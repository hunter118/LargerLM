#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import os
import random
import shutil
import statistics
import struct
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from largerlm.expert_io import stage_batch_experts
from largerlm.staged_moe import run_staged_routed_moe_batch

SCHEMA = "largerlm.glm_moe_tile_sweep.v2"
DEFAULT_MAX_PROMOTION_DRIFT = 1e-5
DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO = 0.98
DEFAULT_MIN_PROMOTION_SAMPLE_COUNT = 3


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_layer(layout: dict[str, Any], layer_id: int) -> dict[str, Any]:
    for layer in layout.get("layers", []):
        if isinstance(layer, dict) and layer.get("layer") == layer_id:
            return layer
    raise SystemExit(f"layer {layer_id} not found in {layout}")


def _component_shape(layer: dict[str, Any], name: str) -> tuple[int, int]:
    for component in layer.get("components", []):
        if isinstance(component, dict) and component.get("name") == name:
            shape = component.get("shape")
            if (
                isinstance(shape, list)
                and len(shape) == 2
                and type(shape[0]) is int
                and type(shape[1]) is int
            ):
                return int(shape[0]), int(shape[1])
    raise SystemExit(f"component {name} not found")


def _parse_experts(value: str) -> tuple[int, ...]:
    experts = tuple(int(item) for item in value.split(",") if item.strip())
    if not experts:
        raise argparse.ArgumentTypeError("expert list must not be empty")
    if any(expert < 0 for expert in experts):
        raise argparse.ArgumentTypeError("expert ids must be non-negative")
    return experts


def _parse_tiles(value: str) -> tuple[int, ...]:
    tiles = tuple(int(item) for item in value.split(",") if item.strip())
    if not tiles:
        raise argparse.ArgumentTypeError("tile list must not be empty")
    if any(tile not in {1, 2, 4} for tile in tiles):
        raise argparse.ArgumentTypeError("tiles must be 1, 2, or 4")
    return tiles


def _parse_vector_swiglu_modes(value: str) -> tuple[bool, ...]:
    modes: list[bool] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"0", "off", "false", "no", "scalar"}:
            modes.append(False)
        elif item in {"1", "on", "true", "yes", "vector"}:
            modes.append(True)
        else:
            raise argparse.ArgumentTypeError(
                "vector SwiGLU modes must be off/on, scalar/vector, or 0/1"
            )
    if not modes:
        raise argparse.ArgumentTypeError("vector SwiGLU mode list must not be empty")
    return tuple(modes)


def _parse_group32_modes(value: str) -> tuple[str, ...]:
    modes: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"auto", "default"}:
            modes.append("auto")
        elif item in {"0", "off", "false", "no", "scalar"}:
            modes.append("off")
        elif item in {"1", "on", "true", "yes", "group32"}:
            modes.append("on")
        else:
            raise argparse.ArgumentTypeError(
                "group32 modes must be auto, off, or on"
            )
    if not modes:
        raise argparse.ArgumentTypeError("group32 mode list must not be empty")
    return tuple(modes)


def _parse_activation_modes(value: str) -> tuple[str, ...]:
    modes: list[str] = []
    aliases = {
        "0": "silu",
        "silu": "silu",
        "exact": "silu",
        "1": "fast-exp",
        "fast": "fast-exp",
        "fast-exp": "fast-exp",
        "2": "linear",
        "linear": "linear",
        "skip-silu": "linear",
    }
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        mode = aliases.get(item)
        if mode is None:
            raise argparse.ArgumentTypeError(
                "activation modes must be silu, fast-exp, or linear"
            )
        modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("activation mode list must not be empty")
    return tuple(modes)


def _config_skip_reason(*, tile: int, vector_swiglu: bool, group32_mode: str) -> str | None:
    if vector_swiglu and tile != 1:
        return "vector SwiGLU is only implemented for MXFP4 token tile 1"
    if group32_mode == "on" and tile != 1:
        return "group32-specialized routed MXFP4 kernels require token tile 1"
    if vector_swiglu and group32_mode != "off":
        return (
            "vector SwiGLU is benchmarked with group32 explicitly off to avoid "
            "ambiguous auto labels"
        )
    return None


def _write_router_jsons(
    *,
    router_json_dir: Path,
    batch_tokens: int,
    experts: tuple[int, ...],
) -> None:
    router_json_dir.mkdir(parents=True, exist_ok=True)
    weight = 1.0 / float(len(experts))
    payload = {"experts": list(experts), "weights": [weight] * len(experts)}
    text = json.dumps(payload, separators=(",", ":"))
    for token in range(batch_tokens):
        (router_json_dir / f"token_{token:06d}.router.json").write_text(
            text,
            encoding="utf-8",
        )


def _write_input(
    path: Path,
    *,
    batch_tokens: int,
    hidden_dim: int,
    mode: str,
    seed: int,
) -> None:
    if mode == "zero":
        row = struct.pack("<" + "f" * hidden_dim, *([0.0] * hidden_dim))
        with path.open("wb") as handle:
            for _ in range(batch_tokens):
                handle.write(row)
        return
    rng = random.Random(seed)
    values = array.array(
        "f",
        (rng.uniform(-0.02, 0.02) for _ in range(batch_tokens * hidden_dim)),
    )
    with path.open("wb") as handle:
        values.tofile(handle)


def _read_f32(path: Path, count: int) -> array.array[float]:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    return values


def _max_abs_diff(lhs: array.array[float], rhs: array.array[float]) -> float:
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def _series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _positive_mean(summary: dict[str, Any], metric: str) -> float | None:
    metric_summary = summary.get(metric)
    if not isinstance(metric_summary, dict):
        return None
    mean = metric_summary.get("mean")
    count = metric_summary.get("count")
    if not isinstance(mean, (int, float)) or not isinstance(count, (int, float)):
        return None
    if count <= 0 or mean <= 0:
        return None
    return float(mean)


def _summary_max_abs_diff(summary: dict[str, Any]) -> float | None:
    value = summary.get("max_abs_diff_vs_first")
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _summary_count(summary: dict[str, Any], metric: str) -> int:
    metric_summary = summary.get(metric)
    if not isinstance(metric_summary, dict):
        return 0
    count = metric_summary.get("count")
    if not isinstance(count, (int, float)) or count <= 0:
        return 0
    return int(count)


def _baseline_config_key(config_summary: dict[str, dict[str, Any]]) -> str | None:
    for key in ("tile1_auto_silu", "tile1_scalar_silu"):
        summary = config_summary.get(key)
        if isinstance(summary, dict) and _positive_mean(summary, "kernel_seconds"):
            return key
    for key in sorted(config_summary):
        summary = config_summary.get(key)
        if isinstance(summary, dict) and _positive_mean(summary, "kernel_seconds"):
            return key
    return None


def _config_comparison(
    config_summary: dict[str, dict[str, Any]],
    *,
    max_promotion_drift: float,
    min_promotion_speedup_ratio: float,
    min_promotion_sample_count: int = DEFAULT_MIN_PROMOTION_SAMPLE_COUNT,
) -> dict[str, Any]:
    baseline_key = _baseline_config_key(config_summary)
    if baseline_key is None:
        return {
            "baseline_config": None,
            "fastest_kernel_config": None,
            "fastest_runner_total_config": None,
            "candidate_config": None,
            "candidate_for_full_replay": False,
            "requires_full_replay_bakeoff": False,
            "max_promotion_drift": max_promotion_drift,
            "min_promotion_speedup_ratio": min_promotion_speedup_ratio,
            "min_promotion_sample_count": min_promotion_sample_count,
            "rows": [],
            "reasons": ["no_config_with_kernel_timing"],
        }

    baseline = config_summary[baseline_key]
    baseline_kernel = _positive_mean(baseline, "kernel_seconds")
    baseline_runner_total = _positive_mean(baseline, "runner_total_seconds")
    rows: list[dict[str, Any]] = []
    fastest_kernel_key: str | None = None
    fastest_kernel_mean: float | None = None
    fastest_runner_key: str | None = None
    fastest_runner_mean: float | None = None
    for key in sorted(config_summary):
        summary = config_summary[key]
        kernel_mean = _positive_mean(summary, "kernel_seconds")
        runner_mean = _positive_mean(summary, "runner_total_seconds")
        if kernel_mean is None:
            continue
        if fastest_kernel_mean is None or kernel_mean < fastest_kernel_mean:
            fastest_kernel_key = key
            fastest_kernel_mean = kernel_mean
        if runner_mean is not None and (
            fastest_runner_mean is None or runner_mean < fastest_runner_mean
        ):
            fastest_runner_key = key
            fastest_runner_mean = runner_mean
        drift = _summary_max_abs_diff(summary)
        kernel_sample_count = _summary_count(summary, "kernel_seconds")
        row = {
            "config": key,
            "is_baseline": key == baseline_key,
            "kernel_sample_count": kernel_sample_count,
            "kernel_seconds_mean": kernel_mean,
            "kernel_ratio_to_baseline": (
                kernel_mean / baseline_kernel
                if baseline_kernel is not None and baseline_kernel > 0
                else None
            ),
            "runner_total_seconds_mean": runner_mean,
            "runner_total_ratio_to_baseline": (
                runner_mean / baseline_runner_total
                if (
                    runner_mean is not None
                    and baseline_runner_total is not None
                    and baseline_runner_total > 0
                )
                else None
            ),
            "max_abs_diff_vs_first": drift,
            "numerically_within_promotion_drift": (
                drift is not None and drift <= max_promotion_drift
            ),
            "meets_min_promotion_sample_count": (
                kernel_sample_count >= min_promotion_sample_count
            ),
        }
        kernel_ratio = row["kernel_ratio_to_baseline"]
        row["microbench_candidate_for_full_replay"] = (
            key != baseline_key
            and isinstance(kernel_ratio, float)
            and kernel_ratio <= min_promotion_speedup_ratio
            and row["numerically_within_promotion_drift"] is True
            and row["meets_min_promotion_sample_count"] is True
        )
        rows.append(row)

    candidate_rows = [
        row
        for row in rows
        if row.get("microbench_candidate_for_full_replay") is True
    ]
    candidate = min(
        candidate_rows,
        key=lambda row: float(row["kernel_seconds_mean"]),
        default=None,
    )
    reasons: list[str] = []
    if candidate is None:
        reasons.append("no_candidate_met_kernel_speedup_and_drift_policy")
    else:
        reasons.append("candidate_met_kernel_speedup_and_drift_policy")
    if fastest_kernel_key == baseline_key:
        reasons.append("baseline_has_fastest_kernel_mean")
    elif fastest_kernel_key:
        reasons.append("non_baseline_has_fastest_kernel_mean")
    return {
        "baseline_config": baseline_key,
        "fastest_kernel_config": fastest_kernel_key,
        "fastest_runner_total_config": fastest_runner_key,
        "candidate_config": candidate.get("config") if candidate else None,
        "candidate_for_full_replay": candidate is not None,
        "requires_full_replay_bakeoff": candidate is not None,
        "max_promotion_drift": max_promotion_drift,
        "min_promotion_speedup_ratio": min_promotion_speedup_ratio,
        "min_promotion_sample_count": min_promotion_sample_count,
        "rows": rows,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat real GLM routed-MoE layer runs across MXFP4 token tiles.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument(
        "--experts",
        type=_parse_experts,
        default="11,79,92,103,154,212,236,254",
    )
    parser.add_argument("--tiles", type=_parse_tiles, default="1,2")
    parser.add_argument(
        "--vector-swiglu-modes",
        type=_parse_vector_swiglu_modes,
        default="off",
        help="Comma-separated experimental LARGERLM_MOE_MXFP4_VECTOR_SWIGLU modes.",
    )
    parser.add_argument(
        "--group32-modes",
        type=_parse_group32_modes,
        default="auto",
        help="Comma-separated experimental LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED modes.",
    )
    parser.add_argument(
        "--activation-modes",
        type=_parse_activation_modes,
        default="silu",
        help="Comma-separated experimental LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION modes.",
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--input-mode", choices=("random", "zero"), default="random")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-stage-mib", type=float, default=512.0)
    parser.add_argument("--max-compact-stage-mib", type=float, default=512.0)
    parser.add_argument("--copy-chunk-mib", type=float, default=8.0)
    parser.add_argument("--max-slot-mib", type=int, default=256)
    parser.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    parser.add_argument("--moe-token-block", default="auto")
    parser.add_argument(
        "--max-promotion-drift",
        type=float,
        default=DEFAULT_MAX_PROMOTION_DRIFT,
        help=(
            "Maximum output drift for marking a microbench candidate as ready "
            "for full replay/bakeoff consideration."
        ),
    )
    parser.add_argument(
        "--min-promotion-speedup-ratio",
        type=float,
        default=DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO,
        help=(
            "Candidate kernel mean must be at most this ratio of baseline "
            "kernel mean before it is recommended for full replay/bakeoff."
        ),
    )
    parser.add_argument(
        "--min-promotion-sample-count",
        type=int,
        default=DEFAULT_MIN_PROMOTION_SAMPLE_COUNT,
        help=(
            "Minimum per-config kernel timing sample count before a microbench "
            "candidate can be recommended for full replay/bakeoff."
        ),
    )
    parser.add_argument("--write-result", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    if args.batch_tokens <= 0:
        raise SystemExit("--batch-tokens must be positive")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.max_promotion_drift < 0:
        raise SystemExit("--max-promotion-drift must be non-negative")
    if not (0 < args.min_promotion_speedup_ratio < 1):
        raise SystemExit("--min-promotion-speedup-ratio must be between 0 and 1")
    if args.min_promotion_sample_count <= 0:
        raise SystemExit("--min-promotion-sample-count must be positive")

    expert_layout = args.prepared_dir / "experts" / "layout.json"
    layout = _load_json(expert_layout)
    layer = _find_layer(layout, args.layer)
    hidden_dim, _ = _component_shape(layer, "down_proj.weight")
    num_experts = int(layer.get("num_experts", 0))
    for expert in args.experts:
        if expert >= num_experts:
            raise SystemExit(f"expert {expert} out of range for {num_experts} experts")

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-tile-sweep-"))
    cleaned = False
    old_tile_env = os.environ.get("LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE")
    old_vector_swiglu_env = os.environ.get("LARGERLM_MOE_MXFP4_VECTOR_SWIGLU")
    old_group32_env = os.environ.get("LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED")
    old_activation_env = os.environ.get("LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION")
    try:
        router_json_dir = work_dir / "router_json"
        input_f32 = work_dir / "input.f32"
        stage_file = work_dir / "experts.stage.bin"
        stage_manifest = work_dir / "experts.stage.manifest.json"

        _write_router_jsons(
            router_json_dir=router_json_dir,
            batch_tokens=args.batch_tokens,
            experts=args.experts,
        )
        _write_input(
            input_f32,
            batch_tokens=args.batch_tokens,
            hidden_dim=hidden_dim,
            mode=args.input_mode,
            seed=args.seed,
        )

        stage_started = time.perf_counter()
        stage = stage_batch_experts(
            expert_layout,
            layer=args.layer,
            router_json_dir=router_json_dir,
            stage_file_path=stage_file,
            manifest_path=stage_manifest,
            max_stage_mib=args.max_stage_mib,
            copy_chunk_mib=args.copy_chunk_mib,
        )
        stage_elapsed = time.perf_counter() - stage_started

        output_count = args.batch_tokens * hidden_dim
        baseline_output: array.array[float] | None = None
        records: list[dict[str, Any]] = []
        schedule: list[tuple[int, bool, str, str, int]] = []
        skipped_configs: list[dict[str, Any]] = []

        def maybe_schedule(
            tile: int,
            vector_swiglu: bool,
            group32_mode: str,
            activation_mode: str,
            index: int,
        ) -> None:
            skip_reason = _config_skip_reason(
                tile=tile,
                vector_swiglu=vector_swiglu,
                group32_mode=group32_mode,
            )
            if skip_reason is not None:
                skipped_configs.append(
                    {
                        "tile": tile,
                        "vector_swiglu": vector_swiglu,
                        "group32_mode": group32_mode,
                        "activation_mode": activation_mode,
                        "repeat_index": index,
                        "reason": skip_reason,
                    }
                )
                return
            schedule.append(
                (
                    tile,
                    vector_swiglu,
                    group32_mode,
                    activation_mode,
                    index,
                )
            )

        if args.order == "grouped":
            for tile in args.tiles:
                for vector_swiglu in args.vector_swiglu_modes:
                    for group32_mode in args.group32_modes:
                        for activation_mode in args.activation_modes:
                            for index in range(args.repeat):
                                maybe_schedule(
                                    tile,
                                    vector_swiglu,
                                    group32_mode,
                                    activation_mode,
                                    index,
                                )
        else:
            for index in range(args.repeat):
                for tile in args.tiles:
                    for vector_swiglu in args.vector_swiglu_modes:
                        for group32_mode in args.group32_modes:
                            for activation_mode in args.activation_modes:
                                maybe_schedule(
                                    tile,
                                    vector_swiglu,
                                    group32_mode,
                                    activation_mode,
                                    index,
                                )
        for tile, vector_swiglu, group32_mode, activation_mode, index in schedule:
            os.environ["LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE"] = str(tile)
            if vector_swiglu:
                os.environ["LARGERLM_MOE_MXFP4_VECTOR_SWIGLU"] = "1"
            else:
                os.environ.pop("LARGERLM_MOE_MXFP4_VECTOR_SWIGLU", None)
            if group32_mode == "on":
                os.environ["LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED"] = "1"
            elif group32_mode == "off":
                os.environ["LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED"] = "0"
            else:
                os.environ.pop("LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED", None)
            if activation_mode == "silu":
                os.environ.pop("LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION", None)
            else:
                os.environ["LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION"] = activation_mode
            try:
                mode_label = "vector" if vector_swiglu else "scalar"
                if group32_mode == "on":
                    mode_label = "group32"
                elif group32_mode == "auto":
                    mode_label = "auto"
                act_label = activation_mode.replace("-", "")
                output_dir = (
                    work_dir / f"run_tile{tile}_{mode_label}_{act_label}_rep{index:02d}"
                )
                output_f32 = (
                    work_dir / f"output_tile{tile}_{mode_label}_{act_label}_rep{index:02d}.f32"
                )
                run_started = time.perf_counter()
                moe = run_staged_routed_moe_batch(
                    runner_path=args.runner,
                    stage_manifest_path=stage_manifest,
                    input_f32_path=input_f32,
                    output_f32_path=output_f32,
                    output_dir=output_dir,
                    max_compact_stage_mib=args.max_compact_stage_mib,
                    copy_chunk_mib=args.copy_chunk_mib,
                    max_slot_mib=args.max_slot_mib,
                    max_runner_scratch_mib=args.max_runner_scratch_mib,
                    moe_token_block=args.moe_token_block,
                    static_capacity_per_expert=args.batch_tokens,
                    write_static_capacity_json=False,
                    echo_runner_output=False,
                )
                run_elapsed = time.perf_counter() - run_started
                output = _read_f32(output_f32, output_count)
                if baseline_output is None:
                    baseline_output = output
                    max_abs_diff = 0.0
                else:
                    max_abs_diff = _max_abs_diff(baseline_output, output)
                records.append(
                    {
                        "tile": tile,
                        "vector_swiglu": vector_swiglu,
                        "group32_mode": group32_mode,
                        "group32_specialized": moe.moe_mxfp4_group32_specialized,
                        "activation_mode": activation_mode,
                        "runner_mxfp4_token_tile": moe.moe_mxfp4_token_tile,
                        "runner_mxfp4_vector_swiglu": moe.moe_mxfp4_vector_swiglu,
                        "runner_mxfp4_group32_specialized": (
                            moe.moe_mxfp4_group32_specialized
                        ),
                        "runner_mxfp4_swiglu_activation": (
                            moe.moe_mxfp4_swiglu_activation
                        ),
                        "repeat_index": index,
                        "run_elapsed_seconds": run_elapsed,
                        "kernel_seconds": moe.moe_timing_kernel_seconds,
                        "mxfp4_swiglu_kernel_seconds": (
                            moe.moe_timing_mxfp4_swiglu_kernel_seconds
                        ),
                        "mxfp4_down_add_kernel_seconds": (
                            moe.moe_timing_mxfp4_down_add_kernel_seconds
                        ),
                        "runner_total_seconds": moe.moe_timing_total_seconds,
                        "expert_read_seconds": moe.moe_timing_expert_read_seconds,
                        "input_read_seconds": moe.moe_timing_input_read_seconds,
                        "output_read_seconds": moe.moe_timing_output_read_seconds,
                        "output_write_seconds": moe.moe_timing_output_write_seconds,
                        "output_accumulator": moe.moe_output_accumulator,
                        "output_accumulator_bytes": moe.moe_output_accumulator_bytes,
                        "max_abs_diff_vs_first": max_abs_diff,
                        "effective_token_block": moe.effective_moe_token_block,
                        "estimated_peak_bytes": moe.moe_estimated_peak_bytes,
                    }
                )
            finally:
                if old_tile_env is None:
                    os.environ.pop("LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE", None)
                else:
                    os.environ["LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE"] = old_tile_env
                if old_vector_swiglu_env is None:
                    os.environ.pop("LARGERLM_MOE_MXFP4_VECTOR_SWIGLU", None)
                else:
                    os.environ["LARGERLM_MOE_MXFP4_VECTOR_SWIGLU"] = old_vector_swiglu_env
                if old_group32_env is None:
                    os.environ.pop("LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED", None)
                else:
                    os.environ["LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED"] = old_group32_env
                if old_activation_env is None:
                    os.environ.pop("LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION", None)
                else:
                    os.environ["LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION"] = old_activation_env

        by_tile: dict[str, dict[str, Any]] = {}
        for tile in args.tiles:
            tile_records = [record for record in records if record["tile"] == tile]
            by_tile[str(tile)] = {
                "kernel_seconds": _series([r["kernel_seconds"] for r in tile_records]),
                "runner_total_seconds": _series(
                    [r["runner_total_seconds"] for r in tile_records]
                ),
                "mxfp4_swiglu_kernel_seconds": _series(
                    [
                        r["mxfp4_swiglu_kernel_seconds"]
                        for r in tile_records
                        if r["mxfp4_swiglu_kernel_seconds"] is not None
                    ]
                ),
                "mxfp4_down_add_kernel_seconds": _series(
                    [
                        r["mxfp4_down_add_kernel_seconds"]
                        for r in tile_records
                        if r["mxfp4_down_add_kernel_seconds"] is not None
                    ]
                ),
                "run_elapsed_seconds": _series(
                    [r["run_elapsed_seconds"] for r in tile_records]
                ),
                "max_abs_diff_vs_first": max(
                    (r["max_abs_diff_vs_first"] for r in tile_records),
                    default=0.0,
                ),
            }
        by_config: dict[str, dict[str, Any]] = {}
        for tile in args.tiles:
            for vector_swiglu in args.vector_swiglu_modes:
                for group32_mode in args.group32_modes:
                    if (
                        _config_skip_reason(
                            tile=tile,
                            vector_swiglu=vector_swiglu,
                            group32_mode=group32_mode,
                        )
                        is not None
                    ):
                        continue
                    for activation_mode in args.activation_modes:
                        config_records = [
                            record
                            for record in records
                            if record["tile"] == tile
                            and record["vector_swiglu"] == vector_swiglu
                            and record["group32_mode"] == group32_mode
                            and record["activation_mode"] == activation_mode
                        ]
                        act_label = activation_mode.replace("-", "")
                        if group32_mode == "on":
                            kernel_label = "group32"
                        elif group32_mode == "auto":
                            kernel_label = "auto"
                        else:
                            kernel_label = "vector" if vector_swiglu else "scalar"
                        key = f"tile{tile}_{kernel_label}_{act_label}"
                        by_config[key] = {
                            "kernel_seconds": _series(
                                [r["kernel_seconds"] for r in config_records]
                            ),
                            "runner_total_seconds": _series(
                                [r["runner_total_seconds"] for r in config_records]
                            ),
                            "mxfp4_swiglu_kernel_seconds": _series(
                                [
                                    r["mxfp4_swiglu_kernel_seconds"]
                                    for r in config_records
                                    if r["mxfp4_swiglu_kernel_seconds"] is not None
                                ]
                            ),
                            "mxfp4_down_add_kernel_seconds": _series(
                                [
                                    r["mxfp4_down_add_kernel_seconds"]
                                    for r in config_records
                                    if r["mxfp4_down_add_kernel_seconds"] is not None
                                ]
                            ),
                            "run_elapsed_seconds": _series(
                                [r["run_elapsed_seconds"] for r in config_records]
                            ),
                            "max_abs_diff_vs_first": max(
                                (r["max_abs_diff_vs_first"] for r in config_records),
                                default=0.0,
                            ),
                        }

        payload = {
            "schema": SCHEMA,
            "prepared_dir": args.prepared_dir,
            "runner": args.runner,
            "work_dir": work_dir,
            "layer": args.layer,
            "batch_tokens": args.batch_tokens,
            "experts": args.experts,
            "tiles": args.tiles,
            "vector_swiglu_modes": args.vector_swiglu_modes,
            "group32_modes": args.group32_modes,
            "activation_modes": args.activation_modes,
            "repeat": args.repeat,
            "order": args.order,
            "input_mode": args.input_mode,
            "seed": args.seed,
            "hidden_dim": hidden_dim,
            "skipped_configs": skipped_configs,
            "stage_elapsed_seconds": stage_elapsed,
            "stage_staged_bytes": stage.staged_bytes,
            "stage_copy_elapsed_seconds": stage.copy_elapsed_seconds,
            "stage_copy_throughput_gib_per_second": (
                stage.copy_throughput_gib_per_second
            ),
            "stage_raw_range_count": stage.io_summary.raw_range_count,
            "stage_coalesced_range_count": stage.io_summary.coalesced_range_count,
            "tile_summary": by_tile,
            "config_summary": by_config,
            "config_comparison": _config_comparison(
                by_config,
                max_promotion_drift=args.max_promotion_drift,
                min_promotion_speedup_ratio=args.min_promotion_speedup_ratio,
                min_promotion_sample_count=args.min_promotion_sample_count,
            ),
            "records": records,
            "work_dir_cleaned": False,
        }
        if not args.keep_work_dir:
            shutil.rmtree(work_dir)
            cleaned = True
        payload["work_dir_cleaned"] = cleaned
        if args.write_result is not None:
            args.write_result.parent.mkdir(parents=True, exist_ok=True)
            args.write_result.write_text(
                json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
        return 0
    finally:
        if old_tile_env is None:
            os.environ.pop("LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE", None)
        else:
            os.environ["LARGERLM_MOE_MXFP4_BATCH_TOKEN_TILE"] = old_tile_env
        if old_vector_swiglu_env is None:
            os.environ.pop("LARGERLM_MOE_MXFP4_VECTOR_SWIGLU", None)
        else:
            os.environ["LARGERLM_MOE_MXFP4_VECTOR_SWIGLU"] = old_vector_swiglu_env
        if old_group32_env is None:
            os.environ.pop("LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED", None)
        else:
            os.environ["LARGERLM_MOE_MXFP4_GROUP32_SPECIALIZED"] = old_group32_env
        if old_activation_env is None:
            os.environ.pop("LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION", None)
        else:
            os.environ["LARGERLM_MOE_MXFP4_SWIGLU_ACTIVATION"] = old_activation_env
        if not args.keep_work_dir and not cleaned and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
