#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.prefill_execute import (
    BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV,
    run_prefill_attention_projection_batch,
)

DEFAULT_MAX_PROMOTION_DRIFT = 1e-5
DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO = 0.98
DEFAULT_MIN_PROMOTION_SAMPLE_COUNT = 3


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_tensor(layout: dict[str, Any], *, layer: int, suffix: str) -> dict[str, Any]:
    target = f"model.layers.{layer}{suffix}"
    for tensor in layout.get("tensors", []):
        if isinstance(tensor, dict) and tensor.get("name") == target:
            return tensor
    raise SystemExit(f"tensor {target} not found")


def _matrix_out_in(tensor: dict[str, Any]) -> tuple[int, int]:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise SystemExit(f"tensor {tensor.get('name')} does not have a matrix shape")
    out_dim = int(shape[0])
    in_dim = int(shape[1])
    if tensor.get("dtype") == "U32":
        in_dim *= 8
    if out_dim <= 0 or in_dim <= 0:
        raise SystemExit(f"tensor {tensor.get('name')} has invalid shape {shape}")
    return out_dim, in_dim


def _parse_modes(value: str) -> tuple[str, ...]:
    modes: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"fused", "auto", "default", "on", "1", "true", "yes"}:
            modes.append("fused")
        elif item in {"separate", "off", "0", "false", "no"}:
            modes.append("separate")
        else:
            raise argparse.ArgumentTypeError("modes must be fused or separate")
    if not modes:
        raise argparse.ArgumentTypeError("mode list must not be empty")
    return tuple(modes)


def _write_f32(path: Path, *, count: int, seed: int, scale: float) -> None:
    rng = random.Random(seed)
    values = array.array("f", (rng.uniform(-scale, scale) for _ in range(count)))
    with path.open("wb") as handle:
        values.tofile(handle)


def _read_f32(path: Path, count: int) -> array.array[float]:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    if len(values) != count:
        raise SystemExit(f"{path} had {len(values)} f32 values, expected {count}")
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


def _series_count(series: dict[str, Any]) -> int:
    value = series.get("count")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _series_mean(series: dict[str, Any]) -> float | None:
    value = series.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def _config_comparison(
    by_mode: dict[str, dict[str, Any]],
    *,
    max_promotion_drift: float = DEFAULT_MAX_PROMOTION_DRIFT,
    min_promotion_speedup_ratio: float = DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO,
    min_promotion_sample_count: int = DEFAULT_MIN_PROMOTION_SAMPLE_COUNT,
) -> dict[str, Any]:
    if not by_mode:
        return {
            "baseline_mode": None,
            "fastest_wall_mode": None,
            "candidate_mode": None,
            "candidate_for_full_replay": False,
            "requires_full_replay_bakeoff": False,
            "max_promotion_drift": max_promotion_drift,
            "min_promotion_speedup_ratio": min_promotion_speedup_ratio,
            "min_promotion_sample_count": min_promotion_sample_count,
            "rows": [],
            "reasons": ["no_modes_observed"],
        }
    baseline_mode = "fused" if "fused" in by_mode else next(iter(by_mode))
    baseline_wall = _series_mean(
        dict(by_mode.get(baseline_mode, {}).get("elapsed_wall_seconds") or {})
    )
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for mode, summary in by_mode.items():
        wall_series = dict(summary.get("elapsed_wall_seconds") or {})
        wall_mean = _series_mean(wall_series)
        sample_count = _series_count(wall_series)
        max_diff = float(summary.get("max_abs_diff_vs_first") or 0.0)
        wall_ratio = _ratio(wall_mean, baseline_wall)
        enough_samples = sample_count >= min_promotion_sample_count
        numerically_close = max_diff <= max_promotion_drift
        speedup = (
            wall_ratio is not None and wall_ratio <= min_promotion_speedup_ratio
        )
        candidate = (
            mode != baseline_mode
            and enough_samples
            and numerically_close
            and speedup
        )
        row = {
            "mode": mode,
            "is_baseline": mode == baseline_mode,
            "wall_seconds_mean": wall_mean,
            "wall_ratio_to_baseline": wall_ratio,
            "wall_sample_count": sample_count,
            "max_abs_diff_vs_first": max_diff,
            "meets_min_promotion_sample_count": enough_samples,
            "numerically_within_promotion_drift": numerically_close,
            "microbench_candidate_for_full_replay": candidate,
        }
        rows.append(row)
        if candidate:
            candidate_rows.append(row)
    fastest_wall = min(
        (row for row in rows if row["wall_seconds_mean"] is not None),
        key=lambda row: float(row["wall_seconds_mean"]),
        default=None,
    )
    candidate = min(
        candidate_rows,
        key=lambda row: float(row["wall_seconds_mean"]),
        default=None,
    )
    reasons: list[str] = []
    if candidate is None:
        reasons.append("no_candidate_met_wall_speedup_and_drift_policy")
    else:
        reasons.append("candidate_met_wall_speedup_and_drift_policy")
    if fastest_wall and fastest_wall["mode"] == baseline_mode:
        reasons.append("baseline_has_fastest_wall_mean")
    elif fastest_wall:
        reasons.append("non_baseline_has_fastest_wall_mean")
    return {
        "baseline_mode": baseline_mode,
        "fastest_wall_mode": fastest_wall["mode"] if fastest_wall is not None else None,
        "candidate_mode": candidate["mode"] if candidate is not None else None,
        "candidate_for_full_replay": candidate is not None,
        "requires_full_replay_bakeoff": candidate is not None,
        "max_promotion_drift": max_promotion_drift,
        "min_promotion_speedup_ratio": min_promotion_speedup_ratio,
        "min_promotion_sample_count": min_promotion_sample_count,
        "rows": rows,
        "reasons": reasons,
    }


def _set_projection_mode(mode: str) -> str | None:
    previous = os.environ.get(BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV)
    if mode == "fused":
        os.environ.pop(BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV, None)
    else:
        os.environ[BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV] = "1"
    return previous


def _restore_projection_mode(previous: str | None) -> None:
    if previous is None:
        os.environ.pop(BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV, None)
    else:
        os.environ[BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV] = previous


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fused and separate GLM attention projection prefill paths.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument(
        "--modes",
        type=_parse_modes,
        default="fused,separate",
        help="Comma-separated modes: fused, separate.",
    )
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-resident-matrix-mib", type=int, default=256)
    parser.add_argument("--max-runner-scratch-mib", type=int, default=256)
    parser.add_argument(
        "--max-promotion-drift",
        type=float,
        default=DEFAULT_MAX_PROMOTION_DRIFT,
    )
    parser.add_argument(
        "--min-promotion-speedup-ratio",
        type=float,
        default=DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO,
    )
    parser.add_argument(
        "--min-promotion-sample-count",
        type=int,
        default=DEFAULT_MIN_PROMOTION_SAMPLE_COUNT,
    )
    parser.add_argument("--write-result", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    if args.batch_tokens <= 1:
        raise SystemExit("--batch-tokens must be greater than one for fusion comparison")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.max_resident_matrix_mib <= 0 or args.max_runner_scratch_mib <= 0:
        raise SystemExit("memory limits must be positive")
    if args.max_promotion_drift < 0:
        raise SystemExit("--max-promotion-drift must be non-negative")
    if not (0 < args.min_promotion_speedup_ratio < 1):
        raise SystemExit("--min-promotion-speedup-ratio must be between 0 and 1")
    if args.min_promotion_sample_count <= 0:
        raise SystemExit("--min-promotion-sample-count must be positive")

    resident_layout = args.prepared_dir / "resident" / "layout.json"
    layout = _load_json(resident_layout)
    q_a = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".self_attn.q_a_proj.weight",
    )
    q_b = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".self_attn.q_b_proj.weight",
    )
    kv_a = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".self_attn.kv_a_proj_with_mqa.weight",
    )
    _, hidden_dim = _matrix_out_in(q_a)
    q_b_dim, _ = _matrix_out_in(q_b)
    kv_a_dim, _ = _matrix_out_in(kv_a)

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-attn-proj-fusion-"))
    cleaned = False
    previous_env = os.environ.get(BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV)
    try:
        input_f32 = work_dir / "hidden.f32"
        _write_f32(
            input_f32,
            count=args.batch_tokens * hidden_dim,
            seed=args.seed,
            scale=args.input_scale,
        )
        schedule: list[tuple[str, int]] = []
        if args.order == "grouped":
            for mode in args.modes:
                for index in range(args.repeat):
                    schedule.append((mode, index))
        else:
            for index in range(args.repeat):
                for mode in args.modes:
                    schedule.append((mode, index))

        baseline_outputs: dict[str, array.array[float]] | None = None
        records: list[dict[str, Any]] = []
        for mode, repeat_index in schedule:
            output_dir = work_dir / f"{mode}_rep{repeat_index:02d}"
            old_env = _set_projection_mode(mode)
            started = time.perf_counter()
            try:
                result = run_prefill_attention_projection_batch(
                    runner_path=args.runner,
                    resident_layout_path=resident_layout,
                    layer=args.layer,
                    input_f32_path=input_f32,
                    output_dir=output_dir,
                    batch_tokens=args.batch_tokens,
                    rms_norm_eps=args.rms_norm_eps,
                    max_resident_matrix_mib=args.max_resident_matrix_mib,
                    max_runner_scratch_mib=args.max_runner_scratch_mib,
                    prefill_linear_backend="custom-metal",
                    echo_runner_output=False,
                )
            finally:
                _restore_projection_mode(old_env)
            elapsed = time.perf_counter() - started
            outputs = {
                "q_b": _read_f32(result.q_b_proj.output_path, args.batch_tokens * q_b_dim),
                "kv_a_lora": _read_f32(
                    result.kv_a_lora_path,
                    args.batch_tokens * result.kv_lora_dim,
                ),
                "kv_a_rope": _read_f32(
                    result.kv_a_rope_path,
                    args.batch_tokens * result.kv_rope_dim,
                ),
            }
            if result.kv_b_proj is not None:
                outputs["kv_b"] = _read_f32(
                    result.kv_b_proj.output_path,
                    args.batch_tokens * result.kv_b_dim,
                )
            if baseline_outputs is None:
                baseline_outputs = outputs
                diffs = {key: 0.0 for key in outputs}
            else:
                diffs = {
                    key: _max_abs_diff(baseline_outputs[key], value)
                    for key, value in outputs.items()
                }
            records.append(
                {
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "elapsed_wall_seconds": elapsed,
                    "estimated_peak_bytes": result.estimated_peak_bytes,
                    "attention_value_source": result.attention_value_source,
                    "command_count_estimate": (
                        1
                        if "--run-attn-projections" in result.q_b_proj.command
                        else 7
                    ),
                    "q_b_backend": result.q_b_proj.backend,
                    "kv_b_backend": result.kv_b_proj.backend
                    if result.kv_b_proj is not None
                    else None,
                    "max_abs_diff_outputs_vs_first": diffs,
                    "max_abs_diff_vs_first": max(diffs.values(), default=0.0),
                }
            )

        by_mode: dict[str, dict[str, Any]] = {}
        for mode in args.modes:
            mode_records = [record for record in records if record["mode"] == mode]
            by_mode[mode] = {
                "elapsed_wall_seconds": _series(
                    [float(record["elapsed_wall_seconds"]) for record in mode_records]
                ),
                "estimated_peak_bytes_max": max(
                    (int(record["estimated_peak_bytes"]) for record in mode_records),
                    default=0,
                ),
                "command_count_estimate": sorted(
                    {
                        int(record["command_count_estimate"])
                        for record in mode_records
                    }
                ),
                "max_abs_diff_vs_first": max(
                    (float(record["max_abs_diff_vs_first"]) for record in mode_records),
                    default=0.0,
                ),
            }

        payload = {
            "schema": "largerlm.glm_attention_projection_fusion_sweep.v1",
            "prepared_dir": args.prepared_dir,
            "runner": args.runner,
            "resident_layout": resident_layout,
            "layer": args.layer,
            "batch_tokens": args.batch_tokens,
            "hidden_dim": hidden_dim,
            "q_b_dim": q_b_dim,
            "kv_a_dim": kv_a_dim,
            "modes": list(args.modes),
            "repeat": args.repeat,
            "order": args.order,
            "seed": args.seed,
            "input_scale": args.input_scale,
            "rms_norm_eps": args.rms_norm_eps,
            "max_resident_matrix_mib": args.max_resident_matrix_mib,
            "max_runner_scratch_mib": args.max_runner_scratch_mib,
            "records": records,
            "by_mode": by_mode,
            "config_comparison": _config_comparison(
                by_mode,
                max_promotion_drift=args.max_promotion_drift,
                min_promotion_speedup_ratio=args.min_promotion_speedup_ratio,
                min_promotion_sample_count=args.min_promotion_sample_count,
            ),
            "work_dir": work_dir,
            "work_dir_cleaned": False,
        }
        if args.write_result is not None:
            args.write_result.parent.mkdir(parents=True, exist_ok=True)
            args.write_result.write_text(
                json.dumps(payload, default=_json_default, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    finally:
        _restore_projection_mode(previous_env)
        if not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
            cleaned = True
        if args.write_result is not None and args.write_result.exists():
            try:
                payload = json.loads(args.write_result.read_text(encoding="utf-8"))
                payload["work_dir_cleaned"] = cleaned
                args.write_result.write_text(
                    json.dumps(payload, default=_json_default, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
            except (OSError, json.JSONDecodeError):
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
