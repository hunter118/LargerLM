#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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


def _write_f32(
    path: Path,
    *,
    count: int,
    mode: str,
    seed: int,
    scale: float,
) -> None:
    if mode == "zero":
        values = array.array("f", [0.0]) * count
    else:
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
            "fastest_backend_mode": None,
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
    baseline_mode = "auto" if "auto" in by_mode else next(iter(by_mode))
    baseline = by_mode.get(baseline_mode, {})
    baseline_backend = _series_mean(
        dict(baseline.get("timing_backend_seconds") or {})
    )
    baseline_wall = _series_mean(dict(baseline.get("elapsed_wall_seconds") or {}))
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for mode, summary in by_mode.items():
        backend_series = dict(summary.get("timing_backend_seconds") or {})
        wall_series = dict(summary.get("elapsed_wall_seconds") or {})
        backend_mean = _series_mean(backend_series)
        wall_mean = _series_mean(wall_series)
        sample_count = _series_count(backend_series)
        projection_diff = float(summary.get("max_abs_diff_projection_vs_first") or 0.0)
        output_diff = float(summary.get("max_abs_diff_output_vs_first") or 0.0)
        max_diff = max(projection_diff, output_diff)
        backend_ratio = _ratio(backend_mean, baseline_backend)
        wall_ratio = _ratio(wall_mean, baseline_wall)
        enough_samples = sample_count >= min_promotion_sample_count
        numerically_close = max_diff <= max_promotion_drift
        speedup = (
            backend_ratio is not None
            and backend_ratio <= min_promotion_speedup_ratio
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
            "backend_seconds_mean": backend_mean,
            "backend_ratio_to_baseline": backend_ratio,
            "wall_seconds_mean": wall_mean,
            "wall_ratio_to_baseline": wall_ratio,
            "backend_sample_count": sample_count,
            "max_abs_diff_projection_vs_first": projection_diff,
            "max_abs_diff_output_vs_first": output_diff,
            "max_abs_diff_vs_first": max_diff,
            "meets_min_promotion_sample_count": enough_samples,
            "numerically_within_promotion_drift": numerically_close,
            "microbench_candidate_for_full_replay": candidate,
        }
        rows.append(row)
        if candidate:
            candidate_rows.append(row)
    fastest_backend = min(
        (row for row in rows if row["backend_seconds_mean"] is not None),
        key=lambda row: float(row["backend_seconds_mean"]),
        default=None,
    )
    fastest_wall = min(
        (row for row in rows if row["wall_seconds_mean"] is not None),
        key=lambda row: float(row["wall_seconds_mean"]),
        default=None,
    )
    candidate = min(
        candidate_rows,
        key=lambda row: float(row["backend_seconds_mean"]),
        default=None,
    )
    reasons: list[str] = []
    if candidate is None:
        reasons.append("no_candidate_met_backend_speedup_and_drift_policy")
    else:
        reasons.append("candidate_met_backend_speedup_and_drift_policy")
    if fastest_backend and fastest_backend["mode"] == baseline_mode:
        reasons.append("baseline_has_fastest_backend_mean")
    elif fastest_backend:
        reasons.append("non_baseline_has_fastest_backend_mean")
    return {
        "baseline_mode": baseline_mode,
        "fastest_backend_mode": (
            fastest_backend["mode"] if fastest_backend is not None else None
        ),
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


def _parse_runner_stdout(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("timing backend:"):
            parsed["timing_backend_seconds"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("MXFP4 token tile:"):
            parsed["mxfp4_token_tile"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("MXFP4 group32 path:"):
            parsed["mxfp4_group32_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("estimated peak:"):
            parsed["estimated_peak_bytes"] = int(line.split(":", 1)[1].strip())
    return parsed


def _run_one(
    *,
    runner: Path,
    resident_layout: Path,
    layer: int,
    input_f32: Path,
    residual_f32: Path,
    output_f32: Path,
    projection_f32: Path,
    batch_tokens: int,
    token_tile: int,
    group32_mode: str,
    max_resident_matrix_mib: int,
    max_runner_scratch_mib: int,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["LARGERLM_MXFP4_BATCH_TOKEN_TILE"] = str(token_tile)
    if group32_mode == "auto":
        env.pop("LARGERLM_MXFP4_GROUP32_SPECIALIZED", None)
    elif group32_mode == "off":
        env["LARGERLM_MXFP4_GROUP32_SPECIALIZED"] = "0"
    else:
        env["LARGERLM_MXFP4_GROUP32_SPECIALIZED"] = "1"
    command = [
        str(runner),
        "--resident-layout",
        str(resident_layout),
        "--layer",
        str(layer),
        "--run-attn-output-batch",
        "--input-f32",
        str(input_f32),
        "--residual-f32",
        str(residual_f32),
        "--batch-tokens",
        str(batch_tokens),
        "--output-f32",
        str(output_f32),
        "--projection-f32",
        str(projection_f32),
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(
            f"runner failed for group32={group32_mode}, tile={token_tile}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    parsed = _parse_runner_stdout(result.stdout)
    parsed.update(
        {
            "command": command,
            "elapsed_wall_seconds": elapsed,
            "stderr": result.stderr,
        }
    )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare resident MXFP4 group32 kernels on GLM attention o_proj.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--token-tile", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument(
        "--group32-modes",
        type=_parse_group32_modes,
        default="off,auto",
        help="Comma-separated modes: off, auto, on.",
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--input-mode", choices=("random", "zero"), default="random")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--residual-scale", type=float, default=0.02)
    parser.add_argument("--max-resident-matrix-mib", type=int, default=512)
    parser.add_argument("--max-runner-scratch-mib", type=int, default=512)
    parser.add_argument(
        "--max-promotion-drift",
        type=float,
        default=DEFAULT_MAX_PROMOTION_DRIFT,
        help=(
            "Maximum projection/output drift for marking a resident projection "
            "microbench candidate as ready for full replay consideration."
        ),
    )
    parser.add_argument(
        "--min-promotion-speedup-ratio",
        type=float,
        default=DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO,
        help=(
            "Candidate backend mean must be at most this ratio of baseline "
            "backend mean before it is recommended for full replay/bakeoff."
        ),
    )
    parser.add_argument(
        "--min-promotion-sample-count",
        type=int,
        default=DEFAULT_MIN_PROMOTION_SAMPLE_COUNT,
        help=(
            "Minimum per-mode backend timing sample count before a microbench "
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
    weight = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".self_attn.o_proj.weight",
    )
    scales = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".self_attn.o_proj.scales",
    )
    if weight.get("dtype") != "U32" or scales.get("dtype") != "U8":
        raise SystemExit(
            "this sweep expects MLX MXFP4 o_proj tensors: U32 weight and U8 scales"
        )
    weight_shape = weight.get("shape")
    scales_shape = scales.get("shape")
    if (
        not isinstance(weight_shape, list)
        or len(weight_shape) != 2
        or not isinstance(scales_shape, list)
        or len(scales_shape) != 2
    ):
        raise SystemExit("unexpected o_proj weight/scales shapes")
    out_dim = int(weight_shape[0])
    in_dim = int(weight_shape[1]) * 8
    group_size = in_dim // int(scales_shape[1])
    if group_size <= 0 or in_dim % int(scales_shape[1]) != 0:
        raise SystemExit("could not infer MXFP4 group size")

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-resident-mxfp4-group32-"))
    cleaned = False
    try:
        input_f32 = work_dir / "attn_value.f32"
        residual_f32 = work_dir / "residual.f32"
        _write_f32(
            input_f32,
            count=args.batch_tokens * in_dim,
            mode=args.input_mode,
            seed=args.seed,
            scale=args.input_scale,
        )
        _write_f32(
            residual_f32,
            count=args.batch_tokens * out_dim,
            mode=args.input_mode,
            seed=args.seed + 1,
            scale=args.residual_scale,
        )

        schedule: list[tuple[str, int]] = []
        if args.order == "grouped":
            for mode in args.group32_modes:
                for index in range(args.repeat):
                    schedule.append((mode, index))
        else:
            for index in range(args.repeat):
                for mode in args.group32_modes:
                    schedule.append((mode, index))

        projection_count = args.batch_tokens * out_dim
        baseline_projection: array.array[float] | None = None
        baseline_output: array.array[float] | None = None
        records: list[dict[str, Any]] = []
        for group32_mode, repeat_index in schedule:
            projection_f32 = (
                work_dir / f"projection_{group32_mode}_rep{repeat_index:02d}.f32"
            )
            output_f32 = work_dir / f"output_{group32_mode}_rep{repeat_index:02d}.f32"
            parsed = _run_one(
                runner=args.runner,
                resident_layout=resident_layout,
                layer=args.layer,
                input_f32=input_f32,
                residual_f32=residual_f32,
                output_f32=output_f32,
                projection_f32=projection_f32,
                batch_tokens=args.batch_tokens,
                token_tile=args.token_tile,
                group32_mode=group32_mode,
                max_resident_matrix_mib=args.max_resident_matrix_mib,
                max_runner_scratch_mib=args.max_runner_scratch_mib,
            )
            projection = _read_f32(projection_f32, projection_count)
            output = _read_f32(output_f32, projection_count)
            if baseline_projection is None or baseline_output is None:
                baseline_projection = projection
                baseline_output = output
                projection_diff = 0.0
                output_diff = 0.0
            else:
                projection_diff = _max_abs_diff(baseline_projection, projection)
                output_diff = _max_abs_diff(baseline_output, output)
            records.append(
                {
                    "group32_mode": group32_mode,
                    "repeat_index": repeat_index,
                    "token_tile": args.token_tile,
                    "mxfp4_token_tile": parsed.get("mxfp4_token_tile"),
                    "mxfp4_group32_path": parsed.get("mxfp4_group32_path"),
                    "timing_backend_seconds": parsed.get("timing_backend_seconds"),
                    "elapsed_wall_seconds": parsed["elapsed_wall_seconds"],
                    "estimated_peak_bytes": parsed.get("estimated_peak_bytes"),
                    "max_abs_diff_projection_vs_first": projection_diff,
                    "max_abs_diff_output_vs_first": output_diff,
                }
            )

        by_mode: dict[str, dict[str, Any]] = {}
        for mode in args.group32_modes:
            mode_records = [record for record in records if record["group32_mode"] == mode]
            by_mode[mode] = {
                "timing_backend_seconds": _series(
                    [
                        float(record["timing_backend_seconds"])
                        for record in mode_records
                        if record["timing_backend_seconds"] is not None
                    ]
                ),
                "elapsed_wall_seconds": _series(
                    [float(record["elapsed_wall_seconds"]) for record in mode_records]
                ),
                "max_abs_diff_projection_vs_first": max(
                    (
                        float(record["max_abs_diff_projection_vs_first"])
                        for record in mode_records
                    ),
                    default=0.0,
                ),
                "max_abs_diff_output_vs_first": max(
                    (float(record["max_abs_diff_output_vs_first"]) for record in mode_records),
                    default=0.0,
                ),
                "observed_group32_paths": sorted(
                    {
                        str(record["mxfp4_group32_path"])
                        for record in mode_records
                        if record["mxfp4_group32_path"] is not None
                    }
                ),
            }

        payload = {
            "schema": "largerlm.glm_resident_mxfp4_group32_sweep.v1",
            "prepared_dir": args.prepared_dir,
            "runner": args.runner,
            "resident_layout": resident_layout,
            "work_dir": work_dir,
            "layer": args.layer,
            "tensor": weight["name"],
            "batch_tokens": args.batch_tokens,
            "out_dim": out_dim,
            "in_dim": in_dim,
            "group_size": group_size,
            "token_tile": args.token_tile,
            "group32_modes": args.group32_modes,
            "repeat": args.repeat,
            "input_mode": args.input_mode,
            "records": records,
            "by_mode": by_mode,
            "config_comparison": _config_comparison(
                by_mode,
                max_promotion_drift=args.max_promotion_drift,
                min_promotion_speedup_ratio=args.min_promotion_speedup_ratio,
                min_promotion_sample_count=args.min_promotion_sample_count,
            ),
            "work_dir_cleaned": False,
        }
        if not args.keep_work_dir:
            shutil.rmtree(work_dir)
            cleaned = True
            payload["work_dir_cleaned"] = True
        if args.write_result:
            args.write_result.parent.mkdir(parents=True, exist_ok=True)
            args.write_result.write_text(
                json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
        return 0
    finally:
        if not args.keep_work_dir and not cleaned and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
