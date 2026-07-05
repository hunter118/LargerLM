#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from largerlm.prefill_execute import run_prefill_mla_attention_batch

SCHEMA = "largerlm.glm_mla_attention_cache_sweep.v1"
DEFAULT_MAX_PROMOTION_DRIFT = 1e-5
DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO = 0.98
DEFAULT_MIN_PROMOTION_SAMPLE_COUNT = 3


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _parse_cache_modes(value: str) -> tuple[str, ...]:
    modes: list[str] = []
    aliases = {
        "default": "key-value",
        "both": "key-value",
        "kv": "key-value",
        "key-value": "key-value",
        "key_value": "key-value",
        "key": "key-only",
        "key-only": "key-only",
        "key_only": "key-only",
        "value": "value-only",
        "value-only": "value-only",
        "value_only": "value-only",
        "none": "none",
        "off": "none",
        "no-cache": "none",
        "no_cache": "none",
    }
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        mode = aliases.get(item)
        if mode is None:
            raise argparse.ArgumentTypeError(
                "cache modes must be key-value, key-only, value-only, or none"
            )
        modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("cache mode list must not be empty")
    return tuple(dict.fromkeys(modes))


def _cache_mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "key-value":
        return True, True
    if mode == "key-only":
        return True, False
    if mode == "value-only":
        return False, True
    if mode == "none":
        return False, False
    raise ValueError(f"unsupported cache mode: {mode}")


def _write_f32(path: Path, *, count: int, mode: str, seed: int, scale: float) -> None:
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
            "fastest_total_mode": None,
            "fastest_kernel_mode": None,
            "candidate_mode": None,
            "candidate_for_full_replay": False,
            "requires_full_replay_bakeoff": False,
            "max_promotion_drift": max_promotion_drift,
            "min_promotion_speedup_ratio": min_promotion_speedup_ratio,
            "min_promotion_sample_count": min_promotion_sample_count,
            "rows": [],
            "reasons": ["no_modes_observed"],
        }
    baseline_mode = "key-value" if "key-value" in by_mode else next(iter(by_mode))
    baseline = by_mode.get(baseline_mode, {})
    baseline_total = _series_mean(dict(baseline.get("timing_total_seconds") or {}))
    baseline_kernel = _series_mean(dict(baseline.get("timing_kernel_seconds") or {}))
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for mode, summary in by_mode.items():
        total_series = dict(summary.get("timing_total_seconds") or {})
        kernel_series = dict(summary.get("timing_kernel_seconds") or {})
        total_mean = _series_mean(total_series)
        kernel_mean = _series_mean(kernel_series)
        sample_count = _series_count(total_series)
        max_diff = float(summary.get("max_abs_diff_output_vs_first") or 0.0)
        total_ratio = _ratio(total_mean, baseline_total)
        kernel_ratio = _ratio(kernel_mean, baseline_kernel)
        enough_samples = sample_count >= min_promotion_sample_count
        numerically_close = max_diff <= max_promotion_drift
        speedup = total_ratio is not None and total_ratio <= min_promotion_speedup_ratio
        candidate = (
            mode != baseline_mode
            and enough_samples
            and numerically_close
            and speedup
        )
        row = {
            "mode": mode,
            "is_baseline": mode == baseline_mode,
            "total_seconds_mean": total_mean,
            "total_ratio_to_baseline": total_ratio,
            "kernel_seconds_mean": kernel_mean,
            "kernel_ratio_to_baseline": kernel_ratio,
            "total_sample_count": sample_count,
            "max_abs_diff_output_vs_first": max_diff,
            "meets_min_promotion_sample_count": enough_samples,
            "numerically_within_promotion_drift": numerically_close,
            "microbench_candidate_for_full_replay": candidate,
        }
        rows.append(row)
        if candidate:
            candidate_rows.append(row)
    fastest_total = min(
        (row for row in rows if row["total_seconds_mean"] is not None),
        key=lambda row: float(row["total_seconds_mean"]),
        default=None,
    )
    fastest_kernel = min(
        (row for row in rows if row["kernel_seconds_mean"] is not None),
        key=lambda row: float(row["kernel_seconds_mean"]),
        default=None,
    )
    candidate = min(
        candidate_rows,
        key=lambda row: float(row["total_seconds_mean"]),
        default=None,
    )
    reasons: list[str] = []
    if candidate is None:
        reasons.append("no_candidate_met_total_speedup_and_drift_policy")
    else:
        reasons.append("candidate_met_total_speedup_and_drift_policy")
    if fastest_total and fastest_total["mode"] == baseline_mode:
        reasons.append("baseline_has_fastest_total_mean")
    elif fastest_total:
        reasons.append("non_baseline_has_fastest_total_mean")
    return {
        "baseline_mode": baseline_mode,
        "fastest_total_mode": (
            fastest_total["mode"] if fastest_total is not None else None
        ),
        "fastest_kernel_mode": (
            fastest_kernel["mode"] if fastest_kernel is not None else None
        ),
        "candidate_mode": candidate["mode"] if candidate is not None else None,
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
        description="Compare bounded GLM MLA attention key/value cache modes.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=53)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--qk-nope-dim", type=int, default=192)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--v-head-dim", type=int, default=256)
    parser.add_argument("--kv-lora-dim", type=int, default=512)
    parser.add_argument("--rope-theta", type=float, default=8_000_000.0)
    parser.add_argument("--rope-interleave", action="store_true", default=True)
    parser.add_argument("--no-rope-interleave", dest="rope_interleave", action="store_false")
    parser.add_argument(
        "--cache-modes",
        type=_parse_cache_modes,
        default="key-value,key-only,value-only,none",
    )
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--input-mode", choices=("random", "zero"), default="random")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--q-scale", type=float, default=0.02)
    parser.add_argument("--max-cache-file-mib", type=int, default=32768)
    parser.add_argument("--max-cache-read-mib", type=int, default=256)
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

    for name in (
        "layer",
        "context_length",
        "batch_tokens",
        "num_heads",
        "qk_nope_dim",
        "rope_dim",
        "v_head_dim",
        "kv_lora_dim",
        "repeat",
        "max_cache_file_mib",
        "max_cache_read_mib",
        "max_resident_matrix_mib",
        "max_runner_scratch_mib",
    ):
        if getattr(args, name) <= 0 and name != "layer":
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.layer < 0 or args.start_position < 0:
        raise SystemExit("--layer and --start-position must be non-negative")
    if args.start_position + args.batch_tokens > args.context_length:
        raise SystemExit("--start-position + --batch-tokens exceeds --context-length")
    if args.max_promotion_drift < 0:
        raise SystemExit("--max-promotion-drift must be non-negative")
    if not (0 < args.min_promotion_speedup_ratio < 1):
        raise SystemExit("--min-promotion-speedup-ratio must be between 0 and 1")
    if args.min_promotion_sample_count <= 0:
        raise SystemExit("--min-promotion-sample-count must be positive")

    resident_layout = args.prepared_dir / "resident" / "layout.json"
    cache_layout = args.prepared_dir / "decode_cache_layout.json"
    cache_file = args.prepared_dir / "decode_cache.bin"
    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-glm-mla-cache-sweep-"))
    cleaned = False
    try:
        q_nope = work_dir / "q_nope.f32"
        q_rope = work_dir / "q_rope.f32"
        _write_f32(
            q_nope,
            count=args.batch_tokens * args.num_heads * args.qk_nope_dim,
            mode=args.input_mode,
            seed=args.seed,
            scale=args.q_scale,
        )
        _write_f32(
            q_rope,
            count=args.batch_tokens * args.num_heads * args.rope_dim,
            mode=args.input_mode,
            seed=args.seed + 1,
            scale=args.q_scale,
        )

        schedule: list[tuple[str, int]] = []
        if args.order == "grouped":
            for mode in args.cache_modes:
                for index in range(args.repeat):
                    schedule.append((mode, index))
        else:
            for index in range(args.repeat):
                for mode in args.cache_modes:
                    schedule.append((mode, index))

        output_count = args.batch_tokens * args.num_heads * args.v_head_dim
        baseline_output: array.array[float] | None = None
        records: list[dict[str, Any]] = []
        for mode, repeat_index in schedule:
            mla_key_cache, mla_value_cache = _cache_mode_flags(mode)
            output = work_dir / f"output_{mode}_rep{repeat_index:02d}.f32"
            started = time.perf_counter()
            result = run_prefill_mla_attention_batch(
                runner_path=args.runner,
                resident_layout_path=resident_layout,
                cache_layout_path=cache_layout,
                cache_file_path=cache_file,
                layer=args.layer,
                q_nope_f32_path=q_nope,
                q_rope_f32_path=q_rope,
                output_f32_path=output,
                context_length=args.context_length,
                start_position=args.start_position,
                batch_tokens=args.batch_tokens,
                num_heads=args.num_heads,
                qk_nope_dim=args.qk_nope_dim,
                rope_dim=args.rope_dim,
                v_head_dim=args.v_head_dim,
                kv_lora_dim=args.kv_lora_dim,
                mla_key_cache=mla_key_cache,
                mla_value_cache=mla_value_cache,
                rope_theta=args.rope_theta,
                rope_interleave=args.rope_interleave,
                max_cache_file_mib=args.max_cache_file_mib,
                max_cache_read_mib=args.max_cache_read_mib,
                max_resident_matrix_mib=args.max_resident_matrix_mib,
                max_runner_scratch_mib=args.max_runner_scratch_mib,
                echo_runner_output=False,
            )
            wall_elapsed = time.perf_counter() - started
            values = _read_f32(output, output_count)
            if baseline_output is None:
                baseline_output = values
                max_abs_diff = 0.0
            else:
                max_abs_diff = _max_abs_diff(baseline_output, values)
            value_cache_disabled_reason = None
            if mla_value_cache and not result.mla_value_cache:
                value_cache_disabled_reason = "scratch_cap_or_indexed_path"
            records.append(
                {
                    "cache_mode": mode,
                    "repeat_index": repeat_index,
                    "requested_mla_key_cache": mla_key_cache,
                    "requested_mla_value_cache": mla_value_cache,
                    "mla_key_cache": result.mla_key_cache,
                    "mla_value_cache": result.mla_value_cache,
                    "mla_value_cache_disabled_reason": value_cache_disabled_reason,
                    "mla_key_cache_bytes": result.mla_key_cache_bytes,
                    "mla_value_cache_bytes": result.mla_value_cache_bytes,
                    "estimated_peak_bytes": result.estimated_peak_bytes,
                    "cache_read_bytes": result.cache_read_bytes,
                    "cache_f32_bytes": result.cache_f32_bytes,
                    "kv_b_matrix_bytes": result.kv_b_matrix_bytes,
                    "kv_b_f32_bytes": result.kv_b_f32_bytes,
                    "output_bytes": result.output_bytes,
                    "attention_value_source": result.attention_value_source,
                    "wall_elapsed_seconds": wall_elapsed,
                    "timing_input_seconds": result.mla_timing_input_elapsed_seconds,
                    "timing_cache_read_seconds": (
                        result.mla_timing_cache_read_elapsed_seconds
                    ),
                    "timing_value_read_seconds": (
                        result.mla_timing_value_read_elapsed_seconds
                    ),
                    "timing_metal_setup_seconds": (
                        result.mla_timing_metal_setup_elapsed_seconds
                    ),
                    "timing_kernel_seconds": result.mla_timing_kernel_elapsed_seconds,
                    "timing_write_seconds": result.mla_timing_write_elapsed_seconds,
                    "timing_total_seconds": result.mla_timing_total_elapsed_seconds,
                    "max_abs_diff_output_vs_first": max_abs_diff,
                }
            )

        by_mode: dict[str, dict[str, Any]] = {}
        for mode in args.cache_modes:
            mode_records = [record for record in records if record["cache_mode"] == mode]
            by_mode[mode] = {
                "wall_elapsed_seconds": _series(
                    [float(record["wall_elapsed_seconds"]) for record in mode_records]
                ),
                "timing_input_seconds": _series(
                    [
                        float(record["timing_input_seconds"])
                        for record in mode_records
                        if record["timing_input_seconds"] is not None
                    ]
                ),
                "timing_cache_read_seconds": _series(
                    [
                        float(record["timing_cache_read_seconds"])
                        for record in mode_records
                        if record["timing_cache_read_seconds"] is not None
                    ]
                ),
                "timing_value_read_seconds": _series(
                    [
                        float(record["timing_value_read_seconds"])
                        for record in mode_records
                        if record["timing_value_read_seconds"] is not None
                    ]
                ),
                "timing_metal_setup_seconds": _series(
                    [
                        float(record["timing_metal_setup_seconds"])
                        for record in mode_records
                        if record["timing_metal_setup_seconds"] is not None
                    ]
                ),
                "timing_kernel_seconds": _series(
                    [
                        float(record["timing_kernel_seconds"])
                        for record in mode_records
                        if record["timing_kernel_seconds"] is not None
                    ]
                ),
                "timing_write_seconds": _series(
                    [
                        float(record["timing_write_seconds"])
                        for record in mode_records
                        if record["timing_write_seconds"] is not None
                    ]
                ),
                "timing_total_seconds": _series(
                    [
                        float(record["timing_total_seconds"])
                        for record in mode_records
                        if record["timing_total_seconds"] is not None
                    ]
                ),
                "max_abs_diff_output_vs_first": max(
                    (
                        float(record["max_abs_diff_output_vs_first"])
                        for record in mode_records
                    ),
                    default=0.0,
                ),
                "max_estimated_peak_bytes": max(
                    (
                        int(record["estimated_peak_bytes"])
                        for record in mode_records
                        if record["estimated_peak_bytes"] is not None
                    ),
                    default=0,
                ),
                "observed_mla_key_cache": sorted(
                    {bool(record["mla_key_cache"]) for record in mode_records}
                ),
                "observed_mla_value_cache": sorted(
                    {bool(record["mla_value_cache"]) for record in mode_records}
                ),
                "requested_mla_value_cache_disabled_count": sum(
                    1
                    for record in mode_records
                    if record.get("requested_mla_value_cache")
                    and not record.get("mla_value_cache")
                ),
                "requested_mla_value_cache_disabled_reasons": sorted(
                    {
                        str(record["mla_value_cache_disabled_reason"])
                        for record in mode_records
                        if record.get("mla_value_cache_disabled_reason")
                    }
                ),
            }

        payload = {
            "schema": SCHEMA,
            "prepared_dir": args.prepared_dir,
            "runner": args.runner,
            "resident_layout": resident_layout,
            "cache_layout": cache_layout,
            "cache_file": cache_file,
            "work_dir": work_dir,
            "layer": args.layer,
            "context_length": args.context_length,
            "start_position": args.start_position,
            "batch_tokens": args.batch_tokens,
            "num_heads": args.num_heads,
            "qk_nope_dim": args.qk_nope_dim,
            "rope_dim": args.rope_dim,
            "v_head_dim": args.v_head_dim,
            "kv_lora_dim": args.kv_lora_dim,
            "rope_theta": args.rope_theta,
            "rope_interleave": args.rope_interleave,
            "cache_modes": args.cache_modes,
            "repeat": args.repeat,
            "order": args.order,
            "input_mode": args.input_mode,
            "max_cache_file_mib": args.max_cache_file_mib,
            "max_cache_read_mib": args.max_cache_read_mib,
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
        if not args.keep_work_dir and not cleaned and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
