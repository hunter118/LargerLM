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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from largerlm.prefill_execute import run_prefill_rope_batch

DEFAULT_MAX_PROMOTION_DRIFT = 1e-5
DEFAULT_MIN_PROMOTION_SPEEDUP_RATIO = 0.98
DEFAULT_MIN_PROMOTION_SAMPLE_COUNT = 3


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


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
        dict(by_mode.get(baseline_mode, {}).get("elapsed_seconds") or {})
    )
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for mode, summary in by_mode.items():
        elapsed_series = dict(summary.get("elapsed_seconds") or {})
        wall_mean = _series_mean(elapsed_series)
        sample_count = _series_count(elapsed_series)
        diffs = summary.get("max_abs_diff_vs_first")
        max_diff = (
            max(float(value) for value in diffs.values())
            if isinstance(diffs, dict) and diffs
            else 0.0
        )
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


def _write_random_f32(path: Path, *, count: int, seed: int, scale: float) -> None:
    rng = random.Random(seed)
    values = array.array("f", (rng.uniform(-scale, scale) for _ in range(count)))
    with path.open("wb") as handle:
        values.tofile(handle)


def _read_f32(path: Path) -> array.array[float]:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // 4)
    return values


def _max_abs_diff(lhs: array.array[float], rhs: array.array[float]) -> float:
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare old and fused GLM-sized RoPE split batch paths.",
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--qk-nope-dim", type=int, default=128)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-interleave", action="store_true")
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--input-scale", type=float, default=0.02)
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

    if args.batch_tokens <= 0 or args.num_heads <= 0:
        raise SystemExit("batch tokens and num heads must be positive")
    if args.qk_nope_dim <= 0 or args.rope_dim <= 0 or args.rope_dim % 2 != 0:
        raise SystemExit("qk nope dim must be positive and rope dim must be positive/even")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.max_promotion_drift < 0:
        raise SystemExit("--max-promotion-drift must be non-negative")
    if not (0 < args.min_promotion_speedup_ratio < 1):
        raise SystemExit("--min-promotion-speedup-ratio must be between 0 and 1")
    if args.min_promotion_sample_count <= 0:
        raise SystemExit("--min-promotion-sample-count must be positive")

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-glm-rope-split-"))
    cleaned = False
    old_disable = os.environ.get("LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH")
    try:
        q_b = work_dir / "q_b.f32"
        k_rope = work_dir / "k_rope.f32"
        _write_random_f32(
            q_b,
            count=args.batch_tokens
            * args.num_heads
            * (args.qk_nope_dim + args.rope_dim),
            seed=args.seed,
            scale=args.input_scale,
        )
        _write_random_f32(
            k_rope,
            count=args.batch_tokens * args.rope_dim,
            seed=args.seed + 1,
            scale=args.input_scale,
        )

        schedule: list[tuple[str, int]] = []
        modes = ("old", "fused")
        if args.order == "grouped":
            for mode in modes:
                for index in range(args.repeat):
                    schedule.append((mode, index))
        else:
            for index in range(args.repeat):
                for mode in modes:
                    schedule.append((mode, index))

        baseline: dict[str, array.array[float]] | None = None
        records: list[dict[str, Any]] = []
        for mode, index in schedule:
            if mode == "old":
                os.environ["LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH"] = "1"
            else:
                os.environ.pop("LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH", None)
            output_dir = work_dir / f"{mode}_rep{index:02d}"
            started = time.perf_counter()
            result = run_prefill_rope_batch(
                runner_path=args.runner,
                q_b_f32_path=q_b,
                k_rope_f32_path=k_rope,
                output_dir=output_dir,
                batch_tokens=args.batch_tokens,
                num_heads=args.num_heads,
                qk_nope_dim=args.qk_nope_dim,
                rope_dim=args.rope_dim,
                start_position=args.start_position,
                rope_theta=args.rope_theta,
                rope_interleave=args.rope_interleave,
                max_runner_scratch_mib=args.max_runner_scratch_mib,
                echo_runner_output=False,
            )
            elapsed = time.perf_counter() - started
            outputs = {
                "q_nope": _read_f32(result.q_nope_path),
                "q_rope": _read_f32(result.q_rope_path),
                "q_rotated": _read_f32(result.q_rope_rotated_path),
                "k_rotated": _read_f32(result.k_rope_rotated_path),
            }
            if baseline is None:
                baseline = outputs
                diffs = {name: 0.0 for name in outputs}
            else:
                diffs = {
                    name: _max_abs_diff(baseline[name], values)
                    for name, values in outputs.items()
                }
            records.append(
                {
                    "mode": mode,
                    "repeat_index": index,
                    "elapsed_seconds": elapsed,
                    "command": result.command,
                    "estimated_peak_bytes": result.estimated_peak_bytes,
                    "max_abs_diff_vs_first": diffs,
                }
            )

        by_mode: dict[str, dict[str, Any]] = {}
        for mode in modes:
            mode_records = [record for record in records if record["mode"] == mode]
            by_mode[mode] = {
                "elapsed_seconds": _series(
                    [float(record["elapsed_seconds"]) for record in mode_records]
                ),
                "estimated_peak_bytes": _series(
                    [float(record["estimated_peak_bytes"]) for record in mode_records]
                ),
                "max_abs_diff_vs_first": {
                    name: max(
                        (
                            float(record["max_abs_diff_vs_first"][name])
                            for record in mode_records
                        ),
                        default=0.0,
                    )
                    for name in ("q_nope", "q_rope", "q_rotated", "k_rotated")
                },
                "commands": sorted(
                    {
                        str(record["command"][1])
                        for record in mode_records
                        if len(record["command"]) > 1
                    }
                ),
            }

        payload = {
            "schema": "largerlm.glm_rope_split_sweep.v1",
            "runner": args.runner,
            "work_dir": work_dir,
            "batch_tokens": args.batch_tokens,
            "num_heads": args.num_heads,
            "qk_nope_dim": args.qk_nope_dim,
            "rope_dim": args.rope_dim,
            "start_position": args.start_position,
            "rope_theta": args.rope_theta,
            "rope_interleave": args.rope_interleave,
            "repeat": args.repeat,
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
        if old_disable is None:
            os.environ.pop("LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH", None)
        else:
            os.environ["LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH"] = old_disable
        if not args.keep_work_dir and not cleaned and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
