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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from largerlm.decode_cache import DecodeCacheSegment, load_decode_cache_layout
from largerlm.prefill_execute import (
    PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV,
    write_prefill_kv_cache_batch,
)

DEFAULT_PREPARED_DIR = Path("artifacts/glm-5.2-mxfp4/largerlm-prepared")
DEFAULT_MAX_PROMOTION_DIFF_BYTES = 0
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


def _parse_chunk_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    return modes or ["default"]


def _chunk_bytes_for_mode(mode: str, *, transient_bytes_per_token: int) -> int | None:
    lowered = mode.lower()
    if lowered == "default":
        return None
    if lowered in {"one-row", "row", "scalar-row"}:
        return transient_bytes_per_token
    suffixes = (("kib", 1024), ("kb", 1000), ("mib", 1024**2), ("mb", 1000**2))
    for suffix, scale in suffixes:
        if lowered.endswith(suffix):
            number = lowered[: -len(suffix)]
            if not number:
                raise ValueError(f"invalid chunk mode {mode!r}")
            return int(float(number) * scale)
    return int(lowered, 10)


@contextmanager
def _temporary_chunk_bytes(value: int | None) -> Iterator[None]:
    previous = os.environ.get(PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV)
    try:
        if value is None:
            os.environ.pop(PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV, None)
        else:
            os.environ[PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV] = str(value)
        yield
    finally:
        if previous is None:
            os.environ.pop(PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV, None)
        else:
            os.environ[PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV] = previous


def _write_random_f32(path: Path, *, count: int, seed: int, scale: float) -> None:
    rng = random.Random(seed)
    values = array.array("f", (rng.uniform(-scale, scale) for _ in range(count)))
    with path.open("wb") as handle:
        values.tofile(handle)


def _write_synthetic_cache_layout(path: Path, *, segment: DecodeCacheSegment) -> int:
    total_bytes = segment.token_stride_bytes * segment.max_context_tokens
    payload = {
        "version": 1,
        "model_type": "glm_moe_dsa_synthetic_cache_write_sweep",
        "max_context_tokens": segment.max_context_tokens,
        "dtype": segment.dtype,
        "dtype_bytes": segment.dtype_bytes,
        "alignment": 1,
        "total_bytes": total_bytes,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": segment.layer,
                "offset": 0,
                "width": segment.width,
                "dtype": segment.dtype,
                "dtype_bytes": segment.dtype_bytes,
                "token_stride_bytes": segment.token_stride_bytes,
                "max_context_tokens": segment.max_context_tokens,
                "total_bytes": total_bytes,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return total_bytes


def _first_mla_segment(prepared_dir: Path, *, layer: int | None) -> DecodeCacheSegment:
    layout = load_decode_cache_layout(prepared_dir / "decode_cache_layout.json")
    segments = [segment for segment in layout.segments if segment.kind == "mla_kv"]
    if layer is not None:
        segments = [segment for segment in segments if segment.layer == layer]
    if not segments:
        suffix = f" for layer {layer}" if layer is not None else ""
        raise SystemExit(f"no mla_kv cache segment found{suffix}")
    return segments[0]


def _max_diff_bytes(lhs: bytes, rhs: bytes) -> int:
    length_delta = abs(len(lhs) - len(rhs))
    diff = sum(1 for a, b in zip(lhs, rhs) if a != b)
    return diff + length_delta


def _config_comparison(
    by_mode: dict[str, dict[str, Any]],
    *,
    max_promotion_diff_bytes: int = DEFAULT_MAX_PROMOTION_DIFF_BYTES,
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
            "rows": [],
            "reasons": ["no_modes_observed"],
        }
    baseline_mode = "default" if "default" in by_mode else next(iter(by_mode))
    baseline_wall = _series_mean(
        dict(by_mode.get(baseline_mode, {}).get("elapsed_seconds") or {})
    )
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for mode, summary in by_mode.items():
        elapsed_series = dict(summary.get("elapsed_seconds") or {})
        wall_mean = _series_mean(elapsed_series)
        sample_count = _series_count(elapsed_series)
        diff_bytes = int(summary.get("max_diff_bytes_vs_baseline") or 0)
        wall_ratio = _ratio(wall_mean, baseline_wall)
        enough_samples = sample_count >= min_promotion_sample_count
        byte_identical = diff_bytes <= max_promotion_diff_bytes
        speedup = (
            wall_ratio is not None and wall_ratio <= min_promotion_speedup_ratio
        )
        candidate = (
            mode != baseline_mode
            and enough_samples
            and byte_identical
            and speedup
        )
        row = {
            "mode": mode,
            "is_baseline": mode == baseline_mode,
            "wall_seconds_mean": wall_mean,
            "wall_ratio_to_baseline": wall_ratio,
            "wall_sample_count": sample_count,
            "max_diff_bytes_vs_baseline": diff_bytes,
            "meets_min_promotion_sample_count": enough_samples,
            "byte_identical_within_promotion_policy": byte_identical,
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
        reasons.append("no_candidate_met_wall_speedup_and_byte_policy")
    else:
        reasons.append("candidate_met_wall_speedup_and_byte_policy")
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
        "max_promotion_diff_bytes": max_promotion_diff_bytes,
        "min_promotion_speedup_ratio": min_promotion_speedup_ratio,
        "min_promotion_sample_count": min_promotion_sample_count,
        "rows": rows,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep bounded GLM prefill KV cache write chunk sizes.",
    )
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument(
        "--chunk-modes",
        type=_parse_chunk_modes,
        default=_parse_chunk_modes("default,one-row,16KiB,256KiB,1MiB"),
    )
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--max-cache-file-mib", type=float, default=256.0)
    parser.add_argument("--max-cache-write-mib", type=float, default=256.0)
    parser.add_argument(
        "--max-promotion-diff-bytes",
        type=int,
        default=DEFAULT_MAX_PROMOTION_DIFF_BYTES,
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

    if args.batch_tokens <= 0 or args.repeat <= 0:
        raise SystemExit("batch tokens and repeat must be positive")
    if args.start_position < 0:
        raise SystemExit("start position must be non-negative")
    if args.max_promotion_diff_bytes < 0:
        raise SystemExit("max promotion diff bytes must be non-negative")

    source_segment = _first_mla_segment(args.prepared_dir, layer=args.layer)
    synthetic_segment = DecodeCacheSegment(
        kind="mla_kv",
        layer=source_segment.layer,
        offset=0,
        width=source_segment.width,
        dtype=source_segment.dtype,
        dtype_bytes=source_segment.dtype_bytes,
        max_context_tokens=args.start_position + args.batch_tokens,
    )
    input_row_bytes = synthetic_segment.width * 4
    transient_bytes_per_token = input_row_bytes + synthetic_segment.token_stride_bytes
    mode_chunk_bytes = {
        mode: _chunk_bytes_for_mode(
            mode,
            transient_bytes_per_token=transient_bytes_per_token,
        )
        for mode in args.chunk_modes
    }

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-cache-write-sweep-"))
    records: list[dict[str, Any]] = []
    cache_snapshots: dict[str, bytes] = {}
    try:
        cache_layout = work_dir / "decode_cache_layout.json"
        total_cache_bytes = _write_synthetic_cache_layout(
            cache_layout,
            segment=synthetic_segment,
        )
        input_path = work_dir / "kv_a.f32"
        _write_random_f32(
            input_path,
            count=args.batch_tokens * synthetic_segment.width,
            seed=args.seed,
            scale=args.input_scale,
        )
        sequence = list(args.chunk_modes) * args.repeat
        if args.order == "grouped":
            sequence = [mode for mode in args.chunk_modes for _ in range(args.repeat)]
        for run_index, mode in enumerate(sequence):
            cache_file = work_dir / f"decode_cache_{run_index}_{mode}.bin"
            cache_file.write_bytes(b"\0" * total_cache_bytes)
            requested_chunk_bytes = mode_chunk_bytes[mode]
            started = time.perf_counter()
            with _temporary_chunk_bytes(requested_chunk_bytes):
                result = write_prefill_kv_cache_batch(
                    cache_layout_path=cache_layout,
                    cache_file_path=cache_file,
                    layer=synthetic_segment.layer,
                    input_f32_path=input_path,
                    start_position=args.start_position,
                    batch_tokens=args.batch_tokens,
                    max_cache_file_mib=args.max_cache_file_mib,
                    max_cache_write_mib=args.max_cache_write_mib,
                )
            elapsed = time.perf_counter() - started
            snapshot = cache_file.read_bytes()
            cache_snapshots.setdefault(mode, snapshot)
            records.append(
                {
                    "run_index": run_index,
                    "mode": mode,
                    "elapsed_seconds": elapsed,
                    "requested_chunk_bytes": requested_chunk_bytes,
                    "encoder": result.encoder,
                    "write_chunks": result.write_chunks,
                    "write_chunk_tokens": result.write_chunk_tokens,
                    "write_chunk_bytes": result.write_chunk_bytes,
                    "estimated_peak_bytes": result.estimated_peak_bytes,
                    "encoded_bytes": result.encoded_bytes,
                }
            )
        baseline_mode = "default" if "default" in cache_snapshots else args.chunk_modes[0]
        baseline_snapshot = cache_snapshots[baseline_mode]
        by_mode: dict[str, dict[str, Any]] = {}
        for mode in args.chunk_modes:
            mode_records = [record for record in records if record["mode"] == mode]
            elapsed_values = [
                float(record["elapsed_seconds"]) for record in mode_records
            ]
            peak_values = [
                int(record["estimated_peak_bytes"]) for record in mode_records
            ]
            by_mode[mode] = {
                "elapsed_seconds": _series(elapsed_values),
                "estimated_peak_bytes": _series([float(value) for value in peak_values]),
                "requested_chunk_bytes": mode_chunk_bytes[mode],
                "observed_write_chunks": sorted(
                    {int(record["write_chunks"]) for record in mode_records}
                ),
                "observed_write_chunk_tokens": sorted(
                    {int(record["write_chunk_tokens"]) for record in mode_records}
                ),
                "max_diff_bytes_vs_baseline": _max_diff_bytes(
                    cache_snapshots[mode],
                    baseline_snapshot,
                ),
            }
        payload: dict[str, Any] = {
            "schema": "largerlm.glm_prefill_cache_write_sweep.v1",
            "prepared_dir": args.prepared_dir,
            "source_cache_layout": args.prepared_dir / "decode_cache_layout.json",
            "layer": synthetic_segment.layer,
            "batch_tokens": args.batch_tokens,
            "start_position": args.start_position,
            "width": synthetic_segment.width,
            "dtype": synthetic_segment.dtype,
            "dtype_bytes": synthetic_segment.dtype_bytes,
            "token_stride_bytes": synthetic_segment.token_stride_bytes,
            "input_bytes": args.batch_tokens * input_row_bytes,
            "encoded_bytes": args.batch_tokens * synthetic_segment.token_stride_bytes,
            "transient_bytes_per_token": transient_bytes_per_token,
            "repeat": args.repeat,
            "order": args.order,
            "records": records,
            "by_mode": by_mode,
            "config_comparison": _config_comparison(
                by_mode,
                max_promotion_diff_bytes=args.max_promotion_diff_bytes,
                min_promotion_speedup_ratio=args.min_promotion_speedup_ratio,
                min_promotion_sample_count=args.min_promotion_sample_count,
            ),
            "work_dir": work_dir,
            "work_dir_cleaned": not args.keep_work_dir,
        }
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    text = json.dumps(payload, default=_json_default, indent=2, sort_keys=True)
    if args.write_result is not None:
        args.write_result.parent.mkdir(parents=True, exist_ok=True)
        args.write_result.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
