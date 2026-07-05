from __future__ import annotations

import pytest

from scripts.glm_mla_attention_cache_sweep import (
    _cache_mode_flags,
    _config_comparison,
    _parse_cache_modes,
)


def _summary(
    *,
    total_mean: float,
    kernel_mean: float,
    max_diff: float,
    count: float = 3.0,
) -> dict[str, object]:
    return {
        "timing_total_seconds": {
            "count": count,
            "min": total_mean,
            "mean": total_mean,
            "max": total_mean,
            "stdev": 0.0,
        },
        "timing_kernel_seconds": {
            "count": count,
            "min": kernel_mean,
            "mean": kernel_mean,
            "max": kernel_mean,
            "stdev": 0.0,
        },
        "max_abs_diff_output_vs_first": max_diff,
    }


def test_parse_cache_modes_normalizes_aliases() -> None:
    assert _parse_cache_modes("default,key,value,off,kv") == (
        "key-value",
        "key-only",
        "value-only",
        "none",
    )
    assert _cache_mode_flags("key-value") == (True, True)
    assert _cache_mode_flags("key-only") == (True, False)
    assert _cache_mode_flags("value-only") == (False, True)
    assert _cache_mode_flags("none") == (False, False)


def test_parse_cache_modes_rejects_empty_or_unknown() -> None:
    with pytest.raises(argparse_error()):
        _parse_cache_modes("")
    with pytest.raises(argparse_error()):
        _parse_cache_modes("mystery")


def argparse_error() -> type[Exception]:
    import argparse

    return argparse.ArgumentTypeError


def test_config_comparison_keeps_key_value_baseline_when_fastest() -> None:
    comparison = _config_comparison(
        {
            "key-value": _summary(total_mean=1.0, kernel_mean=0.5, max_diff=0.0),
            "key-only": _summary(total_mean=1.2, kernel_mean=0.6, max_diff=1e-6),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["baseline_mode"] == "key-value"
    assert comparison["fastest_total_mode"] == "key-value"
    assert comparison["candidate_mode"] is None
    assert comparison["candidate_for_full_replay"] is False
    assert comparison["reasons"] == [
        "no_candidate_met_total_speedup_and_drift_policy",
        "baseline_has_fastest_total_mean",
    ]


def test_config_comparison_can_promote_faster_cache_mode() -> None:
    comparison = _config_comparison(
        {
            "key-value": _summary(total_mean=1.0, kernel_mean=0.5, max_diff=0.0),
            "key-only": _summary(total_mean=0.9, kernel_mean=0.45, max_diff=1e-6),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["candidate_mode"] == "key-only"
    assert comparison["candidate_for_full_replay"] is True
    candidate = next(row for row in comparison["rows"] if row["mode"] == "key-only")
    assert candidate["total_ratio_to_baseline"] == 0.9
    assert candidate["microbench_candidate_for_full_replay"] is True


def test_config_comparison_rejects_drifting_cache_mode() -> None:
    comparison = _config_comparison(
        {
            "key-value": _summary(total_mean=1.0, kernel_mean=0.5, max_diff=0.0),
            "key-only": _summary(total_mean=0.9, kernel_mean=0.45, max_diff=1e-3),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["candidate_mode"] is None
    candidate = next(row for row in comparison["rows"] if row["mode"] == "key-only")
    assert candidate["numerically_within_promotion_drift"] is False
    assert candidate["microbench_candidate_for_full_replay"] is False


def test_config_comparison_requires_minimum_cache_samples() -> None:
    comparison = _config_comparison(
        {
            "key-value": _summary(total_mean=1.0, kernel_mean=0.5, max_diff=0.0),
            "key-only": _summary(
                total_mean=0.9,
                kernel_mean=0.45,
                max_diff=1e-6,
                count=1.0,
            ),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["candidate_mode"] is None
    candidate = next(row for row in comparison["rows"] if row["mode"] == "key-only")
    assert candidate["total_sample_count"] == 1
    assert candidate["meets_min_promotion_sample_count"] is False
    assert candidate["microbench_candidate_for_full_replay"] is False
