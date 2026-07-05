from __future__ import annotations

from scripts.glm_resident_mxfp4_group32_sweep import _config_comparison


def _summary(
    *,
    backend_mean: float,
    wall_mean: float,
    max_diff: float,
    count: float = 3.0,
) -> dict[str, object]:
    return {
        "timing_backend_seconds": {
            "count": count,
            "min": backend_mean,
            "mean": backend_mean,
            "max": backend_mean,
            "stdev": 0.0,
        },
        "elapsed_wall_seconds": {
            "count": count,
            "min": wall_mean,
            "mean": wall_mean,
            "max": wall_mean,
            "stdev": 0.0,
        },
        "max_abs_diff_projection_vs_first": max_diff,
        "max_abs_diff_output_vs_first": max_diff,
    }


def test_config_comparison_recommends_faster_resident_mode() -> None:
    comparison = _config_comparison(
        {
            "off": _summary(backend_mean=1.0, wall_mean=1.2, max_diff=0.0),
            "auto": _summary(backend_mean=0.8, wall_mean=1.0, max_diff=1e-6),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["baseline_mode"] == "auto"
    assert comparison["fastest_backend_mode"] == "auto"
    assert comparison["candidate_mode"] is None
    assert comparison["candidate_for_full_replay"] is False
    assert comparison["reasons"] == [
        "no_candidate_met_backend_speedup_and_drift_policy",
        "baseline_has_fastest_backend_mean",
    ]


def test_config_comparison_can_promote_non_baseline_resident_mode() -> None:
    comparison = _config_comparison(
        {
            "auto": _summary(backend_mean=1.0, wall_mean=1.2, max_diff=0.0),
            "off": _summary(backend_mean=0.9, wall_mean=1.1, max_diff=1e-6),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["baseline_mode"] == "auto"
    assert comparison["fastest_backend_mode"] == "off"
    assert comparison["candidate_mode"] == "off"
    assert comparison["candidate_for_full_replay"] is True
    candidate = next(row for row in comparison["rows"] if row["mode"] == "off")
    assert candidate["backend_ratio_to_baseline"] == 0.9
    assert candidate["microbench_candidate_for_full_replay"] is True


def test_config_comparison_rejects_drifting_resident_mode() -> None:
    comparison = _config_comparison(
        {
            "auto": _summary(backend_mean=1.0, wall_mean=1.2, max_diff=0.0),
            "off": _summary(backend_mean=0.9, wall_mean=1.1, max_diff=1e-3),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["candidate_mode"] is None
    candidate = next(row for row in comparison["rows"] if row["mode"] == "off")
    assert candidate["numerically_within_promotion_drift"] is False
    assert candidate["microbench_candidate_for_full_replay"] is False


def test_config_comparison_requires_minimum_resident_samples() -> None:
    comparison = _config_comparison(
        {
            "auto": _summary(backend_mean=1.0, wall_mean=1.2, max_diff=0.0),
            "off": _summary(
                backend_mean=0.9,
                wall_mean=1.1,
                max_diff=1e-6,
                count=1.0,
            ),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["candidate_mode"] is None
    candidate = next(row for row in comparison["rows"] if row["mode"] == "off")
    assert candidate["backend_sample_count"] == 1
    assert candidate["meets_min_promotion_sample_count"] is False
    assert candidate["microbench_candidate_for_full_replay"] is False
