from __future__ import annotations

from scripts.glm_moe_tile_sweep import _config_comparison, _config_skip_reason


def test_config_skip_reason_allows_default_tile1_group32_auto() -> None:
    assert (
        _config_skip_reason(tile=1, vector_swiglu=False, group32_mode="auto")
        is None
    )
    assert (
        _config_skip_reason(tile=1, vector_swiglu=False, group32_mode="on")
        is None
    )


def test_config_skip_reason_rejects_group32_on_multi_token_tiles() -> None:
    reason = _config_skip_reason(tile=2, vector_swiglu=False, group32_mode="on")
    assert reason is not None
    assert "token tile 1" in reason


def test_config_skip_reason_keeps_vector_swiglu_unambiguous() -> None:
    assert (
        _config_skip_reason(tile=1, vector_swiglu=True, group32_mode="off")
        is None
    )

    reason = _config_skip_reason(tile=1, vector_swiglu=True, group32_mode="auto")
    assert reason is not None
    assert "ambiguous auto labels" in reason


def test_config_skip_reason_rejects_vector_swiglu_multi_token_tiles() -> None:
    reason = _config_skip_reason(tile=4, vector_swiglu=True, group32_mode="off")
    assert reason is not None
    assert "only implemented" in reason


def _summary(
    *,
    kernel_mean: float,
    runner_mean: float,
    max_diff: float,
) -> dict[str, object]:
    return {
        "kernel_seconds": {
            "count": 3.0,
            "min": kernel_mean,
            "mean": kernel_mean,
            "max": kernel_mean,
            "stdev": 0.0,
        },
        "runner_total_seconds": {
            "count": 3.0,
            "min": runner_mean,
            "mean": runner_mean,
            "max": runner_mean,
            "stdev": 0.0,
        },
        "max_abs_diff_vs_first": max_diff,
    }


def test_config_comparison_recommends_kernel_candidate() -> None:
    comparison = _config_comparison(
        {
            "tile1_scalar_silu": _summary(
                kernel_mean=1.0,
                runner_mean=1.2,
                max_diff=0.0,
            ),
            "tile1_vector_silu": _summary(
                kernel_mean=0.9,
                runner_mean=1.1,
                max_diff=1e-6,
            ),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["baseline_config"] == "tile1_scalar_silu"
    assert comparison["fastest_kernel_config"] == "tile1_vector_silu"
    assert comparison["candidate_config"] == "tile1_vector_silu"
    assert comparison["candidate_for_full_replay"] is True
    assert comparison["requires_full_replay_bakeoff"] is True
    candidate_row = next(
        row for row in comparison["rows"] if row["config"] == "tile1_vector_silu"
    )
    assert candidate_row["kernel_ratio_to_baseline"] == 0.9
    assert candidate_row["microbench_candidate_for_full_replay"] is True


def test_config_comparison_rejects_fast_but_drifting_candidate() -> None:
    comparison = _config_comparison(
        {
            "tile1_auto_silu": _summary(
                kernel_mean=1.0,
                runner_mean=1.2,
                max_diff=0.0,
            ),
            "tile1_vector_silu": _summary(
                kernel_mean=0.9,
                runner_mean=1.1,
                max_diff=1e-3,
            ),
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["baseline_config"] == "tile1_auto_silu"
    assert comparison["fastest_kernel_config"] == "tile1_vector_silu"
    assert comparison["candidate_config"] is None
    assert comparison["candidate_for_full_replay"] is False
    candidate_row = next(
        row for row in comparison["rows"] if row["config"] == "tile1_vector_silu"
    )
    assert candidate_row["numerically_within_promotion_drift"] is False
    assert candidate_row["microbench_candidate_for_full_replay"] is False


def test_config_comparison_requires_minimum_sample_count() -> None:
    comparison = _config_comparison(
        {
            "tile1_auto_silu": _summary(
                kernel_mean=1.0,
                runner_mean=1.2,
                max_diff=0.0,
            ),
            "tile1_scalar_silu": {
                **_summary(
                    kernel_mean=0.9,
                    runner_mean=1.1,
                    max_diff=1e-6,
                ),
                "kernel_seconds": {
                    "count": 1.0,
                    "min": 0.9,
                    "mean": 0.9,
                    "max": 0.9,
                    "stdev": 0.0,
                },
            },
        },
        max_promotion_drift=1e-5,
        min_promotion_speedup_ratio=0.98,
    )

    assert comparison["candidate_config"] is None
    assert comparison["candidate_for_full_replay"] is False
    candidate_row = next(
        row for row in comparison["rows"] if row["config"] == "tile1_scalar_silu"
    )
    assert candidate_row["kernel_sample_count"] == 1
    assert candidate_row["meets_min_promotion_sample_count"] is False
    assert candidate_row["microbench_candidate_for_full_replay"] is False
