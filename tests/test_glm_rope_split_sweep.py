from __future__ import annotations

from scripts.glm_rope_split_sweep import _config_comparison


def test_config_comparison_keeps_fused_baseline_when_fastest() -> None:
    comparison = _config_comparison(
        {
            "old": {
                "elapsed_seconds": {"count": 4, "mean": 0.11},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 1e-6},
            },
            "fused": {
                "elapsed_seconds": {"count": 4, "mean": 0.10},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 0.0},
            },
        }
    )

    assert comparison["baseline_mode"] == "fused"
    assert comparison["fastest_wall_mode"] == "fused"
    assert comparison["candidate_mode"] is None
    assert comparison["candidate_for_full_replay"] is False
    assert comparison["requires_full_replay_bakeoff"] is False
    assert "baseline_has_fastest_wall_mean" in comparison["reasons"]


def test_config_comparison_can_promote_old_mode() -> None:
    comparison = _config_comparison(
        {
            "old": {
                "elapsed_seconds": {"count": 4, "mean": 0.09},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 1e-6},
            },
            "fused": {
                "elapsed_seconds": {"count": 4, "mean": 0.10},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 0.0},
            },
        }
    )

    assert comparison["candidate_mode"] == "old"
    assert comparison["candidate_for_full_replay"] is True
    assert comparison["requires_full_replay_bakeoff"] is True
    assert "candidate_met_wall_speedup_and_drift_policy" in comparison["reasons"]


def test_config_comparison_rejects_drift() -> None:
    comparison = _config_comparison(
        {
            "old": {
                "elapsed_seconds": {"count": 4, "mean": 0.09},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 1e-2},
            },
            "fused": {
                "elapsed_seconds": {"count": 4, "mean": 0.10},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 0.0},
            },
        },
        max_promotion_drift=1e-5,
    )

    assert comparison["candidate_for_full_replay"] is False
    old = next(row for row in comparison["rows"] if row["mode"] == "old")
    assert old["numerically_within_promotion_drift"] is False


def test_config_comparison_requires_samples() -> None:
    comparison = _config_comparison(
        {
            "old": {
                "elapsed_seconds": {"count": 2, "mean": 0.09},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 1e-6},
            },
            "fused": {
                "elapsed_seconds": {"count": 2, "mean": 0.10},
                "max_abs_diff_vs_first": {"q_nope": 0.0, "q_rope": 0.0},
            },
        },
        min_promotion_sample_count=3,
    )

    assert comparison["candidate_for_full_replay"] is False
    old = next(row for row in comparison["rows"] if row["mode"] == "old")
    assert old["meets_min_promotion_sample_count"] is False
