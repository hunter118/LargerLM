from __future__ import annotations

import argparse

import pytest

from scripts.glm_attention_projection_fusion_sweep import (
    _config_comparison,
    _parse_modes,
)


def test_parse_modes_accepts_aliases() -> None:
    assert _parse_modes("fused,separate,on,off") == (
        "fused",
        "separate",
        "fused",
        "separate",
    )


def test_parse_modes_rejects_unknown() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_modes("fused,maybe")


def test_config_comparison_keeps_fused_baseline_when_fastest() -> None:
    comparison = _config_comparison(
        {
            "fused": {
                "elapsed_wall_seconds": {"count": 4, "mean": 0.10},
                "max_abs_diff_vs_first": 0.0,
            },
            "separate": {
                "elapsed_wall_seconds": {"count": 4, "mean": 0.14},
                "max_abs_diff_vs_first": 1e-6,
            },
        }
    )

    assert comparison["baseline_mode"] == "fused"
    assert comparison["fastest_wall_mode"] == "fused"
    assert comparison["candidate_mode"] is None
    assert comparison["candidate_for_full_replay"] is False
    assert comparison["requires_full_replay_bakeoff"] is False
    assert "baseline_has_fastest_wall_mean" in comparison["reasons"]


def test_config_comparison_can_promote_separate_mode() -> None:
    comparison = _config_comparison(
        {
            "fused": {
                "elapsed_wall_seconds": {"count": 4, "mean": 0.10},
                "max_abs_diff_vs_first": 0.0,
            },
            "separate": {
                "elapsed_wall_seconds": {"count": 4, "mean": 0.09},
                "max_abs_diff_vs_first": 1e-6,
            },
        }
    )

    assert comparison["candidate_mode"] == "separate"
    assert comparison["candidate_for_full_replay"] is True
    assert comparison["requires_full_replay_bakeoff"] is True
    assert "candidate_met_wall_speedup_and_drift_policy" in comparison["reasons"]


def test_config_comparison_rejects_drift() -> None:
    comparison = _config_comparison(
        {
            "fused": {
                "elapsed_wall_seconds": {"count": 4, "mean": 0.10},
                "max_abs_diff_vs_first": 0.0,
            },
            "separate": {
                "elapsed_wall_seconds": {"count": 4, "mean": 0.09},
                "max_abs_diff_vs_first": 1e-2,
            },
        },
        max_promotion_drift=1e-5,
    )

    assert comparison["candidate_for_full_replay"] is False
    separate = next(row for row in comparison["rows"] if row["mode"] == "separate")
    assert separate["numerically_within_promotion_drift"] is False


def test_config_comparison_requires_samples() -> None:
    comparison = _config_comparison(
        {
            "fused": {
                "elapsed_wall_seconds": {"count": 2, "mean": 0.10},
                "max_abs_diff_vs_first": 0.0,
            },
            "separate": {
                "elapsed_wall_seconds": {"count": 2, "mean": 0.09},
                "max_abs_diff_vs_first": 1e-6,
            },
        },
        min_promotion_sample_count=3,
    )

    assert comparison["candidate_for_full_replay"] is False
    separate = next(row for row in comparison["rows"] if row["mode"] == "separate")
    assert separate["meets_min_promotion_sample_count"] is False
