from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.glm_router_hybrid_policy_analysis import (
    _iter_router_records_stream,
    _read_consistency_observations,
    analyze_router_hybrid_policy,
    analyze_router_hybrid_policy_records,
)


def _layer(layer: int, *, margin: float, elapsed: float, near_1e_5: int) -> dict:
    return {
        "layer": layer,
        "staged_mlp": {
            "router_gate_proj": {"elapsed_seconds": elapsed},
            "router_margin_summary": {
                "token_count": 4,
                "min_effective_score_margin": margin,
                "min_topk_score_margin": margin,
                "mean_effective_score_margin": margin * 2.0,
                "effective_near_tie_counts": {
                    "le_1e-05": near_1e_5,
                    "le_1e-04": near_1e_5 + 1,
                },
                "topk_near_tie_counts": {
                    "le_1e-05": near_1e_5,
                    "le_1e-04": near_1e_5 + 1,
                },
            },
        },
    }


def _payload() -> dict:
    return {
        "token_result": {
            "prompt_prefill": {
                "chunks": [
                    {
                        "layers": [
                            _layer(3, margin=1.0e-6, elapsed=10.0, near_1e_5=1),
                            _layer(4, margin=1.0e-4, elapsed=20.0, near_1e_5=0),
                        ]
                    }
                ]
            }
        }
    }


def test_hybrid_policy_blocks_global_custom_but_keeps_layer_candidate() -> None:
    result = analyze_router_hybrid_policy(
        _payload(),
        observed_logit_drift=1.0e-5,
        safety_multipliers=(1.0, 4.0),
        absolute_thresholds=(1.0e-5, 1.0e-4),
        promotion_safety_multiplier=4.0,
        custom_to_mpsgraph_router_ratio=0.5,
    )

    assert result["summary"]["routed_layer_count"] == 2
    assert result["summary"]["token_layer_count"] == 8
    assert result["summary"]["min_effective_score_margin"] == pytest.approx(1.0e-6)
    assert result["recommendation"]["global_custom_promotable"] is False
    assert result["recommendation"]["layer_hybrid_candidate"] is True
    assert result["recommendation"]["safe_layer_count_at_promotion_threshold"] == 1
    assert result["recommendation"]["fallback_layer_count_at_promotion_threshold"] == 1

    promotion_row = next(
        row for row in result["thresholds"] if "drift_x_4" in row["sources"]
    )
    assert promotion_row["fallback_layer_count"] == 1
    assert promotion_row["safe_layer_count"] == 1
    assert promotion_row["router_elapsed_seconds"][
        "static_layer_policy_estimate"
    ] == pytest.approx(20.0)
    assert promotion_row["router_elapsed_seconds"][
        "online_custom_first_layer_fallback_estimate"
    ] == pytest.approx(25.0)


def test_threshold_near_tie_counts_use_compact_summary_keys() -> None:
    result = analyze_router_hybrid_policy(
        _payload(),
        observed_logit_drift=1.0e-5,
        safety_multipliers=(1.0,),
        absolute_thresholds=(1.0e-5, 1.0e-3),
        promotion_safety_multiplier=1.0,
        custom_to_mpsgraph_router_ratio=0.5,
    )

    row = next(row for row in result["thresholds"] if row["threshold_key"] == "le_1e-05")
    assert row["effective_near_tie_token_layer_count"] == 1
    assert row["topk_near_tie_token_layer_count"] == 1

    missing_summary_row = next(
        row for row in result["thresholds"] if row["threshold_key"] == "le_1e-03"
    )
    assert missing_summary_row["effective_near_tie_token_layer_count"] is None
    assert missing_summary_row["topk_near_tie_token_layer_count"] is None


def test_consistency_observations_read_drift_and_backend_ratio(tmp_path: Path) -> None:
    path = tmp_path / "consistency.json"
    path.write_text(
        json.dumps(
            {
                "comparisons": {
                    "custom_resident_vs_mpsgraph_resident": {
                        "logits_max_abs": 5.5e-6
                    }
                },
                "custom_resident_linear": {"elapsed_seconds": 2.0},
                "mpsgraph_resident_linear": {"elapsed_seconds": 5.0},
            }
        ),
        encoding="utf-8",
    )

    assert _read_consistency_observations(path) == {
        "observed_logit_drift": 5.5e-6,
        "custom_to_mpsgraph_router_ratio": 0.4,
    }


def test_stream_router_records_match_payload_analysis(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_payload(), indent=2), encoding="utf-8")

    records = _iter_router_records_stream(path)
    result = analyze_router_hybrid_policy_records(
        records,
        observed_logit_drift=1.0e-5,
        safety_multipliers=(4.0,),
        absolute_thresholds=(1.0e-5,),
        promotion_safety_multiplier=4.0,
        custom_to_mpsgraph_router_ratio=0.5,
    )

    assert [(record.chunk_index, record.layer) for record in records] == [(0, 3), (0, 4)]
    assert result["summary"]["routed_layer_count"] == 2
    assert result["summary"]["router_elapsed_seconds_total"] == pytest.approx(30.0)
