from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "expert_order_repack_plan.py"
    spec = importlib.util.spec_from_file_location("expert_order_repack_plan", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_plan_selects_repack_candidates(tmp_path: Path) -> None:
    module = _load_module()
    analysis = tmp_path / "analysis.json"
    layout = tmp_path / "layout.json"
    experts = tmp_path / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
    (experts / "layer_001.bin").write_bytes(b"abcd")
    (experts / "layer_002.bin").write_bytes(b"wxyz")
    layout.write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "quantization": "mlx-mxfp4",
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_001.bin",
                    },
                    {
                        "layer": 2,
                        "num_experts": 4,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_002.bin",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    analysis.write_text(
        json.dumps(
            {
                "schema": "largerlm.stage_hotspot_layout_analysis.v2",
                "source_result_count": 2,
                "coactivation_candidates": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_order": [2, 0, 1, 3],
                        "sample_count": 2,
                        "observed_expert_count": 3,
                        "total_current_range_count": 5,
                        "total_candidate_range_count": 2,
                        "total_range_reduction": 3,
                        "total_range_reduction_fraction": 0.6,
                        "total_copy_elapsed_seconds": 0.7,
                        "simulated_rows": [
                            {
                                "current_range_count": 3,
                                "candidate_range_count": 1,
                            }
                        ],
                    },
                    {
                        "layer": 2,
                        "num_experts": 4,
                        "expert_order": [0, 1, 2, 3],
                        "sample_count": 1,
                        "observed_expert_count": 2,
                        "total_current_range_count": 2,
                        "total_candidate_range_count": 2,
                        "total_range_reduction": 0,
                        "total_range_reduction_fraction": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = module.build_plan(
        analysis_path=analysis,
        expert_layout_path=layout,
        max_layers=5,
        min_range_reduction=1,
        min_reduction_fraction=0.0,
    )

    assert plan["schema"] == "largerlm.expert_order_repack_plan.v1"
    assert plan["dry_run"] is True
    assert plan["safe_to_apply_to_existing_layout"] is False
    assert plan["requires_layer_file_repack"] is True
    assert plan["source_analysis_schema"] == "largerlm.stage_hotspot_layout_analysis.v2"
    assert plan["min_sample_count"] == 0
    assert plan["selected_layer_count"] == 1
    assert plan["total_repack_read_bytes"] == 4
    assert plan["total_repack_write_bytes"] == 4
    assert plan["total_repack_io_bytes"] == 8
    selected = plan["selected_layers"][0]
    assert selected["layer"] == 1
    assert selected["current_expert_order"] == [0, 1, 2, 3]
    assert selected["proposed_expert_order"] == [2, 0, 1, 3]
    assert selected["layout_patch"] == {
        "layer": 1,
        "expert_order": [2, 0, 1, 3],
    }
    assert selected["total_range_reduction"] == 3


def test_build_plan_filters_candidates_by_min_sample_count(tmp_path: Path) -> None:
    module = _load_module()
    experts = tmp_path / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
    analysis = tmp_path / "analysis.json"
    (experts / "layer_001.bin").write_bytes(b"abcd")
    (experts / "layer_002.bin").write_bytes(b"wxyz")
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_001.bin",
                    },
                    {
                        "layer": 2,
                        "num_experts": 4,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_002.bin",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    analysis.write_text(
        json.dumps(
            {
                "coactivation_candidates": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_order": [2, 0, 1, 3],
                        "sample_count": 1,
                        "total_range_reduction": 9,
                        "total_range_reduction_fraction": 0.9,
                    },
                    {
                        "layer": 2,
                        "num_experts": 4,
                        "expert_order": [1, 0, 2, 3],
                        "sample_count": 2,
                        "total_range_reduction": 2,
                        "total_range_reduction_fraction": 0.5,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = module.build_plan(
        analysis_path=analysis,
        expert_layout_path=layout,
        max_layers=5,
        min_range_reduction=1,
        min_reduction_fraction=0.0,
        min_sample_count=2,
    )

    assert plan["min_sample_count"] == 2
    assert plan["selected_layer_count"] == 1
    assert plan["selected_layers"][0]["layer"] == 2
    assert plan["selected_layers"][0]["sample_count"] == 2
    assert plan["total_repack_write_bytes"] == 4


def test_build_plan_skips_bad_layer_file_size(tmp_path: Path) -> None:
    module = _load_module()
    experts = tmp_path / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
    analysis = tmp_path / "analysis.json"
    (experts / "layer_001.bin").write_bytes(b"bad")
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_001.bin",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    analysis.write_text(
        json.dumps(
            {
                "coactivation_candidates": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_order": [2, 0, 1, 3],
                        "total_range_reduction": 3,
                        "total_range_reduction_fraction": 0.6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = module.build_plan(
        analysis_path=analysis,
        expert_layout_path=layout,
        max_layers=1,
        min_range_reduction=1,
        min_reduction_fraction=0.0,
    )

    assert plan["selected_layer_count"] == 0
    assert plan["skipped_layers"][0]["reason"] == (
        "layer file size does not match layout"
    )
