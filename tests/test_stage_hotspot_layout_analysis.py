from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage_hotspot_layout_analysis.py"
    spec = importlib.util.spec_from_file_location("stage_hotspot_layout_analysis", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_result(
    path: Path,
    *,
    layer: int,
    selected_experts: list[int],
    coalesced_ranges: int,
    copy_seconds: float,
) -> None:
    row = {
        "chunk_index": 0,
        "layer": layer,
        "tile_index": 0,
        "selected_experts": selected_experts,
        "selected_expert_count": len(selected_experts),
        "raw_range_count": coalesced_ranges,
        "coalesced_range_count": coalesced_ranges,
        "planned_read_bytes": 1000 * coalesced_ranges,
        "copy_elapsed_seconds": copy_seconds,
        "copy_read_calls": coalesced_ranges,
    }
    path.write_text(
        json.dumps(
            {
                "token_result": {
                    "prefill_actual_read_time": {
                        "expert_stage_io_stage_count": 1,
                        "expert_stage_copy_hotspots": [row],
                        "expert_stage_range_hotspots": [row],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_analyze_results_builds_coactivation_order(tmp_path: Path) -> None:
    module = _load_module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    layout = tmp_path / "layout.json"
    _write_result(
        first,
        layer=1,
        selected_experts=[0, 3, 7],
        coalesced_ranges=3,
        copy_seconds=0.3,
    )
    _write_result(
        second,
        layer=1,
        selected_experts=[3, 4, 7],
        coalesced_ranges=2,
        copy_seconds=0.2,
    )
    layout.write_text(
        json.dumps({"layers": [{"layer": 1, "num_experts": 8}]}),
        encoding="utf-8",
    )

    analysis = module.analyze_results(
        [first, second],
        expert_layout_path=layout,
    )

    assert analysis["schema"] == "largerlm.stage_hotspot_layout_analysis.v2"
    assert analysis["source_result_count"] == 2
    assert analysis["row_count"] == 2
    assert analysis["source_results"][0]["row_count"] == 1
    assert analysis["source_results"][0]["top_layout_targets"][0]["sources"] == [
        {"source": "copy", "rank": 1},
        {"source": "ranges", "rank": 1},
    ]
    candidate = analysis["coactivation_candidates"][0]
    assert candidate["layer"] == 1
    assert candidate["num_experts"] == 8
    assert set(candidate["observed_expert_order"]) == {0, 3, 4, 7}
    assert candidate["expert_order"][:4] == candidate["observed_expert_order"]
    assert candidate["unobserved_expert_count"] == 4
    assert candidate["total_current_range_count"] == 5
    assert candidate["total_candidate_range_count"] == 2
    assert candidate["total_range_reduction"] == 3
    assert candidate["total_range_reduction_fraction"] == 0.6
