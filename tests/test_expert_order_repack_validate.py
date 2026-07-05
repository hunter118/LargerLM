from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "expert_order_repack_validate.py"
    spec = importlib.util.spec_from_file_location("expert_order_repack_validate", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_repack_reports_slot_and_range_delta(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    repacked = tmp_path / "repacked"
    source.mkdir()
    repacked.mkdir()
    (source / "layer_001.bin").write_bytes(b"aabbccdd")
    (repacked / "layer_001.bin").write_bytes(b"ccaabbdd")
    source_layout = source / "layout.json"
    repacked_layout = repacked / "layout.json"
    source_layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repacked_layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                        "expert_order": [2, 0, 1, 3],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": 1,
                        "selected_experts": [0, 1, 2],
                        "source_result": "sample.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = module.validate_repack(
        source_layout_path=source_layout,
        repacked_layout_path=repacked_layout,
        analysis_path=analysis,
        sample_count=3,
        sample_bytes=2,
        align_bytes=1,
    )

    assert result["schema"] == "largerlm.expert_order_repack_validate.v1"
    assert result["slot_validation_ok"] is True
    assert result["slot_validation"][0]["layer"] == 1
    assert result["old_coalesced_range_count"] == 1
    assert result["new_coalesced_range_count"] == 1
    assert result["range_reduction"] == 0


def test_validate_repack_range_delta_can_improve(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source"
    repacked = tmp_path / "repacked"
    source.mkdir()
    repacked.mkdir()
    (source / "layer_001.bin").write_bytes(b"aabbccdd")
    (repacked / "layer_001.bin").write_bytes(b"ddccaabb")
    source_layout = source / "layout.json"
    repacked_layout = repacked / "layout.json"
    source_layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repacked_layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                        "expert_order": [3, 2, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer": 1,
                        "selected_experts": [0, 2, 3],
                        "source_result": "sample.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = module.validate_repack(
        source_layout_path=source_layout,
        repacked_layout_path=repacked_layout,
        analysis_path=analysis,
        sample_count=4,
        sample_bytes=2,
        align_bytes=1,
    )

    assert result["slot_validation_ok"] is True
    assert result["old_coalesced_range_count"] == 2
    assert result["new_coalesced_range_count"] == 1
    assert result["range_reduction"] == 1


def test_validate_repack_uses_source_expert_order_for_incremental_repack(
    tmp_path: Path,
) -> None:
    module = _load_module()
    source = tmp_path / "source"
    repacked = tmp_path / "repacked"
    source.mkdir()
    repacked.mkdir()
    # Logical experts 2,0,1,3 are already physically packed in source slots 0..3.
    (source / "layer_001.bin").write_bytes(b"ccaabbdd")
    # The incremental repack changes the physical order to logical 3,2,0,1.
    (repacked / "layer_001.bin").write_bytes(b"ddccaabb")
    source_layout = source / "layout.json"
    repacked_layout = repacked / "layout.json"
    source_layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                        "expert_order": [2, 0, 1, 3],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repacked_layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                        "expert_order": [3, 2, 0, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = module.validate_repack(
        source_layout_path=source_layout,
        repacked_layout_path=repacked_layout,
        analysis_path=None,
        sample_count=4,
        sample_bytes=2,
        align_bytes=1,
    )

    assert result["slot_validation_ok"] is True
    checks = result["slot_validation"][0]["checks"]
    assert [(item["expert"], item["source_slot"]) for item in checks] == [
        (3, 3),
        (2, 0),
        (0, 1),
        (1, 2),
    ]
