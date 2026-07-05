from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "expert_order_repack_execute.py"
    spec = importlib.util.spec_from_file_location("expert_order_repack_execute", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    experts = tmp_path / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
    (experts / "layer_001.bin").write_bytes(b"aabbccdd")
    (experts / "layer_002.bin").write_bytes(b"11223344")
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "quantization": "mlx-mxfp4",
                "num_layers": 2,
                "num_experts": 4,
                "component_order": ["gate_proj.weight"],
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                        "components": [],
                    },
                    {
                        "layer": 2,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_002.bin",
                        "components": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "largerlm.expert_order_repack_plan.v1",
                "expert_layout": str(layout),
                "selected_layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 2,
                        "layer_file": "layer_001.bin",
                        "expected_layer_file_bytes": 8,
                        "current_expert_order": [0, 1, 2, 3],
                        "proposed_expert_order": [2, 0, 1, 3],
                        "total_current_range_count": 3,
                        "total_candidate_range_count": 1,
                        "total_range_reduction": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return layout, plan


def test_repack_manifest_dry_run_does_not_create_output(tmp_path: Path) -> None:
    module = _load_module()
    _layout, plan = _write_fixture(tmp_path)
    output = tmp_path / "out"

    manifest = module.build_repack_manifest(
        plan_path=plan,
        output_dir=output,
        execute=False,
        copy_chunk_bytes=3,
        min_free_bytes_after_write=0,
    )

    assert manifest["executed"] is False
    assert manifest["selected_layer_count"] == 1
    assert manifest["repack_read_bytes"] == 8
    assert manifest["repack_write_bytes"] == 8
    assert manifest["hardlink_referenced_bytes"] == 8
    assert manifest["write_guard"]["ok"] is True
    assert manifest["write_guard"]["repack_write_bytes"] == 8
    assert not output.exists()


def test_repack_execute_reorders_selected_layer_and_hardlinks_others(
    tmp_path: Path,
) -> None:
    module = _load_module()
    layout, plan = _write_fixture(tmp_path)
    output = tmp_path / "out"

    manifest = module.build_repack_manifest(
        plan_path=plan,
        output_dir=output,
        execute=True,
        copy_chunk_bytes=3,
        min_free_bytes_after_write=0,
    )

    assert manifest["executed"] is True
    assert (output / "layer_001.bin").read_bytes() == b"ccaabbdd"
    assert (output / "layer_002.bin").read_bytes() == b"11223344"
    patched = json.loads((output / "layout.json").read_text(encoding="utf-8"))
    assert patched["layers"][0]["expert_order"] == [2, 0, 1, 3]
    assert "expert_order" not in patched["layers"][1]
    layer_ops = {item["layer"]: item for item in manifest["layer_operations"]}
    assert layer_ops[1]["operation"] == "repack"
    assert layer_ops[1]["bytes"] == 8
    assert layer_ops[1]["read_calls"] == 4
    assert layer_ops[2]["operation"] == "hardlink"
    assert os.stat(layout.parent / "layer_002.bin").st_ino == os.stat(
        output / "layer_002.bin"
    ).st_ino


def test_repack_execute_rejects_nonempty_output_dir(tmp_path: Path) -> None:
    module = _load_module()
    _layout, plan = _write_fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing").write_text("x", encoding="utf-8")

    try:
        module.build_repack_manifest(
            plan_path=plan,
            output_dir=output,
            execute=True,
            copy_chunk_bytes=3,
            min_free_bytes_after_write=0,
        )
    except SystemExit as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("expected non-empty output dir rejection")


def test_repack_execute_rejects_write_cap_before_output(tmp_path: Path) -> None:
    module = _load_module()
    _layout, plan = _write_fixture(tmp_path)
    output = tmp_path / "out"

    try:
        module.build_repack_manifest(
            plan_path=plan,
            output_dir=output,
            execute=True,
            copy_chunk_bytes=3,
            max_repack_write_bytes=7,
            min_free_bytes_after_write=0,
        )
    except SystemExit as exc:
        assert "repack write guard failed" in str(exc)
    else:
        raise AssertionError("expected write cap rejection")
    assert not output.exists()
