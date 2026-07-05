from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from largerlm.prepared import load_prepared_manifest


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepared_manifest_variant.py"
    )
    spec = importlib.util.spec_from_file_location("prepared_manifest_variant", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_layout(path: Path, *, layer_file: str, expert_order: list[int] | None) -> None:
    layout: dict[str, object] = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "num_layers": 1,
        "num_experts": 2,
        "layers": [
            {
                "layer": 0,
                "num_experts": 2,
                "expert_slot_bytes": 16,
                "layer_file": layer_file,
                "components": [
                    {
                        "name": "gate_proj.weight",
                        "offset": 0,
                        "size": 16,
                        "dtype": "U32",
                        "shape": [4, 1],
                    }
                ],
            }
        ],
    }
    if expert_order is not None:
        layer = layout["layers"][0]
        assert isinstance(layer, dict)
        layer["expert_order"] = expert_order
    path.write_text(json.dumps(layout), encoding="utf-8")


def _write_prepared(root: Path) -> Path:
    prepared = root / "prepared"
    model = root / "model"
    experts = prepared / "experts"
    repacked = prepared / "experts-repacked"
    resident = prepared / "resident"
    for directory in (model, experts, repacked, resident):
        directory.mkdir(parents=True)

    (experts / "layer_000.bin").write_bytes(b"a" * 32)
    _write_layout(experts / "layout.json", layer_file="layer_000.bin", expert_order=None)
    (repacked / "layer_000.bin").write_bytes(b"b" * 32)
    _write_layout(
        repacked / "layout.json",
        layer_file="layer_000.bin",
        expert_order=[1, 0],
    )

    (resident / "resident.bin").write_bytes(b"\0" * 16)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "total_bytes": 16,
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 16,
                        "dtype": "F32",
                        "shape": [1, 4],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (prepared / "decode_cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 32,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    (prepared / "decode_cache.bin").write_bytes(b"\0" * 32)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_dir": str(model),
                "experts_layout": "experts/layout.json",
                "resident_layout": "resident/layout.json",
                "decode_cache_layout": "decode_cache_layout.json",
                "decode_cache_file": "decode_cache.bin",
                "max_context_tokens": 4,
            }
        ),
        encoding="utf-8",
    )
    return prepared


def test_prepared_manifest_variant_writes_valid_shadow_manifest(tmp_path: Path) -> None:
    module = _load_script()
    prepared = _write_prepared(tmp_path)
    output = prepared / "manifest-repacked-layer0.json"

    result = module.write_validated_variant_manifest(
        source_manifest=prepared,
        experts_layout=Path("experts-repacked/layout.json"),
        output_manifest=output,
        label="layer0",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experts_layout"] == "experts-repacked/layout.json"
    assert payload["resident_layout"] == "resident/layout.json"
    assert payload["manifest_variant"]["label"] == "layer0"
    assert result["expert_layout_bytes"] == 32
    loaded = load_prepared_manifest(output)
    assert loaded.experts_layout == prepared / "experts-repacked" / "layout.json"
    assert loaded.expert_layout_bytes == 32


def test_prepared_manifest_variant_rejects_output_in_different_base(
    tmp_path: Path,
) -> None:
    module = _load_script()
    prepared = _write_prepared(tmp_path)

    with pytest.raises(SystemExit, match="same directory"):
        module.write_validated_variant_manifest(
            source_manifest=prepared,
            experts_layout=Path("experts-repacked/layout.json"),
            output_manifest=tmp_path / "manifest-repacked.json",
            label="bad",
        )


def test_prepared_manifest_variant_rejects_unvalidated_layout(tmp_path: Path) -> None:
    module = _load_script()
    prepared = _write_prepared(tmp_path)
    bad = prepared / "experts-bad"
    bad.mkdir()
    (bad / "layer_000.bin").write_bytes(b"x")
    _write_layout(bad / "layout.json", layer_file="layer_000.bin", expert_order=[0, 1])
    output = prepared / "manifest-bad.json"

    with pytest.raises(SystemExit, match="failed prepared validation"):
        module.write_validated_variant_manifest(
            source_manifest=prepared,
            experts_layout=Path("experts-bad/layout.json"),
            output_manifest=output,
            label="bad",
        )
    assert not output.exists()
