from __future__ import annotations

import json
from pathlib import Path

import pytest

import largerlm.layout as layout_module
from largerlm.layout import (
    ComponentLayout,
    LayerLayout,
    PackedExpertsLayout,
    ResidentTensorLayout,
    ResidentWeightsLayout,
)


def test_packed_experts_layout_write_is_atomic_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "layout.json"
    path.write_text('{"old": true}', encoding="utf-8")
    layout = PackedExpertsLayout(
        version=1,
        model_type="glm_moe_dsa",
        config_sha256=None,
        quantization="largerlm-affine-int4",
        group_size=64,
        num_layers=1,
        num_experts=1,
        component_order=("gate_proj.weight",),
        layers=(
            LayerLayout(
                layer=0,
                num_experts=1,
                expert_slot_bytes=4,
                layer_file="layer_000.bin",
                components=(
                    ComponentLayout(
                        name="gate_proj.weight",
                        offset=0,
                        size=4,
                        dtype="U32",
                        shape=(1, 1),
                    ),
                ),
            ),
        ),
    )

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(layout_module.Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        layout.write(path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not path.with_name(path.name + ".tmp").exists()


def test_resident_weights_layout_write_is_atomic_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "layout.json"
    path.write_text('{"old": true}', encoding="utf-8")
    layout = ResidentWeightsLayout(
        version=1,
        model_type="glm_moe_dsa",
        config_sha256=None,
        alignment=64,
        weight_file="resident.bin",
        total_bytes=4,
        tensors=(
            ResidentTensorLayout(
                name="model.embed_tokens.weight",
                offset=0,
                size=4,
                dtype="F32",
                shape=(1,),
                category="embeddings",
            ),
        ),
        router=None,
    )

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(layout_module.Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        layout.write(path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not path.with_name(path.name + ".tmp").exists()
