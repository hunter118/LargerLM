from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import largerlm.layout as layout_module
import largerlm.resident as resident_module
from largerlm.resident import build_resident_layout, pack_resident_weights


def _write_checkpoint(root: Path) -> dict[str, bytes]:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "scoring_func": " sigmoid ",
                "norm_topk_prob": True,
                "routed_scaling_factor": 2.5,
                "n_group": 1,
                "topk_group": 1,
                "topk_method": " noaux_tc ",
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        "model.embed_tokens.weight": b"abcde",
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": b"dense123",
        "model.layers.1.mlp.gate.weight": b"xyz",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight": b"01234567",
        "model.layers.2.self_attn.indexer.wk.weight": b"mtp-indexer",
        "model.layers.2.mlp.experts.0.gate_proj.weight": b"mtp-routed",
    }
    payload = bytearray()
    header = {}
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [start, len(payload)],
        }
    shard = root / "model-00001-of-00001.safetensors"
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    return tensors


def _write_resident_safetensors(
    root: Path,
    *,
    config: dict[str, object],
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    payload = bytearray()
    header = {}
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    shard = root / "model-00001-of-00001.safetensors"
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )


def _resident_split_config() -> dict[str, object]:
    return {
        "model_type": "glm_moe_dsa",
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
    }


def test_resident_layout_excludes_routed_experts(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    layout, _tensors = build_resident_layout(tmp_path, alignment=64)

    assert [tensor.name for tensor in layout.tensors] == [
        "model.embed_tokens.weight",
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
    ]
    assert [tensor.offset for tensor in layout.tensors] == [0, 64, 128]
    assert [tensor.category for tensor in layout.tensors] == [
        "embeddings",
        "dense_mlp",
        "routers",
    ]
    assert layout.total_bytes == 192
    assert layout.router == {
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "num_experts_per_tok": 1,
    }


def test_resident_layout_splits_fused_dense_gate_up_tensor(tmp_path: Path) -> None:
    gate_bytes = b"G" * (16 * 8 * 4)
    up_bytes = b"U" * (16 * 8 * 4)
    down_bytes = b"D" * (8 * 16 * 4)
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.mlp.switch_mlp.gate_up_proj.weight": (
                "F32",
                [32, 8],
                gate_bytes + up_bytes,
            ),
            "model.layers.0.mlp.switch_mlp.down_proj.weight": (
                "F32",
                [8, 16],
                down_bytes,
            ),
            "model.layers.1.mlp.gate.weight": ("F32", [2, 8], b"R" * (2 * 8 * 4)),
            "model.layers.1.mlp.experts.0.gate_proj.weight": (
                "BF16",
                [16, 8],
                b"E" * (16 * 8 * 2),
            ),
        },
    )

    layout, tensors = build_resident_layout(tmp_path, alignment=16)
    tensor_by_name = {tensor.name: tensor for tensor in tensors}

    assert "model.layers.0.mlp.switch_mlp.gate_up_proj.weight" not in tensor_by_name
    assert tensor_by_name[
        "model.layers.0.mlp.switch_mlp.gate_proj.weight"
    ].shape == (16, 8)
    assert tensor_by_name[
        "model.layers.0.mlp.switch_mlp.up_proj.weight"
    ].shape == (16, 8)
    assert {
        tensor.name: tensor.category
        for tensor in layout.tensors
        if "switch_mlp" in tensor.name
    } == {
        "model.layers.0.mlp.switch_mlp.down_proj.weight": "dense_mlp",
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": "dense_mlp",
        "model.layers.0.mlp.switch_mlp.up_proj.weight": "dense_mlp",
    }

    output = tmp_path / "resident"
    report = pack_resident_weights(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=31,
        alignment=16,
        disk_safety_margin_bytes=0,
    )
    assert report.fused_gate_up_source_tensor_count == 1
    assert report.fused_gate_up_expanded_tensor_count == 2
    assert report.fused_gate_up_expanded_bytes == len(gate_bytes) + len(up_bytes)
    data = (output / "resident.bin").read_bytes()
    layout_by_name = {tensor.name: tensor for tensor in report.layout.tensors}
    gate = layout_by_name["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    up = layout_by_name["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    assert data[gate.offset : gate.offset + gate.size] == gate_bytes
    assert data[up.offset : up.offset + up.size] == up_bytes
    assert b"G" * 8 in data
    assert b"U" * 8 in data
    assert b"E" * 8 not in data


def test_resident_layout_accepts_affine_int4_resident_linear(
    tmp_path: Path,
) -> None:
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.self_attn.q_a_proj.weight": (
                "U32",
                [2, 1],
                b"W" * (2 * 4),
            ),
            "model.layers.0.self_attn.q_a_proj.scales": (
                "BF16",
                [2, 1],
                b"S" * (2 * 2),
            ),
            "model.layers.0.self_attn.q_a_proj.biases": (
                "BF16",
                [2, 1],
                b"B" * (2 * 2),
            ),
        },
    )

    layout, tensors = build_resident_layout(tmp_path, alignment=16)

    assert {tensor.name for tensor in tensors} == {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_proj.scales",
        "model.layers.0.self_attn.q_a_proj.biases",
    }
    by_name = {tensor.name: tensor for tensor in layout.tensors}
    assert by_name["model.layers.0.self_attn.q_a_proj.weight"].dtype == "U32"
    assert by_name["model.layers.0.self_attn.q_a_proj.weight"].shape == (2, 1)
    assert by_name["model.layers.0.self_attn.q_a_proj.scales"].shape == (2, 1)
    assert by_name["model.layers.0.self_attn.q_a_proj.biases"].shape == (2, 1)


def test_resident_layout_accepts_mxfp4_resident_linear_metadata(
    tmp_path: Path,
) -> None:
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.self_attn.q_a_proj.weight": (
                "U32",
                [2, 1],
                b"W" * (2 * 4),
            ),
            "model.layers.0.self_attn.q_a_proj.scales": (
                "U8",
                [2, 1],
                b"S" * 2,
            ),
        },
    )

    layout, tensors = build_resident_layout(tmp_path, alignment=16)

    assert {tensor.name for tensor in tensors} == {
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_proj.scales",
    }
    by_name = {tensor.name: tensor for tensor in layout.tensors}
    assert by_name["model.layers.0.self_attn.q_a_proj.weight"].dtype == "U32"
    assert by_name["model.layers.0.self_attn.q_a_proj.weight"].shape == (2, 1)
    assert by_name["model.layers.0.self_attn.q_a_proj.scales"].dtype == "U8"
    assert by_name["model.layers.0.self_attn.q_a_proj.scales"].shape == (2, 1)


def test_resident_layout_accepts_mxfp4_3d_attention_metadata(
    tmp_path: Path,
) -> None:
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.self_attn.embed_q.weight": (
                "U32",
                [2, 4, 1],
                b"W" * (2 * 4 * 4),
            ),
            "model.layers.0.self_attn.embed_q.scales": (
                "U8",
                [2, 4, 1],
                b"S" * (2 * 4),
            ),
        },
    )

    layout, tensors = build_resident_layout(tmp_path, alignment=16)

    assert {tensor.name for tensor in tensors} == {
        "model.layers.0.self_attn.embed_q.weight",
        "model.layers.0.self_attn.embed_q.scales",
    }
    by_name = {tensor.name: tensor for tensor in layout.tensors}
    assert by_name["model.layers.0.self_attn.embed_q.weight"].shape == (2, 4, 1)
    assert by_name["model.layers.0.self_attn.embed_q.scales"].shape == (2, 4, 1)


def test_resident_layout_splits_fused_shared_gate_up_tensor(tmp_path: Path) -> None:
    gate_bytes = b"S" * (16 * 8 * 4)
    up_bytes = b"T" * (16 * 8 * 4)
    down_bytes = b"V" * (8 * 16 * 4)
    config = _resident_split_config() | {"n_shared_experts": 1}
    _write_resident_safetensors(
        tmp_path,
        config=config,
        tensors={
            "model.layers.1.mlp.shared_experts.gate_up.weight": (
                "F32",
                [32, 8],
                gate_bytes + up_bytes,
            ),
            "model.layers.1.mlp.shared_experts.down_proj.weight": (
                "F32",
                [8, 16],
                down_bytes,
            ),
            "model.layers.1.mlp.gate.weight": ("F32", [2, 8], b"R" * (2 * 8 * 4)),
        },
    )

    output = tmp_path / "resident"
    report = pack_resident_weights(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=29,
        alignment=16,
        disk_safety_margin_bytes=0,
    )

    assert report.fused_gate_up_source_tensor_count == 1
    assert report.fused_gate_up_expanded_tensor_count == 2
    layout_by_name = {tensor.name: tensor for tensor in report.layout.tensors}
    gate = layout_by_name["model.layers.1.mlp.shared_experts.gate_proj.weight"]
    up = layout_by_name["model.layers.1.mlp.shared_experts.up_proj.weight"]
    assert gate.category == "shared_experts"
    assert up.category == "shared_experts"
    data = (output / "resident.bin").read_bytes()
    assert data[gate.offset : gate.offset + gate.size] == gate_bytes
    assert data[up.offset : up.offset + up.size] == up_bytes


def test_resident_layout_normalizes_dense_w_component_aliases(
    tmp_path: Path,
) -> None:
    gate_bytes = b"G" * (16 * 8 * 4)
    up_bytes = b"U" * (16 * 8 * 4)
    down_bytes = b"D" * (8 * 16 * 4)
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.mlp.switch_mlp.w1.weight": (
                "F32",
                [16, 8],
                gate_bytes,
            ),
            "model.layers.0.mlp.switch_mlp.w3.weight": (
                "F32",
                [16, 8],
                up_bytes,
            ),
            "model.layers.0.mlp.switch_mlp.w2.weight": (
                "F32",
                [8, 16],
                down_bytes,
            ),
            "model.layers.1.mlp.gate.weight": ("F32", [2, 8], b"R" * (2 * 8 * 4)),
        },
    )

    layout, tensors = build_resident_layout(tmp_path, alignment=16)
    tensor_by_name = {tensor.name: tensor for tensor in tensors}

    assert "model.layers.0.mlp.switch_mlp.w1.weight" not in tensor_by_name
    assert "model.layers.0.mlp.switch_mlp.w2.weight" not in tensor_by_name
    assert "model.layers.0.mlp.switch_mlp.w3.weight" not in tensor_by_name
    assert {
        tensor.name: tensor.category
        for tensor in layout.tensors
        if "switch_mlp" in tensor.name
    } == {
        "model.layers.0.mlp.switch_mlp.down_proj.weight": "dense_mlp",
        "model.layers.0.mlp.switch_mlp.gate_proj.weight": "dense_mlp",
        "model.layers.0.mlp.switch_mlp.up_proj.weight": "dense_mlp",
    }

    output = tmp_path / "resident"
    report = pack_resident_weights(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=31,
        alignment=16,
        disk_safety_margin_bytes=0,
    )
    assert report.component_alias_source_tensor_count == 3
    assert report.component_alias_renamed_tensor_count == 3
    assert report.component_alias_bytes == (
        len(gate_bytes) + len(up_bytes) + len(down_bytes)
    )
    data = (output / "resident.bin").read_bytes()
    layout_by_name = {tensor.name: tensor for tensor in report.layout.tensors}
    gate = layout_by_name["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    up = layout_by_name["model.layers.0.mlp.switch_mlp.up_proj.weight"]
    down = layout_by_name["model.layers.0.mlp.switch_mlp.down_proj.weight"]
    assert data[gate.offset : gate.offset + gate.size] == gate_bytes
    assert data[up.offset : up.offset + up.size] == up_bytes
    assert data[down.offset : down.offset + down.size] == down_bytes


def test_resident_layout_normalizes_shared_w_component_aliases(
    tmp_path: Path,
) -> None:
    gate_bytes = b"S" * (16 * 8 * 4)
    up_bytes = b"T" * (16 * 8 * 4)
    down_bytes = b"V" * (8 * 16 * 4)
    config = _resident_split_config() | {"n_shared_experts": 1}
    _write_resident_safetensors(
        tmp_path,
        config=config,
        tensors={
            "model.layers.1.mlp.shared_experts.w1.weight": (
                "F32",
                [16, 8],
                gate_bytes,
            ),
            "model.layers.1.mlp.shared_experts.w3.weight": (
                "F32",
                [16, 8],
                up_bytes,
            ),
            "model.layers.1.mlp.shared_experts.w2.weight": (
                "F32",
                [8, 16],
                down_bytes,
            ),
            "model.layers.1.mlp.gate.weight": ("F32", [2, 8], b"R" * (2 * 8 * 4)),
        },
    )

    output = tmp_path / "resident"
    report = pack_resident_weights(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=29,
        alignment=16,
        disk_safety_margin_bytes=0,
    )

    assert report.component_alias_source_tensor_count == 3
    assert report.component_alias_renamed_tensor_count == 3
    layout_by_name = {tensor.name: tensor for tensor in report.layout.tensors}
    gate = layout_by_name["model.layers.1.mlp.shared_experts.gate_proj.weight"]
    up = layout_by_name["model.layers.1.mlp.shared_experts.up_proj.weight"]
    down = layout_by_name["model.layers.1.mlp.shared_experts.down_proj.weight"]
    assert gate.category == "shared_experts"
    assert up.category == "shared_experts"
    assert down.category == "shared_experts"
    data = (output / "resident.bin").read_bytes()
    assert data[gate.offset : gate.offset + gate.size] == gate_bytes
    assert data[up.offset : up.offset + up.size] == up_bytes
    assert data[down.offset : down.offset + down.size] == down_bytes


def test_resident_layout_rejects_duplicate_fused_gate_up_alias(
    tmp_path: Path,
) -> None:
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.mlp.switch_mlp.gate_up_proj.weight": (
                "F32",
                [32, 8],
                b"F" * (32 * 8 * 4),
            ),
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": (
                "F32",
                [16, 8],
                b"G" * (16 * 8 * 4),
            ),
        },
    )

    with pytest.raises(
        resident_module.ResidentPackerError,
        match="duplicate resident tensor",
    ):
        build_resident_layout(tmp_path)


def test_resident_layout_rejects_duplicate_w_component_alias(
    tmp_path: Path,
) -> None:
    _write_resident_safetensors(
        tmp_path,
        config=_resident_split_config(),
        tensors={
            "model.layers.0.mlp.switch_mlp.w1.weight": (
                "F32",
                [16, 8],
                b"W" * (16 * 8 * 4),
            ),
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": (
                "F32",
                [16, 8],
                b"G" * (16 * 8 * 4),
            ),
        },
    )

    with pytest.raises(
        resident_module.ResidentPackerError,
        match="duplicate resident tensor",
    ):
        build_resident_layout(tmp_path)


def test_pack_resident_execute_writes_aligned_blob(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    output = tmp_path / "resident"

    report = pack_resident_weights(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=2,
        disk_safety_margin_bytes=0,
    )

    data = (output / "resident.bin").read_bytes()
    assert report.disk_checked is True
    assert (output / "layout.json").exists()
    assert len(data) == 192
    assert data[:5] == b"abcde"
    assert data[64:72] == b"dense123"
    assert data[128:131] == b"xyz"
    assert b"01234567" not in data
    assert b"mtp-indexer" not in data
    assert b"mtp-routed" not in data


def test_pack_resident_removes_partial_weight_file_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checkpoint(tmp_path)
    output = tmp_path / "resident"

    def fail_copy(**kwargs) -> None:
        os.pwrite(kwargs["dst_fd"], b"partial", kwargs["dst_offset"])
        raise resident_module.ResidentPackerError("copy exploded")

    monkeypatch.setattr(resident_module, "_copy_slice", fail_copy)

    with pytest.raises(resident_module.ResidentPackerError, match="copy exploded"):
        pack_resident_weights(
            tmp_path,
            output,
            dry_run=False,
            chunk_size=2,
            disk_safety_margin_bytes=0,
        )

    assert not (output / "resident.bin").exists()
    assert not (output / "layout.json").exists()


def test_pack_resident_preserves_existing_layout_when_atomic_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checkpoint(tmp_path)
    output = tmp_path / "resident"
    output.mkdir()
    layout_path = output / "layout.json"
    layout_path.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(layout_module.Path, "replace", fail_replace)

    with pytest.raises(resident_module.ResidentPackerError, match="replace failed"):
        pack_resident_weights(
            tmp_path,
            output,
            dry_run=False,
            force=True,
            chunk_size=2,
            disk_safety_margin_bytes=0,
        )

    assert json.loads(layout_path.read_text(encoding="utf-8")) == {"old": True}
    assert not layout_path.with_name(layout_path.name + ".tmp").exists()
    assert not (output / "resident.bin").exists()
