from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import largerlm.layout as layout_module
import largerlm.packer as packer_module
from largerlm.layout import DEFAULT_EXPERT_COMPONENTS
from largerlm.layout import MXFP4_EXPERT_COMPONENTS
from largerlm.packer import PackerError, build_packed_layout, pack_experts
from largerlm.safety import SafetyError
from largerlm.safetensors import SafetensorsError


def _write_config(root: Path) -> None:
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
            }
        ),
        encoding="utf-8",
    )


def _write_mxfp4_config(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 32,
                "intermediate_size": 32,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_safetensors(
    root: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    shard = root / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)

    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    name: shard.name
                    for name in tensors
                },
            }
        ),
        encoding="utf-8",
    )


def _f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def _expected_affine_word(values: list[float]) -> int:
    lo = min(values)
    hi = max(values)
    scale = (hi - lo) / 15.0
    word = 0
    for lane, value in enumerate(values):
        q = 0 if scale == 0 else int(round((value - lo) / scale))
        q = max(0, min(15, q))
        word |= (q & 0xF) << (lane * 4)
    return word


def _affine_component_meta(component: str) -> tuple[str, list[int], int]:
    base, kind = component.rsplit(".", 1)
    out_dim = 8 if base == "down_proj" else 16
    in_dim = 16 if base == "down_proj" else 8
    if kind == "weight":
        return "U32", [out_dim, in_dim // 8], out_dim * (in_dim // 8) * 4
    return "BF16", [out_dim, in_dim // 8], out_dim * (in_dim // 8) * 2


def _make_checkpoint(root: Path) -> dict[str, bytes]:
    _write_config(root)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    raw_tensors: dict[str, bytes] = {}
    for i, component in enumerate(DEFAULT_EXPERT_COMPONENTS):
        dtype, shape, size = _affine_component_meta(component)
        # Fused tensor layout: expert 0 slice followed by expert 1 slice.
        e0 = bytes([i + 1]) * size
        e1 = bytes([101 + i]) * size
        dense = bytes([201 + i]) * (2 * size)
        tensors[f"model.layers.0.mlp.switch_mlp.{component}"] = (
            dtype,
            [2] + shape,
            dense,
        )
        tensors[f"model.layers.1.mlp.switch_mlp.{component}"] = (
            dtype,
            [2] + shape,
            e0 + e1,
        )
        raw_tensors[f"model.layers.0.mlp.switch_mlp.{component}"] = dense
        raw_tensors[f"model.layers.1.mlp.switch_mlp.{component}"] = e0 + e1
    _write_safetensors(root, tensors)
    return raw_tensors


def _make_raw_bf16_checkpoint(root: Path) -> None:
    _write_config(root)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    shapes = {
        "gate_proj": [2, 16, 8],
        "up_proj": [2, 16, 8],
        "down_proj": [2, 8, 16],
    }
    for index, (component, shape) in enumerate(shapes.items(), start=1):
        element_count = shape[0] * shape[1] * shape[2]
        tensors[f"model.layers.1.mlp.experts.{component}.weight"] = (
            "BF16",
            shape,
            _f32_to_bf16(float(index)) * element_count,
        )
    _write_safetensors(root, tensors)


def _make_unsupported_quantized_expert_checkpoint(root: Path) -> None:
    _write_config(root)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert in (0, 1):
        stem = f"model.layers.1.mlp.experts.{expert}.gate_proj"
        tensors[f"{stem}.qweight"] = ("U32", [16, 1], b"\0" * (16 * 4))
        tensors[f"{stem}.qzeros"] = ("U32", [16, 1], b"\0" * (16 * 4))
        tensors[f"{stem}.g_idx"] = ("I32", [1], b"\0" * 4)
    _write_safetensors(root, tensors)


def test_build_packed_layout_from_fused_switch_mlp(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path)
    layout, _sources = build_packed_layout(tmp_path, group_size=8)

    assert layout.total_bytes == 2 * 384
    assert len(layout.layers) == 1
    layer = layout.layers[0]
    assert layer.layer == 1
    assert layer.layer_file == "layer_001.bin"
    assert layer.expert_slot_bytes == 384
    assert [component.name for component in layer.components] == list(
        DEFAULT_EXPERT_COMPONENTS
    )


def test_build_packed_layout_reports_unsupported_quantized_expert_names(
    tmp_path: Path,
) -> None:
    _make_unsupported_quantized_expert_checkpoint(tmp_path)

    with pytest.raises(
        PackerError,
        match="GPTQ/AWQ/bitsandbytes-style.*MLX affine-int4.*qweight",
    ):
        build_packed_layout(tmp_path, group_size=8)


def test_build_packed_layout_accepts_mxfp4_scales_only_experts(
    tmp_path: Path,
) -> None:
    _write_mxfp4_config(tmp_path)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for component in ("gate_proj", "up_proj", "down_proj"):
        out_dim = 32
        tensors[f"model.layers.1.mlp.switch_mlp.{component}.weight"] = (
            "U32",
            [2, out_dim, 4],
            b"\0" * (2 * out_dim * 4 * 4),
        )
        tensors[f"model.layers.1.mlp.switch_mlp.{component}.scales"] = (
            "U8",
            [2, out_dim, 1],
            b"\0" * (2 * out_dim),
        )
    _write_safetensors(tmp_path, tensors)

    layout, _sources = build_packed_layout(tmp_path, group_size=8)

    assert layout.quantization == "mlx-mxfp4"
    assert layout.group_size == 32
    assert layout.component_order == MXFP4_EXPERT_COMPONENTS
    assert len(layout.layers) == 1
    assert layout.layers[0].expert_slot_bytes == 1632
    assert [component.name for component in layout.layers[0].components] == list(
        MXFP4_EXPERT_COMPONENTS
    )


def test_pack_experts_mxfp4_execute_streams_expected_bytes(tmp_path: Path) -> None:
    _write_mxfp4_config(tmp_path)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for index, component in enumerate(("gate_proj", "up_proj", "down_proj")):
        out_dim = 32
        weight_size = out_dim * 4 * 4
        scale_size = out_dim
        tensors[f"model.layers.1.mlp.switch_mlp.{component}.weight"] = (
            "U32",
            [2, out_dim, 4],
            bytes([10 + index]) * (2 * weight_size),
        )
        tensors[f"model.layers.1.mlp.switch_mlp.{component}.scales"] = (
            "U8",
            [2, out_dim, 1],
            bytes([40 + index]) * (2 * scale_size),
        )
    _write_safetensors(tmp_path, tensors)

    report = pack_experts(
        tmp_path,
        tmp_path / "packed",
        dry_run=False,
        chunk_size=7,
        group_size=8,
        disk_safety_margin_bytes=0,
    )

    assert report.layout.quantization == "mlx-mxfp4"
    assert report.layout.layers[0].expert_slot_bytes == 1632
    expected = bytearray()
    for _expert in range(2):
        for index, component in enumerate(("gate_proj", "up_proj", "down_proj")):
            out_dim = 32
            expected.extend(bytes([10 + index]) * (out_dim * 4 * 4))
            expected.extend(bytes([40 + index]) * out_dim)
    assert (tmp_path / "packed" / "layer_001.bin").read_bytes() == expected


def test_pack_experts_execute_streams_expected_bytes(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path)
    output = tmp_path / "packed"

    report = pack_experts(tmp_path, output, dry_run=False, chunk_size=3, group_size=8)

    expected = bytearray()
    for expert in range(2):
        for i, _component in enumerate(DEFAULT_EXPERT_COMPONENTS):
            value = i + 1 if expert == 0 else 101 + i
            _dtype, _shape, size = _affine_component_meta(_component)
            expected.extend(bytes([value]) * size)

    assert report.disk_checked is True
    assert report.raw_quantization_extra_heap_bytes == 0
    assert report.raw_quantization_max_source_block_bytes == 0
    assert report.raw_quantization_max_output_block_bytes == 0
    assert report.raw_quantization_max_rows_per_block == 0
    assert (output / "layout.json").exists()
    assert (output / "layer_001.bin").read_bytes() == bytes(expected)


def test_raw_quantizer_read_exact_uses_single_mutable_buffer(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    fd = os.open(source, os.O_RDONLY)
    try:
        raw = packer_module._read_exact(fd, 1, 8, 3)
    finally:
        os.close(fd)

    assert isinstance(raw, bytearray)
    assert raw == bytearray(b"12345678")


def test_raw_quantizer_returns_mutable_output_blocks_without_copy() -> None:
    raw = b"".join(_f32_to_bf16(float(value)) for value in range(8))

    weight, scales, biases = packer_module._quantize_affine_int4_rows(
        raw,
        dtype="BF16",
        rows=1,
        in_dim=8,
        group_size=8,
    )

    assert isinstance(weight, bytearray)
    assert isinstance(scales, bytearray)
    assert isinstance(biases, bytearray)
    assert struct.unpack_from("<I", weight, 0) == (
        _expected_affine_word([float(i) for i in range(8)]),
    )
    assert scales[:2] == _f32_to_bf16(7.0 / 15.0)
    assert biases[:2] == _f32_to_bf16(0.0)


def test_pack_experts_raw_quantization_heap_budget_counts_streamed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_raw_bf16_checkpoint(tmp_path)
    output = tmp_path / "packed"
    monkeypatch.setattr(packer_module, "estimate_pack_peak_heap_bytes", lambda **_: 100)

    report = pack_experts(
        tmp_path,
        output,
        dry_run=True,
        chunk_size=64,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
        max_heap_bytes=132,
    )

    assert report.estimated_peak_heap_bytes == 132
    assert report.raw_quantization_extra_heap_bytes == 32
    assert report.raw_quantization_max_source_block_bytes == 64
    assert report.raw_quantization_max_output_block_bytes == 32
    assert report.raw_quantization_max_rows_per_block == 4
    with pytest.raises(SafetyError, match="estimated packer heap 132 exceeds"):
        pack_experts(
            tmp_path,
            output,
            dry_run=False,
            chunk_size=64,
            disk_safety_margin_bytes=0,
            quantize_raw_to_int4=True,
            group_size=8,
            max_heap_bytes=131,
        )
    assert not output.exists()


def test_pack_experts_raw_quantization_streams_row_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_raw_bf16_checkpoint(tmp_path)
    output = tmp_path / "packed"
    read_sizes: list[int] = []
    original_read_exact = packer_module._read_exact

    def spy_read_exact(
        fd: int,
        offset: int,
        size: int,
        chunk_size: int,
    ) -> bytearray:
        read_sizes.append(size)
        return original_read_exact(fd, offset, size, chunk_size)

    monkeypatch.setattr(packer_module, "_read_exact", spy_read_exact)

    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=64,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    assert report.layout.total_bytes == 768
    assert report.raw_quantization_max_source_block_bytes <= 64
    assert report.raw_quantization_max_output_block_bytes == 32
    assert read_sizes
    assert max(read_sizes) <= 64
    assert len(read_sizes) == 24
    assert (output / "layer_001.bin").stat().st_size == 768


def test_pack_experts_raw_quantization_reports_oversized_row_block_budget(
    tmp_path: Path,
) -> None:
    _make_raw_bf16_checkpoint(tmp_path)
    output = tmp_path / "packed"

    report = pack_experts(
        tmp_path,
        output,
        dry_run=True,
        chunk_size=1,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    assert report.raw_quantization_max_source_block_bytes == 32
    assert report.raw_quantization_max_output_block_bytes == 16
    assert report.raw_quantization_extra_heap_bytes == 47
    assert report.raw_quantization_max_rows_per_block == 1


def test_pack_experts_removes_partial_layer_file_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_checkpoint(tmp_path)
    output = tmp_path / "packed"

    def fail_copy(**kwargs) -> None:
        os.pwrite(kwargs["dst_fd"], b"partial", kwargs["dst_offset"])
        raise PackerError("copy exploded")

    monkeypatch.setattr(packer_module, "_copy_slice", fail_copy)

    with pytest.raises(PackerError, match="copy exploded"):
        pack_experts(
            tmp_path,
            output,
            dry_run=False,
            chunk_size=3,
            group_size=8,
            disk_safety_margin_bytes=0,
        )

    assert not (output / "layer_001.bin").exists()
    assert not (output / "layout.json").exists()


def test_pack_experts_preserves_existing_layout_when_atomic_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_checkpoint(tmp_path)
    output = tmp_path / "packed"
    output.mkdir()
    layout_path = output / "layout.json"
    layout_path.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(layout_module.Path, "replace", fail_replace)

    with pytest.raises(PackerError, match="replace failed"):
        pack_experts(
            tmp_path,
            output,
            dry_run=False,
            force=True,
            chunk_size=3,
            group_size=8,
            disk_safety_margin_bytes=0,
        )

    assert json.loads(layout_path.read_text(encoding="utf-8")) == {"old": True}
    assert not layout_path.with_name(layout_path.name + ".tmp").exists()
    assert not (output / "layer_001.bin").exists()


def test_pack_experts_defaults_to_dry_run(tmp_path: Path) -> None:
    _make_checkpoint(tmp_path)
    output = tmp_path / "packed"

    report = pack_experts(tmp_path, output, group_size=8)

    assert report.dry_run is True
    assert not output.exists()


def test_pack_experts_quantizes_raw_bf16_experts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    constant = _f32_to_bf16(2.0) * 64
    tensors = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": constant,
        "model.layers.1.mlp.experts.0.up_proj.weight": constant,
        "model.layers.1.mlp.experts.0.down_proj.weight": constant,
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "BF16",
            "shape": [8, 8],
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "packed"
    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=17,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    assert report.layout.quantization == "largerlm-affine-int4"
    layer = report.layout.layers[0]
    assert layer.expert_slot_bytes == 192
    data = (output / "layer_001.bin").read_bytes()
    assert len(data) == 192
    for base in (0, 64, 128):
        assert data[base : base + 32] == b"\x00" * 32
        assert data[base + 32 : base + 48] == b"\x00" * 16
        assert data[base + 48 : base + 64] == _f32_to_bf16(2.0) * 8


def test_pack_experts_quantizes_raw_bf16_affine_lanes(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    values = [float(i) for i in range(8)]
    row = b"".join(_f32_to_bf16(value) for value in values)
    matrix = row * 8
    tensors = {
        f"model.layers.1.mlp.experts.0.{component}.weight": matrix
        for component in ("gate_proj", "up_proj", "down_proj")
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "BF16",
            "shape": [8, 8],
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "packed"
    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    assert report.layout.layers[0].expert_slot_bytes == 192
    data = (output / "layer_001.bin").read_bytes()
    expected_word = _expected_affine_word(values)
    for base in (0, 64, 128):
        assert struct.unpack_from("<I", data, base) == (expected_word,)
        assert data[base + 32 : base + 34] == _f32_to_bf16(7.0 / 15.0)
        assert data[base + 48 : base + 50] == _f32_to_bf16(0.0)


def test_pack_experts_raw_quantization_rejects_nonfinite_and_cleans_partial(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    good_matrix = b"".join(struct.pack("<f", 1.0) for _ in range(64))
    bad_values = [float("nan")] + [1.0] * 63
    bad_matrix = b"".join(struct.pack("<f", value) for value in bad_values)
    tensors = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": (
            "F32",
            [8, 8],
            bad_matrix,
        ),
        "model.layers.1.mlp.experts.0.up_proj.weight": (
            "F32",
            [8, 8],
            good_matrix,
        ),
        "model.layers.1.mlp.experts.0.down_proj.weight": (
            "F32",
            [8, 8],
            good_matrix,
        ),
    }
    _write_safetensors(tmp_path, tensors)

    output = tmp_path / "packed"
    with pytest.raises(PackerError, match="non-finite value"):
        pack_experts(
            tmp_path,
            output,
            dry_run=False,
            chunk_size=17,
            disk_safety_margin_bytes=0,
            quantize_raw_to_int4=True,
            group_size=8,
        )

    assert not (output / "layer_001.bin").exists()
    assert not (output / "layout.json").exists()


def test_pack_experts_quantizes_fused_raw_bf16_experts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    expert0 = _f32_to_bf16(2.0) * 64
    expert1 = _f32_to_bf16(3.0) * 64
    tensors = {
        "model.layers.1.mlp.experts.gate_proj.weight": expert0 + expert1,
        "model.layers.1.mlp.experts.up_proj.weight": expert0 + expert1,
        "model.layers.1.mlp.experts.down_proj.weight": expert0 + expert1,
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "BF16",
            "shape": [2, 8, 8],
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "packed"
    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=17,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    layer = report.layout.layers[0]
    assert layer.expert_slot_bytes == 192
    assert layer.components[0].shape == (8, 1)
    data = (output / "layer_001.bin").read_bytes()
    assert len(data) == 2 * 192
    for expert, value in enumerate((2.0, 3.0)):
        slot_base = expert * 192
        for component_base in (0, 64, 128):
            base = slot_base + component_base
            assert data[base : base + 32] == b"\x00" * 32
            assert data[base + 32 : base + 48] == b"\x00" * 16
            assert data[base + 48 : base + 64] == _f32_to_bf16(value) * 8


def test_pack_experts_quantizes_fused_raw_w_alias_experts(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    shapes = {
        "w1": [2, 16, 8],
        "w3": [2, 16, 8],
        "w2": [2, 8, 16],
    }
    values = {
        "w1": (2.0, 3.0),
        "w3": (4.0, 5.0),
        "w2": (6.0, 7.0),
    }
    tensors = {}
    for alias, (expert0, expert1) in values.items():
        element_count = shapes[alias][1] * shapes[alias][2]
        tensors[f"model.layers.1.mlp.experts.{alias}.weight"] = (
            "BF16",
            shapes[alias],
            _f32_to_bf16(expert0) * element_count
            + _f32_to_bf16(expert1) * element_count,
        )
    _write_safetensors(tmp_path, tensors)

    output = tmp_path / "packed"
    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=17,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    layer = report.layout.layers[0]
    assert [component.name for component in layer.components] == list(
        DEFAULT_EXPERT_COMPONENTS
    )
    data = (output / "layer_001.bin").read_bytes()
    component_by_name = {component.name: component for component in layer.components}
    expected_biases = {
        "gate_proj.biases": (2.0, 3.0),
        "up_proj.biases": (4.0, 5.0),
        "down_proj.biases": (6.0, 7.0),
    }
    for expert, slot_base in enumerate((0, layer.expert_slot_bytes)):
        for component_name, per_expert_values in expected_biases.items():
            component = component_by_name[component_name]
            start = slot_base + component.offset
            assert data[start : start + component.size] == (
                _f32_to_bf16(per_expert_values[expert]) * (component.size // 2)
            )


def test_pack_experts_quantizes_fused_raw_gate_up_experts(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)

    def matrix(value: float, rows: int, cols: int) -> bytes:
        return _f32_to_bf16(value) * (rows * cols)

    gate_up = (
        matrix(2.0, 16, 8)
        + matrix(4.0, 16, 8)
        + matrix(3.0, 16, 8)
        + matrix(5.0, 16, 8)
    )
    down = matrix(6.0, 8, 16) + matrix(7.0, 8, 16)
    tensors = {
        "model.layers.1.mlp.experts.gate_up_proj.weight": (
            "BF16",
            [2, 32, 8],
            gate_up,
        ),
        "model.layers.1.mlp.experts.down_proj.weight": (
            "BF16",
            [2, 8, 16],
            down,
        ),
    }
    _write_safetensors(tmp_path, tensors)

    layout, sources = build_packed_layout(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
    )
    gate0 = sources[(1, 0)]["gate_proj.weight"]
    up0 = sources[(1, 0)]["up_proj.weight"]
    assert gate0.tensor.name == up0.tensor.name
    assert gate0.logical_shape == (16, 8)
    assert up0.logical_shape == (16, 8)
    assert up0.offset == gate0.offset + gate0.size

    output = tmp_path / "packed"
    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=17,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )

    assert report.layout.total_bytes == layout.total_bytes
    layer = report.layout.layers[0]
    component_by_name = {component.name: component for component in layer.components}
    data = (output / "layer_001.bin").read_bytes()
    expected_biases = {
        "gate_proj.biases": (2.0, 3.0),
        "up_proj.biases": (4.0, 5.0),
        "down_proj.biases": (6.0, 7.0),
    }
    for expert, slot_base in enumerate((0, layer.expert_slot_bytes)):
        for component_name, per_expert_values in expected_biases.items():
            component = component_by_name[component_name]
            start = slot_base + component.offset
            assert data[start : start + component.size] == (
                _f32_to_bf16(per_expert_values[expert]) * (component.size // 2)
            )


def test_pack_experts_copies_fused_quantized_gate_up_components(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)

    def fill(value: int, size: int) -> bytes:
        return bytes([value]) * size

    tensors = {
        "model.layers.1.mlp.experts.gate_up_proj.weight": (
            "U32",
            [2, 32, 1],
            fill(0x11, 64)
            + fill(0x12, 64)
            + fill(0x21, 64)
            + fill(0x22, 64),
        ),
        "model.layers.1.mlp.experts.gate_up_proj.scales": (
            "F16",
            [2, 32, 1],
            fill(0x31, 32)
            + fill(0x32, 32)
            + fill(0x41, 32)
            + fill(0x42, 32),
        ),
        "model.layers.1.mlp.experts.gate_up_proj.biases": (
            "F16",
            [2, 32, 1],
            fill(0x51, 32)
            + fill(0x52, 32)
            + fill(0x61, 32)
            + fill(0x62, 32),
        ),
        "model.layers.1.mlp.experts.down_proj.weight": (
            "U32",
            [2, 8, 2],
            fill(0x71, 64) + fill(0x72, 64),
        ),
        "model.layers.1.mlp.experts.down_proj.scales": (
            "BF16",
            [2, 8, 2],
            fill(0x81, 32) + fill(0x82, 32),
        ),
        "model.layers.1.mlp.experts.down_proj.biases": (
            "BF16",
            [2, 8, 2],
            fill(0x91, 32) + fill(0x92, 32),
        ),
    }
    _write_safetensors(tmp_path, tensors)

    layout, sources = build_packed_layout(tmp_path, group_size=8)

    gate0 = sources[(1, 0)]["gate_proj.weight"]
    up0 = sources[(1, 0)]["up_proj.weight"]
    assert gate0.tensor.name == up0.tensor.name
    assert gate0.logical_shape == (16, 1)
    assert up0.logical_shape == (16, 1)
    assert up0.offset == gate0.offset + gate0.size
    assert sources[(1, 0)]["gate_proj.scales"].dtype == "F16"
    assert sources[(1, 0)]["up_proj.biases"].dtype == "F16"

    output = tmp_path / "packed"
    report = pack_experts(
        tmp_path,
        output,
        dry_run=False,
        chunk_size=17,
        disk_safety_margin_bytes=0,
        group_size=8,
    )

    assert report.layout.total_bytes == layout.total_bytes
    layer = report.layout.layers[0]
    component_by_name = {component.name: component for component in layer.components}
    data = (output / "layer_001.bin").read_bytes()
    expected_bytes = {
        "gate_proj.weight": (0x11, 0x21),
        "gate_proj.scales": (0x31, 0x41),
        "gate_proj.biases": (0x51, 0x61),
        "up_proj.weight": (0x12, 0x22),
        "up_proj.scales": (0x32, 0x42),
        "up_proj.biases": (0x52, 0x62),
        "down_proj.weight": (0x71, 0x72),
        "down_proj.scales": (0x81, 0x82),
        "down_proj.biases": (0x91, 0x92),
    }
    for expert, slot_base in enumerate((0, layer.expert_slot_bytes)):
        for component_name, values in expected_bytes.items():
            component = component_by_name[component_name]
            start = slot_base + component.offset
            assert data[start : start + component.size] == fill(
                values[expert],
                component.size,
            )


def test_build_packed_layout_rejects_fused_raw_gate_up_shape_mismatch(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    tensors = {
        "model.layers.1.mlp.experts.gate_up_proj.weight": (
            "BF16",
            [2, 16, 16],
            _f32_to_bf16(1.0) * (2 * 16 * 16),
        ),
        "model.layers.1.mlp.experts.down_proj.weight": (
            "BF16",
            [2, 8, 16],
            _f32_to_bf16(1.0) * (2 * 8 * 16),
        ),
    }
    _write_safetensors(tmp_path, tensors)

    with pytest.raises(
        PackerError,
        match=r"fused raw routed expert component gate_up_proj\.weight shape",
    ):
        build_packed_layout(
            tmp_path,
            quantize_raw_to_int4=True,
            group_size=8,
        )


def test_build_packed_layout_rejects_duplicate_component_aliases(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    matrix = _f32_to_bf16(2.0) * (16 * 8)
    tensors = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": ("BF16", [16, 8], matrix),
        "model.layers.1.mlp.experts.0.w1.weight": ("BF16", [16, 8], matrix),
        "model.layers.1.mlp.experts.0.up_proj.weight": ("BF16", [16, 8], matrix),
        "model.layers.1.mlp.experts.0.down_proj.weight": (
            "BF16",
            [8, 16],
            matrix,
        ),
    }
    _write_safetensors(tmp_path, tensors)

    with pytest.raises(PackerError, match="duplicate routed expert component"):
        build_packed_layout(
            tmp_path,
            quantize_raw_to_int4=True,
            group_size=8,
            layers={1},
        )


def test_build_packed_layout_rejects_raw_dtype_shape_byte_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    tensors = {
        f"model.layers.1.mlp.experts.0.{component}.weight": (
            _f32_to_bf16(1.0) * 63
        )
        for component in ("gate_proj", "up_proj", "down_proj")
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "BF16",
            "shape": [8, 8],
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SafetensorsError, match="expects 128 bytes.*span 126 bytes"):
        build_packed_layout(
            tmp_path,
            quantize_raw_to_int4=True,
            group_size=8,
        )


def test_build_packed_layout_rejects_out_of_range_per_expert_id(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert in (0, 1):
        for component in ("gate_proj", "up_proj", "down_proj"):
            shape = [8, 16] if component == "down_proj" else [16, 8]
            tensors[f"model.layers.1.mlp.experts.{expert}.{component}.weight"] = (
                "BF16",
                shape,
                _f32_to_bf16(1.0) * (shape[0] * shape[1]),
            )
    tensors["model.layers.1.mlp.experts.2.gate_proj.weight"] = (
        "BF16",
        [16, 8],
        _f32_to_bf16(1.0) * (16 * 8),
    )
    _write_safetensors(tmp_path, tensors)

    with pytest.raises(
        PackerError,
        match=r"routed expert id 2 .* outside configured range \[0, 2\)",
    ):
        build_packed_layout(
            tmp_path,
            quantize_raw_to_int4=True,
            group_size=8,
        )


def test_build_packed_layout_rejects_same_size_shape_mismatch(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert, shape in ((0, [8, 8]), (1, [4, 16])):
        for component in ("gate_proj", "up_proj", "down_proj"):
            tensors[
                f"model.layers.1.mlp.experts.{expert}.{component}.weight"
            ] = ("BF16", shape, _f32_to_bf16(1.0) * 64)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PackerError, match="shape .* does not match config expected"):
        build_packed_layout(
            tmp_path,
            quantize_raw_to_int4=True,
            group_size=8,
        )


def test_build_packed_layout_rejects_raw_config_shape_mismatch(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for component in ("gate_proj", "up_proj", "down_proj"):
        shape = [8, 8]
        if component == "down_proj":
            shape = [4, 16]
        tensors[f"model.layers.1.mlp.experts.0.{component}.weight"] = (
            "BF16",
            shape,
            _f32_to_bf16(1.0) * 64,
        )
    shard = tmp_path / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PackerError,
        match=r"down_proj\.weight shape .* does not match config expected",
    ):
        build_packed_layout(
            tmp_path,
            quantize_raw_to_int4=True,
            group_size=8,
        )


def test_build_packed_layout_rejects_packed_affine_shape_mismatch(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for i, component in enumerate(DEFAULT_EXPERT_COMPONENTS):
        dtype, shape, size = _affine_component_meta(component)
        if component == "gate_proj.weight":
            shape = [8, 2]
        data = bytes([i + 1]) * size
        tensors[f"model.layers.1.mlp.experts.0.{component}"] = (dtype, shape, data)
        tensors[f"model.layers.1.mlp.experts.1.{component}"] = (dtype, shape, data)
    _write_safetensors(tmp_path, tensors)

    with pytest.raises(
        PackerError,
        match=r"gate_proj\.weight shape .* does not match expected",
    ):
        build_packed_layout(tmp_path, group_size=8)


def test_build_packed_layout_accepts_f16_affine_metadata(tmp_path: Path) -> None:
    _write_config(tmp_path)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for i, component in enumerate(DEFAULT_EXPERT_COMPONENTS):
        dtype, shape, size = _affine_component_meta(component)
        if component.endswith((".scales", ".biases")):
            dtype = "F16"
        data = bytes([i + 1]) * size
        tensors[f"model.layers.1.mlp.experts.0.{component}"] = (dtype, shape, data)
        tensors[f"model.layers.1.mlp.experts.1.{component}"] = (dtype, shape, data)
    _write_safetensors(tmp_path, tensors)

    layout, _sources = build_packed_layout(tmp_path, group_size=8)

    by_name = {component.name: component for component in layout.layers[0].components}
    assert by_name["gate_proj.weight"].dtype == "U32"
    assert by_name["gate_proj.scales"].dtype == "F16"
    assert by_name["gate_proj.biases"].dtype == "F16"


def test_build_packed_layout_accepts_common_affine_dtype_aliases(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for i, component in enumerate(DEFAULT_EXPERT_COMPONENTS):
        dtype, shape, size = _affine_component_meta(component)
        if component.endswith(".weight"):
            dtype = "uint32"
        elif component.endswith((".scales", ".biases")):
            dtype = "bfloat16"
        data = bytes([i + 1]) * size
        tensors[f"model.layers.1.mlp.experts.0.{component}"] = (dtype, shape, data)
        tensors[f"model.layers.1.mlp.experts.1.{component}"] = (dtype, shape, data)
    _write_safetensors(tmp_path, tensors)

    layout, _sources = build_packed_layout(tmp_path, group_size=8)

    by_name = {component.name: component for component in layout.layers[0].components}
    assert by_name["gate_proj.weight"].dtype == "uint32"
    assert by_name["gate_proj.scales"].dtype == "bfloat16"
    assert by_name["gate_proj.biases"].dtype == "bfloat16"
