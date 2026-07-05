from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from largerlm.cli import main as cli_main
from largerlm import dsa_indexer as dsa_indexer_module
from largerlm.dsa_indexer import (
    DSAIndexerError,
    compute_dsa_topk_batch,
    run_dsa_indexer_batch,
    write_dsa_index_cache_batch,
)


def _pack(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _write_resident(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir()
    tensors: list[dict[str, object]] = []
    payload = bytearray()

    def add(name: str, values: list[float], shape: list[int]) -> None:
        data = _pack(values)
        offset = len(payload)
        payload.extend(data)
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": len(data),
                "dtype": "F32",
                "shape": shape,
                "category": "attention",
            }
        )

    prefix = "model.layers.0.self_attn.indexer"
    add(f"{prefix}.wk.weight", [1.0, 0.0, 0.0, 1.0], [2, 2])
    add(f"{prefix}.k_norm.weight", [1.0, 1.0], [2])
    add(f"{prefix}.k_norm.bias", [0.0, 0.0], [2])
    add(f"{prefix}.wq_b.weight", [1.0, 0.0, 0.0, 1.0], [2, 2])
    add(f"{prefix}.weights_proj.weight", [1.0, 1.0], [1, 2])

    (resident / "resident.bin").write_bytes(payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "alignment": 1,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": tensors,
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def _pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def _write_mxfp4_resident(root: Path) -> Path:
    resident = root / "resident_mxfp4"
    resident.mkdir()
    tensors: list[dict[str, object]] = []
    payload = bytearray()

    def add_mxfp4(name: str, row_codes: list[int]) -> None:
        weight = b"".join(struct.pack("<I", _pack8(code)) for code in row_codes)
        scale = bytes([127]) * len(row_codes)
        offset = len(payload)
        payload.extend(weight)
        tensors.append(
            {
                "name": f"{name}.weight",
                "offset": offset,
                "size": len(weight),
                "dtype": "U32",
                "shape": [len(row_codes), 1],
                "category": "attention",
            }
        )
        offset = len(payload)
        payload.extend(scale)
        tensors.append(
            {
                "name": f"{name}.scales",
                "offset": offset,
                "size": len(scale),
                "dtype": "U8",
                "shape": [len(row_codes), 1],
                "category": "attention",
            }
        )

    def add_f32(name: str, values: list[float], shape: list[int]) -> None:
        data = _pack(values)
        offset = len(payload)
        payload.extend(data)
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": len(data),
                "dtype": "F32",
                "shape": shape,
                "category": "attention",
            }
        )

    prefix = "model.layers.0.self_attn.indexer"
    add_mxfp4(f"{prefix}.wk", [2, 1])
    add_f32(f"{prefix}.k_norm.weight", [1.0, 1.0], [2])
    add_f32(f"{prefix}.k_norm.bias", [0.0, 0.0], [2])
    add_mxfp4(f"{prefix}.wq_b", [2, 1])
    add_mxfp4(f"{prefix}.weights_proj", [2])

    (resident / "resident.bin").write_bytes(payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "alignment": 1,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": tensors,
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def _write_cache(root: Path) -> tuple[Path, Path]:
    segment_bytes = 4 * 2 * 4
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "max_context_tokens": 4,
        "dtype": "F32",
        "dtype_bytes": 4,
        "alignment": 1,
        "total_bytes": segment_bytes,
        "segments": [
            {
                "kind": "dsa_index",
                "layer": 0,
                "offset": 0,
                "width": 2,
                "dtype": "F32",
                "dtype_bytes": 4,
                "max_context_tokens": 4,
            }
        ],
    }
    layout_path = root / "cache_layout.json"
    cache_path = root / "cache.bin"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    cache_path.write_bytes(b"\0" * segment_bytes)
    return layout_path, cache_path


def _mutate_json(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_interleave_fixture(root: Path) -> tuple[Path, Path, Path]:
    resident = root / "resident4"
    resident.mkdir()
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, values: list[float], shape: list[int]) -> None:
        data = _pack(values)
        offset = len(payload)
        payload.extend(data)
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": len(data),
                "dtype": "F32",
                "shape": shape,
                "category": "attention",
            }
        )

    prefix = "model.layers.0.self_attn.indexer"
    identity4 = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    add(f"{prefix}.wk.weight", identity4, [4, 4])
    add(f"{prefix}.k_norm.weight", [1.0, 1.0, 1.0, 1.0], [4])
    add(f"{prefix}.k_norm.bias", [0.0, 0.0, 0.0, 0.0], [4])
    add(f"{prefix}.wq_b.weight", identity4, [4, 4])
    add(f"{prefix}.weights_proj.weight", [1.0, 1.0, 1.0, 1.0], [1, 4])
    (resident / "resident.bin").write_bytes(payload)
    layout_path = resident / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 1,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )

    cache_layout = root / "cache_layout4.json"
    cache_file = root / "cache4.bin"
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 2,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 1,
                "total_bytes": 32,
                "segments": [
                    {
                        "kind": "dsa_index",
                        "layer": 0,
                        "offset": 0,
                        "width": 4,
                        "dtype": "F32",
                        "dtype_bytes": 4,
                        "max_context_tokens": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_file.write_bytes(b"\0" * 32)
    return layout_path, cache_layout, cache_file


def _read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def test_write_dsa_index_cache_batch_writes_layernormed_rows(tmp_path: Path) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))

    result = write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    assert result.cache_write_bytes == 16
    values = _read_f32(cache_file)
    expected = 1.0 / math.sqrt(1.0 + 4e-6)
    assert values[:4] == pytest.approx((expected, -expected, -expected, expected))


def test_write_dsa_index_cache_batch_supports_interleaved_rope(tmp_path: Path) -> None:
    resident, cache_layout, cache_file = _write_interleave_fixture(tmp_path)
    hidden = tmp_path / "hidden4.f32"
    hidden_values = [1.0, 2.0, 4.0, 8.0]
    hidden.write_bytes(_pack(hidden_values))

    result = write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=1,
        batch_tokens=1,
        qk_rope_dim=4,
        rope_interleave=True,
        layer_norm_eps=1e-6,
    )

    mean = sum(hidden_values) / len(hidden_values)
    variance = sum((value - mean) ** 2 for value in hidden_values) / len(hidden_values)
    normed = [(value - mean) / math.sqrt(variance + 1e-6) for value in hidden_values]
    cos0 = math.cos(1.0)
    sin0 = math.sin(1.0)
    angle1 = 1.0 / (10000.0 ** 0.5)
    cos1 = math.cos(angle1)
    sin1 = math.sin(angle1)
    expected = (
        normed[0] * cos0 - normed[1] * sin0,
        normed[1] * cos0 + normed[0] * sin0,
        normed[2] * cos1 - normed[3] * sin1,
        normed[3] * cos1 + normed[2] * sin1,
    )
    assert result.rope_interleave is True
    assert _read_f32(cache_file)[4:8] == pytest.approx(expected)


def test_write_dsa_index_cache_batch_numpy_fast_path_writes_bf16_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if dsa_indexer_module._np is None:
        pytest.skip("numpy fast path is not available")
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    _mutate_json(
        cache_layout,
        lambda payload: (
            payload.update({"dtype": "BF16", "dtype_bytes": 2, "total_bytes": 16}),
            payload["segments"][0].update({"dtype": "BF16", "dtype_bytes": 2}),
        ),
    )
    cache_file.write_bytes(b"\0" * 16)
    hidden = tmp_path / "hidden_bf16_cache.f32"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))

    def fail_matvec(*args: object, **kwargs: object) -> object:
        raise AssertionError("batch DSA cache write should not call _matvec")

    monkeypatch.setattr(dsa_indexer_module, "_matvec", fail_matvec)

    result = write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    raw = cache_file.read_bytes()
    values = [
        dsa_indexer_module._bf16_to_f32(raw, offset)
        for offset in range(0, 8, 2)
    ]
    expected = 1.0 / math.sqrt(1.0 + 4e-6)
    assert result.cache_dtype == "BF16"
    assert values == pytest.approx([expected, -expected, -expected, expected], abs=0.01)


def test_compute_dsa_topk_batch_respects_causal_prefix(tmp_path: Path) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    indices = tmp_path / "topk.json"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    q_resid.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    result = compute_dsa_topk_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        q_resid_f32_path=q_resid,
        output_indices_path=indices,
        start_position=0,
        batch_tokens=2,
        context_length=2,
        index_topk=1,
        index_n_heads=1,
        qk_rope_dim=0,
    )

    assert result.topk_indices == ((0,), (1,))
    payload = json.loads(indices.read_text(encoding="utf-8"))
    assert payload["topk_indices"] == [[0], [1]]


def test_compute_dsa_topk_batch_numpy_batch_path_bypasses_matvec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if dsa_indexer_module._np is None:
        pytest.skip("numpy fast path is not available")
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    q_resid.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    def fail_matvec(*args: object, **kwargs: object) -> object:
        raise AssertionError("batch DSA top-k should not call _matvec")

    monkeypatch.setattr(dsa_indexer_module, "_matvec", fail_matvec)

    result = compute_dsa_topk_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        q_resid_f32_path=q_resid,
        output_indices_path=None,
        start_position=0,
        batch_tokens=2,
        context_length=2,
        index_topk=1,
        index_n_heads=1,
        qk_rope_dim=0,
    )

    assert result.topk_indices == ((0,), (1,))


def test_compute_dsa_topk_batch_json_replace_failure_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    indices = tmp_path / "topk.json"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    q_resid.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    indices.write_text('{"old": true}', encoding="utf-8")
    write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(dsa_indexer_module.Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        compute_dsa_topk_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=hidden,
            q_resid_f32_path=q_resid,
            output_indices_path=indices,
            start_position=0,
            batch_tokens=2,
            context_length=2,
            index_topk=1,
            index_n_heads=1,
            qk_rope_dim=0,
        )

    assert json.loads(indices.read_text(encoding="utf-8")) == {"old": True}
    assert not indices.with_name(indices.name + ".tmp").exists()


def test_compute_dsa_topk_batch_writes_padded_u32_rows(tmp_path: Path) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    indices_u32 = tmp_path / "topk.u32"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    q_resid.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    result = compute_dsa_topk_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        q_resid_f32_path=q_resid,
        output_indices_path=None,
        output_indices_u32_path=indices_u32,
        start_position=0,
        batch_tokens=2,
        context_length=2,
        index_topk=2,
        index_n_heads=1,
        qk_rope_dim=0,
        collect_topk_indices=False,
    )

    assert result.topk_indices == ()
    assert result.topk_indices_collected is False
    assert result.output_indices_u32_bytes == 24
    assert struct.unpack("<6I", indices_u32.read_bytes()) == (1, 0, 0, 2, 1, 0)


def test_compute_dsa_topk_batch_fast_paths_singleton_u32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    indices_u32 = tmp_path / "topk_singleton.u32"
    hidden.write_bytes(_pack([1.0, 0.0]))
    q_resid.write_bytes(_pack([1.0, 0.0]))
    write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=1,
        qk_rope_dim=0,
    )

    def fail_tensor_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("singleton u32 DSA top-k should not load indexer matrices")

    monkeypatch.setattr(dsa_indexer_module, "_tensor_to_f32_array", fail_tensor_load)

    result = compute_dsa_topk_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        q_resid_f32_path=q_resid,
        output_indices_path=None,
        output_indices_u32_path=indices_u32,
        start_position=0,
        batch_tokens=1,
        context_length=1,
        index_topk=2,
        index_n_heads=1,
        qk_rope_dim=0,
        collect_topk_indices=False,
    )

    assert result.cache_read_bytes == 0
    assert result.topk_indices == ()
    assert result.output_indices_u32_bytes == 12
    assert struct.unpack("<3I", indices_u32.read_bytes()) == (1, 0, 0)


def test_compute_dsa_topk_batch_u32_replace_failure_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    indices_u32 = tmp_path / "topk.u32"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    q_resid.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))
    indices_u32.write_bytes(b"old")
    write_dsa_index_cache_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        start_position=0,
        batch_tokens=2,
        qk_rope_dim=0,
    )

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(dsa_indexer_module.Path, "replace", fail_replace)

    with pytest.raises(DSAIndexerError, match="replace failed"):
        compute_dsa_topk_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=hidden,
            q_resid_f32_path=q_resid,
            output_indices_path=None,
            output_indices_u32_path=indices_u32,
            start_position=0,
            batch_tokens=2,
            context_length=2,
            index_topk=2,
            index_n_heads=1,
            qk_rope_dim=0,
        )

    assert indices_u32.read_bytes() == b"old"
    assert not indices_u32.with_name(indices_u32.name + ".tmp").exists()


def test_run_dsa_indexer_batch_composes_cache_write_and_topk(tmp_path: Path) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    hidden.write_bytes(_pack([1.0, 0.0]))
    q_resid.write_bytes(_pack([1.0, 0.0]))

    result = run_dsa_indexer_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        q_resid_f32_path=q_resid,
        output_indices_path=None,
        start_position=0,
        batch_tokens=1,
        context_length=1,
        index_topk=1,
        index_n_heads=1,
        qk_rope_dim=0,
    )

    assert result.cache_write.batch_tokens == 1
    assert result.topk.topk_indices == ((0,),)


def test_dsa_indexer_batch_supports_mxfp4_indexer_matrices(tmp_path: Path) -> None:
    resident = _write_mxfp4_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden_mxfp4.f32"
    q_resid = tmp_path / "q_resid_mxfp4.f32"
    hidden.write_bytes(_pack(([1.0 / 8.0] * 8) + ([0.5 / 8.0] * 8)))
    q_resid.write_bytes(_pack(([1.0 / 8.0] * 8) + ([0.5 / 8.0] * 8)))

    result = run_dsa_indexer_batch(
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=0,
        hidden_f32_path=hidden,
        q_resid_f32_path=q_resid,
        output_indices_path=None,
        start_position=0,
        batch_tokens=2,
        context_length=2,
        index_topk=1,
        index_n_heads=1,
        qk_rope_dim=0,
    )

    assert result.cache_write.hidden_dim == 8
    assert result.cache_write.index_head_dim == 2
    assert result.cache_write.resident_matrix_bytes == 26
    assert result.cache_write.resident_matrix_f32_bytes == 80
    assert result.topk.q_lora_dim == 8
    assert result.topk.resident_matrix_bytes == 15
    assert result.topk.resident_matrix_f32_bytes == 96
    assert result.topk.topk_indices == ((0,), (0,))


def test_compute_dsa_topk_batch_rejects_cache_read_over_cap(tmp_path: Path) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    hidden.write_bytes(_pack([1.0, 0.0]))
    q_resid.write_bytes(_pack([1.0, 0.0]))

    with pytest.raises(DSAIndexerError, match="cache read"):
        compute_dsa_topk_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=hidden,
            q_resid_f32_path=q_resid,
            output_indices_path=None,
            start_position=0,
            batch_tokens=1,
            context_length=1,
            index_topk=1,
            index_n_heads=1,
            qk_rope_dim=0,
            max_cache_read_mib=0.000001,
        )


def test_decode_cache_rows_numpy_supports_bf16() -> None:
    if dsa_indexer_module._np is None:
        pytest.skip("numpy fast path is not available")
    raw = struct.pack(
        "<4H",
        dsa_indexer_module._f32_to_bf16_bits(1.0),
        dsa_indexer_module._f32_to_bf16_bits(-2.0),
        dsa_indexer_module._f32_to_bf16_bits(0.5),
        dsa_indexer_module._f32_to_bf16_bits(3.0),
    )

    rows = dsa_indexer_module._decode_cache_rows_np(
        raw,
        dtype="BF16",
        width=2,
        rows=2,
    )

    assert rows is not None
    assert rows.shape == (2, 2)
    assert rows.reshape(-1).tolist() == pytest.approx([1.0, -2.0, 0.5, 3.0])


def test_tensor_to_f32_array_numpy_supports_bf16(tmp_path: Path) -> None:
    if dsa_indexer_module._np is None:
        pytest.skip("numpy fast path is not available")
    raw = struct.pack(
        "<3H",
        dsa_indexer_module._f32_to_bf16_bits(1.0),
        dsa_indexer_module._f32_to_bf16_bits(-0.5),
        dsa_indexer_module._f32_to_bf16_bits(4.0),
    )
    weights = tmp_path / "resident.bin"
    weights.write_bytes(raw)
    tensor = {
        "name": "model.layers.0.self_attn.indexer.test.weight",
        "offset": 0,
        "size": len(raw),
        "dtype": "BF16",
        "shape": [3],
    }

    values = dsa_indexer_module._tensor_to_f32_array(weights, tensor)

    assert list(values) == pytest.approx([1.0, -0.5, 4.0])


def test_write_dsa_index_cache_batch_rejects_boolean_matrix_shape(
    tmp_path: Path,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["shape"][0] = True

    _mutate_json(resident, mutate)

    with pytest.raises(DSAIndexerError, match="indexer wk must have shape"):
        write_dsa_index_cache_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=tmp_path / "missing-hidden.f32",
            start_position=0,
            batch_tokens=2,
            qk_rope_dim=0,
        )


def test_write_dsa_index_cache_batch_rejects_boolean_vector_shape(
    tmp_path: Path,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[1]["shape"][0] = False

    _mutate_json(resident, mutate)

    with pytest.raises(DSAIndexerError, match="indexer k_norm.weight must have shape"):
        write_dsa_index_cache_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=tmp_path / "missing-hidden.f32",
            start_position=0,
            batch_tokens=2,
            qk_rope_dim=0,
        )


def test_write_dsa_index_cache_batch_rejects_boolean_tensor_offset(
    tmp_path: Path,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(_pack([1.0, 0.0, 0.0, 1.0]))

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["offset"] = True

    _mutate_json(resident, mutate)

    with pytest.raises(DSAIndexerError, match="offset must be an integer"):
        write_dsa_index_cache_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=hidden,
            start_position=0,
            batch_tokens=2,
            qk_rope_dim=0,
        )


def test_compute_dsa_topk_batch_rejects_boolean_matrix_size(
    tmp_path: Path,
) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[3]["size"] = True

    _mutate_json(resident, mutate)

    with pytest.raises(DSAIndexerError, match="indexer wq_b size must be an integer"):
        compute_dsa_topk_batch(
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=0,
            hidden_f32_path=tmp_path / "missing-hidden.f32",
            q_resid_f32_path=tmp_path / "missing-q-resid.f32",
            output_indices_path=None,
            start_position=0,
            batch_tokens=2,
            context_length=2,
            index_topk=1,
            index_n_heads=1,
            qk_rope_dim=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layer": True}, "layer must be an integer"),
        ({"start_position": False}, "start_position must be an integer"),
        ({"batch_tokens": True}, "batch_tokens must be an integer"),
        ({"qk_rope_dim": False}, "qk_rope_dim must be an integer"),
        ({"max_resident_matrix_mib": True}, "max_resident_matrix_mib must be a finite number"),
        ({"max_cache_file_mib": False}, "max_cache_file_mib must be a finite number"),
        ({"rope_theta": True}, "rope_theta must be a finite number"),
    ],
)
def test_write_dsa_index_cache_batch_rejects_boolean_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    args: dict[str, object] = {
        "resident_layout_path": tmp_path / "missing-layout.json",
        "cache_layout_path": tmp_path / "missing-cache-layout.json",
        "cache_file_path": tmp_path / "missing-cache.bin",
        "layer": 0,
        "hidden_f32_path": tmp_path / "missing-hidden.f32",
        "start_position": 0,
        "batch_tokens": 1,
        "qk_rope_dim": 0,
    }
    args.update(kwargs)

    with pytest.raises(DSAIndexerError, match=message):
        write_dsa_index_cache_batch(**args)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layer": True}, "layer must be an integer"),
        ({"context_length": False}, "context_length must be an integer"),
        ({"index_topk": True}, "index_topk must be an integer"),
        ({"index_n_heads": False}, "index_n_heads must be an integer"),
        ({"max_cache_read_mib": True}, "max_cache_read_mib must be a finite number"),
        ({"rope_theta": False}, "rope_theta must be a finite number"),
    ],
)
def test_compute_dsa_topk_batch_rejects_boolean_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    args: dict[str, object] = {
        "resident_layout_path": tmp_path / "missing-layout.json",
        "cache_layout_path": tmp_path / "missing-cache-layout.json",
        "cache_file_path": tmp_path / "missing-cache.bin",
        "layer": 0,
        "hidden_f32_path": tmp_path / "missing-hidden.f32",
        "q_resid_f32_path": tmp_path / "missing-q-resid.f32",
        "output_indices_path": None,
        "start_position": 0,
        "batch_tokens": 1,
        "context_length": 1,
        "index_topk": 1,
        "index_n_heads": 1,
        "qk_rope_dim": 0,
    }
    args.update(kwargs)

    with pytest.raises(DSAIndexerError, match=message):
        compute_dsa_topk_batch(**args)


def test_dsa_indexer_batch_cli_writes_topk_json(tmp_path: Path) -> None:
    resident = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    hidden = tmp_path / "hidden.f32"
    q_resid = tmp_path / "q_resid.f32"
    indices = tmp_path / "indices.json"
    indices_u32 = tmp_path / "indices.u32"
    hidden.write_bytes(_pack([1.0, 0.0]))
    q_resid.write_bytes(_pack([1.0, 0.0]))

    status = cli_main(
        [
            "dsa-indexer-batch",
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--layer",
            "0",
            "--hidden-f32",
            str(hidden),
            "--q-resid-f32",
            str(q_resid),
            "--output-indices-json",
            str(indices),
            "--output-indices-u32",
            str(indices_u32),
            "--start-position",
            "0",
            "--batch-tokens",
            "1",
            "--context-length",
            "1",
            "--index-topk",
            "1",
            "--index-n-heads",
            "1",
            "--qk-rope-dim",
            "0",
            "--rope-interleave",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(indices.read_text(encoding="utf-8"))
    assert payload["rope_interleave"] is True
    assert payload["topk_indices"] == [[0]]
    assert struct.unpack("<2I", indices_u32.read_bytes()) == (1, 0)
