from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from largerlm.safetensors import (
    HEADER_MANIFEST_NAME,
    SafetensorsError,
    categorize_tensor_for_moe_layers,
    iter_tensor_metadata,
    scan_checkpoint,
    validate_local_safetensors_header,
)


def _write_shard(path: Path, header: dict, payload: bytes) -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def _write_index(
    root: Path,
    weight_map: dict[str, str],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": metadata or {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _write_header_manifest(
    root: Path,
    *,
    index: dict[str, object],
    shards: dict[str, dict[str, object]],
) -> None:
    (root / HEADER_MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "index": index, "shards": shards}),
        encoding="utf-8",
    )


def test_iter_tensor_metadata_reads_indexed_header_entries(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "__metadata__": {"format": "pt"},
        },
        b"\0" * 16,
    )
    _write_index(
        tmp_path,
        {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"},
    )

    tensors = iter_tensor_metadata(tmp_path)

    assert len(tensors) == 1
    assert tensors[0].name == "model.embed_tokens.weight"
    assert tensors[0].shape == (2, 2)
    assert tensors[0].nbytes == 16


def test_validate_local_safetensors_header_is_header_only_and_bounded(
    tmp_path: Path,
) -> None:
    shard = "model-00001-of-00001.safetensors"
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        }
    }
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {"model.embed_tokens.weight": shard},
    }
    _write_shard(tmp_path / shard, header, b"\0" * 4)

    check = validate_local_safetensors_header(tmp_path, shard, index=index)

    assert check.ok is True
    assert check.tensor_count == 1
    assert check.file_size == (tmp_path / shard).stat().st_size

    (tmp_path / shard).write_bytes(b"x" * check.file_size)

    check = validate_local_safetensors_header(tmp_path, shard, index=index)

    assert check.ok is False
    assert check.error is not None
    assert "header length" in check.error


def test_iter_tensor_metadata_reads_header_manifest_without_shards(
    tmp_path: Path,
) -> None:
    header = {
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16],
        },
        "lm_head.weight": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [16, 20],
        },
        "__metadata__": {"format": "pt"},
    }
    index = {
        "metadata": {"total_size": 20},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
            "lm_head.weight": "model-00001-of-00001.safetensors",
        },
    }
    _write_index(
        tmp_path,
        {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
            "lm_head.weight": "model-00001-of-00001.safetensors",
        },
        metadata={"total_size": 20},
    )
    _write_header_manifest(
        tmp_path,
        index=index,
        shards={
            "model-00001-of-00001.safetensors": {
                "file_size": 128,
                "data_start": 108,
                "header": header,
            }
        },
    )

    tensors = iter_tensor_metadata(tmp_path)

    assert [tensor.name for tensor in tensors] == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    assert {tensor.shard for tensor in tensors} == {
        "model-00001-of-00001.safetensors"
    }
    assert sum(tensor.nbytes for tensor in tensors) == 20


def test_iter_tensor_metadata_prefers_local_shards_over_header_manifest(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "local.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
        },
        b"\0" * 4,
    )
    _write_header_manifest(
        tmp_path,
        index={
            "metadata": {"total_size": 8},
            "weight_map": {
                "manifest.weight": "model-00001-of-00001.safetensors",
            },
        },
        shards={
            "model-00001-of-00001.safetensors": {
                "file_size": 12,
                "data_start": 8,
                "header": {
                    "manifest.weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
            }
        },
    )

    tensors = iter_tensor_metadata(tmp_path)

    assert [tensor.name for tensor in tensors] == ["local.weight"]


def test_iter_tensor_metadata_can_prefer_header_manifest_over_local_shards(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "local.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
        },
        b"\0" * 4,
    )
    _write_header_manifest(
        tmp_path,
        index={
            "metadata": {"total_size": 4},
            "weight_map": {
                "manifest.weight": "model-00001-of-00001.safetensors",
            },
        },
        shards={
            "model-00001-of-00001.safetensors": {
                "file_size": 12,
                "data_start": 8,
                "header": {
                    "manifest.weight": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    }
                },
            }
        },
    )

    tensors = iter_tensor_metadata(tmp_path, prefer_header_manifest=True)

    assert [tensor.name for tensor in tensors] == ["manifest.weight"]


def test_iter_tensor_metadata_accepts_common_dtype_aliases(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "uint32",
                "shape": [2],
                "data_offsets": [0, 8],
            },
            "model.layers.0.mlp.experts.0.gate_proj.scales": {
                "dtype": "float16",
                "shape": [4],
                "data_offsets": [8, 16],
            },
            "model.layers.0.mlp.experts.0.gate_proj.biases": {
                "dtype": "bfloat16",
                "shape": [1, 2],
                "data_offsets": [16, 20],
            },
        },
        b"\0" * 20,
    )
    _write_index(
        tmp_path,
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": (
                "model-00001-of-00001.safetensors"
            ),
            "model.layers.0.mlp.experts.0.gate_proj.scales": (
                "model-00001-of-00001.safetensors"
            ),
            "model.layers.0.mlp.experts.0.gate_proj.biases": (
                "model-00001-of-00001.safetensors"
            ),
        },
        metadata={"total_size": 20},
    )

    tensors = iter_tensor_metadata(tmp_path)

    by_name = {tensor.name: tensor for tensor in tensors}
    assert by_name[
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    ].dtype == "uint32"
    assert by_name[
        "model.layers.0.mlp.experts.0.gate_proj.scales"
    ].dtype == "float16"
    assert by_name[
        "model.layers.0.mlp.experts.0.gate_proj.biases"
    ].dtype == "bfloat16"
    assert sum(tensor.nbytes for tensor in tensors) == 20


def test_iter_tensor_metadata_reads_single_unindexed_safetensors(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "lm_head.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [16, 20],
            },
            "__metadata__": {"format": "pt"},
        },
        b"\0" * 20,
    )

    tensors = iter_tensor_metadata(tmp_path)

    assert [tensor.name for tensor in tensors] == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    assert {tensor.shard for tensor in tensors} == {"model.safetensors"}
    assert sum(tensor.nbytes for tensor in tensors) == 20


def test_scan_checkpoint_reads_single_unindexed_safetensors(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model.safetensors",
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [4, 8],
            },
        },
        b"\0" * 8,
    )

    stats = scan_checkpoint(tmp_path)

    assert stats.tensor_count == 2
    assert stats.routed_expert_bytes == 4
    assert stats.resident_bytes == 4


def test_iter_tensor_metadata_rejects_multiple_unindexed_safetensors(
    tmp_path: Path,
) -> None:
    for index in range(2):
        _write_shard(
            tmp_path / f"model-{index}.safetensors",
            {
                f"tensor_{index}.weight": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            },
            b"\0" * 4,
        )

    with pytest.raises(SafetensorsError, match="sharded safetensors checkpoints"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_accepts_matching_total_size(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "lm_head.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [16, 20],
            },
        },
        b"\0" * 20,
    )
    _write_index(
        tmp_path,
        {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
            "lm_head.weight": "model-00001-of-00001.safetensors",
        },
        metadata={"total_size": 20},
    )

    tensors = iter_tensor_metadata(tmp_path)

    assert sum(tensor.nbytes for tensor in tensors) == 20


def test_iter_tensor_metadata_rejects_total_size_mismatch(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        b"\0" * 4,
    )
    _write_index(
        tmp_path,
        {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"},
        metadata={"total_size": 8},
    )

    with pytest.raises(
        SafetensorsError,
        match="metadata.total_size 8 does not match indexed tensor bytes 4",
    ):
        iter_tensor_metadata(tmp_path)


@pytest.mark.parametrize("total_size", ["4", -1, True])
def test_iter_tensor_metadata_rejects_invalid_total_size(
    tmp_path: Path,
    total_size: object,
) -> None:
    _write_index(
        tmp_path,
        {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"},
        metadata={"total_size": total_size},
    )

    with pytest.raises(
        SafetensorsError,
        match="metadata.total_size must be a non-negative integer",
    ):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_weight_map_header_mismatch(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {},
        b"",
    )
    _write_index(
        tmp_path,
        {"missing.weight": "model-00001-of-00001.safetensors"},
    )

    with pytest.raises(SafetensorsError, match="missing.weight"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_absolute_shard_path(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {"bad.weight": str(tmp_path / "model-00001-of-00001.safetensors")},
    )

    with pytest.raises(SafetensorsError, match="escapes model directory"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_parent_relative_shard_path(
    tmp_path: Path,
) -> None:
    _write_index(
        tmp_path,
        {"bad.weight": "../model-00001-of-00001.safetensors"},
    )

    with pytest.raises(SafetensorsError, match="escapes model directory"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_unindexed_header_tensor(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "listed.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
            "extra.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [4, 8],
            },
            "__metadata__": {"format": "pt"},
        },
        b"\0" * 8,
    )
    _write_index(tmp_path, {"listed.weight": "model-00001-of-00001.safetensors"})

    with pytest.raises(SafetensorsError, match="not listed in weight_map"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_out_of_bounds_offsets(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "bad.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 32],
            }
        },
        b"\0" * 16,
    )
    _write_index(tmp_path, {"bad.weight": "model-00001-of-00001.safetensors"})

    with pytest.raises(SafetensorsError, match="exceeds shard payload"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_overlapping_offsets(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "first.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [0, 4],
            },
            "second.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [2, 6],
            },
        },
        b"\0" * 6,
    )
    _write_index(
        tmp_path,
        {
            "first.weight": "model-00001-of-00001.safetensors",
            "second.weight": "model-00001-of-00001.safetensors",
        },
    )

    with pytest.raises(SafetensorsError, match="overlaps previous tensor data"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_unindexed_data_gap(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "first.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [0, 4],
            },
            "second.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [8, 12],
            },
        },
        b"\0" * 12,
    )
    _write_index(
        tmp_path,
        {
            "first.weight": "model-00001-of-00001.safetensors",
            "second.weight": "model-00001-of-00001.safetensors",
        },
    )

    with pytest.raises(SafetensorsError, match="unindexed data gap"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_unindexed_trailing_data(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "first.weight": {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [0, 4],
            },
        },
        b"\0" * 8,
    )
    _write_index(
        tmp_path,
        {"first.weight": "model-00001-of-00001.safetensors"},
    )

    with pytest.raises(SafetensorsError, match="unindexed trailing data"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_dtype_shape_size_mismatch(
    tmp_path: Path,
) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "bad.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 12],
            }
        },
        b"\0" * 12,
    )
    _write_index(tmp_path, {"bad.weight": "model-00001-of-00001.safetensors"})

    with pytest.raises(SafetensorsError, match="expects 16 bytes.*span 12 bytes"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_unknown_dtype(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "bad.weight": {
                "dtype": "Q4",
                "shape": [4],
                "data_offsets": [0, 4],
            }
        },
        b"\0" * 4,
    )
    _write_index(tmp_path, {"bad.weight": "model-00001-of-00001.safetensors"})

    with pytest.raises(SafetensorsError, match="unsupported safetensors dtype"):
        iter_tensor_metadata(tmp_path)


def test_iter_tensor_metadata_rejects_non_integer_shape_dim(tmp_path: Path) -> None:
    _write_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "bad.weight": {
                "dtype": "F32",
                "shape": [1.5],
                "data_offsets": [0, 4],
            }
        },
        b"\0" * 4,
    )
    _write_index(tmp_path, {"bad.weight": "model-00001-of-00001.safetensors"})

    with pytest.raises(SafetensorsError, match=r"shape\[0\] must be an integer"):
        iter_tensor_metadata(tmp_path)


def test_scan_checkpoint_excludes_ignored_extra_layers_from_resident_bytes(
    tmp_path: Path,
) -> None:
    tensors = {
        "model.embed_tokens.weight": b"a" * 4,
        "model.layers.0.mlp.gate.weight": b"b" * 8,
        "model.layers.1.mlp.experts.0.gate_proj.weight": b"c" * 16,
        "model.layers.2.self_attn.indexer.wk.weight": b"d" * 32,
    }
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [start, len(payload)],
        }
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", header, bytes(payload))
    _write_index(
        tmp_path,
        {name: "model-00001-of-00001.safetensors" for name in tensors},
    )

    stats = scan_checkpoint(
        tmp_path,
        category_fn=lambda name: categorize_tensor_for_moe_layers(
            name,
            {1},
            num_hidden_layers=2,
        ),
    )

    assert stats.total_bytes == 60
    assert stats.routed_expert_bytes == 16
    assert stats.resident_bytes == 12
    assert stats.by_category["ignored_extra_layers"] == 32


def test_scan_checkpoint_classifies_fused_w_alias_experts(
    tmp_path: Path,
) -> None:
    tensors = {
        "model.embed_tokens.weight": b"a" * 4,
        "model.layers.1.mlp.experts.w1.weight": b"b" * 16,
        "model.layers.1.mlp.experts.w2.weight": b"c" * 8,
        "model.layers.1.mlp.experts.w3.weight": b"d" * 12,
    }
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [start, len(payload)],
        }
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", header, bytes(payload))
    _write_index(
        tmp_path,
        {name: "model-00001-of-00001.safetensors" for name in tensors},
    )

    stats = scan_checkpoint(
        tmp_path,
        category_fn=lambda name: categorize_tensor_for_moe_layers(
            name,
            {1},
            num_hidden_layers=2,
        ),
    )

    assert stats.routed_expert_bytes == 36
    assert stats.resident_bytes == 4
    assert stats.by_category["routed_experts"] == 36


def test_scan_checkpoint_classifies_fused_gate_up_experts(
    tmp_path: Path,
) -> None:
    tensors = {
        "model.embed_tokens.weight": b"a" * 4,
        "model.layers.1.mlp.experts.gate_up_proj.weight": b"b" * 32,
        "model.layers.1.mlp.experts.down_proj.weight": b"c" * 8,
    }
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [start, len(payload)],
        }
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", header, bytes(payload))
    _write_index(
        tmp_path,
        {name: "model-00001-of-00001.safetensors" for name in tensors},
    )

    stats = scan_checkpoint(
        tmp_path,
        category_fn=lambda name: categorize_tensor_for_moe_layers(
            name,
            {1},
            num_hidden_layers=2,
        ),
    )

    assert stats.routed_expert_bytes == 40
    assert stats.resident_bytes == 4
    assert stats.by_category["routed_experts"] == 40
