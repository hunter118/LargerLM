from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

import largerlm.preflight as preflight_module
from largerlm.cli import main as cli_main
from largerlm.hardware import HardwareInfo
from largerlm.preflight import preflight_glm_checkpoint
from largerlm.safetensors import HEADER_MANIFEST_NAME, TensorMeta
from test_tokenizer import write_simple_tokenizer


def _zeros(dtype: str, shape: list[int]) -> bytes:
    count = 1
    for dim in shape:
        count *= dim
    if dtype == "F32":
        return b"\0" * (count * 4)
    if dtype == "BF16":
        return b"\0" * (count * 2)
    raise AssertionError(dtype)


def _write_checkpoint(
    root: Path,
    *,
    omit: set[str] | None = None,
    include_indexer: bool = False,
    include_extra_layer: bool = False,
    include_router_bias: bool = False,
    include_shared_expert: bool = False,
    topk_method: str | None = None,
    tie_word_embeddings: bool | None = None,
    vocab_size: int | None = None,
    config_overrides: dict[str, object] | None = None,
    shape_overrides: dict[str, list[int]] | None = None,
    extra_tensors: dict[str, tuple[str, list[int], bytes]] | None = None,
) -> None:
    omit = omit or set()
    shape_overrides = shape_overrides or {}
    extra_tensors = extra_tensors or {}
    config = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 8,
        "intermediate_size": 8,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "n_routed_experts": 2,
        "n_shared_experts": 1 if include_shared_expert else 0,
        "num_experts_per_tok": 1,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "num_attention_heads": 2,
        "q_lora_rank": 2,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 2,
        "v_head_dim": 1,
    }
    if topk_method is not None:
        config["topk_method"] = topk_method
    if tie_word_embeddings is not None:
        config["tie_word_embeddings"] = tie_word_embeddings
    if vocab_size is not None:
        config["vocab_size"] = vocab_size
    if include_indexer:
        config.update(
            {
                "index_head_dim": 2,
                "index_n_heads": 1,
                "index_topk": 2,
                "indexer_types": ["full", "shared"],
            }
        )
    if config_overrides:
        config.update(config_overrides)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, tuple[str, list[int], bytes]] = {}

    def add(name: str, dtype: str, shape: list[int]) -> None:
        if name in omit:
            return
        shape = shape_overrides.get(name, shape)
        tensors[name] = (dtype, shape, _zeros(dtype, shape))

    add("model.embed_tokens.weight", "F32", [4, 8])
    add("model.norm.weight", "F32", [8])
    add("lm_head.weight", "F32", [4, 8])
    for layer in (0, 1):
        prefix = f"model.layers.{layer}"
        add(f"{prefix}.input_layernorm.weight", "F32", [8])
        add(f"{prefix}.self_attn.q_a_proj.weight", "F32", [2, 8])
        add(f"{prefix}.self_attn.q_a_layernorm.weight", "F32", [2])
        add(f"{prefix}.self_attn.q_b_proj.weight", "F32", [6, 2])
        add(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", "F32", [4, 8])
        add(f"{prefix}.self_attn.kv_a_layernorm.weight", "F32", [2])
        add(f"{prefix}.self_attn.kv_b_proj.weight", "F32", [4, 2])
        add(f"{prefix}.self_attn.o_proj.weight", "F32", [8, 2])
        add(f"{prefix}.post_attention_layernorm.weight", "F32", [8])

    add("model.layers.1.mlp.gate.weight", "F32", [2, 8])
    if include_router_bias:
        add("model.layers.1.mlp.gate.e_score_correction_bias", "F32", [2])
    if include_shared_expert:
        for component in ("gate_proj", "up_proj"):
            add(f"model.layers.1.mlp.shared_experts.{component}.weight", "F32", [8, 8])
        add("model.layers.1.mlp.shared_experts.down_proj.weight", "F32", [8, 8])
    if include_indexer:
        prefix = "model.layers.0.self_attn.indexer"
        add(f"{prefix}.wk.weight", "F32", [2, 8])
        add(f"{prefix}.wq_b.weight", "F32", [2, 2])
        add(f"{prefix}.weights_proj.weight", "F32", [1, 8])
        add(f"{prefix}.k_norm.weight", "F32", [2])
        add(f"{prefix}.k_norm.bias", "F32", [2])
    for component in ("gate_proj", "up_proj", "down_proj"):
        add(f"model.layers.0.mlp.switch_mlp.{component}.weight", "F32", [8, 8])
    for expert in (0, 1):
        for component in ("gate_proj", "up_proj", "down_proj"):
            add(f"model.layers.1.mlp.experts.{expert}.{component}.weight", "BF16", [8, 8])
    if include_extra_layer:
        add("model.layers.2.self_attn.indexer.wk.weight", "F32", [2, 8])
        add("model.layers.2.mlp.experts.0.gate_proj.weight", "BF16", [8, 8])
    tensors.update(extra_tensors)

    payload = bytearray()
    header: dict[str, dict[str, object]] = {}
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard = root / "model-00001-of-00001.safetensors"
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
    write_simple_tokenizer(root / "simple_tokenizer.json")


def _replace_safetensors_shard_with_header_manifest(root: Path) -> None:
    shard = root / "model-00001-of-00001.safetensors"
    raw = shard.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_len
    header = json.loads(raw[8:data_start])
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    (root / HEADER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "index": index,
                "shards": {
                    shard.name: {
                        "file_size": len(raw),
                        "data_start": data_start,
                        "header": header,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    shard.unlink()


def _mlx_affine_int4_expert_fixture() -> tuple[
    set[str],
    dict[str, tuple[str, list[int], bytes]],
]:
    omit: set[str] = set()
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert in (0, 1):
        for component in ("gate_proj", "up_proj", "down_proj"):
            stem = f"model.layers.1.mlp.experts.{expert}.{component}"
            omit.add(f"{stem}.weight")
            tensors[f"{stem}.weight"] = ("U32", [8, 1], b"\0" * (8 * 4))
            tensors[f"{stem}.scales"] = ("BF16", [8, 1], b"\0" * (8 * 2))
            tensors[f"{stem}.biases"] = ("BF16", [8, 1], b"\0" * (8 * 2))
    return omit, tensors


def _mlx_mxfp4_expert_fixture() -> tuple[
    set[str],
    dict[str, tuple[str, list[int], bytes]],
]:
    omit: set[str] = set()
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert in (0, 1):
        for component in ("gate_proj", "up_proj", "down_proj"):
            stem = f"model.layers.1.mlp.experts.{expert}.{component}"
            source = f"{stem}.weight"
            omit.add(source)
            out_dim = 32
            tensors[source] = ("U32", [out_dim, 4], b"\0" * (out_dim * 4 * 4))
            tensors[f"{stem}.scales"] = ("U8", [out_dim, 1], b"\0" * out_dim)
    return omit, tensors


def test_preflight_glm_checkpoint_accepts_complete_raw_checkpoint(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.attention_layers_ok == 2
    assert report.tensor_coverage.router_layers_ok == 1
    assert report.tensor_coverage.router_bias_layers_checked == 1
    assert report.tensor_coverage.router_bias_layers_found == 0
    assert report.tensor_coverage.dense_layers_ok == 1
    assert report.expert_coverage.checked is True
    assert report.expert_coverage.quantization_mode == "largerlm-affine-int4"
    assert report.resident_coverage.checked is True
    assert report.tokenizer.found is True
    assert report.effective_unified_memory_bytes == 128 * 1024**3
    assert report.effective_unified_memory_source == "explicit"
    assert report.effective_system_reserve_bytes == 24 * 1024**3
    assert isinstance(report.hardware_chip_name, str)
    assert report.hardware_chip_name
    assert report.recommended_max_live_working_set_bytes == (
        8 * 1024**3 + report.resident_coverage.packed_bytes_estimate
    )
    assert report.recommended_min_free_unified_memory_bytes == 24 * 1024**3
    assert report.public_glm_5_2_shape["matches"] is False
    assert "hidden_size" in report.public_glm_5_2_shape["mismatched_fields"]
    assert not [issue for issue in report.issues if issue.severity == "error"]


def test_preflight_glm_checkpoint_accepts_header_manifest_without_shards(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    _replace_safetensors_shard_with_header_manifest(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.checkpoint_tensor_count > 0
    assert report.checkpoint_total_bytes > 0
    assert report.tensor_coverage.dense_layers_ok == 1
    assert report.expert_coverage.checked is True
    assert report.resident_coverage.checked is True
    assert not [issue for issue in report.issues if issue.severity == "error"]


def test_preflight_metadata_only_ignores_partial_local_shards(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    _replace_safetensors_shard_with_header_manifest(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"partial")

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
        prefer_header_manifest=True,
    )

    assert report.ok is True
    assert report.checkpoint_tensor_count > 0
    assert report.expert_coverage.checked is True
    assert report.resident_coverage.checked is True
    assert not [issue for issue in report.issues if issue.severity == "error"]


def test_preflight_reports_mlx_quantization_metadata(tmp_path: Path) -> None:
    omit, extra_tensors = _mlx_affine_int4_expert_fixture()
    _write_checkpoint(
        tmp_path,
        omit=omit,
        extra_tensors=extra_tensors,
        config_overrides={
            "quantization": {
                "bits": 4,
                "group_size": 8,
            },
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.expert_coverage.quantization_mode == "mlx-affine-int4"
    assert report.quantization.detected is True
    assert report.quantization.source == "quantization"
    assert report.quantization.bits == 4
    assert report.quantization.group_size == 8
    assert report.quantization.bits_match_target is True
    assert report.quantization.group_size_match_target is True
    assert report.quantization.compatible_with_mlx_affine_int4 is True


def test_preflight_reports_mxfp4_scales_only_expert_layout(
    tmp_path: Path,
) -> None:
    omit, extra_tensors = _mlx_mxfp4_expert_fixture()
    _write_checkpoint(
        tmp_path,
        omit=omit,
        extra_tensors=extra_tensors,
        config_overrides={
            "hidden_size": 32,
            "intermediate_size": 32,
            "moe_intermediate_size": 32,
        },
        shape_overrides={
            "model.layers.1.mlp.gate.weight": [2, 32],
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": [32, 32],
            "model.layers.0.mlp.switch_mlp.up_proj.weight": [32, 32],
            "model.layers.0.mlp.switch_mlp.down_proj.weight": [32, 32],
            "model.layers.0.self_attn.q_a_proj.weight": [2, 32],
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": [4, 32],
            "model.layers.0.self_attn.o_proj.weight": [32, 2],
            "model.layers.1.self_attn.q_a_proj.weight": [2, 32],
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight": [4, 32],
            "model.layers.1.self_attn.o_proj.weight": [32, 2],
            "model.layers.0.input_layernorm.weight": [32],
            "model.layers.0.post_attention_layernorm.weight": [32],
            "model.layers.1.input_layernorm.weight": [32],
            "model.layers.1.post_attention_layernorm.weight": [32],
            "model.embed_tokens.weight": [4, 32],
            "model.norm.weight": [32],
            "lm_head.weight": [4, 32],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.expert_coverage.checked is True
    assert report.expert_coverage.quantization_mode == "mlx-mxfp4"
    assert not any(
        issue.code == "mxfp4_expert_runtime_unsupported" for issue in report.issues
    )


def test_preflight_rejects_mlx_quantization_group_mismatch(tmp_path: Path) -> None:
    omit, extra_tensors = _mlx_affine_int4_expert_fixture()
    _write_checkpoint(
        tmp_path,
        omit=omit,
        extra_tensors=extra_tensors,
        config_overrides={
            "quantization": {
                "bits": 4,
                "group_size": 64,
            },
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.quantization.group_size_match_target is False
    assert any(
        issue.code == "config_quantization_group_size_mismatch"
        for issue in report.issues
    )


def test_preflight_warns_when_raw_conversion_ignores_quantization_metadata(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        config_overrides={
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 8,
                "group_size": 16,
            },
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.quantization.source == "quantization_config"
    assert report.quantization.method == "gptq"
    assert report.quantization.bits_match_target is False
    assert report.quantization.group_size_match_target is False
    assert any(
        issue.code == "config_quantization_ignored_for_raw_conversion"
        for issue in report.issues
    )


def test_preflight_explains_redundant_raw_conversion_for_mlx_affine_int4(
    tmp_path: Path,
) -> None:
    omit, extra_tensors = _mlx_affine_int4_expert_fixture()
    _write_checkpoint(
        tmp_path,
        omit=omit,
        extra_tensors=extra_tensors,
        config_overrides={
            "quantization": {
                "bits": 4,
                "group_size": 8,
            },
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert any(
        issue.code == "config_quantization_raw_conversion_redundant"
        and issue.severity == "warning"
        for issue in report.issues
    )
    assert any(
        issue.code == "expert_coverage_failed"
        and "remove --quantize-bf16-affine-int4" in issue.message
        for issue in report.issues
    )


def test_preflight_accepts_resident_affine_int4_projection_layout(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"model.layers.0.self_attn.q_a_proj.weight"},
        extra_tensors={
            "model.layers.0.self_attn.q_a_proj.weight": (
                "U32",
                [2, 1],
                b"\0" * (2 * 4),
            ),
            "model.layers.0.self_attn.q_a_proj.scales": (
                "BF16",
                [2, 1],
                b"\0" * (2 * 2),
            ),
            "model.layers.0.self_attn.q_a_proj.biases": (
                "BF16",
                [2, 1],
                b"\0" * (2 * 2),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.attention_layers_ok == 2
    assert not [
        issue
        for issue in report.issues
        if issue.code == "resident_affine_int4_layout_invalid"
    ]


def test_preflight_accepts_resident_affine_int4_router_layout(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"model.layers.1.mlp.gate.weight"},
        extra_tensors={
            "model.layers.1.mlp.gate.weight": (
                "U32",
                [2, 1],
                b"\0" * (2 * 4),
            ),
            "model.layers.1.mlp.gate.scales": (
                "BF16",
                [2, 1],
                b"\0" * (2 * 2),
            ),
            "model.layers.1.mlp.gate.biases": (
                "BF16",
                [2, 1],
                b"\0" * (2 * 2),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.router_layers_ok == 1
    assert not [
        issue
        for issue in report.issues
        if issue.code == "resident_affine_int4_layout_invalid"
    ]
    assert not [
        issue for issue in report.issues if issue.code == "router_shape_mismatch"
    ]


def test_preflight_accepts_resident_affine_int4_lm_head_layout(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"lm_head.weight"},
        vocab_size=4,
        extra_tensors={
            "lm_head.weight": (
                "U32",
                [4, 1],
                b"\0" * (4 * 4),
            ),
            "lm_head.scales": (
                "BF16",
                [4, 1],
                b"\0" * (4 * 2),
            ),
            "lm_head.biases": (
                "BF16",
                [4, 1],
                b"\0" * (4 * 2),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.lm_head is True
    assert not [
        issue
        for issue in report.issues
        if issue.code == "resident_affine_int4_layout_invalid"
    ]
    assert not [
        issue for issue in report.issues if issue.code == "global_shape_mismatch"
    ]


def test_preflight_accepts_mxfp4_global_logical_shapes(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"model.embed_tokens.weight", "lm_head.weight"},
        vocab_size=4,
        extra_tensors={
            "model.embed_tokens.weight": (
                "U32",
                [4, 1],
                b"\0" * (4 * 4),
            ),
            "model.embed_tokens.scales": (
                "U8",
                [4, 1],
                b"\0" * 4,
            ),
            "lm_head.weight": (
                "U32",
                [4, 1],
                b"\0" * (4 * 4),
            ),
            "lm_head.scales": (
                "U8",
                [4, 1],
                b"\0" * 4,
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.embedding is True
    assert report.tensor_coverage.lm_head is True
    assert not any(
        issue.code == "mxfp4_resident_runtime_unsupported"
        for issue in report.issues
    )
    assert not [
        issue
        for issue in report.issues
        if issue.code == "resident_affine_int4_layout_invalid"
    ]
    assert not [
        issue for issue in report.issues if issue.code == "global_shape_mismatch"
    ]


def test_preflight_reports_mxfp4_non_2d_resident_runtime_gap(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        extra_tensors={
            "model.layers.0.self_attn.unsupported_alias.weight": (
                "U32",
                [2, 4, 1],
                b"\0" * (2 * 4 * 4),
            ),
            "model.layers.0.self_attn.unsupported_alias.scales": (
                "U8",
                [2, 4, 1],
                b"\0" * (2 * 4),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(
        item for item in report.issues if item.code == "mxfp4_resident_runtime_unsupported"
    )
    assert "model.layers.0.self_attn.unsupported_alias.weight" in issue.message


def test_preflight_skips_mxfp4_switch_mlp_routed_tensors_for_resident_gap() -> None:
    tensors = [
        TensorMeta(
            name="model.layers.3.mlp.switch_mlp.gate_proj.weight",
            shard="model.safetensors",
            dtype="U32",
            shape=(256, 2048, 1),
            data_offsets=(0, 256 * 2048 * 4),
            data_start=8,
        ),
        TensorMeta(
            name="model.layers.3.mlp.switch_mlp.gate_proj.scales",
            shard="model.safetensors",
            dtype="U8",
            shape=(256, 2048, 1),
            data_offsets=(256 * 2048 * 4, 256 * 2048 * 5),
            data_start=8,
        ),
    ]

    assert preflight_module._resident_mxfp4_runtime_unsupported_names(tensors) == ()


def test_preflight_accepts_absorbed_kv_b_attention_aliases(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"model.layers.0.self_attn.kv_b_proj.weight"},
        extra_tensors={
            "model.layers.0.self_attn.embed_q.weight": (
                "F32",
                [2, 2, 1],
                b"\0" * (2 * 2 * 1 * 4),
            ),
            "model.layers.0.self_attn.unembed_out.weight": (
                "F32",
                [2, 1, 2],
                b"\0" * (2 * 1 * 2 * 4),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.attention_layers_ok == 2
    assert not [
        issue
        for issue in report.issues
        if issue.code in {"missing_resident_tensors", "attention_shape_mismatch"}
    ]


def test_preflight_reports_unsupported_expert_quantized_tensor_layout(
    tmp_path: Path,
) -> None:
    omit = {
        f"model.layers.1.mlp.experts.{expert}.{component}.weight"
        for expert in (0, 1)
        for component in ("gate_proj", "up_proj", "down_proj")
    }
    extra_tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert in (0, 1):
        stem = f"model.layers.1.mlp.experts.{expert}.gate_proj"
        extra_tensors[f"{stem}.qweight"] = ("U32", [8, 1], b"\0" * (8 * 4))
        extra_tensors[f"{stem}.qzeros"] = ("U32", [8, 1], b"\0" * (8 * 4))
        extra_tensors[f"{stem}.g_idx"] = ("I32", [1], b"\0" * 4)
    _write_checkpoint(tmp_path, omit=omit, extra_tensors=extra_tensors)

    report = preflight_glm_checkpoint(
        tmp_path,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.quantization.detected is False
    issue = next(
        item
        for item in report.issues
        if item.code == "unsupported_expert_quantized_tensor_layout"
    )
    assert "qweight" in issue.message
    assert "MLX affine-int4" in issue.message


def test_preflight_accepts_fused_dense_gate_up_resident_tensor(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.0.mlp.switch_mlp.up_proj.weight",
        },
        extra_tensors={
            "model.layers.0.mlp.switch_mlp.gate_up_proj.weight": (
                "F32",
                [16, 8],
                b"\0" * (16 * 8 * 4),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.dense_layers_ok == 1
    assert not [
        issue
        for issue in report.issues
        if issue.severity == "error" and "dense MLP" in issue.message
    ]


def test_preflight_accepts_dense_w_component_resident_aliases(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        omit={
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.layers.0.mlp.switch_mlp.up_proj.weight",
            "model.layers.0.mlp.switch_mlp.down_proj.weight",
        },
        extra_tensors={
            "model.layers.0.mlp.switch_mlp.w1.weight": (
                "F32",
                [8, 8],
                b"\0" * (8 * 8 * 4),
            ),
            "model.layers.0.mlp.switch_mlp.w3.weight": (
                "F32",
                [8, 8],
                b"\0" * (8 * 8 * 4),
            ),
            "model.layers.0.mlp.switch_mlp.w2.weight": (
                "F32",
                [8, 8],
                b"\0" * (8 * 8 * 4),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.dense_layers_ok == 1
    assert not [
        issue
        for issue in report.issues
        if issue.severity == "error" and "dense MLP" in issue.message
    ]


def test_preflight_accepts_shared_w_component_resident_aliases(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        include_shared_expert=True,
        omit={
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            "model.layers.1.mlp.shared_experts.down_proj.weight",
        },
        extra_tensors={
            "model.layers.1.mlp.shared_experts.w1.weight": (
                "F32",
                [8, 8],
                b"\0" * (8 * 8 * 4),
            ),
            "model.layers.1.mlp.shared_experts.w3.weight": (
                "F32",
                [8, 8],
                b"\0" * (8 * 8 * 4),
            ),
            "model.layers.1.mlp.shared_experts.w2.weight": (
                "F32",
                [8, 8],
                b"\0" * (8 * 8 * 4),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert not [
        issue
        for issue in report.issues
        if issue.severity == "error" and "shared expert" in issue.message
    ]


def test_preflight_accepts_resident_affine_int4_shared_expert_layout(
    tmp_path: Path,
) -> None:
    omit = {
        f"model.layers.1.mlp.shared_experts.{component}.weight"
        for component in ("gate_proj", "up_proj", "down_proj")
    }
    extra_tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for component in ("gate_proj", "up_proj", "down_proj"):
        stem = f"model.layers.1.mlp.shared_experts.{component}"
        extra_tensors[f"{stem}.weight"] = ("U32", [8, 1], b"\0" * (8 * 4))
        extra_tensors[f"{stem}.scales"] = ("BF16", [8, 1], b"\0" * (8 * 2))
        extra_tensors[f"{stem}.biases"] = ("BF16", [8, 1], b"\0" * (8 * 2))
    _write_checkpoint(
        tmp_path,
        include_shared_expert=True,
        omit=omit,
        extra_tensors=extra_tensors,
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.shared_layers_ok == 1
    assert not [
        issue
        for issue in report.issues
        if issue.code == "resident_affine_int4_layout_invalid"
    ]


def test_preflight_records_structured_apple_silicon_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checkpoint(tmp_path)
    monkeypatch.setattr(
        preflight_module,
        "detect_hardware",
        lambda: HardwareInfo(
            chip_name="Apple M5 Max",
            unified_memory_bytes=128 * 1024**3,
            gpu_cores=40,
            apple_silicon_generation=5,
            apple_silicon_tier="Max",
        ),
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
    )

    assert report.hardware_chip_name == "Apple M5 Max"
    assert report.hardware_apple_silicon_generation == 5
    assert report.hardware_apple_silicon_tier == "Max"
    assert report.effective_unified_memory_source == "detected"


def test_preflight_require_public_glm_5_2_shape_reports_issue(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(
        item for item in report.issues if item.code == "public_glm_5_2_shape_mismatch"
    )
    assert issue.severity == "error"
    assert "hidden_size" in issue.message
    assert "num_hidden_layers" in issue.message


def test_preflight_require_public_glm_5_2_shape_stops_before_checkpoint_scan(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "num_attention_heads": 2,
                "q_lora_rank": 2,
                "kv_lora_rank": 2,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
            }
        ),
        encoding="utf-8",
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.checkpoint_tensor_count == 0
    assert report.tensor_coverage.attention_layers_checked == 0
    assert report.expert_coverage.checked is False
    assert report.resident_coverage.checked is False
    assert report.plan is None
    assert report.tokenizer.error == (
        "not checked because the public GLM-5.2 gate failed"
    )
    issue = next(
        item for item in report.issues if item.code == "public_glm_5_2_shape_mismatch"
    )
    assert "hidden_size" in issue.message


def test_preflight_require_public_glm_5_2_shape_requires_4bit_before_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
                "num_attention_heads": 2,
                "q_lora_rank": 2,
                "kv_lora_rank": 2,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preflight_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quant_bits=8,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.checkpoint_tensor_count == 0
    assert report.expert_coverage.checked is False
    issue = next(
        item for item in report.issues if item.code == "public_glm_5_2_requires_4bit"
    )
    assert issue.severity == "error"
    assert report.tokenizer.error == (
        "not checked because the public GLM-5.2 gate failed"
    )


def test_preflight_rejects_non_boolean_public_shape_gate(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=1,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert any(
        item.code == "invalid_require_public_glm_5_2_shape"
        for item in report.issues
    )


def test_preflight_warns_when_lm_head_uses_tied_embeddings(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, omit={"lm_head.weight"}, tie_word_embeddings=True)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.lm_head is False
    assert report.tensor_coverage.tied_lm_head is True
    issue = next(item for item in report.issues if item.code == "tied_lm_head")
    assert issue.severity == "warning"


def test_preflight_rejects_missing_lm_head_when_embeddings_are_not_tied(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path, omit={"lm_head.weight"}, tie_word_embeddings=False)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.tensor_coverage.lm_head is False
    assert report.tensor_coverage.tied_lm_head is False
    assert "lm_head.weight" in report.tensor_coverage.missing_global
    issue = next(item for item in report.issues if item.code == "missing_lm_head")
    assert issue.severity == "error"
    assert "tie_word_embeddings=false" in issue.message


def test_preflight_explicit_system_reserve_overrides_large_memory_default(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
        system_reserve_bytes=16 * 1024**3,
    )

    assert report.ok is True
    assert report.recommended_min_free_unified_memory_bytes == 16 * 1024**3


def test_preflight_warns_when_resident_budget_exceeds_memory(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=1 * 1024**3,
    )

    assert report.ok is True
    assert report.plan is not None
    assert report.plan.resident_memory_fits_budget is False
    issue = next(
        item for item in report.issues if item.code == "resident_memory_budget_exceeded"
    )
    assert issue.severity == "warning"
    assert "exceed unified memory" in issue.message


def test_preflight_reports_invalid_planning_budget(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
        runtime_buffer_bytes=0,
    )

    assert report.ok is False
    issue = next(
        item for item in report.issues if item.code == "invalid_runtime_buffer_bytes"
    )
    assert issue.severity == "error"
    assert issue.message == "runtime_buffer_bytes must be positive"


def test_preflight_reports_invalid_disk_safety_margin(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        output_dir=tmp_path / "packed",
        disk_safety_margin_bytes=-1,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.disk_budget is None
    issue = next(item for item in report.issues if item.code == "invalid_disk_safety_margin")
    assert issue.severity == "error"


@pytest.mark.parametrize(
    ("kwargs", "code", "message"),
    (
        ({"quant_bits": True}, "invalid_quant_bits", "quant_bits must be an integer"),
        ({"group_size": 8.5}, "invalid_group_size", "group_size must be an integer"),
        (
            {"max_context_tokens": True},
            "invalid_max_context_tokens",
            "max_context_tokens must be an integer",
        ),
        (
            {"max_cache_bytes": 1.5},
            "invalid_max_cache_bytes",
            "max_cache_bytes must be an integer",
        ),
        (
            {"disk_safety_margin_bytes": False},
            "invalid_disk_safety_margin",
            "disk_safety_margin_bytes must be an integer",
        ),
        (
            {"unified_memory_bytes": True},
            "invalid_unified_memory_bytes",
            "unified_memory_bytes must be an integer",
        ),
        (
            {"system_reserve_bytes": 1.5},
            "invalid_system_reserve_bytes",
            "system_reserve_bytes must be an integer",
        ),
        (
            {"runtime_buffer_bytes": False},
            "invalid_runtime_buffer_bytes",
            "runtime_buffer_bytes must be an integer",
        ),
        (
            {"page_cache_fraction": True},
            "invalid_page_cache_fraction",
            "page_cache_fraction must be numeric",
        ),
        (
            {"cold_read_gib_per_second": False},
            "invalid_cold_read_gib_per_second",
            "cold_read_gib_per_second must be numeric",
        ),
    ),
)
def test_preflight_rejects_non_integer_budget_inputs(
    tmp_path: Path,
    kwargs: dict[str, object],
    code: str,
    message: str,
) -> None:
    _write_checkpoint(tmp_path)

    params: dict[str, object] = {
        "quantize_raw_to_int4": True,
        "group_size": 8,
        "max_context_tokens": 4,
        "max_cache_bytes": 1024 * 1024,
        "disk_safety_margin_bytes": 0,
        "unified_memory_bytes": 128 * 1024**3,
    }
    params.update(kwargs)
    report = preflight_glm_checkpoint(tmp_path, **params)

    assert report.ok is False
    issue = next(item for item in report.issues if item.code == code)
    assert issue.severity == "error"
    assert issue.message == message


def test_preflight_rejects_context_above_model_max_position(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_position_embeddings"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(item for item in report.issues if item.code == "context_exceeds_model_max")
    assert issue.severity == "error"
    assert "max_position_embeddings 4" in issue.message


def test_preflight_reports_ignored_extra_layer_bytes(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, include_extra_layer=True)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.checkpoint_ignored_bytes == 192


def test_plan_scan_safetensors_uses_glm_layer_boundary_for_extra_layers(
    tmp_path: Path,
    capsys,
) -> None:
    _write_checkpoint(tmp_path, include_extra_layer=True)

    status = cli_main(["plan", str(tmp_path), "--scan-safetensors", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    stats = payload["checkpoint_stats"]
    assert stats["by_category"]["ignored_extra_layers"] == 192
    assert stats["routed_expert_bytes"] == 768
    assert stats["resident_bytes"] == stats["total_bytes"] - 768 - 192
    assert payload["resident_bytes_estimate"] == stats["resident_bytes"]
    assert payload["resident_bytes_estimate_source"] == "checkpoint_scan"
    suggested_decode = payload["suggested_decode_guard_flags"]
    assert suggested_decode["source"] == "plan"
    assert suggested_decode["decode_read_bytes_per_token"] == 192
    assert "--decode-max-routed-read-gib-per-token" in suggested_decode["argv"]


def test_plan_cli_writes_decode_launch_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_checkpoint(tmp_path)
    profile_path = tmp_path / "plan-launch-profile.json"

    status = cli_main(
        [
            "plan",
            str(tmp_path),
            "--cold-read-gib-s",
            "16",
            "--write-launch-profile",
            str(profile_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile == payload["suggested_launch_profile"]
    assert profile["source"] == "plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["decode_guard_flags"] == payload[
        "suggested_decode_guard_flags"
    ]
    assert "--decode-max-routed-read-gib-per-token" in profile["argv"]
    assert "--prefill-ssd-read-gib-s" in profile["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in profile["argv"]


def test_plan_cli_rejects_non_finite_scaled_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_checkpoint(tmp_path)

    status = cli_main(
        [
            "plan",
            str(tmp_path),
            "--unified-memory-gib",
            "nan",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "--unified-memory-gib must be finite" in captured.err


def test_preflight_reports_router_correction_bias_coverage(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        include_router_bias=True,
        topk_method="noaux_tc",
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.router_bias_layers_found == 1
    assert not [
        issue
        for issue in report.issues
        if issue.code == "router_correction_bias_incomplete"
    ]


def test_preflight_counts_block_sparse_router_correction_bias(
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path,
        topk_method="noaux_tc",
        extra_tensors={
            "model.layers.1.block_sparse_moe.gate.e_score_correction_bias": (
                "F32",
                [2],
                b"\0" * (2 * 4),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.router_bias_layers_found == 1
    assert not [
        issue
        for issue in report.issues
        if issue.code == "router_correction_bias_incomplete"
    ]


def test_preflight_warns_when_noaux_router_bias_is_missing(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        topk_method=" noaux_tc ",
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    issue = next(
        issue
        for issue in report.issues
        if issue.code == "router_correction_bias_incomplete"
    )
    assert issue.severity == "warning"
    assert "found 0/1" in issue.message


def test_preflight_reports_missing_resident_tensor(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"model.layers.0.self_attn.o_proj.weight"},
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert any(issue.code == "missing_resident_tensors" for issue in report.issues)


def test_preflight_reports_attention_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        shape_overrides={
            "model.layers.0.self_attn.kv_b_proj.weight": [3, 2],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(
        issue for issue in report.issues if issue.code == "attention_shape_mismatch"
    )
    assert ".self_attn.kv_b_proj.weight" in issue.message
    assert "[3, 2] != [4, 2]" in issue.message


def test_preflight_reports_global_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        shape_overrides={
            "model.embed_tokens.weight": [4, 7],
            "model.norm.weight": [7],
            "lm_head.weight": [4, 7],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "global_shape_mismatch")
    assert "embed_tokens.weight hidden dim 7 != 8" in issue.message
    assert "norm.weight shape [7] != [8]" in issue.message
    assert "lm_head.weight hidden dim 7 != 8" in issue.message


def test_preflight_reports_global_vocab_size_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, vocab_size=5)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "global_shape_mismatch")
    assert "embed_tokens.weight vocab rows 4 != 5" in issue.message
    assert "lm_head.weight vocab rows 4 != 5" in issue.message


def test_preflight_reports_missing_dense_mlp_tensor(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        omit={"model.layers.0.mlp.switch_mlp.up_proj.weight"},
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.tensor_coverage.dense_layers_ok == 0
    assert any("dense MLP up_proj" in item for item in report.tensor_coverage.missing_by_layer)


def test_preflight_reports_dense_mlp_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        shape_overrides={
            "model.layers.0.mlp.switch_mlp.down_proj.weight": [7, 8],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "mlp_shape_mismatch")
    assert "dense MLP down_proj" in issue.message
    assert "[7, 8] != [8, 8]" in issue.message


def test_preflight_reports_shared_expert_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        include_shared_expert=True,
        shape_overrides={
            "model.layers.1.mlp.shared_experts.gate_proj.weight": [7, 8],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "mlp_shape_mismatch")
    assert "shared expert gate_proj" in issue.message
    assert "[7, 8] != [8, 8]" in issue.message


def test_preflight_reports_raw_routed_expert_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        shape_overrides={
            "model.layers.1.mlp.experts.0.down_proj.weight": [4, 16],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "expert_coverage_failed")
    assert "down_proj.weight shape" in issue.message
    assert "does not match config expected" in issue.message


def test_preflight_reports_out_of_range_routed_expert_id(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        extra_tensors={
            "model.layers.1.mlp.experts.2.gate_proj.weight": (
                "BF16",
                [8, 8],
                _zeros("BF16", [8, 8]),
            ),
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "expert_coverage_failed")
    assert "routed expert id 2" in issue.message
    assert "outside configured range [0, 2)" in issue.message


def test_preflight_reports_router_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        include_router_bias=True,
        shape_overrides={
            "model.layers.1.mlp.gate.weight": [3, 8],
            "model.layers.1.mlp.gate.e_score_correction_bias": [3],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(issue for issue in report.issues if issue.code == "router_shape_mismatch")
    assert "router gate.weight" in issue.message
    assert "router correction bias" in issue.message


def test_preflight_checks_full_dsa_indexer_tensors(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path, include_indexer=True)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.tensor_coverage.indexer_layers_checked == 1
    assert report.tensor_coverage.indexer_layers_ok == 1
    assert report.dsa_index_head_dim == 2
    assert report.dsa_index_n_heads == 1
    assert report.dsa_index_topk == 2
    assert report.dsa_full_indexer_q_output_dim == 2


def test_preflight_reports_missing_dsa_indexer_tensor(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        include_indexer=True,
        omit={"model.layers.0.self_attn.indexer.weights_proj.weight"},
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.tensor_coverage.indexer_layers_checked == 1
    assert report.tensor_coverage.indexer_layers_ok == 0
    assert any(
        "indexer .self_attn.indexer.weights_proj.weight" in item
        for item in report.tensor_coverage.missing_by_layer
    )


def test_preflight_reports_dsa_indexer_shape_mismatch(tmp_path: Path) -> None:
    _write_checkpoint(
        tmp_path,
        include_indexer=True,
        shape_overrides={
            "model.layers.0.self_attn.indexer.wq_b.weight": [3, 2],
        },
    )

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=4,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    issue = next(
        issue for issue in report.issues if issue.code == "dsa_indexer_shape_mismatch"
    )
    assert ".self_attn.indexer.wq_b.weight" in issue.message
    assert "[3, 2] != [2, 2]" in issue.message


def test_preflight_decode_cache_issue_reports_safe_context(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    report = preflight_glm_checkpoint(
        tmp_path,
        quantize_raw_to_int4=True,
        group_size=8,
        max_context_tokens=1024,
        max_cache_bytes=1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    issue = next(
        issue for issue in report.issues if issue.code == "decode_cache_budget_exceeded"
    )
    assert report.ok is False
    assert "safe context" in issue.message
    assert "64 tokens" in issue.message


def test_preflight_glm_cli_returns_success_for_complete_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_checkpoint(tmp_path)
    report_path = tmp_path / "reports" / "preflight-report.json"

    status = cli_main(
        [
            "preflight-glm",
            str(tmp_path),
            "--quantize-bf16-affine-int4",
            "--group-size",
            "8",
            "--max-context-tokens",
            "4",
            "--max-cache-gib",
            "1",
            "--output-dir",
            str(tmp_path / "packed"),
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--write-report",
            str(report_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload == payload
    assert not report_path.with_name(f"{report_path.name}.tmp").exists()
    assert payload["effective_unified_memory_bytes"] == 128 * 1024**3
    assert payload["effective_unified_memory_source"] == "explicit"
    assert payload["effective_system_reserve_bytes"] == 24 * 1024**3
    assert isinstance(payload["hardware_chip_name"], str)
    assert "hardware_unified_memory_bytes" in payload
    assert "hardware_gpu_cores" in payload
    assert "hardware_apple_silicon_generation" in payload
    assert "hardware_apple_silicon_tier" in payload
    assert payload["public_glm_5_2_shape"]["matches"] is False
    assert "hidden_size" in payload["public_glm_5_2_shape"]["mismatched_fields"]


def test_preflight_glm_cli_metadata_only_ignores_partial_shard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_checkpoint(tmp_path)
    _replace_safetensors_shard_with_header_manifest(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"partial")

    status = cli_main(
        [
            "preflight-glm",
            str(tmp_path),
            "--metadata-only",
            "--quantize-bf16-affine-int4",
            "--group-size",
            "8",
            "--max-context-tokens",
            "4",
            "--max-cache-gib",
            "1",
            "--output-dir",
            str(tmp_path / "packed"),
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_tensor_count"] > 0
    assert payload["expert_coverage"]["checked"] is True
    assert payload["resident_coverage"]["checked"] is True


def test_preflight_glm_cli_require_public_glm_5_2_shape_returns_not_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_checkpoint(tmp_path)

    status = cli_main(
        [
            "preflight-glm",
            str(tmp_path),
            "--quantize-bf16-affine-int4",
            "--require-public-glm-5-2-shape",
            "--group-size",
            "8",
            "--max-context-tokens",
            "4",
            "--max-cache-gib",
            "1",
            "--output-dir",
            str(tmp_path / "packed"),
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--json",
        ]
    )

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["public_glm_5_2_shape"]["matches"] is False
    issue = next(
        item for item in payload["issues"]
        if item["code"] == "public_glm_5_2_shape_mismatch"
    )
    assert "hidden_size" in issue["message"]


def test_preflight_glm_cli_require_public_glm_5_2_shape_rejects_non_4bit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_checkpoint(tmp_path)
    monkeypatch.setattr(
        preflight_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )

    status = cli_main(
        [
            "preflight-glm",
            str(tmp_path),
            "--quantize-bf16-affine-int4",
            "--quant-bits",
            "8",
            "--require-public-glm-5-2-shape",
            "--group-size",
            "8",
            "--max-context-tokens",
            "4",
            "--max-cache-gib",
            "1",
            "--output-dir",
            str(tmp_path / "packed"),
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--json",
        ]
    )

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_tensor_count"] == 0
    assert payload["expert_coverage"]["checked"] is False
    issue = next(
        item for item in payload["issues"]
        if item["code"] == "public_glm_5_2_requires_4bit"
    )
    assert issue["severity"] == "error"
    assert payload["tokenizer"]["error"] == (
        "not checked because the public GLM-5.2 gate failed"
    )
