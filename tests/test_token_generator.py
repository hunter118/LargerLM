from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from largerlm import generation_guard as generation_guard_module
from largerlm.cli import main as cli_main
from largerlm.generation_guard import (
    GenerationGuardError,
    SystemMemorySnapshot,
    check_generation_runtime,
    check_live_memory_budget,
    estimate_prompt_prefill_live_memory,
)
from largerlm.layout import config_sha256
from largerlm.safety import DiskBudget
from largerlm.token_generator import (
    TokenGenerationResult,
    TokenGeneratorError,
    _auto_prefill_prompt_chunk_plan,
    _auto_prefill_prompt_chunk_tokens,
    _prefill_actual_read_time_summary,
    generate_token_ids,
)


COMPONENTS = [
    ("gate_proj.weight", 0, 32, "U32", [8, 1]),
    ("gate_proj.scales", 32, 16, "BF16", [8, 1]),
    ("gate_proj.biases", 48, 16, "BF16", [8, 1]),
    ("up_proj.weight", 64, 32, "U32", [8, 1]),
    ("up_proj.scales", 96, 16, "BF16", [8, 1]),
    ("up_proj.biases", 112, 16, "BF16", [8, 1]),
    ("down_proj.weight", 128, 32, "U32", [8, 1]),
    ("down_proj.scales", 160, 16, "BF16", [8, 1]),
    ("down_proj.biases", 176, 16, "BF16", [8, 1]),
]


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(value) for value in values)


def pack8(values: list[int]) -> int:
    out = 0
    for index, value in enumerate(values):
        out |= (value & 0xF) << (index * 4)
    return out


def test_live_memory_budget_rejects_low_system_headroom() -> None:
    snapshot = SystemMemorySnapshot(
        total_bytes=128 * 1024**3,
        available_bytes=10 * 1024**2,
        page_size=16 * 1024,
        source="test",
    )

    with pytest.raises(GenerationGuardError, match="available unified memory") as exc_info:
        check_live_memory_budget(
            estimated_live_working_set_bytes=8 * 1024**2,
            max_live_working_set_bytes=64 * 1024**2,
            min_available_memory_bytes=8 * 1024**2,
            snapshot=snapshot,
        )
    assert exc_info.value.payload == {
        "code": "available_unified_memory_below_required",
        "estimated_live_working_set_bytes": 8 * 1024**2,
        "max_live_working_set_bytes": 64 * 1024**2,
        "min_available_memory_bytes": 8 * 1024**2,
        "required_available_memory_bytes": 16 * 1024**2,
        "system_available_memory_bytes": 10 * 1024**2,
        "system_total_memory_bytes": 128 * 1024**3,
        "system_memory_source": "test",
        "available_memory_ok": False,
        "resident_backing_bytes": 0,
        "nonresident_peak_bytes": None,
        "extra_live_working_set_bytes": 0,
    }


def test_system_memory_snapshot_darwin_uses_native_total_when_sysctl_process_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_text(args: list[str]) -> str | None:
        if args == ["/usr/bin/vm_stat"]:
            return (
                "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
                "Pages free: 2.\n"
                "Pages inactive: 3.\n"
                "Pages speculative: 5.\n"
            )
        return None

    monkeypatch.setattr(generation_guard_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(generation_guard_module, "_run_text", fake_run_text)
    monkeypatch.setattr(
        generation_guard_module,
        "_darwin_hw_memsize_sysctlbyname",
        lambda: 128 * 1024**3,
    )

    snapshot = generation_guard_module.system_memory_snapshot()

    assert snapshot is not None
    assert snapshot.total_bytes == 128 * 1024**3
    assert snapshot.available_bytes == 10 * 4096
    assert snapshot.source == "vm_stat"


def test_system_memory_snapshot_darwin_falls_back_to_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sysconf(name: str) -> int:
        values = {
            "SC_PHYS_PAGES": 4,
            "SC_PAGE_SIZE": 8192,
        }
        return values[name]

    monkeypatch.setattr(
        generation_guard_module,
        "_darwin_hw_memsize_sysctlbyname",
        lambda: None,
    )
    monkeypatch.setattr(generation_guard_module.os, "sysconf", fake_sysconf)
    monkeypatch.setattr(generation_guard_module, "_run_text", lambda args: None)

    assert generation_guard_module._darwin_hw_memsize_bytes() == 32768


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {
                "estimated_live_working_set_bytes": True,
                "max_live_working_set_bytes": 64,
                "min_available_memory_bytes": 0,
            },
            "estimated live working set must be an integer",
        ),
        (
            {
                "estimated_live_working_set_bytes": 0,
                "max_live_working_set_bytes": False,
                "min_available_memory_bytes": 0,
            },
            "max live working set must be an integer",
        ),
        (
            {
                "estimated_live_working_set_bytes": 0,
                "max_live_working_set_bytes": None,
                "min_available_memory_bytes": True,
            },
            "min free unified memory must be an integer",
        ),
        (
            {
                "estimated_live_working_set_bytes": 0,
                "max_live_working_set_bytes": None,
                "min_available_memory_bytes": 0,
                "resident_backing_bytes": False,
            },
            "resident_backing_bytes must be an integer",
        ),
        (
            {
                "estimated_live_working_set_bytes": 0,
                "max_live_working_set_bytes": None,
                "min_available_memory_bytes": 0,
                "nonresident_peak_bytes": True,
            },
            "nonresident_peak_bytes must be an integer",
        ),
    ),
)
def test_live_memory_budget_rejects_boolean_integer_fields(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(GenerationGuardError, match=message):
        check_live_memory_budget(**kwargs)


def test_prompt_prefill_live_memory_estimate_accounts_for_cache_write_cap() -> None:
    estimate = estimate_prompt_prefill_live_memory(
        max_prompt_batch_mib=64,
        max_runner_scratch_mib=128,
        max_cache_read_mib=32,
        max_cache_write_mib=512,
        copy_chunk_mib=8,
    )

    assert estimate.prompt_batch_bytes == 64 * 1024**2
    assert estimate.runner_scratch_bytes == 128 * 1024**2
    assert estimate.cache_write_bytes == 512 * 1024**2
    assert estimate.estimated_live_working_set_bytes == (512 + 128) * 1024**2


def _call_generate_token_ids_for_validation(**overrides: object) -> None:
    kwargs = {
        "runner_path": "unused-runner",
        "expert_layout_path": "unused-experts.json",
        "resident_layout_path": "unused-resident.json",
        "cache_layout_path": "unused-cache-layout.json",
        "cache_file_path": "unused-cache.bin",
        "prompt_token_ids": [0],
        "max_new_tokens": 1,
        "num_heads": 1,
        "qk_nope_dim": 1,
        "rope_dim": 1,
        "v_head_dim": 1,
    }
    kwargs.update(overrides)
    generate_token_ids(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"prompt_token_ids": [0, 1.5]}, "prompt_token_ids must be an integer"),
        ({"prompt_token_ids": [True]}, "prompt_token_ids must be an integer"),
        ({"max_new_tokens": 1.5}, "max_new_tokens must be an integer"),
        ({"max_new_tokens": True}, "max_new_tokens must be an integer"),
        ({"logits_top_k": 1.5}, "logits_top_k must be an integer"),
        ({"top_k": 1.5}, "top_k must be an integer"),
        ({"max_k": True}, "max_k must be an integer"),
        ({"top_k": 4, "max_k": 2}, "top_k/max_k"),
        ({"cache_dtype_bytes": 3}, "cache_dtype_bytes must be 2 or 4"),
        ({"logits_chunk_rows": 1.5}, "logits_chunk_rows must be an integer"),
        ({"router_n_group": 1.5}, "router_n_group must be an integer"),
        ({"sampling_seed": False}, "sampling_seed must be an integer"),
        (
            {"batch_prefill_prompt": True, "prefill_prompt_chunk_tokens": -1},
            "prefill_prompt_chunk_tokens must be non-negative",
        ),
        (
            {"prefill_mpsgraph_min_batch_tokens": False},
            "prefill_mpsgraph_min_batch_tokens must be an integer",
        ),
        (
            {"prefill_mpsgraph_min_matrix_dim": 1.5},
            "prefill_mpsgraph_min_matrix_dim must be an integer",
        ),
        (
            {"prefill_min_accelerated_flop_fraction": 1.5},
            "prefill_min_accelerated_flop_fraction must be 0..1",
        ),
        (
            {"prefill_min_accelerated_flop_fraction": True},
            "prefill_min_accelerated_flop_fraction must be 0..1",
        ),
        (
            {"prefill_router_hybrid_margin_threshold": True},
            "prefill_router_hybrid_margin_threshold must be a non-negative finite number",
        ),
        (
            {"prefill_router_hybrid_margin_threshold": -1.0},
            "prefill_router_hybrid_margin_threshold must be a non-negative finite number",
        ),
        ({"eos_token_ids": [False]}, "eos token ids must be an integer"),
        ({"eos_token_id": 1.5}, "eos token ids must be an integer"),
    ),
)
def test_generate_token_ids_rejects_non_integer_control_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TokenGeneratorError, match=message):
        _call_generate_token_ids_for_validation(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"sampling_temperature": True}, "temperature must be numeric"),
        ({"sampling_temperature": float("nan")}, "temperature must be finite"),
        ({"sampling_top_p": False}, "top_p must be numeric"),
        ({"sampling_top_p": 1.5}, "top_p must be in \\(0, 1\\]"),
        ({"routed_scaling_factor": float("inf")}, "routed_scaling_factor must be finite"),
        ({"attention_scale": 0.0}, "attention_scale must be positive"),
        ({"rope_theta": float("nan")}, "rope_theta must be finite"),
        (
            {"prefill_expert_stage_align_kib": 0.0},
            "prefill_expert_stage_align_kib must be positive",
        ),
        (
            {"prefill_max_routed_read_amplification": True},
            "prefill_max_routed_read_amplification must be numeric",
        ),
        (
            {"prefill_max_routed_read_amplification": -1.0},
            "prefill_max_routed_read_amplification must be non-negative",
        ),
        (
            {"prefill_max_routed_read_gib": True},
            "prefill_max_routed_read_gib must be numeric",
        ),
        (
            {"prefill_max_routed_read_gib": -1.0},
            "prefill_max_routed_read_gib must be non-negative",
        ),
        (
            {"prefill_max_stage_raw_ranges": True},
            "prefill_max_stage_raw_ranges must be an integer",
        ),
        (
            {"prefill_max_stage_raw_ranges": -1},
            "prefill_max_stage_raw_ranges must be non-negative",
        ),
        (
            {"prefill_max_stage_coalesced_ranges": True},
            "prefill_max_stage_coalesced_ranges must be an integer",
        ),
        (
            {"prefill_max_stage_coalesced_ranges": -1},
            "prefill_max_stage_coalesced_ranges must be non-negative",
        ),
        (
            {"prefill_expert_stage_tiling": 1},
            "prefill_expert_stage_tiling must be a boolean",
        ),
        (
            {"prefill_persistent_moe_plan_server": 1},
            "prefill_persistent_moe_plan_server must be a boolean",
        ),
        (
            {"prefill_persistent_resident_linear_server": 1},
            "prefill_persistent_resident_linear_server must be a boolean",
        ),
        (
            {"prefill_persistent_attention_projection_server": 1},
            "prefill_persistent_attention_projection_server must be a boolean",
        ),
        (
            {"prefill_persistent_attention_output_server": 1},
            "prefill_persistent_attention_output_server must be a boolean",
        ),
        (
            {"prefill_persistent_shared_expert_server": 1},
            "prefill_persistent_shared_expert_server must be a boolean",
        ),
        (
            {"prefill_persistent_rope_split_server": 1},
            "prefill_persistent_rope_split_server must be a boolean",
        ),
        (
            {"prefill_persistent_mla_attention_server": 1},
            "prefill_persistent_mla_attention_server must be a boolean",
        ),
        (
            {"prefill_persistent_rmsnorm_server": 1},
            "prefill_persistent_rmsnorm_server must be a boolean",
        ),
        (
            {"decode_mla_key_cache": 1},
            "decode_mla_key_cache must be a boolean",
        ),
        (
            {"prefill_moe_output_accumulator": "bad"},
            "moe_output_accumulator must be env, file, or memory",
        ),
        (
            {"prefill_ssd_read_gib_per_second": True},
            "prefill_ssd_read_gib_per_second must be numeric",
        ),
        (
            {"prefill_max_routed_read_seconds": -1.0},
            "prefill_max_routed_read_seconds must be non-negative",
        ),
        (
            {"prefill_max_routed_read_seconds": 1.0},
            "prefill_ssd_read_gib_per_second must be positive",
        ),
        (
            {"decode_max_routed_read_gib_per_token": -1.0},
            "decode_max_routed_read_gib_per_token must be non-negative",
        ),
        (
            {"decode_max_routed_read_seconds_per_token": -1.0},
            "decode_max_routed_read_seconds_per_token must be non-negative",
        ),
        (
            {"decode_max_routed_read_seconds_per_token": 1.0},
            "prefill_ssd_read_gib_per_second must be positive",
        ),
        (
            {"prefill_mpsgraph_min_batch_tokens": 0},
            "prefill_mpsgraph_min_batch_tokens must be positive",
        ),
    ),
)
def test_generate_token_ids_rejects_invalid_numeric_control_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TokenGeneratorError, match=message):
        _call_generate_token_ids_for_validation(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"logits_max_chunk_mib": 0.0}, "logits_max_chunk_mib"),
        ({"max_cache_file_mib": 0.0}, "max_cache_file_mib"),
        ({"max_cache_read_mib": 0.0}, "max_cache_read_mib"),
        ({"max_embedding_row_mib": 0.0}, "max_embedding_row_mib"),
        ({"max_runner_scratch_mib": float("nan")}, "max_runner_scratch_mib"),
        ({"max_live_working_set_mib": -1.0}, "max_live_working_set_mib"),
        ({"min_free_unified_memory_mib": -1.0}, "min_free_unified_memory_mib"),
        ({"extra_live_working_set_bytes": -1}, "extra_live_working_set_bytes"),
    ),
)
def test_generation_runtime_rejects_invalid_memory_caps(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    kwargs = {
        "expert_layout_path": expert,
        "resident_layout_path": resident,
        "cache_layout_path": cache_layout,
        "cache_file_path": cache_file,
        "requested_context_tokens": 1,
        "layers": {1},
        "dense_layers": None,
        "top_k": 2,
        "max_k": 2,
        "num_heads": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "include_shared_expert": False,
        "logits_top_k": 2,
        "logits_chunk_rows": None,
        "logits_max_chunk_mib": 1.0,
        "rms_norm_eps": 0.0,
        "max_slot_mib": 1.0,
        "max_router_mib": 1.0,
        "max_resident_matrix_mib": 1.0,
        "max_cache_file_mib": 1.0,
        "max_cache_read_mib": 1.0,
        "max_runner_scratch_mib": 64.0,
        "cache_dtype_bytes": 2,
        "metal_final_logits": False,
        "max_live_working_set_mib": 8192.0,
        "min_free_unified_memory_mib": 0.0,
        "extra_live_working_set_bytes": 0,
    }
    kwargs.update(overrides)

    with pytest.raises(GenerationGuardError, match=message):
        check_generation_runtime(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"requested_context_tokens": True},
            "requested_context_tokens must be an integer",
        ),
        ({"top_k": False}, "top_k must be an integer"),
        ({"max_k": 2.5}, "max_k must be an integer"),
        ({"num_heads": True}, "num_heads must be an integer"),
        ({"qk_nope_dim": 1.5}, "qk_nope_dim must be an integer"),
        ({"rope_dim": False}, "rope_dim must be an integer"),
        ({"v_head_dim": 1.5}, "v_head_dim must be an integer"),
        ({"logits_top_k": True}, "logits_top_k must be an integer"),
        ({"logits_chunk_rows": False}, "logits_chunk_rows must be an integer"),
        ({"cache_dtype_bytes": True}, "cache_dtype_bytes must be an integer"),
        (
            {"extra_live_working_set_bytes": False},
            "extra_live_working_set_bytes must be an integer",
        ),
        ({"dsa_index_topk": True}, "dsa_index_topk must be an integer"),
        ({"dsa_index_head_dim": False}, "dsa_index_head_dim must be an integer"),
        ({"expected_vocab_size": False}, "expected_vocab_size must be an integer"),
        ({"expected_hidden_size": False}, "expected_hidden_size must be an integer"),
        ({"layers": {True}}, "layers must be an integer"),
        ({"dense_layers": {False}}, "dense_layers must be an integer"),
    ),
)
def test_generation_runtime_rejects_non_integer_control_values(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    kwargs = {
        "expert_layout_path": expert,
        "resident_layout_path": resident,
        "cache_layout_path": cache_layout,
        "cache_file_path": cache_file,
        "requested_context_tokens": 1,
        "layers": {1},
        "dense_layers": None,
        "top_k": 2,
        "max_k": 2,
        "num_heads": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "include_shared_expert": False,
        "logits_top_k": 2,
        "logits_chunk_rows": None,
        "logits_max_chunk_mib": 1.0,
        "rms_norm_eps": 0.0,
        "max_slot_mib": 1.0,
        "max_router_mib": 1.0,
        "max_resident_matrix_mib": 1.0,
        "max_cache_file_mib": 1.0,
        "max_cache_read_mib": 1.0,
        "max_runner_scratch_mib": 64.0,
        "cache_dtype_bytes": 2,
        "metal_final_logits": False,
        "max_live_working_set_mib": 8192.0,
        "min_free_unified_memory_mib": 0.0,
        "extra_live_working_set_bytes": 0,
    }
    kwargs.update(overrides)

    with pytest.raises(GenerationGuardError, match=message):
        check_generation_runtime(**kwargs)


def _add_tensor(
    tensors: list[dict],
    payload: bytearray,
    name: str,
    dtype: str,
    shape: list[int],
    data: bytes,
    category: str,
) -> None:
    tensors.append(
        {
            "name": name,
            "offset": len(payload),
            "size": len(data),
            "dtype": dtype,
            "shape": shape,
            "category": category,
        }
    )
    payload.extend(data)


def write_fixture(root: Path, *, include_dsa: bool = False) -> tuple[Path, Path, Path, Path]:
    experts = root / "experts"
    resident = root / "resident"
    experts.mkdir()
    resident.mkdir()
    expert_layout = experts / "layout.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "quantization": "mlx-affine-int4",
                "group_size": 8,
                "num_layers": 2,
                "num_experts": 2,
                "component_order": [name for name, *_ in COMPONENTS],
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 2,
                        "expert_slot_bytes": 192,
                        "layer_file": "layer_001.bin",
                        "components": [
                            {
                                "name": name,
                                "offset": offset,
                                "size": size,
                                "dtype": dtype,
                                "shape": shape,
                            }
                            for name, offset, size, dtype, shape in COMPONENTS
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (experts / "layer_001.bin").write_bytes(b"\0" * (2 * 192))

    tensors: list[dict] = []
    payload = bytearray()
    _add_tensor(
        tensors,
        payload,
        "model.embed_tokens.weight",
        "F32",
        [4, 8],
        f32([1.0] * 8 + [0.0] * 8 + [2.0] * 8 + [-1.0] * 8),
        "embeddings",
    )
    _add_tensor(tensors, payload, "model.norm.weight", "F32", [8], f32([1.0] * 8), "norms")
    _add_tensor(
        tensors,
        payload,
        "lm_head.weight",
        "F32",
        [4, 8],
        f32([1.0] * 8 + [0.0] * 8 + [2.0] * 8 + [-1.0] * 8),
        "lm_head",
    )
    layer_specs = [
        ("input_layernorm.weight", "F32", [8], 32, "norms"),
        ("self_attn.q_a_layernorm.weight", "F32", [2], 8, "norms"),
        ("self_attn.kv_a_layernorm.weight", "F32", [2], 8, "norms"),
        ("post_attention_layernorm.weight", "F32", [8], 32, "norms"),
        ("self_attn.q_a_proj.weight", "F32", [2, 8], 64, "attention"),
        ("self_attn.q_b_proj.weight", "F32", [6, 2], 48, "attention"),
        ("self_attn.kv_a_proj_with_mqa.weight", "F32", [4, 8], 128, "attention"),
        ("self_attn.kv_b_proj.weight", "F32", [4, 2], 32, "attention"),
        ("self_attn.o_proj.weight", "F32", [8, 2], 64, "attention"),
        ("mlp.gate.weight", "F32", [2, 8], 64, "routers"),
    ]
    for layer in (0, 1):
        for suffix, dtype, shape, size, category in layer_specs:
            if layer == 0 and suffix == "mlp.gate.weight":
                continue
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.{suffix}",
                dtype,
                shape,
                b"\0" * size,
                category,
            )
        if layer == 0:
            for component in ("gate_proj", "up_proj", "down_proj"):
                _add_tensor(
                    tensors,
                    payload,
                    f"model.layers.0.mlp.{component}.weight",
                    "F32",
                    [8, 8],
                    b"\0" * 256,
                    "dense_mlp",
                )
        elif include_dsa:
            for suffix, shape, data in (
                ("self_attn.indexer.wk.weight", [2, 8], b"\0" * 64),
                ("self_attn.indexer.k_norm.weight", [2], b"\0" * 8),
                ("self_attn.indexer.k_norm.bias", [2], b"\0" * 8),
                ("self_attn.indexer.wq_b.weight", [2, 2], b"\0" * 16),
                ("self_attn.indexer.weights_proj.weight", [1, 8], b"\0" * 32),
            ):
                _add_tensor(
                    tensors,
                    payload,
                    f"model.layers.{layer}.{suffix}",
                    "F32",
                    shape,
                    data,
                    "dsa_indexer",
                )
    resident_layout = resident / "layout.json"
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    (resident / "resident.bin").write_bytes(payload)

    cache_total_bytes = 96
    cache_segments = [
        {
            "kind": "mla_kv",
            "layer": 0,
            "offset": 0,
            "width": 4,
            "dtype": "BF16",
            "dtype_bytes": 2,
            "token_stride_bytes": 8,
            "max_context_tokens": 4,
            "total_bytes": 32,
        },
        {
            "kind": "mla_kv",
            "layer": 1,
            "offset": 64,
            "width": 4,
            "dtype": "BF16",
            "dtype_bytes": 2,
            "token_stride_bytes": 8,
            "max_context_tokens": 4,
            "total_bytes": 32,
        },
    ]
    if include_dsa:
        cache_segments.append(
            {
                "kind": "dsa_index",
                "layer": 1,
                "offset": 96,
                "width": 2,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "token_stride_bytes": 4,
                "max_context_tokens": 4,
                "total_bytes": 16,
            }
        )
        cache_total_bytes = 112

    cache_layout = root / "cache_layout.json"
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": cache_total_bytes,
                "segments": cache_segments,
            }
        ),
        encoding="utf-8",
    )
    cache_file = root / "decode_cache.bin"
    cache_file.write_bytes(b"\0" * cache_total_bytes)
    return expert_layout, resident_layout, cache_layout, cache_file


def remove_resident_tensor(resident_layout: Path, tensor_name: str) -> None:
    payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    payload["tensors"] = [
        tensor
        for tensor in payload["tensors"]
        if tensor.get("name") != tensor_name
    ]
    resident_layout.write_text(json.dumps(payload), encoding="utf-8")


def replace_lm_head_with_affine(resident_layout: Path) -> None:
    payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    weight_file = payload["weight_file"]
    assert isinstance(weight_file, str)
    resident_bin = resident_layout.parent / weight_file
    data = bytearray(resident_bin.read_bytes())
    payload["tensors"] = [
        tensor
        for tensor in payload["tensors"]
        if tensor.get("name") != "lm_head.weight"
    ]

    def add(name: str, dtype: str, shape: list[int], blob: bytes) -> None:
        payload["tensors"].append(
            {
                "name": name,
                "offset": len(data),
                "size": len(blob),
                "dtype": dtype,
                "shape": shape,
                "category": "lm_head",
            }
        )
        data.extend(blob)

    add(
        "lm_head.weight",
        "U32",
        [4, 1],
        struct.pack("<4I", *(pack8([value] * 8) for value in (0, 1, 2, 3))),
    )
    add("lm_head.scales", "BF16", [4, 1], bf16([1.0] * 4))
    add("lm_head.biases", "BF16", [4, 1], bf16([0.0] * 4))
    payload["total_bytes"] = len(data)
    resident_layout.write_text(json.dumps(payload), encoding="utf-8")
    resident_bin.write_bytes(data)


def replace_embedding_with_mxfp4(resident_layout: Path) -> None:
    payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    weight_file = payload["weight_file"]
    assert isinstance(weight_file, str)
    resident_bin = resident_layout.parent / weight_file
    data = bytearray(resident_bin.read_bytes())
    payload["tensors"] = [
        tensor
        for tensor in payload["tensors"]
        if tensor.get("name") != "model.embed_tokens.weight"
    ]

    def add(name: str, dtype: str, shape: list[int], blob: bytes) -> None:
        payload["tensors"].append(
            {
                "name": name,
                "offset": len(data),
                "size": len(blob),
                "dtype": dtype,
                "shape": shape,
                "category": "embeddings",
            }
        )
        data.extend(blob)

    add(
        "model.embed_tokens.weight",
        "U32",
        [4, 1],
        struct.pack("<4I", *(pack8([value] * 8) for value in (2, 1, 0, 10))),
    )
    add("model.embed_tokens.scales", "U8", [4, 1], bytes([127] * 4))
    payload["total_bytes"] = len(data)
    resident_layout.write_text(json.dumps(payload), encoding="utf-8")
    resident_bin.write_bytes(data)


def _inflate_resident_backing(resident_layout: Path, target_bytes: int) -> None:
    payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    assert target_bytes > payload["total_bytes"]
    weight_file = payload["weight_file"]
    assert isinstance(weight_file, str)
    payload["total_bytes"] = target_bytes
    resident_layout.write_text(json.dumps(payload), encoding="utf-8")
    (resident_layout.parent / weight_file).write_bytes(b"\0" * target_bytes)


def test_generation_runtime_live_working_set_includes_resident_backing(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    guard = check_generation_runtime(
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        requested_context_tokens=1,
        layers={1},
        dense_layers=None,
        top_k=2,
        max_k=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        include_shared_expert=False,
        logits_top_k=2,
        logits_chunk_rows=None,
        logits_max_chunk_mib=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1.0,
        max_router_mib=1.0,
        max_resident_matrix_mib=1.0,
        max_cache_file_mib=1.0,
        max_cache_read_mib=1.0,
        max_runner_scratch_mib=64.0,
        cache_dtype_bytes=2,
        metal_final_logits=False,
        max_live_working_set_mib=8192.0,
        min_free_unified_memory_mib=0.0,
    )
    resident_total = json.loads(resident.read_text(encoding="utf-8"))["total_bytes"]
    nonresident_peak = max(
        guard.max_layer_peak_bytes,
        guard.final_logits_budget.estimated_peak_bytes,
        guard.embedding_budget.row_bytes + guard.embedding_budget.output_bytes,
    )

    live = guard.live_memory_budget
    assert live.resident_backing_bytes == resident_total
    assert live.nonresident_peak_bytes == nonresident_peak
    assert live.extra_live_working_set_bytes == 0
    assert live.estimated_live_working_set_bytes == resident_total + nonresident_peak


def test_generation_runtime_accounts_for_decode_mla_key_cache(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    guard = check_generation_runtime(
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        requested_context_tokens=3,
        layers={1},
        dense_layers=None,
        top_k=2,
        max_k=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        include_shared_expert=False,
        logits_top_k=2,
        logits_chunk_rows=None,
        logits_max_chunk_mib=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1.0,
        max_router_mib=1.0,
        max_resident_matrix_mib=1.0,
        max_cache_file_mib=1.0,
        max_cache_read_mib=1.0,
        max_runner_scratch_mib=64.0,
        cache_dtype_bytes=2,
        metal_final_logits=False,
        decode_mla_key_cache=True,
        max_live_working_set_mib=8192.0,
        min_free_unified_memory_mib=0.0,
    )

    budget = guard.layer_budgets[0]
    assert budget.decoder_mla_key_cache is True
    assert budget.decoder_mla_key_cache_bytes == 24
    assert budget.decoder_mla_attention_peak_bytes >= (
        budget.decoder_mla_key_cache_bytes
    )


def test_generation_runtime_accepts_affine_int4_lm_head_budget(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    replace_lm_head_with_affine(resident)

    guard = check_generation_runtime(
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        requested_context_tokens=1,
        layers={1},
        dense_layers=None,
        top_k=2,
        max_k=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        include_shared_expert=False,
        logits_top_k=2,
        logits_chunk_rows=2,
        logits_max_chunk_mib=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1.0,
        max_router_mib=1.0,
        max_resident_matrix_mib=1.0,
        max_cache_file_mib=1.0,
        max_cache_read_mib=1.0,
        max_runner_scratch_mib=64.0,
        cache_dtype_bytes=2,
        metal_final_logits=True,
        max_live_working_set_mib=8192.0,
        min_free_unified_memory_mib=0.0,
    )

    budget = guard.final_logits_budget
    assert budget.head_tensor == "lm_head.weight"
    assert budget.dtype == "affine-int4"
    assert budget.hidden_dim == 8
    assert budget.vocab_size == 4
    assert budget.chunk_rows == 2
    assert budget.chunk_bytes == 16
    assert budget.read_bytes == 32
    assert budget.estimated_peak_bytes == 2 * 1024 * 1024 + 8 * 4 + 2 * 4


def test_generation_runtime_accepts_mxfp4_embedding_budget(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    replace_embedding_with_mxfp4(resident)

    guard = check_generation_runtime(
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        requested_context_tokens=1,
        layers={1},
        dense_layers=None,
        top_k=2,
        max_k=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        include_shared_expert=False,
        logits_top_k=2,
        logits_chunk_rows=None,
        logits_max_chunk_mib=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1.0,
        max_router_mib=1.0,
        max_resident_matrix_mib=1.0,
        max_cache_file_mib=1.0,
        max_cache_read_mib=1.0,
        max_runner_scratch_mib=64.0,
        cache_dtype_bytes=2,
        metal_final_logits=False,
        max_live_working_set_mib=8192.0,
        min_free_unified_memory_mib=0.0,
    )

    assert guard.embedding_budget.dtype == "mlx-mxfp4"
    assert guard.embedding_budget.hidden_dim == 8
    assert guard.embedding_budget.row_bytes == 5
    assert guard.embedding_budget.output_bytes == 32


def test_generation_runtime_rejects_resident_backing_over_live_cap(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    _inflate_resident_backing(resident, 2 * 1024**2)

    with pytest.raises(GenerationGuardError, match="live working set"):
        check_generation_runtime(
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            requested_context_tokens=1,
            layers={1},
            dense_layers=None,
            top_k=2,
            max_k=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            include_shared_expert=False,
            logits_top_k=2,
            logits_chunk_rows=None,
            logits_max_chunk_mib=1.0,
            rms_norm_eps=0.0,
            max_slot_mib=1.0,
            max_router_mib=1.0,
            max_resident_matrix_mib=1.0,
            max_cache_file_mib=1.0,
            max_cache_read_mib=1.0,
            max_runner_scratch_mib=64.0,
            cache_dtype_bytes=2,
            metal_final_logits=False,
            max_live_working_set_mib=3.0,
            min_free_unified_memory_mib=0.0,
        )


def test_generation_runtime_rejects_missing_lm_head_when_tied_disallowed(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    remove_resident_tensor(resident, "lm_head.weight")

    with pytest.raises(GenerationGuardError, match="lm_head.weight not found"):
        check_generation_runtime(
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            requested_context_tokens=1,
            layers={1},
            dense_layers=None,
            top_k=2,
            max_k=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            include_shared_expert=False,
            logits_top_k=2,
            logits_chunk_rows=None,
            logits_max_chunk_mib=1.0,
            rms_norm_eps=0.0,
            max_slot_mib=1.0,
            max_router_mib=1.0,
            max_resident_matrix_mib=1.0,
            max_cache_file_mib=1.0,
            max_cache_read_mib=1.0,
            max_runner_scratch_mib=64.0,
            cache_dtype_bytes=2,
            metal_final_logits=False,
            allow_tied_embeddings=False,
            max_live_working_set_mib=8192.0,
            min_free_unified_memory_mib=0.0,
            extra_live_working_set_bytes=0,
        )


def test_generation_runtime_rejects_missing_embedding(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    remove_resident_tensor(resident, "model.embed_tokens.weight")

    with pytest.raises(GenerationGuardError, match="embed_tokens.weight not found"):
        check_generation_runtime(
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            requested_context_tokens=1,
            layers={1},
            dense_layers=None,
            top_k=2,
            max_k=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            include_shared_expert=False,
            logits_top_k=2,
            logits_chunk_rows=None,
            logits_max_chunk_mib=1.0,
            rms_norm_eps=0.0,
            max_slot_mib=1.0,
            max_router_mib=1.0,
            max_resident_matrix_mib=1.0,
            max_cache_file_mib=1.0,
            max_cache_read_mib=1.0,
            max_runner_scratch_mib=64.0,
            cache_dtype_bytes=2,
            metal_final_logits=False,
            max_live_working_set_mib=8192.0,
            min_free_unified_memory_mib=0.0,
            extra_live_working_set_bytes=0,
        )


def test_generation_runtime_rejects_vocab_size_mismatch(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)

    with pytest.raises(
        GenerationGuardError,
        match="lm_head/embedding vocab size 4 does not match config vocab_size 5",
    ):
        check_generation_runtime(
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            requested_context_tokens=1,
            layers={1},
            dense_layers=None,
            top_k=2,
            max_k=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            include_shared_expert=False,
            logits_top_k=2,
            logits_chunk_rows=None,
            logits_max_chunk_mib=1.0,
            rms_norm_eps=0.0,
            max_slot_mib=1.0,
            max_router_mib=1.0,
            max_resident_matrix_mib=1.0,
            max_cache_file_mib=1.0,
            max_cache_read_mib=1.0,
            max_runner_scratch_mib=64.0,
            cache_dtype_bytes=2,
            metal_final_logits=False,
            expected_vocab_size=5,
            max_live_working_set_mib=8192.0,
            min_free_unified_memory_mib=0.0,
            extra_live_working_set_bytes=0,
        )


def test_generation_runtime_rejects_hidden_size_mismatch(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)

    with pytest.raises(
        GenerationGuardError,
        match="lm_head/embedding hidden dim 8 does not match config hidden_size 7",
    ):
        check_generation_runtime(
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            requested_context_tokens=1,
            layers={1},
            dense_layers=None,
            top_k=2,
            max_k=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            include_shared_expert=False,
            logits_top_k=2,
            logits_chunk_rows=None,
            logits_max_chunk_mib=1.0,
            rms_norm_eps=0.0,
            max_slot_mib=1.0,
            max_router_mib=1.0,
            max_resident_matrix_mib=1.0,
            max_cache_file_mib=1.0,
            max_cache_read_mib=1.0,
            max_runner_scratch_mib=64.0,
            cache_dtype_bytes=2,
            metal_final_logits=False,
            expected_hidden_size=7,
            max_live_working_set_mib=8192.0,
            min_free_unified_memory_mib=0.0,
            extra_live_working_set_bytes=0,
        )


def write_fake_runner(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import shutil
import struct
import sys
from pathlib import Path

def arg(name):
    i = sys.argv.index(name)
    return sys.argv[i + 1]

def write(path, values):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(struct.pack(f"<{len(values)}f", *values))

if "--run-rmsnorm-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    suffix = arg("--norm-suffix")
    dim = 2 if "q_a_layernorm" in suffix or "kv_a_layernorm" in suffix else 8
    write(arg("--output-f32"), [0.0] * (batch * dim))
elif "--run-resident-linear-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    suffix = arg("--tensor-suffix")
    if suffix.endswith("q_a_proj.weight"):
        out_dim = 2
    elif suffix.endswith("q_b_proj.weight"):
        out_dim = 6
    elif suffix.endswith("kv_a_proj_with_mqa.weight"):
        out_dim = 4
    elif suffix.endswith("kv_b_proj.weight"):
        out_dim = 4
    elif suffix.endswith("o_proj.weight"):
        out_dim = 8
    elif suffix.endswith("mlp.gate_proj.weight"):
        out_dim = 8
    elif suffix.endswith("mlp.up_proj.weight"):
        out_dim = 8
    elif suffix.endswith("mlp.down_proj.weight"):
        out_dim = 8
    else:
        raise SystemExit(f"unknown tensor suffix {suffix}")
    write(arg("--output-f32"), [0.0] * (batch * out_dim))
elif "--run-rope-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    heads = int(arg("--num-heads"))
    rope = int(arg("--rope-dim"))
    write(arg("--output-q-f32"), [0.0] * (batch * heads * rope))
    write(arg("--output-k-f32"), [0.0] * (batch * rope))
elif "--run-mla-attention-indexed-batch" in sys.argv or "--run-mla-attention-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    heads = int(arg("--num-heads"))
    v_head = int(arg("--v-head-dim"))
    write(arg("--output-f32"), [0.0] * (batch * heads * v_head))
else:
    shutil.copyfile(arg("--input-f32"), arg("--output-f32"))
""",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


def add_dsa_index_segment(cache_layout: Path, cache_file: Path) -> None:
    payload = json.loads(cache_layout.read_text(encoding="utf-8"))
    payload["segments"].append(
        {
            "kind": "dsa_index",
            "layer": 0,
            "offset": 96,
            "width": 4,
            "dtype": "BF16",
            "dtype_bytes": 2,
            "token_stride_bytes": 8,
            "max_context_tokens": 4,
            "total_bytes": 32,
        }
    )
    payload["total_bytes"] = 128
    cache_layout.write_text(json.dumps(payload), encoding="utf-8")
    cache_file.write_bytes(b"\0" * 128)


def test_auto_prefill_prompt_chunk_respects_expert_stage_caps(tmp_path: Path) -> None:
    expert_dir = tmp_path / "experts"
    resident_dir = tmp_path / "resident"
    expert_dir.mkdir()
    resident_dir.mkdir()
    expert_layout = expert_dir / "layout.json"
    resident_layout = resident_dir / "layout.json"
    cache_layout = tmp_path / "cache_layout.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 4,
                        "expert_slot_bytes": 192,
                        "layer_file": "layer_001.bin",
                        "components": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 256,
                        "dtype": "F32",
                        "shape": [8, 8],
                    },
                    {
                        "name": "model.layers.1.self_attn.q_a_proj.weight",
                        "offset": 256,
                        "size": 256,
                        "dtype": "F32",
                        "shape": [8, 8],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 8,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 64,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": 4,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "max_context_tokens": 8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = _auto_prefill_prompt_chunk_plan(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=8,
        start_position=0,
        layers={1},
        dense_layers=None,
        work_dir=None,
        top_k=2,
        max_prompt_batch_mib=1,
        max_cache_read_mib=1,
        max_cache_write_mib=1,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=(2 * (192 + 4096)) / (1024 * 1024),
        prefill_max_compact_stage_mib=(2 * 192) / (1024 * 1024),
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
    )

    assert plan.chunk_tokens == 1
    assert plan.raw_tokens == 1
    assert plan.limiting_cap_tokens == 1
    assert {cap.name for cap in plan.limiting_caps} == {"expert_stage_layer_1"}
    assert plan.hidden_dim == 8
    assert plan.per_token_activation_bytes > 0
    assert _auto_prefill_prompt_chunk_tokens(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=8,
        start_position=0,
        layers={1},
        dense_layers=None,
        work_dir=None,
        top_k=2,
        max_prompt_batch_mib=1,
        max_cache_read_mib=1,
        max_cache_write_mib=1,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=(2 * (192 + 4096)) / (1024 * 1024),
        prefill_max_compact_stage_mib=(2 * 192) / (1024 * 1024),
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
    ) == 1


def test_auto_prefill_prompt_chunk_plan_reports_mpp_blocking_caps(
    tmp_path: Path,
) -> None:
    expert_layout = tmp_path / "experts.json"
    resident_layout = tmp_path / "resident.json"
    cache_layout = tmp_path / "cache.json"
    expert_layout.write_text(
        json.dumps({"version": 1, "model_type": "glm_moe_dsa", "layers": []}),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "offset": 4096,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 256,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 0,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )

    plan = _auto_prefill_prompt_chunk_plan(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=256,
        start_position=0,
        layers={0},
        dense_layers={0},
        work_dir=None,
        top_k=1,
        max_prompt_batch_mib=(64 * 32 * 4) / (1024 * 1024),
        max_cache_read_mib=1,
        max_cache_write_mib=1,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        tile_tokens=1,
    )

    assert plan.chunk_tokens == 64
    assert plan.mpp_tensor_ops_candidate_min_batch_tokens == 128
    assert plan.mpp_tensor_ops_candidate_min_matrix_dim == 32
    assert plan.mpp_tensor_ops_dimension_candidate_matrix_count == 1
    assert plan.mpp_tensor_ops_candidate_reachable_under_caps is False
    assert plan.mpp_tensor_ops_candidate_blocking_cap_names == (
        "prompt_batch_bytes",
    )
    assert plan.mpp_tensor_ops_candidate_blocking_cap_summary == {
        "prompt_batch_bytes": 1
    }
    assert plan.mpp_tensor_ops_candidate_non_expert_blocking_cap_names == (
        "prompt_batch_bytes",
    )
    assert plan.mpp_tensor_ops_candidate_non_expert_blocking_cap_summary == {
        "prompt_batch_bytes": 1
    }
    assert plan.mpp_tensor_ops_candidate_stage_tiling_plan is None
    assert (
        plan.mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable
        is False
    )
    assert plan.mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers == (
        "prompt_batch_bytes",
    )
    assert [cap.name for cap in plan.mpp_tensor_ops_candidate_blocking_caps] == [
        "prompt_batch_bytes"
    ]


def test_auto_prefill_prompt_chunk_plan_keeps_near_tile_safety_cap(
    tmp_path: Path,
) -> None:
    expert_layout = tmp_path / "experts.json"
    resident_layout = tmp_path / "resident.json"
    cache_layout = tmp_path / "cache.json"
    expert_layout.write_text(
        json.dumps({"version": 1, "model_type": "glm_moe_dsa", "layers": []}),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "offset": 4096,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 4096,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 0,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )

    def plan_for_cap(tokens: int):
        return _auto_prefill_prompt_chunk_plan(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            prompt_tokens=2048,
            start_position=0,
            layers={0},
            dense_layers={0},
            work_dir=None,
            top_k=1,
            max_prompt_batch_mib=(tokens * 32 * 4) / (1024 * 1024),
            max_cache_read_mib=1024,
            max_cache_write_mib=1024,
            max_runner_scratch_mib=1024,
            prefill_max_stage_mib=1024,
            prefill_max_compact_stage_mib=1024,
            prefill_expert_stage_align_kib=4,
            prefill_stage_disk_margin_mib=0,
            dsa_indexer_types=None,
            dsa_index_topk=None,
        )

    near_cap = plan_for_cap(113)
    aligned_cap = plan_for_cap(200)

    assert near_cap.raw_tokens == 113
    assert near_cap.chunk_tokens == 113
    assert {cap.name for cap in near_cap.limiting_caps} == {"prompt_batch_bytes"}
    assert aligned_cap.raw_tokens == 200
    assert aligned_cap.chunk_tokens == 192


def test_auto_prefill_prompt_chunk_plan_reports_stage_tiling_counterfactual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout = tmp_path / "experts.json"
    resident_layout = tmp_path / "resident.json"
    cache_layout = tmp_path / "cache.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 8,
                        "expert_slot_bytes": 1024,
                        "layer_file": "layer_001.bin",
                        "components": [],
                    },
                    {
                        "layer": 2,
                        "num_experts": 8,
                        "expert_slot_bytes": 1024,
                        "layer_file": "layer_002.bin",
                        "components": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [128, 32],
                    },
                    {
                        "name": "model.layers.1.self_attn.q_a_proj.weight",
                        "offset": 4096,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                    {
                        "name": "model.layers.2.self_attn.q_a_proj.weight",
                        "offset": 8192,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 256,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 0,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_disk_budget(
        output_dir: str | Path,
        required_bytes: int,
        *,
        safety_margin_bytes: int = 0,
    ) -> DiskBudget:
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=100_000,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr("largerlm.token_generator.disk_budget", fake_disk_budget)

    plan = _auto_prefill_prompt_chunk_plan(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=256,
        start_position=0,
        layers={1, 2},
        dense_layers=None,
        work_dir=tmp_path / "work",
        top_k=2,
        max_prompt_batch_mib=1,
        max_cache_read_mib=1,
        max_cache_write_mib=1,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=(2 * (1024 + 4096)) / (1024 * 1024),
        prefill_max_compact_stage_mib=(2 * 1024) / (1024 * 1024),
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        tile_tokens=1,
    )

    assert plan.chunk_tokens == 1
    assert plan.mpp_tensor_ops_candidate_reachable_under_caps is False
    assert plan.mpp_tensor_ops_candidate_blocking_cap_summary == {
        "expert_stage": 2,
        "work_dir_disk_bytes": 1,
    }
    assert plan.mpp_tensor_ops_candidate_non_expert_blocking_cap_names == (
        "work_dir_disk_bytes",
    )
    assert plan.mpp_tensor_ops_candidate_streamed_stage_disk_cap_tokens == 156
    assert (
        plan.mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable
        is True
    )
    assert plan.mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers == ()
    tiling = plan.mpp_tensor_ops_candidate_stage_tiling_plan
    assert tiling is not None
    assert tiling.target_chunk_tokens == 128
    assert tiling.target_assignments_per_layer == 256
    assert tiling.layer_count == 2
    assert tiling.layers_requiring_tiling == 2
    assert tiling.max_target_unique_experts_per_layer == 8
    assert tiling.max_experts_per_stage_tile == 2
    assert tiling.max_stage_tile_count_per_layer == 4
    assert tiling.total_stage_tile_count == 8
    assert tiling.max_stage_tile_bytes == 2 * (1024 + 4096)
    assert tiling.max_compact_stage_tile_bytes == 2 * 1024
    assert tiling.can_tile_all_layers is True


def test_auto_prefill_prompt_chunk_uses_tiled_stage_caps_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout = tmp_path / "experts.json"
    resident_layout = tmp_path / "resident.json"
    cache_layout = tmp_path / "cache.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 8,
                        "expert_slot_bytes": 1024,
                        "layer_file": "layer_001.bin",
                        "components": [],
                    },
                    {
                        "layer": 2,
                        "num_experts": 8,
                        "expert_slot_bytes": 1024,
                        "layer_file": "layer_002.bin",
                        "components": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [128, 32],
                    },
                    {
                        "name": "model.layers.1.self_attn.q_a_proj.weight",
                        "offset": 4096,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                    {
                        "name": "model.layers.2.self_attn.q_a_proj.weight",
                        "offset": 8192,
                        "size": 4096,
                        "dtype": "F32",
                        "shape": [32, 32],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 256,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 0,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_disk_budget(
        output_dir: str | Path,
        required_bytes: int,
        *,
        safety_margin_bytes: int = 0,
    ) -> DiskBudget:
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=100_000,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr("largerlm.token_generator.disk_budget", fake_disk_budget)
    common = {
        "expert_layout_path": expert_layout,
        "resident_layout_path": resident_layout,
        "cache_layout_path": cache_layout,
        "prompt_tokens": 256,
        "start_position": 0,
        "layers": {1, 2},
        "dense_layers": None,
        "work_dir": tmp_path / "work",
        "top_k": 2,
        "max_prompt_batch_mib": 1,
        "max_cache_read_mib": 1,
        "max_cache_write_mib": 1,
        "max_runner_scratch_mib": 64,
        "prefill_max_stage_mib": (2 * (1024 + 4096)) / (1024 * 1024),
        "prefill_max_compact_stage_mib": (2 * 1024) / (1024 * 1024),
        "prefill_expert_stage_align_kib": 4,
        "prefill_stage_disk_margin_mib": 0,
        "dsa_indexer_types": None,
        "dsa_index_topk": None,
        "tile_tokens": 1,
    }

    untiled = _auto_prefill_prompt_chunk_plan(**common)
    tiled = _auto_prefill_prompt_chunk_plan(
        **common,
        prefill_expert_stage_tiling=True,
    )

    assert untiled.chunk_tokens == 1
    assert untiled.mpp_tensor_ops_candidate_reachable_under_caps is False
    assert tiled.chunk_tokens == 137
    assert tiled.expert_stage_tiling is True
    assert tiled.max_tiled_stage_plus_compact_disk_bytes == 12_288
    assert tiled.mpp_tensor_ops_candidate_reachable_under_caps is True
    assert tiled.mpp_tensor_ops_candidate_blocking_cap_names == ()
    work_cap = next(cap for cap in tiled.caps if cap.name == "work_dir_disk_bytes")
    assert work_cap.detail == "streamed_tiled_stage"
    assert {cap.name for cap in tiled.limiting_caps} == {"work_dir_disk_bytes"}


def test_auto_prefill_prompt_chunk_accounts_for_routed_stage_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)

    def fake_disk_budget(
        output_dir: str | Path,
        required_bytes: int,
        *,
        safety_margin_bytes: int = 0,
    ) -> DiskBudget:
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=20_000,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr("largerlm.token_generator.disk_budget", fake_disk_budget)

    chunk_tokens = _auto_prefill_prompt_chunk_tokens(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=3,
        start_position=0,
        layers=None,
        dense_layers={0},
        work_dir=tmp_path / "work",
        top_k=2,
        max_prompt_batch_mib=1,
        max_cache_read_mib=1,
        max_cache_write_mib=1,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
    )

    assert chunk_tokens == 2


def test_auto_prefill_prompt_chunk_accounts_for_compact_stage_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)

    def fake_disk_budget(
        output_dir: str | Path,
        required_bytes: int,
        *,
        safety_margin_bytes: int = 0,
    ) -> DiskBudget:
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=18_000,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr("largerlm.token_generator.disk_budget", fake_disk_budget)

    chunk_tokens = _auto_prefill_prompt_chunk_tokens(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=3,
        start_position=0,
        layers=None,
        dense_layers={0},
        work_dir=tmp_path / "work",
        top_k=2,
        max_prompt_batch_mib=1,
        max_cache_read_mib=1,
        max_cache_write_mib=1,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
    )

    assert chunk_tokens == 1


def test_auto_prefill_prompt_chunk_rejects_invalid_resident_layout(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)
    resident_layout.write_text(json.dumps({"tensors": {}}), encoding="utf-8")

    with pytest.raises(TokenGeneratorError, match="missing tensors array"):
        _auto_prefill_prompt_chunk_tokens(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            prompt_tokens=3,
            start_position=0,
            layers=None,
            dense_layers={0},
            work_dir=tmp_path / "work",
            top_k=2,
            max_prompt_batch_mib=1,
            max_cache_read_mib=1,
            max_cache_write_mib=1,
            max_runner_scratch_mib=64,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            prefill_expert_stage_align_kib=4,
            prefill_stage_disk_margin_mib=0,
            dsa_indexer_types=None,
            dsa_index_topk=None,
        )


def test_auto_prefill_prompt_chunk_rejects_boolean_resident_shape(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)
    payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    payload["tensors"][0]["shape"][1] = True
    resident_layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TokenGeneratorError, match="shape must use integer rows and cols"):
        _auto_prefill_prompt_chunk_tokens(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            prompt_tokens=3,
            start_position=0,
            layers=None,
            dense_layers={0},
            work_dir=tmp_path / "work",
            top_k=2,
            max_prompt_batch_mib=1,
            max_cache_read_mib=1,
            max_cache_write_mib=1,
            max_runner_scratch_mib=64,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            prefill_expert_stage_align_kib=4,
            prefill_stage_disk_margin_mib=0,
            dsa_indexer_types=None,
            dsa_index_topk=None,
        )


def test_auto_prefill_prompt_chunk_rejects_boolean_expert_slot(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)
    payload = json.loads(expert_layout.read_text(encoding="utf-8"))
    payload["layers"][0]["expert_slot_bytes"] = False
    expert_layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        TokenGeneratorError,
        match="expert layout expert_slot_bytes must be an integer",
    ):
        _auto_prefill_prompt_chunk_tokens(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            prompt_tokens=3,
            start_position=0,
            layers=None,
            dense_layers={0},
            work_dir=tmp_path / "work",
            top_k=2,
            max_prompt_batch_mib=1,
            max_cache_read_mib=1,
            max_cache_write_mib=1,
            max_runner_scratch_mib=64,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            prefill_expert_stage_align_kib=4,
            prefill_stage_disk_margin_mib=0,
            dsa_indexer_types=None,
            dsa_index_topk=None,
        )


def test_auto_prefill_prompt_chunk_rejects_missing_embedding_hidden_dim(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)
    payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    payload["tensors"] = [
        tensor
        for tensor in payload["tensors"]
        if not tensor["name"].endswith(".embed_tokens.weight")
    ]
    resident_layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TokenGeneratorError, match="missing embed_tokens hidden dim"):
        _auto_prefill_prompt_chunk_tokens(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            prompt_tokens=3,
            start_position=0,
            layers=None,
            dense_layers={0},
            work_dir=tmp_path / "work",
            top_k=2,
            max_prompt_batch_mib=1,
            max_cache_read_mib=1,
            max_cache_write_mib=1,
            max_runner_scratch_mib=64,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            prefill_expert_stage_align_kib=4,
            prefill_stage_disk_margin_mib=0,
            dsa_indexer_types=None,
            dsa_index_topk=None,
        )


def test_auto_prefill_prompt_chunk_rejects_invalid_cache_layout(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, _cache_file = write_fixture(tmp_path)
    cache_layout.write_text(json.dumps({"not": "a decode cache layout"}), encoding="utf-8")

    with pytest.raises(TokenGeneratorError, match="failed to load cache layout"):
        _auto_prefill_prompt_chunk_tokens(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            prompt_tokens=3,
            start_position=0,
            layers=None,
            dense_layers={0},
            work_dir=tmp_path / "work",
            top_k=2,
            max_prompt_batch_mib=1,
            max_cache_read_mib=1,
            max_cache_write_mib=1,
            max_runner_scratch_mib=64,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            prefill_expert_stage_align_kib=4,
            prefill_stage_disk_margin_mib=0,
            dsa_indexer_types=None,
            dsa_index_topk=None,
        )


def test_auto_prefill_prompt_chunk_accounts_for_mpsgraph_conversion_scratch(
    tmp_path: Path,
) -> None:
    expert_dir = tmp_path / "experts"
    resident_dir = tmp_path / "resident"
    expert_dir.mkdir()
    resident_dir.mkdir()
    expert_layout = expert_dir / "layout.json"
    resident_layout = resident_dir / "layout.json"
    cache_layout = tmp_path / "cache_layout.json"
    expert_layout.write_text(
        json.dumps({"version": 1, "model_type": "glm_moe_dsa", "layers": []}),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 16 * 8 * 4,
                        "dtype": "F32",
                        "shape": [16, 8],
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "offset": 16 * 8 * 4,
                        "size": 32 * 32 * 2,
                        "dtype": "BF16",
                        "shape": [32, 32],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 256,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 0,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )
    mpsgraph_matrix_scratch = 2 * 1024 * 1024 + 32 * 32 * 2
    per_token_activation = (32 + 32) * 4
    max_scratch_mib = (
        mpsgraph_matrix_scratch + 127 * per_token_activation
    ) / (1024 * 1024)

    auto_chunk = _auto_prefill_prompt_chunk_tokens(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=256,
        start_position=0,
        layers={0},
        dense_layers=None,
        work_dir=None,
        top_k=1,
        max_prompt_batch_mib=1024,
        max_cache_read_mib=1024,
        max_cache_write_mib=1024,
        max_runner_scratch_mib=max_scratch_mib,
        prefill_max_stage_mib=1024,
        prefill_max_compact_stage_mib=1024,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
        tile_tokens=1,
    )
    auto_plan = _auto_prefill_prompt_chunk_plan(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=256,
        start_position=0,
        layers={0},
        dense_layers=None,
        work_dir=None,
        top_k=1,
        max_prompt_batch_mib=1024,
        max_cache_read_mib=1024,
        max_cache_write_mib=1024,
        max_runner_scratch_mib=max_scratch_mib,
        prefill_max_stage_mib=1024,
        prefill_max_compact_stage_mib=1024,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
        tile_tokens=1,
    )
    custom_chunk = _auto_prefill_prompt_chunk_tokens(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=256,
        start_position=0,
        layers={0},
        dense_layers=None,
        work_dir=None,
        top_k=1,
        max_prompt_batch_mib=1024,
        max_cache_read_mib=1024,
        max_cache_write_mib=1024,
        max_runner_scratch_mib=max_scratch_mib,
        prefill_max_stage_mib=1024,
        prefill_max_compact_stage_mib=1024,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        prefill_linear_backend="custom-metal",
        tile_tokens=1,
    )
    relaxed_auto_chunk = _auto_prefill_prompt_chunk_tokens(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=256,
        start_position=0,
        layers={0},
        dense_layers=None,
        work_dir=None,
        top_k=1,
        max_prompt_batch_mib=1024,
        max_cache_read_mib=1024,
        max_cache_write_mib=1024,
        max_runner_scratch_mib=max_scratch_mib,
        prefill_max_stage_mib=1024,
        prefill_max_compact_stage_mib=1024,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=512,
        prefill_mpsgraph_min_matrix_dim=32,
        tile_tokens=1,
    )

    assert auto_chunk == 127
    assert auto_plan.chunk_tokens == 127
    assert auto_plan.max_matrix_scratch_bytes == 2 * 1024 * 1024
    assert auto_plan.next_token_matrix_scratch_bytes == mpsgraph_matrix_scratch
    assert {cap.name for cap in auto_plan.limiting_caps} == {
        "runner_scratch_bytes"
    }
    assert custom_chunk == 135
    assert relaxed_auto_chunk == custom_chunk


def test_auto_prefill_prompt_chunk_accounts_for_affine_int4_triplet_scratch(
    tmp_path: Path,
) -> None:
    expert_layout = tmp_path / "expert_layout.json"
    resident_layout = tmp_path / "resident_layout.json"
    cache_layout = tmp_path / "cache_layout.json"
    out_dim = 1024
    packed_cols = 511
    groups = 511
    in_dim = packed_cols * 8
    weight_bytes = out_dim * packed_cols * 4
    meta_bytes = out_dim * groups * 2
    total_triplet_bytes = weight_bytes + 2 * meta_bytes
    expert_layout.write_text(
        json.dumps({"version": 1, "model_type": "glm_moe_dsa", "layers": []}),
        encoding="utf-8",
    )
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "weight_file": "resident.bin",
                "tensors": [
                    {
                        "name": "model.embed_tokens.weight",
                        "offset": 0,
                        "size": 16 * 8 * 4,
                        "dtype": "F32",
                        "shape": [16, 8],
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "offset": 16 * 8 * 4,
                        "size": weight_bytes,
                        "dtype": "U32",
                        "shape": [out_dim, packed_cols],
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.scales",
                        "offset": 16 * 8 * 4 + weight_bytes,
                        "size": meta_bytes,
                        "dtype": "BF16",
                        "shape": [out_dim, groups],
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.biases",
                        "offset": 16 * 8 * 4 + weight_bytes + meta_bytes,
                        "size": meta_bytes,
                        "dtype": "BF16",
                        "shape": [out_dim, groups],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 16,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 0,
                "segments": [],
            }
        ),
        encoding="utf-8",
    )

    plan = _auto_prefill_prompt_chunk_plan(
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        prompt_tokens=8,
        start_position=0,
        layers={0},
        dense_layers=None,
        work_dir=None,
        top_k=1,
        max_prompt_batch_mib=1024,
        max_cache_read_mib=1024,
        max_cache_write_mib=1024,
        max_runner_scratch_mib=64,
        prefill_max_stage_mib=1024,
        prefill_max_compact_stage_mib=1024,
        prefill_expert_stage_align_kib=4,
        prefill_stage_disk_margin_mib=0,
        dsa_indexer_types=None,
        dsa_index_topk=None,
        prefill_linear_backend="custom-metal",
        tile_tokens=1,
    )

    assert plan.chunk_tokens == 8
    assert plan.per_token_activation_bytes == (out_dim + in_dim) * 4
    assert plan.max_matrix_scratch_bytes == 4 * 1024 * 1024
    assert total_triplet_bytes > 2 * 1024 * 1024


def write_config(
    path: Path,
    *,
    mixed_layers: bool = False,
    eos_token_id: object | None = None,
) -> None:
    payload = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "moe_intermediate_size": 8,
        "n_routed_experts": 2,
        "n_shared_experts": 0,
        "num_experts_per_tok": 2,
        "num_attention_heads": 2,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 2,
        "v_head_dim": 1,
        "scoring_func": "raw",
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
    }
    if mixed_layers:
        payload["mlp_layer_types"] = ["dense", "sparse"]
    if eos_token_id is not None:
        payload["eos_token_id"] = eos_token_id
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _set_layout_config_sha256(layout_path: Path, digest: str) -> None:
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["config_sha256"] = digest
    layout_path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_token_ids_runs_greedy_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import largerlm.token_generator as token_generator_module

    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)
    decode_mla_key_cache_calls: list[bool] = []
    original_run_decode_layers = token_generator_module.run_decode_layers

    def spy_run_decode_layers(**kwargs):
        decode_mla_key_cache_calls.append(bool(kwargs["mla_key_cache"]))
        return original_run_decode_layers(**kwargs)

    monkeypatch.setattr(
        "largerlm.token_generator.run_decode_layers",
        spy_run_decode_layers,
    )

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0],
        max_new_tokens=2,
        layers={1},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        prefill_ssd_read_gib_per_second=16.0,
        decode_max_routed_read_seconds_per_token=1.0,
        decode_mla_key_cache=True,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (2, 2)
    assert [step.position for step in result.steps] == [0, 1]
    assert result.elapsed_seconds >= 0.0
    assert result.estimated_read_bytes > 0
    assert result.estimated_embedding_read_bytes > 0
    assert result.estimated_expert_read_bytes > 0
    assert result.estimated_logits_read_bytes > 0
    assert result.estimated_read_bytes == (
        result.estimated_embedding_read_bytes
        + result.estimated_expert_read_bytes
        + result.estimated_cache_read_bytes
        + result.estimated_logits_read_bytes
    )
    assert all(step.estimated_read_bytes > 0 for step in result.steps)
    assert all(step.logits_elapsed_seconds >= 0.0 for step in result.steps)
    assert result.steps[0].decode_layers
    assert result.steps[0].decode_layers[0].layer == 1
    assert result.steps[0].decode_layers[0].kind == "moe"
    assert result.steps[0].decode_layers[0].expert_read_bytes == 2 * 192
    assert result.steps[0].decode_layers[0].cache_read_bytes > 0
    assert result.steps[0].decode_layers[0].elapsed_seconds >= 0.0
    assert result.runtime_guard is not None
    assert result.runtime_guard.requested_context_tokens == 3
    assert result.runtime_guard.max_layer_cache_read_bytes == 24
    runtime_budget = result.runtime_guard.layer_budgets[0]
    assert runtime_budget.decoder_mla_key_cache is True
    assert runtime_budget.decoder_mla_key_cache_bytes == 24
    assert decode_mla_key_cache_calls == [True, True]
    decode_actual = result.decode_actual_read_time
    assert decode_actual is not None
    assert decode_actual["source"] == "generation_actual_decode"
    assert decode_actual["decode_step_count"] == 2
    assert decode_actual["decode_read_bytes_per_token"] == 384
    assert decode_actual["planned_decode_routed_read_bytes"] == 768
    assert decode_actual["actual_decode_routed_read_bytes"] == 768
    assert decode_actual["actual_decode_routed_read_bytes_ok"] is True
    assert decode_actual["actual_decode_routed_read_seconds"] == pytest.approx(
        768 / (16 * 1024**3)
    )
    assert decode_actual["decode_max_routed_read_seconds_per_token"] == 1.0
    assert decode_actual["total_decode_max_routed_read_seconds"] == 2.0
    assert decode_actual["total_decode_routed_read_seconds_ok"] is True
    assert not result.runtime_guard.final_logits_budget.metal
    assert (
        result.runtime_guard.live_memory_budget.max_live_working_set_bytes
        == 8192 * 1024**2
    )
    assert result.runtime_guard.live_memory_budget.estimated_live_working_set_bytes > 0


def test_generate_token_ids_rejects_live_working_set_over_cap(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    work_dir = tmp_path / "work"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="live working set"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_live_working_set_mib=0.001,
            echo_runner_output=False,
        )

    assert not work_dir.exists()


def test_generate_token_ids_rechecks_live_memory_during_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)
    calls = {"count": 0}

    def fail_live_memory_recheck(**kwargs):
        calls["count"] += 1
        raise GenerationGuardError("available unified memory fell below reserve")

    monkeypatch.setattr(
        "largerlm.token_generator.check_live_memory_budget",
        fail_live_memory_recheck,
    )

    with pytest.raises(TokenGeneratorError, match="during generation"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=tmp_path / "work",
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            min_free_unified_memory_gib=0.001,
            echo_runner_output=False,
        )

    assert calls["count"] == 1


def test_generate_token_ids_min_free_forces_runtime_guard_when_preflight_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)
    captured: dict[str, object] = {}

    def fail_runtime_guard(**kwargs):
        captured.update(kwargs)
        raise GenerationGuardError("available unified memory fell below reserve")

    monkeypatch.setattr(
        "largerlm.token_generator.check_generation_runtime",
        fail_runtime_guard,
    )

    with pytest.raises(TokenGeneratorError, match="available unified memory"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=tmp_path / "work",
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            preflight_runtime=False,
            min_free_unified_memory_gib=0.001,
            echo_runner_output=False,
        )

    assert captured["min_free_unified_memory_mib"] == pytest.approx(1.024)
    assert not (tmp_path / "work").exists()


def test_generate_token_ids_rechecks_live_memory_before_final_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)
    calls = {"count": 0}

    def staged_live_memory_recheck(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise GenerationGuardError("available unified memory fell below reserve")

    def fail_final_logits(*args, **kwargs):
        raise AssertionError("final logits should not run after guard failure")

    monkeypatch.setattr(
        "largerlm.token_generator.check_live_memory_budget",
        staged_live_memory_recheck,
    )
    monkeypatch.setattr(
        "largerlm.token_generator.compute_final_logits",
        fail_final_logits,
    )

    with pytest.raises(TokenGeneratorError, match="during generation"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=tmp_path / "work",
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            min_free_unified_memory_gib=0.001,
            echo_runner_output=False,
        )

    assert calls["count"] == 2


def test_generate_token_ids_passes_live_memory_guard_to_batch_prefill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)
    captured: dict[str, object] = {}

    def fake_run_prompt_prefill(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_last_hidden_f32_path"]).write_bytes(f32([1.0] * 8))
        return SimpleNamespace(
            chunks=(),
            total_embedding_read_bytes=0,
            total_expert_stage_serial_read_bytes=4096,
            total_expert_stage_unique_requested_bytes=1536,
            total_expert_stage_planned_read_bytes=2048,
            total_expert_stage_waste_bytes=512,
            total_expert_stage_coalesced_savings_bytes=2048,
            total_expert_stage_planned_read_seconds=2048 / (18.25 * 1024**3),
            total_expert_stage_copy_elapsed_seconds=0.125,
            total_expert_stage_copy_throughput_gib_per_second=(
                (2048 / 1024**3) / 0.125
            ),
            prefill_ssd_read_gib_per_second=18.25,
            prefill_max_routed_read_seconds=99.0,
            total_expert_stage_read_seconds_ok=True,
            total_expert_stage_copy_seconds_ok=True,
            prefill_max_stage_raw_ranges=7,
            prefill_max_stage_coalesced_ranges=3,
            total_expert_stage_raw_ranges=5,
            total_expert_stage_coalesced_ranges=2,
            max_expert_stage_raw_ranges=4,
            max_expert_stage_coalesced_ranges=1,
            total_expert_stage_raw_ranges_ok=True,
            total_expert_stage_coalesced_ranges_ok=True,
            total_expert_stage_read_advice_attempted_ranges=3,
            total_expert_stage_read_advice_calls=2,
            total_expert_stage_read_advice_bytes=1536,
            total_expert_stage_read_advice_failures=1,
            total_expert_stage_copy_read_calls=8,
            total_expert_stage_copy_write_calls=8,
            total_expert_stage_copy_average_read_bytes=256.0,
            total_expert_stage_copy_average_write_bytes=256.0,
            total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib={
                "8": 8,
                "16": 4,
                "32": 2,
            },
            total_expert_stage_assignment_read_amplification=0.5,
            total_expert_stage_unique_read_amplification=2048 / 1536,
            max_expert_stage_unique_read_amplification=1.5,
            max_expert_stage_stage_budget_utilization=0.75,
            linear_backend_counts={"mpsgraph-f32": 1, "custom-metal": 2},
            linear_backend_flops={"mpsgraph-f32": 100, "custom-metal": 300},
            linear_backend_elapsed_seconds={
                "mpsgraph-f32": 0.01,
                "custom-metal": 0.03,
            },
            linear_backend_estimated_tflops={
                "mpsgraph-f32": 0.00001,
                "custom-metal": 0.00001,
            },
            linear_backend_component_stats={
                "attention.o_proj": {
                    "linear_backend_counts": {"custom-metal": 1},
                    "linear_backend_flops": {"custom-metal": 300},
                    "linear_backend_elapsed_seconds": {"custom-metal": 0.03},
                    "linear_backend_estimated_tflops": {"custom-metal": 0.00001},
                },
                "moe.router_gate_proj": {
                    "linear_backend_counts": {"mpsgraph-f32": 1},
                    "linear_backend_flops": {"mpsgraph-f32": 100},
                    "linear_backend_elapsed_seconds": {"mpsgraph-f32": 0.01},
                    "linear_backend_estimated_tflops": {"mpsgraph-f32": 0.00001},
                },
            },
            total_linear_estimated_flops=400,
            accelerated_linear_estimated_flops=100,
            custom_linear_estimated_flops=300,
            unsupported_linear_estimated_flops=0,
            accelerated_linear_flop_fraction=0.25,
            prefill_acceleration_coverage={
                "ok": True,
                "accelerated_flop_fraction": 0.25,
                "streamed_routed_expert_estimated_flops": 128,
            },
            prefill_acceleration_frontier={
                "suggested_guard_flags": {
                    "prefill_prompt_chunk_tokens": 4,
                    "argv": ("--prefill-prompt-chunk-tokens", "4"),
                }
            },
        )

    def fake_runtime_guard(**kwargs):
        return SimpleNamespace(
            read_bytes_per_token=0,
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=0,
                max_live_working_set_bytes=None,
                min_available_memory_bytes=0,
            ),
        )

    monkeypatch.setattr(
        "largerlm.token_generator.check_generation_runtime",
        fake_runtime_guard,
    )
    monkeypatch.setattr(
        "largerlm.token_generator.run_prompt_prefill",
        fake_run_prompt_prefill,
    )

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 1],
        max_new_tokens=2,
        layers={1},
        work_dir=tmp_path / "work",
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=2,
        prefill_ssd_read_gib_per_second=18.25,
        prefill_max_routed_read_seconds=99.0,
        prefill_max_stage_raw_ranges=7,
        prefill_max_stage_coalesced_ranges=3,
        prefill_expert_stage_tiling=True,
        prefill_persistent_attention_output_server=True,
        prefill_persistent_shared_expert_server=True,
        prefill_persistent_mla_attention_server=True,
        prefill_persistent_rmsnorm_server=True,
        require_prefill_acceleration=True,
        prefill_min_accelerated_flop_fraction=0.5,
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
        prefill_router_hybrid_margin_threshold=1e-5,
        dsa_index_topk=4,
        preflight_runtime=False,
        max_live_working_set_mib=123.0,
        min_free_unified_memory_gib=0.125,
        echo_runner_output=False,
    )

    assert captured["max_live_working_set_mib"] == 123.0
    assert captured["min_free_unified_memory_gib"] == 0.125
    assert captured["prefill_ssd_read_gib_per_second"] == 18.25
    assert captured["prefill_max_routed_read_seconds"] == 99.0
    assert captured["prefill_min_accelerated_flop_fraction"] == 0.5
    assert captured["router_hybrid_margin_threshold"] == 1e-5
    assert captured["expert_stage_max_raw_ranges"] == 7
    assert captured["expert_stage_max_coalesced_ranges"] == 3
    assert captured["expert_stage_tiling"] is True
    assert captured["persistent_attention_output_server"] is True
    assert captured["persistent_shared_expert_server"] is True
    assert captured["persistent_mla_attention_server"] is True
    assert captured["persistent_rmsnorm_server"] is True
    assert captured["write_dsa_future_cache"] is False
    actual = result.prefill_actual_read_time
    assert actual is not None
    assert actual["source"] == "generation_actual_prefill"
    assert actual["total_expert_stage_serial_read_bytes"] == 4096
    assert actual["total_expert_stage_unique_requested_bytes"] == 1536
    assert actual["total_expert_stage_planned_read_bytes"] == 2048
    assert actual["total_expert_stage_waste_bytes"] == 512
    assert actual["total_expert_stage_coalesced_savings_bytes"] == 2048
    assert actual["total_expert_stage_planned_read_seconds"] == pytest.approx(
        2048 / (18.25 * 1024**3)
    )
    assert actual["prefill_ssd_read_gib_per_second"] == 18.25
    assert actual["prefill_max_routed_read_seconds"] == 99.0
    assert actual["total_expert_stage_read_seconds_ok"] is True
    assert actual["total_expert_stage_copy_seconds_ok"] is True
    assert actual["prefill_max_stage_raw_ranges"] == 7
    assert actual["prefill_max_stage_coalesced_ranges"] == 3
    assert actual["total_expert_stage_raw_ranges"] == 5
    assert actual["total_expert_stage_coalesced_ranges"] == 2
    assert actual["max_expert_stage_raw_ranges"] == 4
    assert actual["max_expert_stage_coalesced_ranges"] == 1
    assert actual["total_expert_stage_raw_ranges_ok"] is True
    assert actual["total_expert_stage_coalesced_ranges_ok"] is True
    assert actual["total_expert_stage_read_advice_attempted_ranges"] == 3
    assert actual["total_expert_stage_read_advice_calls"] == 2
    assert actual["total_expert_stage_read_advice_bytes"] == 1536
    assert actual["total_expert_stage_read_advice_failures"] == 1
    assert actual["total_expert_stage_copy_read_calls"] == 8
    assert actual["total_expert_stage_copy_write_calls"] == 8
    assert actual["total_expert_stage_copy_average_read_bytes"] == 256.0
    assert actual["total_expert_stage_copy_average_write_bytes"] == 256.0
    assert actual[
        "total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib"
    ] == {
        "8": 8,
        "16": 4,
        "32": 2,
    }
    assert actual["total_expert_stage_assignment_read_amplification"] == 0.5
    assert actual["total_expert_stage_unique_read_amplification"] == pytest.approx(
        2048 / 1536
    )
    assert actual["max_expert_stage_unique_read_amplification"] == 1.5
    assert actual["max_expert_stage_stage_budget_utilization"] == 0.75
    assert actual["total_expert_stage_copy_elapsed_seconds"] == 0.125
    assert actual[
        "total_expert_stage_copy_throughput_gib_per_second"
    ] == pytest.approx((2048 / 1024**3) / 0.125)
    actual_coverage = result.prefill_actual_acceleration_coverage
    assert actual_coverage is not None
    assert actual_coverage["source"] == "generation_actual_prefill"
    assert actual_coverage["required"] is True
    assert actual_coverage["min_accelerated_flop_fraction"] == 0.5
    assert actual_coverage["accelerated_flop_fraction"] == 0.25
    assert actual_coverage["streamed_routed_expert_estimated_flops"] == 128
    actual_frontier = result.prefill_actual_acceleration_frontier
    assert actual_frontier is not None
    assert actual_frontier["source"] == "generation_actual_prefill"
    assert actual_frontier["suggested_guard_flags"]["prefill_prompt_chunk_tokens"] == 4
    actual_linear = result.prefill_actual_linear_backend
    assert actual_linear is not None
    assert actual_linear["source"] == "generation_actual_prefill"
    assert actual_linear["configured_backend"] == "auto"
    assert actual_linear["auto_policy"] == {
        "mpsgraph_min_batch_tokens": 128,
        "mpsgraph_min_matrix_dim": 32,
    }
    assert actual_linear["linear_backend_counts"] == {
        "mpsgraph-f32": 1,
        "custom-metal": 2,
    }
    assert actual_linear["linear_backend_flops"] == {
        "mpsgraph-f32": 100,
        "custom-metal": 300,
    }
    assert actual_linear["total_linear_estimated_flops"] == 400
    assert actual_linear["accelerated_linear_flop_fraction"] == 0.25
    components = actual_linear["linear_backend_component_stats"]
    assert components["attention.o_proj"]["linear_backend_counts"] == {
        "custom-metal": 1,
    }
    assert components["attention.o_proj"]["linear_backend_flops"] == {
        "custom-metal": 300,
    }
    assert components["moe.router_gate_proj"]["linear_backend_counts"] == {
        "mpsgraph-f32": 1,
    }


def test_prefill_actual_read_time_reports_actual_copy_seconds_guard() -> None:
    summary = _prefill_actual_read_time_summary(
        SimpleNamespace(
            total_expert_stage_planned_read_bytes=1024,
            total_expert_stage_planned_read_seconds=0.01,
            total_expert_stage_copy_elapsed_seconds=2.0,
            total_expert_stage_copy_throughput_gib_per_second=(
                (1024 / 1024**3) / 2.0
            ),
            prefill_ssd_read_gib_per_second=10.0,
            prefill_max_routed_read_seconds=1.0,
            total_expert_stage_read_seconds_ok=True,
            total_expert_stage_copy_seconds_ok=False,
            expert_stage_io_stage_count=2,
            expert_stage_copy_hotspots=(
                SimpleNamespace(
                    chunk_index=0,
                    layer=19,
                    tile_index=0,
                    batch_tokens=64,
                    selected_experts=(1, 3, 8),
                    selected_expert_count=3,
                    total_assignments=128,
                    raw_range_count=40,
                    coalesced_range_count=24,
                    planned_read_bytes=1024,
                    staged_bytes=1024,
                    unique_requested_bytes=768,
                    waste_bytes=256,
                    copy_chunk_bytes=64 * 1024 * 1024,
                    copy_elapsed_seconds=0.25,
                    copy_throughput_gib_per_second=0.000004,
                    copy_read_calls=24,
                    copy_write_calls=24,
                    copy_average_read_bytes=1024 / 24,
                    stage_budget_utilization=0.5,
                    unique_read_amplification=4 / 3,
                ),
            ),
            expert_stage_range_hotspots=(
                SimpleNamespace(
                    chunk_index=0,
                    layer=7,
                    tile_index=1,
                    raw_range_count=48,
                    coalesced_range_count=32,
                    planned_read_bytes=2048,
                    copy_elapsed_seconds=0.125,
                ),
            ),
        )
    )

    assert summary is not None
    assert summary["total_expert_stage_read_seconds_ok"] is True
    assert summary["total_expert_stage_copy_seconds_ok"] is False
    assert summary["total_expert_stage_copy_elapsed_seconds"] == 2.0
    assert summary["expert_stage_io_stage_count"] == 2
    copy_hotspot = summary["expert_stage_copy_hotspots"][0]
    assert copy_hotspot["layer"] == 19
    assert copy_hotspot["selected_experts"] == [1, 3, 8]
    assert copy_hotspot["coalesced_range_count"] == 24
    assert copy_hotspot["copy_read_calls"] == 24
    range_hotspot = summary["expert_stage_range_hotspots"][0]
    assert range_hotspot["layer"] == 7
    assert range_hotspot["tile_index"] == 1


def test_generate_token_ids_stops_on_any_eos_token_id(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0],
        max_new_tokens=3,
        eos_token_ids=[2, 3],
        layers={1},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (2,)
    assert len(result.steps) == 1


def test_generate_token_ids_cleans_intermediate_work_files(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    work_dir = tmp_path / "work"
    write_fake_runner(runner)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0],
        max_new_tokens=2,
        layers={1},
        work_dir=work_dir,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (2, 2)
    assert not list(work_dir.rglob("*.f32"))
    assert not list(work_dir.glob("pos_*_layers"))


def test_generate_token_ids_cleans_auto_work_dir_on_embedding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)
    auto_root = tmp_path / "auto_generate_failure"

    def fake_mkdtemp(*args, **kwargs) -> str:
        del args, kwargs
        auto_root.mkdir()
        return str(auto_root)

    monkeypatch.setattr("largerlm.token_generator.tempfile.mkdtemp", fake_mkdtemp)

    with pytest.raises(TokenGeneratorError, match="embedding row"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            max_embedding_row_mib=1 / (1024 * 1024),
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            echo_runner_output=False,
        )

    assert not auto_root.exists()


def test_generate_token_ids_rejects_cache_context_overflow(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="exceeds cache context"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1, 2],
            max_new_tokens=2,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
        )


def test_generate_token_ids_runtime_preflight_rejects_cache_read_before_work_dir(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    work_dir = tmp_path / "work"
    missing_runner = tmp_path / "missing-runner"

    with pytest.raises(TokenGeneratorError, match="decoder cache read"):
        generate_token_ids(
            runner_path=missing_runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_cache_read_mib=0.000001,
        )

    assert not work_dir.exists()


def test_generate_token_ids_rejects_decode_routed_read_cap_before_work_dir(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    work_dir = tmp_path / "work"
    missing_runner = tmp_path / "missing-runner"

    with pytest.raises(
        TokenGeneratorError,
        match="decode routed expert read .* bytes/token exceeds cap",
    ):
        generate_token_ids(
            runner_path=missing_runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            preflight_runtime=False,
            decode_max_routed_read_gib_per_token=1e-9,
        )

    assert not work_dir.exists()


def test_generate_token_ids_runtime_preflight_rejects_short_expert_before_work_dir(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    (expert.parent / "layer_001.bin").write_bytes(b"\0")
    work_dir = tmp_path / "work"
    missing_runner = tmp_path / "missing-runner"

    with pytest.raises(TokenGeneratorError, match="expert layer file"):
        generate_token_ids(
            runner_path=missing_runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
        )

    assert not work_dir.exists()


def test_generate_token_ids_runtime_preflight_rejects_short_resident_before_work_dir(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    (resident.parent / "resident.bin").write_bytes(b"\0")
    work_dir = tmp_path / "work"
    missing_runner = tmp_path / "missing-runner"

    with pytest.raises(TokenGeneratorError, match="resident weight file"):
        generate_token_ids(
            runner_path=missing_runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
        )

    assert not work_dir.exists()


def test_generate_token_ids_rejects_dsa_index_cache_by_default(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    add_dsa_index_segment(cache_layout, cache_file)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="DSA/indexer"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            echo_runner_output=False,
        )


def test_generate_token_ids_can_explicitly_allow_missing_dsa_indexer(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    add_dsa_index_segment(cache_layout, cache_file)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0],
        max_new_tokens=2,
        layers={1},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        allow_missing_dsa_indexer=True,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (2,)
    assert result.runtime_guard is not None
    assert result.runtime_guard.dsa_index_layers == (0,)
    assert result.runtime_guard.dsa_index_cache_bytes == 32
    assert result.runtime_guard.allow_missing_dsa_indexer is True


def test_generate_token_ids_runs_dsa_decode_when_schedule_present(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path, include_dsa=True)
    runner = tmp_path / "runner.py"
    work_dir = tmp_path / "work"
    write_fake_runner(runner)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0],
        max_new_tokens=2,
        layers={1},
        work_dir=work_dir,
        keep_work_dir=True,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["none", "full"],
        dsa_index_topk=1,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (2, 2)
    assert result.runtime_guard is not None
    assert result.runtime_guard.dsa_indexer_runtime is True
    budget = result.runtime_guard.layer_budgets[0]
    assert budget.dsa_indexer_mode == "full"
    assert budget.dsa_index_head_dim == 2
    assert budget.dsa_index_cache_read_bytes > 0
    assert budget.decoder_mla_cache_read_bytes > 0
    assert result.runtime_guard.max_layer_cache_read_bytes > 0
    topk_files = list(work_dir.rglob("dsa_topk.u32"))
    assert len(topk_files) >= 1
    assert all(path.stat().st_size == 8 for path in topk_files)


def test_generate_token_ids_skips_dsa_cache_when_request_fits_topk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path, include_dsa=True)
    runner = tmp_path / "runner.py"
    work_dir = tmp_path / "work"
    write_fake_runner(runner)

    def fail_dsa_cache_write(**kwargs: object) -> object:
        raise AssertionError("short requests should not write DSA future cache")

    monkeypatch.setattr(
        "largerlm.prefill_execute.run_dsa_indexer_batch",
        fail_dsa_cache_write,
    )

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0],
        max_new_tokens=2,
        layers={1},
        work_dir=work_dir,
        keep_work_dir=True,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        router_score="raw",
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["none", "full"],
        dsa_index_topk=4,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (2, 2)
    assert not list(work_dir.rglob("dsa_topk.u32"))
    assert all(
        record.dsa_indexer_mode == "none"
        for step in result.steps
        for record in step.decode_layers
    )
    assert all(
        "--run-mla-attention-indexed-batch" not in record.command
        for step in result.steps
        for record in step.decode_layers
    )


def test_generate_token_ids_keeps_dsa_cache_when_request_exceeds_topk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path, include_dsa=True)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    def fail_dsa_cache_write(**kwargs: object) -> object:
        raise AssertionError("longer requests still need DSA future cache")

    monkeypatch.setattr(
        "largerlm.prefill_execute.run_dsa_indexer_batch",
        fail_dsa_cache_write,
    )

    with pytest.raises(AssertionError, match="longer requests still need"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=3,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            dsa_indexer_types=["none", "full"],
            dsa_index_topk=2,
            dsa_index_n_heads=1,
            dsa_qk_rope_dim=2,
            echo_runner_output=False,
        )


def test_generate_token_ids_rejects_invalid_dsa_schedule(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path, include_dsa=True)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="invalid DSA indexer_type"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            dsa_indexer_types=["none", "typo"],
            dsa_index_topk=2,
            dsa_index_n_heads=1,
            echo_runner_output=False,
        )


def test_generate_token_ids_rejects_dsa_schedule_without_full_layer(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path, include_dsa=True)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="no full DSA indexer"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            dsa_indexer_types=["none", "none"],
            dsa_index_topk=2,
            dsa_index_n_heads=1,
            echo_runner_output=False,
        )


def test_generate_token_ids_rejects_selected_shared_dsa_without_selected_full(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path, include_dsa=True)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="selected DSA layer 1 is shared"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            router_score="raw",
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            dsa_indexer_types=["full", "shared"],
            dsa_index_topk=2,
            dsa_index_n_heads=1,
            dsa_index_head_dim=2,
            echo_runner_output=False,
        )


def test_generate_token_ids_rejects_sampling_without_candidates(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    write_fake_runner(runner)

    with pytest.raises(TokenGeneratorError, match="logits-top-k"):
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            max_new_tokens=1,
            layers={1},
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            sampling_temperature=1.0,
            logits_top_k=1,
        )


def test_generate_token_ids_cli_derives_from_config(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    config = tmp_path / "config.json"
    write_fake_runner(runner)
    write_config(config)

    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--max-k",
            "2",
            "--logits-top-k",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
        ]
    )

    assert status == 0


def test_generate_token_ids_cli_rejects_layout_config_hash_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    config = tmp_path / "config.json"
    write_config(config)
    digest = config_sha256(config)
    assert digest is not None
    _set_layout_config_sha256(expert, digest)
    _set_layout_config_sha256(resident, digest)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["routed_scaling_factor"] = 2.5
    config.write_text(json.dumps(payload), encoding="utf-8")

    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            "unused-runner",
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    assert "config_sha256 mismatch" in capsys.readouterr().err


def test_generate_token_ids_cli_derives_router_defaults_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    config = tmp_path / "config.json"
    write_config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload.update(
        {
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "scoring_func": "sigmoid",
            "n_group": 1,
            "topk_group": 1,
        }
    )
    config.write_text(json.dumps(payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            "unused-runner",
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["router_score"] == "sigmoid"
    assert captured["routed_scaling_factor"] == 2.5
    assert captured["norm_topk_prob"] is True
    assert captured["no_norm_topk_prob"] is False
    assert captured["router_n_group"] == 1
    assert captured["router_topk_group"] == 1

    captured.clear()
    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            "unused-runner",
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--router-score",
            "raw",
            "--routed-scaling-factor",
            "1.25",
            "--no-norm-topk-prob",
            "--router-n-group",
            "2",
            "--router-topk-group",
            "2",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["router_score"] == "raw"
    assert captured["routed_scaling_factor"] == 1.25
    assert captured["norm_topk_prob"] is False
    assert captured["no_norm_topk_prob"] is True
    assert captured["router_n_group"] == 2
    assert captured["router_topk_group"] == 2


def test_generate_token_ids_cli_derives_eos_list_from_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    config = tmp_path / "config.json"
    write_fake_runner(runner)
    write_config(config, eos_token_id=[2, 3])

    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "3",
            "--max-k",
            "2",
            "--logits-top-k",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generated_token_ids"] == [2]


def test_generate_token_ids_cli_derives_dense_layers_from_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file = write_fixture(tmp_path)
    runner = tmp_path / "runner.py"
    config = tmp_path / "config.json"
    write_fake_runner(runner)
    write_config(config, mixed_layers=True)

    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--max-k",
            "2",
            "--logits-top-k",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime_guard"]["dense_layers"] == [0]
    assert [item["layer_kind"] for item in payload["runtime_guard"]["layer_budgets"]] == [
        "dense",
        "moe",
    ]
