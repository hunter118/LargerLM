from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from largerlm.latent_value_collapse import build_latent_value_collapse_plan


SCALE_E8M0_ONE = 127


def _pack8(code: int) -> bytes:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def _write_prepared(
    tmp_path: Path,
    *,
    layers: int = 2,
    heads: int = 2,
    hidden_dim: int = 2,
    kv_lora_dim: int = 8,
    v_head_dim: int = 8,
) -> Path:
    prepared = tmp_path / "prepared"
    resident = prepared / "resident"
    resident.mkdir(parents=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, dtype: str, shape: list[int], data: bytes) -> None:
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": "attention",
            }
        )
        payload.extend(data)

    value_dim = heads * v_head_dim
    value_groups = value_dim // 8
    qk_nope_groups = 1
    kv_groups = kv_lora_dim // 8
    for layer in range(layers):
        prefix = f"model.layers.{layer}.self_attn"
        add(
            f"{prefix}.o_proj.weight",
            "U32",
            [hidden_dim, value_groups],
            _pack8(2) * hidden_dim * value_groups,
        )
        add(
            f"{prefix}.o_proj.scales",
            "U8",
            [hidden_dim, value_groups],
            bytes([SCALE_E8M0_ONE]) * hidden_dim * value_groups,
        )
        add(
            f"{prefix}.embed_q.weight",
            "U32",
            [heads, kv_lora_dim, qk_nope_groups],
            _pack8(2) * heads * kv_lora_dim * qk_nope_groups,
        )
        add(
            f"{prefix}.embed_q.scales",
            "U8",
            [heads, kv_lora_dim, qk_nope_groups],
            bytes([SCALE_E8M0_ONE]) * heads * kv_lora_dim * qk_nope_groups,
        )
        add(
            f"{prefix}.unembed_out.weight",
            "U32",
            [heads, v_head_dim, kv_groups],
            _pack8(2) * heads * v_head_dim * kv_groups,
        )
        add(
            f"{prefix}.unembed_out.scales",
            "U8",
            [heads, v_head_dim, kv_groups],
            bytes([SCALE_E8M0_ONE]) * heads * v_head_dim * kv_groups,
        )

    (resident / "resident.bin").write_bytes(bytes(payload))
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps({"version": 1, "resident_layout": "resident/layout.json"}),
        encoding="utf-8",
    )
    return prepared


def test_latent_value_collapse_plan_rejects_exact_per_head_cache_shape(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path)

    plan = build_latent_value_collapse_plan(prepared)

    assert plan.layer_count == 2
    assert plan.num_heads == 2
    assert plan.hidden_dim == 2
    assert plan.attention_value_dim == 16
    assert plan.kv_lora_dim == 8
    assert plan.current_o_proj_bytes_per_token == 40
    assert plan.context1_cache_bytes == 64
    assert plan.context1_cache_read_ratio == pytest.approx(64 / 40)
    assert plan.exact_per_head_cache_bytes == 128
    assert plan.exact_per_head_cache_read_ratio == pytest.approx(128 / 40)
    assert plan.exact_per_head_int4_floor_bytes == 32
    assert plan.exact_per_head_int4_floor_read_ratio == pytest.approx(32 / 40)
    assert plan.current_o_proj_runtime_fma_per_token == 64
    assert plan.context1_runtime_fma_per_token == 32
    assert plan.context1_runtime_fma_ratio == pytest.approx(0.5)
    assert plan.exact_per_head_runtime_fma_per_token == 64
    assert plan.exact_per_head_runtime_fma_ratio == pytest.approx(1.0)
    assert plan.break_even_shared_head_groups == 0
    assert plan.independent_attention_head_groups == 2
    assert plan.exact_all_context_cache_recommended is False
    assert plan.decision.startswith("exact_all_context_per_head_cache_not_viable")

    report = plan.to_report()
    assert report["schema"] == "largerlm.latent_value_collapse_plan.v1"
    assert report["ratios"]["exact_per_head_cache_read"] == pytest.approx(128 / 40)
    assert report["constraints"]["exact_all_context_cache_recommended"] is False
