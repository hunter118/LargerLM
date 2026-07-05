from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "glm_context1_o_proj_collapse_plan.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "glm_context1_o_proj_collapse_plan", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
planner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(planner)


def _add_tensor(
    tensors: list[dict[str, object]],
    name: str,
    dtype: str,
    shape: list[int],
    size: int,
) -> None:
    tensors.append(
        {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "offset": sum(int(item["size"]) for item in tensors),
            "size": size,
        }
    )


def _write_prepared(tmp_path: Path, *, layers: int = 2) -> Path:
    prepared = tmp_path / "prepared"
    resident = prepared / "resident"
    resident.mkdir(parents=True)
    tensors: list[dict[str, object]] = []
    for layer in range(layers):
        prefix = f"model.layers.{layer}.self_attn"
        _add_tensor(tensors, f"{prefix}.o_proj.weight", "U32", [4, 1], 16)
        _add_tensor(tensors, f"{prefix}.o_proj.scales", "U8", [4, 1], 4)
        _add_tensor(tensors, f"{prefix}.embed_q.weight", "U32", [2, 8, 1], 64)
        _add_tensor(tensors, f"{prefix}.embed_q.scales", "U8", [2, 8, 1], 16)
        _add_tensor(tensors, f"{prefix}.unembed_out.weight", "U32", [2, 4, 1], 32)
        _add_tensor(tensors, f"{prefix}.unembed_out.scales", "U8", [2, 4, 1], 8)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "weight_file": "resident.bin",
                "total_bytes": sum(int(item["size"]) for item in tensors),
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps({"resident_layout": "resident/layout.json"}),
        encoding="utf-8",
    )
    return prepared


def test_build_plan_estimates_collapsed_cache(tmp_path: Path) -> None:
    prepared = _write_prepared(tmp_path, layers=2)

    plan = planner.build_plan(
        prepared,
        telemetry_json=None,
        target_dtype="bf16",
        max_cache_mib=1,
    )

    assert plan["schema"] == planner.SCHEMA
    assert plan["layers_supported"] == 2
    assert plan["dims"] == {
        "hidden_dim": 4,
        "attention_value_dim": 8,
        "num_heads": 2,
        "v_head_dim": 4,
        "qk_nope_dim": 8,
        "kv_lora_dim": 8,
    }
    assert plan["bytes"]["current_o_proj_storage_per_token"] == 40
    assert plan["bytes"]["collapsed_bf16_total"] == 128
    assert plan["bytes"]["collapsed_f32_total"] == 256
    assert plan["bytes"]["existing_full_mla_kv_b_f32_total"] == 1536
    assert plan["bytes"]["existing_value_mla_kv_b_f32_total"] == 512
    assert plan["bytes"]["chosen_cache_fits_limit"] is True
    assert plan["build_work"]["fma_total"] == 512


def test_build_plan_compares_telemetry(tmp_path: Path) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    telemetry = tmp_path / "telemetry.json"
    telemetry.write_text(
        json.dumps(
                {
                    "generated_token_ids": [1, 2],
                    "decode_attn_output_bytes_read": [100, 100],
                }
        ),
        encoding="utf-8",
    )

    plan = planner.build_plan(
        prepared,
        telemetry_json=telemetry,
        target_dtype="bf16",
        max_cache_mib=1,
    )

    comparison = plan["telemetry_comparison"]
    assert comparison["attn_output_bytes_per_token"] == 100
    assert comparison["chosen_cache_vs_observed_ratio"] == 0.64
    assert comparison["chosen_cache_savings_per_token"] == 36
