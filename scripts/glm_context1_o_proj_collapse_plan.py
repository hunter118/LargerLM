#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "largerlm.context1_o_proj_collapse_plan.v1"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _tensors_by_name(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise SystemExit("resident layout is missing a tensors list")
    out: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            out[name] = item
    return out


def _shape(tensor: dict[str, Any], rank: int, label: str) -> list[int]:
    raw = tensor.get("shape")
    if not isinstance(raw, list) or len(raw) != rank:
        raise ValueError(f"{label} must have rank {rank}")
    dims: list[int] = []
    for dim in raw:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{label} has invalid shape {raw!r}")
        dims.append(dim)
    return dims


def _require_dtype(tensor: dict[str, Any], dtype: str, label: str) -> None:
    if tensor.get("dtype") != dtype:
        raise ValueError(f"{label} must be {dtype}, got {tensor.get('dtype')!r}")


def _companion(tensors: dict[str, dict[str, Any]], weight_name: str) -> dict[str, Any]:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"{weight_name} is not a .weight tensor")
    scale_name = weight_name[: -len(".weight")] + ".scales"
    try:
        return tensors[scale_name]
    except KeyError as exc:
        raise ValueError(f"missing companion scale tensor {scale_name}") from exc


def _mxfp4_matrix_dims(
    tensors: dict[str, dict[str, Any]],
    weight: dict[str, Any],
    label: str,
) -> tuple[int, int, int, int]:
    _require_dtype(weight, "U32", label)
    scales = _companion(tensors, str(weight["name"]))
    _require_dtype(scales, "U8", label + ".scales")
    out_dim, packed_cols = _shape(weight, 2, label)
    scale_out, scale_groups = _shape(scales, 2, label + ".scales")
    if scale_out != out_dim:
        raise ValueError(f"{label} scales row count mismatch")
    in_dim = packed_cols * 8
    if scale_groups <= 0 or in_dim % scale_groups != 0:
        raise ValueError(f"{label} scales do not divide logical input dim")
    return out_dim, in_dim, scale_groups, in_dim // scale_groups


def _mxfp4_tensor3d_dims(
    tensors: dict[str, dict[str, Any]],
    weight: dict[str, Any],
    label: str,
) -> tuple[int, int, int, int, int]:
    _require_dtype(weight, "U32", label)
    scales = _companion(tensors, str(weight["name"]))
    _require_dtype(scales, "U8", label + ".scales")
    dim0, dim1, packed_dim2 = _shape(weight, 3, label)
    scale0, scale1, scale_groups = _shape(scales, 3, label + ".scales")
    if (scale0, scale1) != (dim0, dim1):
        raise ValueError(f"{label} scales prefix dims mismatch")
    dim2 = packed_dim2 * 8
    if scale_groups <= 0 or dim2 % scale_groups != 0:
        raise ValueError(f"{label} scales do not divide logical dim2")
    return dim0, dim1, dim2, scale_groups, dim2 // scale_groups


def _tensor_size(tensor: dict[str, Any]) -> int:
    value = tensor.get("size")
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{tensor.get('name')} has invalid size")
    return value


def _layer_ids(tensors: dict[str, dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for name in tensors:
        match = LAYER_RE.match(name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(set(ids))


def _telemetry_attn_bytes(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    values = payload.get("decode_attn_output_bytes_read")
    tokens = payload.get("generated_token_ids")
    if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
        raise SystemExit(f"{path} does not contain decode_attn_output_bytes_read")
    token_count = len(tokens) if isinstance(tokens, list) else len(values)
    total = sum(values)
    return {
        "path": str(path),
        "token_count": token_count,
        "attn_output_bytes_total": total,
        "attn_output_bytes_per_token": total / token_count if token_count else None,
    }


def build_plan(
    prepared_dir: Path,
    *,
    telemetry_json: Path | None,
    target_dtype: str,
    max_cache_mib: float,
) -> dict[str, Any]:
    manifest_path = prepared_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    resident_layout_rel = manifest.get("resident_layout", "resident/layout.json")
    if not isinstance(resident_layout_rel, str):
        raise SystemExit("manifest resident_layout must be a string")
    resident_layout_path = prepared_dir / resident_layout_rel
    layout = _read_json(resident_layout_path)
    tensors = _tensors_by_name(layout)

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    hidden_dim: int | None = None
    attn_value_dim: int | None = None
    kv_lora_dim: int | None = None
    num_heads: int | None = None
    v_head_dim: int | None = None
    qk_nope_dim: int | None = None

    for layer in _layer_ids(tensors):
        prefix = f"model.layers.{layer}.self_attn"
        try:
            o_weight = tensors[f"{prefix}.o_proj.weight"]
            o_scales = _companion(tensors, str(o_weight["name"]))
            out_dim, in_dim, o_scale_groups, o_group_size = _mxfp4_matrix_dims(
                tensors, o_weight, f"layer {layer} o_proj"
            )
            embed = tensors[f"{prefix}.embed_q.weight"]
            unembed = tensors[f"{prefix}.unembed_out.weight"]
            heads, embed_kv, embed_qk, _embed_groups, _embed_group = (
                _mxfp4_tensor3d_dims(tensors, embed, f"layer {layer} embed_q")
            )
            u_heads, u_value, u_kv, _u_groups, _u_group = _mxfp4_tensor3d_dims(
                tensors, unembed, f"layer {layer} unembed_out"
            )
            if heads != u_heads or embed_kv != u_kv:
                raise ValueError("embed_q/unembed_out logical dims mismatch")
            if in_dim != heads * u_value:
                raise ValueError(
                    f"o_proj input {in_dim} does not match heads*v_head {heads * u_value}"
                )
            if hidden_dim is None:
                hidden_dim = out_dim
                attn_value_dim = in_dim
                kv_lora_dim = u_kv
                num_heads = heads
                v_head_dim = u_value
                qk_nope_dim = embed_qk
            elif (
                hidden_dim,
                attn_value_dim,
                kv_lora_dim,
                num_heads,
                v_head_dim,
                qk_nope_dim,
            ) != (out_dim, in_dim, u_kv, heads, u_value, embed_qk):
                raise ValueError("layer dims differ; mixed-shape collapse is not supported yet")
            supported.append(
                {
                    "layer": layer,
                    "o_proj_storage_bytes": _tensor_size(o_weight)
                    + _tensor_size(o_scales),
                    "unembed_out_storage_bytes": _tensor_size(unembed)
                    + _tensor_size(_companion(tensors, str(unembed["name"]))),
                    "o_proj_group_size": o_group_size,
                    "o_proj_scale_groups": o_scale_groups,
                }
            )
        except (KeyError, ValueError) as exc:
            unsupported.append({"layer": layer, "reason": str(exc)})

    if not supported:
        raise SystemExit("no collapsible attention output layers found")

    assert hidden_dim is not None
    assert attn_value_dim is not None
    assert kv_lora_dim is not None
    assert num_heads is not None
    assert v_head_dim is not None
    assert qk_nope_dim is not None

    layer_count = len(supported)
    collapsed_f32_per_layer = hidden_dim * kv_lora_dim * 4
    collapsed_bf16_per_layer = hidden_dim * kv_lora_dim * 2
    current_o_proj_per_token = sum(item["o_proj_storage_bytes"] for item in supported)
    full_kv_b_f32_per_layer = num_heads * (qk_nope_dim + v_head_dim) * kv_lora_dim * 4
    value_kv_b_f32_per_layer = num_heads * v_head_dim * kv_lora_dim * 4
    chosen_per_layer = (
        collapsed_bf16_per_layer if target_dtype == "bf16" else collapsed_f32_per_layer
    )
    chosen_total = chosen_per_layer * layer_count
    max_cache_bytes = int(max_cache_mib * 1024 * 1024)

    telemetry = _telemetry_attn_bytes(telemetry_json)
    plan = {
        "schema": SCHEMA,
        "prepared_dir": str(prepared_dir),
        "manifest_path": str(manifest_path),
        "resident_layout_path": str(resident_layout_path),
        "layers_analyzed": len(_layer_ids(tensors)),
        "layers_supported": layer_count,
        "unsupported_layers": unsupported,
        "context_limit": "decode/context_length_1_only",
        "dims": {
            "hidden_dim": hidden_dim,
            "attention_value_dim": attn_value_dim,
            "num_heads": num_heads,
            "v_head_dim": v_head_dim,
            "qk_nope_dim": qk_nope_dim,
            "kv_lora_dim": kv_lora_dim,
        },
        "bytes": {
            "current_o_proj_storage_per_token": current_o_proj_per_token,
            "collapsed_f32_per_layer": collapsed_f32_per_layer,
            "collapsed_f32_total": collapsed_f32_per_layer * layer_count,
            "collapsed_bf16_per_layer": collapsed_bf16_per_layer,
            "collapsed_bf16_total": collapsed_bf16_per_layer * layer_count,
            "existing_full_mla_kv_b_f32_total": full_kv_b_f32_per_layer * layer_count,
            "existing_value_mla_kv_b_f32_total": value_kv_b_f32_per_layer * layer_count,
            "chosen_cache_dtype": target_dtype,
            "chosen_cache_total": chosen_total,
            "chosen_cache_fits_limit": chosen_total <= max_cache_bytes,
            "max_cache_bytes": max_cache_bytes,
            "staged_per_token_savings_vs_o_proj": max(
                0, current_o_proj_per_token - chosen_total
            ),
        },
        "build_work": {
            "fma_per_layer": hidden_dim * attn_value_dim * kv_lora_dim,
            "fma_total": hidden_dim * attn_value_dim * kv_lora_dim * layer_count,
            "two_op_flops_total": 2 * hidden_dim * attn_value_dim * kv_lora_dim * layer_count,
            "requires_offline_builder": True,
            "recommended_builder": (
                "Metal/Accelerate tiled per-layer job; do not use a Python/CPU nested loop"
            ),
        },
        "runtime_plan": {
            "replace_kernels": [
                "glm_mla_attention_context1_f32",
                "glm_mxfp4_matvec_add_gs32_simd for self_attn.o_proj",
            ],
            "new_hot_kernel": "collapsed_bf16_or_f32_matvec_add_from_current_kv_a",
            "can_skip_rope_split_for_context1": True,
            "can_reduce_or_replace_mla_kv_b_f32_cache": True,
            "production_default": False,
        },
        "telemetry_comparison": telemetry,
    }
    if telemetry and telemetry.get("attn_output_bytes_per_token"):
        observed = float(telemetry["attn_output_bytes_per_token"])
        plan["telemetry_comparison"]["chosen_cache_vs_observed_ratio"] = (
            chosen_total / observed
        )
        plan["telemetry_comparison"]["chosen_cache_savings_per_token"] = max(
            0.0, observed - chosen_total
        )
    return plan


def _format_mib(value: int | float) -> str:
    return f"{float(value) / 1024 / 1024:.1f} MiB"


def print_summary(plan: dict[str, Any]) -> None:
    dims = plan["dims"]
    bytes_ = plan["bytes"]
    print("GLM context=1 o_proj collapse plan")
    print(f"  prepared:             {plan['prepared_dir']}")
    print(f"  supported layers:     {plan['layers_supported']}/{plan['layers_analyzed']}")
    print(
        "  dims:                 "
        f"hidden={dims['hidden_dim']} value={dims['attention_value_dim']} "
        f"kv_lora={dims['kv_lora_dim']}"
    )
    print(
        "  current o_proj read:  "
        f"{_format_mib(bytes_['current_o_proj_storage_per_token'])}/token"
    )
    print(
        "  collapsed bf16:       "
        f"{_format_mib(bytes_['collapsed_bf16_total'])} total"
    )
    print(
        "  collapsed f32:        "
        f"{_format_mib(bytes_['collapsed_f32_total'])} total"
    )
    print(
        "  existing KV-B f32:    "
        f"{_format_mib(bytes_['existing_full_mla_kv_b_f32_total'])} total"
    )
    print(
        "  chosen cache:         "
        f"{bytes_['chosen_cache_dtype']} {_format_mib(bytes_['chosen_cache_total'])} "
        f"fits={bytes_['chosen_cache_fits_limit']}"
    )
    telemetry = plan.get("telemetry_comparison")
    if isinstance(telemetry, dict) and telemetry.get("attn_output_bytes_per_token"):
        print(
            "  observed o_proj read: "
            f"{_format_mib(telemetry['attn_output_bytes_per_token'])}/token"
        )
        print(
            "  observed ratio:       "
            f"{telemetry['chosen_cache_vs_observed_ratio']:.3f}"
        )
    print(
        "  build work:           "
        f"{plan['build_work']['fma_total'] / 1e12:.2f}T FMA "
        "(offline, resumable builder required)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan a context=1 GLM o_proj*B_v collapsed attention-output cache."
    )
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--telemetry-json", type=Path, default=None)
    parser.add_argument("--target-dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--max-cache-mib", type=float, default=2048.0)
    parser.add_argument("--write-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    plan = build_plan(
        args.prepared_dir,
        telemetry_json=args.telemetry_json,
        target_dtype=args.target_dtype,
        max_cache_mib=args.max_cache_mib,
    )
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print_summary(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
