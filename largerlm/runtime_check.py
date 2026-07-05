from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .resident_affine import (
    ResidentAffineLayoutError,
    resident_affine_int4_layout_info,
    resident_mxfp4_layout_info,
)


class RuntimeCheckError(RuntimeError):
    """Raised when a packed runtime layout would exceed configured limits."""


@dataclass(frozen=True)
class LayerRuntimeBudget:
    expert_layout_path: Path
    resident_layout_path: Path
    layer: int
    layer_kind: str
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    top_k: int
    expert_slot_bytes: int
    aligned_slot_bytes: int
    router_bytes: int
    router_logits_bytes: int
    read_bytes_per_token: int
    include_shared_expert: bool
    shared_intermediate_dim: int | None
    shared_max_matrix_bytes: int
    shared_max_aligned_matrix_bytes: int
    include_attention_projections: bool
    attention_q_lora_dim: int | None
    attention_q_output_dim: int | None
    attention_kv_lora_dim: int | None
    attention_kv_rope_dim: int | None
    attention_kv_output_dim: int | None
    attention_read_bytes_per_token: int
    attention_max_matrix_bytes: int
    attention_max_aligned_matrix_bytes: int
    attention_projection_peak_bytes: int
    include_decoder_layer: bool
    decoder_context_length: int | None
    decoder_num_heads: int | None
    decoder_qk_nope_dim: int | None
    decoder_rope_dim: int | None
    decoder_v_head_dim: int | None
    decoder_mla_key_cache: bool
    decoder_mla_key_cache_bytes: int
    dsa_indexer_mode: str
    dsa_index_topk: int | None
    dsa_index_head_dim: int | None
    dsa_index_cache_read_bytes: int
    decoder_mla_cache_read_bytes: int
    decoder_cache_read_bytes: int
    decoder_cache_f32_bytes: int
    decoder_mla_attention_peak_bytes: int
    decoder_attention_output_peak_bytes: int
    router_stage_peak_bytes: int
    moe_stage_peak_bytes: int
    estimated_peak_bytes: int
    max_k: int
    max_slot_bytes: int
    max_router_bytes: int
    max_resident_matrix_bytes: int
    max_cache_read_bytes: int
    max_runner_scratch_bytes: int


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeCheckError(f"failed to read layout {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeCheckError(f"failed to parse layout {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeCheckError(f"layout {p} must be a JSON object")
    return payload


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_int(value: object) -> bool:
    return type(value) is int


def _require_int(value: object, label: str) -> int:
    if not _is_int(value):
        raise RuntimeCheckError(f"{label} must be an integer")
    return int(value)


def _int_field(payload: dict[str, Any], field: str, label: str) -> int:
    return _require_int(payload.get(field), f"{label} {field}")


def _find_layer(layout: dict[str, Any], layer_id: int) -> dict[str, Any]:
    layers = layout.get("layers")
    if not isinstance(layers, list):
        raise RuntimeCheckError("expert layout missing layers array")
    for item in layers:
        if (
            isinstance(item, dict)
            and _is_int(item.get("layer"))
            and item.get("layer") == layer_id
        ):
            return item
    raise RuntimeCheckError(f"layer {layer_id} not found in expert layout")


def _component(layer: dict[str, Any], name: str) -> dict[str, Any]:
    components = layer.get("components")
    if not isinstance(components, list):
        raise RuntimeCheckError("expert layer missing components array")
    for item in components:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    raise RuntimeCheckError(f"expert layout missing component {name}")


def _shape2(component: dict[str, Any], name: str) -> tuple[int, int]:
    shape = component.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise RuntimeCheckError(f"component {name} must have a 2-D integer shape")
    return int(shape[0]), int(shape[1])


def _contains_layer(name: str, layer_id: int) -> bool:
    return f".layers.{layer_id}." in name


def _find_router(resident_layout: dict[str, Any], layer_id: int) -> dict[str, Any]:
    tensors = resident_layout.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeCheckError("resident layout missing tensors array")
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name_ok = name.endswith(".gate.weight") or ".mlp.gate.weight" in name
        if name_ok and _contains_layer(name, layer_id):
            return item
    raise RuntimeCheckError(f"router tensor for layer {layer_id} not found")


def _router_dims(
    resident_layout: dict[str, Any],
    router: dict[str, Any],
) -> tuple[int, int, int, int]:
    try:
        mxfp4 = resident_mxfp4_layout_info(resident_layout, router)
    except ResidentAffineLayoutError as exc:
        raise RuntimeCheckError(str(exc)) from exc
    if mxfp4 is not None:
        return (
            mxfp4.out_dim,
            mxfp4.in_dim,
            mxfp4.total_bytes,
            _align_up(mxfp4.total_bytes, 2 * 1024 * 1024),
        )
    try:
        affine = resident_affine_int4_layout_info(resident_layout, router)
    except ResidentAffineLayoutError as exc:
        raise RuntimeCheckError(str(exc)) from exc
    if affine is not None:
        return (
            affine.out_dim,
            affine.in_dim,
            affine.total_bytes,
            _align_up(affine.total_bytes, 2 * 1024 * 1024),
        )
    shape = router.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise RuntimeCheckError("router tensor must have shape [num_experts, hidden_dim]")
    size = _int_field(router, "size", "router tensor")
    return int(shape[0]), int(shape[1]), size, size


def _find_shared_matrix(
    resident_layout: dict[str, Any],
    layer_id: int,
    component: str,
) -> dict[str, Any]:
    tensors = resident_layout.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeCheckError("resident layout missing tensors array")
    suffix = f".{component}.weight"
    for family in (".shared_experts.", ".shared_expert."):
        for item in tensors:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str):
                continue
            if _contains_layer(name, layer_id) and family in name and name.endswith(suffix):
                return item
    raise RuntimeCheckError(f"shared expert component {component}.weight not found")


def _find_resident_tensor_by_suffix(
    resident_layout: dict[str, Any],
    layer_id: int,
    suffix: str,
) -> dict[str, Any]:
    tensors = resident_layout.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeCheckError("resident layout missing tensors array")
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        if _contains_layer(name, layer_id) and name.endswith(suffix):
            return item
    raise RuntimeCheckError(f"resident tensor {suffix} for layer {layer_id} not found")


def _find_resident_tensor_by_suffix_optional(
    resident_layout: dict[str, Any],
    layer_id: int,
    suffix: str,
) -> dict[str, Any] | None:
    try:
        return _find_resident_tensor_by_suffix(resident_layout, layer_id, suffix)
    except RuntimeCheckError:
        return None


def _find_resident_tensor_by_name(
    resident_layout: dict[str, Any],
    target_name: str,
) -> dict[str, Any] | None:
    tensors = resident_layout.get("tensors")
    if not isinstance(tensors, list):
        raise RuntimeCheckError("resident layout missing tensors array")
    for item in tensors:
        if not isinstance(item, dict):
            continue
        if item.get("name") == target_name:
            return item
    return None


def _find_dense_mlp_matrix(
    resident_layout: dict[str, Any],
    layer_id: int,
    component: str,
) -> dict[str, Any]:
    suffixes = (
        f".mlp.{component}.weight",
        f".mlp.switch_mlp.{component}.weight",
        f".switch_mlp.{component}.weight",
    )
    for suffix in suffixes:
        try:
            return _find_resident_tensor_by_suffix(resident_layout, layer_id, suffix)
        except RuntimeCheckError:
            pass
    raise RuntimeCheckError(
        f"dense MLP tensor {component}.weight for layer {layer_id} not found"
    )


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"F32", "float32"}:
        return 4
    if dtype in {"BF16", "bfloat16", "F16", "float16"}:
        return 2
    return 0


def _shape3(tensor: dict[str, Any], label: str) -> tuple[int, int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or not all(_is_int(dim) and int(dim) > 0 for dim in shape)
    ):
        raise RuntimeCheckError(f"{label} must have a 3-D positive integer shape")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _tensor3d_dims(
    resident_layout: dict[str, Any],
    tensor: dict[str, Any],
    label: str,
) -> tuple[int, int, int, int, int]:
    dtype = str(tensor.get("dtype") or "")
    d0, d1, d2 = _shape3(tensor, label)
    size = _int_field(tensor, "size", label)
    if dtype in {"U32", "uint32", "UINT32"}:
        name = tensor.get("name")
        if not isinstance(name, str) or not name.endswith(".weight"):
            raise RuntimeCheckError(f"{label} MXFP4 tensor must end with .weight")
        scale_name = name[:-7] + ".scales"
        scales = _find_resident_tensor_by_name(resident_layout, scale_name)
        if scales is None:
            raise RuntimeCheckError(f"{label} MXFP4 tensor missing {scale_name}")
        scale_dtype = str(scales.get("dtype") or "")
        if scale_dtype not in {"U8", "uint8", "UINT8"}:
            raise RuntimeCheckError(f"{label} MXFP4 scales must have U8 dtype")
        s0, s1, groups = _shape3(scales, f"{label}.scales")
        if s0 != d0 or s1 != d1:
            raise RuntimeCheckError(
                f"{label} MXFP4 scales must match first two weight dims"
            )
        logical_d2 = d2 * 8
        if logical_d2 % groups != 0:
            raise RuntimeCheckError(
                f"{label} MXFP4 groups do not divide logical dim {logical_d2}"
            )
        group_size = logical_d2 // groups
        if group_size <= 0 or group_size % 8 != 0:
            raise RuntimeCheckError(
                f"{label} MXFP4 group size must be a positive multiple of 8"
            )
        expected_weight = d0 * d1 * d2 * 4
        scale_size = _int_field(scales, "size", f"{label}.scales")
        expected_scales = d0 * d1 * groups
        if size != expected_weight or scale_size != expected_scales:
            raise RuntimeCheckError(
                f"{label} MXFP4 size mismatch: weight {size}/{expected_weight}, "
                f"scales {scale_size}/{expected_scales}"
            )
        f32_bytes = d0 * d1 * logical_d2 * 4
        return d0, d1, logical_d2, size + scale_size, f32_bytes
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise RuntimeCheckError(f"unsupported resident 3D tensor dtype {dtype} for {label}")
    expected = d0 * d1 * d2 * dtype_nbytes
    if size != expected:
        raise RuntimeCheckError(f"{label} size {size} does not match expected {expected}")
    return d0, d1, d2, size, d0 * d1 * d2 * 4


def _matrix_dims(
    resident_layout: dict[str, Any],
    matrix: dict[str, Any],
    label: str,
) -> tuple[int, int, int]:
    try:
        mxfp4 = resident_mxfp4_layout_info(resident_layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise RuntimeCheckError(str(exc)) from exc
    if mxfp4 is not None:
        return mxfp4.out_dim, mxfp4.in_dim, mxfp4.total_bytes
    try:
        affine = resident_affine_int4_layout_info(resident_layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise RuntimeCheckError(str(exc)) from exc
    if affine is not None:
        return affine.out_dim, affine.in_dim, affine.total_bytes
    shape = matrix.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise RuntimeCheckError(f"{label} must have a 2-D integer shape")
    dtype = str(matrix.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise RuntimeCheckError(f"unsupported resident matrix dtype {dtype} for {label}")
    size = _int_field(matrix, "size", label)
    expected = int(shape[0]) * int(shape[1]) * dtype_nbytes
    if size != expected:
        raise RuntimeCheckError(
            f"{label} size {size} does not match expected {expected}"
        )
    return int(shape[0]), int(shape[1]), size


def _vector_dim(vector: dict[str, Any], label: str) -> tuple[int, int]:
    shape = vector.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 1
        or not _is_int(shape[0])
    ):
        raise RuntimeCheckError(f"{label} must have a 1-D integer shape")
    dtype = str(vector.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise RuntimeCheckError(f"unsupported resident vector dtype {dtype} for {label}")
    dim = int(shape[0])
    size = _int_field(vector, "size", label)
    expected = dim * dtype_nbytes
    if size != expected:
        raise RuntimeCheckError(
            f"{label} size {size} does not match expected {expected}"
        )
    return dim, size


def _shared_dims(
    resident_layout: dict[str, Any],
    matrix: dict[str, Any],
    name: str,
) -> tuple[int, int, int]:
    try:
        mxfp4 = resident_mxfp4_layout_info(resident_layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise RuntimeCheckError(str(exc)) from exc
    if mxfp4 is not None:
        return mxfp4.out_dim, mxfp4.in_dim, mxfp4.total_bytes
    try:
        affine = resident_affine_int4_layout_info(resident_layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise RuntimeCheckError(str(exc)) from exc
    if affine is not None:
        return affine.out_dim, affine.in_dim, affine.total_bytes
    shape = matrix.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise RuntimeCheckError(f"shared expert {name} must have a 2-D integer shape")
    size = _int_field(matrix, "size", f"shared expert {name}")
    dtype = str(matrix.get("dtype") or "")
    dtype_bytes = _dtype_bytes(dtype)
    if dtype_bytes == 0:
        raise RuntimeCheckError(f"unsupported shared expert dtype {dtype}")
    expected = int(shape[0]) * int(shape[1]) * dtype_bytes
    if size != expected:
        raise RuntimeCheckError(
            f"shared expert {name} size {size} does not match expected {expected}"
        )
    return int(shape[0]), int(shape[1]), size


def check_layer_runtime(
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    *,
    layer: int,
    dense_mlp: bool = False,
    top_k: int = 8,
    max_k: int = 8,
    max_slot_bytes: int = 256 * 1024**2,
    max_router_bytes: int = 64 * 1024**2,
    max_resident_matrix_bytes: int = 512 * 1024**2,
    max_runner_scratch_bytes: int = 4 * 1024**3,
    max_cache_read_bytes: int = 256 * 1024**2,
    include_shared_expert: bool = False,
    include_attention_projections: bool = False,
    include_decoder_layer: bool = False,
    context_length: int | None = None,
    num_heads: int | None = None,
    qk_nope_dim: int | None = None,
    rope_dim: int | None = None,
    v_head_dim: int | None = None,
    cache_dtype_bytes: int = 2,
    decode_mla_key_cache: bool = False,
    dsa_indexer_mode: str = "none",
    dsa_index_topk: int | None = None,
    dsa_index_head_dim: int | None = None,
) -> LayerRuntimeBudget:
    layer = _require_int(layer, "layer")
    top_k = _require_int(top_k, "top_k")
    max_k = _require_int(max_k, "max_k")
    max_slot_bytes = _require_int(max_slot_bytes, "max_slot_bytes")
    max_router_bytes = _require_int(max_router_bytes, "max_router_bytes")
    max_resident_matrix_bytes = _require_int(
        max_resident_matrix_bytes, "max_resident_matrix_bytes"
    )
    max_runner_scratch_bytes = _require_int(
        max_runner_scratch_bytes, "max_runner_scratch_bytes"
    )
    max_cache_read_bytes = _require_int(max_cache_read_bytes, "max_cache_read_bytes")
    cache_dtype_bytes = _require_int(cache_dtype_bytes, "cache_dtype_bytes")
    if type(decode_mla_key_cache) is not bool:
        raise RuntimeCheckError("decode_mla_key_cache must be a boolean")
    if context_length is not None:
        context_length = _require_int(context_length, "context_length")
    if num_heads is not None:
        num_heads = _require_int(num_heads, "num_heads")
    if qk_nope_dim is not None:
        qk_nope_dim = _require_int(qk_nope_dim, "qk_nope_dim")
    if rope_dim is not None:
        rope_dim = _require_int(rope_dim, "rope_dim")
    if v_head_dim is not None:
        v_head_dim = _require_int(v_head_dim, "v_head_dim")
    if dsa_index_topk is not None:
        dsa_index_topk = _require_int(dsa_index_topk, "dsa_index_topk")
    if dsa_index_head_dim is not None:
        dsa_index_head_dim = _require_int(dsa_index_head_dim, "dsa_index_head_dim")

    if layer < 0:
        raise RuntimeCheckError("layer must be non-negative")
    if max_k <= 0 or max_k > 64:
        raise RuntimeCheckError("max_k must be in 1..64")
    if not dense_mlp and (top_k <= 0 or top_k > max_k):
        raise RuntimeCheckError("top_k must be in 1..max_k")

    resident_layout = _load_json(resident_layout_path)
    if max_resident_matrix_bytes <= 0:
        raise RuntimeCheckError("max_resident_matrix_bytes must be positive")
    if max_cache_read_bytes <= 0:
        raise RuntimeCheckError("max_cache_read_bytes must be positive")

    if dense_mlp:
        dense_gate = _find_dense_mlp_matrix(resident_layout, layer, "gate_proj")
        dense_up = _find_dense_mlp_matrix(resident_layout, layer, "up_proj")
        dense_down = _find_dense_mlp_matrix(resident_layout, layer, "down_proj")
        gate_out, gate_in, gate_bytes = _matrix_dims(
            resident_layout, dense_gate, "mlp.gate_proj.weight"
        )
        up_out, up_in, up_bytes = _matrix_dims(
            resident_layout, dense_up, "mlp.up_proj.weight"
        )
        down_out, down_in, down_bytes = _matrix_dims(
            resident_layout, dense_down, "mlp.down_proj.weight"
        )
        if gate_out != up_out or gate_in != up_in or down_out != gate_in or down_in != gate_out:
            raise RuntimeCheckError(
                "dense MLP shapes are inconsistent: "
                f"gate=[{gate_out},{gate_in}], up=[{up_out},{up_in}], "
                f"down=[{down_out},{down_in}]"
            )
        dense_max_matrix_bytes = max(gate_bytes, up_bytes, down_bytes)
        if dense_max_matrix_bytes > max_resident_matrix_bytes:
            raise RuntimeCheckError(
                f"resident dense MLP matrix {dense_max_matrix_bytes} bytes "
                f"exceeds limit {max_resident_matrix_bytes}"
            )
        hidden_dim = gate_in
        intermediate_dim = gate_out
        num_experts = 0
        slot_bytes = 0
        aligned_slot = 0
        router_bytes = 0
        logits_bytes = 0
        router_stage_peak = 0
        shared_intermediate_dim: int | None = None
        shared_max_matrix_bytes = 0
        shared_max_aligned_matrix_bytes = 0
        scratch_intermediate_dim = intermediate_dim
        dense_mlp_matrix_peak = _align_up(dense_max_matrix_bytes, 2 * 1024 * 1024)
    else:
        expert_layout = _load_json(expert_layout_path)
        expert_layer = _find_layer(expert_layout, layer)
        gate_w = _component(expert_layer, "gate_proj.weight")
        down_w = _component(expert_layer, "down_proj.weight")
        gate_out, gate_packed_in = _shape2(gate_w, "gate_proj.weight")
        down_out, down_packed_in = _shape2(down_w, "down_proj.weight")

        hidden_dim = down_out
        intermediate_dim = gate_out
        gate_in = gate_packed_in * 8
        down_in = down_packed_in * 8
        if gate_in != hidden_dim or down_in != intermediate_dim:
            raise RuntimeCheckError(
                "expert shapes are inconsistent: "
                f"gate_in={gate_in}, hidden={hidden_dim}, "
                f"down_in={down_in}, intermediate={intermediate_dim}"
            )

        num_experts = _int_field(expert_layer, "num_experts", "expert layer")
        slot_bytes = _int_field(expert_layer, "expert_slot_bytes", "expert layer")
        if num_experts <= 0 or slot_bytes <= 0:
            raise RuntimeCheckError("expert layer missing num_experts or expert_slot_bytes")
        if top_k > num_experts:
            raise RuntimeCheckError(
                f"top_k {top_k} exceeds layer expert count {num_experts}"
            )
        if slot_bytes > max_slot_bytes:
            raise RuntimeCheckError(
                f"expert slot {slot_bytes} bytes exceeds limit {max_slot_bytes}"
            )

        router = _find_router(resident_layout, layer)
        (
            router_experts,
            router_hidden,
            router_bytes,
            router_peak_storage_bytes,
        ) = _router_dims(
            resident_layout,
            router,
        )
        if router_experts != num_experts:
            raise RuntimeCheckError(
                f"router has {router_experts} experts but expert layout has {num_experts}"
            )
        if router_hidden != hidden_dim:
            raise RuntimeCheckError(
                f"router hidden dim {router_hidden} does not match expert hidden dim {hidden_dim}"
            )
        if router_bytes <= 0:
            raise RuntimeCheckError("router tensor size must be positive")
        if router_bytes > max_router_bytes:
            raise RuntimeCheckError(
                f"router tensor {router_bytes} bytes exceeds limit {max_router_bytes}"
            )

        logits_bytes = num_experts * 4
        aligned_slot = _align_up(slot_bytes, 2 * 1024 * 1024)
        router_stage_peak = (
            2 * router_peak_storage_bytes
            + 2 * hidden_dim * 4
            + 2 * logits_bytes
        )
        scratch_intermediate_dim = intermediate_dim
        shared_intermediate_dim = None
        shared_max_matrix_bytes = 0
        shared_max_aligned_matrix_bytes = 0
        if include_shared_expert:
            shared_gate = _find_shared_matrix(resident_layout, layer, "gate_proj")
            shared_up = _find_shared_matrix(resident_layout, layer, "up_proj")
            shared_down = _find_shared_matrix(resident_layout, layer, "down_proj")
            gate_out, gate_in, gate_bytes = _shared_dims(
                resident_layout,
                shared_gate,
                "gate_proj.weight",
            )
            up_out, up_in, up_bytes = _shared_dims(
                resident_layout,
                shared_up,
                "up_proj.weight",
            )
            down_out, down_in, down_bytes = _shared_dims(
                resident_layout,
                shared_down,
                "down_proj.weight",
            )
            if gate_out != up_out or gate_in != up_in or down_out != gate_in or down_in != gate_out:
                raise RuntimeCheckError(
                    "shared expert shapes are inconsistent: "
                    f"gate=[{gate_out},{gate_in}], up=[{up_out},{up_in}], "
                    f"down=[{down_out},{down_in}]"
                )
            if gate_in != hidden_dim:
                raise RuntimeCheckError(
                    f"shared expert hidden dim {gate_in} does not match routed hidden dim {hidden_dim}"
                )
            shared_intermediate_dim = gate_out
            scratch_intermediate_dim = max(scratch_intermediate_dim, shared_intermediate_dim)
            shared_max_matrix_bytes = max(
                gate_bytes,
                up_bytes,
                down_bytes,
            )
            shared_max_aligned_matrix_bytes = _align_up(
                shared_max_matrix_bytes, 2 * 1024 * 1024
            )
        dense_mlp_matrix_peak = 0
    if include_decoder_layer:
        include_attention_projections = True
        if dsa_indexer_mode not in {"none", "full", "shared"}:
            raise RuntimeCheckError("dsa_indexer_mode must be none, full, or shared")
        if (
            context_length is None
            or num_heads is None
            or qk_nope_dim is None
            or rope_dim is None
            or v_head_dim is None
        ):
            raise RuntimeCheckError(
                "decoder layer check requires context_length, num_heads, "
                "qk_nope_dim, rope_dim, and v_head_dim"
            )
        if context_length <= 0 or num_heads <= 0 or qk_nope_dim <= 0:
            raise RuntimeCheckError("decoder layer dimensions must be positive")
        if rope_dim <= 0 or v_head_dim <= 0 or rope_dim % 2 != 0:
            raise RuntimeCheckError("rope_dim must be positive and even; v_head_dim positive")
        if cache_dtype_bytes not in {2, 4}:
            raise RuntimeCheckError("cache_dtype_bytes must be 2 or 4")
        if dsa_indexer_mode in {"full", "shared"}:
            if dsa_index_topk is None or dsa_index_topk <= 0:
                raise RuntimeCheckError("dsa_index_topk must be positive for DSA decoder layers")
            if dsa_indexer_mode == "full" and (
                dsa_index_head_dim is None or dsa_index_head_dim <= 0
            ):
                raise RuntimeCheckError(
                    "dsa_index_head_dim must be positive for full DSA decoder layers"
                )

    hidden_bytes = hidden_dim * 4
    intermediate_bytes = scratch_intermediate_dim * 4
    if dense_mlp:
        moe_stage_peak = dense_mlp_matrix_peak + 5 * hidden_bytes + 3 * intermediate_bytes
    else:
        moe_stage_peak = (
            aligned_slot
            + logits_bytes
            + shared_max_aligned_matrix_bytes
            + 5 * hidden_bytes
            + 3 * intermediate_bytes
        )

    attention_q_lora_dim: int | None = None
    attention_q_output_dim: int | None = None
    attention_kv_lora_dim: int | None = None
    attention_kv_rope_dim: int | None = None
    attention_kv_output_dim: int | None = None
    attention_read_bytes_per_token = 0
    attention_max_matrix_bytes = 0
    attention_max_aligned_matrix_bytes = 0
    attention_projection_peak = 0
    decoder_cache_read_bytes = 0
    dsa_index_cache_read_bytes = 0
    decoder_mla_cache_read_bytes = 0
    decoder_cache_f32_bytes = 0
    mla_key_cache_bytes = 0
    decoder_mla_attention_peak = 0
    decoder_attention_output_peak = 0
    if include_attention_projections:
        input_norm = _find_resident_tensor_by_suffix(
            resident_layout, layer, ".input_layernorm.weight"
        )
        q_norm = _find_resident_tensor_by_suffix(
            resident_layout, layer, ".self_attn.q_a_layernorm.weight"
        )
        kv_norm = _find_resident_tensor_by_suffix(
            resident_layout, layer, ".self_attn.kv_a_layernorm.weight"
        )
        q_a = _find_resident_tensor_by_suffix(
            resident_layout, layer, ".self_attn.q_a_proj.weight"
        )
        q_b = _find_resident_tensor_by_suffix(
            resident_layout, layer, ".self_attn.q_b_proj.weight"
        )
        kv_a = _find_resident_tensor_by_suffix(
            resident_layout, layer, ".self_attn.kv_a_proj_with_mqa.weight"
        )
        kv_b = _find_resident_tensor_by_suffix_optional(
            resident_layout, layer, ".self_attn.kv_b_proj.weight"
        )
        embed_q: dict[str, Any] | None = None
        unembed_out: dict[str, Any] | None = None
        if kv_b is None:
            embed_q = _find_resident_tensor_by_suffix(
                resident_layout, layer, ".self_attn.embed_q.weight"
            )
            unembed_out = _find_resident_tensor_by_suffix(
                resident_layout, layer, ".self_attn.unembed_out.weight"
            )
        o_proj: dict[str, Any] | None = None
        post_attention_norm: dict[str, Any] | None = None
        if include_decoder_layer:
            o_proj = _find_resident_tensor_by_suffix(
                resident_layout, layer, ".self_attn.o_proj.weight"
            )
            post_attention_norm = _find_resident_tensor_by_suffix(
                resident_layout, layer, ".post_attention_layernorm.weight"
            )
        input_norm_dim, _input_norm_bytes = _vector_dim(input_norm, "input_layernorm")
        q_norm_dim, _q_norm_bytes = _vector_dim(q_norm, "q_a_layernorm")
        kv_norm_dim, _kv_norm_bytes = _vector_dim(kv_norm, "kv_a_layernorm")
        q_a_out, q_a_in, q_a_bytes = _matrix_dims(
            resident_layout, q_a, "q_a_proj.weight"
        )
        q_b_out, q_b_in, q_b_bytes = _matrix_dims(
            resident_layout, q_b, "q_b_proj.weight"
        )
        kv_a_out, kv_a_in, kv_a_bytes = _matrix_dims(
            resident_layout, kv_a, "kv_a_proj_with_mqa.weight"
        )
        kv_b_attention_source_f32_bytes = 0
        kv_b_projection_output_bytes = 0
        kv_b_alias_max_matrix_bytes = 0
        if kv_b is not None:
            kv_b_out, kv_b_in, kv_b_bytes = _matrix_dims(
                resident_layout, kv_b, "kv_b_proj.weight"
            )
            kv_b_attention_storage_bytes = kv_b_bytes
            kv_b_projection_output_bytes = kv_b_out * 4
        else:
            assert embed_q is not None
            assert unembed_out is not None
            (
                embed_heads,
                embed_kv_lora,
                embed_qk_nope,
                embed_storage_bytes,
                embed_f32_bytes,
            ) = _tensor3d_dims(resident_layout, embed_q, "embed_q.weight")
            (
                unembed_heads,
                unembed_v_head,
                unembed_kv_lora,
                unembed_storage_bytes,
                unembed_f32_bytes,
            ) = _tensor3d_dims(resident_layout, unembed_out, "unembed_out.weight")
            if (
                embed_kv_lora != kv_norm_dim
                or unembed_kv_lora != kv_norm_dim
                or embed_heads != unembed_heads
            ):
                raise RuntimeCheckError(
                    "absorbed attention alias dims are inconsistent: "
                    f"embed_q=[{embed_heads},{embed_kv_lora},{embed_qk_nope}], "
                    f"unembed_out=[{unembed_heads},{unembed_v_head},{unembed_kv_lora}], "
                    f"kv_norm={kv_norm_dim}"
                )
            kv_b_in = kv_norm_dim
            kv_b_out = embed_heads * (embed_qk_nope + unembed_v_head)
            kv_b_bytes = 0
            kv_b_attention_storage_bytes = embed_storage_bytes + unembed_storage_bytes
            kv_b_attention_source_f32_bytes = embed_f32_bytes + unembed_f32_bytes
            kv_b_alias_max_matrix_bytes = max(
                embed_storage_bytes,
                unembed_storage_bytes,
                embed_f32_bytes,
                unembed_f32_bytes,
            )
        if q_a_in != hidden_dim or kv_a_in != hidden_dim or input_norm_dim != hidden_dim:
            raise RuntimeCheckError(
                "attention input dims are inconsistent: "
                f"hidden={hidden_dim}, q_a_in={q_a_in}, kv_a_in={kv_a_in}, "
                f"input_norm={input_norm_dim}"
            )
        if q_a_out != q_b_in or q_norm_dim != q_a_out:
            raise RuntimeCheckError(
                "attention Q dims are inconsistent: "
                f"q_a_out={q_a_out}, q_b_in={q_b_in}, q_norm={q_norm_dim}"
            )
        if kv_norm_dim != kv_b_in or kv_a_out < kv_norm_dim:
            raise RuntimeCheckError(
                "attention KV dims are inconsistent: "
                f"kv_a_out={kv_a_out}, kv_norm={kv_norm_dim}, kv_b_in={kv_b_in}"
            )
        attention_max_matrix_bytes = max(
            q_a_bytes,
            q_b_bytes,
            kv_a_bytes,
            kv_b_bytes,
            kv_b_alias_max_matrix_bytes,
        )
        if attention_max_matrix_bytes > max_resident_matrix_bytes:
            raise RuntimeCheckError(
                f"resident attention matrix {attention_max_matrix_bytes} bytes "
                f"exceeds limit {max_resident_matrix_bytes}"
            )
        attention_q_lora_dim = q_a_out
        attention_q_output_dim = q_b_out
        attention_kv_lora_dim = kv_norm_dim
        attention_kv_rope_dim = kv_a_out - kv_norm_dim
        attention_kv_output_dim = kv_b_out
        attention_read_bytes_per_token = q_a_bytes + q_b_bytes + kv_a_bytes + kv_b_bytes
        attention_max_aligned_matrix_bytes = _align_up(
            attention_max_matrix_bytes, 2 * 1024 * 1024
        )
        q_lora_bytes = q_a_out * 4
        q_out_bytes = q_b_out * 4
        kv_a_out_bytes = kv_a_out * 4
        kv_lora_bytes = kv_norm_dim * 4
        kv_out_bytes = kv_b_projection_output_bytes
        attention_projection_peak = (
            attention_max_aligned_matrix_bytes
            + 2 * hidden_bytes
            + 2 * q_lora_bytes
            + q_out_bytes
            + kv_a_out_bytes
            + kv_lora_bytes
            + kv_out_bytes
        )
        if include_decoder_layer:
            assert context_length is not None
            assert num_heads is not None
            assert qk_nope_dim is not None
            assert rope_dim is not None
            assert v_head_dim is not None
            assert post_attention_norm is not None
            post_norm_dim, _post_norm_bytes = _vector_dim(
                post_attention_norm, "post_attention_layernorm"
            )
            if post_norm_dim != hidden_dim:
                raise RuntimeCheckError(
                    f"post_attention_layernorm dim {post_norm_dim} "
                    f"does not match hidden dim {hidden_dim}"
                )
            if q_b_out != num_heads * (qk_nope_dim + rope_dim):
                raise RuntimeCheckError(
                    "decoder Q dims are inconsistent: "
                    f"q_b_out={q_b_out}, expected={num_heads * (qk_nope_dim + rope_dim)}"
                )
            if kv_b_in != kv_norm_dim:
                raise RuntimeCheckError(
                    f"kv_b_in={kv_b_in} does not match kv_lora_dim={kv_norm_dim}"
                )
            if kv_b_out != num_heads * (qk_nope_dim + v_head_dim):
                raise RuntimeCheckError(
                    "decoder KV-B dims are inconsistent: "
                    f"kv_b_out={kv_b_out}, "
                    f"expected={num_heads * (qk_nope_dim + v_head_dim)}"
                )
            assert o_proj is not None
            o_out, o_in, o_bytes = _matrix_dims(
                resident_layout, o_proj, "o_proj.weight"
            )
            expected_o_in = num_heads * v_head_dim
            if o_in != expected_o_in or o_out != hidden_dim:
                raise RuntimeCheckError(
                    "decoder output projection dims are inconsistent: "
                    f"o_proj=[{o_out},{o_in}], expected=[{hidden_dim},{expected_o_in}]"
                )
            if o_bytes > max_resident_matrix_bytes:
                raise RuntimeCheckError(
                    f"resident attention matrix {o_bytes} bytes exceeds limit "
                    f"{max_resident_matrix_bytes}"
                )
            attention_max_matrix_bytes = max(attention_max_matrix_bytes, o_bytes)
            attention_max_aligned_matrix_bytes = _align_up(
                attention_max_matrix_bytes, 2 * 1024 * 1024
            )
            cache_width = kv_norm_dim + rope_dim
            if dsa_indexer_mode == "none":
                decoder_mla_cache_read_bytes = context_length * cache_width * cache_dtype_bytes
                decoder_cache_f32_bytes = context_length * cache_width * 4
            else:
                assert dsa_index_topk is not None
                selected = min(dsa_index_topk, context_length)
                decoder_mla_cache_read_bytes = selected * cache_width * cache_dtype_bytes
                decoder_cache_f32_bytes = selected * cache_width * 4
                if dsa_indexer_mode == "full":
                    assert dsa_index_head_dim is not None
                    dsa_index_cache_read_bytes = (
                        context_length * dsa_index_head_dim * cache_dtype_bytes
                    )
            decoder_cache_read_bytes = (
                dsa_index_cache_read_bytes + decoder_mla_cache_read_bytes
            )
            if decoder_cache_read_bytes > max_cache_read_bytes:
                raise RuntimeCheckError(
                    f"decoder cache read {decoder_cache_read_bytes} bytes exceeds "
                    f"limit {max_cache_read_bytes}"
                )
            q_nope_bytes = num_heads * qk_nope_dim * 4
            q_rope_bytes = num_heads * rope_dim * 4
            kv_b_f32_bytes = kv_b_out * kv_b_in * 4
            attn_value_bytes = num_heads * v_head_dim * 4
            visible_context = (
                context_length
                if dsa_indexer_mode == "none"
                else min(int(dsa_index_topk or 0), context_length)
            )
            mla_weights_bytes = visible_context * num_heads * 4
            mla_value_cache_bytes = visible_context * num_heads * v_head_dim * 4
            mla_key_cache_enabled = (
                decode_mla_key_cache
                or os.environ.get("LARGERLM_MLA_KEY_CACHE") is not None
            )
            mla_key_cache_bytes = (
                visible_context * num_heads * qk_nope_dim * 4
                if mla_key_cache_enabled
                else 0
            )
            mla_rope_cache_bytes = (
                visible_context * rope_dim * 4
                if (
                    mla_key_cache_bytes > 0
                    and os.environ.get("LARGERLM_MLA_ROPE_CACHE") is not None
                )
                else 0
            )
            decoder_mla_attention_base_peak = (
                decoder_cache_read_bytes
                + decoder_cache_f32_bytes
                + kv_b_attention_storage_bytes
                + kv_b_attention_source_f32_bytes
                + kv_b_f32_bytes
                + q_nope_bytes
                + q_rope_bytes
                + attn_value_bytes
                + mla_weights_bytes
            )
            decoder_mla_attention_peak = decoder_mla_attention_base_peak
            if decoder_mla_attention_peak + mla_key_cache_bytes <= max_runner_scratch_bytes:
                decoder_mla_attention_peak += mla_key_cache_bytes
            if (
                mla_key_cache_bytes > 0
                and decoder_mla_attention_peak + mla_rope_cache_bytes
                <= max_runner_scratch_bytes
            ):
                decoder_mla_attention_peak += mla_rope_cache_bytes
            if (
                os.environ.get("LARGERLM_MLA_DISABLE_VALUE_CACHE") is None
                and os.environ.get("LARGERLM_MLA_VALUE_CACHE_SINGLETON") is not None
                and decoder_mla_attention_peak + mla_value_cache_bytes
                <= max_runner_scratch_bytes
            ):
                decoder_mla_attention_peak += mla_value_cache_bytes
            decoder_attention_output_peak = (
                _align_up(o_bytes, 2 * 1024 * 1024)
                + attn_value_bytes
                + 2 * hidden_bytes
            )

    estimated_peak = max(router_stage_peak, moe_stage_peak)
    if attention_projection_peak > estimated_peak:
        estimated_peak = attention_projection_peak
    if decoder_mla_attention_peak > estimated_peak:
        estimated_peak = decoder_mla_attention_peak
    if decoder_attention_output_peak > estimated_peak:
        estimated_peak = decoder_attention_output_peak
    if estimated_peak > max_runner_scratch_bytes:
        raise RuntimeCheckError(
            f"estimated runner scratch {estimated_peak} bytes exceeds limit "
            f"{max_runner_scratch_bytes}"
        )

    return LayerRuntimeBudget(
        expert_layout_path=Path(expert_layout_path),
        resident_layout_path=Path(resident_layout_path),
        layer=layer,
        layer_kind="dense" if dense_mlp else "moe",
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=0 if dense_mlp else top_k,
        expert_slot_bytes=slot_bytes,
        aligned_slot_bytes=aligned_slot,
        router_bytes=router_bytes,
        router_logits_bytes=logits_bytes,
        read_bytes_per_token=0 if dense_mlp else top_k * slot_bytes,
        include_shared_expert=False if dense_mlp else include_shared_expert,
        shared_intermediate_dim=shared_intermediate_dim,
        shared_max_matrix_bytes=shared_max_matrix_bytes,
        shared_max_aligned_matrix_bytes=shared_max_aligned_matrix_bytes,
        include_attention_projections=include_attention_projections,
        attention_q_lora_dim=attention_q_lora_dim,
        attention_q_output_dim=attention_q_output_dim,
        attention_kv_lora_dim=attention_kv_lora_dim,
        attention_kv_rope_dim=attention_kv_rope_dim,
        attention_kv_output_dim=attention_kv_output_dim,
        attention_read_bytes_per_token=attention_read_bytes_per_token,
        attention_max_matrix_bytes=attention_max_matrix_bytes,
        attention_max_aligned_matrix_bytes=attention_max_aligned_matrix_bytes,
        attention_projection_peak_bytes=attention_projection_peak,
        include_decoder_layer=include_decoder_layer,
        decoder_context_length=context_length if include_decoder_layer else None,
        decoder_num_heads=num_heads if include_decoder_layer else None,
        decoder_qk_nope_dim=qk_nope_dim if include_decoder_layer else None,
        decoder_rope_dim=rope_dim if include_decoder_layer else None,
        decoder_v_head_dim=v_head_dim if include_decoder_layer else None,
        decoder_mla_key_cache=(
            bool(decode_mla_key_cache)
            if include_decoder_layer
            else False
        ),
        decoder_mla_key_cache_bytes=mla_key_cache_bytes,
        dsa_indexer_mode=dsa_indexer_mode if include_decoder_layer else "none",
        dsa_index_topk=dsa_index_topk if include_decoder_layer else None,
        dsa_index_head_dim=dsa_index_head_dim if include_decoder_layer else None,
        dsa_index_cache_read_bytes=dsa_index_cache_read_bytes,
        decoder_mla_cache_read_bytes=decoder_mla_cache_read_bytes,
        decoder_cache_read_bytes=decoder_cache_read_bytes,
        decoder_cache_f32_bytes=decoder_cache_f32_bytes,
        decoder_mla_attention_peak_bytes=decoder_mla_attention_peak,
        decoder_attention_output_peak_bytes=decoder_attention_output_peak,
        router_stage_peak_bytes=router_stage_peak,
        moe_stage_peak_bytes=moe_stage_peak,
        estimated_peak_bytes=estimated_peak,
        max_k=max_k,
        max_slot_bytes=max_slot_bytes,
        max_router_bytes=max_router_bytes,
        max_resident_matrix_bytes=max_resident_matrix_bytes,
        max_cache_read_bytes=max_cache_read_bytes,
        max_runner_scratch_bytes=max_runner_scratch_bytes,
    )
