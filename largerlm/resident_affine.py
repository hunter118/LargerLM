from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ResidentAffineLayoutError(ValueError):
    """Raised when a resident affine-int4 triplet is malformed."""


AFFINE_INT4_WEIGHT_DTYPES = {"U32", "UINT32"}
AFFINE_INT4_META_DTYPES = {"BF16", "BFLOAT16", "F16", "FLOAT16"}
MXFP4_SCALE_DTYPES = {"U8", "UINT8"}


@dataclass(frozen=True)
class ResidentAffineInt4LayoutInfo:
    name: str
    weight: dict[str, Any]
    scales: dict[str, Any]
    biases: dict[str, Any]
    out_dim: int
    in_dim: int
    group_size: int
    total_bytes: int


@dataclass(frozen=True)
class ResidentMxfp4LayoutInfo:
    name: str
    weight: dict[str, Any]
    scales: dict[str, Any]
    out_dim: int
    in_dim: int
    group_size: int
    total_bytes: int


def is_affine_int4_weight_dtype(dtype: object) -> bool:
    return isinstance(dtype, str) and dtype.upper() in AFFINE_INT4_WEIGHT_DTYPES


def is_affine_int4_meta_dtype(dtype: object) -> bool:
    return isinstance(dtype, str) and dtype.upper() in AFFINE_INT4_META_DTYPES


def is_mxfp4_scale_dtype(dtype: object) -> bool:
    return isinstance(dtype, str) and dtype.upper() in MXFP4_SCALE_DTYPES


def _shape2(tensor: dict[str, Any], label: str) -> tuple[int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or type(shape[0]) is not int
        or type(shape[1]) is not int
    ):
        raise ResidentAffineLayoutError(f"{label} must have 2-D integer shape")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ResidentAffineLayoutError(f"{label} dimensions must be positive")
    return int(shape[0]), int(shape[1])


def _size(tensor: dict[str, Any], label: str) -> int:
    value = tensor.get("size")
    if type(value) is not int or value < 0:
        raise ResidentAffineLayoutError(f"{label} size must be a non-negative integer")
    return int(value)


def _tensor_name(tensor: dict[str, Any], label: str) -> str:
    name = tensor.get("name")
    if not isinstance(name, str) or not name:
        raise ResidentAffineLayoutError(f"{label} name must be a non-empty string")
    return name


def resident_layout_tensors_by_name(
    resident_layout: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    tensors = resident_layout.get("tensors")
    if not isinstance(tensors, list):
        raise ResidentAffineLayoutError("resident layout missing tensors array")
    result: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            result[name] = item
    return result


def resident_affine_int4_layout_info(
    resident_layout: dict[str, Any],
    weight: dict[str, Any],
    *,
    tensors_by_name: dict[str, dict[str, Any]] | None = None,
) -> ResidentAffineInt4LayoutInfo | None:
    if not is_affine_int4_weight_dtype(weight.get("dtype")):
        return None
    weight_name = _tensor_name(weight, "resident affine-int4 weight")
    if not weight_name.endswith(".weight"):
        raise ResidentAffineLayoutError(
            f"resident affine-int4 tensor {weight_name} must end with .weight"
        )
    by_name = (
        tensors_by_name
        if tensors_by_name is not None
        else resident_layout_tensors_by_name(resident_layout)
    )
    base = weight_name[: -len(".weight")]
    scales = by_name.get(f"{base}.scales")
    biases = by_name.get(f"{base}.biases")
    if scales is None or biases is None:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 weight {weight_name} is missing scales/biases companions"
        )
    if not is_affine_int4_meta_dtype(scales.get("dtype")) or not is_affine_int4_meta_dtype(
        biases.get("dtype")
    ):
        raise ResidentAffineLayoutError(
            f"resident affine-int4 metadata for {weight_name} must be BF16/F16"
        )
    out_dim, packed_cols = _shape2(weight, weight_name)
    scale_shape = _shape2(scales, f"{base}.scales")
    bias_shape = _shape2(biases, f"{base}.biases")
    if scale_shape != bias_shape or scale_shape[0] != out_dim:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 metadata for {weight_name} must have shape [out_dim, groups]"
        )
    in_dim = packed_cols * 8
    groups = scale_shape[1]
    if groups <= 0 or in_dim % groups != 0:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 groups for {weight_name} do not divide logical input dim {in_dim}"
        )
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 group size for {weight_name} must be a positive multiple of 8"
        )
    expected_weight_bytes = out_dim * packed_cols * 4
    expected_meta_bytes = out_dim * groups * 2
    if _size(weight, weight_name) != expected_weight_bytes:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 weight {weight_name} size does not match packed shape"
        )
    if _size(scales, f"{base}.scales") != expected_meta_bytes:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 scales for {weight_name} size does not match metadata shape"
        )
    if _size(biases, f"{base}.biases") != expected_meta_bytes:
        raise ResidentAffineLayoutError(
            f"resident affine-int4 biases for {weight_name} size does not match metadata shape"
        )
    total_bytes = _size(weight, weight_name) + _size(scales, f"{base}.scales") + _size(
        biases,
        f"{base}.biases",
    )
    return ResidentAffineInt4LayoutInfo(
        name=weight_name,
        weight=weight,
        scales=scales,
        biases=biases,
        out_dim=out_dim,
        in_dim=in_dim,
        group_size=group_size,
        total_bytes=total_bytes,
    )


def resident_mxfp4_layout_info(
    resident_layout: dict[str, Any],
    weight: dict[str, Any],
    *,
    tensors_by_name: dict[str, dict[str, Any]] | None = None,
) -> ResidentMxfp4LayoutInfo | None:
    if not is_affine_int4_weight_dtype(weight.get("dtype")):
        return None
    weight_name = _tensor_name(weight, "resident MXFP4 weight")
    if not weight_name.endswith(".weight"):
        raise ResidentAffineLayoutError(
            f"resident MXFP4 tensor {weight_name} must end with .weight"
        )
    by_name = (
        tensors_by_name
        if tensors_by_name is not None
        else resident_layout_tensors_by_name(resident_layout)
    )
    base = weight_name[: -len(".weight")]
    scales = by_name.get(f"{base}.scales")
    biases = by_name.get(f"{base}.biases")
    if scales is None or biases is not None:
        return None
    if not is_mxfp4_scale_dtype(scales.get("dtype")):
        return None
    out_dim, packed_cols = _shape2(weight, weight_name)
    scale_rows, groups = _shape2(scales, f"{base}.scales")
    if scale_rows != out_dim:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 scales for {weight_name} must have shape [out_dim, groups]"
        )
    in_dim = packed_cols * 8
    if groups <= 0 or in_dim % groups != 0:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 groups for {weight_name} do not divide logical input dim {in_dim}"
        )
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 group size for {weight_name} must be a positive multiple of 8"
        )
    expected_weight_bytes = out_dim * packed_cols * 4
    expected_scale_bytes = out_dim * groups
    if _size(weight, weight_name) != expected_weight_bytes:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 weight {weight_name} size does not match packed shape"
        )
    if _size(scales, f"{base}.scales") != expected_scale_bytes:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 scales for {weight_name} size does not match metadata shape"
        )
    total_bytes = _size(weight, weight_name) + _size(scales, f"{base}.scales")
    return ResidentMxfp4LayoutInfo(
        name=weight_name,
        weight=weight,
        scales=scales,
        out_dim=out_dim,
        in_dim=in_dim,
        group_size=group_size,
        total_bytes=total_bytes,
    )


def resident_layout_logical_shape(
    resident_layout: dict[str, Any],
    tensor: dict[str, Any],
    *,
    tensors_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, ...]:
    mxfp4 = resident_mxfp4_layout_info(
        resident_layout,
        tensor,
        tensors_by_name=tensors_by_name,
    )
    if mxfp4 is not None:
        return mxfp4.out_dim, mxfp4.in_dim
    affine = resident_affine_int4_layout_info(
        resident_layout,
        tensor,
        tensors_by_name=tensors_by_name,
    )
    if affine is not None:
        return affine.out_dim, affine.in_dim
    shape = tensor.get("shape")
    if not isinstance(shape, (list, tuple)) or not all(type(dim) is int for dim in shape):
        raise ResidentAffineLayoutError("resident tensor shape must be an integer array")
    return tuple(int(dim) for dim in shape)
