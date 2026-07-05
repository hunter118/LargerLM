from __future__ import annotations

import heapq
import json
import math
import os
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Optional fast path; pure-Python fallback remains below.
    import numpy as _np
except Exception:  # pragma: no cover - depends on the local runtime image.
    _np = None  # type: ignore[assignment]

from .decode_cache import DecodeCacheSegment, load_decode_cache_layout
from .resident_affine import (
    ResidentAffineLayoutError,
    ResidentMxfp4LayoutInfo,
    resident_mxfp4_layout_info,
)


class DSAIndexerError(RuntimeError):
    """Raised when a DSA indexer baseline step cannot run safely."""


@dataclass(frozen=True)
class DSAIndexerCacheWriteResult:
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    input_path: Path
    layer: int
    start_position: int
    batch_tokens: int
    hidden_dim: int
    index_head_dim: int
    qk_rope_dim: int
    rope_interleave: bool
    cache_dtype: str
    cache_write_bytes: int
    cache_segment_offset: int
    first_write_offset: int
    resident_matrix_bytes: int
    resident_matrix_f32_bytes: int
    estimated_peak_bytes: int


@dataclass(frozen=True)
class DSAIndexerTopKResult:
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    hidden_path: Path
    q_resid_path: Path
    output_indices_path: Path | None
    output_indices_u32_path: Path | None
    layer: int
    start_position: int
    batch_tokens: int
    context_length: int
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    q_lora_dim: int
    qk_rope_dim: int
    rope_interleave: bool
    cache_read_bytes: int
    resident_matrix_bytes: int
    resident_matrix_f32_bytes: int
    estimated_peak_bytes: int
    output_indices_u32_bytes: int
    topk_indices_collected: bool
    topk_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class DSAIndexerBatchResult:
    cache_write: DSAIndexerCacheWriteResult
    topk: DSAIndexerTopKResult


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DSAIndexerError(f"failed to read JSON {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DSAIndexerError(f"failed to parse JSON {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DSAIndexerError(f"{p} must contain a JSON object")
    return payload


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _replace_atomic(tmp_path: Path, path: Path) -> None:
    try:
        tmp_path.replace(path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_atomic(tmp_path, path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"F32", "float32"}:
        return 4
    if dtype in {"BF16", "bfloat16", "F16", "float16"}:
        return 2
    return 0


def _mxfp4_info(
    layout: dict[str, Any],
    tensor: dict[str, Any],
) -> ResidentMxfp4LayoutInfo | None:
    try:
        return resident_mxfp4_layout_info(layout, tensor)
    except ResidentAffineLayoutError as exc:
        raise DSAIndexerError(str(exc)) from exc


def _is_int(value: object) -> bool:
    return type(value) is int


def _require_int(value: object, label: str) -> int:
    if not _is_int(value):
        raise DSAIndexerError(f"{label} must be an integer")
    return int(value)


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DSAIndexerError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DSAIndexerError(f"{label} must be a finite number")
    return number


def _int_field(tensor: dict[str, Any], field: str, label: str) -> int:
    return _require_int(tensor.get(field), f"{label} {field}")


def _find_layer_tensor(layout: dict[str, Any], *, layer: int, suffix: str) -> dict[str, Any]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise DSAIndexerError("resident layout missing tensors array")
    needle = f".layers.{layer}."
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        if isinstance(name, str) and needle in name and name.endswith(suffix):
            return tensor
    raise DSAIndexerError(f"resident tensor for layer {layer} suffix {suffix} not found")


def _shape1(tensor: dict[str, Any], label: str) -> int:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) < 1 or not _is_int(shape[0]):
        raise DSAIndexerError(f"{label} must have shape [dim]")
    dim = int(shape[0])
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    size = _int_field(tensor, "size", label)
    if dim <= 0 or dtype_nbytes == 0 or size != dim * dtype_nbytes:
        raise DSAIndexerError(f"{label} has invalid shape, dtype, or size")
    return dim


def _shape2(
    tensor: dict[str, Any],
    label: str,
    *,
    layout: dict[str, Any] | None = None,
) -> tuple[int, int]:
    if layout is not None:
        mxfp4 = _mxfp4_info(layout, tensor)
        if mxfp4 is not None:
            return mxfp4.out_dim, mxfp4.in_dim
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise DSAIndexerError(f"{label} must have shape [rows, cols]")
    rows, cols = int(shape[0]), int(shape[1])
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    size = _int_field(tensor, "size", label)
    if rows <= 0 or cols <= 0 or dtype_nbytes == 0 or size != rows * cols * dtype_nbytes:
        raise DSAIndexerError(f"{label} has invalid shape, dtype, or size")
    return rows, cols


def _matrix_storage_bytes(
    layout: dict[str, Any],
    tensor: dict[str, Any],
    label: str,
) -> int:
    mxfp4 = _mxfp4_info(layout, tensor)
    if mxfp4 is not None:
        return mxfp4.total_bytes
    return _int_field(tensor, "size", label)


def _resident_weight_path(layout_path: Path, layout: dict[str, Any]) -> Path:
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise DSAIndexerError("resident layout missing weight_file")
    path = layout_path.parent / weight_file
    if not path.exists():
        raise DSAIndexerError(f"resident weight file not found: {path}")
    return path


def _bf16_to_f32(raw: bytes, offset: int) -> float:
    bits = int.from_bytes(raw[offset : offset + 2], "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def _read_tensor_bytes(weight_path: Path, tensor: dict[str, Any], label: str) -> bytes:
    size = _int_field(tensor, "size", label)
    offset = _int_field(tensor, "offset", label)
    if size <= 0 or offset < 0:
        raise DSAIndexerError("resident tensor has invalid offset or size")
    try:
        fd = os.open(weight_path, os.O_RDONLY)
        try:
            raw = os.pread(fd, size, offset)
        finally:
            os.close(fd)
    except OSError as exc:
        raise DSAIndexerError(f"failed to read resident tensor from {weight_path}: {exc}") from exc
    if len(raw) != size:
        raise DSAIndexerError(f"short read for resident tensor {tensor.get('name')}")
    return raw


def _mxfp4_e2m1_to_f32(value: int) -> float:
    lookup = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    mag = value & 0x7
    sign = -1.0 if value & 0x8 else 1.0
    return sign * lookup[mag]


def _mxfp4_e8m0_to_f32(value: int) -> float:
    return struct.unpack("<f", ((value & 0xFF) << 23).to_bytes(4, "little"))[0]


def _mxfp4_to_f32_array(
    weight_raw: bytes,
    scale_raw: bytes,
    info: ResidentMxfp4LayoutInfo,
) -> array:
    packed_cols = info.in_dim // 8
    groups = info.in_dim // info.group_size
    packed_per_group = info.group_size // 8
    expected_weight = info.out_dim * packed_cols * 4
    expected_scale = info.out_dim * groups
    if len(weight_raw) != expected_weight or len(scale_raw) != expected_scale:
        raise DSAIndexerError(f"resident MXFP4 matrix {info.name} raw byte mismatch")
    values = array("f", [0.0]) * (info.out_dim * info.in_dim)
    for row in range(info.out_dim):
        out_base = row * info.in_dim
        weight_base = row * packed_cols * 4
        scale_base = row * groups
        for group in range(groups):
            scale = _mxfp4_e8m0_to_f32(scale_raw[scale_base + group])
            packed_group_base = weight_base + group * packed_per_group * 4
            x_group_base = group * info.group_size
            for packed_index in range(packed_per_group):
                packed = struct.unpack_from(
                    "<I",
                    weight_raw,
                    packed_group_base + packed_index * 4,
                )[0]
                x_base = out_base + x_group_base + packed_index * 8
                for nibble in range(8):
                    values[x_base + nibble] = (
                        _mxfp4_e2m1_to_f32((packed >> (nibble * 4)) & 0xF)
                        * scale
                    )
    return values


def _numpy_f32_to_array(values: Any) -> array:
    out = array("f")
    out.frombytes(
        _np.asarray(values, dtype=_np.float32).tobytes()  # type: ignore[union-attr]
    )
    return out


def _tensor_to_f32_array(
    weight_path: Path,
    tensor: dict[str, Any],
    *,
    layout: dict[str, Any] | None = None,
) -> array:
    dtype = str(tensor.get("dtype") or "")
    label = str(tensor.get("name") or "resident tensor")
    if layout is not None:
        mxfp4 = _mxfp4_info(layout, tensor)
        if mxfp4 is not None:
            weight_raw = _read_tensor_bytes(weight_path, mxfp4.weight, mxfp4.name)
            scale_label = f"{mxfp4.name[: -len('.weight')]}.scales"
            scale_raw = _read_tensor_bytes(weight_path, mxfp4.scales, scale_label)
            return _mxfp4_to_f32_array(weight_raw, scale_raw, mxfp4)

    raw = _read_tensor_bytes(weight_path, tensor, label)

    values = array("f")
    if dtype in {"F32", "float32"}:
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    if dtype in {"BF16", "bfloat16"}:
        if _np is not None:
            words = _np.frombuffer(raw, dtype="<u2").astype(_np.uint32)  # type: ignore[union-attr]
            return _numpy_f32_to_array((words << 16).view(_np.float32))  # type: ignore[union-attr]
        for idx in range(0, len(raw), 2):
            values.append(_bf16_to_f32(raw, idx))
        return values
    if dtype in {"F16", "float16"}:
        if _np is not None:
            return _numpy_f32_to_array(
                _np.frombuffer(raw, dtype="<f2").astype(_np.float32)  # type: ignore[union-attr]
            )
        for idx in range(0, len(raw), 2):
            values.append(struct.unpack_from("<e", raw, idx)[0])
        return values
    raise DSAIndexerError(f"unsupported resident dtype {dtype}")


def _read_f32_row(handle: Any, row_dim: int, label: str) -> array:
    row_bytes = row_dim * 4
    raw = handle.read(row_bytes)
    if len(raw) != row_bytes:
        raise DSAIndexerError(f"failed to read full {label} row")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _validate_f32_file(path: Path, *, rows: int, row_dim: int, label: str) -> int:
    expected = rows * row_dim * 4
    try:
        actual = path.stat().st_size
    except OSError as exc:
        raise DSAIndexerError(f"failed to stat {label} file {path}: {exc}") from exc
    if actual != expected:
        raise DSAIndexerError(f"{label} bytes {actual} do not match expected {expected}")
    return actual


def _matvec(matrix: array, rows: int, cols: int, vector: array) -> array:
    if len(vector) != cols:
        raise DSAIndexerError(f"matvec input dim {len(vector)} does not match {cols}")
    if _np is not None:
        try:
            matrix_np = _np.frombuffer(  # type: ignore[union-attr]
                matrix,
                dtype=_np.float32,  # type: ignore[union-attr]
                count=rows * cols,
            ).reshape(rows, cols)
            vector_np = _np.frombuffer(  # type: ignore[union-attr]
                vector,
                dtype=_np.float32,  # type: ignore[union-attr]
                count=cols,
            )
            result = matrix_np @ vector_np
            out = array("f")
            out.frombytes(
                _np.asarray(result, dtype=_np.float32).tobytes()  # type: ignore[union-attr]
            )
            return out
        except (BufferError, TypeError, ValueError):
            pass
    out = array("f", [0.0]) * rows
    for row in range(rows):
        base = row * cols
        acc = 0.0
        for col in range(cols):
            acc += float(matrix[base + col]) * float(vector[col])
        out[row] = acc
    return out


def _read_f32_matrix_np(path: Path, *, rows: int, cols: int, label: str) -> Any | None:
    if _np is None:
        return None
    values = _np.fromfile(path, dtype="<f4", count=rows * cols)  # type: ignore[union-attr]
    if values.size != rows * cols:
        raise DSAIndexerError(f"failed to read full {label} matrix")
    return values.reshape(rows, cols)


def _array_to_f32_matrix_np(values: array, *, rows: int, cols: int) -> Any | None:
    if _np is None:
        return None
    try:
        return _np.frombuffer(  # type: ignore[union-attr]
            values,
            dtype=_np.float32,  # type: ignore[union-attr]
            count=rows * cols,
        ).reshape(rows, cols)
    except (BufferError, TypeError, ValueError):
        return None


def _layer_norm(values: array, weight: array, bias: array, eps: float) -> array:
    dim = len(values)
    if len(weight) != dim or len(bias) != dim:
        raise DSAIndexerError("indexer k_norm vector dims do not match wk output")
    mean = sum(float(value) for value in values) / dim
    variance = sum((float(value) - mean) ** 2 for value in values) / dim
    scale = 1.0 / math.sqrt(variance + eps)
    out = array("f", [0.0]) * dim
    for idx, value in enumerate(values):
        out[idx] = (float(value) - mean) * scale * float(weight[idx]) + float(bias[idx])
    return out


def _apply_rope(
    values: array,
    *,
    rope_dim: int,
    position: int,
    theta: float,
    interleave: bool = False,
) -> None:
    if rope_dim == 0:
        return
    if rope_dim < 0 or rope_dim > len(values) or rope_dim % 2 != 0:
        raise DSAIndexerError("qk_rope_dim must be even and within index_head_dim")
    half = rope_dim // 2
    original = values[:rope_dim]
    for idx in range(half):
        angle = position / (theta ** ((2.0 * idx) / rope_dim))
        cos_v = math.cos(angle)
        sin_v = math.sin(angle)
        if interleave:
            even = 2 * idx
            odd = even + 1
            x1 = float(original[even])
            x2 = float(original[odd])
            values[even] = x1 * cos_v - x2 * sin_v
            values[odd] = x2 * cos_v + x1 * sin_v
        else:
            x1 = float(original[idx])
            x2 = float(original[idx + half])
            values[idx] = x1 * cos_v - x2 * sin_v
            values[idx + half] = x2 * cos_v + x1 * sin_v


def _apply_rope_batch_np(
    values: Any,
    *,
    rope_dim: int,
    positions: Any,
    theta: float,
    interleave: bool = False,
) -> bool:
    if _np is None:
        return False
    if rope_dim == 0:
        return True
    if rope_dim < 0 or rope_dim > values.shape[1] or rope_dim % 2 != 0:
        raise DSAIndexerError("qk_rope_dim must be even and within index_head_dim")
    half = rope_dim // 2
    dim_index = _np.arange(half, dtype=_np.float64)  # type: ignore[union-attr]
    angles = (
        _np.asarray(positions, dtype=_np.float64)[:, None]  # type: ignore[union-attr]
        / (float(theta) ** ((2.0 * dim_index) / float(rope_dim)))
    )
    cos_v = _np.cos(angles)  # type: ignore[union-attr]
    sin_v = _np.sin(angles)  # type: ignore[union-attr]
    original = values[:, :rope_dim].astype(_np.float64, copy=True)  # type: ignore[union-attr]
    if interleave:
        even = original[:, 0::2]
        odd = original[:, 1::2]
        values[:, 0:rope_dim:2] = (even * cos_v - odd * sin_v).astype(  # type: ignore[union-attr]
            _np.float32
        )
        values[:, 1:rope_dim:2] = (odd * cos_v + even * sin_v).astype(  # type: ignore[union-attr]
            _np.float32
        )
    else:
        left = original[:, :half]
        right = original[:, half:rope_dim]
        values[:, :half] = (left * cos_v - right * sin_v).astype(_np.float32)  # type: ignore[union-attr]
        values[:, half:rope_dim] = (right * cos_v + left * sin_v).astype(  # type: ignore[union-attr]
            _np.float32
        )
    return True


def _f32_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return (bits + 0x8000) >> 16


def _encode_cache_row(values: array, *, dtype: str, width: int) -> bytes:
    if len(values) != width:
        raise DSAIndexerError("cache row width mismatch")
    if dtype in {"F32", "float32"}:
        return struct.pack(f"<{width}f", *values)
    if dtype in {"BF16", "bfloat16"}:
        return struct.pack(f"<{width}H", *(_f32_to_bf16_bits(value) for value in values))
    raise DSAIndexerError(f"dsa index cache write supports BF16 or F32, got {dtype}")


def _encode_cache_rows_np(
    values: Any,
    *,
    dtype: str,
    width: int,
    rows: int,
) -> bytes | None:
    if _np is None:
        return None
    if values.shape != (rows, width):
        raise DSAIndexerError("cache row matrix width mismatch")
    if dtype in {"F32", "float32"}:
        return _np.asarray(values, dtype="<f4").tobytes()  # type: ignore[union-attr]
    if dtype in {"BF16", "bfloat16"}:
        f32 = _np.asarray(values, dtype="<f4")  # type: ignore[union-attr]
        bits = f32.view(_np.uint32)  # type: ignore[union-attr]
        return ((bits + 0x8000) >> 16).astype("<u2").tobytes()
    return None


def _decode_cache_row(raw: bytes, *, dtype: str, width: int) -> array:
    values = array("f")
    if dtype in {"F32", "float32"}:
        if len(raw) != width * 4:
            raise DSAIndexerError("F32 cache row has invalid byte length")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    if dtype in {"BF16", "bfloat16"}:
        if len(raw) != width * 2:
            raise DSAIndexerError("BF16 cache row has invalid byte length")
        for idx in range(0, len(raw), 2):
            values.append(_bf16_to_f32(raw, idx))
        return values
    raise DSAIndexerError(f"dsa index cache read supports BF16 or F32, got {dtype}")


def _decode_cache_rows_np(
    raw: bytes,
    *,
    dtype: str,
    width: int,
    rows: int,
) -> Any | None:
    if _np is None:
        return None
    if rows < 0 or width <= 0:
        raise DSAIndexerError("cache row dimensions are invalid")
    if dtype in {"F32", "float32"}:
        expected = rows * width * 4
        if len(raw) != expected:
            raise DSAIndexerError("F32 cache rows have invalid byte length")
        return _np.frombuffer(raw, dtype="<f4").reshape(rows, width)  # type: ignore[union-attr]
    if dtype in {"BF16", "bfloat16"}:
        expected = rows * width * 2
        if len(raw) != expected:
            raise DSAIndexerError("BF16 cache rows have invalid byte length")
        words = _np.frombuffer(raw, dtype="<u2").astype(_np.uint32)  # type: ignore[union-attr]
        return (words << 16).view(_np.float32).reshape(rows, width)  # type: ignore[union-attr]
    return None


def _dsa_segment(cache_layout_path: Path, *, layer: int) -> tuple[Any, DecodeCacheSegment]:
    layout = load_decode_cache_layout(cache_layout_path)
    for segment in layout.segments:
        if segment.kind == "dsa_index" and segment.layer == layer:
            return layout, segment
    raise DSAIndexerError(f"dsa_index cache segment for layer {layer} not found")


def _check_cache_file(
    cache_file_path: Path,
    *,
    layout_total_bytes: int,
    max_cache_file_bytes: int,
) -> int:
    try:
        size = cache_file_path.stat().st_size
    except OSError as exc:
        raise DSAIndexerError(f"failed to stat cache file {cache_file_path}: {exc}") from exc
    if size < layout_total_bytes:
        raise DSAIndexerError(
            f"cache file has {size} bytes, expected at least {layout_total_bytes}"
        )
    if size > max_cache_file_bytes:
        raise DSAIndexerError(
            f"cache file has {size} bytes, exceeds limit {max_cache_file_bytes}"
        )
    return size


def _write_dsa_index_cache_batch_np(
    *,
    cache_path: Path,
    segment: DecodeCacheSegment,
    hidden_path: Path,
    wk_values: array,
    norm_weight: array,
    norm_bias: array,
    start_position: int,
    batch_tokens: int,
    hidden_dim: int,
    head_dim: int,
    qk_rope_dim: int,
    rope_theta: float,
    rope_interleave: bool,
    layer_norm_eps: float,
) -> bool:
    if _np is None:
        return False
    wk_np = _array_to_f32_matrix_np(wk_values, rows=head_dim, cols=hidden_dim)
    if wk_np is None:
        return False
    hidden_np = _read_f32_matrix_np(
        hidden_path,
        rows=batch_tokens,
        cols=hidden_dim,
        label="hidden",
    )
    if hidden_np is None:
        return False
    norm_weight_np = _np.frombuffer(  # type: ignore[union-attr]
        norm_weight,
        dtype=_np.float32,  # type: ignore[union-attr]
        count=head_dim,
    ).astype(_np.float64)  # type: ignore[union-attr]
    norm_bias_np = _np.frombuffer(  # type: ignore[union-attr]
        norm_bias,
        dtype=_np.float32,  # type: ignore[union-attr]
        count=head_dim,
    ).astype(_np.float64)  # type: ignore[union-attr]
    k_np = hidden_np @ wk_np.T
    k64 = k_np.astype(_np.float64, copy=False)  # type: ignore[union-attr]
    mean = k64.mean(axis=1, keepdims=True)
    variance = ((k64 - mean) ** 2).mean(axis=1, keepdims=True)
    k_np = ((k64 - mean) / _np.sqrt(variance + layer_norm_eps))  # type: ignore[union-attr]
    k_np = (k_np * norm_weight_np + norm_bias_np).astype(_np.float32)  # type: ignore[union-attr]
    positions = _np.arange(  # type: ignore[union-attr]
        start_position,
        start_position + batch_tokens,
        dtype=_np.int64,
    )
    _apply_rope_batch_np(
        k_np,
        rope_dim=qk_rope_dim,
        positions=positions,
        theta=rope_theta,
        interleave=rope_interleave,
    )
    encoded = _encode_cache_rows_np(
        k_np,
        dtype=segment.dtype,
        width=head_dim,
        rows=batch_tokens,
    )
    if encoded is None:
        return False
    row_bytes = head_dim * segment.dtype_bytes
    first_write_offset = segment.offset + start_position * segment.token_stride_bytes
    try:
        with cache_path.open("r+b") as cache_file:
            if segment.token_stride_bytes == row_bytes:
                cache_file.seek(first_write_offset)
                cache_file.write(encoded)
            else:
                for token_index in range(batch_tokens):
                    row_start = token_index * row_bytes
                    write_offset = (
                        segment.offset
                        + (start_position + token_index) * segment.token_stride_bytes
                    )
                    cache_file.seek(write_offset)
                    cache_file.write(encoded[row_start : row_start + row_bytes])
    except OSError as exc:
        raise DSAIndexerError(f"failed to write DSA index cache rows: {exc}") from exc
    return True


def write_dsa_index_cache_batch(
    *,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    layer: int,
    hidden_f32_path: str | Path,
    start_position: int,
    batch_tokens: int,
    qk_rope_dim: int,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    layer_norm_eps: float = 1e-6,
    max_resident_matrix_mib: int = 512,
    max_cache_file_mib: float = 32768.0,
    max_cache_write_mib: float = 4096.0,
    max_runner_scratch_mib: int = 4096,
) -> DSAIndexerCacheWriteResult:
    layer = _require_int(layer, "layer")
    start_position = _require_int(start_position, "start_position")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    qk_rope_dim = _require_int(qk_rope_dim, "qk_rope_dim")
    max_resident_matrix_mib = _require_number(
        max_resident_matrix_mib, "max_resident_matrix_mib"
    )
    max_runner_scratch_mib = _require_number(
        max_runner_scratch_mib, "max_runner_scratch_mib"
    )
    max_cache_file_mib = _require_number(max_cache_file_mib, "max_cache_file_mib")
    max_cache_write_mib = _require_number(max_cache_write_mib, "max_cache_write_mib")
    rope_theta = _require_number(rope_theta, "rope_theta")
    layer_norm_eps = _require_number(layer_norm_eps, "layer_norm_eps")
    if layer < 0 or start_position < 0:
        raise DSAIndexerError("layer and start_position must be non-negative")
    if batch_tokens <= 0:
        raise DSAIndexerError("batch_tokens must be positive")
    if rope_theta <= 0 or layer_norm_eps <= 0:
        raise DSAIndexerError("rope_theta and layer_norm_eps must be positive")

    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_path = _resident_weight_path(layout_path, layout)
    cache_layout_p = Path(cache_layout_path)
    cache_path = Path(cache_file_path)
    cache_layout, segment = _dsa_segment(cache_layout_p, layer=layer)
    max_cache_file_bytes = int(max_cache_file_mib * 1024 * 1024)
    max_cache_write_bytes = int(max_cache_write_mib * 1024 * 1024)
    _check_cache_file(
        cache_path,
        layout_total_bytes=cache_layout.total_bytes,
        max_cache_file_bytes=max_cache_file_bytes,
    )
    if start_position + batch_tokens > segment.max_context_tokens:
        raise DSAIndexerError("dsa index cache write exceeds max context")
    cache_write_bytes = batch_tokens * segment.token_stride_bytes
    if cache_write_bytes > max_cache_write_bytes:
        raise DSAIndexerError(
            f"cache write {cache_write_bytes} bytes exceeds limit {max_cache_write_bytes}"
        )

    wk = _find_layer_tensor(layout, layer=layer, suffix=".self_attn.indexer.wk.weight")
    k_norm_weight = _find_layer_tensor(
        layout, layer=layer, suffix=".self_attn.indexer.k_norm.weight"
    )
    k_norm_bias = _find_layer_tensor(
        layout, layer=layer, suffix=".self_attn.indexer.k_norm.bias"
    )
    head_dim, hidden_dim = _shape2(wk, "indexer wk", layout=layout)
    if head_dim != segment.width:
        raise DSAIndexerError(
            f"indexer wk output dim {head_dim} does not match dsa cache width {segment.width}"
        )
    if _shape1(k_norm_weight, "indexer k_norm.weight") != head_dim:
        raise DSAIndexerError("indexer k_norm.weight dim mismatch")
    if _shape1(k_norm_bias, "indexer k_norm.bias") != head_dim:
        raise DSAIndexerError("indexer k_norm.bias dim mismatch")
    if qk_rope_dim < 0 or qk_rope_dim > head_dim or qk_rope_dim % 2 != 0:
        raise DSAIndexerError("qk_rope_dim must be even and within index_head_dim")

    hidden_path = Path(hidden_f32_path)
    _validate_f32_file(hidden_path, rows=batch_tokens, row_dim=hidden_dim, label="hidden")

    wk_bytes = _matrix_storage_bytes(layout, wk, "indexer wk")
    k_norm_weight_bytes = _int_field(k_norm_weight, "size", "indexer k_norm.weight")
    k_norm_bias_bytes = _int_field(k_norm_bias, "size", "indexer k_norm.bias")
    resident_matrix_bytes = wk_bytes + k_norm_weight_bytes + k_norm_bias_bytes
    resident_matrix_f32_bytes = (head_dim * hidden_dim + 2 * head_dim) * 4
    max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    if wk_bytes > max_matrix_bytes or resident_matrix_f32_bytes > max_matrix_bytes:
        raise DSAIndexerError("indexer wk exceeds resident matrix limit")
    estimated_peak = resident_matrix_f32_bytes + hidden_dim * 4 + head_dim * 8
    batch_estimated_peak = (
        resident_matrix_f32_bytes
        + batch_tokens * hidden_dim * 4
        + batch_tokens * head_dim * 4 * 4
        + cache_write_bytes
        + batch_tokens * 4 * 3
    )
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise DSAIndexerError(
            f"estimated peak {estimated_peak} bytes exceeds scratch limit {max_scratch_bytes}"
        )

    wk_values = _tensor_to_f32_array(weight_path, wk, layout=layout)
    norm_weight = _tensor_to_f32_array(weight_path, k_norm_weight)
    norm_bias = _tensor_to_f32_array(weight_path, k_norm_bias)
    first_write_offset = segment.offset + start_position * segment.token_stride_bytes
    batch_fast_path_used = False
    if _np is not None and batch_estimated_peak <= max_scratch_bytes:
        try:
            batch_fast_path_used = _write_dsa_index_cache_batch_np(
                cache_path=cache_path,
                segment=segment,
                hidden_path=hidden_path,
                wk_values=wk_values,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                start_position=start_position,
                batch_tokens=batch_tokens,
                hidden_dim=hidden_dim,
                head_dim=head_dim,
                qk_rope_dim=qk_rope_dim,
                rope_theta=rope_theta,
                rope_interleave=rope_interleave,
                layer_norm_eps=layer_norm_eps,
            )
        except (BufferError, TypeError, ValueError):
            batch_fast_path_used = False
    if batch_fast_path_used:
        estimated_peak = max(estimated_peak, batch_estimated_peak)
        return DSAIndexerCacheWriteResult(
            resident_layout_path=layout_path,
            cache_layout_path=cache_layout_p,
            cache_file_path=cache_path,
            input_path=hidden_path,
            layer=layer,
            start_position=start_position,
            batch_tokens=batch_tokens,
            hidden_dim=hidden_dim,
            index_head_dim=head_dim,
            qk_rope_dim=qk_rope_dim,
            rope_interleave=rope_interleave,
            cache_dtype=segment.dtype,
            cache_write_bytes=cache_write_bytes,
            cache_segment_offset=segment.offset,
            first_write_offset=first_write_offset,
            resident_matrix_bytes=resident_matrix_bytes,
            resident_matrix_f32_bytes=resident_matrix_f32_bytes,
            estimated_peak_bytes=estimated_peak,
        )
    try:
        with hidden_path.open("rb") as hidden_file, cache_path.open("r+b") as cache_file:
            for token_index in range(batch_tokens):
                hidden = _read_f32_row(hidden_file, hidden_dim, "hidden")
                k = _matvec(wk_values, head_dim, hidden_dim, hidden)
                k = _layer_norm(k, norm_weight, norm_bias, layer_norm_eps)
                _apply_rope(
                    k,
                    rope_dim=qk_rope_dim,
                    position=start_position + token_index,
                    theta=rope_theta,
                    interleave=rope_interleave,
                )
                encoded = _encode_cache_row(k, dtype=segment.dtype, width=head_dim)
                write_offset = (
                    segment.offset
                    + (start_position + token_index) * segment.token_stride_bytes
                )
                cache_file.seek(write_offset)
                cache_file.write(encoded)
    except OSError as exc:
        raise DSAIndexerError(f"failed to write DSA index cache rows: {exc}") from exc

    return DSAIndexerCacheWriteResult(
        resident_layout_path=layout_path,
        cache_layout_path=cache_layout_p,
        cache_file_path=cache_path,
        input_path=hidden_path,
        layer=layer,
        start_position=start_position,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        index_head_dim=head_dim,
        qk_rope_dim=qk_rope_dim,
        rope_interleave=rope_interleave,
        cache_dtype=segment.dtype,
        cache_write_bytes=cache_write_bytes,
        cache_segment_offset=segment.offset,
        first_write_offset=first_write_offset,
        resident_matrix_bytes=resident_matrix_bytes,
        resident_matrix_f32_bytes=resident_matrix_f32_bytes,
        estimated_peak_bytes=estimated_peak,
    )


def _write_topk_json(path: Path, result: DSAIndexerTopKResult) -> None:
    payload = {
        "layer": result.layer,
        "start_position": result.start_position,
        "batch_tokens": result.batch_tokens,
        "context_length": result.context_length,
        "index_topk": result.index_topk,
        "index_n_heads": result.index_n_heads,
        "index_head_dim": result.index_head_dim,
        "q_lora_dim": result.q_lora_dim,
        "qk_rope_dim": result.qk_rope_dim,
        "rope_interleave": result.rope_interleave,
        "topk_indices": [list(row) for row in result.topk_indices],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, payload)


def _write_topk_u32_row(handle: Any, row: tuple[int, ...], *, index_topk: int) -> None:
    if len(row) > index_topk:
        raise DSAIndexerError("top-k row is longer than index_topk")
    values = [len(row), *row, *([0] * (index_topk - len(row)))]
    handle.write(struct.pack(f"<{len(values)}I", *values))


def _score_dsa_prefix_np(
    *,
    fd_cache: int,
    segment: DecodeCacheSegment,
    q: array,
    head_weights: array,
    index_n_heads: int,
    head_dim: int,
    prefix_tokens: int,
    topk_count: int,
    softmax_scale: float,
) -> tuple[int, ...] | None:
    if _np is None:
        return None
    if prefix_tokens <= 0 or topk_count <= 0:
        return ()
    try:
        q_np = _np.frombuffer(  # type: ignore[union-attr]
            q,
            dtype=_np.float32,  # type: ignore[union-attr]
            count=index_n_heads * head_dim,
        ).reshape(index_n_heads, head_dim)
        head_weights_np = _np.frombuffer(  # type: ignore[union-attr]
            head_weights,
            dtype=_np.float32,  # type: ignore[union-attr]
            count=index_n_heads,
        )
    except (BufferError, TypeError, ValueError):
        return None
    read_bytes = prefix_tokens * segment.token_stride_bytes
    raw = os.pread(fd_cache, read_bytes, segment.offset)
    if len(raw) != read_bytes:
        raise DSAIndexerError("short read from dsa index cache")
    k_np = _decode_cache_rows_np(
        raw,
        dtype=segment.dtype,
        width=head_dim,
        rows=prefix_tokens,
    )
    if k_np is None:
        return None
    dots = k_np @ q_np.T
    scores = _np.maximum(dots * softmax_scale, 0.0) @ head_weights_np  # type: ignore[union-attr]
    order = _np.lexsort(  # type: ignore[union-attr]
        (
            _np.arange(prefix_tokens, dtype=_np.int64),  # type: ignore[union-attr]
            -scores,
        )
    )
    return tuple(int(token) for token in order[:topk_count])


def _compute_dsa_topk_rows_batch_np(
    *,
    fd_cache: int,
    segment: DecodeCacheSegment,
    hidden_path: Path,
    q_resid_path: Path,
    wq_values: array,
    weight_values: array,
    start_position: int,
    batch_tokens: int,
    context_length: int,
    index_topk: int,
    index_n_heads: int,
    head_dim: int,
    hidden_dim: int,
    q_lora_dim: int,
    q_rows: int,
    qk_rope_dim: int,
    rope_theta: float,
    rope_interleave: bool,
    softmax_scale: float,
    head_weight_scale: float,
) -> list[tuple[int, ...]] | None:
    if _np is None:
        return None
    wq_np = _array_to_f32_matrix_np(wq_values, rows=q_rows, cols=q_lora_dim)
    weight_np = _array_to_f32_matrix_np(
        weight_values,
        rows=index_n_heads,
        cols=hidden_dim,
    )
    if wq_np is None or weight_np is None:
        return None
    hidden_np = _read_f32_matrix_np(
        hidden_path,
        rows=batch_tokens,
        cols=hidden_dim,
        label="hidden",
    )
    q_resid_np = _read_f32_matrix_np(
        q_resid_path,
        rows=batch_tokens,
        cols=q_lora_dim,
        label="q_resid",
    )
    if hidden_np is None or q_resid_np is None:
        return None

    q_batch = q_resid_np @ wq_np.T
    q_heads = q_batch.reshape(batch_tokens * index_n_heads, head_dim)
    positions = _np.repeat(  # type: ignore[union-attr]
        _np.arange(  # type: ignore[union-attr]
            start_position,
            start_position + batch_tokens,
            dtype=_np.int64,
        ),
        index_n_heads,
    )
    _apply_rope_batch_np(
        q_heads,
        rope_dim=qk_rope_dim,
        positions=positions,
        theta=rope_theta,
        interleave=rope_interleave,
    )
    q_batch = q_heads.reshape(batch_tokens, index_n_heads, head_dim)
    head_weights_batch = (hidden_np @ weight_np.T) * float(head_weight_scale)

    max_prefix_tokens = min(context_length, start_position + batch_tokens)
    if max_prefix_tokens <= 0:
        return [() for _ in range(batch_tokens)]
    read_bytes = max_prefix_tokens * segment.token_stride_bytes
    raw = os.pread(fd_cache, read_bytes, segment.offset)
    if len(raw) != read_bytes:
        raise DSAIndexerError("short read from dsa index cache")
    k_np = _decode_cache_rows_np(
        raw,
        dtype=segment.dtype,
        width=head_dim,
        rows=max_prefix_tokens,
    )
    if k_np is None:
        return None

    rows: list[tuple[int, ...]] = []
    token_order = _np.arange(max_prefix_tokens, dtype=_np.int64)  # type: ignore[union-attr]
    for token_index in range(batch_tokens):
        position = start_position + token_index
        prefix_tokens = min(context_length, position + 1)
        topk_count = min(index_topk, prefix_tokens)
        if topk_count <= 0:
            rows.append(())
            continue
        dots = k_np[:prefix_tokens] @ q_batch[token_index].T
        scores = (
            _np.maximum(dots * softmax_scale, 0.0)  # type: ignore[union-attr]
            @ head_weights_batch[token_index]
        )
        order = _np.lexsort(  # type: ignore[union-attr]
            (
                token_order[:prefix_tokens],
                -scores,
            )
        )
        rows.append(tuple(int(token) for token in order[:topk_count]))
    return rows


def compute_dsa_topk_batch(
    *,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    layer: int,
    hidden_f32_path: str | Path,
    q_resid_f32_path: str | Path,
    output_indices_path: str | Path | None,
    output_indices_u32_path: str | Path | None = None,
    start_position: int,
    batch_tokens: int,
    context_length: int,
    index_topk: int,
    index_n_heads: int,
    qk_rope_dim: int,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    max_resident_matrix_mib: int = 512,
    max_cache_file_mib: float = 32768.0,
    max_cache_read_mib: float = 4096.0,
    max_runner_scratch_mib: int = 4096,
    collect_topk_indices: bool = True,
) -> DSAIndexerTopKResult:
    layer = _require_int(layer, "layer")
    start_position = _require_int(start_position, "start_position")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    context_length = _require_int(context_length, "context_length")
    index_topk = _require_int(index_topk, "index_topk")
    index_n_heads = _require_int(index_n_heads, "index_n_heads")
    qk_rope_dim = _require_int(qk_rope_dim, "qk_rope_dim")
    max_resident_matrix_mib = _require_number(
        max_resident_matrix_mib, "max_resident_matrix_mib"
    )
    max_runner_scratch_mib = _require_number(
        max_runner_scratch_mib, "max_runner_scratch_mib"
    )
    max_cache_file_mib = _require_number(max_cache_file_mib, "max_cache_file_mib")
    max_cache_read_mib = _require_number(max_cache_read_mib, "max_cache_read_mib")
    rope_theta = _require_number(rope_theta, "rope_theta")
    if layer < 0 or start_position < 0:
        raise DSAIndexerError("layer and start_position must be non-negative")
    if batch_tokens <= 0 or context_length <= 0:
        raise DSAIndexerError("batch_tokens and context_length must be positive")
    if start_position + batch_tokens > context_length:
        raise DSAIndexerError("query positions exceed context_length")
    if index_topk <= 0 or index_n_heads <= 0:
        raise DSAIndexerError("index_topk and index_n_heads must be positive")
    if rope_theta <= 0:
        raise DSAIndexerError("rope_theta must be positive")

    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_path = _resident_weight_path(layout_path, layout)
    cache_layout_p = Path(cache_layout_path)
    cache_path = Path(cache_file_path)
    cache_layout, segment = _dsa_segment(cache_layout_p, layer=layer)
    max_cache_file_bytes = int(max_cache_file_mib * 1024 * 1024)
    _check_cache_file(
        cache_path,
        layout_total_bytes=cache_layout.total_bytes,
        max_cache_file_bytes=max_cache_file_bytes,
    )
    if context_length > segment.max_context_tokens:
        raise DSAIndexerError("context_length exceeds dsa index cache max context")

    wq_b = _find_layer_tensor(layout, layer=layer, suffix=".self_attn.indexer.wq_b.weight")
    weights_proj = _find_layer_tensor(
        layout, layer=layer, suffix=".self_attn.indexer.weights_proj.weight"
    )
    q_rows, q_lora_dim = _shape2(wq_b, "indexer wq_b", layout=layout)
    weight_rows, hidden_dim = _shape2(
        weights_proj,
        "indexer weights_proj",
        layout=layout,
    )
    head_dim = segment.width
    if q_rows != index_n_heads * head_dim:
        raise DSAIndexerError("indexer wq_b rows do not match index_n_heads*head_dim")
    if weight_rows != index_n_heads:
        raise DSAIndexerError("indexer weights_proj rows do not match index_n_heads")
    if qk_rope_dim < 0 or qk_rope_dim > head_dim or qk_rope_dim % 2 != 0:
        raise DSAIndexerError("qk_rope_dim must be even and within index_head_dim")

    hidden_path = Path(hidden_f32_path)
    q_resid_path = Path(q_resid_f32_path)
    _validate_f32_file(hidden_path, rows=batch_tokens, row_dim=hidden_dim, label="hidden")
    _validate_f32_file(q_resid_path, rows=batch_tokens, row_dim=q_lora_dim, label="q_resid")

    wq_b_bytes = _matrix_storage_bytes(layout, wq_b, "indexer wq_b")
    weights_proj_bytes = _matrix_storage_bytes(
        layout,
        weights_proj,
        "indexer weights_proj",
    )
    resident_matrix_bytes = wq_b_bytes + weights_proj_bytes
    resident_matrix_f32_bytes = (q_rows * q_lora_dim + weight_rows * hidden_dim) * 4
    max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    if (
        wq_b_bytes > max_matrix_bytes
        or weights_proj_bytes > max_matrix_bytes
        or resident_matrix_f32_bytes > max_matrix_bytes
    ):
        raise DSAIndexerError("indexer resident matrices exceed resident matrix limit")

    cache_read_bytes = sum(
        min(context_length, start_position + idx + 1) * segment.token_stride_bytes
        for idx in range(batch_tokens)
    )
    max_cache_read_bytes = int(max_cache_read_mib * 1024 * 1024)
    if cache_read_bytes > max_cache_read_bytes:
        raise DSAIndexerError(
            f"cache read {cache_read_bytes} bytes exceeds limit {max_cache_read_bytes}"
        )
    estimated_peak = (
        resident_matrix_f32_bytes
        + hidden_dim * 4
        + q_lora_dim * 4
        + index_n_heads * head_dim * 4
        + segment.token_stride_bytes
        + index_topk * 16
    )
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise DSAIndexerError(
            f"estimated peak {estimated_peak} bytes exceeds scratch limit {max_scratch_bytes}"
        )

    topk_rows: list[tuple[int, ...]] = []
    output_u32_path = (
        Path(output_indices_u32_path) if output_indices_u32_path is not None else None
    )
    output_indices_u32_bytes = 0
    fast_singleton_prefix = (
        start_position == 0
        and batch_tokens == 1
        and context_length == 1
        and output_indices_path is None
        and output_u32_path is not None
        and not collect_topk_indices
    )
    if fast_singleton_prefix:
        output_u32_tmp_path = output_u32_path.with_name(output_u32_path.name + ".tmp")
        cleanup_u32_tmp = False
        try:
            output_u32_path.parent.mkdir(parents=True, exist_ok=True)
            cleanup_u32_tmp = True
            with output_u32_tmp_path.open("wb") as u32_handle:
                _write_topk_u32_row(u32_handle, (0,), index_topk=index_topk)
                output_indices_u32_bytes = (index_topk + 1) * 4
            _replace_atomic(output_u32_tmp_path, output_u32_path)
            cleanup_u32_tmp = False
        except OSError as exc:
            if cleanup_u32_tmp:
                _remove_partial_file(output_u32_tmp_path)
            raise DSAIndexerError(f"failed to compute DSA top-k: {exc}") from exc
        except Exception:
            if cleanup_u32_tmp:
                _remove_partial_file(output_u32_tmp_path)
            raise
        return DSAIndexerTopKResult(
            resident_layout_path=layout_path,
            cache_layout_path=cache_layout_p,
            cache_file_path=cache_path,
            hidden_path=hidden_path,
            q_resid_path=q_resid_path,
            output_indices_path=None,
            output_indices_u32_path=output_u32_path,
            layer=layer,
            start_position=start_position,
            batch_tokens=batch_tokens,
            context_length=context_length,
            index_topk=index_topk,
            index_n_heads=index_n_heads,
            index_head_dim=head_dim,
            q_lora_dim=q_lora_dim,
            qk_rope_dim=qk_rope_dim,
            rope_interleave=rope_interleave,
            cache_read_bytes=0,
            resident_matrix_bytes=resident_matrix_bytes,
            resident_matrix_f32_bytes=resident_matrix_f32_bytes,
            estimated_peak_bytes=estimated_peak,
            output_indices_u32_bytes=output_indices_u32_bytes,
            topk_indices_collected=False,
            topk_indices=(),
        )

    wq_values = _tensor_to_f32_array(weight_path, wq_b, layout=layout)
    weight_values = _tensor_to_f32_array(weight_path, weights_proj, layout=layout)
    output_u32_tmp_path: Path | None = None
    cleanup_u32_tmp = False
    softmax_scale = head_dim ** -0.5
    head_weight_scale = index_n_heads ** -0.5
    max_prefix_tokens = min(context_length, start_position + batch_tokens)
    batch_topk_estimated_peak = (
        resident_matrix_f32_bytes
        + batch_tokens * hidden_dim * 4
        + batch_tokens * q_lora_dim * 4
        + batch_tokens * q_rows * 4
        + batch_tokens * index_n_heads * 4
        + max_prefix_tokens * segment.token_stride_bytes
        + max_prefix_tokens * index_n_heads * 4
        + max_prefix_tokens * 4
        + batch_tokens * min(index_topk, max_prefix_tokens) * 16
    )
    if _np is not None and batch_topk_estimated_peak <= max_scratch_bytes:
        fd_cache = os.open(cache_path, os.O_RDONLY)
        try:
            batch_rows = _compute_dsa_topk_rows_batch_np(
                fd_cache=fd_cache,
                segment=segment,
                hidden_path=hidden_path,
                q_resid_path=q_resid_path,
                wq_values=wq_values,
                weight_values=weight_values,
                start_position=start_position,
                batch_tokens=batch_tokens,
                context_length=context_length,
                index_topk=index_topk,
                index_n_heads=index_n_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                q_lora_dim=q_lora_dim,
                q_rows=q_rows,
                qk_rope_dim=qk_rope_dim,
                rope_theta=rope_theta,
                rope_interleave=rope_interleave,
                softmax_scale=softmax_scale,
                head_weight_scale=head_weight_scale,
            )
        finally:
            os.close(fd_cache)
        if batch_rows is not None:
            try:
                if output_u32_path is not None:
                    output_u32_path.parent.mkdir(parents=True, exist_ok=True)
                    output_u32_tmp_path = output_u32_path.with_name(
                        output_u32_path.name + ".tmp"
                    )
                    cleanup_u32_tmp = True
                    with output_u32_tmp_path.open("wb") as u32_handle:
                        for row in batch_rows:
                            _write_topk_u32_row(
                                u32_handle,
                                row,
                                index_topk=index_topk,
                            )
                            output_indices_u32_bytes += (index_topk + 1) * 4
                    _replace_atomic(output_u32_tmp_path, output_u32_path)
                    cleanup_u32_tmp = False
            except OSError as exc:
                if cleanup_u32_tmp and output_u32_tmp_path is not None:
                    _remove_partial_file(output_u32_tmp_path)
                raise DSAIndexerError(f"failed to compute DSA top-k: {exc}") from exc
            topk_rows = list(batch_rows) if collect_topk_indices else []
            output_path = (
                Path(output_indices_path) if output_indices_path is not None else None
            )
            result = DSAIndexerTopKResult(
                resident_layout_path=layout_path,
                cache_layout_path=cache_layout_p,
                cache_file_path=cache_path,
                hidden_path=hidden_path,
                q_resid_path=q_resid_path,
                output_indices_path=output_path,
                output_indices_u32_path=output_u32_path,
                layer=layer,
                start_position=start_position,
                batch_tokens=batch_tokens,
                context_length=context_length,
                index_topk=index_topk,
                index_n_heads=index_n_heads,
                index_head_dim=head_dim,
                q_lora_dim=q_lora_dim,
                qk_rope_dim=qk_rope_dim,
                rope_interleave=rope_interleave,
                cache_read_bytes=cache_read_bytes,
                resident_matrix_bytes=resident_matrix_bytes,
                resident_matrix_f32_bytes=resident_matrix_f32_bytes,
                estimated_peak_bytes=max(estimated_peak, batch_topk_estimated_peak),
                output_indices_u32_bytes=output_indices_u32_bytes,
                topk_indices_collected=collect_topk_indices,
                topk_indices=tuple(topk_rows),
            )
            if output_path is not None:
                _write_topk_json(output_path, result)
            return result
    u32_handle = None
    try:
        fd_cache = os.open(cache_path, os.O_RDONLY)
        try:
            if output_u32_path is not None:
                output_u32_path.parent.mkdir(parents=True, exist_ok=True)
                output_u32_tmp_path = output_u32_path.with_name(output_u32_path.name + ".tmp")
                cleanup_u32_tmp = True
                u32_handle = output_u32_tmp_path.open("wb")
            with hidden_path.open("rb") as hidden_file, q_resid_path.open("rb") as q_file:
                for token_index in range(batch_tokens):
                    position = start_position + token_index
                    prefix_tokens = min(context_length, position + 1)
                    hidden = _read_f32_row(hidden_file, hidden_dim, "hidden")
                    q_resid = _read_f32_row(q_file, q_lora_dim, "q_resid")
                    q = _matvec(wq_values, q_rows, q_lora_dim, q_resid)
                    for head in range(index_n_heads):
                        start = head * head_dim
                        q_head = q[start : start + head_dim]
                        _apply_rope(
                            q_head,
                            rope_dim=qk_rope_dim,
                            position=position,
                            theta=rope_theta,
                            interleave=rope_interleave,
                        )
                        q[start : start + head_dim] = q_head
                    head_weights = _matvec(
                        weight_values,
                        index_n_heads,
                        hidden_dim,
                        hidden,
                    )
                    for head in range(index_n_heads):
                        head_weights[head] = float(head_weights[head]) * head_weight_scale

                    topk_count = min(index_topk, prefix_tokens)
                    row = _score_dsa_prefix_np(
                        fd_cache=fd_cache,
                        segment=segment,
                        q=q,
                        head_weights=head_weights,
                        index_n_heads=index_n_heads,
                        head_dim=head_dim,
                        prefix_tokens=prefix_tokens,
                        topk_count=topk_count,
                        softmax_scale=softmax_scale,
                    )
                    if row is not None:
                        if collect_topk_indices:
                            topk_rows.append(row)
                        if u32_handle is not None:
                            _write_topk_u32_row(u32_handle, row, index_topk=index_topk)
                            output_indices_u32_bytes += (index_topk + 1) * 4
                        continue
                    heap: list[tuple[float, int]] = []
                    for token_pos in range(prefix_tokens):
                        offset = segment.offset + token_pos * segment.token_stride_bytes
                        raw = os.pread(fd_cache, segment.token_stride_bytes, offset)
                        if len(raw) != segment.token_stride_bytes:
                            raise DSAIndexerError("short read from dsa index cache")
                        k = _decode_cache_row(raw, dtype=segment.dtype, width=head_dim)
                        score = 0.0
                        for head in range(index_n_heads):
                            q_base = head * head_dim
                            dot = 0.0
                            for dim in range(head_dim):
                                dot += float(q[q_base + dim]) * float(k[dim])
                            score += float(head_weights[head]) * max(0.0, dot * softmax_scale)
                        item = (score, token_pos)
                        if len(heap) < topk_count:
                            heapq.heappush(heap, item)
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)
                    row = tuple(
                        token
                        for _score, token in sorted(heap, key=lambda item: (-item[0], item[1]))
                    )
                    if collect_topk_indices:
                        topk_rows.append(row)
                    if u32_handle is not None:
                        _write_topk_u32_row(u32_handle, row, index_topk=index_topk)
                        output_indices_u32_bytes += (index_topk + 1) * 4
        finally:
            try:
                if u32_handle is not None:
                    u32_handle.close()
                    u32_handle = None
            finally:
                os.close(fd_cache)
        if output_u32_tmp_path is not None and output_u32_path is not None:
            _replace_atomic(output_u32_tmp_path, output_u32_path)
            cleanup_u32_tmp = False
    except OSError as exc:
        if cleanup_u32_tmp and output_u32_tmp_path is not None:
            _remove_partial_file(output_u32_tmp_path)
        raise DSAIndexerError(f"failed to compute DSA top-k: {exc}") from exc
    except Exception:
        if u32_handle is not None:
            try:
                u32_handle.close()
            except OSError:
                pass
        if cleanup_u32_tmp and output_u32_tmp_path is not None:
            _remove_partial_file(output_u32_tmp_path)
        raise

    output_path = Path(output_indices_path) if output_indices_path is not None else None
    result = DSAIndexerTopKResult(
        resident_layout_path=layout_path,
        cache_layout_path=cache_layout_p,
        cache_file_path=cache_path,
        hidden_path=hidden_path,
        q_resid_path=q_resid_path,
        output_indices_path=output_path,
        output_indices_u32_path=output_u32_path,
        layer=layer,
        start_position=start_position,
        batch_tokens=batch_tokens,
        context_length=context_length,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=head_dim,
        q_lora_dim=q_lora_dim,
        qk_rope_dim=qk_rope_dim,
        rope_interleave=rope_interleave,
        cache_read_bytes=cache_read_bytes,
        resident_matrix_bytes=resident_matrix_bytes,
        resident_matrix_f32_bytes=resident_matrix_f32_bytes,
        estimated_peak_bytes=estimated_peak,
        output_indices_u32_bytes=output_indices_u32_bytes,
        topk_indices_collected=collect_topk_indices,
        topk_indices=tuple(topk_rows),
    )
    if output_path is not None:
        _write_topk_json(output_path, result)
    return result


def run_dsa_indexer_batch(
    *,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    layer: int,
    hidden_f32_path: str | Path,
    q_resid_f32_path: str | Path,
    output_indices_path: str | Path | None,
    output_indices_u32_path: str | Path | None = None,
    start_position: int,
    batch_tokens: int,
    context_length: int,
    index_topk: int,
    index_n_heads: int,
    qk_rope_dim: int,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    layer_norm_eps: float = 1e-6,
    max_resident_matrix_mib: int = 512,
    max_cache_file_mib: float = 32768.0,
    max_cache_write_mib: float = 4096.0,
    max_cache_read_mib: float = 4096.0,
    max_runner_scratch_mib: int = 4096,
    collect_topk_indices: bool = True,
) -> DSAIndexerBatchResult:
    cache_write = write_dsa_index_cache_batch(
        resident_layout_path=resident_layout_path,
        cache_layout_path=cache_layout_path,
        cache_file_path=cache_file_path,
        layer=layer,
        hidden_f32_path=hidden_f32_path,
        start_position=start_position,
        batch_tokens=batch_tokens,
        qk_rope_dim=qk_rope_dim,
        rope_theta=rope_theta,
        rope_interleave=rope_interleave,
        layer_norm_eps=layer_norm_eps,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_cache_file_mib=max_cache_file_mib,
        max_cache_write_mib=max_cache_write_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
    )
    topk = compute_dsa_topk_batch(
        resident_layout_path=resident_layout_path,
        cache_layout_path=cache_layout_path,
        cache_file_path=cache_file_path,
        layer=layer,
        hidden_f32_path=hidden_f32_path,
        q_resid_f32_path=q_resid_f32_path,
        output_indices_path=output_indices_path,
        output_indices_u32_path=output_indices_u32_path,
        start_position=start_position,
        batch_tokens=batch_tokens,
        context_length=context_length,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        qk_rope_dim=qk_rope_dim,
        rope_theta=rope_theta,
        rope_interleave=rope_interleave,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_cache_file_mib=max_cache_file_mib,
        max_cache_read_mib=max_cache_read_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        collect_topk_indices=collect_topk_indices,
    )
    return DSAIndexerBatchResult(cache_write=cache_write, topk=topk)
