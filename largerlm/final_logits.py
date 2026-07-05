from __future__ import annotations

import heapq
import json
import math
import os
import subprocess
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .resident_affine import (
    ResidentAffineInt4LayoutInfo,
    ResidentAffineLayoutError,
    resident_affine_int4_layout_info,
)


class FinalLogitsError(RuntimeError):
    """Raised when final logits cannot be computed within configured limits."""


@dataclass(frozen=True)
class LogitRecord:
    token_id: int
    logit: float


@dataclass(frozen=True)
class FinalLogitsResult:
    resident_layout_path: Path
    input_path: Path
    output_logits_path: Path | None
    output_topk_path: Path | None
    norm_tensor: str | None
    head_tensor: str
    hidden_dim: int
    vocab_size: int
    dtype: str
    chunk_rows: int
    chunks: int
    read_bytes: int
    topk: tuple[LogitRecord, ...]


@dataclass(frozen=True)
class ResidentMxfp4LayoutInfo:
    name: str
    weight: dict[str, Any]
    scales: dict[str, Any]
    out_dim: int
    in_dim: int
    group_size: int
    total_bytes: int


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FinalLogitsError(f"failed to read resident layout {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FinalLogitsError(f"failed to parse resident layout {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalLogitsError(f"resident layout {p} must be a JSON object")
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
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _replace_atomic(tmp_path, path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def load_final_logits_topk_json(path: str | Path, *, top_k: int) -> tuple[LogitRecord, ...]:
    top_k = _require_int(top_k, "top_k")
    if top_k <= 0:
        raise FinalLogitsError("top_k must be positive")
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FinalLogitsError(f"failed to read Metal final logits top-k JSON {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FinalLogitsError(f"failed to parse Metal final logits top-k JSON {p}: {exc}") from exc
    raw_records = payload.get("topk")
    if not isinstance(raw_records, list):
        raise FinalLogitsError("Metal final logits top-k JSON missing topk array")
    parsed_records: list[LogitRecord] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            raise FinalLogitsError("Metal final logits top-k entries must be objects")
        token_id = _require_int(item.get("token_id"), f"topk[{index}].token_id")
        if token_id < 0:
            raise FinalLogitsError(f"topk[{index}].token_id must be non-negative")
        logit = _require_number(item.get("logit"), f"topk[{index}].logit")
        parsed_records.append(LogitRecord(token_id=token_id, logit=logit))
    top = tuple(parsed_records)
    if len(top) != top_k:
        raise FinalLogitsError(
            f"Metal final logits returned {len(top)} records, expected {top_k}"
        )
    return top


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.upper()
    if normalized in {"F32", "FLOAT32"}:
        return 4
    if normalized in {"BF16", "BFLOAT16", "F16", "FLOAT16"}:
        return 2
    return 0


def _is_u32_dtype(dtype: object) -> bool:
    return isinstance(dtype, str) and dtype.upper() in {"U32", "UINT32"}


def _is_u8_dtype(dtype: object) -> bool:
    return isinstance(dtype, str) and dtype.upper() in {"U8", "UINT8"}


def _is_int(value: object) -> bool:
    return type(value) is int


def _require_int(value: object, label: str) -> int:
    if not _is_int(value):
        raise FinalLogitsError(f"{label} must be an integer")
    return int(value)


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalLogitsError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FinalLogitsError(f"{label} must be a finite number")
    return parsed


def _require_nonnegative_number(value: object, label: str) -> float:
    parsed = _require_number(value, label)
    if parsed < 0:
        raise FinalLogitsError(f"{label} must be non-negative")
    return parsed


def _int_field(tensor: dict[str, Any], field: str, label: str) -> int:
    return _require_int(tensor.get(field), f"{label} {field}")


def _f16_to_f32(raw: bytes, offset: int) -> float:
    return struct.unpack_from("<e", raw, offset)[0]


def _bf16_to_f32(raw: bytes, offset: int) -> float:
    bits = int.from_bytes(raw[offset : offset + 2], "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def _value_at(raw: bytes, dtype: str, index: int) -> float:
    normalized = dtype.upper()
    if normalized in {"F32", "FLOAT32"}:
        return struct.unpack_from("<f", raw, index * 4)[0]
    if normalized in {"BF16", "BFLOAT16"}:
        return _bf16_to_f32(raw, index * 2)
    if normalized in {"F16", "FLOAT16"}:
        return _f16_to_f32(raw, index * 2)
    raise FinalLogitsError(f"unsupported tensor dtype {dtype}")


def _read_f32_vector(path: str | Path) -> list[float]:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise FinalLogitsError(f"failed to read hidden f32 file {p}: {exc}") from exc
    if len(data) == 0 or len(data) % 4 != 0:
        raise FinalLogitsError("hidden f32 file must contain a non-empty f32 vector")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def _shape2(tensor: dict[str, Any], label: str) -> tuple[int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise FinalLogitsError(f"{label} must have shape [rows, cols]")
    rows, cols = int(shape[0]), int(shape[1])
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise FinalLogitsError(f"unsupported dtype {dtype} for {label}")
    size = _int_field(tensor, "size", label)
    expected = rows * cols * dtype_nbytes
    if size != expected:
        raise FinalLogitsError(f"{label} size {size} does not match expected {expected}")
    return rows, cols


def _raw_shape2(tensor: dict[str, Any], label: str) -> tuple[int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise FinalLogitsError(f"{label} must have shape [rows, cols]")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise FinalLogitsError(f"{label} dimensions must be positive")
    return rows, cols


def _affine_info(
    layout: dict[str, Any],
    tensor: dict[str, Any],
) -> ResidentAffineInt4LayoutInfo | None:
    try:
        return resident_affine_int4_layout_info(layout, tensor)
    except ResidentAffineLayoutError as exc:
        raise FinalLogitsError(str(exc)) from exc


def _mxfp4_info(
    layout: dict[str, Any],
    tensor: dict[str, Any],
) -> ResidentMxfp4LayoutInfo | None:
    if not _is_u32_dtype(tensor.get("dtype")):
        return None
    name = tensor.get("name")
    if not isinstance(name, str) or not name.endswith(".weight"):
        return None
    by_name = {
        str(item.get("name")): item
        for item in _tensors(layout)
        if isinstance(item.get("name"), str)
    }
    base = name[: -len(".weight")]
    scales = by_name.get(f"{base}.scales")
    biases = by_name.get(f"{base}.biases")
    if scales is None or biases is not None:
        return None
    if not _is_u8_dtype(scales.get("dtype")):
        return None
    out_dim, packed_cols = _raw_shape2(tensor, name)
    scale_rows, groups = _raw_shape2(scales, f"{base}.scales")
    if scale_rows != out_dim:
        raise FinalLogitsError(
            f"resident MXFP4 scales for {name} must have shape [out_dim, groups]"
        )
    in_dim = packed_cols * 8
    if groups <= 0 or in_dim % groups != 0:
        raise FinalLogitsError(
            f"resident MXFP4 groups for {name} do not divide logical input dim {in_dim}"
        )
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise FinalLogitsError(
            f"resident MXFP4 group size for {name} must be a positive multiple of 8"
        )
    weight_size = _int_field(tensor, "size", name)
    scale_size = _int_field(scales, "size", f"{base}.scales")
    expected_weight = out_dim * packed_cols * 4
    expected_scale = out_dim * groups
    if weight_size != expected_weight:
        raise FinalLogitsError(
            f"resident MXFP4 weight {name} size does not match packed shape"
        )
    if scale_size != expected_scale:
        raise FinalLogitsError(
            f"resident MXFP4 scales for {name} size does not match metadata shape"
        )
    return ResidentMxfp4LayoutInfo(
        name=name,
        weight=tensor,
        scales=scales,
        out_dim=out_dim,
        in_dim=in_dim,
        group_size=group_size,
        total_bytes=weight_size + scale_size,
    )


def _head_shape(
    layout: dict[str, Any],
    tensor: dict[str, Any],
    label: str,
) -> tuple[int, int, ResidentAffineInt4LayoutInfo | None, ResidentMxfp4LayoutInfo | None]:
    mxfp4 = _mxfp4_info(layout, tensor)
    if mxfp4 is not None:
        return mxfp4.out_dim, mxfp4.in_dim, None, mxfp4
    affine = _affine_info(layout, tensor)
    if affine is not None:
        return affine.out_dim, affine.in_dim, affine, None
    rows, cols = _shape2(tensor, label)
    return rows, cols, None, None


def _affine_row_bytes(info: ResidentAffineInt4LayoutInfo) -> tuple[int, int, int, int]:
    if info.out_dim <= 0:
        raise FinalLogitsError(f"resident affine-int4 head {info.name} has no rows")
    if info.weight.get("size") % info.out_dim != 0:
        raise FinalLogitsError(
            f"resident affine-int4 weight {info.name} size is not row-aligned"
        )
    if info.scales.get("size") % info.out_dim != 0:
        raise FinalLogitsError(
            f"resident affine-int4 scales for {info.name} size is not row-aligned"
        )
    if info.biases.get("size") % info.out_dim != 0:
        raise FinalLogitsError(
            f"resident affine-int4 biases for {info.name} size is not row-aligned"
        )
    weight_row_bytes = int(info.weight["size"]) // info.out_dim
    scale_row_bytes = int(info.scales["size"]) // info.out_dim
    bias_row_bytes = int(info.biases["size"]) // info.out_dim
    return (
        weight_row_bytes,
        scale_row_bytes,
        bias_row_bytes,
        weight_row_bytes + scale_row_bytes + bias_row_bytes,
    )


def _mxfp4_row_bytes(info: ResidentMxfp4LayoutInfo) -> tuple[int, int, int]:
    if info.out_dim <= 0:
        raise FinalLogitsError(f"resident MXFP4 head {info.name} has no rows")
    if info.weight.get("size") % info.out_dim != 0:
        raise FinalLogitsError(
            f"resident MXFP4 weight {info.name} size is not row-aligned"
        )
    if info.scales.get("size") % info.out_dim != 0:
        raise FinalLogitsError(
            f"resident MXFP4 scales for {info.name} size is not row-aligned"
        )
    weight_row_bytes = int(info.weight["size"]) // info.out_dim
    scale_row_bytes = int(info.scales["size"]) // info.out_dim
    return weight_row_bytes, scale_row_bytes, weight_row_bytes + scale_row_bytes


def _shape1(tensor: dict[str, Any], label: str) -> int:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) < 1 or not _is_int(shape[0]):
        raise FinalLogitsError(f"{label} must have shape [dim]")
    dim = int(shape[0])
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise FinalLogitsError(f"unsupported dtype {dtype} for {label}")
    size = _int_field(tensor, "size", label)
    expected = dim * dtype_nbytes
    if size != expected:
        raise FinalLogitsError(f"{label} size {size} does not match expected {expected}")
    return dim


def _check_tensor_backing_span(
    weight_path: Path,
    tensor: dict[str, Any],
    label: str,
) -> None:
    offset = tensor.get("offset")
    size = tensor.get("size")
    if type(offset) is not int or offset < 0 or type(size) is not int or size < 0:
        raise FinalLogitsError(f"{label} must have non-negative offset and size")
    try:
        file_bytes = weight_path.stat().st_size
    except OSError as exc:
        raise FinalLogitsError(
            f"failed to stat resident weight file {weight_path}: {exc}"
        ) from exc
    end = int(offset) + int(size)
    if end > file_bytes:
        raise FinalLogitsError(
            f"{label} extends beyond resident weight file: {end} > {file_bytes}"
        )


def _tensors(layout: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise FinalLogitsError("resident layout missing tensors array")
    return [tensor for tensor in tensors if isinstance(tensor, dict)]


def _find_global_tensor(
    layout: dict[str, Any],
    suffixes: tuple[str, ...],
    *,
    label: str,
    required: bool = True,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for tensor in _tensors(layout):
        name = tensor.get("name")
        if not isinstance(name, str) or ".layers." in name:
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            matches.append(tensor)
    if not matches:
        if required:
            raise FinalLogitsError(f"{label} tensor not found")
        return None
    matches.sort(key=lambda item: len(str(item.get("name") or "")))
    return matches[0]


def _read_tensor_vector_f32(weight_path: Path, tensor: dict[str, Any], dim: int) -> list[float]:
    dtype = str(tensor.get("dtype") or "")
    _check_tensor_backing_span(
        weight_path,
        tensor,
        str(tensor.get("name") or "resident vector"),
    )
    offset = int(tensor.get("offset") or 0)
    size = int(tensor.get("size") or 0)
    fd = os.open(weight_path, os.O_RDONLY)
    try:
        raw = os.pread(fd, size, offset)
        if len(raw) != size:
            raise FinalLogitsError(f"short read for tensor {tensor.get('name')}")
    finally:
        os.close(fd)
    if dim * _dtype_bytes(dtype) != size:
        raise FinalLogitsError(f"vector {tensor.get('name')} size mismatch")
    return [_value_at(raw, dtype, i) for i in range(dim)]


def _rmsnorm(values: list[float], weights: list[float], eps: float) -> list[float]:
    if len(values) != len(weights):
        raise FinalLogitsError("RMSNorm weight dim does not match hidden dim")
    inv = 1.0 / math.sqrt(sum(v * v for v in values) / len(values) + eps)
    return [value * inv * weight for value, weight in zip(values, weights)]


def _mxfp4_e2m1_to_f32(value: int) -> float:
    mag = value & 0x7
    if mag == 0:
        return 0.0
    lookup = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    sign = -1.0 if value & 0x8 else 1.0
    return sign * lookup[mag]


def _mxfp4_e8m0_to_f32(value: int) -> float:
    return struct.unpack("<f", ((value & 0xFF) << 23).to_bytes(4, "little"))[0]


def _write_topk(path: str | Path, records: tuple[LogitRecord, ...]) -> None:
    payload = {
        "topk": [
            {"token_id": record.token_id, "logit": record.logit}
            for record in records
        ]
    }
    _write_json_atomic(Path(path), payload)


def compute_final_logits(
    resident_layout_path: str | Path,
    input_f32_path: str | Path,
    *,
    output_logits_f32_path: str | Path | None = None,
    output_topk_json_path: str | Path | None = None,
    top_k: int = 1,
    rms_norm_eps: float = 1e-5,
    chunk_rows: int | None = None,
    max_chunk_bytes: int = 64 * 1024**2,
    max_output_logits_bytes: int = 64 * 1024**2,
    allow_tied_embeddings: bool = True,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
    skip_final_norm: bool = False,
) -> FinalLogitsResult:
    top_k = _require_int(top_k, "top_k")
    max_chunk_bytes = _require_int(max_chunk_bytes, "max_chunk_bytes")
    max_output_logits_bytes = _require_int(
        max_output_logits_bytes, "max_output_logits_bytes"
    )
    if chunk_rows is not None:
        chunk_rows = _require_int(chunk_rows, "chunk_rows")
    if expected_vocab_size is not None:
        expected_vocab_size = _require_int(expected_vocab_size, "expected_vocab_size")
    if expected_hidden_size is not None:
        expected_hidden_size = _require_int(expected_hidden_size, "expected_hidden_size")
    rms_norm_eps = _require_nonnegative_number(rms_norm_eps, "rms_norm_eps")
    if top_k <= 0:
        raise FinalLogitsError("top_k must be positive")
    if expected_vocab_size is not None and expected_vocab_size <= 0:
        raise FinalLogitsError("expected_vocab_size must be positive")
    if expected_hidden_size is not None and expected_hidden_size <= 0:
        raise FinalLogitsError("expected_hidden_size must be positive")
    if chunk_rows is not None and chunk_rows <= 0:
        raise FinalLogitsError("chunk_rows must be positive")
    if max_chunk_bytes <= 0 or max_output_logits_bytes <= 0:
        raise FinalLogitsError("max chunk/output bytes must be positive")

    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise FinalLogitsError("resident layout missing weight_file")
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise FinalLogitsError(f"resident weight file not found: {weight_path}")

    head = _find_global_tensor(
        layout,
        ("lm_head.weight", ".lm_head.weight"),
        label="lm_head",
        required=False,
    )
    if head is None and allow_tied_embeddings:
        head = _find_global_tensor(
            layout,
            (
                "model.embed_tokens.weight",
                ".embed_tokens.weight",
                "transformer.word_embeddings.weight",
                ".word_embeddings.weight",
            ),
            label="tied embedding",
            required=False,
        )
    if head is None:
        raise FinalLogitsError("lm_head.weight not found")

    vocab_size, hidden_dim, affine, mxfp4 = _head_shape(
        layout,
        head,
        "lm_head/embedding",
    )
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise FinalLogitsError(
            f"lm_head/embedding vocab size {vocab_size} does not match "
            f"expected_vocab_size {expected_vocab_size}"
        )
    if expected_hidden_size is not None and hidden_dim != expected_hidden_size:
        raise FinalLogitsError(
            f"lm_head/embedding hidden dim {hidden_dim} does not match "
            f"expected_hidden_size {expected_hidden_size}"
        )
    if mxfp4 is not None:
        _check_tensor_backing_span(weight_path, mxfp4.weight, "resident MXFP4 lm_head weight")
        _check_tensor_backing_span(weight_path, mxfp4.scales, "resident MXFP4 lm_head scales")
    elif affine is not None:
        _check_tensor_backing_span(weight_path, affine.weight, "resident affine-int4 lm_head weight")
        _check_tensor_backing_span(weight_path, affine.scales, "resident affine-int4 lm_head scales")
        _check_tensor_backing_span(weight_path, affine.biases, "resident affine-int4 lm_head biases")
    else:
        _check_tensor_backing_span(
            weight_path,
            head,
            str(head.get("name") or "lm_head/embedding"),
        )
    if top_k > vocab_size:
        raise FinalLogitsError(f"top_k {top_k} exceeds vocab size {vocab_size}")
    hidden = _read_f32_vector(input_f32_path)
    if len(hidden) != hidden_dim:
        raise FinalLogitsError(
            f"hidden dim {len(hidden)} does not match lm_head input dim {hidden_dim}"
        )

    norm_name: str | None = None
    if not skip_final_norm:
        norm = _find_global_tensor(
            layout,
            (
                "model.norm.weight",
                ".model.norm.weight",
                "transformer.norm.weight",
                ".transformer.norm.weight",
                "norm.weight",
                ".norm.weight",
            ),
            label="final norm",
            required=True,
        )
        assert norm is not None
        norm_dim = _shape1(norm, "final norm")
        if norm_dim != hidden_dim:
            raise FinalLogitsError(
                f"final norm dim {norm_dim} does not match hidden dim {hidden_dim}"
            )
        _check_tensor_backing_span(
            weight_path,
            norm,
            str(norm.get("name") or "final norm"),
        )
        weights = _read_tensor_vector_f32(weight_path, norm, norm_dim)
        hidden = _rmsnorm(hidden, weights, rms_norm_eps)
        norm_name = str(norm.get("name"))

    dtype = (
        "mlx-mxfp4"
        if mxfp4 is not None
        else ("affine-int4" if affine is not None else str(head.get("dtype") or ""))
    )
    if mxfp4 is not None:
        weight_row_bytes, scale_row_bytes, row_bytes = _mxfp4_row_bytes(mxfp4)
        bias_row_bytes = 0
        dtype_nbytes = 0
    elif affine is not None:
        weight_row_bytes, scale_row_bytes, bias_row_bytes, row_bytes = _affine_row_bytes(
            affine
        )
        dtype_nbytes = 0
    else:
        dtype_nbytes = _dtype_bytes(dtype)
        row_bytes = hidden_dim * dtype_nbytes
    if row_bytes > max_chunk_bytes:
        raise FinalLogitsError(
            f"one lm_head row {row_bytes} bytes exceeds chunk limit {max_chunk_bytes}"
        )
    rows_per_chunk = chunk_rows or max(1, max_chunk_bytes // row_bytes)
    rows_per_chunk = min(rows_per_chunk, vocab_size)
    if rows_per_chunk <= 0:
        raise FinalLogitsError("chunk_rows must be positive")
    chunk_bytes = rows_per_chunk * row_bytes
    if chunk_bytes > max_chunk_bytes:
        raise FinalLogitsError(
            f"logit chunk {chunk_bytes} bytes exceeds limit {max_chunk_bytes}"
        )

    logits_out = None
    out_path: Path | None = Path(output_logits_f32_path) if output_logits_f32_path else None
    logits_tmp_path: Path | None = None
    cleanup_logits_tmp = False
    try:
        if out_path is not None:
            output_bytes = vocab_size * 4
            if output_bytes > max_output_logits_bytes:
                raise FinalLogitsError(
                    f"output logits {output_bytes} bytes exceeds limit "
                    f"{max_output_logits_bytes}"
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            logits_tmp_path = out_path.with_name(out_path.name + ".tmp")
            cleanup_logits_tmp = True
            logits_out = os.open(logits_tmp_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(logits_out, output_bytes)

        fd = os.open(weight_path, os.O_RDONLY)
        heap: list[tuple[float, int]] = []
        chunks = 0
        read_bytes = 0
        try:
            for row_start in range(0, vocab_size, rows_per_chunk):
                rows = min(rows_per_chunk, vocab_size - row_start)
                if mxfp4 is not None:
                    weight_size = rows * weight_row_bytes
                    scale_size = rows * scale_row_bytes
                    weight_raw = os.pread(
                        fd,
                        weight_size,
                        int(mxfp4.weight["offset"]) + row_start * weight_row_bytes,
                    )
                    scale_raw = os.pread(
                        fd,
                        scale_size,
                        int(mxfp4.scales["offset"]) + row_start * scale_row_bytes,
                    )
                    if len(weight_raw) != weight_size or len(scale_raw) != scale_size:
                        raise FinalLogitsError(
                            f"short MXFP4 lm_head read at row {row_start}"
                        )
                    raw_size = weight_size + scale_size
                elif affine is not None:
                    weight_size = rows * weight_row_bytes
                    scale_size = rows * scale_row_bytes
                    bias_size = rows * bias_row_bytes
                    weight_raw = os.pread(
                        fd,
                        weight_size,
                        int(affine.weight["offset"]) + row_start * weight_row_bytes,
                    )
                    scale_raw = os.pread(
                        fd,
                        scale_size,
                        int(affine.scales["offset"]) + row_start * scale_row_bytes,
                    )
                    bias_raw = os.pread(
                        fd,
                        bias_size,
                        int(affine.biases["offset"]) + row_start * bias_row_bytes,
                    )
                    if (
                        len(weight_raw) != weight_size
                        or len(scale_raw) != scale_size
                        or len(bias_raw) != bias_size
                    ):
                        raise FinalLogitsError(
                            f"short affine-int4 lm_head read at row {row_start}"
                        )
                    raw_size = weight_size + scale_size + bias_size
                else:
                    raw_size = rows * row_bytes
                    raw_offset = int(head.get("offset") or 0) + row_start * row_bytes
                    raw = os.pread(fd, raw_size, raw_offset)
                    if len(raw) != raw_size:
                        raise FinalLogitsError(
                            f"short lm_head read at row {row_start}: got {len(raw)}, "
                            f"expected {raw_size}"
                        )
                read_bytes += raw_size
                chunks += 1
                chunk_logits = bytearray(rows * 4) if logits_out is not None else None
                for local_row in range(rows):
                    acc = 0.0
                    if mxfp4 is not None:
                        groups = hidden_dim // mxfp4.group_size
                        packed_per_group = mxfp4.group_size // 8
                        weight_base = local_row * weight_row_bytes
                        meta_base = local_row * groups
                        for group in range(groups):
                            scale = _mxfp4_e8m0_to_f32(scale_raw[meta_base + group])
                            base_packed = group * packed_per_group
                            base_x = group * mxfp4.group_size
                            for packed_index in range(packed_per_group):
                                packed = struct.unpack_from(
                                    "<I",
                                    weight_raw,
                                    weight_base + (base_packed + packed_index) * 4,
                                )[0]
                                for nibble_index in range(8):
                                    weight = _mxfp4_e2m1_to_f32(
                                        (packed >> (nibble_index * 4)) & 0xF
                                    ) * scale
                                    acc += weight * hidden[
                                        base_x + packed_index * 8 + nibble_index
                                    ]
                    elif affine is not None:
                        groups = hidden_dim // affine.group_size
                        packed_per_group = affine.group_size // 8
                        weight_base = local_row * weight_row_bytes
                        meta_base = local_row * groups
                        scale_dtype = str(affine.scales.get("dtype") or "")
                        bias_dtype = str(affine.biases.get("dtype") or "")
                        for group in range(groups):
                            scale = _value_at(scale_raw, scale_dtype, meta_base + group)
                            bias = _value_at(bias_raw, bias_dtype, meta_base + group)
                            base_packed = group * packed_per_group
                            base_x = group * affine.group_size
                            for packed_index in range(packed_per_group):
                                packed = struct.unpack_from(
                                    "<I",
                                    weight_raw,
                                    weight_base + (base_packed + packed_index) * 4,
                                )[0]
                                for nibble_index in range(8):
                                    weight = (
                                        float((packed >> (nibble_index * 4)) & 0xF)
                                        * scale
                                        + bias
                                    )
                                    acc += weight * hidden[
                                        base_x + packed_index * 8 + nibble_index
                                    ]
                    else:
                        base = local_row * hidden_dim
                        for i, value in enumerate(hidden):
                            acc += value * _value_at(raw, dtype, base + i)
                    token_id = row_start + local_row
                    if len(heap) < top_k:
                        heapq.heappush(heap, (acc, token_id))
                    elif acc > heap[0][0]:
                        heapq.heapreplace(heap, (acc, token_id))
                    if chunk_logits is not None:
                        struct.pack_into("<f", chunk_logits, local_row * 4, acc)
                if logits_out is not None and chunk_logits is not None:
                    os.pwrite(logits_out, chunk_logits, row_start * 4)
        finally:
            os.close(fd)
            if logits_out is not None:
                os.close(logits_out)
                logits_out = None

        top = tuple(
            LogitRecord(token_id=token_id, logit=float(logit))
            for logit, token_id in sorted(heap, key=lambda item: (-item[0], item[1]))
        )
        if logits_tmp_path is not None and out_path is not None:
            _replace_atomic(logits_tmp_path, out_path)
            cleanup_logits_tmp = False

        topk_path = Path(output_topk_json_path) if output_topk_json_path else None
        if topk_path is not None:
            topk_path.parent.mkdir(parents=True, exist_ok=True)
            _write_topk(topk_path, top)
    except Exception:
        if logits_out is not None:
            try:
                os.close(logits_out)
            except OSError:
                pass
        if cleanup_logits_tmp and logits_tmp_path is not None:
            _remove_partial_file(logits_tmp_path)
        raise

    return FinalLogitsResult(
        resident_layout_path=layout_path,
        input_path=Path(input_f32_path),
        output_logits_path=out_path,
        output_topk_path=topk_path,
        norm_tensor=norm_name,
        head_tensor=str(head.get("name")),
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dtype=dtype,
        chunk_rows=rows_per_chunk,
        chunks=chunks,
        read_bytes=read_bytes,
        topk=top,
    )


def compute_final_logits_metal(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    input_f32_path: str | Path,
    output_topk_json_path: str | Path | None = None,
    top_k: int = 1,
    rms_norm_eps: float = 1e-5,
    chunk_rows: int | None = None,
    max_chunk_bytes: int = 64 * 1024**2,
    max_runner_scratch_bytes: int = 4 * 1024**3,
    allow_tied_embeddings: bool = True,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
    skip_final_norm: bool = False,
    echo_runner_output: bool = True,
) -> FinalLogitsResult:
    top_k = _require_int(top_k, "top_k")
    max_chunk_bytes = _require_int(max_chunk_bytes, "max_chunk_bytes")
    max_runner_scratch_bytes = _require_int(
        max_runner_scratch_bytes, "max_runner_scratch_bytes"
    )
    if chunk_rows is not None:
        chunk_rows = _require_int(chunk_rows, "chunk_rows")
    if expected_vocab_size is not None:
        expected_vocab_size = _require_int(expected_vocab_size, "expected_vocab_size")
    if expected_hidden_size is not None:
        expected_hidden_size = _require_int(expected_hidden_size, "expected_hidden_size")
    rms_norm_eps = _require_nonnegative_number(rms_norm_eps, "rms_norm_eps")
    if top_k <= 0:
        raise FinalLogitsError("top_k must be positive")
    if expected_vocab_size is not None and expected_vocab_size <= 0:
        raise FinalLogitsError("expected_vocab_size must be positive")
    if expected_hidden_size is not None and expected_hidden_size <= 0:
        raise FinalLogitsError("expected_hidden_size must be positive")
    if top_k > 64:
        raise FinalLogitsError("Metal final logits currently supports top_k <= 64")
    if chunk_rows is not None and chunk_rows <= 0:
        raise FinalLogitsError("chunk_rows must be positive")
    if max_chunk_bytes <= 0 or max_runner_scratch_bytes <= 0:
        raise FinalLogitsError("max chunk/scratch bytes must be positive")

    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise FinalLogitsError("resident layout missing weight_file")
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise FinalLogitsError(f"resident weight file not found: {weight_path}")
    head = _find_global_tensor(
        layout,
        ("lm_head.weight", ".lm_head.weight"),
        label="lm_head",
        required=False,
    )
    if head is None and allow_tied_embeddings:
        head = _find_global_tensor(
            layout,
            (
                "model.embed_tokens.weight",
                ".embed_tokens.weight",
                "transformer.word_embeddings.weight",
                ".word_embeddings.weight",
            ),
            label="tied embedding",
            required=False,
        )
    if head is None:
        raise FinalLogitsError("lm_head.weight not found")
    vocab_size, hidden_dim, affine, mxfp4 = _head_shape(
        layout,
        head,
        "lm_head/embedding",
    )
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise FinalLogitsError(
            f"lm_head/embedding vocab size {vocab_size} does not match "
            f"expected_vocab_size {expected_vocab_size}"
        )
    if expected_hidden_size is not None and hidden_dim != expected_hidden_size:
        raise FinalLogitsError(
            f"lm_head/embedding hidden dim {hidden_dim} does not match "
            f"expected_hidden_size {expected_hidden_size}"
        )
    if mxfp4 is not None:
        _check_tensor_backing_span(weight_path, mxfp4.weight, "resident MXFP4 lm_head weight")
        _check_tensor_backing_span(weight_path, mxfp4.scales, "resident MXFP4 lm_head scales")
    elif affine is not None:
        _check_tensor_backing_span(weight_path, affine.weight, "resident affine-int4 lm_head weight")
        _check_tensor_backing_span(weight_path, affine.scales, "resident affine-int4 lm_head scales")
        _check_tensor_backing_span(weight_path, affine.biases, "resident affine-int4 lm_head biases")
    else:
        _check_tensor_backing_span(
            weight_path,
            head,
            str(head.get("name") or "lm_head/embedding"),
        )
    if top_k > vocab_size:
        raise FinalLogitsError(f"top_k {top_k} exceeds vocab size {vocab_size}")
    hidden = _read_f32_vector(input_f32_path)
    if len(hidden) != hidden_dim:
        raise FinalLogitsError(
            f"hidden dim {len(hidden)} does not match lm_head input dim {hidden_dim}"
        )

    norm_name: str | None = None
    if not skip_final_norm:
        norm = _find_global_tensor(
            layout,
            (
                "model.norm.weight",
                ".model.norm.weight",
                "transformer.norm.weight",
                ".transformer.norm.weight",
                "norm.weight",
                ".norm.weight",
            ),
            label="final norm",
            required=True,
        )
        assert norm is not None
        norm_dim = _shape1(norm, "final norm")
        if norm_dim != hidden_dim:
            raise FinalLogitsError(
                f"final norm dim {norm_dim} does not match hidden dim {hidden_dim}"
            )
        _check_tensor_backing_span(
            weight_path,
            norm,
            str(norm.get("name") or "final norm"),
        )
        norm_name = str(norm.get("name"))

    dtype = (
        "mlx-mxfp4"
        if mxfp4 is not None
        else ("affine-int4" if affine is not None else str(head.get("dtype") or ""))
    )
    if mxfp4 is not None:
        row_bytes = _mxfp4_row_bytes(mxfp4)[2]
    elif affine is not None:
        row_bytes = _affine_row_bytes(affine)[3]
    else:
        dtype_nbytes = _dtype_bytes(dtype)
        row_bytes = hidden_dim * dtype_nbytes
    if row_bytes > max_chunk_bytes:
        raise FinalLogitsError(
            f"one lm_head row {row_bytes} bytes exceeds chunk limit {max_chunk_bytes}"
        )
    rows_per_chunk = chunk_rows or max(1, max_chunk_bytes // row_bytes)
    rows_per_chunk = min(rows_per_chunk, vocab_size)
    chunk_bytes = rows_per_chunk * row_bytes
    if chunk_bytes > max_chunk_bytes:
        raise FinalLogitsError(
            f"logit chunk {chunk_bytes} bytes exceeds limit {max_chunk_bytes}"
        )

    def _run(out_topk: Path) -> None:
        cmd = [
            str(runner_path),
            "--resident-layout",
            str(layout_path),
            "--run-final-logits",
            "--input-f32",
            str(input_f32_path),
            "--output-topk-json",
            str(out_topk),
            "--top-k",
            str(top_k),
            "--rms-norm-eps",
            f"{rms_norm_eps:.9g}",
            "--max-chunk-mib",
            f"{max_chunk_bytes / 1024**2:.9g}",
            "--max-runner-scratch-mib",
            f"{max_runner_scratch_bytes / 1024**2:.9g}",
        ]
        if chunk_rows is not None:
            cmd.extend(["--chunk-rows", str(chunk_rows)])
        if not allow_tied_embeddings:
            cmd.append("--no-tied-embeddings")
        if skip_final_norm:
            cmd.append("--skip-final-norm")
        completed = subprocess.run(cmd, text=True, capture_output=True)
        if echo_runner_output:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="")
        if completed.returncode != 0:
            detail = ""
            if completed.stdout:
                detail += f"\nstdout:\n{completed.stdout[-4000:]}"
            if completed.stderr:
                detail += f"\nstderr:\n{completed.stderr[-4000:]}"
            raise FinalLogitsError(
                f"Metal final logits failed with exit {completed.returncode}: "
                f"{' '.join(cmd)}{detail}"
            )

    topk_path = Path(output_topk_json_path) if output_topk_json_path else None
    if topk_path is not None:
        topk_path.parent.mkdir(parents=True, exist_ok=True)
        _run(topk_path)
        top = load_final_logits_topk_json(topk_path, top_k=top_k)
    else:
        with tempfile.TemporaryDirectory(prefix="largerlm-final-logits-metal-") as tmp_s:
            tmp_topk = Path(tmp_s) / "topk.json"
            _run(tmp_topk)
            top = load_final_logits_topk_json(tmp_topk, top_k=top_k)

    chunks = (vocab_size + rows_per_chunk - 1) // rows_per_chunk
    return FinalLogitsResult(
        resident_layout_path=layout_path,
        input_path=Path(input_f32_path),
        output_logits_path=None,
        output_topk_path=topk_path,
        norm_tensor=norm_name,
        head_tensor=str(head.get("name")),
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dtype=dtype,
        chunk_rows=rows_per_chunk,
        chunks=chunks,
        read_bytes=(
            mxfp4.total_bytes
            if mxfp4 is not None
            else affine.total_bytes
            if affine is not None
            else _int_field(head, "size", str(head.get("name") or "lm_head/embedding"))
        ),
        topk=top,
    )
