from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class EmbeddingError(RuntimeError):
    """Raised when a token embedding cannot be streamed safely."""


@dataclass(frozen=True)
class EmbeddingResult:
    resident_layout_path: Path
    output_path: Path
    tensor: str
    token_id: int
    vocab_size: int
    hidden_dim: int
    dtype: str
    read_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class EmbeddingBatchResult:
    resident_layout_path: Path
    output_path: Path
    tensor: str
    token_count: int
    first_token_id: int | None
    last_token_id: int | None
    vocab_size: int
    hidden_dim: int
    dtype: str
    read_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class _EmbeddingMetadata:
    layout_path: Path
    weight_path: Path
    tensor: dict[str, Any]
    scales: dict[str, Any] | None
    vocab_size: int
    hidden_dim: int
    dtype: str
    row_bytes: int
    weight_row_bytes: int
    scale_row_bytes: int
    group_size: int


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EmbeddingError(f"failed to read resident layout {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EmbeddingError(f"failed to parse resident layout {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EmbeddingError(f"resident layout {p} must be a JSON object")
    return payload


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"F32", "float32"}:
        return 4
    if dtype in {"BF16", "bfloat16", "F16", "float16"}:
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
        raise EmbeddingError(f"{label} must be an integer")
    return int(value)


def _optional_positive_int(value: object | None, label: str) -> int | None:
    if value is None:
        return None
    value = _require_int(value, label)
    if value <= 0:
        raise EmbeddingError(f"{label} must be positive")
    return value


def _int_field(tensor: dict[str, Any], field: str, label: str) -> int:
    return _require_int(tensor.get(field), f"{label} {field}")


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _value_at(raw: bytes, dtype: str, index: int) -> float:
    if dtype in {"F32", "float32"}:
        return struct.unpack_from("<f", raw, index * 4)[0]
    if dtype in {"BF16", "bfloat16"}:
        bits = int.from_bytes(raw[index * 2 : index * 2 + 2], "little") << 16
        return struct.unpack("<f", bits.to_bytes(4, "little"))[0]
    if dtype in {"F16", "float16"}:
        return struct.unpack_from("<e", raw, index * 2)[0]
    raise EmbeddingError(f"unsupported embedding dtype {dtype}")


def _row_to_f32_bytes(raw: bytes, dtype: str, hidden: int) -> bytes:
    if dtype in {"F32", "float32"}:
        return raw
    values = [_value_at(raw, dtype, i) for i in range(hidden)]
    return struct.pack(f"<{hidden}f", *values)


def _mxfp4_e2m1_to_f32(value: int) -> float:
    mag = value & 0x7
    if mag == 0:
        return 0.0
    lookup = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    sign = -1.0 if value & 0x8 else 1.0
    return sign * lookup[mag]


def _mxfp4_e8m0_to_f32(value: int) -> float:
    return struct.unpack("<f", ((value & 0xFF) << 23).to_bytes(4, "little"))[0]


def _mxfp4_row_to_f32_bytes(
    weight_raw: bytes,
    scale_raw: bytes,
    *,
    hidden: int,
    group_size: int,
) -> bytes:
    groups = hidden // group_size
    packed_per_group = group_size // 8
    values: list[float] = []
    for group in range(groups):
        scale = _mxfp4_e8m0_to_f32(scale_raw[group])
        base_packed = group * packed_per_group
        for packed_index in range(packed_per_group):
            packed = struct.unpack_from(
                "<I",
                weight_raw,
                (base_packed + packed_index) * 4,
            )[0]
            for nibble_index in range(8):
                values.append(
                    _mxfp4_e2m1_to_f32((packed >> (nibble_index * 4)) & 0xF)
                    * scale
                )
    return struct.pack(f"<{hidden}f", *values)


def _find_embedding(layout: dict[str, Any]) -> dict[str, Any]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise EmbeddingError("resident layout missing tensors array")
    suffixes = (
        "model.embed_tokens.weight",
        ".embed_tokens.weight",
        "transformer.word_embeddings.weight",
        ".word_embeddings.weight",
    )
    matches: list[dict[str, Any]] = []
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        if not isinstance(name, str) or ".layers." in name:
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            matches.append(tensor)
    if not matches:
        raise EmbeddingError("embedding tensor not found")
    matches.sort(key=lambda item: len(str(item.get("name") or "")))
    return matches[0]


def _shape2(tensor: dict[str, Any]) -> tuple[int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise EmbeddingError("embedding tensor must have shape [vocab, hidden]")
    vocab, hidden = int(shape[0]), int(shape[1])
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise EmbeddingError(f"unsupported embedding dtype {dtype}")
    size = _int_field(tensor, "size", "embedding")
    expected = vocab * hidden * dtype_nbytes
    if size != expected:
        raise EmbeddingError(f"embedding size {size} does not match expected {expected}")
    return vocab, hidden


def _raw_shape2(tensor: dict[str, Any], label: str) -> tuple[int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise EmbeddingError(f"{label} must have shape [vocab, hidden]")
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0:
        raise EmbeddingError(f"{label} dimensions must be positive")
    return rows, cols


def _tensors_by_name(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise EmbeddingError("resident layout missing tensors array")
    result: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            result[name] = item
    return result


def _mxfp4_embedding_metadata(
    layout: dict[str, Any],
    tensor: dict[str, Any],
) -> tuple[dict[str, Any], int, int, int, int, int] | None:
    if not _is_u32_dtype(tensor.get("dtype")):
        return None
    name = tensor.get("name")
    if not isinstance(name, str) or not name.endswith(".weight"):
        return None
    base = name[: -len(".weight")]
    by_name = _tensors_by_name(layout)
    scales = by_name.get(f"{base}.scales")
    biases = by_name.get(f"{base}.biases")
    if scales is None or biases is not None:
        return None
    if not _is_u8_dtype(scales.get("dtype")):
        return None
    vocab, packed_cols = _raw_shape2(tensor, "embedding tensor")
    scale_rows, groups = _raw_shape2(scales, "embedding scales")
    if scale_rows != vocab:
        raise EmbeddingError("embedding MXFP4 scales must have shape [vocab, groups]")
    hidden = packed_cols * 8
    if groups <= 0 or hidden % groups != 0:
        raise EmbeddingError(
            f"embedding MXFP4 groups do not divide logical hidden dim {hidden}"
        )
    group_size = hidden // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise EmbeddingError("embedding MXFP4 group size must be a positive multiple of 8")
    weight_size = _int_field(tensor, "size", "embedding")
    scale_size = _int_field(scales, "size", "embedding scales")
    expected_weight = vocab * packed_cols * 4
    expected_scale = vocab * groups
    if weight_size != expected_weight:
        raise EmbeddingError(
            f"embedding MXFP4 weight size {weight_size} does not match expected {expected_weight}"
        )
    if scale_size != expected_scale:
        raise EmbeddingError(
            f"embedding MXFP4 scale size {scale_size} does not match expected {expected_scale}"
        )
    weight_row_bytes = packed_cols * 4
    scale_row_bytes = groups
    return scales, vocab, hidden, weight_row_bytes, scale_row_bytes, group_size


def _check_tensor_backing_span(
    weight_path: Path,
    tensor: dict[str, Any],
    label: str,
) -> None:
    offset = tensor.get("offset")
    size = tensor.get("size")
    if type(offset) is not int or offset < 0 or type(size) is not int or size < 0:
        raise EmbeddingError(f"{label} must have non-negative offset and size")
    try:
        file_bytes = weight_path.stat().st_size
    except OSError as exc:
        raise EmbeddingError(
            f"failed to stat resident weight file {weight_path}: {exc}"
        ) from exc
    end = int(offset) + int(size)
    if end > file_bytes:
        raise EmbeddingError(
            f"{label} extends beyond resident weight file: {end} > {file_bytes}"
        )


def _load_embedding_metadata(
    resident_layout_path: str | Path,
) -> _EmbeddingMetadata:
    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise EmbeddingError("resident layout missing weight_file")
    tensor = _find_embedding(layout)
    dtype = str(tensor.get("dtype") or "")
    scales = None
    group_size = 0
    mxfp4 = _mxfp4_embedding_metadata(layout, tensor)
    if mxfp4 is not None:
        scales, vocab, hidden, weight_row_bytes, scale_row_bytes, group_size = mxfp4
        row_bytes = weight_row_bytes + scale_row_bytes
        dtype = "mlx-mxfp4"
    else:
        vocab, hidden = _shape2(tensor)
        weight_row_bytes = hidden * _dtype_bytes(dtype)
        scale_row_bytes = 0
        row_bytes = weight_row_bytes
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise EmbeddingError(f"resident weight file not found: {weight_path}")
    _check_tensor_backing_span(
        weight_path,
        tensor,
        str(tensor.get("name") or "embedding"),
    )
    if scales is not None:
        _check_tensor_backing_span(
            weight_path,
            scales,
            str(scales.get("name") or "embedding scales"),
        )
    return _EmbeddingMetadata(
        layout_path=layout_path,
        weight_path=weight_path,
        tensor=tensor,
        scales=scales,
        vocab_size=vocab,
        hidden_dim=hidden,
        dtype=dtype,
        row_bytes=row_bytes,
        weight_row_bytes=weight_row_bytes,
        scale_row_bytes=scale_row_bytes,
        group_size=group_size,
    )


def _check_embedding_expectations(
    *,
    vocab: int,
    hidden: int,
    expected_vocab_size: int | None,
    expected_hidden_size: int | None,
) -> None:
    if expected_vocab_size is not None and vocab != expected_vocab_size:
        raise EmbeddingError(
            f"embedding vocab size {vocab} does not match "
            f"expected_vocab_size {expected_vocab_size}"
        )
    if expected_hidden_size is not None and hidden != expected_hidden_size:
        raise EmbeddingError(
            f"embedding hidden dim {hidden} does not match "
            f"expected_hidden_size {expected_hidden_size}"
        )


def embed_token(
    resident_layout_path: str | Path,
    *,
    token_id: int,
    output_f32_path: str | Path,
    max_row_bytes: int = 64 * 1024**2,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
) -> EmbeddingResult:
    token_id = _require_int(token_id, "token_id")
    max_row_bytes = _require_int(max_row_bytes, "max_row_bytes")
    expected_vocab_size = _optional_positive_int(
        expected_vocab_size,
        "expected_vocab_size",
    )
    expected_hidden_size = _optional_positive_int(
        expected_hidden_size,
        "expected_hidden_size",
    )
    if token_id < 0:
        raise EmbeddingError("token_id must be non-negative")
    if max_row_bytes <= 0:
        raise EmbeddingError("max_row_bytes must be positive")
    meta = _load_embedding_metadata(resident_layout_path)
    _check_embedding_expectations(
        vocab=meta.vocab_size,
        hidden=meta.hidden_dim,
        expected_vocab_size=expected_vocab_size,
        expected_hidden_size=expected_hidden_size,
    )
    if token_id >= meta.vocab_size:
        raise EmbeddingError(f"token_id {token_id} exceeds vocab size {meta.vocab_size}")
    if meta.row_bytes > max_row_bytes:
        raise EmbeddingError(
            f"embedding row {meta.row_bytes} bytes exceeds limit {max_row_bytes}"
        )
    fd = os.open(meta.weight_path, os.O_RDONLY)
    try:
        raw_offset = int(meta.tensor.get("offset") or 0) + token_id * meta.weight_row_bytes
        raw = os.pread(fd, meta.weight_row_bytes, raw_offset)
        if meta.scales is not None:
            scale_offset = (
                int(meta.scales.get("offset") or 0) + token_id * meta.scale_row_bytes
            )
            scale_raw = os.pread(fd, meta.scale_row_bytes, scale_offset)
        else:
            scale_raw = b""
    finally:
        os.close(fd)
    if len(raw) != meta.weight_row_bytes:
        raise EmbeddingError(
            f"short embedding read: got {len(raw)} bytes, expected {meta.weight_row_bytes}"
        )
    if meta.scales is not None and len(scale_raw) != meta.scale_row_bytes:
        raise EmbeddingError(
            f"short embedding scale read: got {len(scale_raw)} bytes, "
            f"expected {meta.scale_row_bytes}"
        )

    out = Path(output_f32_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if meta.scales is not None:
        out.write_bytes(
            _mxfp4_row_to_f32_bytes(
                raw,
                scale_raw,
                hidden=meta.hidden_dim,
                group_size=meta.group_size,
            )
        )
    else:
        out.write_bytes(_row_to_f32_bytes(raw, meta.dtype, meta.hidden_dim))
    return EmbeddingResult(
        resident_layout_path=meta.layout_path,
        output_path=out,
        tensor=str(meta.tensor.get("name")),
        token_id=token_id,
        vocab_size=meta.vocab_size,
        hidden_dim=meta.hidden_dim,
        dtype=meta.dtype,
        read_bytes=meta.row_bytes,
        output_bytes=meta.hidden_dim * 4,
    )


def embed_tokens_batch(
    resident_layout_path: str | Path,
    *,
    token_ids: Sequence[int],
    output_f32_path: str | Path,
    max_row_bytes: int = 64 * 1024**2,
    max_output_bytes: int = 4 * 1024**3,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
) -> EmbeddingBatchResult:
    if not token_ids:
        raise EmbeddingError("token_ids must not be empty")
    tokens: list[int] = []
    for position, raw_token_id in enumerate(token_ids):
        token_id = _require_int(raw_token_id, f"token_id at position {position}")
        if token_id < 0:
            raise EmbeddingError(f"token_id at position {position} must be non-negative")
        tokens.append(token_id)
    max_row_bytes = _require_int(max_row_bytes, "max_row_bytes")
    max_output_bytes = _require_int(max_output_bytes, "max_output_bytes")
    expected_vocab_size = _optional_positive_int(
        expected_vocab_size,
        "expected_vocab_size",
    )
    expected_hidden_size = _optional_positive_int(
        expected_hidden_size,
        "expected_hidden_size",
    )
    if max_row_bytes <= 0:
        raise EmbeddingError("max_row_bytes must be positive")
    if max_output_bytes <= 0:
        raise EmbeddingError("max_output_bytes must be positive")
    meta = _load_embedding_metadata(resident_layout_path)
    _check_embedding_expectations(
        vocab=meta.vocab_size,
        hidden=meta.hidden_dim,
        expected_vocab_size=expected_vocab_size,
        expected_hidden_size=expected_hidden_size,
    )
    if meta.row_bytes > max_row_bytes:
        raise EmbeddingError(
            f"embedding row {meta.row_bytes} bytes exceeds limit {max_row_bytes}"
        )
    output_bytes = len(tokens) * meta.hidden_dim * 4
    if output_bytes > max_output_bytes:
        raise EmbeddingError(
            f"embedding batch output {output_bytes} bytes exceeds limit {max_output_bytes}"
        )
    for position, token_id in enumerate(tokens):
        if token_id >= meta.vocab_size:
            raise EmbeddingError(
                f"token_id {token_id} at position {position} exceeds vocab size {meta.vocab_size}"
            )

    out = Path(output_f32_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tensor_offset = int(meta.tensor.get("offset") or 0)
    scales_offset = int(meta.scales.get("offset") or 0) if meta.scales is not None else 0
    read_bytes = 0
    weight_fd = os.open(meta.weight_path, os.O_RDONLY)
    try:
        try:
            with out.open("wb") as output:
                for token_id in tokens:
                    raw_offset = tensor_offset + token_id * meta.weight_row_bytes
                    raw = os.pread(weight_fd, meta.weight_row_bytes, raw_offset)
                    if len(raw) != meta.weight_row_bytes:
                        raise EmbeddingError(
                            f"short embedding read for token {token_id}: "
                            f"got {len(raw)} bytes, expected {meta.weight_row_bytes}"
                        )
                    if meta.scales is not None:
                        scale_offset = scales_offset + token_id * meta.scale_row_bytes
                        scale_raw = os.pread(weight_fd, meta.scale_row_bytes, scale_offset)
                        if len(scale_raw) != meta.scale_row_bytes:
                            raise EmbeddingError(
                                f"short embedding scale read for token {token_id}: "
                                f"got {len(scale_raw)} bytes, expected {meta.scale_row_bytes}"
                            )
                        output.write(
                            _mxfp4_row_to_f32_bytes(
                                raw,
                                scale_raw,
                                hidden=meta.hidden_dim,
                                group_size=meta.group_size,
                            )
                        )
                    else:
                        output.write(_row_to_f32_bytes(raw, meta.dtype, meta.hidden_dim))
                    read_bytes += meta.row_bytes
        except (OSError, EmbeddingError):
            _remove_partial_file(out)
            raise
    finally:
        os.close(weight_fd)

    return EmbeddingBatchResult(
        resident_layout_path=meta.layout_path,
        output_path=out,
        tensor=str(meta.tensor.get("name")),
        token_count=len(tokens),
        first_token_id=tokens[0],
        last_token_id=tokens[-1],
        vocab_size=meta.vocab_size,
        hidden_dim=meta.hidden_dim,
        dtype=meta.dtype,
        read_bytes=read_bytes,
        output_bytes=output_bytes,
    )
