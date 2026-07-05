from __future__ import annotations

import json
import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, ModelConfig
from .safety import DiskBudget, disk_budget


class DecodeCacheError(RuntimeError):
    """Raised when decode cache metadata cannot be built safely."""


_DTYPE_BYTES = {
    "BF16": 2,
    "bfloat16": 2,
    "F16": 2,
    "float16": 2,
    "F32": 4,
    "float32": 4,
}


@dataclass(frozen=True)
class DecodeCacheSegment:
    kind: str
    layer: int
    offset: int
    width: int
    dtype: str
    dtype_bytes: int
    max_context_tokens: int

    @property
    def token_stride_bytes(self) -> int:
        return self.width * self.dtype_bytes

    @property
    def total_bytes(self) -> int:
        return self.token_stride_bytes * self.max_context_tokens

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "layer": self.layer,
            "offset": self.offset,
            "width": self.width,
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "token_stride_bytes": self.token_stride_bytes,
            "max_context_tokens": self.max_context_tokens,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class DecodeCacheLayout:
    version: int
    model_type: str
    max_context_tokens: int
    dtype: str
    dtype_bytes: int
    alignment: int
    total_bytes: int
    segments: tuple[DecodeCacheSegment, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "max_context_tokens": self.max_context_tokens,
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "alignment": self.alignment,
            "total_bytes": self.total_bytes,
            "segments": [segment.to_json() for segment in self.segments],
        }


@dataclass(frozen=True)
class DecodeCacheInitResult:
    layout_path: Path
    cache_file_path: Path
    total_bytes: int
    existed: bool
    sparse: bool
    disk_budget: DiskBudget


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise DecodeCacheError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise DecodeCacheError(f"{name} must be an integer") from exc


def _positive_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed <= 0:
        raise DecodeCacheError(f"{name} must be positive")
    return parsed


def _nonnegative_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed < 0:
        raise DecodeCacheError(f"{name} must be non-negative")
    return parsed


def _optional_nonnegative_integer_value(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer_value(name, value)


def _dtype_bytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise DecodeCacheError(f"unsupported cache dtype {dtype}") from exc


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecodeCacheError(f"decode cache layout field {key} must be an integer")
    return int(value)


def load_decode_cache_layout(path: str | Path) -> DecodeCacheLayout:
    layout_path = Path(path)
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DecodeCacheError(f"failed to read decode cache layout {layout_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecodeCacheError(f"failed to parse decode cache layout {layout_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecodeCacheError("decode cache layout must be a JSON object")
    version = _require_int(payload, "version")
    if version != 1:
        raise DecodeCacheError(f"unsupported decode cache layout version {version}")
    model_type = payload.get("model_type")
    dtype = payload.get("dtype")
    if not isinstance(model_type, str) or not isinstance(dtype, str):
        raise DecodeCacheError("decode cache layout missing model_type or dtype")
    dtype_bytes = _require_int(payload, "dtype_bytes")
    if dtype_bytes != _dtype_bytes(dtype):
        raise DecodeCacheError("decode cache layout dtype_bytes does not match dtype")
    max_context_tokens = _require_int(payload, "max_context_tokens")
    alignment = _require_int(payload, "alignment")
    total_bytes = _require_int(payload, "total_bytes")
    if max_context_tokens <= 0 or alignment <= 0 or total_bytes < 0:
        raise DecodeCacheError("decode cache layout has invalid positive-size fields")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise DecodeCacheError("decode cache layout missing segments array")

    segments: list[DecodeCacheSegment] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise DecodeCacheError("decode cache segment must be a JSON object")
        kind = raw.get("kind")
        dtype_segment = raw.get("dtype")
        if not isinstance(kind, str) or not isinstance(dtype_segment, str):
            raise DecodeCacheError("decode cache segment missing kind or dtype")
        segment = DecodeCacheSegment(
            kind=kind,
            layer=_require_int(raw, "layer"),
            offset=_require_int(raw, "offset"),
            width=_require_int(raw, "width"),
            dtype=dtype_segment,
            dtype_bytes=_require_int(raw, "dtype_bytes"),
            max_context_tokens=_require_int(raw, "max_context_tokens"),
        )
        if segment.offset < 0 or segment.width <= 0 or segment.max_context_tokens <= 0:
            raise DecodeCacheError("decode cache segment has invalid dimensions")
        if segment.dtype != dtype or segment.dtype_bytes != dtype_bytes:
            raise DecodeCacheError("decode cache segment dtype does not match layout")
        if segment.offset + segment.total_bytes > total_bytes:
            raise DecodeCacheError("decode cache segment exceeds layout total_bytes")
        segments.append(segment)

    return DecodeCacheLayout(
        version=version,
        model_type=model_type,
        max_context_tokens=max_context_tokens,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        alignment=alignment,
        total_bytes=total_bytes,
        segments=tuple(segments),
    )


def init_decode_cache_file(
    layout_path: str | Path,
    cache_file_path: str | Path,
    *,
    force: bool = False,
    max_cache_bytes: int | None = None,
    disk_safety_margin_bytes: int = 16 * 1024**3,
) -> DecodeCacheInitResult:
    layout = load_decode_cache_layout(layout_path)
    max_cache_bytes = _optional_nonnegative_integer_value(
        "max_cache_bytes",
        max_cache_bytes,
    )
    disk_safety_margin_bytes = _nonnegative_integer_value(
        "disk_safety_margin_bytes",
        disk_safety_margin_bytes,
    )
    if max_cache_bytes is not None and layout.total_bytes > max_cache_bytes:
        raise DecodeCacheError(
            f"decode cache {layout.total_bytes} bytes exceeds limit {max_cache_bytes}"
        )
    out = Path(cache_file_path)
    existed = out.exists()
    if existed and not force:
        raise DecodeCacheError(f"{out} already exists; use --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    budget = disk_budget(
        out,
        layout.total_bytes,
        safety_margin_bytes=disk_safety_margin_bytes,
    )
    if not budget.ok:
        raise DecodeCacheError(
            "not enough free disk for decode cache: "
            f"need {layout.total_bytes + disk_safety_margin_bytes} bytes including "
            f"margin, have {budget.available_bytes} bytes"
        )

    flags = os.O_RDWR | os.O_CREAT
    flags |= os.O_TRUNC if force else os.O_EXCL
    try:
        fd = os.open(out, flags, 0o644)
    except OSError as exc:
        raise DecodeCacheError(f"failed to open decode cache file {out}: {exc}") from exc
    try:
        try:
            os.ftruncate(fd, layout.total_bytes)
        except OSError as exc:
            _remove_partial_file(out)
            raise DecodeCacheError(
                f"failed to initialize decode cache file {out}: {exc}"
            ) from exc
    finally:
        os.close(fd)
    return DecodeCacheInitResult(
        layout_path=Path(layout_path),
        cache_file_path=out,
        total_bytes=layout.total_bytes,
        existed=existed,
        sparse=True,
        disk_budget=budget,
    )


def build_decode_cache_layout(
    config: ModelConfig,
    *,
    max_context_tokens: int,
    dtype: str = "BF16",
    alignment: int = 64,
    max_cache_bytes: int | None = None,
) -> DecodeCacheLayout:
    max_context_tokens = _positive_integer_value(
        "max_context_tokens",
        max_context_tokens,
    )
    alignment = _positive_integer_value("alignment", alignment)
    max_cache_bytes = _optional_nonnegative_integer_value(
        "max_cache_bytes",
        max_cache_bytes,
    )
    if (
        config.max_position_embeddings is not None
        and max_context_tokens > int(config.max_position_embeddings)
    ):
        raise DecodeCacheError(
            "max_context_tokens "
            f"{max_context_tokens} exceeds model max_position_embeddings "
            f"{config.max_position_embeddings}"
        )
    dtype_bytes = _dtype_bytes(dtype)
    cache_width = config.mla_cache_width
    if cache_width is None:
        raise ConfigError("config is missing kv_lora_rank/qk_rope_head_dim")

    offset = 0
    segments: list[DecodeCacheSegment] = []

    def add_segment(kind: str, layer: int, width: int) -> None:
        nonlocal offset
        if width <= 0:
            raise DecodeCacheError(f"{kind} width must be positive")
        offset = _align_up(offset, alignment)
        segment = DecodeCacheSegment(
            kind=kind,
            layer=layer,
            offset=offset,
            width=width,
            dtype=dtype,
            dtype_bytes=dtype_bytes,
            max_context_tokens=max_context_tokens,
        )
        segments.append(segment)
        offset += segment.total_bytes

    for layer in range(config.num_hidden_layers):
        add_segment("mla_kv", layer, cache_width)

    if config.indexer_types is not None and config.index_head_dim:
        for layer, indexer_type in enumerate(config.indexer_types):
            if layer >= config.num_hidden_layers:
                break
            if indexer_type == "full":
                add_segment("dsa_index", layer, int(config.index_head_dim))

    total_bytes = _align_up(offset, alignment)
    if max_cache_bytes is not None and total_bytes > max_cache_bytes:
        raise DecodeCacheError(
            f"decode cache {total_bytes} bytes exceeds limit {max_cache_bytes}"
        )

    return DecodeCacheLayout(
        version=1,
        model_type=config.model_type,
        max_context_tokens=max_context_tokens,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        alignment=alignment,
        total_bytes=total_bytes,
        segments=tuple(segments),
    )
