from __future__ import annotations

import math
import operator
import os
import platform
import re
import subprocess
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import first_shared_indexer_without_previous_full
from .decode_cache import load_decode_cache_layout
from .decode_driver import layers_from_expert_layout
from .final_logits import (
    _affine_row_bytes,
    _dtype_bytes,
    _find_global_tensor,
    _head_shape,
    _load_json,
    _mxfp4_info,
    _mxfp4_row_bytes,
    _shape1,
    _shape2,
)
from .prepared import PreparedManifestError, validate_layout_backing_files
from .runtime_check import LayerRuntimeBudget, check_layer_runtime


class GenerationGuardError(RuntimeError):
    """Raised when a generation request would exceed configured runtime limits."""

    def __init__(
        self,
        message: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.payload = payload or {}


@dataclass(frozen=True)
class FinalLogitsBudget:
    head_tensor: str
    norm_tensor: str | None
    hidden_dim: int
    vocab_size: int
    dtype: str
    top_k: int
    chunk_rows: int
    chunks: int
    row_bytes: int
    chunk_bytes: int
    read_bytes: int
    estimated_peak_bytes: int
    max_chunk_bytes: int
    max_runner_scratch_bytes: int | None
    metal: bool


@dataclass(frozen=True)
class EmbeddingBudget:
    tensor: str
    hidden_dim: int
    vocab_size: int
    dtype: str
    row_bytes: int
    output_bytes: int
    max_row_bytes: int


@dataclass(frozen=True)
class SystemMemorySnapshot:
    total_bytes: int | None
    available_bytes: int | None
    page_size: int | None
    source: str


@dataclass(frozen=True)
class LiveMemoryBudget:
    estimated_live_working_set_bytes: int
    max_live_working_set_bytes: int | None
    min_available_memory_bytes: int
    system_available_bytes: int | None
    system_total_bytes: int | None
    system_source: str | None
    resident_backing_bytes: int = 0
    nonresident_peak_bytes: int | None = None
    extra_live_working_set_bytes: int = 0


@dataclass(frozen=True)
class PromptPrefillLiveMemoryEstimate:
    prompt_batch_bytes: int
    runner_scratch_bytes: int
    cache_read_bytes: int
    cache_write_bytes: int
    stage_copy_bytes: int
    estimated_live_working_set_bytes: int


@dataclass(frozen=True)
class GenerationRuntimeGuard:
    expert_layout_path: Path
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    requested_context_tokens: int
    cache_context_tokens: int
    cache_file_bytes: int
    layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    layer_budgets: tuple[LayerRuntimeBudget, ...]
    max_layer_peak_bytes: int
    max_layer_cache_read_bytes: int
    read_bytes_per_token: int
    dsa_index_layers: tuple[int, ...]
    dsa_index_cache_bytes: int
    dsa_indexer_runtime: bool
    allow_missing_dsa_indexer: bool
    embedding_budget: EmbeddingBudget
    final_logits_budget: FinalLogitsBudget
    live_memory_budget: LiveMemoryBudget


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _run_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except Exception:
        return None
    return result.stdout


def _parse_vm_stat(text: str, *, total_bytes: int | None = None) -> SystemMemorySnapshot:
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", text)
    page_size = int(page_match.group(1)) if page_match else 4096
    pages: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([^:]+):\s+([\d,]+)\.", line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        pages[key] = int(match.group(2).replace(",", ""))
    available_pages = (
        pages.get("pages free", 0)
        + pages.get("pages inactive", 0)
        + pages.get("pages speculative", 0)
    )
    available_bytes = available_pages * page_size if available_pages else None
    return SystemMemorySnapshot(
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        page_size=page_size,
        source="vm_stat",
    )


def _parse_meminfo(text: str) -> SystemMemorySnapshot:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            kib = int(parts[0])
        except ValueError:
            continue
        values[key] = kib * 1024
    return SystemMemorySnapshot(
        total_bytes=values.get("MemTotal"),
        available_bytes=values.get("MemAvailable", values.get("MemFree")),
        page_size=None,
        source="/proc/meminfo",
    )


def _darwin_hw_memsize_sysctlbyname() -> int | None:
    try:
        libc = ctypes.CDLL(None)
        value = ctypes.c_uint64()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        sysctlbyname = libc.sysctlbyname
        rc = sysctlbyname(
            b"hw.memsize",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        )
    except Exception:
        return None
    if rc != 0 or value.value <= 0:
        return None
    return int(value.value)


def _darwin_hw_memsize_sysconf() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _darwin_hw_memsize_bytes() -> int | None:
    total_bytes = _darwin_hw_memsize_sysctlbyname()
    if total_bytes is not None:
        return total_bytes
    total_bytes = _darwin_hw_memsize_sysconf()
    if total_bytes is not None:
        return total_bytes
    total_text = _run_text(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    if total_text:
        try:
            total_bytes = int(total_text.strip())
        except ValueError:
            total_bytes = None
    return total_bytes


def system_memory_snapshot() -> SystemMemorySnapshot | None:
    if platform.system() == "Darwin":
        total_bytes = _darwin_hw_memsize_bytes()
        vm_text = _run_text(["/usr/bin/vm_stat"])
        if vm_text:
            return _parse_vm_stat(vm_text, total_bytes=total_bytes)
        if total_bytes is not None:
            return SystemMemorySnapshot(
                total_bytes=total_bytes,
                available_bytes=None,
                page_size=None,
                source="sysctl",
            )
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            return _parse_meminfo(meminfo.read_text(encoding="utf-8"))
        except OSError:
            return None
    return None


def check_live_memory_budget(
    *,
    estimated_live_working_set_bytes: int,
    max_live_working_set_bytes: int | None,
    min_available_memory_bytes: int = 0,
    snapshot: SystemMemorySnapshot | None = None,
    resident_backing_bytes: int = 0,
    nonresident_peak_bytes: int | None = None,
    extra_live_working_set_bytes: int = 0,
) -> LiveMemoryBudget:
    estimated_live_working_set_bytes = _nonnegative_integer_value(
        "estimated live working set",
        estimated_live_working_set_bytes,
    )
    resident_backing_bytes = _nonnegative_integer_value(
        "resident_backing_bytes",
        resident_backing_bytes,
    )
    extra_live_working_set_bytes = _nonnegative_integer_value(
        "extra_live_working_set_bytes",
        extra_live_working_set_bytes,
    )
    if nonresident_peak_bytes is not None:
        nonresident_peak_bytes = _nonnegative_integer_value(
            "nonresident_peak_bytes",
            nonresident_peak_bytes,
        )
    if max_live_working_set_bytes is not None:
        max_live_working_set_bytes = _integer_value(
            "max live working set",
            max_live_working_set_bytes,
        )
        if max_live_working_set_bytes <= 0:
            max_live_working_set_bytes = None
    min_available_memory_bytes = _nonnegative_integer_value(
        "min free unified memory",
        min_available_memory_bytes,
    )
    if (
        max_live_working_set_bytes is not None
        and estimated_live_working_set_bytes > max_live_working_set_bytes
    ):
        raise GenerationGuardError(
            f"estimated live working set {estimated_live_working_set_bytes} bytes "
            f"exceeds configured limit {max_live_working_set_bytes}",
            payload={
                "code": "estimated_live_working_set_exceeds_limit",
                "estimated_live_working_set_bytes": estimated_live_working_set_bytes,
                "max_live_working_set_bytes": max_live_working_set_bytes,
                "min_available_memory_bytes": min_available_memory_bytes,
                "resident_backing_bytes": resident_backing_bytes,
                "nonresident_peak_bytes": nonresident_peak_bytes,
                "extra_live_working_set_bytes": extra_live_working_set_bytes,
            },
        )

    detected = snapshot
    if min_available_memory_bytes > 0 and detected is None:
        detected = system_memory_snapshot()
    available = detected.available_bytes if detected is not None else None
    total = detected.total_bytes if detected is not None else None
    source = detected.source if detected is not None else None
    if min_available_memory_bytes > 0:
        if available is None:
            raise GenerationGuardError(
                "could not inspect system available memory for "
                "min free unified memory guard",
                payload={
                    "code": "system_available_memory_unavailable",
                    "estimated_live_working_set_bytes": (
                        estimated_live_working_set_bytes
                    ),
                    "min_available_memory_bytes": min_available_memory_bytes,
                    "system_available_memory_bytes": available,
                    "system_total_memory_bytes": total,
                    "system_memory_source": source,
                    "resident_backing_bytes": resident_backing_bytes,
                    "nonresident_peak_bytes": nonresident_peak_bytes,
                    "extra_live_working_set_bytes": extra_live_working_set_bytes,
                },
            )
        required_available = (
            estimated_live_working_set_bytes + min_available_memory_bytes
        )
        if available < required_available:
            raise GenerationGuardError(
                f"available unified memory {available} bytes is below required "
                f"{required_available} bytes including live working set and reserve",
                payload={
                    "code": "available_unified_memory_below_required",
                    "estimated_live_working_set_bytes": (
                        estimated_live_working_set_bytes
                    ),
                    "max_live_working_set_bytes": max_live_working_set_bytes,
                    "min_available_memory_bytes": min_available_memory_bytes,
                    "required_available_memory_bytes": required_available,
                    "system_available_memory_bytes": available,
                    "system_total_memory_bytes": total,
                    "system_memory_source": source,
                    "available_memory_ok": False,
                    "resident_backing_bytes": resident_backing_bytes,
                    "nonresident_peak_bytes": nonresident_peak_bytes,
                    "extra_live_working_set_bytes": extra_live_working_set_bytes,
                },
            )
    return LiveMemoryBudget(
        estimated_live_working_set_bytes=estimated_live_working_set_bytes,
        max_live_working_set_bytes=max_live_working_set_bytes,
        min_available_memory_bytes=min_available_memory_bytes,
        system_available_bytes=available,
        system_total_bytes=total,
        system_source=source,
        resident_backing_bytes=resident_backing_bytes,
        nonresident_peak_bytes=nonresident_peak_bytes,
        extra_live_working_set_bytes=extra_live_working_set_bytes,
    )


def _numeric_limit(name: str, value: float | int) -> float:
    if isinstance(value, bool):
        raise GenerationGuardError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GenerationGuardError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise GenerationGuardError(f"{name} must be finite")
    return parsed


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise GenerationGuardError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise GenerationGuardError(f"{name} must be an integer") from exc


def _positive_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed <= 0:
        raise GenerationGuardError(f"{name} must be positive")
    return parsed


def _nonnegative_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed < 0:
        raise GenerationGuardError(f"{name} must be non-negative")
    return parsed


def _layer_id_set(values: Iterable[object], *, name: str) -> set[int]:
    try:
        return {_nonnegative_integer_value(name, layer) for layer in values}
    except TypeError as exc:
        raise GenerationGuardError(f"{name} must be iterable") from exc


def _positive_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed <= 0:
        raise GenerationGuardError(f"{name} must be positive")
    return parsed


def _nonnegative_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed < 0:
        raise GenerationGuardError(f"{name} must be non-negative")
    return parsed


def estimate_prompt_prefill_live_memory(
    *,
    max_prompt_batch_mib: float,
    max_runner_scratch_mib: float,
    max_cache_read_mib: float,
    max_cache_write_mib: float,
    copy_chunk_mib: float,
) -> PromptPrefillLiveMemoryEstimate:
    values = {
        "max_prompt_batch_mib": max_prompt_batch_mib,
        "max_runner_scratch_mib": max_runner_scratch_mib,
        "max_cache_read_mib": max_cache_read_mib,
        "max_cache_write_mib": max_cache_write_mib,
        "copy_chunk_mib": copy_chunk_mib,
    }
    for label, value in values.items():
        values[label] = _positive_limit(label, value)
    max_prompt_batch_mib = values["max_prompt_batch_mib"]
    max_runner_scratch_mib = values["max_runner_scratch_mib"]
    max_cache_read_mib = values["max_cache_read_mib"]
    max_cache_write_mib = values["max_cache_write_mib"]
    copy_chunk_mib = values["copy_chunk_mib"]

    prompt_batch_bytes = int(max_prompt_batch_mib * 1024**2)
    runner_scratch_bytes = int(max_runner_scratch_mib * 1024**2)
    cache_read_bytes = int(max_cache_read_mib * 1024**2)
    cache_write_bytes = int(max_cache_write_mib * 1024**2)
    stage_copy_bytes = int(copy_chunk_mib * 1024**2)

    # These caps are not all live at once: prompt batch buffers, attention cache
    # I/O, and expert stage copies happen in separate substeps. Keep the estimate
    # conservative by pairing each phase with the runner scratch ceiling.
    estimated = max(
        prompt_batch_bytes + runner_scratch_bytes,
        cache_read_bytes + runner_scratch_bytes,
        cache_write_bytes + runner_scratch_bytes,
        stage_copy_bytes + runner_scratch_bytes,
    )
    return PromptPrefillLiveMemoryEstimate(
        prompt_batch_bytes=prompt_batch_bytes,
        runner_scratch_bytes=runner_scratch_bytes,
        cache_read_bytes=cache_read_bytes,
        cache_write_bytes=cache_write_bytes,
        stage_copy_bytes=stage_copy_bytes,
        estimated_live_working_set_bytes=estimated,
    )


def _embedding_budget(
    resident_layout_path: str | Path,
    *,
    max_row_bytes: int,
    expected_vocab_size: int | None,
    expected_hidden_size: int | None,
) -> EmbeddingBudget:
    if max_row_bytes <= 0:
        raise GenerationGuardError("embedding max row bytes must be positive")
    layout = _load_json(resident_layout_path)
    embedding = _find_global_tensor(
        layout,
        (
            "model.embed_tokens.weight",
            ".embed_tokens.weight",
            "transformer.word_embeddings.weight",
            ".word_embeddings.weight",
        ),
        label="embed_tokens.weight",
        required=False,
    )
    if embedding is None:
        raise GenerationGuardError("embed_tokens.weight not found")
    mxfp4 = _mxfp4_info(layout, embedding)
    if mxfp4 is not None:
        vocab_size = mxfp4.out_dim
        hidden_dim = mxfp4.in_dim
    else:
        vocab_size, hidden_dim = _shape2(embedding, "embed_tokens.weight")
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise GenerationGuardError(
            f"embed_tokens.weight vocab size {vocab_size} does not match "
            f"config vocab_size {expected_vocab_size}"
        )
    if expected_hidden_size is not None and hidden_dim != expected_hidden_size:
        raise GenerationGuardError(
            f"embed_tokens.weight hidden dim {hidden_dim} does not match "
            f"config hidden_size {expected_hidden_size}"
        )
    dtype = "mlx-mxfp4" if mxfp4 is not None else str(embedding.get("dtype") or "")
    if mxfp4 is not None:
        row_bytes = _mxfp4_row_bytes(mxfp4)[2]
    else:
        dtype_nbytes = _dtype_bytes(dtype)
        if dtype_nbytes <= 0:
            raise GenerationGuardError(f"unsupported embed_tokens dtype {dtype}")
        row_bytes = hidden_dim * dtype_nbytes
    if row_bytes > max_row_bytes:
        raise GenerationGuardError(
            f"embedding row {row_bytes} bytes exceeds limit {max_row_bytes}"
        )
    return EmbeddingBudget(
        tensor=str(embedding.get("name")),
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dtype=dtype,
        row_bytes=row_bytes,
        output_bytes=hidden_dim * 4,
        max_row_bytes=max_row_bytes,
    )


def _final_logits_budget(
    resident_layout_path: str | Path,
    *,
    top_k: int,
    chunk_rows: int | None,
    max_chunk_bytes: int,
    max_runner_scratch_bytes: int | None,
    rms_norm_eps: float,
    allow_tied_embeddings: bool,
    expected_vocab_size: int | None,
    expected_hidden_size: int | None,
    skip_final_norm: bool,
    metal: bool,
) -> FinalLogitsBudget:
    del rms_norm_eps
    if top_k <= 0:
        raise GenerationGuardError("logits_top_k must be positive")
    if metal and top_k > 64:
        raise GenerationGuardError("Metal final logits supports logits_top_k <= 64")
    if max_chunk_bytes <= 0:
        raise GenerationGuardError("logits max chunk bytes must be positive")
    if max_runner_scratch_bytes is not None and max_runner_scratch_bytes <= 0:
        raise GenerationGuardError("max runner scratch bytes must be positive")

    layout = _load_json(resident_layout_path)
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
        raise GenerationGuardError("lm_head.weight not found")
    vocab_size, hidden_dim, affine, mxfp4 = _head_shape(
        layout,
        head,
        "lm_head/embedding",
    )
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise GenerationGuardError(
            f"lm_head/embedding vocab size {vocab_size} does not match "
            f"config vocab_size {expected_vocab_size}"
        )
    if expected_hidden_size is not None and hidden_dim != expected_hidden_size:
        raise GenerationGuardError(
            f"lm_head/embedding hidden dim {hidden_dim} does not match "
            f"config hidden_size {expected_hidden_size}"
        )
    if top_k > vocab_size:
        raise GenerationGuardError(f"logits_top_k {top_k} exceeds vocab size {vocab_size}")
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
        if dtype_nbytes <= 0:
            raise GenerationGuardError(f"unsupported lm_head dtype {dtype}")
        row_bytes = hidden_dim * dtype_nbytes
    if row_bytes > max_chunk_bytes:
        raise GenerationGuardError(
            f"one lm_head row {row_bytes} bytes exceeds chunk limit {max_chunk_bytes}"
        )
    rows = chunk_rows or max(1, max_chunk_bytes // row_bytes)
    rows = min(rows, vocab_size)
    if rows <= 0:
        raise GenerationGuardError("logits chunk rows must be positive")
    chunk_bytes = rows * row_bytes
    if chunk_bytes > max_chunk_bytes:
        raise GenerationGuardError(
            f"logits chunk {chunk_bytes} bytes exceeds limit {max_chunk_bytes}"
        )
    chunks = (vocab_size + rows - 1) // rows
    norm_name = None
    hidden_bytes = hidden_dim * 4
    norm_peak = 0
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
            raise GenerationGuardError(
                f"final norm dim {norm_dim} does not match hidden dim {hidden_dim}"
            )
        norm_name = str(norm.get("name"))
        norm_peak = 3 * hidden_bytes

    output_bytes = rows * 4
    if metal:
        chunk_peak = _align_up(chunk_bytes, 2 * 1024 * 1024) + hidden_bytes + output_bytes
    else:
        chunk_peak = chunk_bytes + 2 * hidden_bytes + output_bytes
    peak = max(norm_peak, chunk_peak)
    if max_runner_scratch_bytes is not None and peak > max_runner_scratch_bytes:
        raise GenerationGuardError(
            f"final logits peak {peak} bytes exceeds runner scratch limit "
            f"{max_runner_scratch_bytes}"
        )
    return FinalLogitsBudget(
        head_tensor=str(head.get("name")),
        norm_tensor=norm_name,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dtype=dtype,
        top_k=top_k,
        chunk_rows=rows,
        chunks=chunks,
        row_bytes=row_bytes,
        chunk_bytes=chunk_bytes,
        read_bytes=vocab_size * row_bytes,
        estimated_peak_bytes=peak,
        max_chunk_bytes=max_chunk_bytes,
        max_runner_scratch_bytes=max_runner_scratch_bytes,
        metal=metal,
    )


def check_generation_runtime(
    *,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    requested_context_tokens: int,
    layers: Iterable[int] | None,
    dense_layers: Iterable[int] | None = None,
    top_k: int,
    max_k: int,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    include_shared_expert: bool,
    logits_top_k: int,
    logits_chunk_rows: int | None,
    logits_max_chunk_mib: float,
    rms_norm_eps: float,
    max_slot_mib: float,
    max_router_mib: float,
    max_resident_matrix_mib: float,
    max_cache_file_mib: float,
    max_cache_read_mib: float,
    max_runner_scratch_mib: float,
    cache_dtype_bytes: int,
    metal_final_logits: bool,
    allow_tied_embeddings: bool = True,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
    max_embedding_row_mib: float = 64.0,
    max_live_working_set_mib: float | None = 8192.0,
    min_free_unified_memory_mib: float = 0.0,
    extra_live_working_set_bytes: int = 0,
    system_memory: SystemMemorySnapshot | None = None,
    decode_mla_key_cache: bool = False,
    dsa_indexer_runtime: bool = False,
    dsa_indexer_types: Iterable[str] | None = None,
    dsa_index_topk: int | None = None,
    dsa_index_head_dim: int | None = None,
    allow_missing_dsa_indexer: bool = False,
) -> GenerationRuntimeGuard:
    requested_context_tokens = _positive_integer_value(
        "requested_context_tokens",
        requested_context_tokens,
    )
    top_k = _positive_integer_value("top_k", top_k)
    max_k = _positive_integer_value("max_k", max_k)
    num_heads = _positive_integer_value("num_heads", num_heads)
    qk_nope_dim = _positive_integer_value("qk_nope_dim", qk_nope_dim)
    rope_dim = _positive_integer_value("rope_dim", rope_dim)
    v_head_dim = _positive_integer_value("v_head_dim", v_head_dim)
    logits_top_k = _positive_integer_value("logits_top_k", logits_top_k)
    if logits_chunk_rows is not None:
        logits_chunk_rows = _positive_integer_value(
            "logits_chunk_rows",
            logits_chunk_rows,
        )
    cache_dtype_bytes = _positive_integer_value(
        "cache_dtype_bytes",
        cache_dtype_bytes,
    )
    extra_live_working_set_bytes = _nonnegative_integer_value(
        "extra_live_working_set_bytes",
        extra_live_working_set_bytes,
    )
    if type(decode_mla_key_cache) is not bool:
        raise GenerationGuardError("decode_mla_key_cache must be a boolean")
    if dsa_index_topk is not None:
        dsa_index_topk = _positive_integer_value("dsa_index_topk", dsa_index_topk)
    if dsa_index_head_dim is not None:
        dsa_index_head_dim = _positive_integer_value(
            "dsa_index_head_dim",
            dsa_index_head_dim,
        )
    if expected_vocab_size is not None:
        expected_vocab_size = _positive_integer_value(
            "expected_vocab_size",
            expected_vocab_size,
        )
    if expected_hidden_size is not None:
        expected_hidden_size = _positive_integer_value(
            "expected_hidden_size",
            expected_hidden_size,
        )
    logits_max_chunk_mib = _positive_limit(
        "logits_max_chunk_mib",
        logits_max_chunk_mib,
    )
    max_slot_mib = _positive_limit("max_slot_mib", max_slot_mib)
    max_router_mib = _positive_limit("max_router_mib", max_router_mib)
    max_resident_matrix_mib = _positive_limit(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_cache_file_mib = _positive_limit("max_cache_file_mib", max_cache_file_mib)
    max_cache_read_mib = _positive_limit("max_cache_read_mib", max_cache_read_mib)
    max_runner_scratch_mib = _positive_limit(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    max_embedding_row_mib = _positive_limit(
        "max_embedding_row_mib",
        max_embedding_row_mib,
    )
    min_free_unified_memory_mib = _nonnegative_limit(
        "min_free_unified_memory_mib",
        min_free_unified_memory_mib,
    )
    if max_live_working_set_mib is not None:
        max_live_working_set_mib = _nonnegative_limit(
            "max_live_working_set_mib",
            max_live_working_set_mib,
        )
    cache_layout = load_decode_cache_layout(cache_layout_path)
    if requested_context_tokens > cache_layout.max_context_tokens:
        raise GenerationGuardError(
            f"requested context {requested_context_tokens} exceeds cache context "
            f"{cache_layout.max_context_tokens}"
        )
    max_cache_file_bytes = int(max_cache_file_mib * 1024**2)
    if cache_layout.total_bytes > max_cache_file_bytes:
        raise GenerationGuardError(
            f"decode cache layout {cache_layout.total_bytes} bytes exceeds limit "
            f"{max_cache_file_bytes}"
        )
    cache_file = Path(cache_file_path)
    try:
        cache_file_bytes = os.path.getsize(cache_file)
    except OSError as exc:
        raise GenerationGuardError(f"failed to stat cache file {cache_file}: {exc}") from exc
    if cache_file_bytes < cache_layout.total_bytes:
        raise GenerationGuardError(
            f"cache file {cache_file} is {cache_file_bytes} bytes, expected at least "
            f"{cache_layout.total_bytes}"
        )
    try:
        backing = validate_layout_backing_files(
            expert_layout_path,
            resident_layout_path,
        )
    except PreparedManifestError as exc:
        raise GenerationGuardError(str(exc)) from exc
    dsa_segments = tuple(
        segment for segment in cache_layout.segments if segment.kind == "dsa_index"
    )
    if dsa_segments and not (dsa_indexer_runtime or allow_missing_dsa_indexer):
        layers = ",".join(str(segment.layer) for segment in dsa_segments[:8])
        more = "" if len(dsa_segments) <= 8 else f", +{len(dsa_segments) - 8} more"
        raise GenerationGuardError(
            "decode cache contains DSA/indexer segments but the runtime does not "
            f"compute indexer cache for this request; layers={layers}{more}. "
            "Enable DSA indexer runtime, or use --allow-missing-dsa-indexer only "
            "for explicit debugging."
        )
    dsa_index_layers = tuple(segment.layer for segment in dsa_segments)
    dsa_segment_by_layer = {segment.layer: segment for segment in dsa_segments}
    dsa_index_cache_bytes = sum(segment.total_bytes for segment in dsa_segments)

    expert_layer_ids = set(layers_from_expert_layout(expert_layout_path))
    dense_layer_ids = _layer_id_set(dense_layers or (), name="dense_layers")
    overlap = expert_layer_ids & dense_layer_ids
    if overlap:
        joined = ",".join(str(layer) for layer in sorted(overlap))
        raise GenerationGuardError(f"dense layers overlap expert layout layers: {joined}")
    if layers is None:
        requested_layer_ids = set(expert_layer_ids) | set(dense_layer_ids)
    else:
        requested_layer_ids = _layer_id_set(layers, name="layers")
    unknown_layer_ids = requested_layer_ids - expert_layer_ids - dense_layer_ids
    if unknown_layer_ids:
        joined = ",".join(str(layer) for layer in sorted(unknown_layer_ids))
        raise GenerationGuardError(f"layers not found in dense or expert layouts: {joined}")
    layer_ids = tuple(sorted(requested_layer_ids))
    if not layer_ids:
        raise GenerationGuardError("no layers selected")
    dsa_types = tuple(str(item).lower() for item in dsa_indexer_types or ())
    invalid_dsa_types = sorted(set(dsa_types) - {"none", "full", "shared"})
    if invalid_dsa_types:
        joined = ",".join(invalid_dsa_types)
        raise GenerationGuardError(f"invalid DSA indexer_type values: {joined}")
    bad_shared_layer = first_shared_indexer_without_previous_full(
        dsa_types,
        selected_layers=layer_ids,
    )
    if bad_shared_layer is not None:
        raise GenerationGuardError(
            f"selected DSA layer {bad_shared_layer} is shared but no previous "
            "selected full-indexer layer is available"
        )
    if dsa_indexer_runtime:
        if not dsa_types:
            raise GenerationGuardError("DSA indexer runtime requires indexer_types")
        if len(dsa_types) <= max(layer_ids):
            raise GenerationGuardError("DSA indexer_types does not cover selected layers")
        if dsa_index_topk is None or dsa_index_topk <= 0:
            raise GenerationGuardError("DSA index_topk must be positive")
        full_layers = {layer for layer in layer_ids if dsa_types[layer] == "full"}
        missing_cache_layers = sorted(full_layers - set(dsa_segment_by_layer))
        if missing_cache_layers:
            joined = ",".join(str(layer) for layer in missing_cache_layers)
            raise GenerationGuardError(
                "DSA full-indexer layers are missing dsa_index cache segments: "
                f"{joined}"
            )
        selected_dsa_cache_layers = set(dsa_segment_by_layer) & set(layer_ids)
        if selected_dsa_cache_layers and not full_layers and not allow_missing_dsa_indexer:
            joined = ",".join(str(layer) for layer in sorted(selected_dsa_cache_layers))
            raise GenerationGuardError(
                "decode cache has selected dsa_index segments but schedule has no "
                f"full DSA indexer layers: {joined}"
            )
    resolved_dsa_index_head_dim = dsa_index_head_dim
    if resolved_dsa_index_head_dim is None and dsa_segments:
        widths = {segment.width for segment in dsa_segments}
        if len(widths) == 1:
            resolved_dsa_index_head_dim = next(iter(widths))
    if resolved_dsa_index_head_dim is not None and dsa_indexer_runtime and dsa_types:
        mismatched_width_layers = sorted(
            layer
            for layer in layer_ids
            if dsa_types[layer] == "full"
            and layer in dsa_segment_by_layer
            and dsa_segment_by_layer[layer].width != resolved_dsa_index_head_dim
        )
        if mismatched_width_layers:
            joined = ",".join(str(layer) for layer in mismatched_width_layers)
            raise GenerationGuardError(
                "dsa_index_head_dim does not match dsa_index cache width for layers: "
                f"{joined}"
            )
    max_slot_bytes = int(max_slot_mib * 1024**2)
    max_router_bytes = int(max_router_mib * 1024**2)
    max_resident_matrix_bytes = int(max_resident_matrix_mib * 1024**2)
    max_cache_read_bytes = int(max_cache_read_mib * 1024**2)
    max_runner_scratch_bytes = int(max_runner_scratch_mib * 1024**2)
    budgets = tuple(
        check_layer_runtime(
            expert_layout_path,
            resident_layout_path,
            layer=layer,
            dense_mlp=layer in dense_layer_ids,
            top_k=top_k,
            max_k=max_k,
            max_slot_bytes=max_slot_bytes,
            max_router_bytes=max_router_bytes,
            max_resident_matrix_bytes=max_resident_matrix_bytes,
            max_cache_read_bytes=max_cache_read_bytes,
            max_runner_scratch_bytes=max_runner_scratch_bytes,
            include_shared_expert=include_shared_expert,
            include_decoder_layer=True,
            context_length=requested_context_tokens,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            cache_dtype_bytes=cache_dtype_bytes,
            decode_mla_key_cache=decode_mla_key_cache,
            dsa_indexer_mode=(
                dsa_types[layer]
                if dsa_types and dsa_types[layer] in {"full", "shared"}
                else "none"
            ),
            dsa_index_topk=dsa_index_topk,
            dsa_index_head_dim=resolved_dsa_index_head_dim,
        )
        for layer in layer_ids
    )
    final_budget = _final_logits_budget(
        resident_layout_path,
        top_k=logits_top_k,
        chunk_rows=logits_chunk_rows,
        max_chunk_bytes=int(logits_max_chunk_mib * 1024**2),
        max_runner_scratch_bytes=max_runner_scratch_bytes if metal_final_logits else None,
        rms_norm_eps=rms_norm_eps,
        allow_tied_embeddings=allow_tied_embeddings,
        expected_vocab_size=expected_vocab_size,
        expected_hidden_size=expected_hidden_size,
        skip_final_norm=False,
        metal=metal_final_logits,
    )
    embedding_budget = _embedding_budget(
        resident_layout_path,
        max_row_bytes=int(max_embedding_row_mib * 1024**2),
        expected_vocab_size=expected_vocab_size,
        expected_hidden_size=expected_hidden_size,
    )
    nonresident_peak_bytes = max(
        max(b.estimated_peak_bytes for b in budgets),
        final_budget.estimated_peak_bytes,
        embedding_budget.row_bytes + embedding_budget.output_bytes,
        int(extra_live_working_set_bytes),
    )
    resident_backing_bytes = backing.resident_layout_bytes
    max_live_working_set_bytes = (
        int(max_live_working_set_mib * 1024**2)
        if max_live_working_set_mib is not None
        else None
    )
    live_budget = check_live_memory_budget(
        estimated_live_working_set_bytes=resident_backing_bytes
        + nonresident_peak_bytes,
        max_live_working_set_bytes=max_live_working_set_bytes,
        min_available_memory_bytes=int(min_free_unified_memory_mib * 1024**2),
        snapshot=system_memory,
        resident_backing_bytes=resident_backing_bytes,
        nonresident_peak_bytes=nonresident_peak_bytes,
        extra_live_working_set_bytes=extra_live_working_set_bytes,
    )
    return GenerationRuntimeGuard(
        expert_layout_path=Path(expert_layout_path),
        resident_layout_path=Path(resident_layout_path),
        cache_layout_path=Path(cache_layout_path),
        cache_file_path=cache_file,
        requested_context_tokens=requested_context_tokens,
        cache_context_tokens=cache_layout.max_context_tokens,
        cache_file_bytes=cache_file_bytes,
        layers=layer_ids,
        dense_layers=tuple(layer for layer in layer_ids if layer in dense_layer_ids),
        layer_budgets=budgets,
        max_layer_peak_bytes=max(b.estimated_peak_bytes for b in budgets),
        max_layer_cache_read_bytes=max(b.decoder_cache_read_bytes for b in budgets),
        read_bytes_per_token=sum(b.read_bytes_per_token for b in budgets),
        dsa_index_layers=dsa_index_layers,
        dsa_index_cache_bytes=dsa_index_cache_bytes,
        dsa_indexer_runtime=dsa_indexer_runtime,
        allow_missing_dsa_indexer=allow_missing_dsa_indexer,
        embedding_budget=embedding_budget,
        final_logits_budget=final_budget,
        live_memory_budget=live_budget,
    )
