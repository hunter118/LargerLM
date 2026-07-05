from __future__ import annotations

import operator
import shutil
from dataclasses import dataclass
from pathlib import Path


class SafetyError(RuntimeError):
    """Raised before an operation that could exhaust memory or disk."""


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise SafetyError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise SafetyError(f"{name} must be an integer") from exc


def _positive_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed <= 0:
        raise SafetyError(f"{name} must be positive")
    return parsed


def _nonnegative_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed < 0:
        raise SafetyError(f"{name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class DiskBudget:
    output_dir: Path
    required_bytes: int
    available_bytes: int
    safety_margin_bytes: int

    @property
    def ok(self) -> bool:
        return self.available_bytes >= self.required_bytes + self.safety_margin_bytes


def check_chunk_budget(chunk_size: int, max_chunk_size: int) -> None:
    chunk_size = _positive_integer_value("chunk size", chunk_size)
    max_chunk_size = _positive_integer_value("max chunk size", max_chunk_size)
    if chunk_size > max_chunk_size:
        raise SafetyError(
            f"chunk size {chunk_size} exceeds safety cap {max_chunk_size}"
        )


def estimate_pack_peak_heap_bytes(
    *,
    chunk_size: int,
    metadata_overhead_bytes: int = 256 * 1024**2,
) -> int:
    chunk_size = _positive_integer_value("chunk size", chunk_size)
    metadata_overhead_bytes = _nonnegative_integer_value(
        "metadata overhead bytes",
        metadata_overhead_bytes,
    )
    # The packer keeps one copied byte chunk live, plus Python objects for the
    # safetensors index/header metadata. The overhead is intentionally generous.
    return chunk_size + metadata_overhead_bytes


def check_memory_budget(estimated_peak_bytes: int, max_memory_bytes: int) -> None:
    estimated_peak_bytes = _nonnegative_integer_value(
        "estimated peak bytes",
        estimated_peak_bytes,
    )
    max_memory_bytes = _positive_integer_value("max memory bytes", max_memory_bytes)
    if estimated_peak_bytes > max_memory_bytes:
        raise SafetyError(
            f"estimated packer heap {estimated_peak_bytes} exceeds configured "
            f"limit {max_memory_bytes}; reduce --chunk-mib or raise the limit"
        )


def disk_budget(
    output_dir: str | Path,
    required_bytes: int,
    *,
    safety_margin_bytes: int = 16 * 1024**3,
) -> DiskBudget:
    required_bytes = _nonnegative_integer_value("required bytes", required_bytes)
    safety_margin_bytes = _nonnegative_integer_value(
        "safety margin bytes",
        safety_margin_bytes,
    )
    path = Path(output_dir)
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    budget = DiskBudget(
        output_dir=path,
        required_bytes=required_bytes,
        available_bytes=int(usage.free),
        safety_margin_bytes=safety_margin_bytes,
    )
    return budget


def check_disk_budget(
    output_dir: str | Path,
    required_bytes: int,
    *,
    safety_margin_bytes: int = 16 * 1024**3,
) -> DiskBudget:
    budget = disk_budget(
        output_dir,
        required_bytes,
        safety_margin_bytes=safety_margin_bytes,
    )
    if not budget.ok:
        raise SafetyError(
            "not enough free disk for packed experts: "
            f"need {required_bytes + safety_margin_bytes} bytes including margin, "
            f"have {budget.available_bytes} bytes"
        )
    return budget
