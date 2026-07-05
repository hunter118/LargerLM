from __future__ import annotations

import operator
import os
import time
from dataclasses import dataclass
from pathlib import Path


MAX_SEQUENTIAL_READ_CHUNK_BYTES = 512 * 1024**2


class DiskBenchmarkError(RuntimeError):
    """Raised when a bounded disk benchmark cannot run safely."""


@dataclass(frozen=True)
class SequentialReadBenchmark:
    path: Path
    file_size_bytes: int
    offset_bytes: int
    requested_bytes: int
    measured_bytes: int
    chunk_bytes: int
    elapsed_seconds: float
    gib_per_second: float
    short_read: bool


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise DiskBenchmarkError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise DiskBenchmarkError(f"{name} must be an integer") from exc


def _positive_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed <= 0:
        raise DiskBenchmarkError(f"{name} must be positive")
    return parsed


def _nonnegative_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed < 0:
        raise DiskBenchmarkError(f"{name} must be non-negative")
    return parsed


def _optional_positive_integer_value(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _positive_integer_value(name, value)


def benchmark_sequential_read(
    path: str | Path,
    *,
    bytes_to_read: int | None = None,
    chunk_bytes: int = 8 * 1024**2,
    offset_bytes: int = 0,
    max_chunk_bytes: int | None = MAX_SEQUENTIAL_READ_CHUNK_BYTES,
) -> SequentialReadBenchmark:
    target = Path(path)
    if bytes_to_read is not None:
        bytes_to_read = _positive_integer_value("bytes_to_read", bytes_to_read)
    chunk_bytes = _positive_integer_value("chunk_bytes", chunk_bytes)
    offset_bytes = _nonnegative_integer_value("offset_bytes", offset_bytes)
    max_chunk_bytes = _optional_positive_integer_value(
        "max_chunk_bytes",
        max_chunk_bytes,
    )
    if max_chunk_bytes is not None and chunk_bytes > max_chunk_bytes:
        raise DiskBenchmarkError(
            f"chunk_bytes {chunk_bytes} exceeds max_chunk_bytes {max_chunk_bytes}"
        )
    try:
        stat = target.stat()
    except OSError as exc:
        raise DiskBenchmarkError(f"failed to stat {target}: {exc}") from exc
    if not target.is_file():
        raise DiskBenchmarkError(f"path is not a regular file: {target}")
    file_size = int(stat.st_size)
    if file_size <= 0:
        raise DiskBenchmarkError(f"file is empty: {target}")
    if offset_bytes >= file_size:
        raise DiskBenchmarkError("offset_bytes must be smaller than file size")
    available = file_size - offset_bytes
    requested = bytes_to_read if bytes_to_read is not None else available
    measured_target = min(requested, available)
    measured = 0
    started = time.perf_counter()
    try:
        fd = os.open(target, os.O_RDONLY)
    except OSError as exc:
        raise DiskBenchmarkError(f"failed to open {target}: {exc}") from exc
    try:
        while measured < measured_target:
            want = min(chunk_bytes, measured_target - measured)
            data = os.pread(fd, want, offset_bytes + measured)
            if not data:
                break
            measured += len(data)
            if len(data) < want:
                break
    except OSError as exc:
        raise DiskBenchmarkError(f"failed to read {target}: {exc}") from exc
    finally:
        os.close(fd)
    elapsed = time.perf_counter() - started
    gib_s = measured / (1024**3) / elapsed if measured > 0 and elapsed > 0 else 0.0
    return SequentialReadBenchmark(
        path=target,
        file_size_bytes=file_size,
        offset_bytes=offset_bytes,
        requested_bytes=requested,
        measured_bytes=measured,
        chunk_bytes=chunk_bytes,
        elapsed_seconds=elapsed,
        gib_per_second=gib_s,
        short_read=measured < requested,
    )
