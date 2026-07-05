from __future__ import annotations

import json
from pathlib import Path

import pytest

from largerlm.cli import main as cli_main
from largerlm.disk_benchmark import (
    DiskBenchmarkError,
    benchmark_sequential_read,
)


def test_benchmark_sequential_read_measures_bounded_file_slice(tmp_path: Path) -> None:
    source = tmp_path / "layer.bin"
    source.write_bytes(bytes(range(256)) * 4096)

    result = benchmark_sequential_read(
        source,
        bytes_to_read=128 * 1024,
        chunk_bytes=16 * 1024,
        offset_bytes=64 * 1024,
    )

    assert result.path == source
    assert result.file_size_bytes == 1024 * 1024
    assert result.offset_bytes == 64 * 1024
    assert result.requested_bytes == 128 * 1024
    assert result.measured_bytes == 128 * 1024
    assert result.chunk_bytes == 16 * 1024
    assert result.elapsed_seconds > 0
    assert result.gib_per_second > 0
    assert result.short_read is False


def test_benchmark_sequential_read_reports_short_file(tmp_path: Path) -> None:
    source = tmp_path / "small.bin"
    source.write_bytes(b"x" * 1024)

    result = benchmark_sequential_read(
        source,
        bytes_to_read=4096,
        chunk_bytes=512,
    )

    assert result.requested_bytes == 4096
    assert result.measured_bytes == 1024
    assert result.short_read is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"bytes_to_read": True}, "bytes_to_read must be an integer"),
        ({"bytes_to_read": 1.5}, "bytes_to_read must be an integer"),
        ({"chunk_bytes": False}, "chunk_bytes must be an integer"),
        ({"chunk_bytes": 0}, "chunk_bytes must be positive"),
        (
            {"chunk_bytes": 513 * 1024**2},
            "chunk_bytes .* exceeds max_chunk_bytes",
        ),
        ({"max_chunk_bytes": False}, "max_chunk_bytes must be an integer"),
        ({"max_chunk_bytes": 0}, "max_chunk_bytes must be positive"),
        ({"offset_bytes": True}, "offset_bytes must be an integer"),
        ({"offset_bytes": -1}, "offset_bytes must be non-negative"),
    ),
)
def test_benchmark_sequential_read_rejects_invalid_controls(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "layer.bin"
    source.write_bytes(b"x" * 1024)

    with pytest.raises(DiskBenchmarkError, match=message):
        benchmark_sequential_read(source, **kwargs)


def test_benchmark_sequential_read_rejects_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.bin"
    source.write_bytes(b"")

    with pytest.raises(DiskBenchmarkError, match="file is empty"):
        benchmark_sequential_read(source)


def test_benchmark_sequential_read_result_jsonable(tmp_path: Path) -> None:
    source = tmp_path / "layer.bin"
    source.write_bytes(b"x" * 1024)

    result = benchmark_sequential_read(source)
    payload = json.dumps(
        {
            "path": str(result.path),
            "measured_bytes": result.measured_bytes,
            "gib_per_second": result.gib_per_second,
        }
    )

    assert "measured_bytes" in payload


def test_disk_read_benchmark_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "layer.bin"
    source.write_bytes(b"x" * (256 * 1024))

    status = cli_main(
        [
            "disk-read-benchmark",
            str(source),
            "--bytes-mib",
            "0.125",
            "--chunk-mib",
            "0.03125",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(source)
    assert payload["requested_bytes"] == 128 * 1024
    assert payload["measured_bytes"] == 128 * 1024
    assert payload["chunk_bytes"] == 32 * 1024
    assert payload["gib_per_second"] > 0


def test_disk_read_benchmark_cli_rejects_chunk_above_safety_cap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "layer.bin"
    source.write_bytes(b"x" * 1024)

    status = cli_main(
        [
            "disk-read-benchmark",
            str(source),
            "--bytes-mib",
            "0.001",
            "--chunk-mib",
            "2",
            "--max-chunk-mib",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "chunk_bytes" in captured.err
    assert "exceeds max_chunk_bytes" in captured.err
