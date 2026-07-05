from __future__ import annotations

from pathlib import Path

import pytest

from largerlm.safety import (
    SafetyError,
    check_chunk_budget,
    check_memory_budget,
    disk_budget,
    estimate_pack_peak_heap_bytes,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"chunk_size": True, "max_chunk_size": 8}, "chunk size must be an integer"),
        ({"chunk_size": 1.5, "max_chunk_size": 8}, "chunk size must be an integer"),
        ({"chunk_size": 1, "max_chunk_size": False}, "max chunk size must be an integer"),
        ({"chunk_size": 0, "max_chunk_size": 8}, "chunk size must be positive"),
        ({"chunk_size": 9, "max_chunk_size": 8}, "exceeds safety cap"),
    ),
)
def test_check_chunk_budget_rejects_invalid_integer_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(SafetyError, match=message):
        check_chunk_budget(**kwargs)


def test_estimate_pack_peak_heap_bytes_rejects_boolean_chunk() -> None:
    with pytest.raises(SafetyError, match="chunk size must be an integer"):
        estimate_pack_peak_heap_bytes(chunk_size=True)


def test_check_memory_budget_rejects_boolean_limits() -> None:
    with pytest.raises(SafetyError, match="estimated peak bytes must be an integer"):
        check_memory_budget(True, 1024)
    with pytest.raises(SafetyError, match="max memory bytes must be an integer"):
        check_memory_budget(0, False)


@pytest.mark.parametrize(
    ("required_bytes", "safety_margin_bytes", "message"),
    (
        (True, 0, "required bytes must be an integer"),
        (1.5, 0, "required bytes must be an integer"),
        (0, False, "safety margin bytes must be an integer"),
        (-1, 0, "required bytes must be non-negative"),
        (0, -1, "safety margin bytes must be non-negative"),
    ),
)
def test_disk_budget_rejects_invalid_integer_controls(
    tmp_path: Path,
    required_bytes: object,
    safety_margin_bytes: object,
    message: str,
) -> None:
    with pytest.raises(SafetyError, match=message):
        disk_budget(
            tmp_path,
            required_bytes,
            safety_margin_bytes=safety_margin_bytes,
        )


def test_disk_budget_returns_integer_budget(tmp_path: Path) -> None:
    budget = disk_budget(tmp_path, 1024, safety_margin_bytes=2048)

    assert budget.output_dir == tmp_path
    assert budget.required_bytes == 1024
    assert budget.safety_margin_bytes == 2048
    assert isinstance(budget.available_bytes, int)
