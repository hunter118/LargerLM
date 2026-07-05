from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import largerlm.decode_cache as decode_cache_module
from largerlm.config import load_config
from largerlm.decode_cache import (
    DecodeCacheError,
    build_decode_cache_layout,
    init_decode_cache_file,
    load_decode_cache_layout,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_decode_cache_layout_matches_glm_mla_and_dsa_widths() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")

    layout = build_decode_cache_layout(cfg, max_context_tokens=1024)

    assert layout.dtype == "BF16"
    assert layout.dtype_bytes == 2
    assert layout.total_bytes == (6 * (16 + 16) + 3 * 16) * 2 * 1024
    assert len(layout.segments) == 9
    assert [segment.kind for segment in layout.segments[:6]] == ["mla_kv"] * 6
    assert [segment.layer for segment in layout.segments[6:]] == [0, 2, 4]
    assert layout.segments[0].offset == 0
    assert layout.segments[1].offset == 64 * 1024
    assert layout.segments[0].token_stride_bytes == 64
    assert layout.segments[6].kind == "dsa_index"
    assert layout.segments[6].token_stride_bytes == 32
    assert layout.to_json()["segments"][0]["total_bytes"] == 64 * 1024


def test_decode_cache_layout_rejects_budget_overflow() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")

    with pytest.raises(DecodeCacheError, match="exceeds limit"):
        build_decode_cache_layout(
            cfg,
            max_context_tokens=1024,
            max_cache_bytes=1,
        )


def test_decode_cache_layout_rejects_context_above_model_max() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")

    with pytest.raises(DecodeCacheError, match="max_position_embeddings"):
        build_decode_cache_layout(cfg, max_context_tokens=1025)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_context_tokens": True}, "max_context_tokens must be an integer"),
        ({"max_context_tokens": 1.5}, "max_context_tokens must be an integer"),
        ({"alignment": False}, "alignment must be an integer"),
        ({"alignment": 1.5}, "alignment must be an integer"),
        ({"max_cache_bytes": True}, "max_cache_bytes must be an integer"),
        ({"max_cache_bytes": -1}, "max_cache_bytes must be non-negative"),
    ),
)
def test_decode_cache_layout_rejects_non_integer_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    params: dict[str, object] = {"max_context_tokens": 4}
    params.update(kwargs)

    with pytest.raises(DecodeCacheError, match=message):
        build_decode_cache_layout(cfg, **params)


def test_init_decode_cache_file_creates_sparse_logical_size(tmp_path: Path) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4)
    layout_path = tmp_path / "cache_layout.json"
    cache_path = tmp_path / "decode_cache.bin"
    layout_path.write_text(json.dumps(layout.to_json()), encoding="utf-8")

    result = init_decode_cache_file(
        layout_path,
        cache_path,
        disk_safety_margin_bytes=0,
    )

    assert result.cache_file_path == cache_path
    assert result.total_bytes == layout.total_bytes
    assert result.sparse is True
    assert result.existed is False
    assert cache_path.stat().st_size == layout.total_bytes
    loaded = load_decode_cache_layout(layout_path)
    assert loaded.total_bytes == layout.total_bytes
    assert len(loaded.segments) == len(layout.segments)


def test_decode_cache_layout_rejects_boolean_top_level_integer_field(
    tmp_path: Path,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4).to_json()
    layout["max_context_tokens"] = True
    layout_path = tmp_path / "cache_layout_bool_top.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(DecodeCacheError, match="max_context_tokens must be an integer"):
        load_decode_cache_layout(layout_path)


def test_decode_cache_layout_rejects_boolean_segment_integer_field(
    tmp_path: Path,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4).to_json()
    layout["segments"][0]["offset"] = False
    layout_path = tmp_path / "cache_layout_bool_segment.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(DecodeCacheError, match="offset must be an integer"):
        load_decode_cache_layout(layout_path)


def test_init_decode_cache_file_requires_force_for_existing_file(tmp_path: Path) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4)
    layout_path = tmp_path / "cache_layout.json"
    cache_path = tmp_path / "decode_cache.bin"
    layout_path.write_text(json.dumps(layout.to_json()), encoding="utf-8")
    cache_path.write_bytes(b"old")

    with pytest.raises(DecodeCacheError, match="already exists"):
        init_decode_cache_file(layout_path, cache_path, disk_safety_margin_bytes=0)

    result = init_decode_cache_file(
        layout_path,
        cache_path,
        force=True,
        disk_safety_margin_bytes=0,
    )
    assert result.existed is True
    assert cache_path.stat().st_size == layout.total_bytes


def test_init_decode_cache_file_rejects_max_cache_budget(tmp_path: Path) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4)
    layout_path = tmp_path / "cache_layout.json"
    layout_path.write_text(json.dumps(layout.to_json()), encoding="utf-8")

    with pytest.raises(DecodeCacheError, match="exceeds limit"):
        init_decode_cache_file(
            layout_path,
            tmp_path / "decode_cache.bin",
            max_cache_bytes=layout.total_bytes - 1,
            disk_safety_margin_bytes=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_cache_bytes": True}, "max_cache_bytes must be an integer"),
        ({"max_cache_bytes": 1.5}, "max_cache_bytes must be an integer"),
        (
            {"disk_safety_margin_bytes": False},
            "disk_safety_margin_bytes must be an integer",
        ),
        (
            {"disk_safety_margin_bytes": -1},
            "disk_safety_margin_bytes must be non-negative",
        ),
    ),
)
def test_init_decode_cache_file_rejects_non_integer_controls(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4)
    layout_path = tmp_path / "cache_layout.json"
    layout_path.write_text(json.dumps(layout.to_json()), encoding="utf-8")

    with pytest.raises(DecodeCacheError, match=message):
        init_decode_cache_file(
            layout_path,
            tmp_path / "decode_cache.bin",
            **kwargs,
        )


def test_init_decode_cache_file_rejects_disk_budget(tmp_path: Path) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4)
    layout_path = tmp_path / "cache_layout.json"
    layout_path.write_text(json.dumps(layout.to_json()), encoding="utf-8")

    with pytest.raises(DecodeCacheError, match="not enough free disk"):
        init_decode_cache_file(
            layout_path,
            tmp_path / "decode_cache.bin",
            disk_safety_margin_bytes=10**30,
        )


def test_init_decode_cache_file_removes_partial_file_on_truncate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    layout = build_decode_cache_layout(cfg, max_context_tokens=4)
    layout_path = tmp_path / "cache_layout.json"
    cache_path = tmp_path / "decode_cache.bin"
    layout_path.write_text(json.dumps(layout.to_json()), encoding="utf-8")

    def fail_truncate(fd: int, size: int) -> None:
        os.write(fd, b"partial")
        raise OSError("truncate exploded")

    monkeypatch.setattr(decode_cache_module.os, "ftruncate", fail_truncate)

    with pytest.raises(DecodeCacheError, match="failed to initialize decode cache"):
        init_decode_cache_file(
            layout_path,
            cache_path,
            disk_safety_margin_bytes=0,
        )

    assert not cache_path.exists()
