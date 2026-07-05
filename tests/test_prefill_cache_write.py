from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from largerlm import prefill_execute
from largerlm.cli import main as cli_main
from largerlm.prefill_execute import PrefillExecuteError, write_prefill_kv_cache_batch


def _f32(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def _bf16(values: tuple[float, ...]) -> bytes:
    return b"".join(_f32_to_bf16(value) for value in values)


def _write_cache_layout(root: Path, *, dtype: str = "BF16", dtype_bytes: int = 2) -> Path:
    segment_total_bytes = 3 * dtype_bytes * 4
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "max_context_tokens": 4,
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "alignment": 8,
        "total_bytes": 8 + segment_total_bytes,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": 1,
                "offset": 8,
                "width": 3,
                "dtype": dtype,
                "dtype_bytes": dtype_bytes,
                "token_stride_bytes": 3 * dtype_bytes,
                "max_context_tokens": 4,
                "total_bytes": segment_total_bytes,
            }
        ],
    }
    path = root / "cache_layout.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    return path


def test_write_prefill_kv_cache_batch_streams_bf16_rows(tmp_path: Path) -> None:
    layout = _write_cache_layout(tmp_path)
    cache = tmp_path / "decode_cache.bin"
    cache.write_bytes(b"\0" * 32)
    input_path = tmp_path / "kv_a.f32"
    input_path.write_bytes(_f32((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))

    result = write_prefill_kv_cache_batch(
        cache_layout_path=layout,
        cache_file_path=cache,
        layer=1,
        input_f32_path=input_path,
        start_position=1,
        batch_tokens=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
    )

    assert result.width == 3
    assert result.dtype == "BF16"
    assert result.token_stride_bytes == 6
    assert result.first_write_offset == 14
    assert result.input_bytes == 24
    assert result.encoded_bytes == 12
    assert result.encoder == "bitcast_bf16"
    assert result.write_chunks == 1
    assert result.write_chunk_tokens == 2
    assert result.estimated_peak_bytes == 36
    data = cache.read_bytes()
    assert data[:14] == b"\0" * 14
    assert data[14:20] == _bf16((1.0, 2.0, 3.0))
    assert data[20:26] == _bf16((4.0, 5.0, 6.0))
    assert data[26:] == b"\0" * 6


def test_write_prefill_kv_cache_batch_chunks_bf16_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prefill_execute, "PREFILL_CACHE_WRITE_CHUNK_BYTES", 10)
    layout = _write_cache_layout(tmp_path)
    cache = tmp_path / "decode_cache.bin"
    cache.write_bytes(b"\0" * 32)
    input_path = tmp_path / "kv_a.f32"
    input_path.write_bytes(_f32((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)))

    result = write_prefill_kv_cache_batch(
        cache_layout_path=layout,
        cache_file_path=cache,
        layer=1,
        input_f32_path=input_path,
        start_position=0,
        batch_tokens=3,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
    )

    assert result.write_chunks == 3
    assert result.write_chunk_tokens == 1
    assert result.estimated_peak_bytes == 18
    assert cache.read_bytes()[8:26] == _bf16(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    )


def test_write_prefill_kv_cache_batch_copies_f32_rows(tmp_path: Path) -> None:
    layout = _write_cache_layout(tmp_path, dtype="F32", dtype_bytes=4)
    cache = tmp_path / "decode_cache.bin"
    cache.write_bytes(b"\0" * 64)
    input_path = tmp_path / "kv_a.f32"
    rows = _f32((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    input_path.write_bytes(rows)

    result = write_prefill_kv_cache_batch(
        cache_layout_path=layout,
        cache_file_path=cache,
        layer=1,
        input_f32_path=input_path,
        start_position=1,
        batch_tokens=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
    )

    assert result.encoder == "copy_f32"
    assert result.encoded_bytes == len(rows)
    assert cache.read_bytes()[20:44] == rows


def test_prefill_cache_write_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_cache_layout(tmp_path)
    cache = tmp_path / "decode_cache.bin"
    cache.write_bytes(b"\0" * 32)
    input_path = tmp_path / "kv_a.f32"
    input_path.write_bytes(_f32((1.0, 2.0, 3.0)))

    status = cli_main(
        [
            "prefill-cache-write",
            str(layout),
            str(cache),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--start-position",
            "0",
            "--batch-tokens",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 1
    assert payload["encoded_bytes"] == 6
    assert payload["first_write_offset"] == 8
    assert payload["encoder"] == "bitcast_bf16"
    assert payload["write_chunks"] == 1
    assert payload["estimated_peak_bytes"] == 18
    assert cache.read_bytes()[8:14] == _bf16((1.0, 2.0, 3.0))


def test_write_prefill_kv_cache_batch_rejects_wrong_input_size(tmp_path: Path) -> None:
    layout = _write_cache_layout(tmp_path)
    cache = tmp_path / "decode_cache.bin"
    cache.write_bytes(b"\0" * 32)
    input_path = tmp_path / "kv_a.f32"
    input_path.write_bytes(_f32((1.0, 2.0)))

    with pytest.raises(PrefillExecuteError, match="input bytes"):
        write_prefill_kv_cache_batch(
            cache_layout_path=layout,
            cache_file_path=cache,
            layer=1,
            input_f32_path=input_path,
            start_position=0,
            batch_tokens=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
        )
