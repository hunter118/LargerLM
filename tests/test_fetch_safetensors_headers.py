from __future__ import annotations

import importlib.util
import io
import json
import struct
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_safetensors_headers.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "fetch_safetensors_headers_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
fetch_headers = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(fetch_headers)


class _Response:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = io.BytesIO(body)
        self._status = status
        self.headers = headers or {}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getcode(self) -> int:
        return self._status


def _unsupported_content_type_error(url: str):
    return fetch_headers.urllib.error.HTTPError(
        url,
        415,
        "Unsupported Media Type",
        {},
        io.BytesIO(b'{"detail":"Unsupported content type"}'),
    )


def test_fetch_json_retries_unsupported_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        calls.append(request.full_url)
        if len(calls) == 1:
            raise _unsupported_content_type_error(request.full_url)
        return _Response(b'{"ok": true}', headers={"Content-Length": "12"})

    monkeypatch.setattr(fetch_headers.urllib.request, "urlopen", fake_urlopen)

    payload = fetch_headers._fetch_json(
        "https://huggingface.co/repo/resolve/main/model.safetensors.index.json",
        headers={},
        max_bytes=128,
        attempts=2,
        retry_delay_seconds=0,
    )

    assert payload == {"ok": True}
    assert len(calls) == 2


def test_fetch_json_retries_unsupported_content_type_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        calls.append(request.full_url)
        if len(calls) == 1:
            return _Response(
                b'{"detail":"Unsupported content type"}',
                headers={"Content-Length": "37"},
            )
        return _Response(b'{"ok": true}', headers={"Content-Length": "12"})

    monkeypatch.setattr(fetch_headers.urllib.request, "urlopen", fake_urlopen)

    payload = fetch_headers._fetch_json(
        "https://huggingface.co/repo/resolve/main/model.safetensors.index.json",
        headers={},
        max_bytes=128,
        attempts=2,
        retry_delay_seconds=0,
    )

    assert payload == {"ok": True}
    assert len(calls) == 2


def test_fetch_range_retries_unsupported_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        calls.append(request.headers["Range"])
        if len(calls) == 1:
            raise _unsupported_content_type_error(request.full_url)
        return _Response(
            b"abcdefgh",
            status=206,
            headers={"Content-Range": "bytes 0-7/16"},
        )

    monkeypatch.setattr(fetch_headers.urllib.request, "urlopen", fake_urlopen)

    raw, file_size = fetch_headers._fetch_range(
        "https://huggingface.co/repo/resolve/main/model-00001.safetensors",
        start=0,
        end=7,
        headers={},
        attempts=2,
        retry_delay_seconds=0,
    )

    assert raw == b"abcdefgh"
    assert file_size == 16
    assert calls == ["bytes=0-7", "bytes=0-7"]


def test_fetch_range_retries_unsupported_content_type_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        calls.append(request.headers["Range"])
        if len(calls) == 1:
            return _Response(
                b'{"detail":"Unsupported content type"}',
                status=200,
                headers={"Content-Length": "37"},
            )
        return _Response(
            b"abcdefgh",
            status=206,
            headers={"Content-Range": "bytes 0-7/16"},
        )

    monkeypatch.setattr(fetch_headers.urllib.request, "urlopen", fake_urlopen)

    raw, file_size = fetch_headers._fetch_range(
        "https://huggingface.co/repo/resolve/main/model-00001.safetensors",
        start=0,
        end=7,
        headers={},
        attempts=2,
        retry_delay_seconds=0,
    )

    assert raw == b"abcdefgh"
    assert file_size == 16
    assert calls == ["bytes=0-7", "bytes=0-7"]


def test_unsupported_content_type_retry_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        raise _unsupported_content_type_error(request.full_url)

    monkeypatch.setattr(fetch_headers.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unsupported content type"):
        fetch_headers._fetch_json(
            "https://huggingface.co/repo/resolve/main/model.safetensors.index.json",
            headers={},
            max_bytes=128,
            attempts=3,
            retry_delay_seconds=0,
        )

    assert calls == 3


def test_fetch_headers_main_can_fetch_optional_small_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    header = {
        "tensor": {
            "dtype": "U8",
            "shape": [1],
            "data_offsets": [0, 1],
        }
    }
    header_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    file_size = 8 + len(header_raw) + 1
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {"tensor": "model-00001-of-00001.safetensors"},
    }

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        url = request.full_url
        range_header = request.headers.get("Range")
        if url.endswith("/model.safetensors.index.json"):
            return _Response(
                json.dumps(index).encode("utf-8"),
                headers={"Content-Length": "100"},
            )
        if url.endswith("/model-00001-of-00001.safetensors"):
            assert range_header is not None
            if range_header == "bytes=0-7":
                return _Response(
                    struct.pack("<Q", len(header_raw)),
                    status=206,
                    headers={"Content-Range": f"bytes 0-7/{file_size}"},
                )
            assert range_header == f"bytes=8-{7 + len(header_raw)}"
            return _Response(
                header_raw,
                status=206,
                headers={
                    "Content-Range": f"bytes 8-{7 + len(header_raw)}/{file_size}"
                },
            )
        if url.endswith("/config.json"):
            raw = b'{"model_type":"glm_moe_dsa"}'
            return _Response(raw, headers={"Content-Length": str(len(raw))})
        if url.endswith("/tokenizer_config.json"):
            raise fetch_headers.urllib.error.HTTPError(
                url,
                404,
                "Not Found",
                {},
                io.BytesIO(b"not found"),
            )
        raise AssertionError(url)

    monkeypatch.setattr(fetch_headers.urllib.request, "urlopen", fake_urlopen)

    status = fetch_headers.main(
        [
            "tiny-repo",
            "--endpoint",
            "https://example.test",
            "--output-dir",
            str(tmp_path),
            "--small-file",
            "config.json",
            "--small-file",
            "tokenizer_config.json",
            "--retry-delay-seconds",
            "0",
        ]
    )

    assert status == 0
    assert (tmp_path / "config.json").read_text(encoding="utf-8") == (
        '{"model_type":"glm_moe_dsa"}'
    )
    manifest = json.loads(
        (tmp_path / fetch_headers.HEADER_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["small_files"] == [
        {"bytes": 28, "path": "config.json", "present": True},
        {"path": "tokenizer_config.json", "present": False},
    ]
    assert manifest["shards"]["model-00001-of-00001.safetensors"]["file_size"] == (
        file_size
    )
    assert "fetched small file: config.json" in capsys.readouterr().err


def test_small_file_path_must_stay_inside_output_dir() -> None:
    with pytest.raises(RuntimeError, match="escapes output directory"):
        fetch_headers._safe_repo_file_path("../config.json")
