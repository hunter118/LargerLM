from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import unquote

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "download_safetensors_ranges.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "download_safetensors_ranges_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
download_ranges = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(download_ranges)


class _Response:
    def __init__(
        self,
        body: bytes,
        *,
        status: int,
        headers: dict[str, str],
        read_timeout_after_bytes: int | None = None,
    ) -> None:
        self._body = io.BytesIO(body)
        self._status = status
        self.headers = headers
        self._read_timeout_after_bytes = read_timeout_after_bytes
        self._bytes_read = 0

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._read_timeout_after_bytes is not None:
            if self._bytes_read >= self._read_timeout_after_bytes:
                raise TimeoutError("The read operation timed out")
            remaining_before_timeout = self._read_timeout_after_bytes - self._bytes_read
            if size < 0:
                size = remaining_before_timeout
            else:
                size = min(size, remaining_before_timeout)
        chunk = self._body.read(size)
        self._bytes_read += len(chunk)
        return chunk

    def getcode(self) -> int:
        return self._status


class _FakeRangeTransport:
    def __init__(
        self,
        files: dict[str, bytes],
        *,
        honor_range: bool = True,
        unsupported_content_type_failures: int = 0,
        short_first_range_bytes: int | None = None,
        read_timeout_after_first_range_bytes: int | None = None,
        transient_urlopen_failures: int = 0,
    ) -> None:
        self.files = files
        self.honor_range = honor_range
        self.unsupported_content_type_failures = unsupported_content_type_failures
        self.unsupported_content_type_seen = 0
        self.short_first_range_bytes = short_first_range_bytes
        self.read_timeout_after_first_range_bytes = read_timeout_after_first_range_bytes
        self.transient_urlopen_failures = transient_urlopen_failures
        self.transient_urlopen_seen = 0
        self.range_headers: list[str] = []

    def urlopen(self, request, timeout: float):  # type: ignore[no-untyped-def]
        if self.transient_urlopen_seen < self.transient_urlopen_failures:
            self.transient_urlopen_seen += 1
            raise download_ranges.urllib.error.URLError(
                "EOF occurred in violation of protocol"
            )

        if self.unsupported_content_type_seen < self.unsupported_content_type_failures:
            self.unsupported_content_type_seen += 1
            raise download_ranges.urllib.error.HTTPError(
                request.full_url,
                415,
                "Unsupported Media Type",
                {},
                io.BytesIO(b'{"detail":"Unsupported content type"}'),
            )

        name = Path(unquote(request.full_url)).name
        body = self.files[name]
        range_header = request.headers.get("Range")
        if not self.honor_range or range_header is None:
            return _Response(
                body,
                status=200,
                headers={"Content-Length": str(len(body))},
            )

        start_raw, end_raw = range_header.removeprefix("bytes=").split("-", 1)
        self.range_headers.append(range_header)
        start = int(start_raw)
        end = int(end_raw)
        chunk = body[start : end + 1]
        if self.short_first_range_bytes is not None and len(self.range_headers) == 1:
            chunk = chunk[: self.short_first_range_bytes]
        return _Response(
            chunk,
            status=206,
            headers={
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {start}-{end}/{len(body)}",
            },
            read_timeout_after_bytes=(
                self.read_timeout_after_first_range_bytes
                if len(self.range_headers) == 1
                else None
            ),
        )


def _write_handoff(
    tmp_path: Path,
    *,
    model_dir: Path,
    base_url: str,
    files: dict[str, bytes],
) -> Path:
    entries = []
    for name, body in files.items():
        entries.append(
            {
                "name": name,
                "url": f"{base_url}/{name}",
                "target_path": str(model_dir / name),
                "issue": "missing",
                "expected_file_bytes": len(body),
                "actual_file_bytes": None,
                "remaining_file_bytes": len(body),
            }
        )
    payload = {
        "schema": "largerlm.external_safetensors_download.v1",
        "version": 1,
        "model_dir": str(model_dir),
        "entries": entries,
    }
    path = tmp_path / "external-download.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture_files() -> dict[str, bytes]:
    return {
        "model-00001-of-00002.safetensors": b"abcdefghij",
        "model-00002-of-00002.safetensors": b"klmnopqrstuv",
    }


def _safetensors_bytes(header: dict[str, object], payload: bytes) -> bytes:
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return len(raw_header).to_bytes(8, "little") + raw_header + payload


def test_range_downloader_downloads_all_entries_with_clean_json(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--chunk-mib",
            "0.000004",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["selected_entries"] == 2
    assert payload["completed_entries"] == 2
    assert payload["downloaded_bytes"] == sum(len(body) for body in files.values())
    for name, body in files.items():
        assert (model_dir / name).read_bytes() == body


def test_range_downloader_selects_shard_range(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "2",
            "--end-index",
            "2",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out)["completed_entries"] == 1
    assert not (model_dir / "model-00001-of-00002.safetensors").exists()
    assert (
        model_dir / "model-00002-of-00002.safetensors"
    ).read_bytes() == files["model-00002-of-00002.safetensors"]


def test_range_downloader_byte_cap_leaves_resumable_partial(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    capped = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--chunk-mib",
            "0.000003",
            "--max-bytes",
            "5",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    first = json.loads(capsys.readouterr().out)
    resumed = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--chunk-mib",
            "0.000003",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    second = json.loads(capsys.readouterr().out)

    target = model_dir / "model-00001-of-00002.safetensors"
    assert capped == 0
    assert first["byte_cap_reached"] is True
    assert first["completed_entries"] == 0
    assert first["downloaded_bytes"] == 5
    assert resumed == 0
    assert second["completed_entries"] == 1
    assert second["downloaded_bytes"] == len(files[target.name]) - 5
    assert target.read_bytes() == files[target.name]


def test_range_downloader_resumes_short_ranged_response_in_same_run(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files, short_first_range_bytes=2)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--chunk-mib",
            "0.000004",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    target = model_dir / "model-00001-of-00002.safetensors"
    assert status == 0
    assert payload["ok"] is True
    assert payload["completed_entries"] == 1
    assert payload["downloaded_bytes"] == len(files[target.name])
    assert target.read_bytes() == files[target.name]
    assert transport.range_headers[:2] == ["bytes=0-3", "bytes=2-5"]
    assert "short ranged response" in captured.err


def test_range_downloader_resumes_read_timeout_after_partial_body(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files, read_timeout_after_first_range_bytes=2)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--chunk-mib",
            "0.000004",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    target = model_dir / "model-00001-of-00002.safetensors"
    assert status == 0
    assert payload["ok"] is True
    assert payload["completed_entries"] == 1
    assert payload["downloaded_bytes"] == len(files[target.name])
    assert target.read_bytes() == files[target.name]
    assert transport.range_headers[:2] == ["bytes=0-3", "bytes=2-5"]
    assert "interrupted ranged response" in captured.err


def test_range_downloader_discards_complete_file_with_mismatched_header(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = {"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    body = _safetensors_bytes(header, b"data")
    files = {"model-00001-of-00001.safetensors": body}
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    wrong = _safetensors_bytes(
        {"weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}},
        b"data",
    )
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(wrong)
    transport = _FakeRangeTransport(files)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["entries"][0]["expected_data_start"] = len(body) - 4
    payload["entries"][0]["expected_header"] = header
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    target = model_dir / "model-00001-of-00001.safetensors"
    assert status == 0
    assert json.loads(captured.out)["downloaded_bytes"] == len(body)
    assert target.read_bytes() == body
    assert transport.range_headers == [f"bytes=0-{len(body) - 1}"]
    assert "mismatched safetensors header" in captured.err


def test_range_downloader_discards_partial_with_incomplete_header(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    header = {"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    body = _safetensors_bytes(header, b"data")
    files = {"model-00001-of-00001.safetensors": body}
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(body[:9])
    transport = _FakeRangeTransport(files)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["entries"][0]["expected_data_start"] = len(body) - 4
    payload["entries"][0]["expected_header"] = header
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    target = model_dir / "model-00001-of-00001.safetensors"
    assert status == 0
    assert target.read_bytes() == body
    assert transport.range_headers == [f"bytes=0-{len(body) - 1}"]
    assert "incomplete safetensors header" in captured.err


def test_range_downloader_retries_unsupported_content_type(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files, unsupported_content_type_failures=1)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )

    assert transport.unsupported_content_type_seen == 1

    assert status == 0
    assert json.loads(capsys.readouterr().out)["completed_entries"] == 1
    assert (
        model_dir / "model-00001-of-00002.safetensors"
    ).read_bytes() == files["model-00001-of-00002.safetensors"]


def test_range_downloader_retries_transient_urlopen_errors(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files, transient_urlopen_failures=2)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--http-retries",
            "3",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert transport.transient_urlopen_seen == 2
    assert status == 0
    assert json.loads(captured.out)["completed_entries"] == 1
    assert "retrying range after transient error" in captured.err
    assert (
        model_dir / "model-00001-of-00002.safetensors"
    ).read_bytes() == files["model-00001-of-00002.safetensors"]


def test_range_downloader_rejects_servers_that_ignore_range(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _fixture_files()
    model_dir = tmp_path / "model"
    transport = _FakeRangeTransport(files, honor_range=False)
    monkeypatch.setattr(download_ranges.urllib.request, "urlopen", transport.urlopen)
    handoff = _write_handoff(
        tmp_path,
        model_dir=model_dir,
        base_url="https://example.test/repo",
        files=files,
    )

    status = download_ranges.main(
        [
            str(handoff),
            "--start-index",
            "1",
            "--end-index",
            "1",
            "--retry-delay-seconds",
            "0",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["ok"] is False
    assert "server did not honor Range" in payload["error"]
    assert not (model_dir / "model-00001-of-00002.safetensors").exists()
