#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_CONTENT_RANGE_RE = re.compile(r"bytes (?P<start>\d+)-(?P<end>\d+)/(?P<size>\d+|\*)")
_READ_BLOCK_BYTES = 1024 * 1024


class DownloadError(RuntimeError):
    pass


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _headers(token_env: str) -> dict[str, str]:
    headers = {"User-Agent": "LargerLM-safetensors-range-download/1"}
    token = os.environ.get(token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _unsupported_content_type_detail(raw: bytes) -> bool:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return b"Unsupported content type" in raw
    if isinstance(payload, dict):
        detail = payload.get("detail")
        return isinstance(detail, str) and detail == "Unsupported content type"
    return False


def _parse_content_range(value: str | None, *, url: str) -> tuple[int, int, int]:
    if value is None:
        raise DownloadError(f"missing Content-Range for ranged response: {url}")
    match = _CONTENT_RANGE_RE.fullmatch(value.strip())
    if match is None or match.group("size") == "*":
        raise DownloadError(f"invalid Content-Range {value!r} for {url}")
    return int(match.group("start")), int(match.group("end")), int(match.group("size"))


def _load_handoff(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DownloadError(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DownloadError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DownloadError(f"{path} must contain a JSON object")
    if payload.get("schema") != "largerlm.external_safetensors_download.v1":
        raise DownloadError(f"{path} schema must be largerlm.external_safetensors_download.v1")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise DownloadError(f"{path} entries must be a list")
    return payload


def _safe_target(model_dir: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise DownloadError(f"unsafe shard name in handoff: {name!r}")
    return model_dir / path


def _actual_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise DownloadError(f"failed to stat {path}: {exc}") from exc


def _has_unsupported_content_type_body(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            raw = f.read(4096)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DownloadError(f"failed to read {path}: {exc}") from exc
    return _unsupported_content_type_detail(raw)


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise DownloadError(f"short safetensors header prefix: {path}")
            header_len = int.from_bytes(prefix, "little")
            if header_len < 2 or header_len > 16 * 1024 * 1024:
                raise DownloadError(
                    f"invalid safetensors header length {header_len}: {path}"
                )
            raw = f.read(header_len)
            if len(raw) != header_len:
                raise DownloadError(f"short safetensors header body: {path}")
    except OSError as exc:
        raise DownloadError(f"failed to read safetensors header {path}: {exc}") from exc
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"failed to parse safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise DownloadError(f"safetensors header must be an object: {path}")
    return header, 8 + header_len


def _entry_optional_int(entry: dict[str, Any], key: str) -> int | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DownloadError(f"entry {entry.get('name')!r} {key} must be an integer")
    if value < 0:
        raise DownloadError(
            f"entry {entry.get('name')!r} {key} must be non-negative"
        )
    return value


def _entry_optional_header(entry: dict[str, Any]) -> dict[str, Any] | None:
    value = entry.get("expected_header")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DownloadError(
            f"entry {entry.get('name')!r} expected_header must be an object"
        )
    return value


def _verify_or_discard_local_prefix(
    *,
    target: Path,
    actual: int,
    expected_data_start: int | None,
    expected_header: dict[str, Any] | None,
) -> int:
    if actual == 0 or expected_data_start is None or expected_header is None:
        return actual
    if actual < expected_data_start:
        _progress(
            f"discarding local partial with incomplete safetensors header: {target} "
            f"({actual} < {expected_data_start} bytes)"
        )
        target.unlink(missing_ok=True)
        return 0
    try:
        header, data_start = _read_safetensors_header(target)
    except DownloadError as exc:
        _progress(f"discarding local partial with invalid safetensors header: {exc}")
        target.unlink(missing_ok=True)
        return 0
    if data_start != expected_data_start:
        _progress(
            f"discarding local partial with mismatched safetensors data_start: "
            f"{target} ({data_start} != {expected_data_start})"
        )
        target.unlink(missing_ok=True)
        return 0
    if header != expected_header:
        _progress(
            f"discarding local partial with mismatched safetensors header: {target}"
        )
        target.unlink(missing_ok=True)
        return 0
    return actual


def _read_error_body(exc: urllib.error.HTTPError) -> bytes:
    try:
        return exc.read(4096)
    except OSError:
        return b""


def _download_range_once(
    *,
    url: str,
    target: Path,
    start: int,
    end: int,
    expected_file_bytes: int,
    headers: dict[str, str],
    timeout_seconds: float,
) -> int:
    request_headers = dict(headers)
    request_headers["Range"] = f"bytes={start}-{end}"
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.getcode()
            if status != 206:
                raw = response.read(4096)
                if _unsupported_content_type_detail(raw):
                    raise DownloadError("transient Unsupported content type response")
                raise DownloadError(
                    f"server did not honor Range for {url}; refusing full shard download"
                )
            range_start, range_end, file_size = _parse_content_range(
                response.headers.get("Content-Range"),
                url=url,
            )
            if range_start != start or range_end != end:
                raise DownloadError(
                    f"server returned Content-Range {range_start}-{range_end}, "
                    f"expected {start}-{end}"
                )
            if file_size != expected_file_bytes:
                raise DownloadError(
                    f"server reports file size {file_size}, expected {expected_file_bytes}"
                )
            written = 0
            with target.open("ab") as f:
                while written < end - start + 1:
                    wanted = min(_READ_BLOCK_BYTES, end - start + 1 - written)
                    try:
                        chunk = response.read(wanted)
                    except (OSError, TimeoutError) as exc:
                        if written > 0:
                            _progress(
                                f"interrupted ranged response for {url}: wrote {written}, "
                                f"expected {end - start + 1}; resuming from byte "
                                f"{start + written}: {exc}"
                            )
                            return written
                        raise
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
    except urllib.error.HTTPError as exc:
        raw = _read_error_body(exc)
        if _unsupported_content_type_detail(raw):
            raise DownloadError("transient Unsupported content type response") from exc
        raise
    expected_range_bytes = end - start + 1
    if written != expected_range_bytes:
        if written > 0:
            _progress(
                f"short ranged response for {url}: wrote {written}, "
                f"expected {expected_range_bytes}; resuming from byte {start + written}"
            )
            return written
        raise DownloadError(
            f"short ranged response for {url}: wrote {written}, expected {expected_range_bytes}"
        )
    return written


def _download_range_with_retries(
    *,
    url: str,
    target: Path,
    start: int,
    end: int,
    expected_file_bytes: int,
    headers: dict[str, str],
    timeout_seconds: float,
    attempts: int,
    retry_delay_seconds: float,
) -> int:
    for attempt in range(1, attempts + 1):
        before = _actual_size(target)
        try:
            return _download_range_once(
                url=url,
                target=target,
                start=start,
                end=end,
                expected_file_bytes=expected_file_bytes,
                headers=headers,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            after = _actual_size(target)
            if after != before:
                raise
            if attempt >= attempts:
                raise
            _progress(
                f"retrying range after transient error "
                f"({attempt}/{attempts}) for {target}: {exc}"
            )
            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
    raise AssertionError("unreachable retry loop exit")


def _entry_int(entry: dict[str, Any], key: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DownloadError(f"entry {entry.get('name')!r} {key} must be an integer")
    return value


def _download_entry(
    *,
    index: int,
    entry: dict[str, Any],
    model_dir: Path,
    headers: dict[str, str],
    chunk_bytes: int,
    timeout_seconds: float,
    attempts: int,
    retry_delay_seconds: float,
    max_bytes_remaining: int | None,
) -> tuple[int, bool]:
    name = entry.get("name")
    url = entry.get("url")
    if not isinstance(name, str) or not name:
        raise DownloadError(f"entry #{index} name must be a non-empty string")
    if not isinstance(url, str) or not url:
        raise DownloadError(f"entry {name!r} url must be a non-empty string")
    expected = _entry_int(entry, "expected_file_bytes")
    expected_data_start = _entry_optional_int(entry, "expected_data_start")
    expected_header = _entry_optional_header(entry)
    target = _safe_target(model_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _has_unsupported_content_type_body(target):
        _progress(f"retrying transient Unsupported content type body for {target}")
        target.unlink()
    actual = _actual_size(target)
    if actual == expected:
        actual = _verify_or_discard_local_prefix(
            target=target,
            actual=actual,
            expected_data_start=expected_data_start,
            expected_header=expected_header,
        )
    if actual == expected:
        _progress(f"ok: {target} ({actual} bytes)")
        return 0, True
    if actual > expected:
        raise DownloadError(f"refusing to resume oversized shard: {target} ({actual} > {expected})")
    actual = _verify_or_discard_local_prefix(
        target=target,
        actual=actual,
        expected_data_start=expected_data_start,
        expected_header=expected_header,
    )

    downloaded = 0
    while actual < expected:
        remaining = expected - actual
        if max_bytes_remaining is not None and max_bytes_remaining <= 0:
            _progress(f"byte cap reached before {target} ({remaining} bytes remaining)")
            return downloaded, False
        span = min(chunk_bytes, remaining)
        if max_bytes_remaining is not None:
            span = min(span, max_bytes_remaining)
        start = actual
        end = actual + span - 1
        _progress(
            f"range download: {target} #{index} bytes {start}-{end}/{expected}",
        )
        wrote = _download_range_with_retries(
            url=url,
            target=target,
            start=start,
            end=end,
            expected_file_bytes=expected,
            headers=headers,
            timeout_seconds=timeout_seconds,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        downloaded += wrote
        actual += wrote
        if max_bytes_remaining is not None:
            max_bytes_remaining -= wrote
    _progress(f"ok: {target} ({actual} bytes)")
    return downloaded, True


def _positive_int_arg(parser: argparse.ArgumentParser, name: str, value: int) -> int:
    if value < 1:
        parser.error(f"{name} must be at least 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download safetensors shards from a LargerLM external download handoff "
            "using bounded HTTP Range requests."
        )
    )
    parser.add_argument("handoff_json")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=999999)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--chunk-mib", type=float, default=16.0)
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    start_index = _positive_int_arg(parser, "--start-index", args.start_index)
    end_index = _positive_int_arg(parser, "--end-index", args.end_index)
    if end_index < start_index:
        parser.error("--end-index must be >= --start-index")
    if args.max_bytes < 0:
        parser.error("--max-bytes must be non-negative")
    if args.chunk_mib <= 0:
        parser.error("--chunk-mib must be positive")
    if args.http_retries < 1:
        parser.error("--http-retries must be at least 1")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    handoff_path = Path(args.handoff_json)
    payload = _load_handoff(handoff_path)
    model_dir = Path(args.model_dir or payload.get("model_dir") or ".")
    entries = [entry for entry in payload["entries"] if isinstance(entry, dict)]
    headers = _headers(args.token_env)
    chunk_bytes = max(1, int(args.chunk_mib * 1024 * 1024))
    max_bytes_remaining = None if args.max_bytes == 0 else args.max_bytes

    downloaded = 0
    completed = 0
    selected = 0
    try:
        for index, entry in enumerate(entries, start=1):
            if index < start_index or index > end_index:
                continue
            selected += 1
            wrote, complete = _download_entry(
                index=index,
                entry=entry,
                model_dir=model_dir,
                headers=headers,
                chunk_bytes=chunk_bytes,
                timeout_seconds=float(args.timeout_seconds),
                attempts=args.http_retries,
                retry_delay_seconds=float(args.retry_delay_seconds),
                max_bytes_remaining=max_bytes_remaining,
            )
            downloaded += wrote
            if complete:
                completed += 1
            if max_bytes_remaining is not None:
                max_bytes_remaining -= wrote
                if max_bytes_remaining <= 0:
                    break
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "selected_entries": selected,
                        "completed_entries": completed,
                        "downloaded_bytes": downloaded,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    result = {
        "ok": True,
        "model_dir": str(model_dir),
        "selected_entries": selected,
        "completed_entries": completed,
        "downloaded_bytes": downloaded,
        "byte_cap_reached": args.max_bytes > 0 and downloaded >= args.max_bytes,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "range downloads finished: "
            f"{completed}/{selected} selected shards complete, {downloaded} bytes downloaded"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
