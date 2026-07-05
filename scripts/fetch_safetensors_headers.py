#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


HEADER_MANIFEST_NAME = "largerlm.safetensors.headers.json"
_CONTENT_RANGE_RE = re.compile(r"bytes (?P<start>\d+)-(?P<end>\d+)/(?P<size>\d+|\*)")
DEFAULT_SMALL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "generation_config.json",
    "chat_template.jinja",
    "tokenization_glm.py",
    "configuration_glm.py",
)


def _headers(token_env: str) -> dict[str, str]:
    headers = {"User-Agent": "LargerLM-safetensors-header-probe/1"}
    token = os.environ.get(token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _resolve_url(endpoint: str, repo: str, revision: str, path: str) -> str:
    endpoint = endpoint.rstrip("/")
    repo_q = urllib.parse.quote(repo.strip("/"), safe="/")
    revision_q = urllib.parse.quote(revision, safe="")
    path_q = urllib.parse.quote(path, safe="/")
    return f"{endpoint}/{repo_q}/resolve/{revision_q}/{path_q}"


def _parse_content_range(value: str | None, *, url: str) -> int:
    if value is None:
        raise RuntimeError(f"missing Content-Range for ranged response: {url}")
    match = _CONTENT_RANGE_RE.fullmatch(value.strip())
    if match is None or match.group("size") == "*":
        raise RuntimeError(f"invalid Content-Range {value!r} for {url}")
    return int(match.group("size"))


def _unsupported_content_type_detail(raw: bytes) -> bool:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return b"Unsupported content type" in raw
    if isinstance(payload, dict):
        detail = payload.get("detail")
        return isinstance(detail, str) and detail == "Unsupported content type"
    return False


def _retry_unsupported_content_type_body(
    raw: bytes,
    *,
    attempt: int,
    attempts: int,
    retry_delay_seconds: float,
    url: str,
) -> bool:
    if not _unsupported_content_type_detail(raw):
        return False
    if attempt + 1 >= attempts:
        raise RuntimeError(
            "server returned transient Unsupported content type response "
            f"after {attempts} attempts for {url}"
        )
    if retry_delay_seconds > 0:
        time.sleep(retry_delay_seconds)
    return True


def _fetch_json(
    url: str,
    *,
    headers: dict[str, str],
    max_bytes: int,
    attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> Any:
    attempts = max(1, attempts)
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > max_bytes:
                    raise RuntimeError(
                        f"refusing to download {url}: Content-Length exceeds cap"
                    )
                raw = response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raw = exc.read(4096)
            if _retry_unsupported_content_type_body(
                raw,
                attempt=attempt,
                attempts=attempts,
                retry_delay_seconds=retry_delay_seconds,
                url=url,
            ):
                continue
            raise
        if len(raw) > max_bytes:
            raise RuntimeError(f"refusing to download {url}: response exceeds cap")
        if _retry_unsupported_content_type_body(
            raw,
            attempt=attempt,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
            url=url,
        ):
            continue
        return json.loads(raw)
    raise AssertionError("unreachable retry loop exit")


def _fetch_range(
    url: str,
    *,
    start: int,
    end: int,
    headers: dict[str, str],
    attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> tuple[bytes, int]:
    if end < start:
        raise ValueError("range end must be >= start")
    attempts = max(1, attempts)
    expected = end - start + 1
    for attempt in range(attempts):
        request_headers = dict(headers)
        request_headers["Range"] = f"bytes={start}-{end}"
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = response.getcode()
                if status != 206:
                    raw = response.read(4096)
                    if _retry_unsupported_content_type_body(
                        raw,
                        attempt=attempt,
                        attempts=attempts,
                        retry_delay_seconds=retry_delay_seconds,
                        url=url,
                    ):
                        continue
                    raise RuntimeError(
                        f"server did not honor Range for {url}; refusing full-weight download"
                    )
                raw = response.read(expected)
                file_size = _parse_content_range(
                    response.headers.get("Content-Range"),
                    url=url,
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read(4096)
            if _retry_unsupported_content_type_body(
                raw,
                attempt=attempt,
                attempts=attempts,
                retry_delay_seconds=retry_delay_seconds,
                url=url,
            ):
                continue
            raise
        if len(raw) != expected:
            raise RuntimeError(
                f"short ranged response for {url}: got {len(raw)}, expected {expected}"
            )
        return raw, file_size
    raise AssertionError("unreachable retry loop exit")


def _fetch_bytes(
    url: str,
    *,
    headers: dict[str, str],
    max_bytes: int,
    attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    missing_ok: bool = False,
) -> bytes | None:
    attempts = max(1, attempts)
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > max_bytes:
                    raise RuntimeError(
                        f"refusing to download {url}: Content-Length exceeds cap"
                    )
                raw = response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            raw = exc.read(4096)
            if _retry_unsupported_content_type_body(
                raw,
                attempt=attempt,
                attempts=attempts,
                retry_delay_seconds=retry_delay_seconds,
                url=url,
            ):
                continue
            raise
        if len(raw) > max_bytes:
            raise RuntimeError(f"refusing to download {url}: response exceeds cap")
        if _retry_unsupported_content_type_body(
            raw,
            attempt=attempt,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
            url=url,
        ):
            continue
        return raw
    raise AssertionError("unreachable retry loop exit")


def _fetch_safetensors_header(
    url: str,
    *,
    headers: dict[str, str],
    max_header_bytes: int,
    attempts: int = 3,
    retry_delay_seconds: float = 1.0,
) -> tuple[dict[str, Any], int, int]:
    prefix, file_size = _fetch_range(
        url,
        start=0,
        end=7,
        headers=headers,
        attempts=attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    header_len = struct.unpack("<Q", prefix)[0]
    if header_len > max_header_bytes:
        raise RuntimeError(
            f"refusing header for {url}: {header_len} bytes exceeds cap {max_header_bytes}"
        )
    header_raw, file_size_2 = _fetch_range(
        url,
        start=8,
        end=7 + header_len,
        headers=headers,
        attempts=attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    if file_size_2 != file_size:
        raise RuntimeError(f"inconsistent Content-Range file sizes for {url}")
    header = json.loads(header_raw)
    if not isinstance(header, dict):
        raise RuntimeError(f"safetensors header for {url} is not a JSON object")
    return header, 8 + header_len, file_size


def _unique_shards(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("model.safetensors.index.json does not contain a weight_map")
    shards = sorted({str(shard) for shard in weight_map.values()})
    if not shards:
        raise RuntimeError("model.safetensors.index.json weight_map is empty")
    return shards


def _safe_repo_file_path(path: str) -> Path:
    if not path:
        raise RuntimeError("small file path must be non-empty")
    parsed = Path(path)
    if parsed.is_absolute() or any(part == ".." for part in parsed.parts):
        raise RuntimeError(f"small file path escapes output directory: {path!r}")
    if any(part == "" for part in parsed.parts):
        raise RuntimeError(f"small file path is invalid: {path!r}")
    return parsed


def _dedupe_preserve_order(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _fetch_optional_small_files(
    *,
    repo: str,
    revision: str,
    endpoint: str,
    output: Path,
    paths: list[str],
    headers: dict[str, str],
    max_file_bytes: int,
    max_total_bytes: int,
    attempts: int,
    retry_delay_seconds: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = 0
    for raw_path in _dedupe_preserve_order(paths):
        relative = _safe_repo_file_path(raw_path)
        url = _resolve_url(endpoint, repo, revision, str(relative))
        data = _fetch_bytes(
            url,
            headers=headers,
            max_bytes=max_file_bytes,
            attempts=attempts,
            retry_delay_seconds=retry_delay_seconds,
            missing_ok=True,
        )
        if data is None:
            results.append({"path": str(relative), "present": False})
            continue
        total += len(data)
        if total > max_total_bytes:
            raise RuntimeError(
                "refusing to download small files: total size exceeds cap"
            )
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        results.append(
            {
                "path": str(relative),
                "present": True,
                "bytes": len(data),
            }
        )
        print(f"fetched small file: {relative} ({len(data)} bytes)", file=sys.stderr)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch only safetensors shard headers with HTTP Range and write a "
            "LargerLM metadata-only header manifest."
        )
    )
    parser.add_argument("repo", help="Hugging Face repo id, for example zai-org/GLM-5.2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--max-index-mib", type=float, default=64.0)
    parser.add_argument("--max-header-mib", type=float, default=256.0)
    parser.add_argument(
        "--fetch-small-files",
        action="store_true",
        help=(
            "also fetch common small checkpoint files such as config.json and "
            "tokenizer metadata, skipping missing files"
        ),
    )
    parser.add_argument(
        "--small-file",
        action="append",
        default=None,
        help=(
            "small repo file to fetch; may be repeated. When omitted with "
            "--fetch-small-files, a conservative default list is used"
        ),
    )
    parser.add_argument("--max-small-file-mib", type=float, default=32.0)
    parser.add_argument("--max-small-files-total-mib", type=float, default=128.0)
    parser.add_argument("--http-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.http_retries < 1:
        parser.error("--http-retries must be at least 1")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be non-negative")
    if args.max_small_file_mib <= 0:
        parser.error("--max-small-file-mib must be positive")
    if args.max_small_files_total_mib <= 0:
        parser.error("--max-small-files-total-mib must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    headers = _headers(args.token_env)
    index_url = _resolve_url(
        args.endpoint,
        args.repo,
        args.revision,
        "model.safetensors.index.json",
    )
    index = _fetch_json(
        index_url,
        headers=headers,
        max_bytes=int(args.max_index_mib * 1024 * 1024),
        attempts=args.http_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    if not isinstance(index, dict):
        raise RuntimeError("model.safetensors.index.json is not a JSON object")

    shards: dict[str, dict[str, Any]] = {}
    max_header_bytes = int(args.max_header_mib * 1024 * 1024)
    for shard in _unique_shards(index):
        shard_url = _resolve_url(args.endpoint, args.repo, args.revision, shard)
        header, data_start, file_size = _fetch_safetensors_header(
            shard_url,
            headers=headers,
            max_header_bytes=max_header_bytes,
            attempts=args.http_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        shards[shard] = {
            "file_size": file_size,
            "data_start": data_start,
            "header": header,
        }
        print(f"fetched header: {shard} ({data_start} metadata bytes)", file=sys.stderr)

    small_file_status: list[dict[str, Any]] = []
    small_file_paths = args.small_file
    if args.fetch_small_files and not small_file_paths:
        small_file_paths = list(DEFAULT_SMALL_FILES)
    if small_file_paths:
        small_file_status = _fetch_optional_small_files(
            repo=args.repo,
            revision=args.revision,
            endpoint=args.endpoint,
            output=output,
            paths=list(small_file_paths),
            headers=headers,
            max_file_bytes=int(args.max_small_file_mib * 1024 * 1024),
            max_total_bytes=int(args.max_small_files_total_mib * 1024 * 1024),
            attempts=args.http_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )

    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / HEADER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "source": {
                    "repo": args.repo,
                    "revision": args.revision,
                    "endpoint": args.endpoint,
                },
                "index": index,
                "shards": shards,
                "small_files": small_file_status,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {output / HEADER_MANIFEST_NAME}")
    print(f"shards: {len(shards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
