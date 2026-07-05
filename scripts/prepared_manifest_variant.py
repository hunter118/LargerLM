#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.prepared import PreparedManifestError, load_prepared_manifest


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(path_or_dir: Path) -> Path:
    return path_or_dir / "manifest.json" if path_or_dir.is_dir() else path_or_dir


def _relative_to_base(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError as exc:
        raise SystemExit(f"{path} must be inside {base}") from exc


def build_variant_manifest(
    *,
    source_manifest: Path,
    experts_layout: Path,
    output_manifest: Path,
    label: str,
) -> dict[str, Any]:
    source_manifest = _manifest_path(source_manifest)
    output_manifest = _manifest_path(output_manifest)
    if output_manifest.parent.resolve() != source_manifest.parent.resolve():
        raise SystemExit(
            "--output-manifest must be in the same directory as the source manifest"
        )
    payload = _load_json_object(source_manifest)
    if payload.get("version") != 1:
        raise SystemExit(f"{source_manifest} must be a version 1 prepared manifest")
    if not isinstance(payload.get("experts_layout"), str):
        raise SystemExit(f"{source_manifest} missing experts_layout")
    base = source_manifest.parent
    if not experts_layout.is_absolute():
        experts_layout = base / experts_layout
    if not experts_layout.exists():
        raise SystemExit(f"{experts_layout} does not exist")

    variant = dict(payload)
    variant["experts_layout"] = _relative_to_base(experts_layout, base=base)
    variant["manifest_variant"] = {
        "schema": "largerlm.prepared_manifest_variant.v1",
        "label": label,
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": _sha256(source_manifest),
        "experts_layout": variant["experts_layout"],
        "experts_layout_sha256": _sha256(experts_layout),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return variant


def write_validated_variant_manifest(
    *,
    source_manifest: Path,
    experts_layout: Path,
    output_manifest: Path,
    label: str,
    force: bool = False,
) -> dict[str, Any]:
    output_manifest = _manifest_path(output_manifest)
    if output_manifest.exists() and not force:
        raise SystemExit(f"{output_manifest} already exists; pass --force to replace it")
    variant = build_variant_manifest(
        source_manifest=source_manifest,
        experts_layout=experts_layout,
        output_manifest=output_manifest,
        label=label,
    )
    tmp = output_manifest.with_name(output_manifest.name + ".validate-tmp")
    try:
        _write_json_atomic(tmp, variant)
        try:
            loaded = load_prepared_manifest(tmp)
        except PreparedManifestError as exc:
            raise SystemExit(f"variant manifest failed prepared validation: {exc}") from exc
        _write_json_atomic(output_manifest, variant)
        return {
            "schema": "largerlm.prepared_manifest_variant_result.v1",
            "ok": True,
            "output_manifest": str(output_manifest),
            "source_manifest": str(_manifest_path(source_manifest)),
            "experts_layout": str(loaded.experts_layout),
            "expert_layout_bytes": loaded.expert_layout_bytes,
            "resident_layout_bytes": loaded.resident_layout_bytes,
            "decode_cache_layout_bytes": loaded.decode_cache_layout_bytes,
            "decode_cache_file_bytes": loaded.decode_cache_file_bytes,
            "manifest_variant": variant["manifest_variant"],
        }
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a small validated prepared manifest variant."
    )
    parser.add_argument("source_manifest", type=Path, help="prepared manifest or dir")
    parser.add_argument(
        "--experts-layout",
        type=Path,
        required=True,
        help="replacement experts layout path, relative to the source manifest directory",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="output manifest path in the same prepared directory",
    )
    parser.add_argument("--label", default="variant")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = write_validated_variant_manifest(
        source_manifest=args.source_manifest,
        experts_layout=args.experts_layout,
        output_manifest=args.output_manifest,
        label=args.label,
        force=args.force,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"wrote {result['output_manifest']}")
        print(f"experts layout: {result['experts_layout']}")
        print(f"expert bytes: {result['expert_layout_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
