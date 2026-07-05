#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _nonnegative_int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _positive_int(value: object) -> int | None:
    parsed = _nonnegative_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
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


def _expert_order(value: object, *, num_experts: int, label: str) -> list[int]:
    order: list[int] = []
    for item in _as_list(value):
        expert = _nonnegative_int(item)
        if expert is None:
            raise SystemExit(f"{label} expert order entries must be integers")
        order.append(expert)
    if len(order) != num_experts:
        raise SystemExit(f"{label} expert order length must match num_experts")
    if sorted(order) != list(range(num_experts)):
        raise SystemExit(f"{label} expert order must be a permutation")
    return order


def _layer_by_id(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw_layer in _as_list(layout.get("layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        if layer_id is not None:
            result[layer_id] = layer
    return result


def _selected_plan_layers(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw_layer in _as_list(plan.get("selected_layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        if layer_id is not None:
            result[layer_id] = layer
    return result


def _current_order(layer: dict[str, Any], *, num_experts: int) -> list[int]:
    if "expert_order" not in layer:
        return list(range(num_experts))
    return _expert_order(
        layer.get("expert_order"),
        num_experts=num_experts,
        label=f"layout layer {layer.get('layer')}",
    )


def _physical_slot_by_expert(order: list[int]) -> dict[int, int]:
    return {expert: slot for slot, expert in enumerate(order)}


def _copy_slot(
    *,
    src_fd: int,
    dst_fd: int,
    src_offset: int,
    dst_offset: int,
    size: int,
    copy_chunk_bytes: int,
) -> tuple[int, int, int]:
    copied = 0
    read_calls = 0
    write_calls = 0
    while copied < size:
        chunk_size = min(copy_chunk_bytes, size - copied)
        data = os.pread(src_fd, chunk_size, src_offset + copied)
        read_calls += 1
        if len(data) != chunk_size:
            raise OSError(
                f"short read at offset {src_offset + copied}: "
                f"{len(data)} != {chunk_size}"
            )
        written_total = 0
        while written_total < len(data):
            written = os.pwrite(
                dst_fd,
                data[written_total:],
                dst_offset + copied + written_total,
            )
            write_calls += 1
            if written <= 0:
                raise OSError("short write while repacking expert layer")
            written_total += written
        copied += chunk_size
    return copied, read_calls, write_calls


def _repack_layer_file(
    *,
    source: Path,
    destination: Path,
    num_experts: int,
    expert_slot_bytes: int,
    current_order: list[int],
    proposed_order: list[int],
    copy_chunk_bytes: int,
) -> dict[str, object]:
    if source.stat().st_size != num_experts * expert_slot_bytes:
        raise SystemExit(f"{source} size does not match layer layout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    source_slot_by_expert = _physical_slot_by_expert(current_order)
    copied_bytes = 0
    read_calls = 0
    write_calls = 0
    try:
        with source.open("rb") as src, tmp.open("wb") as dst:
            os.ftruncate(dst.fileno(), num_experts * expert_slot_bytes)
            for dst_slot, expert in enumerate(proposed_order):
                src_slot = source_slot_by_expert[expert]
                copied, reads, writes = _copy_slot(
                    src_fd=src.fileno(),
                    dst_fd=dst.fileno(),
                    src_offset=src_slot * expert_slot_bytes,
                    dst_offset=dst_slot * expert_slot_bytes,
                    size=expert_slot_bytes,
                    copy_chunk_bytes=copy_chunk_bytes,
                )
                copied_bytes += copied
                read_calls += reads
                write_calls += writes
        tmp.replace(destination)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {
        "operation": "repack",
        "source": str(source),
        "destination": str(destination),
        "bytes": copied_bytes,
        "read_calls": read_calls,
        "write_calls": write_calls,
        "sha256": _sha256(destination),
    }


def _link_layer_file(*, source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError as exc:
        raise SystemExit(
            f"failed to hardlink unchanged layer {source} -> {destination}: {exc}"
        ) from exc
    return {
        "operation": "hardlink",
        "source": str(source),
        "destination": str(destination),
        "bytes": source.stat().st_size,
    }


def _ensure_output_dir(path: Path, *, execute: bool) -> None:
    if not execute:
        return
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"output dir {path} already exists and is not empty")
    path.mkdir(parents=True, exist_ok=True)


def _repack_write_guard(
    *,
    output_dir: Path,
    repack_write_bytes: int,
    max_repack_write_bytes: int,
    min_free_bytes_after_write: int,
) -> dict[str, object]:
    parent = output_dir if output_dir.exists() else output_dir.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    max_ok = (
        True if max_repack_write_bytes <= 0 else repack_write_bytes <= max_repack_write_bytes
    )
    free_after_write = usage.free - repack_write_bytes
    free_ok = free_after_write >= min_free_bytes_after_write
    return {
        "path": str(parent),
        "repack_write_bytes": repack_write_bytes,
        "max_repack_write_bytes": max_repack_write_bytes,
        "max_repack_write_ok": max_ok,
        "disk_free_bytes": usage.free,
        "min_free_bytes_after_write": min_free_bytes_after_write,
        "disk_free_after_write_bytes": free_after_write,
        "disk_free_after_write_ok": free_ok,
        "ok": bool(max_ok and free_ok),
    }


def _patched_layout(layout: dict[str, Any], selected: dict[int, dict[str, Any]]) -> dict[str, Any]:
    patched = copy.deepcopy(layout)
    for layer in _as_list(patched.get("layers")):
        layer_map = _as_mapping(layer)
        layer_id = _nonnegative_int(layer_map.get("layer"))
        if layer_id is None or layer_id not in selected:
            continue
        layer_map["expert_order"] = list(selected[layer_id]["proposed_expert_order"])
    return patched


def build_repack_manifest(
    *,
    plan_path: Path,
    output_dir: Path,
    execute: bool = False,
    copy_chunk_bytes: int = 64 * 1024 * 1024,
    max_repack_write_bytes: int = 8 * 1024**3,
    min_free_bytes_after_write: int = 20 * 1024**3,
) -> dict[str, Any]:
    if copy_chunk_bytes <= 0:
        raise SystemExit("copy chunk bytes must be positive")
    plan = _load_json(plan_path)
    if plan.get("schema") != "largerlm.expert_order_repack_plan.v1":
        raise SystemExit("plan schema must be largerlm.expert_order_repack_plan.v1")
    expert_layout_path = Path(str(plan.get("expert_layout")))
    layout = _load_json(expert_layout_path)
    source_dir = expert_layout_path.parent
    selected = _selected_plan_layers(plan)
    layout_layers = _layer_by_id(layout)

    layer_operations: list[dict[str, object]] = []
    selected_layer_ids = set(selected)
    for raw_layer in _as_list(layout.get("layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        layer_file = layer.get("layer_file")
        num_experts = _positive_int(layer.get("num_experts"))
        slot_bytes = _positive_int(layer.get("expert_slot_bytes"))
        if (
            layer_id is None
            or not isinstance(layer_file, str)
            or num_experts is None
            or slot_bytes is None
        ):
            raise SystemExit("expert layout layer is missing required fields")
        source = source_dir / layer_file
        destination = output_dir / layer_file
        if layer_id in selected_layer_ids:
            plan_layer = selected[layer_id]
            proposed_order = _expert_order(
                plan_layer.get("proposed_expert_order"),
                num_experts=num_experts,
                label=f"plan layer {layer_id}",
            )
            operation: dict[str, object] = {
                "operation": "repack",
                "layer": layer_id,
                "source": str(source),
                "destination": str(destination),
                "expected_layer_file_bytes": num_experts * slot_bytes,
                "copy_chunk_bytes": copy_chunk_bytes,
                "executed": False,
                "total_range_reduction": plan_layer.get("total_range_reduction"),
                "total_current_range_count": plan_layer.get(
                    "total_current_range_count"
                ),
                "total_candidate_range_count": plan_layer.get(
                    "total_candidate_range_count"
                ),
            }
            layer_operations.append(operation)
        else:
            operation = {
                "operation": "hardlink",
                "layer": layer_id,
                "source": str(source),
                "destination": str(destination),
                "bytes": source.stat().st_size if source.exists() else None,
                "executed": False,
            }
            layer_operations.append(operation)

    patched_layout = _patched_layout(layout, selected)
    output_layout = output_dir / "layout.json"
    repack_bytes = sum(
        int(op.get("expected_layer_file_bytes") or op.get("bytes") or 0)
        for op in layer_operations
        if op.get("operation") == "repack"
    )
    hardlink_bytes = sum(
        int(op.get("bytes") or 0)
        for op in layer_operations
        if op.get("operation") == "hardlink"
    )
    write_guard = _repack_write_guard(
        output_dir=output_dir,
        repack_write_bytes=repack_bytes,
        max_repack_write_bytes=max_repack_write_bytes,
        min_free_bytes_after_write=min_free_bytes_after_write,
    )
    if execute and not write_guard["ok"]:
        raise SystemExit(f"repack write guard failed: {write_guard}")
    _ensure_output_dir(output_dir, execute=execute)
    if execute:
        for operation in layer_operations:
            source = Path(str(operation["source"]))
            destination = Path(str(operation["destination"]))
            if operation.get("operation") == "repack":
                layer_id = _nonnegative_int(operation.get("layer"))
                assert layer_id is not None
                plan_layer = selected[layer_id]
                layout_layer = layout_layers[layer_id]
                num_experts = _positive_int(layout_layer.get("num_experts"))
                slot_bytes = _positive_int(layout_layer.get("expert_slot_bytes"))
                assert num_experts is not None and slot_bytes is not None
                current_order = _current_order(layout_layer, num_experts=num_experts)
                proposed_order = _expert_order(
                    plan_layer.get("proposed_expert_order"),
                    num_experts=num_experts,
                    label=f"plan layer {layer_id}",
                )
                operation.update(
                    _repack_layer_file(
                        source=source,
                        destination=destination,
                        num_experts=num_experts,
                        expert_slot_bytes=slot_bytes,
                        current_order=current_order,
                        proposed_order=proposed_order,
                        copy_chunk_bytes=copy_chunk_bytes,
                    )
                )
                operation["executed"] = True
            elif operation.get("operation") == "hardlink":
                operation.update(_link_layer_file(source=source, destination=destination))
                operation["executed"] = True
        _write_json_atomic(output_layout, patched_layout)
    return {
        "schema": "largerlm.expert_order_repack_execute.v1",
        "executed": bool(execute),
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "source_expert_layout": str(expert_layout_path),
        "source_expert_layout_sha256": _sha256(expert_layout_path),
        "output_dir": str(output_dir),
        "output_layout": str(output_layout),
        "copy_chunk_bytes": copy_chunk_bytes,
        "selected_layer_count": len(selected),
        "total_layer_count": len(layout_layers),
        "repack_read_bytes": repack_bytes,
        "repack_write_bytes": repack_bytes,
        "write_guard": write_guard,
        "hardlink_referenced_bytes": hardlink_bytes,
        "layer_operations": layer_operations,
        "output_layout_sha256": _sha256(output_layout) if execute else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute or dry-run a bounded expert_order layer repack."
    )
    parser.add_argument("plan_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-chunk-mib", type=float, default=64.0)
    parser.add_argument("--max-repack-write-gib", type=float, default=8.0)
    parser.add_argument("--min-free-after-write-gib", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.copy_chunk_mib <= 0:
        raise SystemExit("--copy-chunk-mib must be positive")
    if args.max_repack_write_gib < 0:
        raise SystemExit("--max-repack-write-gib must be non-negative")
    if args.min_free_after_write_gib < 0:
        raise SystemExit("--min-free-after-write-gib must be non-negative")
    copy_chunk_bytes = int(args.copy_chunk_mib * 1024 * 1024)
    if copy_chunk_bytes <= 0:
        raise SystemExit("--copy-chunk-mib resolved to zero bytes")
    manifest = build_repack_manifest(
        plan_path=args.plan_json,
        output_dir=args.output_dir,
        execute=args.execute,
        copy_chunk_bytes=copy_chunk_bytes,
        max_repack_write_bytes=int(args.max_repack_write_gib * 1024**3),
        min_free_bytes_after_write=int(args.min_free_after_write_gib * 1024**3),
    )
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.write_manifest, manifest)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
