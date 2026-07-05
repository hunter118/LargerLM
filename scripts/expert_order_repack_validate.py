#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.expert_io import plan_expert_io


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _nonnegative_int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


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


def _layer_by_id(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for raw_layer in _as_list(layout.get("layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        if layer_id is not None:
            result[layer_id] = layer
    return result


def _expert_order(layer: dict[str, Any]) -> list[int] | None:
    order = []
    raw_order = layer.get("expert_order")
    if raw_order is None:
        return None
    for item in _as_list(raw_order):
        expert = _nonnegative_int(item)
        if expert is None:
            return None
        order.append(expert)
    return order


def _sample_slots(order: list[int], *, sample_count: int) -> list[tuple[int, int]]:
    slots = [(slot, expert) for slot, expert in enumerate(order[:sample_count])]
    if order:
        last = (len(order) - 1, order[-1])
        if last not in slots:
            slots.append(last)
    return slots


def _slot_by_logical_expert(layer: dict[str, Any], *, num_experts: int) -> dict[int, int] | None:
    order = _expert_order(layer)
    if order is None:
        order = list(range(num_experts))
    if len(order) != num_experts or sorted(order) != list(range(num_experts)):
        return None
    return {expert: slot for slot, expert in enumerate(order)}


def _slot_head_digest(
    *,
    path: Path,
    slot_index: int,
    slot_bytes: int,
    sample_bytes: int,
) -> str:
    size = min(slot_bytes, sample_bytes)
    with path.open("rb") as f:
        data = os.pread(f.fileno(), size, slot_index * slot_bytes)
    if len(data) != size:
        raise SystemExit(f"short slot sample read from {path}")
    return hashlib.sha256(data).hexdigest()


def _validate_slots(
    *,
    source_layout_path: Path,
    repacked_layout_path: Path,
    sample_count: int,
    sample_bytes: int,
) -> list[dict[str, object]]:
    source_layout = _load_json(source_layout_path)
    repacked_layout = _load_json(repacked_layout_path)
    source_layers = _layer_by_id(source_layout)
    rows = []
    for layer_id, repacked_layer in _layer_by_id(repacked_layout).items():
        order = _expert_order(repacked_layer)
        if order is None:
            continue
        source_layer = source_layers.get(layer_id)
        if source_layer is None:
            rows.append({"layer": layer_id, "ok": False, "reason": "missing source layer"})
            continue
        num_experts = _nonnegative_int(repacked_layer.get("num_experts"))
        slot_bytes = _nonnegative_int(repacked_layer.get("expert_slot_bytes"))
        layer_file = repacked_layer.get("layer_file")
        source_file = source_layer.get("layer_file")
        if (
            num_experts is None
            or slot_bytes is None
            or not isinstance(layer_file, str)
            or not isinstance(source_file, str)
            or len(order) != num_experts
        ):
            rows.append({"layer": layer_id, "ok": False, "reason": "invalid layer metadata"})
            continue
        source_path = source_layout_path.parent / source_file
        repacked_path = repacked_layout_path.parent / layer_file
        source_slot_by_expert = _slot_by_logical_expert(
            source_layer,
            num_experts=num_experts,
        )
        if source_slot_by_expert is None:
            rows.append({"layer": layer_id, "ok": False, "reason": "invalid source expert_order"})
            continue
        checks = []
        ok = True
        for dst_slot, expert in _sample_slots(order, sample_count=sample_count):
            source_slot = source_slot_by_expert.get(expert)
            if source_slot is None:
                ok = False
                checks.append(
                    {
                        "destination_slot": dst_slot,
                        "expert": expert,
                        "source_slot": None,
                        "sample_bytes": min(slot_bytes, sample_bytes),
                        "match": False,
                    }
                )
                continue
            source_digest = _slot_head_digest(
                path=source_path,
                slot_index=source_slot,
                slot_bytes=slot_bytes,
                sample_bytes=sample_bytes,
            )
            repacked_digest = _slot_head_digest(
                path=repacked_path,
                slot_index=dst_slot,
                slot_bytes=slot_bytes,
                sample_bytes=sample_bytes,
            )
            match = source_digest == repacked_digest
            ok = ok and match
            checks.append(
                {
                    "destination_slot": dst_slot,
                    "expert": expert,
                    "source_slot": source_slot,
                    "sample_bytes": min(slot_bytes, sample_bytes),
                    "match": match,
                }
            )
        rows.append(
            {
                "layer": layer_id,
                "ok": ok,
                "sample_count": len(checks),
                "checks": checks,
            }
        )
    return rows


def _range_delta_rows(
    *,
    source_layout_path: Path,
    repacked_layout_path: Path,
    analysis_path: Path | None,
    align_bytes: int,
) -> list[dict[str, object]]:
    if analysis_path is None:
        return []
    analysis = _load_json(analysis_path)
    rows = []
    for raw_row in _as_list(analysis.get("rows")):
        row = _as_mapping(raw_row)
        layer = _nonnegative_int(row.get("layer"))
        experts = [
            expert
            for expert in (_nonnegative_int(item) for item in _as_list(row.get("selected_experts")))
            if expert is not None
        ]
        if layer is None or not experts:
            continue
        old_plan = plan_expert_io(
            source_layout_path,
            layer=layer,
            expert_ids=experts,
            merge_gap_bytes=0,
            align_bytes=align_bytes,
        )
        new_plan = plan_expert_io(
            repacked_layout_path,
            layer=layer,
            expert_ids=experts,
            merge_gap_bytes=0,
            align_bytes=align_bytes,
        )
        rows.append(
            {
                "layer": layer,
                "source_result": row.get("source_result"),
                "selected_expert_count": len(experts),
                "old_coalesced_range_count": old_plan.coalesced_range_count,
                "new_coalesced_range_count": new_plan.coalesced_range_count,
                "range_reduction": (
                    old_plan.coalesced_range_count - new_plan.coalesced_range_count
                ),
            }
        )
    return rows


def validate_repack(
    *,
    source_layout_path: Path,
    repacked_layout_path: Path,
    analysis_path: Path | None,
    sample_count: int,
    sample_bytes: int,
    align_bytes: int = 4096,
) -> dict[str, Any]:
    slot_rows = _validate_slots(
        source_layout_path=source_layout_path,
        repacked_layout_path=repacked_layout_path,
        sample_count=sample_count,
        sample_bytes=sample_bytes,
    )
    range_rows = _range_delta_rows(
        source_layout_path=source_layout_path,
        repacked_layout_path=repacked_layout_path,
        analysis_path=analysis_path,
        align_bytes=align_bytes,
    )
    total_old = sum(int(row["old_coalesced_range_count"]) for row in range_rows)
    total_new = sum(int(row["new_coalesced_range_count"]) for row in range_rows)
    return {
        "schema": "largerlm.expert_order_repack_validate.v1",
        "source_layout": str(source_layout_path),
        "source_layout_sha256": _sha256(source_layout_path),
        "repacked_layout": str(repacked_layout_path),
        "repacked_layout_sha256": _sha256(repacked_layout_path),
        "analysis": str(analysis_path) if analysis_path is not None else None,
        "analysis_sha256": _sha256(analysis_path) if analysis_path is not None else None,
        "sample_count": sample_count,
        "sample_bytes": sample_bytes,
        "align_bytes": align_bytes,
        "slot_validation": slot_rows,
        "slot_validation_ok": all(row.get("ok") is True for row in slot_rows),
        "range_delta_rows": range_rows,
        "old_coalesced_range_count": total_old,
        "new_coalesced_range_count": total_new,
        "range_reduction": total_old - total_new,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate expert_order repack slot samples and range deltas."
    )
    parser.add_argument("source_layout", type=Path)
    parser.add_argument("repacked_layout", type=Path)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-bytes", type=int, default=4096)
    parser.add_argument("--align-bytes", type=int, default=4096)
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")
    if args.sample_bytes <= 0:
        raise SystemExit("--sample-bytes must be positive")
    if args.align_bytes <= 0:
        raise SystemExit("--align-bytes must be positive")
    result = validate_repack(
        source_layout_path=args.source_layout,
        repacked_layout_path=args.repacked_layout,
        analysis_path=args.analysis_json,
        sample_count=args.sample_count,
        sample_bytes=args.sample_bytes,
        align_bytes=args.align_bytes,
    )
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.write_json, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
