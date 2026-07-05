#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


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
            raise SystemExit(f"{label} expert_order entries must be non-negative integers")
        order.append(expert)
    if len(order) != num_experts:
        raise SystemExit(f"{label} expert_order length must match num_experts")
    if sorted(order) != list(range(num_experts)):
        raise SystemExit(f"{label} expert_order must be a permutation")
    return order


def _layer_layouts(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    layers: dict[int, dict[str, Any]] = {}
    for raw_layer in _as_list(layout.get("layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        if layer_id is not None:
            layers[layer_id] = layer
    return layers


def _current_order(layer: dict[str, Any], *, num_experts: int) -> list[int]:
    if "expert_order" not in layer:
        return list(range(num_experts))
    return _expert_order(
        layer.get("expert_order"),
        num_experts=num_experts,
        label=f"layer {layer.get('layer')}",
    )


def _candidate_rows(
    analysis: dict[str, Any],
    *,
    max_layers: int,
    min_range_reduction: int,
    min_reduction_fraction: float,
    min_sample_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for raw in _as_list(analysis.get("coactivation_candidates")):
        candidate = _as_mapping(raw)
        reduction = _nonnegative_int(candidate.get("total_range_reduction"))
        fraction = _finite_float(candidate.get("total_range_reduction_fraction"))
        sample_count = _nonnegative_int(candidate.get("sample_count")) or 0
        if reduction is None or reduction < min_range_reduction:
            continue
        if fraction is None or fraction < min_reduction_fraction:
            continue
        if sample_count < min_sample_count:
            continue
        rows.append(candidate)
        if len(rows) >= max_layers:
            break
    return rows


def build_plan(
    *,
    analysis_path: Path,
    expert_layout_path: Path,
    max_layers: int,
    min_range_reduction: int,
    min_reduction_fraction: float,
    min_sample_count: int = 0,
) -> dict[str, Any]:
    analysis = _load_json(analysis_path)
    layout = _load_json(expert_layout_path)
    layer_by_id = _layer_layouts(layout)
    selected_layers = []
    total_repack_bytes = 0
    max_layer_file_bytes = 0
    skipped_layers = []
    for candidate in _candidate_rows(
        analysis,
        max_layers=max_layers,
        min_range_reduction=min_range_reduction,
        min_reduction_fraction=min_reduction_fraction,
        min_sample_count=min_sample_count,
    ):
        layer_id = _nonnegative_int(candidate.get("layer"))
        if layer_id is None or layer_id not in layer_by_id:
            skipped_layers.append(
                {
                    "layer": layer_id,
                    "reason": "candidate layer missing from expert layout",
                }
            )
            continue
        layer = layer_by_id[layer_id]
        num_experts = _positive_int(layer.get("num_experts"))
        slot_bytes = _positive_int(layer.get("expert_slot_bytes"))
        layer_file = layer.get("layer_file")
        if num_experts is None or slot_bytes is None or not isinstance(layer_file, str):
            skipped_layers.append(
                {
                    "layer": layer_id,
                    "reason": "expert layout layer missing num_experts, slot bytes, or file",
                }
            )
            continue
        proposed_order = _expert_order(
            candidate.get("expert_order"),
            num_experts=num_experts,
            label=f"candidate layer {layer_id}",
        )
        current_order = _current_order(layer, num_experts=num_experts)
        if proposed_order == current_order:
            skipped_layers.append(
                {
                    "layer": layer_id,
                    "reason": "candidate expert_order matches current layout",
                }
            )
            continue
        layer_file_path = expert_layout_path.parent / layer_file
        expected_layer_file_bytes = num_experts * slot_bytes
        layer_file_bytes = (
            layer_file_path.stat().st_size if layer_file_path.exists() else None
        )
        if layer_file_bytes is not None and layer_file_bytes != expected_layer_file_bytes:
            skipped_layers.append(
                {
                    "layer": layer_id,
                    "reason": "layer file size does not match layout",
                    "layer_file": str(layer_file_path),
                    "expected_layer_file_bytes": expected_layer_file_bytes,
                    "layer_file_bytes": layer_file_bytes,
                }
            )
            continue
        total_repack_bytes += expected_layer_file_bytes
        max_layer_file_bytes = max(max_layer_file_bytes, expected_layer_file_bytes)
        selected_layers.append(
            {
                "layer": layer_id,
                "layer_file": layer_file,
                "layer_file_path": str(layer_file_path),
                "layer_file_exists": layer_file_path.exists(),
                "num_experts": num_experts,
                "expert_slot_bytes": slot_bytes,
                "expected_layer_file_bytes": expected_layer_file_bytes,
                "current_expert_order": current_order,
                "proposed_expert_order": proposed_order,
                "layout_patch": {
                    "layer": layer_id,
                    "expert_order": proposed_order,
                },
                "sample_count": candidate.get("sample_count"),
                "observed_expert_count": candidate.get("observed_expert_count"),
                "total_current_range_count": candidate.get("total_current_range_count"),
                "total_candidate_range_count": candidate.get(
                    "total_candidate_range_count"
                ),
                "total_range_reduction": candidate.get("total_range_reduction"),
                "total_range_reduction_fraction": candidate.get(
                    "total_range_reduction_fraction"
                ),
                "total_copy_elapsed_seconds": candidate.get(
                    "total_copy_elapsed_seconds"
                ),
                "simulated_rows": candidate.get("simulated_rows", []),
            }
        )
    return {
        "schema": "largerlm.expert_order_repack_plan.v1",
        "dry_run": True,
        "safe_to_apply_to_existing_layout": False,
        "requires_layer_file_repack": True,
        "notes": [
            "This manifest does not copy expert layer files.",
            "Do not add expert_order to an existing identity-packed layout unless "
            "the matching layer file has been physically repacked in the same order.",
            "Router outputs stay logical expert ids; expert_order maps logical ids "
            "to physical slots.",
        ],
        "source_analysis": str(analysis_path),
        "source_analysis_sha256": _sha256(analysis_path),
        "source_analysis_schema": analysis.get("schema"),
        "source_result_count": analysis.get("source_result_count"),
        "min_sample_count": min_sample_count,
        "expert_layout": str(expert_layout_path),
        "expert_layout_sha256": _sha256(expert_layout_path),
        "expert_layout_model_type": layout.get("model_type"),
        "expert_layout_quantization": layout.get("quantization"),
        "selected_layer_count": len(selected_layers),
        "skipped_layer_count": len(skipped_layers),
        "total_repack_read_bytes": total_repack_bytes,
        "total_repack_write_bytes": total_repack_bytes,
        "total_repack_io_bytes": 2 * total_repack_bytes,
        "max_layer_file_bytes": max_layer_file_bytes,
        "selected_layers": selected_layers,
        "skipped_layers": skipped_layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a dry-run expert_order repack/indirection manifest."
    )
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("expert_layout", type=Path)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--min-range-reduction", type=int, default=1)
    parser.add_argument("--min-reduction-fraction", type=float, default=0.0)
    parser.add_argument(
        "--min-sample-count",
        type=int,
        default=0,
        help=(
            "Only select candidates observed in at least this many hotspot "
            "samples. Defaults to 0 for backward-compatible planning."
        ),
    )
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.max_layers <= 0:
        raise SystemExit("--max-layers must be positive")
    if args.min_range_reduction < 0:
        raise SystemExit("--min-range-reduction must be non-negative")
    if args.min_reduction_fraction < 0.0 or args.min_reduction_fraction > 1.0:
        raise SystemExit("--min-reduction-fraction must be between 0 and 1")
    if args.min_sample_count < 0:
        raise SystemExit("--min-sample-count must be non-negative")
    plan = build_plan(
        analysis_path=args.analysis_json,
        expert_layout_path=args.expert_layout,
        max_layers=args.max_layers,
        min_range_reduction=args.min_range_reduction,
        min_reduction_fraction=args.min_reduction_fraction,
        min_sample_count=args.min_sample_count,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(text + "\n", encoding="utf-8")
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
