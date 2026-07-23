from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


GIB = 1024**3
M5_MAX_128G_SAFE_PIN_BUDGET_BYTES = 10 * GIB
M5_MAX_128G_ADAPTIVE_EXPERT_CACHE_BYTES = 0


class ExpertUsageError(RuntimeError):
    """Raised when expert telemetry or a residency plan is invalid."""


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _positive_int(value: object) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _token_result(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("token_result"), dict):
        return payload["token_result"]
    response = _as_mapping(payload.get("response"))
    largerlm = _as_mapping(response.get("largerlm"))
    if isinstance(largerlm.get("token_result"), dict):
        return largerlm["token_result"]
    return payload


def _expert_list(value: object, *, label: str) -> list[int]:
    experts: list[int] = []
    for item in _as_list(value):
        expert = _nonnegative_int(item)
        if expert is None:
            raise ExpertUsageError(f"{label} entries must be non-negative integers")
        experts.append(expert)
    return experts


def _add_route(
    counts: dict[int, Counter[int]],
    *,
    layer: int,
    experts: Iterable[int],
    count: int,
) -> None:
    layer_counts = counts.setdefault(layer, Counter())
    for expert in experts:
        layer_counts[expert] += count


def _extract_hotspots(
    payload: dict[str, Any],
    counts: dict[int, Counter[int]],
) -> int:
    actual = _as_mapping(_token_result(payload).get("prefill_actual_read_time"))
    if not actual:
        return 0
    seen: set[tuple[int | None, int, int | None]] = set()
    observations = 0
    for field in ("expert_stage_copy_hotspots", "expert_stage_range_hotspots"):
        for raw_row in _as_list(actual.get(field)):
            row = _as_mapping(raw_row)
            layer = _nonnegative_int(row.get("layer"))
            if layer is None:
                continue
            experts = _expert_list(
                row.get("selected_experts"),
                label=f"{field} selected_experts",
            )
            if not experts:
                continue
            key = (
                _nonnegative_int(row.get("chunk_index")),
                layer,
                _nonnegative_int(row.get("tile_index")),
            )
            if key in seen:
                continue
            seen.add(key)
            _add_route(counts, layer=layer, experts=experts, count=1)
            observations += len(experts)
    return observations


def _extract_event(
    payload: dict[str, Any],
    counts: dict[int, Counter[int]],
    *,
    default_layer: int | None,
) -> int:
    expert_field = None
    if "selected_experts" in payload:
        expert_field = "selected_experts"
    elif "experts" in payload:
        expert_field = "experts"
    elif "expert" in payload:
        expert_field = "expert"
    if expert_field is None:
        return 0

    layer = _nonnegative_int(payload.get("layer"))
    if layer is None:
        layer = default_layer
    if layer is None:
        raise ExpertUsageError(
            "route telemetry has no layer; pass default_layer for a per-layer router file"
        )

    if expert_field == "expert":
        expert = _nonnegative_int(payload.get("expert"))
        if expert is None:
            raise ExpertUsageError("expert must be a non-negative integer")
        experts = [expert]
    else:
        experts = _expert_list(payload.get(expert_field), label=expert_field)
    count = _positive_int(payload.get("count")) if "count" in payload else 1
    if count is None:
        raise ExpertUsageError("route count must be a positive integer")
    _add_route(counts, layer=layer, experts=experts, count=count)
    return len(experts) * count


def _extract_json_payload(
    payload: object,
    counts: dict[int, Counter[int]],
    *,
    default_layer: int | None,
    source_kinds: set[str],
) -> int:
    if isinstance(payload, list):
        return sum(
            _extract_json_payload(
                item,
                counts,
                default_layer=default_layer,
                source_kinds=source_kinds,
            )
            for item in payload
        )
    if not isinstance(payload, dict):
        return 0

    hotspot_count = _extract_hotspots(payload, counts)
    if hotspot_count:
        source_kinds.add("largerlm_prefill_hotspots")
        return hotspot_count

    event_count = _extract_event(payload, counts, default_layer=default_layer)
    if event_count:
        source_kinds.add("route_events")
        return event_count

    total = 0
    for field in (
        "events",
        "routes",
        "records",
        "expert_routes",
        "steps",
        "layers",
        "prompt_prefill",
        "probe_generate",
        "probe_decode_layers",
    ):
        if field in payload:
            total += _extract_json_payload(
                payload[field],
                counts,
                default_layer=default_layer,
                source_kinds=source_kinds,
            )
    return total


def _parse_colibri_line(
    line: str,
    counts: dict[int, Counter[int]],
    *,
    path: Path,
    line_number: int,
) -> int:
    fields = line.split()
    if len(fields) != 3:
        raise ExpertUsageError(
            f"{path}:{line_number} must be JSON or 'layer expert count'"
        )
    try:
        layer, expert, count = (int(field) for field in fields)
    except ValueError as exc:
        raise ExpertUsageError(
            f"{path}:{line_number} has a non-integer Colibri usage field"
        ) from exc
    if layer < 0 or expert < 0 or count <= 0:
        raise ExpertUsageError(
            f"{path}:{line_number} requires layer/expert >= 0 and count > 0"
        )
    _add_route(counts, layer=layer, experts=[expert], count=count)
    return count


def build_usage_profile(
    source_paths: Iterable[Path],
    *,
    default_layer: int | None = None,
) -> dict[str, Any]:
    paths = [Path(path) for path in source_paths]
    if not paths:
        raise ExpertUsageError("at least one usage source is required")
    if default_layer is not None and default_layer < 0:
        raise ExpertUsageError("default_layer must be non-negative")

    counts: dict[int, Counter[int]] = {}
    source_kinds: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise ExpertUsageError(f"usage source does not exist: {path}")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            for line_number, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    _parse_colibri_line(
                        line,
                        counts,
                        path=path,
                        line_number=line_number,
                    )
                    source_kinds.add("colibri_usage")
                else:
                    _extract_json_payload(
                        payload,
                        counts,
                        default_layer=default_layer,
                        source_kinds=source_kinds,
                    )
        else:
            _extract_json_payload(
                payload,
                counts,
                default_layer=default_layer,
                source_kinds=source_kinds,
            )

    layers = []
    total_selections = 0
    distinct_experts = 0
    for layer_id in sorted(counts):
        layer_counts = counts[layer_id]
        layer_total = sum(layer_counts.values())
        total_selections += layer_total
        distinct_experts += len(layer_counts)
        experts = [
            {
                "expert": expert,
                "count": count,
                "fraction": count / layer_total if layer_total else 0.0,
            }
            for expert, count in sorted(
                layer_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        layers.append(
            {
                "layer": layer_id,
                "total_selections": layer_total,
                "observed_expert_count": len(experts),
                "experts": experts,
            }
        )

    if not total_selections:
        raise ExpertUsageError("usage sources contained no expert selections")
    return {
        "schema": "largerlm.expert_usage_profile.v1",
        "source_paths": [str(path) for path in paths],
        "source_kinds": sorted(source_kinds),
        "default_layer": default_layer,
        "count_semantics": "expert_selection_observations",
        "complete_route_telemetry": "largerlm_prefill_hotspots" not in source_kinds,
        "total_selections": total_selections,
        "distinct_expert_count": distinct_experts,
        "layer_count": len(layers),
        "layers": layers,
    }


def load_expert_layout(path: Path) -> dict[int, dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpertUsageError(f"cannot read expert layout {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExpertUsageError("expert layout must contain a JSON object")
    layers: dict[int, dict[str, Any]] = {}
    for raw_layer in _as_list(payload.get("layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        num_experts = _positive_int(layer.get("num_experts"))
        slot_bytes = _positive_int(layer.get("expert_slot_bytes"))
        if layer_id is None or num_experts is None or slot_bytes is None:
            raise ExpertUsageError(
                "each layout layer needs non-negative layer and positive "
                "num_experts/expert_slot_bytes"
            )
        if layer_id in layers:
            raise ExpertUsageError(f"duplicate expert layout layer {layer_id}")
        layers[layer_id] = {
            "layer": layer_id,
            "num_experts": num_experts,
            "expert_slot_bytes": slot_bytes,
            "layer_file": layer.get("layer_file"),
        }
    if not layers:
        raise ExpertUsageError("expert layout contains no usable layers")
    return layers


def build_pin_plan(
    profile: dict[str, Any],
    *,
    expert_layout_path: Path,
    max_pin_bytes: int,
    min_count: int = 1,
    max_experts_per_layer: int | None = None,
    target_profile: str | None = None,
) -> dict[str, Any]:
    if profile.get("schema") != "largerlm.expert_usage_profile.v1":
        raise ExpertUsageError("unsupported expert usage profile schema")
    if max_pin_bytes <= 0:
        raise ExpertUsageError("max_pin_bytes must be positive")
    if min_count <= 0:
        raise ExpertUsageError("min_count must be positive")
    if max_experts_per_layer is not None and max_experts_per_layer <= 0:
        raise ExpertUsageError("max_experts_per_layer must be positive")

    layout = load_expert_layout(expert_layout_path)
    candidates = []
    skipped_counts: Counter[str] = Counter()
    total_profile_selections = _nonnegative_int(profile.get("total_selections")) or 0
    for raw_layer in _as_list(profile.get("layers")):
        profile_layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(profile_layer.get("layer"))
        if layer_id is None or layer_id not in layout:
            skipped_counts["layer_missing_from_layout"] += len(
                _as_list(profile_layer.get("experts"))
            )
            continue
        layer_layout = layout[layer_id]
        for raw_expert in _as_list(profile_layer.get("experts")):
            expert_row = _as_mapping(raw_expert)
            expert = _nonnegative_int(expert_row.get("expert"))
            count = _positive_int(expert_row.get("count"))
            if expert is None or count is None:
                skipped_counts["invalid_profile_entry"] += 1
                continue
            if expert >= layer_layout["num_experts"]:
                skipped_counts["expert_out_of_range"] += 1
                continue
            if count < min_count:
                skipped_counts["below_min_count"] += 1
                continue
            slot_bytes = layer_layout["expert_slot_bytes"]
            candidates.append(
                {
                    "layer": layer_id,
                    "expert": expert,
                    "count": count,
                    "expert_slot_bytes": slot_bytes,
                    "value_per_gib": count * GIB / slot_bytes,
                }
            )

    candidates.sort(
        key=lambda item: (
            -item["value_per_gib"],
            -item["count"],
            item["layer"],
            item["expert"],
        )
    )
    selected = []
    selected_by_layer: Counter[int] = Counter()
    selected_bytes = 0
    covered_selections = 0
    for candidate in candidates:
        layer_id = candidate["layer"]
        if (
            max_experts_per_layer is not None
            and selected_by_layer[layer_id] >= max_experts_per_layer
        ):
            skipped_counts["per_layer_cap"] += 1
            continue
        if selected_bytes + candidate["expert_slot_bytes"] > max_pin_bytes:
            skipped_counts["pin_budget"] += 1
            continue
        selected_bytes += candidate["expert_slot_bytes"]
        covered_selections += candidate["count"]
        selected_by_layer[layer_id] += 1
        selected.append(
            {
                **candidate,
                "cumulative_pin_bytes": selected_bytes,
            }
        )

    selected_layers = []
    for layer_id in sorted(selected_by_layer):
        layer_experts = [row for row in selected if row["layer"] == layer_id]
        selected_layers.append(
            {
                "layer": layer_id,
                "selected_expert_count": len(layer_experts),
                "selected_bytes": sum(
                    row["expert_slot_bytes"] for row in layer_experts
                ),
                "covered_profile_selections": sum(row["count"] for row in layer_experts),
                "experts": [row["expert"] for row in layer_experts],
            }
        )

    plan = {
        "schema": "largerlm.expert_pin_plan.v1",
        "quality_preserving": True,
        "changes_routing": False,
        "planner_only": False,
        "runtime_consumable": True,
        "runtime_cli_flag": "--expert-pin-plan",
        "target_profile": target_profile,
        "expert_layout_path": str(expert_layout_path),
        "max_pin_bytes": max_pin_bytes,
        "selected_bytes": selected_bytes,
        "budget_utilization": selected_bytes / max_pin_bytes,
        "selection_count": len(selected),
        "min_count": min_count,
        "max_experts_per_layer": max_experts_per_layer,
        "total_profile_selections": total_profile_selections,
        "covered_profile_selections": covered_selections,
        "profile_hit_fraction": (
            covered_selections / total_profile_selections
            if total_profile_selections
            else 0.0
        ),
        "selection_policy": "greedy_count_per_expert_byte",
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "selected_experts": selected,
        "layers": selected_layers,
    }
    if target_profile == "m5-max-128g-safe":
        plan["memory_envelope"] = {
            "unified_memory_gib": 128,
            "default_expert_cache_policy": "os_page_cache",
            "experimental_upper_bound": False,
            "hard_pinned_expert_budget_gib": max_pin_bytes / GIB,
            "adaptive_evictable_expert_cache_gib": (
                M5_MAX_128G_ADAPTIVE_EXPERT_CACHE_BYTES / GIB
            ),
            "maximum_expert_resident_gib": (
                (max_pin_bytes + M5_MAX_128G_ADAPTIVE_EXPERT_CACHE_BYTES) / GIB
            ),
            "runtime_live_cap_gib": 16,
            "runtime_live_cap_includes_expert_cache": True,
            "minimum_free_unified_memory_gib": 24,
            "remaining_for_os_dense_and_other_resident_gib": (
                128 - 16 - 24
            ),
            "unallocated_headroom_at_maximum_expert_residency_gib": (
                128 - 16 - 24
            ),
            "adaptive_cache_must_shrink_before_minimum_free_guard": False,
            "metal_recommended_working_set_bound": True,
            "requires_runtime_rss_guard": True,
        }
    return plan


def format_colibri_usage(profile: dict[str, Any]) -> str:
    if profile.get("schema") != "largerlm.expert_usage_profile.v1":
        raise ExpertUsageError("unsupported expert usage profile schema")
    rows: list[tuple[int, int, int]] = []
    for raw_layer in _as_list(profile.get("layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        if layer_id is None:
            continue
        for raw_expert in _as_list(layer.get("experts")):
            expert = _as_mapping(raw_expert)
            expert_id = _nonnegative_int(expert.get("expert"))
            count = _positive_int(expert.get("count"))
            if expert_id is not None and count is not None:
                rows.append((layer_id, expert_id, count))
    rows.sort(key=lambda row: (row[0], -row[2], row[1]))
    return "".join(f"{layer} {expert} {count}\n" for layer, expert, count in rows)
