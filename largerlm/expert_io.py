from __future__ import annotations

import fcntl
import json
import math
import os
import sys
import struct
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .safety import disk_budget


class ExpertIOPlanError(RuntimeError):
    """Raised when expert slot I/O cannot be planned safely."""


@dataclass(frozen=True)
class ExpertReadRange:
    layer: int
    layer_file: str
    experts: tuple[int, ...]
    offset: int
    length: int
    aligned_offset: int
    aligned_length: int
    requested_bytes: int
    waste_bytes: int


@dataclass(frozen=True)
class ExpertIOPlan:
    expert_layout_path: Path
    layer: int
    layer_file: str
    layer_file_path: Path
    num_experts: int
    expert_slot_bytes: int
    selected_experts: tuple[int, ...]
    merge_gap_bytes: int
    align_bytes: int
    requested_bytes: int
    raw_span_bytes: int
    read_bytes: int
    waste_bytes: int
    read_amplification: float
    raw_range_count: int
    coalesced_range_count: int
    coalesced_range_savings: int
    ranges: tuple[ExpertReadRange, ...]
    expert_physical_slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class RouterTokenRoute:
    token_index: int
    router_json_path: Path
    experts: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class ExpertTokenAssignment:
    expert: int
    tokens: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class StaticExpertTokenSlot:
    expert: int
    slot: int
    token_index: int
    weight: float


@dataclass(frozen=True)
class StaticExpertOverflow:
    expert: int
    overflow_index: int
    token_index: int
    weight: float


@dataclass(frozen=True)
class StaticExpertUsage:
    expert: int
    assigned_tokens: int
    used_slots: int
    overflow_assignments: int


@dataclass(frozen=True)
class StaticExpertCapacityPlan:
    batch_tokens: int
    selected_experts: tuple[int, ...]
    capacity_per_expert: int
    total_assignments: int
    total_capacity_slots: int
    used_slots: int
    utilization: float
    overflow_assignments: int
    max_tokens_per_expert: int
    requires_overflow_path: bool
    usages: tuple[StaticExpertUsage, ...]
    slots: tuple[StaticExpertTokenSlot, ...]
    overflow: tuple[StaticExpertOverflow, ...]


@dataclass(frozen=True)
class StaticExpertCapacityBinaryReport:
    path: Path
    bytes_written: int
    version: int
    expert_count: int
    capacity_per_expert: int
    slot_records: int
    overflow_records: int


@dataclass(frozen=True)
class StaticExpertCapacityBinaryValidation:
    path: Path
    bytes_read: int
    version: int
    batch_tokens: int
    expert_count: int
    capacity_per_expert: int
    total_assignments: int
    used_slots: int
    overflow_records: int
    slot_records: int
    active_slot_records: int
    inactive_slot_records: int


@dataclass(frozen=True)
class BatchExpertIOPlan:
    expert_layout_path: Path
    router_json_dir: Path
    router_json_glob: str
    layer: int
    batch_tokens: int
    selected_experts: tuple[int, ...]
    total_assignments: int
    serial_read_bytes: int
    unique_requested_bytes: int
    planned_read_bytes: int
    coalesced_savings_bytes: int
    raw_range_count: int
    coalesced_range_count: int
    coalesced_range_savings: int
    assignment_read_amplification: float
    unique_read_amplification: float
    ssd_read_gib_per_second: float
    planned_read_seconds: float | None
    io_plan: ExpertIOPlan
    token_routes: tuple[RouterTokenRoute, ...]
    expert_tokens: tuple[ExpertTokenAssignment, ...]


@dataclass(frozen=True)
class BatchExpertIOTile:
    tile_index: int
    selected_experts: tuple[int, ...]
    active_token_count: int
    total_assignments: int
    unique_requested_bytes: int
    planned_read_bytes: int
    compact_stage_bytes: int
    raw_range_count: int
    coalesced_range_count: int
    planned_read_seconds: float | None


@dataclass(frozen=True)
class BatchExpertIOTilingPlan:
    batch_plan: BatchExpertIOPlan
    max_stage_bytes: int
    max_compact_stage_bytes: int
    tile_count: int
    total_tile_assignments: int
    total_tile_planned_read_bytes: int
    max_tile_planned_read_bytes: int
    max_tile_compact_stage_bytes: int
    tiles: tuple[BatchExpertIOTile, ...]


@dataclass(frozen=True)
class BatchExpertStageIOSummary:
    batch_tokens: int
    selected_expert_count: int
    total_assignments: int
    serial_read_bytes: int
    unique_requested_bytes: int
    planned_read_bytes: int
    staged_bytes: int
    coalesced_savings_bytes: int
    waste_bytes: int
    raw_range_count: int
    coalesced_range_count: int
    coalesced_range_savings: int
    slot_count: int
    max_stage_bytes: int
    stage_budget_utilization: float
    assignment_read_amplification: float
    unique_read_amplification: float
    staged_unique_read_amplification: float
    coalesced_savings_ratio: float
    waste_ratio: float
    ssd_read_gib_per_second: float
    planned_read_seconds: float | None
    max_read_seconds: float
    read_seconds_ok: bool | None
    copy_elapsed_seconds: float | None = None
    copy_throughput_gib_per_second: float | None = None
    copy_seconds_ok: bool | None = None
    copy_read_calls: int = 0
    copy_write_calls: int = 0
    copy_average_read_bytes: float | None = None
    copy_average_write_bytes: float | None = None
    copy_read_call_counterfactuals_by_chunk_mib: dict[str, int] | None = None
    max_raw_ranges: int = 0
    raw_range_count_ok: bool | None = None
    max_coalesced_ranges: int = 0
    coalesced_range_count_ok: bool | None = None


@dataclass(frozen=True)
class ReadAdviceStats:
    supported: bool
    attempted_ranges: int
    calls: int
    advised_bytes: int
    error: str | None


@dataclass(frozen=True)
class CopyRangeStats:
    read_calls: int
    write_calls: int
    copied_bytes: int


@dataclass(frozen=True)
class StagedExpertRange:
    range_index: int
    experts: tuple[int, ...]
    source_offset: int
    source_length: int
    stage_offset: int
    stage_length: int


@dataclass(frozen=True)
class StagedExpertSlot:
    expert: int
    range_index: int
    source_offset: int
    stage_offset: int
    length: int


@dataclass(frozen=True)
class BatchExpertStageResult:
    expert_layout_path: Path
    router_json_dir: Path
    stage_file_path: Path
    manifest_path: Path | None
    layer: int
    batch_tokens: int
    selected_experts: tuple[int, ...]
    expert_slot_bytes: int
    planned_read_bytes: int
    staged_bytes: int
    max_stage_bytes: int
    copy_chunk_bytes: int
    read_advice: ReadAdviceStats
    io_summary: BatchExpertStageIOSummary
    ranges: tuple[StagedExpertRange, ...]
    slots: tuple[StagedExpertSlot, ...]
    batch_plan: BatchExpertIOPlan
    copy_elapsed_seconds: float | None = None
    copy_throughput_gib_per_second: float | None = None


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExpertIOPlanError(f"failed to read expert layout {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExpertIOPlanError(f"failed to parse expert layout {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExpertIOPlanError("expert layout must be a JSON object")
    return payload


def _find_layer(layout: dict[str, Any], layer_id: int) -> dict[str, Any]:
    layers = layout.get("layers")
    if not isinstance(layers, list):
        raise ExpertIOPlanError("expert layout missing layers array")
    for item in layers:
        if (
            isinstance(item, dict)
            and type(item.get("layer")) is int
            and item.get("layer") == layer_id
        ):
            return item
    raise ExpertIOPlanError(f"layer {layer_id} not found in expert layout")


def _align_down(value: int, alignment: int) -> int:
    return (value // alignment) * alignment


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _numeric_limit(name: str, value: float | int) -> float:
    if isinstance(value, bool):
        raise ExpertIOPlanError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExpertIOPlanError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ExpertIOPlanError(f"{name} must be finite")
    return parsed


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ExpertIOPlanError(f"{label} must be an integer")
    return int(value)


def _positive_integer_value(value: object, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed <= 0:
        raise ExpertIOPlanError(f"{label} must be positive")
    return parsed


def _nonnegative_integer_value(value: object, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed < 0:
        raise ExpertIOPlanError(f"{label} must be non-negative")
    return parsed


def _int_field(payload: dict[str, Any], field: str, label: str) -> int:
    return _require_int(payload.get(field), f"{label} {field}")


def _positive_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed <= 0:
        raise ExpertIOPlanError(f"{name} must be positive")
    return parsed


def _nonnegative_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed < 0:
        raise ExpertIOPlanError(f"{name} must be non-negative")
    return parsed


def _safe_layer_file_path(layout_path: Path, layer_file: str) -> Path:
    rel = Path(layer_file)
    if rel.is_absolute() or ".." in rel.parts:
        raise ExpertIOPlanError("layout layer_file must be a relative path inside layout directory")
    if not rel.parts:
        raise ExpertIOPlanError("layout layer_file must be non-empty")
    return layout_path.parent / rel


def _validated_experts(expert_ids: Iterable[int], *, num_experts: int) -> tuple[int, ...]:
    parsed: set[int] = set()
    for raw_expert in expert_ids:
        if type(raw_expert) is not int:
            raise ExpertIOPlanError("expert ids must be integers")
        parsed.add(raw_expert)
    experts = tuple(sorted(parsed))
    if not experts:
        raise ExpertIOPlanError("at least one expert id is required")
    for expert in experts:
        if expert < 0 or expert >= num_experts:
            raise ExpertIOPlanError(
                f"expert id {expert} is outside layer expert range 0..{num_experts - 1}"
            )
    return experts


def _expert_physical_slots(
    layer_layout: dict[str, Any],
    *,
    num_experts: int,
) -> tuple[int, ...]:
    raw_order = layer_layout.get("expert_order")
    if raw_order is None:
        return tuple(range(num_experts))
    if not isinstance(raw_order, list):
        raise ExpertIOPlanError("layout layer expert_order must be an array")
    if len(raw_order) != num_experts:
        raise ExpertIOPlanError(
            "layout layer expert_order length must match num_experts"
        )
    physical_by_expert = [-1] * num_experts
    for physical_slot, raw_expert in enumerate(raw_order):
        if type(raw_expert) is not int:
            raise ExpertIOPlanError("layout layer expert_order entries must be integers")
        if raw_expert < 0 or raw_expert >= num_experts:
            raise ExpertIOPlanError(
                "layout layer expert_order entry is outside layer expert range"
            )
        if physical_by_expert[raw_expert] != -1:
            raise ExpertIOPlanError("layout layer expert_order must not repeat experts")
        physical_by_expert[raw_expert] = physical_slot
    if any(slot < 0 for slot in physical_by_expert):
        raise ExpertIOPlanError("layout layer expert_order must cover every expert")
    return tuple(physical_by_expert)


def _range_from_group(
    *,
    layer: int,
    layer_file: str,
    experts: list[int],
    offset: int,
    end: int,
    slot_bytes: int,
    layer_file_bytes: int,
    align_bytes: int,
) -> ExpertReadRange:
    aligned_offset = _align_down(offset, align_bytes)
    aligned_end = min(_align_up(end, align_bytes), layer_file_bytes)
    requested = len(experts) * slot_bytes
    aligned_length = aligned_end - aligned_offset
    return ExpertReadRange(
        layer=layer,
        layer_file=layer_file,
        experts=tuple(experts),
        offset=offset,
        length=end - offset,
        aligned_offset=aligned_offset,
        aligned_length=aligned_length,
        requested_bytes=requested,
        waste_bytes=aligned_length - requested,
    )


def _merge_aligned_ranges(ranges: list[ExpertReadRange], *, slot_bytes: int) -> list[ExpertReadRange]:
    if not ranges:
        return []
    merged: list[ExpertReadRange] = [ranges[0]]
    for item in ranges[1:]:
        prev = merged[-1]
        prev_end = prev.aligned_offset + prev.aligned_length
        item_end = item.aligned_offset + item.aligned_length
        if item.aligned_offset > prev_end:
            merged.append(item)
            continue

        experts = tuple(sorted(prev.experts + item.experts))
        offset = min(prev.offset, item.offset)
        end = max(prev.offset + prev.length, item.offset + item.length)
        aligned_offset = min(prev.aligned_offset, item.aligned_offset)
        aligned_end = max(prev_end, item_end)
        requested = len(experts) * slot_bytes
        aligned_length = aligned_end - aligned_offset
        merged[-1] = ExpertReadRange(
            layer=prev.layer,
            layer_file=prev.layer_file,
            experts=experts,
            offset=offset,
            length=end - offset,
            aligned_offset=aligned_offset,
            aligned_length=aligned_length,
            requested_bytes=requested,
            waste_bytes=aligned_length - requested,
        )
    return merged


def plan_expert_io(
    expert_layout_path: str | Path,
    *,
    layer: int,
    expert_ids: Iterable[int],
    merge_gap_bytes: int = 0,
    align_bytes: int = 4096,
) -> ExpertIOPlan:
    layer = _require_int(layer, "layer")
    if layer < 0:
        raise ExpertIOPlanError("layer must be non-negative")
    merge_gap_bytes = _nonnegative_integer_value(
        merge_gap_bytes,
        "merge_gap_bytes",
    )
    align_bytes = _positive_integer_value(align_bytes, "align_bytes")

    layout_path = Path(expert_layout_path)
    layout = _load_json(layout_path)
    layer_layout = _find_layer(layout, layer)
    num_experts = _int_field(layer_layout, "num_experts", "layout layer")
    slot_bytes = _int_field(layer_layout, "expert_slot_bytes", "layout layer")
    layer_file = layer_layout.get("layer_file")
    if not isinstance(layer_file, str) or num_experts <= 0 or slot_bytes <= 0:
        raise ExpertIOPlanError(
            "layout layer is missing layer_file, num_experts, or expert_slot_bytes"
        )
    layer_file_path = _safe_layer_file_path(layout_path, layer_file)

    experts = _validated_experts(expert_ids, num_experts=num_experts)
    physical_slots = _expert_physical_slots(layer_layout, num_experts=num_experts)
    layer_file_bytes = num_experts * slot_bytes
    raw_ranges: list[ExpertReadRange] = []
    group_experts: list[int] = []
    group_offset = 0
    group_end = 0
    for expert in sorted(experts, key=lambda item: physical_slots[item]):
        offset = physical_slots[expert] * slot_bytes
        end = offset + slot_bytes
        if not group_experts:
            group_experts = [expert]
            group_offset = offset
            group_end = end
            continue
        gap = offset - group_end
        if gap <= merge_gap_bytes:
            group_experts.append(expert)
            group_end = end
            continue
        raw_ranges.append(
            _range_from_group(
                layer=layer,
                layer_file=layer_file,
                experts=group_experts,
                offset=group_offset,
                end=group_end,
                slot_bytes=slot_bytes,
                layer_file_bytes=layer_file_bytes,
                align_bytes=align_bytes,
            )
        )
        group_experts = [expert]
        group_offset = offset
        group_end = end

    raw_ranges.append(
        _range_from_group(
            layer=layer,
            layer_file=layer_file,
            experts=group_experts,
            offset=group_offset,
            end=group_end,
            slot_bytes=slot_bytes,
            layer_file_bytes=layer_file_bytes,
            align_bytes=align_bytes,
        )
    )
    ranges = tuple(_merge_aligned_ranges(raw_ranges, slot_bytes=slot_bytes))
    requested_bytes = len(experts) * slot_bytes
    raw_span_bytes = sum(item.length for item in ranges)
    read_bytes = sum(item.aligned_length for item in ranges)
    waste_bytes = read_bytes - requested_bytes
    raw_range_count = len(raw_ranges)
    coalesced_range_count = len(ranges)
    return ExpertIOPlan(
        expert_layout_path=layout_path,
        layer=layer,
        layer_file=layer_file,
        layer_file_path=layer_file_path,
        num_experts=num_experts,
        expert_slot_bytes=slot_bytes,
        selected_experts=experts,
        merge_gap_bytes=merge_gap_bytes,
        align_bytes=align_bytes,
        requested_bytes=requested_bytes,
        raw_span_bytes=raw_span_bytes,
        read_bytes=read_bytes,
        waste_bytes=waste_bytes,
        read_amplification=read_bytes / requested_bytes,
        raw_range_count=raw_range_count,
        coalesced_range_count=coalesced_range_count,
        coalesced_range_savings=raw_range_count - coalesced_range_count,
        ranges=ranges,
        expert_physical_slots=(
            () if physical_slots == tuple(range(num_experts)) else physical_slots
        ),
    )


def _router_json_paths(router_json_dir: Path, router_json_glob: str) -> tuple[Path, ...]:
    if not router_json_glob:
        raise ExpertIOPlanError("router_json_glob must be non-empty")
    try:
        paths = tuple(sorted(path for path in router_json_dir.glob(router_json_glob) if path.is_file()))
    except OSError as exc:
        raise ExpertIOPlanError(
            f"failed to list router JSON files in {router_json_dir}: {exc}"
        ) from exc
    if not paths:
        raise ExpertIOPlanError(
            f"no router JSON files matched {router_json_glob!r} in {router_json_dir}"
        )
    return paths


def _load_router_route(path: Path, *, token_index: int) -> RouterTokenRoute:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExpertIOPlanError(f"failed to read router JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExpertIOPlanError(f"failed to parse router JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExpertIOPlanError(f"router JSON {path} must be an object")
    raw_experts = payload.get("experts")
    raw_weights = payload.get("weights")
    if not isinstance(raw_experts, list) or not raw_experts:
        raise ExpertIOPlanError(f"router JSON {path} missing non-empty experts list")
    if not isinstance(raw_weights, list) or len(raw_weights) != len(raw_experts):
        raise ExpertIOPlanError(f"router JSON {path} weights must match experts length")
    experts: list[int] = []
    weights: list[float] = []
    seen: set[int] = set()
    for raw_expert, raw_weight in zip(raw_experts, raw_weights):
        if isinstance(raw_expert, bool) or not isinstance(raw_expert, int):
            raise ExpertIOPlanError(f"router JSON {path} contains non-integer expert id")
        if raw_expert in seen:
            raise ExpertIOPlanError(f"router JSON {path} repeats expert id {raw_expert}")
        seen.add(raw_expert)
        try:
            if isinstance(raw_weight, bool):
                raise TypeError
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ExpertIOPlanError(f"router JSON {path} contains non-numeric weight") from exc
        if not math.isfinite(weight):
            raise ExpertIOPlanError(f"router JSON {path} contains non-finite weight")
        experts.append(raw_expert)
        weights.append(weight)
    return RouterTokenRoute(
        token_index=token_index,
        router_json_path=path,
        experts=tuple(experts),
        weights=tuple(weights),
    )


def plan_batch_expert_io(
    expert_layout_path: str | Path,
    *,
    layer: int,
    router_json_dir: str | Path,
    router_json_glob: str = "*.router.json",
    merge_gap_bytes: int = 0,
    align_bytes: int = 4096,
    ssd_read_gib_per_second: float | int = 0.0,
) -> BatchExpertIOPlan:
    ssd_read_gib_per_second = _nonnegative_limit(
        "ssd_read_gib_per_second",
        ssd_read_gib_per_second,
    )
    router_dir = Path(router_json_dir)
    paths = _router_json_paths(router_dir, router_json_glob)
    token_routes = tuple(
        _load_router_route(path, token_index=index) for index, path in enumerate(paths)
    )
    expert_to_tokens: dict[int, list[int]] = {}
    expert_to_weights: dict[int, list[float]] = {}
    for route in token_routes:
        for expert, weight in zip(route.experts, route.weights):
            expert_to_tokens.setdefault(expert, []).append(route.token_index)
            expert_to_weights.setdefault(expert, []).append(weight)
    selected_experts = tuple(sorted(expert_to_tokens))
    io_plan = plan_expert_io(
        expert_layout_path,
        layer=layer,
        expert_ids=selected_experts,
        merge_gap_bytes=merge_gap_bytes,
        align_bytes=align_bytes,
    )
    total_assignments = sum(len(route.experts) for route in token_routes)
    serial_read_bytes = total_assignments * io_plan.expert_slot_bytes
    unique_requested_bytes = io_plan.requested_bytes
    planned_read_bytes = io_plan.read_bytes
    planned_read_seconds = (
        planned_read_bytes / (ssd_read_gib_per_second * 1024**3)
        if ssd_read_gib_per_second > 0
        else None
    )
    coalesced_savings = serial_read_bytes - planned_read_bytes
    expert_tokens = tuple(
        ExpertTokenAssignment(
            expert=expert,
            tokens=tuple(expert_to_tokens[expert]),
            weights=tuple(expert_to_weights[expert]),
        )
        for expert in selected_experts
    )
    return BatchExpertIOPlan(
        expert_layout_path=io_plan.expert_layout_path,
        router_json_dir=router_dir,
        router_json_glob=router_json_glob,
        layer=layer,
        batch_tokens=len(token_routes),
        selected_experts=selected_experts,
        total_assignments=total_assignments,
        serial_read_bytes=serial_read_bytes,
        unique_requested_bytes=unique_requested_bytes,
        planned_read_bytes=planned_read_bytes,
        coalesced_savings_bytes=coalesced_savings,
        raw_range_count=io_plan.raw_range_count,
        coalesced_range_count=io_plan.coalesced_range_count,
        coalesced_range_savings=io_plan.coalesced_range_savings,
        assignment_read_amplification=planned_read_bytes / serial_read_bytes,
        unique_read_amplification=io_plan.read_amplification,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
        planned_read_seconds=planned_read_seconds,
        io_plan=io_plan,
        token_routes=token_routes,
        expert_tokens=expert_tokens,
    )


def _tile_assignment_counts(
    batch_plan: BatchExpertIOPlan,
    selected_experts: tuple[int, ...],
) -> tuple[int, int]:
    expert_set = set(selected_experts)
    active_tokens = 0
    assignments = 0
    for route in batch_plan.token_routes:
        route_assignments = sum(1 for expert in route.experts if expert in expert_set)
        if route_assignments:
            active_tokens += 1
            assignments += route_assignments
    return active_tokens, assignments


def _batch_expert_io_tile(
    *,
    batch_plan: BatchExpertIOPlan,
    tile_index: int,
    selected_experts: tuple[int, ...],
    merge_gap_bytes: int,
    align_bytes: int,
) -> BatchExpertIOTile:
    io_plan = plan_expert_io(
        batch_plan.expert_layout_path,
        layer=batch_plan.layer,
        expert_ids=selected_experts,
        merge_gap_bytes=merge_gap_bytes,
        align_bytes=align_bytes,
    )
    active_tokens, assignments = _tile_assignment_counts(batch_plan, selected_experts)
    planned_read_seconds = (
        io_plan.read_bytes / (batch_plan.ssd_read_gib_per_second * 1024**3)
        if batch_plan.ssd_read_gib_per_second > 0
        else None
    )
    return BatchExpertIOTile(
        tile_index=tile_index,
        selected_experts=selected_experts,
        active_token_count=active_tokens,
        total_assignments=assignments,
        unique_requested_bytes=io_plan.requested_bytes,
        planned_read_bytes=io_plan.read_bytes,
        compact_stage_bytes=io_plan.requested_bytes,
        raw_range_count=io_plan.raw_range_count,
        coalesced_range_count=io_plan.coalesced_range_count,
        planned_read_seconds=planned_read_seconds,
    )


def plan_batch_expert_io_tiles(
    expert_layout_path: str | Path,
    *,
    layer: int,
    router_json_dir: str | Path,
    router_json_glob: str = "*.router.json",
    merge_gap_bytes: int = 0,
    align_bytes: int = 4096,
    max_stage_mib: float = 4096.0,
    max_compact_stage_mib: float = 4096.0,
    ssd_read_gib_per_second: float | int = 0.0,
) -> BatchExpertIOTilingPlan:
    max_stage_mib = _positive_limit("max_stage_mib", max_stage_mib)
    max_compact_stage_mib = _positive_limit(
        "max_compact_stage_mib",
        max_compact_stage_mib,
    )
    max_stage_bytes = int(max_stage_mib * 1024 * 1024)
    max_compact_stage_bytes = int(max_compact_stage_mib * 1024 * 1024)
    batch_plan = plan_batch_expert_io(
        expert_layout_path,
        layer=layer,
        router_json_dir=router_json_dir,
        router_json_glob=router_json_glob,
        merge_gap_bytes=merge_gap_bytes,
        align_bytes=align_bytes,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
    )
    tiles: list[BatchExpertIOTile] = []
    current: list[int] = []
    for expert in batch_plan.selected_experts:
        candidate = tuple(current + [expert])
        candidate_tile = _batch_expert_io_tile(
            batch_plan=batch_plan,
            tile_index=len(tiles),
            selected_experts=candidate,
            merge_gap_bytes=merge_gap_bytes,
            align_bytes=align_bytes,
        )
        if (
            candidate_tile.planned_read_bytes <= max_stage_bytes
            and candidate_tile.compact_stage_bytes <= max_compact_stage_bytes
        ):
            current.append(expert)
            continue
        if not current:
            raise ExpertIOPlanError(
                f"expert {expert} cannot fit one tile: planned stage bytes "
                f"{candidate_tile.planned_read_bytes} exceed {max_stage_bytes} "
                "or compact stage bytes "
                f"{candidate_tile.compact_stage_bytes} exceed "
                f"{max_compact_stage_bytes}"
            )
        tiles.append(
            _batch_expert_io_tile(
                batch_plan=batch_plan,
                tile_index=len(tiles),
                selected_experts=tuple(current),
                merge_gap_bytes=merge_gap_bytes,
                align_bytes=align_bytes,
            )
        )
        current = [expert]
    if current:
        tiles.append(
            _batch_expert_io_tile(
                batch_plan=batch_plan,
                tile_index=len(tiles),
                selected_experts=tuple(current),
                merge_gap_bytes=merge_gap_bytes,
                align_bytes=align_bytes,
            )
        )
    return BatchExpertIOTilingPlan(
        batch_plan=batch_plan,
        max_stage_bytes=max_stage_bytes,
        max_compact_stage_bytes=max_compact_stage_bytes,
        tile_count=len(tiles),
        total_tile_assignments=sum(tile.total_assignments for tile in tiles),
        total_tile_planned_read_bytes=sum(tile.planned_read_bytes for tile in tiles),
        max_tile_planned_read_bytes=max(
            (tile.planned_read_bytes for tile in tiles),
            default=0,
        ),
        max_tile_compact_stage_bytes=max(
            (tile.compact_stage_bytes for tile in tiles),
            default=0,
        ),
        tiles=tuple(tiles),
    )


def plan_static_expert_capacity(
    batch_plan: BatchExpertIOPlan,
    *,
    capacity_per_expert: int,
) -> StaticExpertCapacityPlan:
    capacity_per_expert = _positive_integer_value(
        capacity_per_expert,
        "capacity_per_expert",
    )
    slots: list[StaticExpertTokenSlot] = []
    overflow: list[StaticExpertOverflow] = []
    usages: list[StaticExpertUsage] = []
    max_tokens_per_expert = 0

    for assignment in batch_plan.expert_tokens:
        assigned = len(assignment.tokens)
        max_tokens_per_expert = max(max_tokens_per_expert, assigned)
        used = min(assigned, capacity_per_expert)
        overflow_count = assigned - used
        usages.append(
            StaticExpertUsage(
                expert=assignment.expert,
                assigned_tokens=assigned,
                used_slots=used,
                overflow_assignments=overflow_count,
            )
        )
        for index, (token_index, weight) in enumerate(
            zip(assignment.tokens, assignment.weights)
        ):
            if index < capacity_per_expert:
                slots.append(
                    StaticExpertTokenSlot(
                        expert=assignment.expert,
                        slot=index,
                        token_index=token_index,
                        weight=weight,
                    )
                )
            else:
                overflow.append(
                    StaticExpertOverflow(
                        expert=assignment.expert,
                        overflow_index=index - capacity_per_expert,
                        token_index=token_index,
                        weight=weight,
                    )
                )

    total_capacity_slots = len(batch_plan.selected_experts) * capacity_per_expert
    used_slots = len(slots)
    utilization = used_slots / total_capacity_slots if total_capacity_slots else 0.0
    return StaticExpertCapacityPlan(
        batch_tokens=batch_plan.batch_tokens,
        selected_experts=batch_plan.selected_experts,
        capacity_per_expert=capacity_per_expert,
        total_assignments=batch_plan.total_assignments,
        total_capacity_slots=total_capacity_slots,
        used_slots=used_slots,
        utilization=float(utilization),
        overflow_assignments=len(overflow),
        max_tokens_per_expert=max_tokens_per_expert,
        requires_overflow_path=bool(overflow),
        usages=tuple(usages),
        slots=tuple(slots),
        overflow=tuple(overflow),
    )


def static_expert_capacity_payload(plan: StaticExpertCapacityPlan) -> dict[str, Any]:
    active_slots = {(item.expert, item.slot): item for item in plan.slots}
    overflow_by_expert: dict[int, list[StaticExpertOverflow]] = {}
    for item in plan.overflow:
        overflow_by_expert.setdefault(item.expert, []).append(item)

    expert_entries: list[dict[str, Any]] = []
    for usage in plan.usages:
        slot_entries: list[dict[str, Any]] = []
        for slot in range(plan.capacity_per_expert):
            item = active_slots.get((usage.expert, slot))
            if item is None:
                slot_entries.append(
                    {
                        "slot": slot,
                        "active": False,
                        "token_index": None,
                        "weight": 0.0,
                    }
                )
            else:
                slot_entries.append(
                    {
                        "slot": slot,
                        "active": True,
                        "token_index": item.token_index,
                        "weight": item.weight,
                    }
                )
        expert_entries.append(
            {
                "expert": usage.expert,
                "assigned_tokens": usage.assigned_tokens,
                "used_slots": usage.used_slots,
                "overflow_assignments": usage.overflow_assignments,
                "slots": slot_entries,
                "overflow": [
                    {
                        "overflow_index": item.overflow_index,
                        "token_index": item.token_index,
                        "weight": item.weight,
                    }
                    for item in overflow_by_expert.get(usage.expert, [])
                ],
            }
        )

    return {
        "version": 1,
        "format": "largerlm.static_expert_capacity.v1",
        "batch_tokens": plan.batch_tokens,
        "selected_experts": list(plan.selected_experts),
        "capacity_per_expert": plan.capacity_per_expert,
        "total_assignments": plan.total_assignments,
        "total_capacity_slots": plan.total_capacity_slots,
        "used_slots": plan.used_slots,
        "utilization": plan.utilization,
        "overflow_assignments": plan.overflow_assignments,
        "max_tokens_per_expert": plan.max_tokens_per_expert,
        "requires_overflow_path": plan.requires_overflow_path,
        "experts": expert_entries,
    }


def write_static_expert_capacity_plan(
    plan: StaticExpertCapacityPlan,
    output_path: str | Path,
    *,
    allow_overflow: bool = False,
) -> Path:
    if plan.requires_overflow_path and not allow_overflow:
        raise ExpertIOPlanError(
            "static expert capacity plan has overflow assignments; "
            "increase capacity_per_expert or allow overflow explicitly"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json_atomic(path, static_expert_capacity_payload(plan))
    except OSError as exc:
        raise ExpertIOPlanError(f"failed to write static capacity plan {path}: {exc}") from exc
    return path


_STATIC_CAPACITY_BINARY_MAGIC = b"LLMSCAP1"
_STATIC_CAPACITY_HEADER = struct.Struct("<8sIIIIIIII")
_STATIC_CAPACITY_EXPERT = struct.Struct("<I")
_STATIC_CAPACITY_SLOT = struct.Struct("<IfI")
_STATIC_CAPACITY_OVERFLOW = struct.Struct("<IIIf")
_STATIC_CAPACITY_INACTIVE_TOKEN = 0xFFFFFFFF


def static_expert_capacity_json_bytes(plan: StaticExpertCapacityPlan) -> int:
    payload = static_expert_capacity_payload(plan)
    return len(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def static_expert_capacity_binary_bytes(plan: StaticExpertCapacityPlan) -> int:
    return static_expert_capacity_binary_bytes_for_counts(
        expert_count=len(plan.selected_experts),
        capacity_per_expert=plan.capacity_per_expert,
        overflow_records=len(plan.overflow),
    )


def static_expert_capacity_binary_bytes_for_counts(
    *,
    expert_count: int,
    capacity_per_expert: int,
    overflow_records: int = 0,
) -> int:
    if type(expert_count) is not int or expert_count < 0:
        raise ExpertIOPlanError("expert_count must be a non-negative integer")
    if type(capacity_per_expert) is not int or capacity_per_expert <= 0:
        raise ExpertIOPlanError("capacity_per_expert must be a positive integer")
    if type(overflow_records) is not int or overflow_records < 0:
        raise ExpertIOPlanError("overflow_records must be a non-negative integer")
    slot_records = expert_count * capacity_per_expert
    return (
        _STATIC_CAPACITY_HEADER.size
        + expert_count * _STATIC_CAPACITY_EXPERT.size
        + slot_records * _STATIC_CAPACITY_SLOT.size
        + overflow_records * _STATIC_CAPACITY_OVERFLOW.size
    )


def _u32(value: int, *, field: str) -> int:
    if value < 0 or value > 0xFFFFFFFF:
        raise ExpertIOPlanError(f"{field}={value} does not fit uint32")
    return value


def write_static_expert_capacity_binary(
    plan: StaticExpertCapacityPlan,
    output_path: str | Path,
    *,
    allow_overflow: bool = False,
) -> StaticExpertCapacityBinaryReport:
    if plan.requires_overflow_path and not allow_overflow:
        raise ExpertIOPlanError(
            "static expert capacity plan has overflow assignments; "
            "increase capacity_per_expert or allow overflow explicitly"
        )
    expert_count = len(plan.selected_experts)
    slot_records = expert_count * plan.capacity_per_expert
    overflow_records = len(plan.overflow)
    for expert in plan.selected_experts:
        _u32(expert, field="expert")
    _u32(plan.batch_tokens, field="batch_tokens")
    _u32(expert_count, field="expert_count")
    _u32(plan.capacity_per_expert, field="capacity_per_expert")
    _u32(plan.total_assignments, field="total_assignments")
    _u32(plan.used_slots, field="used_slots")
    _u32(overflow_records, field="overflow_records")
    _u32(slot_records, field="slot_records")

    active_slots = {(item.expert, item.slot): item for item in plan.slots}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    bytes_written = 0
    try:
        with tmp_path.open("wb") as handle:
            chunk = _STATIC_CAPACITY_HEADER.pack(
                _STATIC_CAPACITY_BINARY_MAGIC,
                1,
                plan.batch_tokens,
                expert_count,
                plan.capacity_per_expert,
                plan.total_assignments,
                plan.used_slots,
                overflow_records,
                0,
            )
            handle.write(chunk)
            bytes_written += len(chunk)
            for expert in plan.selected_experts:
                chunk = _STATIC_CAPACITY_EXPERT.pack(_u32(expert, field="expert"))
                handle.write(chunk)
                bytes_written += len(chunk)
            for expert in plan.selected_experts:
                for slot in range(plan.capacity_per_expert):
                    item = active_slots.get((expert, slot))
                    if item is None:
                        chunk = _STATIC_CAPACITY_SLOT.pack(
                            _STATIC_CAPACITY_INACTIVE_TOKEN,
                            0.0,
                            0,
                        )
                    else:
                        chunk = _STATIC_CAPACITY_SLOT.pack(
                            _u32(item.token_index, field="token_index"),
                            float(item.weight),
                            1,
                        )
                    handle.write(chunk)
                    bytes_written += len(chunk)
            for item in plan.overflow:
                chunk = _STATIC_CAPACITY_OVERFLOW.pack(
                    _u32(item.expert, field="overflow_expert"),
                    _u32(item.overflow_index, field="overflow_index"),
                    _u32(item.token_index, field="overflow_token_index"),
                    float(item.weight),
                )
                handle.write(chunk)
                bytes_written += len(chunk)
        tmp_path.replace(path)
    except OSError as exc:
        _remove_partial_file(tmp_path)
        raise ExpertIOPlanError(f"failed to write static capacity binary {path}: {exc}") from exc
    return StaticExpertCapacityBinaryReport(
        path=path,
        bytes_written=bytes_written,
        version=1,
        expert_count=expert_count,
        capacity_per_expert=plan.capacity_per_expert,
        slot_records=slot_records,
        overflow_records=overflow_records,
    )


def _read_exact(handle, size: int, *, field: str) -> bytes:
    raw = handle.read(size)
    if len(raw) != size:
        raise ExpertIOPlanError(f"static capacity binary ended while reading {field}")
    return raw


def validate_static_expert_capacity_binary(
    path: str | Path,
    *,
    expected_plan: StaticExpertCapacityPlan | None = None,
) -> StaticExpertCapacityBinaryValidation:
    binary_path = Path(path)
    try:
        file_bytes = binary_path.stat().st_size
    except OSError as exc:
        raise ExpertIOPlanError(
            f"failed to stat static capacity binary {binary_path}: {exc}"
        ) from exc
    try:
        with binary_path.open("rb") as handle:
            header_raw = _read_exact(
                handle,
                _STATIC_CAPACITY_HEADER.size,
                field="header",
            )
            (
                magic,
                version,
                batch_tokens,
                expert_count,
                capacity_per_expert,
                total_assignments,
                used_slots,
                overflow_records,
                reserved,
            ) = _STATIC_CAPACITY_HEADER.unpack(header_raw)
            if magic != _STATIC_CAPACITY_BINARY_MAGIC:
                raise ExpertIOPlanError("static capacity binary has invalid magic")
            if version != 1:
                raise ExpertIOPlanError(
                    f"static capacity binary version {version} is not supported"
                )
            if reserved != 0:
                raise ExpertIOPlanError("static capacity binary reserved field must be zero")
            if expert_count <= 0:
                raise ExpertIOPlanError("static capacity binary expert_count must be positive")
            if capacity_per_expert <= 0:
                raise ExpertIOPlanError(
                    "static capacity binary capacity_per_expert must be positive"
                )
            expected_bytes = static_expert_capacity_binary_bytes_for_counts(
                expert_count=expert_count,
                capacity_per_expert=capacity_per_expert,
                overflow_records=overflow_records,
            )
            if file_bytes != expected_bytes:
                raise ExpertIOPlanError(
                    f"static capacity binary has {file_bytes} bytes, expected "
                    f"{expected_bytes}"
                )

            experts: list[int] = []
            previous_expert: int | None = None
            for index in range(expert_count):
                (expert,) = _STATIC_CAPACITY_EXPERT.unpack(
                    _read_exact(handle, _STATIC_CAPACITY_EXPERT.size, field="expert")
                )
                if previous_expert is not None and expert <= previous_expert:
                    raise ExpertIOPlanError(
                        "static capacity binary experts must be sorted and unique"
                    )
                experts.append(expert)
                previous_expert = expert

            expert_set = set(experts)
            active_slot_records = 0
            inactive_slot_records = 0
            slot_records = expert_count * capacity_per_expert
            for _ in range(slot_records):
                token_index, weight, active = _STATIC_CAPACITY_SLOT.unpack(
                    _read_exact(handle, _STATIC_CAPACITY_SLOT.size, field="slot")
                )
                if active not in (0, 1):
                    raise ExpertIOPlanError(
                        "static capacity binary slot active flag must be 0 or 1"
                    )
                if active:
                    if token_index >= batch_tokens:
                        raise ExpertIOPlanError(
                            "static capacity binary slot token_index is outside batch"
                        )
                    if not math.isfinite(weight):
                        raise ExpertIOPlanError(
                            "static capacity binary slot weight must be finite"
                        )
                    active_slot_records += 1
                else:
                    if token_index != _STATIC_CAPACITY_INACTIVE_TOKEN:
                        raise ExpertIOPlanError(
                            "static capacity binary inactive slot token must be sentinel"
                        )
                    if weight != 0.0:
                        raise ExpertIOPlanError(
                            "static capacity binary inactive slot weight must be zero"
                        )
                    inactive_slot_records += 1

            overflow_seen_by_expert: dict[int, set[int]] = {}
            for _ in range(overflow_records):
                expert, overflow_index, token_index, weight = (
                    _STATIC_CAPACITY_OVERFLOW.unpack(
                        _read_exact(
                            handle,
                            _STATIC_CAPACITY_OVERFLOW.size,
                            field="overflow",
                        )
                    )
                )
                if expert not in expert_set:
                    raise ExpertIOPlanError(
                        "static capacity binary overflow expert is not selected"
                    )
                if token_index >= batch_tokens:
                    raise ExpertIOPlanError(
                        "static capacity binary overflow token_index is outside batch"
                    )
                if not math.isfinite(weight):
                    raise ExpertIOPlanError(
                        "static capacity binary overflow weight must be finite"
                    )
                seen = overflow_seen_by_expert.setdefault(expert, set())
                if overflow_index in seen:
                    raise ExpertIOPlanError(
                        "static capacity binary overflow_index repeats for expert"
                    )
                seen.add(overflow_index)

            trailing = handle.read(1)
            if trailing:
                raise ExpertIOPlanError("static capacity binary has trailing bytes")
    except OSError as exc:
        raise ExpertIOPlanError(
            f"failed to read static capacity binary {binary_path}: {exc}"
        ) from exc

    if used_slots != active_slot_records:
        raise ExpertIOPlanError(
            "static capacity binary used_slots does not match active slot records"
        )
    if total_assignments != active_slot_records + overflow_records:
        raise ExpertIOPlanError(
            "static capacity binary total_assignments does not match records"
        )

    if expected_plan is not None:
        if batch_tokens != expected_plan.batch_tokens:
            raise ExpertIOPlanError("static capacity binary batch_tokens mismatches plan")
        if tuple(experts) != tuple(expected_plan.selected_experts):
            raise ExpertIOPlanError("static capacity binary experts mismatch plan")
        if capacity_per_expert != expected_plan.capacity_per_expert:
            raise ExpertIOPlanError(
                "static capacity binary capacity_per_expert mismatches plan"
            )
        if total_assignments != expected_plan.total_assignments:
            raise ExpertIOPlanError(
                "static capacity binary total_assignments mismatches plan"
            )
        if used_slots != expected_plan.used_slots:
            raise ExpertIOPlanError("static capacity binary used_slots mismatches plan")
        if overflow_records != len(expected_plan.overflow):
            raise ExpertIOPlanError(
                "static capacity binary overflow_records mismatches plan"
            )

    return StaticExpertCapacityBinaryValidation(
        path=binary_path,
        bytes_read=file_bytes,
        version=version,
        batch_tokens=batch_tokens,
        expert_count=expert_count,
        capacity_per_expert=capacity_per_expert,
        total_assignments=total_assignments,
        used_slots=used_slots,
        overflow_records=overflow_records,
        slot_records=slot_records,
        active_slot_records=active_slot_records,
        inactive_slot_records=inactive_slot_records,
    )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _copy_exact_range(
    *,
    source,
    destination,
    source_offset: int,
    length: int,
    copy_chunk_bytes: int,
    copy_buffer: bytearray | None = None,
) -> CopyRangeStats:
    if copy_chunk_bytes <= 0:
        raise ExpertIOPlanError("copy_chunk_bytes must be positive")
    remaining = length
    current_offset = source_offset
    read_calls = 0
    write_calls = 0
    copied_bytes = 0
    buffer = copy_buffer
    if buffer is None or len(buffer) < min(copy_chunk_bytes, max(remaining, 1)):
        buffer = bytearray(min(copy_chunk_bytes, max(remaining, 1)))
    view = memoryview(buffer)
    while remaining:
        chunk_size = min(copy_chunk_bytes, remaining)
        chunk_view = view[:chunk_size]
        if hasattr(os, "preadv"):
            read = os.preadv(source.fileno(), [chunk_view], current_offset)
        else:
            chunk = os.pread(source.fileno(), chunk_size, current_offset)
            read = len(chunk)
            if read == chunk_size:
                chunk_view[:] = chunk
        if read != chunk_size:
            raise ExpertIOPlanError(
                f"failed to read {chunk_size} bytes at source offset {current_offset}"
            )
        read_calls += 1
        written = destination.write(chunk_view)
        if written != chunk_size:
            raise ExpertIOPlanError(
                f"failed to write {chunk_size} staged bytes at source offset "
                f"{current_offset}"
            )
        write_calls += 1
        copied_bytes += chunk_size
        remaining -= chunk_size
        current_offset += chunk_size
    return CopyRangeStats(
        read_calls=read_calls,
        write_calls=write_calls,
        copied_bytes=copied_bytes,
    )


_F_RDADVISE = getattr(fcntl, "F_RDADVISE", 44)
_RADVISORY = struct.Struct("<qi4x")
_READ_ADVICE_MAX_CHUNK = (1 << 31) - 1
_COPY_CALL_COUNTERFACTUAL_CHUNK_MIB = (8, 16, 32, 64, 128)


def _copy_read_call_counterfactuals(
    ranges: tuple[ExpertReadRange, ...],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for chunk_mib in _COPY_CALL_COUNTERFACTUAL_CHUNK_MIB:
        chunk_bytes = chunk_mib * 1024 * 1024
        calls = 0
        for read_range in ranges:
            calls += (read_range.aligned_length + chunk_bytes - 1) // chunk_bytes
        result[str(chunk_mib)] = calls
    return result


def _advise_read_ranges(fd: int, ranges: tuple[ExpertReadRange, ...]) -> ReadAdviceStats:
    attempted = len(ranges)
    if attempted == 0:
        return ReadAdviceStats(
            supported=sys.platform == "darwin",
            attempted_ranges=0,
            calls=0,
            advised_bytes=0,
            error=None,
        )
    if sys.platform != "darwin":
        return ReadAdviceStats(
            supported=False,
            attempted_ranges=attempted,
            calls=0,
            advised_bytes=0,
            error="F_RDADVISE is only available on macOS",
        )

    calls = 0
    advised_bytes = 0
    for read_range in ranges:
        offset = int(read_range.aligned_offset)
        remaining = int(read_range.aligned_length)
        while remaining > 0:
            chunk = min(remaining, _READ_ADVICE_MAX_CHUNK)
            try:
                fcntl.fcntl(fd, _F_RDADVISE, _RADVISORY.pack(offset, chunk))
            except OSError as exc:
                return ReadAdviceStats(
                    supported=calls > 0,
                    attempted_ranges=attempted,
                    calls=calls,
                    advised_bytes=advised_bytes,
                    error=str(exc),
                )
            calls += 1
            advised_bytes += chunk
            offset += chunk
            remaining -= chunk
    return ReadAdviceStats(
        supported=True,
        attempted_ranges=attempted,
        calls=calls,
        advised_bytes=advised_bytes,
        error=None,
    )


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def _batch_stage_io_summary(
    *,
    batch_plan: BatchExpertIOPlan,
    staged_bytes: int,
    max_stage_bytes: int,
    slot_count: int,
    max_read_seconds: float,
    copy_elapsed_seconds: float | None = None,
    copy_read_calls: int = 0,
    copy_write_calls: int = 0,
    max_raw_ranges: int = 0,
    max_coalesced_ranges: int = 0,
) -> BatchExpertStageIOSummary:
    unique_requested = batch_plan.unique_requested_bytes
    planned_read = batch_plan.planned_read_bytes
    serial_read = batch_plan.serial_read_bytes
    copy_throughput = (
        (staged_bytes / 1024**3) / copy_elapsed_seconds
        if copy_elapsed_seconds is not None and copy_elapsed_seconds > 0
        else None
    )
    return BatchExpertStageIOSummary(
        batch_tokens=batch_plan.batch_tokens,
        selected_expert_count=len(batch_plan.selected_experts),
        total_assignments=batch_plan.total_assignments,
        serial_read_bytes=serial_read,
        unique_requested_bytes=unique_requested,
        planned_read_bytes=planned_read,
        staged_bytes=staged_bytes,
        coalesced_savings_bytes=batch_plan.coalesced_savings_bytes,
        waste_bytes=batch_plan.io_plan.waste_bytes,
        raw_range_count=batch_plan.raw_range_count,
        coalesced_range_count=batch_plan.coalesced_range_count,
        coalesced_range_savings=batch_plan.coalesced_range_savings,
        slot_count=slot_count,
        max_stage_bytes=max_stage_bytes,
        stage_budget_utilization=(
            staged_bytes / max_stage_bytes if max_stage_bytes else 0.0
        ),
        assignment_read_amplification=batch_plan.assignment_read_amplification,
        unique_read_amplification=batch_plan.unique_read_amplification,
        staged_unique_read_amplification=(
            staged_bytes / unique_requested if unique_requested else 0.0
        ),
        coalesced_savings_ratio=(
            batch_plan.coalesced_savings_bytes / serial_read if serial_read else 0.0
        ),
        waste_ratio=batch_plan.io_plan.waste_bytes / planned_read if planned_read else 0.0,
        ssd_read_gib_per_second=batch_plan.ssd_read_gib_per_second,
        planned_read_seconds=batch_plan.planned_read_seconds,
        max_read_seconds=max_read_seconds,
        read_seconds_ok=(
            None
            if max_read_seconds <= 0 or batch_plan.planned_read_seconds is None
            else batch_plan.planned_read_seconds <= max_read_seconds
        ),
        copy_elapsed_seconds=copy_elapsed_seconds,
        copy_throughput_gib_per_second=copy_throughput,
        copy_seconds_ok=(
            None
            if max_read_seconds <= 0 or copy_elapsed_seconds is None
            else copy_elapsed_seconds <= max_read_seconds
        ),
        copy_read_calls=copy_read_calls,
        copy_write_calls=copy_write_calls,
        copy_average_read_bytes=(
            staged_bytes / copy_read_calls if copy_read_calls else None
        ),
        copy_average_write_bytes=(
            staged_bytes / copy_write_calls if copy_write_calls else None
        ),
        copy_read_call_counterfactuals_by_chunk_mib=(
            _copy_read_call_counterfactuals(batch_plan.io_plan.ranges)
        ),
        max_raw_ranges=max_raw_ranges,
        raw_range_count_ok=(
            None
            if max_raw_ranges <= 0
            else batch_plan.raw_range_count <= max_raw_ranges
        ),
        max_coalesced_ranges=max_coalesced_ranges,
        coalesced_range_count_ok=(
            None
            if max_coalesced_ranges <= 0
            else batch_plan.coalesced_range_count <= max_coalesced_ranges
        ),
    )


def stage_batch_experts(
    expert_layout_path: str | Path,
    *,
    layer: int,
    router_json_dir: str | Path,
    stage_file_path: str | Path,
    manifest_path: str | Path | None = None,
    router_json_glob: str = "*.router.json",
    merge_gap_bytes: int = 0,
    align_bytes: int = 4096,
    max_stage_mib: float = 4096.0,
    copy_chunk_mib: float = 8.0,
    disk_safety_margin_bytes: int = 0,
    ssd_read_gib_per_second: float | int = 0.0,
    max_read_seconds: float | int = 0.0,
    max_raw_ranges: int = 0,
    max_coalesced_ranges: int = 0,
) -> BatchExpertStageResult:
    max_stage_mib = _positive_limit("max_stage_mib", max_stage_mib)
    copy_chunk_mib = _positive_limit("copy_chunk_mib", copy_chunk_mib)
    ssd_read_gib_per_second = _nonnegative_limit(
        "ssd_read_gib_per_second",
        ssd_read_gib_per_second,
    )
    max_read_seconds = _nonnegative_limit("max_read_seconds", max_read_seconds)
    if max_read_seconds > 0 and ssd_read_gib_per_second <= 0:
        raise ExpertIOPlanError(
            "ssd_read_gib_per_second must be positive when max_read_seconds is set"
        )
    disk_safety_margin_bytes = _nonnegative_integer_value(
        disk_safety_margin_bytes,
        "disk_safety_margin_bytes",
    )
    max_raw_ranges = _nonnegative_integer_value(
        max_raw_ranges,
        "max_raw_ranges",
    )
    max_coalesced_ranges = _nonnegative_integer_value(
        max_coalesced_ranges,
        "max_coalesced_ranges",
    )
    max_stage_bytes = int(max_stage_mib * 1024 * 1024)
    copy_chunk_bytes = max(1, int(copy_chunk_mib * 1024 * 1024))
    batch_plan = plan_batch_expert_io(
        expert_layout_path,
        layer=layer,
        router_json_dir=router_json_dir,
        router_json_glob=router_json_glob,
        merge_gap_bytes=merge_gap_bytes,
        align_bytes=align_bytes,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
    )
    if max_raw_ranges > 0 and batch_plan.raw_range_count > max_raw_ranges:
        raise ExpertIOPlanError(
            f"raw range count {batch_plan.raw_range_count} exceeds limit "
            f"{max_raw_ranges}"
        )
    if (
        max_coalesced_ranges > 0
        and batch_plan.coalesced_range_count > max_coalesced_ranges
    ):
        raise ExpertIOPlanError(
            f"coalesced range count {batch_plan.coalesced_range_count} exceeds "
            f"limit {max_coalesced_ranges}"
        )
    if batch_plan.planned_read_bytes > max_stage_bytes:
        raise ExpertIOPlanError(
            f"planned stage bytes {batch_plan.planned_read_bytes} exceed limit "
            f"{max_stage_bytes}"
        )
    if (
        max_read_seconds > 0
        and batch_plan.planned_read_seconds is not None
        and batch_plan.planned_read_seconds > max_read_seconds
    ):
        raise ExpertIOPlanError(
            f"planned stage read time {batch_plan.planned_read_seconds:.6g}s "
            f"exceeds limit {max_read_seconds:.6g}s at "
            f"ssd_read_gib_per_second={ssd_read_gib_per_second:.6g}"
        )
    layer_file_path = batch_plan.io_plan.layer_file_path
    try:
        layer_file_bytes = layer_file_path.stat().st_size
    except OSError as exc:
        raise ExpertIOPlanError(f"failed to stat layer file {layer_file_path}: {exc}") from exc
    expected_layer_bytes = (
        batch_plan.io_plan.num_experts * batch_plan.io_plan.expert_slot_bytes
    )
    if layer_file_bytes < expected_layer_bytes:
        raise ExpertIOPlanError(
            f"layer file has {layer_file_bytes} bytes, expected at least "
            f"{expected_layer_bytes}"
        )

    stage_path = Path(stage_file_path)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    budget = disk_budget(
        stage_path.parent,
        batch_plan.planned_read_bytes,
        safety_margin_bytes=disk_safety_margin_bytes,
    )
    if not budget.ok:
        raise ExpertIOPlanError(
            "not enough free disk for expert stage file: "
            f"need {batch_plan.planned_read_bytes + disk_safety_margin_bytes} bytes "
            f"including margin, have {budget.available_bytes} bytes"
        )
    manifest_p = (
        Path(manifest_path)
        if manifest_path is not None
        else stage_path.with_suffix(stage_path.suffix + ".manifest.json")
    )
    manifest_p.parent.mkdir(parents=True, exist_ok=True)

    staged_ranges: list[StagedExpertRange] = []
    staged_slots: list[StagedExpertSlot] = []
    current_stage_offset = 0
    copy_read_calls = 0
    copy_write_calls = 0
    copy_buffer = bytearray(copy_chunk_bytes)
    read_advice = ReadAdviceStats(
        supported=False,
        attempted_ranges=len(batch_plan.io_plan.ranges),
        calls=0,
        advised_bytes=0,
        error="stage copy did not start",
    )
    copy_started = time.perf_counter()
    try:
        with layer_file_path.open("rb") as source, stage_path.open("wb") as destination:
            read_advice = _advise_read_ranges(
                source.fileno(),
                batch_plan.io_plan.ranges,
            )
            for range_index, read_range in enumerate(batch_plan.io_plan.ranges):
                stage_offset = current_stage_offset
                copy_stats = _copy_exact_range(
                    source=source,
                    destination=destination,
                    source_offset=read_range.aligned_offset,
                    length=read_range.aligned_length,
                    copy_chunk_bytes=copy_chunk_bytes,
                    copy_buffer=copy_buffer,
                )
                copy_read_calls += copy_stats.read_calls
                copy_write_calls += copy_stats.write_calls
                staged_ranges.append(
                    StagedExpertRange(
                        range_index=range_index,
                        experts=read_range.experts,
                        source_offset=read_range.aligned_offset,
                        source_length=read_range.aligned_length,
                        stage_offset=stage_offset,
                        stage_length=read_range.aligned_length,
                    )
                )
                for expert in read_range.experts:
                    physical_slot = (
                        batch_plan.io_plan.expert_physical_slots[expert]
                        if batch_plan.io_plan.expert_physical_slots
                        else expert
                    )
                    source_slot_offset = (
                        physical_slot * batch_plan.io_plan.expert_slot_bytes
                    )
                    staged_slots.append(
                        StagedExpertSlot(
                            expert=expert,
                            range_index=range_index,
                            source_offset=source_slot_offset,
                            stage_offset=stage_offset
                            + (source_slot_offset - read_range.aligned_offset),
                            length=batch_plan.io_plan.expert_slot_bytes,
                        )
                    )
                current_stage_offset += read_range.aligned_length
    except (OSError, ExpertIOPlanError) as exc:
        _remove_partial_file(stage_path)
        raise ExpertIOPlanError(f"failed to stage batch experts: {exc}") from exc
    copy_elapsed_seconds = max(0.0, time.perf_counter() - copy_started)

    try:
        staged_bytes = stage_path.stat().st_size
    except OSError as exc:
        _remove_partial_file(stage_path)
        raise ExpertIOPlanError(f"failed to stat stage file {stage_path}: {exc}") from exc
    if staged_bytes != batch_plan.planned_read_bytes:
        _remove_partial_file(stage_path)
        raise ExpertIOPlanError(
            f"stage file has {staged_bytes} bytes, expected "
            f"{batch_plan.planned_read_bytes}"
        )
    io_summary = _batch_stage_io_summary(
        batch_plan=batch_plan,
        staged_bytes=staged_bytes,
        max_stage_bytes=max_stage_bytes,
        slot_count=len(staged_slots),
        max_read_seconds=max_read_seconds,
        copy_elapsed_seconds=copy_elapsed_seconds,
        copy_read_calls=copy_read_calls,
        copy_write_calls=copy_write_calls,
        max_raw_ranges=max_raw_ranges,
        max_coalesced_ranges=max_coalesced_ranges,
    )
    if io_summary.copy_seconds_ok is False:
        _remove_partial_file(stage_path)
        _remove_partial_file(manifest_p)
        raise ExpertIOPlanError(
            "actual stage copy time "
            f"{io_summary.copy_elapsed_seconds:.6g}s exceeds limit "
            f"{max_read_seconds:.6g}s"
        )
    result = BatchExpertStageResult(
        expert_layout_path=batch_plan.expert_layout_path,
        router_json_dir=batch_plan.router_json_dir,
        stage_file_path=stage_path,
        manifest_path=manifest_p,
        layer=layer,
        batch_tokens=batch_plan.batch_tokens,
        selected_experts=batch_plan.selected_experts,
        expert_slot_bytes=batch_plan.io_plan.expert_slot_bytes,
        planned_read_bytes=batch_plan.planned_read_bytes,
        staged_bytes=staged_bytes,
        max_stage_bytes=max_stage_bytes,
        copy_chunk_bytes=copy_chunk_bytes,
        read_advice=read_advice,
        io_summary=io_summary,
        ranges=tuple(staged_ranges),
        slots=tuple(sorted(staged_slots, key=lambda item: item.expert)),
        batch_plan=batch_plan,
        copy_elapsed_seconds=io_summary.copy_elapsed_seconds,
        copy_throughput_gib_per_second=io_summary.copy_throughput_gib_per_second,
    )
    try:
        _write_json_atomic(manifest_p, _json_ready(result))
    except OSError as exc:
        _remove_partial_file(stage_path)
        raise ExpertIOPlanError(f"failed to write stage manifest {manifest_p}: {exc}") from exc
    return result
