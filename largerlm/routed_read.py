from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .expert_io import (
    ExpertIOPlanError,
    static_expert_capacity_binary_bytes_for_counts,
)


class RoutedExpertReadError(RuntimeError):
    """Raised when routed expert read cost cannot be estimated."""


@dataclass(frozen=True)
class RoutedExpertReadEstimate:
    prompt_chunk_tokens: int
    top_k: int
    chunks_per_prompt: int
    layers: int
    baseline_read_bytes: int
    planned_read_bytes: int
    extra_read_bytes: int
    read_amplification: float
    max_layer_baseline_read_bytes: int
    max_layer_planned_read_bytes: int


@dataclass(frozen=True)
class RoutedStageTempEstimate:
    prompt_chunk_tokens: int
    top_k: int
    chunks_per_prompt: int
    layers: int
    stage_align_bytes: int
    static_capacity_per_expert: int | str | None
    allow_static_capacity_overflow: bool
    max_static_capacity_per_expert: int
    static_capacity_strict_overflow_safe: bool
    max_unique_experts_per_layer: int
    max_stage_raw_ranges: int
    max_stage_coalesced_ranges: int
    max_stage_bytes: int
    max_compact_stage_bytes: int
    max_stage_plus_compact_bytes: int
    max_static_capacity_binary_bytes: int
    max_static_capacity_overflow_records: int
    max_stage_plus_compact_plus_static_bytes: int
    max_chunk_stage_bytes: int
    max_chunk_compact_stage_bytes: int
    max_chunk_stage_plus_compact_bytes: int
    max_chunk_static_capacity_binary_bytes: int
    max_chunk_stage_plus_compact_plus_static_bytes: int
    total_stage_bytes: int
    total_compact_stage_bytes: int
    total_stage_plus_compact_bytes: int
    total_static_capacity_binary_bytes: int
    total_static_capacity_overflow_records: int
    total_stage_plus_compact_plus_static_bytes: int


@dataclass(frozen=True)
class RoutedPrefillChunkCandidate:
    prompt_chunk_tokens: int
    chunks_per_prompt: int
    saturates_all_experts_per_layer: bool
    planned_read_bytes: int
    extra_read_bytes: int
    read_amplification: float
    max_layer_planned_read_bytes: int
    max_stage_plus_compact_bytes: int
    max_chunk_stage_plus_compact_bytes: int
    total_stage_plus_compact_bytes: int
    max_static_capacity_binary_bytes: int
    max_chunk_static_capacity_binary_bytes: int
    total_static_capacity_binary_bytes: int
    max_stage_plus_compact_plus_static_bytes: int
    max_chunk_stage_plus_compact_plus_static_bytes: int
    total_stage_plus_compact_plus_static_bytes: int
    planned_read_seconds: float | None


@dataclass(frozen=True)
class RoutedPrefillChunkFrontier:
    prompt_token_count: int
    top_k: int
    layers: int
    stage_align_bytes: int
    static_capacity_per_expert: int | str | None
    allow_static_capacity_overflow: bool
    baseline_read_bytes: int
    saturation_chunk_tokens: int | None
    candidates: tuple[RoutedPrefillChunkCandidate, ...]


@dataclass(frozen=True)
class _RoutedExpertLayerSpec:
    layer: int
    num_experts: int
    expert_slot_bytes: int


def _load_routed_expert_layer_specs(
    *,
    expert_layout_path: str | Path,
    layers: Iterable[int] | None,
    purpose: str,
) -> tuple[_RoutedExpertLayerSpec, ...]:
    try:
        payload = json.loads(Path(expert_layout_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutedExpertReadError(
            f"failed to inspect expert layout for {purpose}: {exc}"
        ) from exc
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list):
        raise RoutedExpertReadError(
            f"expert layout missing layers array for {purpose}"
        )

    layer_filter = set(layers) if layers is not None else None
    specs: list[_RoutedExpertLayerSpec] = []
    for item in raw_layers:
        if not isinstance(item, dict):
            raise RoutedExpertReadError("expert layout layers must be objects")
        raw_layer = item.get("layer")
        if type(raw_layer) is not int:
            raise RoutedExpertReadError("expert layout layer must be an integer")
        layer = int(raw_layer)
        if layer_filter is not None and layer not in layer_filter:
            continue
        raw_experts = item.get("num_experts")
        raw_slot = item.get("expert_slot_bytes")
        if type(raw_experts) is not int or raw_experts < 0:
            raise RoutedExpertReadError(
                "expert layout num_experts must be a non-negative integer"
            )
        if type(raw_slot) is not int or raw_slot < 0:
            raise RoutedExpertReadError(
                "expert layout expert_slot_bytes must be a non-negative integer"
            )
        specs.append(
            _RoutedExpertLayerSpec(
                layer=layer,
                num_experts=int(raw_experts),
                expert_slot_bytes=int(raw_slot),
            )
        )
    return tuple(specs)


def _normalize_static_capacity_per_expert(value: object) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RoutedExpertReadError(
            "static_capacity_per_expert must be an integer, 'auto', or None"
        )
    if type(value) is int:
        if value <= 0:
            raise RoutedExpertReadError(
                "static_capacity_per_expert must be positive, 'auto', or None"
            )
        return value
    if not isinstance(value, str):
        raise RoutedExpertReadError(
            "static_capacity_per_expert must be an integer, 'auto', or None"
        )
    text = str(value).strip().lower()
    if text in {"", "none"}:
        return None
    if text == "auto":
        return "auto"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise RoutedExpertReadError(
            "static_capacity_per_expert must be positive, 'auto', or None"
        ) from exc
    if parsed <= 0:
        raise RoutedExpertReadError(
            "static_capacity_per_expert must be positive, 'auto', or None"
        )
    return parsed


def _static_capacity_for_chunk(
    value: int | str | None,
    *,
    chunk_tokens: int,
) -> int | None:
    if value == "auto":
        return chunk_tokens
    return value


def _estimate_static_capacity_binary_bytes(
    *,
    selected_experts: int,
    capacity_per_expert: int,
    chunk_tokens: int,
    top_k: int,
    allow_overflow: bool,
) -> tuple[int, int, bool]:
    if selected_experts <= 0:
        return 0, 0, True
    strict_overflow_safe = capacity_per_expert >= chunk_tokens
    overflow_records = 0
    if allow_overflow and not strict_overflow_safe:
        overflow_records = chunk_tokens * top_k
    try:
        binary_bytes = static_expert_capacity_binary_bytes_for_counts(
            expert_count=selected_experts,
            capacity_per_expert=capacity_per_expert,
            overflow_records=overflow_records,
        )
    except ExpertIOPlanError as exc:
        raise RoutedExpertReadError(str(exc)) from exc
    return binary_bytes, overflow_records, strict_overflow_safe


def format_routed_read_guard_flag_float(value: float) -> str:
    return f"{value:.6g}"


def _positive_headroom_factor(value: float | int, *, label: str) -> float:
    if isinstance(value, bool):
        raise RoutedExpertReadError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RoutedExpertReadError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise RoutedExpertReadError(f"{label} must be finite")
    if parsed <= 0:
        raise RoutedExpertReadError(f"{label} must be positive")
    return parsed


def suggest_decode_routed_read_guard_flags(
    *,
    read_bytes_per_token: int | None,
    ssd_read_gib_per_second: float | int | None = None,
    source: str | None = None,
    headroom_factor: float = 1.05,
) -> dict[str, object] | None:
    headroom = _positive_headroom_factor(
        headroom_factor,
        label="headroom_factor",
    )
    if type(read_bytes_per_token) is not int or read_bytes_per_token <= 0:
        return None
    suggested_gib = read_bytes_per_token / 1024**3 * headroom
    argv: list[str] = [
        "--decode-max-routed-read-gib-per-token",
        format_routed_read_guard_flag_float(suggested_gib),
    ]
    suggested: dict[str, object] = {
        "headroom_factor": headroom,
        "decode_read_bytes_per_token": read_bytes_per_token,
        "decode_max_routed_read_gib_per_token": suggested_gib,
    }
    if source:
        suggested["source"] = source
    if (
        not isinstance(ssd_read_gib_per_second, bool)
        and isinstance(ssd_read_gib_per_second, (int, float))
    ):
        ssd_gib_s = float(ssd_read_gib_per_second)
        if math.isfinite(ssd_gib_s) and ssd_gib_s > 0:
            seconds = read_bytes_per_token / (ssd_gib_s * 1024**3)
            suggested_seconds = seconds * headroom
            suggested["prefill_ssd_read_gib_per_second"] = ssd_gib_s
            suggested["decode_read_seconds_per_token"] = seconds
            suggested["decode_max_routed_read_seconds_per_token"] = (
                suggested_seconds
            )
            argv.extend(
                [
                    "--prefill-ssd-read-gib-s",
                    format_routed_read_guard_flag_float(ssd_gib_s),
                    "--decode-max-routed-read-seconds-per-token",
                    format_routed_read_guard_flag_float(suggested_seconds),
                ]
            )
    suggested["argv"] = tuple(argv)
    return suggested


def suggest_routed_read_guard_flags(
    *,
    prompt_chunk_tokens: int | None,
    planned_read_bytes: int | None,
    read_amplification: float | int | None,
    ssd_read_gib_per_second: float | int | None = None,
    planned_read_seconds: float | int | None = None,
    source: str | None = None,
    headroom_factor: float = 1.05,
) -> dict[str, object] | None:
    if isinstance(headroom_factor, bool):
        raise RoutedExpertReadError("headroom_factor must be numeric")
    try:
        headroom = float(headroom_factor)
    except (TypeError, ValueError) as exc:
        raise RoutedExpertReadError("headroom_factor must be numeric") from exc
    if not math.isfinite(headroom):
        raise RoutedExpertReadError("headroom_factor must be finite")
    if headroom <= 0:
        raise RoutedExpertReadError("headroom_factor must be positive")
    if type(prompt_chunk_tokens) is not int or prompt_chunk_tokens <= 0:
        return None
    if type(planned_read_bytes) is not int or planned_read_bytes <= 0:
        return None
    if isinstance(read_amplification, bool) or not isinstance(
        read_amplification,
        (int, float),
    ):
        return None
    read_amp = float(read_amplification)
    if not math.isfinite(read_amp) or read_amp < 0:
        return None

    suggested_amp = max(1.0, read_amp) * headroom
    suggested_gib = planned_read_bytes / 1024**3 * headroom
    argv: list[str] = [
        "--prefill-prompt-chunk-tokens",
        str(prompt_chunk_tokens),
        "--prefill-max-routed-read-amplification",
        format_routed_read_guard_flag_float(suggested_amp),
        "--prefill-max-routed-read-gib",
        format_routed_read_guard_flag_float(suggested_gib),
    ]
    suggested: dict[str, object] = {
        "headroom_factor": headroom,
        "prefill_prompt_chunk_tokens": prompt_chunk_tokens,
        "prefill_max_routed_read_amplification": suggested_amp,
        "prefill_max_routed_read_gib": suggested_gib,
    }
    if source:
        suggested["source"] = source

    if (
        not isinstance(ssd_read_gib_per_second, bool)
        and isinstance(ssd_read_gib_per_second, (int, float))
    ):
        ssd_gib_s = float(ssd_read_gib_per_second)
        if math.isfinite(ssd_gib_s) and ssd_gib_s > 0:
            seconds: float | None = None
            if planned_read_seconds is None:
                seconds = planned_read_bytes / (ssd_gib_s * 1024**3)
            elif (
                not isinstance(planned_read_seconds, bool)
                and isinstance(planned_read_seconds, (int, float))
            ):
                parsed_seconds = float(planned_read_seconds)
                if math.isfinite(parsed_seconds) and parsed_seconds >= 0:
                    seconds = parsed_seconds
            if seconds is not None:
                suggested_seconds = seconds * headroom
                suggested["prefill_ssd_read_gib_per_second"] = ssd_gib_s
                suggested["planned_routed_read_seconds"] = seconds
                suggested["prefill_max_routed_read_seconds"] = suggested_seconds
                argv.extend(
                    [
                        "--prefill-ssd-read-gib-s",
                        format_routed_read_guard_flag_float(ssd_gib_s),
                        "--prefill-max-routed-read-seconds",
                        format_routed_read_guard_flag_float(suggested_seconds),
                    ]
                )

    suggested["argv"] = tuple(argv)
    return suggested


def suggest_routed_stage_temp_guard_flags(
    *,
    prompt_chunk_tokens: int | None,
    max_stage_bytes: int | None,
    max_compact_stage_bytes: int | None,
    max_stage_raw_ranges: int | None = None,
    max_stage_coalesced_ranges: int | None = None,
    max_stage_plus_compact_bytes: int | None = None,
    total_stage_plus_compact_bytes: int | None = None,
    max_static_capacity_binary_bytes: int | None = None,
    total_static_capacity_binary_bytes: int | None = None,
    max_stage_plus_compact_plus_static_bytes: int | None = None,
    total_stage_plus_compact_plus_static_bytes: int | None = None,
    static_capacity_per_expert: object | None = None,
    source: str | None = None,
    headroom_factor: float = 1.05,
) -> dict[str, object] | None:
    if isinstance(headroom_factor, bool):
        raise RoutedExpertReadError("headroom_factor must be numeric")
    try:
        headroom = float(headroom_factor)
    except (TypeError, ValueError) as exc:
        raise RoutedExpertReadError("headroom_factor must be numeric") from exc
    if not math.isfinite(headroom):
        raise RoutedExpertReadError("headroom_factor must be finite")
    if headroom <= 0:
        raise RoutedExpertReadError("headroom_factor must be positive")
    if type(prompt_chunk_tokens) is not int or prompt_chunk_tokens <= 0:
        return None
    if type(max_stage_bytes) is not int or max_stage_bytes <= 0:
        return None
    if type(max_compact_stage_bytes) is not int or max_compact_stage_bytes <= 0:
        return None
    static_capacity = _normalize_static_capacity_per_expert(
        static_capacity_per_expert
    )

    suggested_stage_mib = max_stage_bytes / 1024**2 * headroom
    suggested_compact_mib = max_compact_stage_bytes / 1024**2 * headroom
    argv: list[str] = [
        "--prefill-prompt-chunk-tokens",
        str(prompt_chunk_tokens),
        "--prefill-max-stage-mib",
        format_routed_read_guard_flag_float(suggested_stage_mib),
        "--prefill-max-compact-stage-mib",
        format_routed_read_guard_flag_float(suggested_compact_mib),
    ]
    if static_capacity is not None:
        argv.extend(
            [
                "--prefill-static-capacity-per-expert",
                str(static_capacity),
            ]
        )
    suggested_raw_ranges = None
    if type(max_stage_raw_ranges) is int and max_stage_raw_ranges > 0:
        suggested_raw_ranges = max(1, math.ceil(max_stage_raw_ranges * headroom))
        argv.extend(["--prefill-max-stage-raw-ranges", str(suggested_raw_ranges)])
    suggested_coalesced_ranges = None
    if (
        type(max_stage_coalesced_ranges) is int
        and max_stage_coalesced_ranges > 0
    ):
        suggested_coalesced_ranges = max(
            1,
            math.ceil(max_stage_coalesced_ranges * headroom),
        )
        argv.extend(
            [
                "--prefill-max-stage-coalesced-ranges",
                str(suggested_coalesced_ranges),
            ]
        )
    suggested: dict[str, object] = {
        "headroom_factor": headroom,
        "prefill_prompt_chunk_tokens": prompt_chunk_tokens,
        "prefill_max_stage_mib": suggested_stage_mib,
        "prefill_max_compact_stage_mib": suggested_compact_mib,
        "profile_max_stage_bytes": max_stage_bytes,
        "profile_max_compact_stage_bytes": max_compact_stage_bytes,
    }
    if source:
        suggested["source"] = source
    if static_capacity is not None:
        suggested["prefill_static_capacity_per_expert"] = static_capacity
    if suggested_raw_ranges is not None:
        suggested["prefill_max_stage_raw_ranges"] = suggested_raw_ranges
        suggested["profile_max_stage_raw_ranges"] = max_stage_raw_ranges
    if suggested_coalesced_ranges is not None:
        suggested["prefill_max_stage_coalesced_ranges"] = (
            suggested_coalesced_ranges
        )
        suggested["profile_max_stage_coalesced_ranges"] = (
            max_stage_coalesced_ranges
        )
    if type(max_stage_plus_compact_bytes) is int and max_stage_plus_compact_bytes >= 0:
        suggested["profile_max_stage_plus_compact_bytes"] = (
            max_stage_plus_compact_bytes
        )
    if (
        type(total_stage_plus_compact_bytes) is int
        and total_stage_plus_compact_bytes >= 0
    ):
        suggested["profile_total_stage_plus_compact_bytes"] = (
            total_stage_plus_compact_bytes
        )
    if (
        type(max_static_capacity_binary_bytes) is int
        and max_static_capacity_binary_bytes >= 0
    ):
        suggested["profile_max_static_capacity_binary_bytes"] = (
            max_static_capacity_binary_bytes
        )
    if (
        type(total_static_capacity_binary_bytes) is int
        and total_static_capacity_binary_bytes >= 0
    ):
        suggested["profile_total_static_capacity_binary_bytes"] = (
            total_static_capacity_binary_bytes
        )
    if (
        type(max_stage_plus_compact_plus_static_bytes) is int
        and max_stage_plus_compact_plus_static_bytes >= 0
    ):
        suggested["profile_max_stage_plus_compact_plus_static_bytes"] = (
            max_stage_plus_compact_plus_static_bytes
        )
    if (
        type(total_stage_plus_compact_plus_static_bytes) is int
        and total_stage_plus_compact_plus_static_bytes >= 0
    ):
        suggested["profile_total_stage_plus_compact_plus_static_bytes"] = (
            total_stage_plus_compact_plus_static_bytes
        )
    suggested["argv"] = tuple(argv)
    return suggested


def _suggested_guard_value(
    suggested: dict[str, object] | None,
    field: str,
    flag: str,
) -> str | None:
    if not suggested:
        return None
    argv = suggested.get("argv")
    if isinstance(argv, (list, tuple)):
        items = [str(item) for item in argv]
        try:
            index = items.index(flag)
        except ValueError:
            pass
        else:
            if index + 1 < len(items):
                return items[index + 1]
    value = suggested.get(field)
    if field == "prefill_static_capacity_per_expert":
        try:
            parsed = _normalize_static_capacity_per_expert(value)
        except RoutedExpertReadError:
            return None
        return None if parsed is None else str(parsed)
    if field in {
        "prefill_max_stage_raw_ranges",
        "prefill_max_stage_coalesced_ranges",
    }:
        if type(value) is int and value > 0:
            return str(value)
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return format_routed_read_guard_flag_float(float(value))


def combine_prefill_guard_flags(
    *,
    routed_read_flags: dict[str, object] | None,
    stage_temp_flags: dict[str, object] | None,
    cache_io_flags: dict[str, object] | None = None,
    source: str,
) -> dict[str, object] | None:
    if routed_read_flags is None and stage_temp_flags is None and cache_io_flags is None:
        return None
    prompt_chunk_tokens = None
    for payload in (routed_read_flags, stage_temp_flags):
        if isinstance(payload, dict):
            value = payload.get("prefill_prompt_chunk_tokens")
            if type(value) is int and value > 0:
                prompt_chunk_tokens = value
                break
    argv: list[str] = []
    if prompt_chunk_tokens is not None:
        argv.extend(["--prefill-prompt-chunk-tokens", str(prompt_chunk_tokens)])
    field_flags = (
        (
            routed_read_flags,
            "prefill_max_routed_read_amplification",
            "--prefill-max-routed-read-amplification",
        ),
        (
            routed_read_flags,
            "prefill_max_routed_read_gib",
            "--prefill-max-routed-read-gib",
        ),
        (
            routed_read_flags,
            "prefill_ssd_read_gib_per_second",
            "--prefill-ssd-read-gib-s",
        ),
        (
            routed_read_flags,
            "prefill_max_routed_read_seconds",
            "--prefill-max-routed-read-seconds",
        ),
        (
            stage_temp_flags,
            "prefill_max_stage_mib",
            "--prefill-max-stage-mib",
        ),
        (
            stage_temp_flags,
            "prefill_max_compact_stage_mib",
            "--prefill-max-compact-stage-mib",
        ),
        (
            stage_temp_flags,
            "prefill_static_capacity_per_expert",
            "--prefill-static-capacity-per-expert",
        ),
        (
            stage_temp_flags,
            "prefill_max_stage_raw_ranges",
            "--prefill-max-stage-raw-ranges",
        ),
        (
            stage_temp_flags,
            "prefill_max_stage_coalesced_ranges",
            "--prefill-max-stage-coalesced-ranges",
        ),
        (
            cache_io_flags,
            "max_cache_read_mib",
            "--max-cache-read-mib",
        ),
        (
            cache_io_flags,
            "prefill_max_cache_write_mib",
            "--prefill-max-cache-write-mib",
        ),
    )
    for suggested, field, flag in field_flags:
        value = _suggested_guard_value(suggested, field, flag)
        if value is not None:
            argv.extend([flag, value])
    include_stage_tiling = (
        isinstance(stage_temp_flags, dict)
        and stage_temp_flags.get("prefill_expert_stage_tiling") is True
    )
    if include_stage_tiling:
        argv.append("--prefill-expert-stage-tiling")
    if not argv:
        return None
    payload: dict[str, object] = {
        "source": source,
        "argv": tuple(argv),
    }
    if prompt_chunk_tokens is not None:
        payload["prefill_prompt_chunk_tokens"] = prompt_chunk_tokens
    if routed_read_flags is not None:
        payload["routed_read_guard"] = routed_read_flags
    if stage_temp_flags is not None:
        payload["stage_temp_guard"] = stage_temp_flags
    if cache_io_flags is not None:
        payload["cache_io_guard"] = cache_io_flags
    if include_stage_tiling:
        payload["prefill_expert_stage_tiling"] = True
    return payload


def estimate_routed_expert_read(
    *,
    expert_layout_path: str | Path,
    prompt_token_count: int,
    prompt_chunk_tokens: int,
    top_k: int,
    layers: Iterable[int] | None = None,
) -> RoutedExpertReadEstimate:
    if type(prompt_token_count) is not int or prompt_token_count <= 0:
        raise RoutedExpertReadError("prompt_token_count must be a positive integer")
    if type(prompt_chunk_tokens) is not int or prompt_chunk_tokens <= 0:
        raise RoutedExpertReadError("prompt_chunk_tokens must be a positive integer")
    if type(top_k) is not int or top_k <= 0:
        raise RoutedExpertReadError("top_k must be a positive integer")

    chunks = (prompt_token_count + prompt_chunk_tokens - 1) // prompt_chunk_tokens
    baseline_read = 0
    planned_read = 0
    layer_count = 0
    max_layer_read = 0
    max_layer_planned = 0
    specs = _load_routed_expert_layer_specs(
        expert_layout_path=expert_layout_path,
        layers=layers,
        purpose="routed read estimate",
    )
    for spec in specs:
        experts = spec.num_experts
        slot_bytes = spec.expert_slot_bytes
        if experts <= 0 or slot_bytes <= 0:
            continue
        layer_count += 1
        full_assignments = prompt_token_count * top_k
        full_unique = min(experts, full_assignments)
        layer_baseline = full_unique * slot_bytes
        layer_planned = 0
        remaining = prompt_token_count
        while remaining > 0:
            current = min(prompt_chunk_tokens, remaining)
            chunk_unique = min(experts, current * top_k)
            layer_planned += chunk_unique * slot_bytes
            remaining -= current
        baseline_read += layer_baseline
        planned_read += layer_planned
        max_layer_read = max(max_layer_read, layer_baseline)
        max_layer_planned = max(max_layer_planned, layer_planned)

    extra = max(0, planned_read - baseline_read)
    amplification = (planned_read / baseline_read) if baseline_read else 1.0
    return RoutedExpertReadEstimate(
        prompt_chunk_tokens=prompt_chunk_tokens,
        top_k=top_k,
        chunks_per_prompt=chunks,
        layers=layer_count,
        baseline_read_bytes=baseline_read,
        planned_read_bytes=planned_read,
        extra_read_bytes=extra,
        read_amplification=float(amplification),
        max_layer_baseline_read_bytes=max_layer_read,
        max_layer_planned_read_bytes=max_layer_planned,
    )


def estimate_routed_stage_temp(
    *,
    expert_layout_path: str | Path,
    prompt_token_count: int,
    prompt_chunk_tokens: int,
    top_k: int,
    stage_align_bytes: int,
    layers: Iterable[int] | None = None,
    static_capacity_per_expert: object | None = None,
    allow_static_capacity_overflow: bool = False,
) -> RoutedStageTempEstimate:
    if type(prompt_token_count) is not int or prompt_token_count <= 0:
        raise RoutedExpertReadError("prompt_token_count must be a positive integer")
    if type(prompt_chunk_tokens) is not int or prompt_chunk_tokens <= 0:
        raise RoutedExpertReadError("prompt_chunk_tokens must be a positive integer")
    if type(top_k) is not int or top_k <= 0:
        raise RoutedExpertReadError("top_k must be a positive integer")
    if type(stage_align_bytes) is not int or stage_align_bytes <= 0:
        raise RoutedExpertReadError("stage_align_bytes must be a positive integer")
    static_capacity_request = _normalize_static_capacity_per_expert(
        static_capacity_per_expert
    )
    if not isinstance(allow_static_capacity_overflow, bool):
        raise RoutedExpertReadError("allow_static_capacity_overflow must be a boolean")

    specs = _load_routed_expert_layer_specs(
        expert_layout_path=expert_layout_path,
        layers=layers,
        purpose="routed stage temp estimate",
    )
    chunks = (prompt_token_count + prompt_chunk_tokens - 1) // prompt_chunk_tokens
    total_stage = 0
    total_compact = 0
    total_static_capacity = 0
    total_static_capacity_overflow = 0
    max_unique = 0
    max_stage = 0
    max_compact = 0
    max_stage_plus_compact = 0
    max_static_capacity = 0
    max_static_capacity_per_expert = 0
    max_static_capacity_overflow = 0
    max_stage_plus_compact_plus_static = 0
    max_chunk_stage = 0
    max_chunk_compact = 0
    max_chunk_stage_plus_compact = 0
    max_chunk_static_capacity = 0
    max_chunk_stage_plus_compact_plus_static = 0
    layer_count = 0
    static_capacity_strict_overflow_safe = True
    remaining = prompt_token_count
    while remaining > 0:
        current = min(prompt_chunk_tokens, remaining)
        chunk_static_capacity = 0
        chunk_stage_plus_compact_plus_static = 0
        chunk_stage = 0
        chunk_compact = 0
        for spec in specs:
            experts = spec.num_experts
            slot_bytes = spec.expert_slot_bytes
            if experts <= 0 or slot_bytes <= 0:
                continue
            unique = min(experts, current * top_k)
            stage = unique * (slot_bytes + stage_align_bytes)
            compact = unique * slot_bytes
            stage_plus_compact = stage + compact
            static_binary = 0
            static_overflow = 0
            capacity_per_expert = _static_capacity_for_chunk(
                static_capacity_request,
                chunk_tokens=current,
            )
            if capacity_per_expert is not None:
                (
                    static_binary,
                    static_overflow,
                    strict_overflow_safe,
                ) = _estimate_static_capacity_binary_bytes(
                    selected_experts=unique,
                    capacity_per_expert=capacity_per_expert,
                    chunk_tokens=current,
                    top_k=top_k,
                    allow_overflow=allow_static_capacity_overflow,
                )
                max_static_capacity_per_expert = max(
                    max_static_capacity_per_expert,
                    capacity_per_expert,
                )
                static_capacity_strict_overflow_safe = (
                    static_capacity_strict_overflow_safe and strict_overflow_safe
                )
            chunk_stage += stage
            chunk_compact += compact
            chunk_static_capacity += static_binary
            total_stage += stage
            total_compact += compact
            total_static_capacity += static_binary
            total_static_capacity_overflow += static_overflow
            max_unique = max(max_unique, unique)
            max_stage = max(max_stage, stage)
            max_compact = max(max_compact, compact)
            max_stage_plus_compact = max(
                max_stage_plus_compact,
                stage_plus_compact,
            )
            max_static_capacity = max(max_static_capacity, static_binary)
            max_static_capacity_overflow = max(
                max_static_capacity_overflow,
                static_overflow,
            )
            max_stage_plus_compact_plus_static = max(
                max_stage_plus_compact_plus_static,
                stage_plus_compact + static_binary,
            )
        chunk_stage_plus_compact = chunk_stage + chunk_compact
        chunk_stage_plus_compact_plus_static = (
            chunk_stage_plus_compact + chunk_static_capacity
        )
        max_chunk_stage = max(max_chunk_stage, chunk_stage)
        max_chunk_compact = max(max_chunk_compact, chunk_compact)
        max_chunk_stage_plus_compact = max(
            max_chunk_stage_plus_compact,
            chunk_stage_plus_compact,
        )
        max_chunk_static_capacity = max(
            max_chunk_static_capacity,
            chunk_static_capacity,
        )
        max_chunk_stage_plus_compact_plus_static = max(
            max_chunk_stage_plus_compact_plus_static,
            chunk_stage_plus_compact_plus_static,
        )
        remaining -= current

    layer_count = sum(
        1
        for spec in specs
        if spec.num_experts > 0 and spec.expert_slot_bytes > 0
    )
    return RoutedStageTempEstimate(
        prompt_chunk_tokens=prompt_chunk_tokens,
        top_k=top_k,
        chunks_per_prompt=chunks,
        layers=layer_count,
        stage_align_bytes=stage_align_bytes,
        static_capacity_per_expert=static_capacity_request,
        allow_static_capacity_overflow=allow_static_capacity_overflow,
        max_static_capacity_per_expert=max_static_capacity_per_expert,
        static_capacity_strict_overflow_safe=static_capacity_strict_overflow_safe,
        max_unique_experts_per_layer=max_unique,
        max_stage_raw_ranges=max_unique,
        max_stage_coalesced_ranges=max_unique,
        max_stage_bytes=max_stage,
        max_compact_stage_bytes=max_compact,
        max_stage_plus_compact_bytes=max_stage_plus_compact,
        max_static_capacity_binary_bytes=max_static_capacity,
        max_static_capacity_overflow_records=max_static_capacity_overflow,
        max_stage_plus_compact_plus_static_bytes=max_stage_plus_compact_plus_static,
        max_chunk_stage_bytes=max_chunk_stage,
        max_chunk_compact_stage_bytes=max_chunk_compact,
        max_chunk_stage_plus_compact_bytes=max_chunk_stage_plus_compact,
        max_chunk_static_capacity_binary_bytes=max_chunk_static_capacity,
        max_chunk_stage_plus_compact_plus_static_bytes=(
            max_chunk_stage_plus_compact_plus_static
        ),
        total_stage_bytes=total_stage,
        total_compact_stage_bytes=total_compact,
        total_stage_plus_compact_bytes=total_stage + total_compact,
        total_static_capacity_binary_bytes=total_static_capacity,
        total_static_capacity_overflow_records=total_static_capacity_overflow,
        total_stage_plus_compact_plus_static_bytes=(
            total_stage + total_compact + total_static_capacity
        ),
    )


def _candidate_chunk_tokens(
    *,
    prompt_token_count: int,
    top_k: int,
    specs: tuple[_RoutedExpertLayerSpec, ...],
    include_chunk_tokens: Iterable[int] | None,
) -> tuple[int, ...]:
    candidates: set[int] = {1, prompt_token_count}
    if include_chunk_tokens is not None:
        for item in include_chunk_tokens:
            if type(item) is int and item > 0:
                candidates.add(min(prompt_token_count, int(item)))
    max_experts = max((spec.num_experts for spec in specs), default=0)
    if max_experts > 0:
        saturation = max(1, (max_experts + top_k - 1) // top_k)
        for item in (
            saturation // 4,
            saturation // 2,
            saturation,
            saturation * 2,
            saturation * 4,
        ):
            if item > 0:
                candidates.add(min(prompt_token_count, item))
    value = 1
    while value < prompt_token_count:
        candidates.add(value)
        value *= 2
    candidates.add(prompt_token_count)
    return tuple(sorted(candidates))


def estimate_routed_prefill_chunk_frontier(
    *,
    expert_layout_path: str | Path,
    prompt_token_count: int,
    top_k: int,
    stage_align_bytes: int,
    layers: Iterable[int] | None = None,
    include_chunk_tokens: Iterable[int] | None = None,
    ssd_read_gib_per_second: float | int = 0.0,
    static_capacity_per_expert: object | None = None,
    allow_static_capacity_overflow: bool = False,
) -> RoutedPrefillChunkFrontier:
    if type(prompt_token_count) is not int or prompt_token_count <= 0:
        raise RoutedExpertReadError("prompt_token_count must be a positive integer")
    if type(top_k) is not int or top_k <= 0:
        raise RoutedExpertReadError("top_k must be a positive integer")
    if type(stage_align_bytes) is not int or stage_align_bytes <= 0:
        raise RoutedExpertReadError("stage_align_bytes must be a positive integer")
    if isinstance(ssd_read_gib_per_second, bool):
        raise RoutedExpertReadError("ssd_read_gib_per_second must be numeric")
    try:
        ssd_gib_s = float(ssd_read_gib_per_second)
    except (TypeError, ValueError) as exc:
        raise RoutedExpertReadError("ssd_read_gib_per_second must be numeric") from exc
    if not math.isfinite(ssd_gib_s):
        raise RoutedExpertReadError("ssd_read_gib_per_second must be finite")
    if ssd_gib_s < 0:
        raise RoutedExpertReadError("ssd_read_gib_per_second must be non-negative")
    static_capacity_request = _normalize_static_capacity_per_expert(
        static_capacity_per_expert
    )
    if not isinstance(allow_static_capacity_overflow, bool):
        raise RoutedExpertReadError("allow_static_capacity_overflow must be a boolean")

    specs = _load_routed_expert_layer_specs(
        expert_layout_path=expert_layout_path,
        layers=layers,
        purpose="routed chunk frontier",
    )
    active_specs = tuple(
        spec for spec in specs if spec.num_experts > 0 and spec.expert_slot_bytes > 0
    )
    saturation_chunk = None
    if active_specs:
        saturation_chunk = max(
            1,
            (max(spec.num_experts for spec in active_specs) + top_k - 1) // top_k,
        )
    chunks = _candidate_chunk_tokens(
        prompt_token_count=prompt_token_count,
        top_k=top_k,
        specs=active_specs,
        include_chunk_tokens=include_chunk_tokens,
    )
    candidates: list[RoutedPrefillChunkCandidate] = []
    baseline_read = 0
    for chunk in chunks:
        read = estimate_routed_expert_read(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            prompt_chunk_tokens=chunk,
            top_k=top_k,
            layers=layers,
        )
        stage = estimate_routed_stage_temp(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            prompt_chunk_tokens=chunk,
            top_k=top_k,
            stage_align_bytes=stage_align_bytes,
            layers=layers,
            static_capacity_per_expert=static_capacity_request,
            allow_static_capacity_overflow=allow_static_capacity_overflow,
        )
        baseline_read = max(baseline_read, read.baseline_read_bytes)
        planned_seconds = (
            read.planned_read_bytes / (ssd_gib_s * 1024**3)
            if ssd_gib_s > 0
            else None
        )
        candidates.append(
            RoutedPrefillChunkCandidate(
                prompt_chunk_tokens=chunk,
                chunks_per_prompt=read.chunks_per_prompt,
                saturates_all_experts_per_layer=(
                    saturation_chunk is not None and chunk >= saturation_chunk
                ),
                planned_read_bytes=read.planned_read_bytes,
                extra_read_bytes=read.extra_read_bytes,
                read_amplification=read.read_amplification,
                max_layer_planned_read_bytes=read.max_layer_planned_read_bytes,
                max_stage_plus_compact_bytes=stage.max_stage_plus_compact_bytes,
                max_chunk_stage_plus_compact_bytes=(
                    stage.max_chunk_stage_plus_compact_bytes
                ),
                total_stage_plus_compact_bytes=stage.total_stage_plus_compact_bytes,
                max_static_capacity_binary_bytes=(
                    stage.max_static_capacity_binary_bytes
                ),
                max_chunk_static_capacity_binary_bytes=(
                    stage.max_chunk_static_capacity_binary_bytes
                ),
                total_static_capacity_binary_bytes=(
                    stage.total_static_capacity_binary_bytes
                ),
                max_stage_plus_compact_plus_static_bytes=(
                    stage.max_stage_plus_compact_plus_static_bytes
                ),
                max_chunk_stage_plus_compact_plus_static_bytes=(
                    stage.max_chunk_stage_plus_compact_plus_static_bytes
                ),
                total_stage_plus_compact_plus_static_bytes=(
                    stage.total_stage_plus_compact_plus_static_bytes
                ),
                planned_read_seconds=planned_seconds,
            )
        )
    return RoutedPrefillChunkFrontier(
        prompt_token_count=prompt_token_count,
        top_k=top_k,
        layers=len(active_specs),
        stage_align_bytes=stage_align_bytes,
        static_capacity_per_expert=static_capacity_request,
        allow_static_capacity_overflow=allow_static_capacity_overflow,
        baseline_read_bytes=baseline_read,
        saturation_chunk_tokens=saturation_chunk,
        candidates=tuple(candidates),
    )


def minimum_prompt_chunk_tokens_for_routed_read_limits(
    *,
    expert_layout_path: str | Path,
    prompt_token_count: int,
    top_k: int,
    layers: Iterable[int] | None = None,
    max_read_amplification: float = 0.0,
    max_planned_read_bytes: int = 0,
) -> int | None:
    if isinstance(max_read_amplification, bool):
        raise RoutedExpertReadError("max_read_amplification must be numeric")
    try:
        max_amp = float(max_read_amplification)
    except (TypeError, ValueError) as exc:
        raise RoutedExpertReadError("max_read_amplification must be numeric") from exc
    if not math.isfinite(max_amp):
        raise RoutedExpertReadError("max_read_amplification must be finite")
    if max_amp < 0:
        raise RoutedExpertReadError("max_read_amplification must be non-negative")
    if type(max_planned_read_bytes) is not int:
        raise RoutedExpertReadError("max_planned_read_bytes must be an integer")
    if max_planned_read_bytes < 0:
        raise RoutedExpertReadError("max_planned_read_bytes must be non-negative")
    if max_amp <= 0 and max_planned_read_bytes <= 0:
        return 1

    def fits(chunk_tokens: int) -> bool:
        estimate = estimate_routed_expert_read(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            prompt_chunk_tokens=chunk_tokens,
            top_k=top_k,
            layers=layers,
        )
        return (
            (max_amp <= 0 or estimate.read_amplification <= max_amp)
            and (
                max_planned_read_bytes <= 0
                or estimate.planned_read_bytes <= max_planned_read_bytes
            )
        )

    if not fits(prompt_token_count):
        return None
    lo = 1
    hi = prompt_token_count
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
