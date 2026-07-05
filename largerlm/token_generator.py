from __future__ import annotations

import json
import math
import operator
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .decode_cache import DecodeCacheError, load_decode_cache_layout
from .decode_driver import DecodeDriverError, DecodeLayerRecord, run_decode_layers
from .embedding import EmbeddingError, embed_token
from .final_logits import (
    FinalLogitsError,
    FinalLogitsResult,
    LogitRecord,
    compute_final_logits,
    compute_final_logits_metal,
    load_final_logits_topk_json,
)
from .generation_guard import (
    GenerationGuardError,
    GenerationRuntimeGuard,
    check_generation_runtime,
    check_live_memory_budget,
    estimate_prompt_prefill_live_memory,
)
from .prefill_execute import (
    AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    AUTO_MPSGRAPH_MIN_DIM,
    PREFILL_LINEAR_BACKENDS,
    PREFILL_LINEAR_F32_CONVERSION_BACKENDS,
    PREFILL_LINEAR_MPSGRAPH_DTYPES,
)
from .prefill_plan import DEFAULT_MPP_MIN_TOKENS, MPP_TENSOR_OPS_MIN_MATRIX_DIM
from .prompt_prefill import (
    PromptPrefillError,
    PromptPrefillResult,
    StaticCapacityPerExpert,
    _normalize_static_capacity_per_expert,
    _reject_unsafe_strict_static_capacity,
    run_prompt_prefill,
)
from .resident_affine import (
    ResidentAffineLayoutError,
    is_affine_int4_weight_dtype,
    is_mxfp4_scale_dtype,
    resident_affine_int4_layout_info,
    resident_layout_tensors_by_name,
    resident_mxfp4_layout_info,
)
from .routed_read import (
    RoutedExpertReadError,
    estimate_routed_expert_read,
    minimum_prompt_chunk_tokens_for_routed_read_limits,
)
from .safety import disk_budget
from .staged_moe import (
    MoEOutputAccumulator,
    MoETokenBlock,
    StagedMoEError,
    _normalize_moe_output_accumulator,
)


class TokenGeneratorError(RuntimeError):
    """Raised when token-id generation cannot run safely."""


def _integer_value(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TokenGeneratorError(f"{label} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TokenGeneratorError(f"{label} must be an integer") from exc


def _nonnegative_integer_value(value: object, *, label: str) -> int:
    parsed = _integer_value(value, label=label)
    if parsed < 0:
        raise TokenGeneratorError(f"{label} must be non-negative")
    return parsed


def _positive_integer_value(value: object, *, label: str) -> int:
    parsed = _integer_value(value, label=label)
    if parsed <= 0:
        raise TokenGeneratorError(f"{label} must be positive")
    return parsed


def _finite_float_value(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TokenGeneratorError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TokenGeneratorError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise TokenGeneratorError(f"{label} must be finite")
    return parsed


def _nonnegative_float_value(value: object, *, label: str) -> float:
    parsed = _finite_float_value(value, label=label)
    if parsed < 0:
        raise TokenGeneratorError(f"{label} must be non-negative")
    return parsed


def _positive_float_value(value: object, *, label: str) -> float:
    parsed = _finite_float_value(value, label=label)
    if parsed <= 0:
        raise TokenGeneratorError(f"{label} must be positive")
    return parsed


def _normalize_token_ids(
    values: Iterable[object],
    *,
    label: str,
) -> tuple[int, ...]:
    try:
        tokens = tuple(_nonnegative_integer_value(token, label=label) for token in values)
    except TypeError as exc:
        raise TokenGeneratorError(f"{label} must be iterable") from exc
    if not tokens:
        raise TokenGeneratorError(f"{label} must be non-empty")
    return tokens


@dataclass(frozen=True)
class GeneratedStep:
    position: int
    input_token_id: int
    selected_token_id: int
    topk: tuple[LogitRecord, ...]
    elapsed_seconds: float = 0.0
    embedding_read_bytes: int = 0
    expert_read_bytes: int = 0
    cache_read_bytes: int = 0
    logits_read_bytes: int = 0
    logits_elapsed_seconds: float = 0.0
    decode_layers: tuple[DecodeLayerRecord, ...] = ()

    @property
    def estimated_read_bytes(self) -> int:
        return (
            self.embedding_read_bytes
            + self.expert_read_bytes
            + self.cache_read_bytes
            + self.logits_read_bytes
        )


@dataclass(frozen=True)
class TokenGenerationResult:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    steps: tuple[GeneratedStep, ...]
    work_dir: Path
    kept_work_dir: bool
    max_context_tokens: int
    sampling_temperature: float
    sampling_top_p: float
    elapsed_seconds: float = 0.0
    estimated_read_bytes: int = 0
    estimated_embedding_read_bytes: int = 0
    estimated_expert_read_bytes: int = 0
    estimated_cache_read_bytes: int = 0
    estimated_logits_read_bytes: int = 0
    runtime_guard: GenerationRuntimeGuard | None = None
    prompt_prefill: PromptPrefillResult | None = None
    auto_prefill_prompt_chunk_plan: AutoPrefillPromptChunkPlan | None = None
    max_safe_prefill_prompt_chunk_plan: AutoPrefillPromptChunkPlan | None = None
    prefill_prompt_chunk_plan_drift: dict[str, object] | None = None
    applied_launch_profile: dict[str, object] | None = None
    prefill_actual_read_time: dict[str, object] | None = None
    prefill_actual_acceleration_coverage: dict[str, object] | None = None
    prefill_actual_acceleration_frontier: dict[str, object] | None = None
    prefill_actual_linear_backend: dict[str, object] | None = None
    decode_actual_read_time: dict[str, object] | None = None


@dataclass(frozen=True)
class AutoPrefillChunkCap:
    name: str
    tokens: int
    bytes_available: int | None = None
    bytes_per_token: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class AutoPrefillExpertStageTilingPlan:
    target_chunk_tokens: int
    target_assignments_per_layer: int
    layer_count: int
    layers_requiring_tiling: int
    max_target_unique_experts_per_layer: int
    max_experts_per_stage_tile: int
    max_stage_tile_count_per_layer: int
    total_stage_tile_count: int
    max_stage_tile_bytes: int
    max_compact_stage_tile_bytes: int
    max_stage_plus_compact_tile_bytes: int
    can_tile_all_layers: bool
    blocker: str | None = None


@dataclass(frozen=True)
class AutoPrefillPromptChunkPlan:
    prompt_tokens: int
    start_position: int
    raw_tokens: int
    chunk_tokens: int
    tile_tokens: int
    limiting_cap_tokens: int
    limiting_caps: tuple[AutoPrefillChunkCap, ...]
    caps: tuple[AutoPrefillChunkCap, ...]
    hidden_dim: int
    per_token_activation_bytes: int
    max_matrix_scratch_bytes: int
    next_token_matrix_scratch_bytes: int | None
    usable_disk_bytes: int
    per_token_disk_bytes: int
    expert_stage_tiling: bool = False
    max_tiled_stage_plus_compact_disk_bytes: int = 0
    mpp_tensor_ops_candidate_min_batch_tokens: int = DEFAULT_MPP_MIN_TOKENS
    mpp_tensor_ops_candidate_min_matrix_dim: int = MPP_TENSOR_OPS_MIN_MATRIX_DIM
    mpp_tensor_ops_dimension_candidate_matrix_count: int = 0
    mpp_tensor_ops_candidate_reachable_under_caps: bool = False
    mpp_tensor_ops_candidate_blocking_cap_names: tuple[str, ...] = ()
    mpp_tensor_ops_candidate_blocking_caps: tuple[AutoPrefillChunkCap, ...] = ()
    mpp_tensor_ops_candidate_blocking_cap_summary: dict[str, int] = field(
        default_factory=dict
    )
    mpp_tensor_ops_candidate_non_expert_blocking_cap_names: tuple[str, ...] = ()
    mpp_tensor_ops_candidate_non_expert_blocking_cap_summary: dict[str, int] = field(
        default_factory=dict
    )
    mpp_tensor_ops_candidate_streamed_stage_disk_cap_tokens: int | None = None
    mpp_tensor_ops_candidate_stage_tiling_plan: (
        AutoPrefillExpertStageTilingPlan | None
    ) = None
    mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable: bool = False
    mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers: tuple[str, ...] = ()


def _chunk_plan_field(plan: object | None, key: str) -> object | None:
    if plan is None:
        return None
    if isinstance(plan, dict):
        return plan.get(key)
    return getattr(plan, key, None)


def _chunk_plan_tokens(plan: object | None) -> int | None:
    value = _chunk_plan_field(plan, "chunk_tokens")
    if type(value) is int:
        return int(value)
    return None


def _chunk_plan_prompt_tokens(plan: object | None) -> int | None:
    value = _chunk_plan_field(plan, "prompt_tokens")
    if type(value) is int:
        return int(value)
    return None


def _chunk_plan_limiting_cap_names(plan: object | None) -> tuple[str, ...]:
    names = _chunk_plan_field(plan, "limiting_cap_names")
    if isinstance(names, (list, tuple)):
        return tuple(str(item) for item in names if isinstance(item, str))
    caps = _chunk_plan_field(plan, "limiting_caps")
    if not isinstance(caps, (list, tuple)):
        return ()
    resolved: list[str] = []
    for cap in caps:
        name = cap.get("name") if isinstance(cap, dict) else getattr(cap, "name", None)
        if isinstance(name, str):
            resolved.append(name)
    return tuple(resolved)


def _profile_prompt_chunk_plan(
    applied_launch_profile: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(applied_launch_profile, dict):
        return None
    plan = applied_launch_profile.get("prefill_prompt_chunk_plan")
    return plan if isinstance(plan, dict) else None


def prefill_prompt_chunk_plan_drift_summary(
    *,
    applied_launch_profile: dict[str, object] | None,
    actual_prompt_chunk_tokens: int | None,
    actual_auto_plan: AutoPrefillPromptChunkPlan | None,
    actual_max_safe_plan: AutoPrefillPromptChunkPlan | None,
) -> dict[str, object] | None:
    profile_plan = _profile_prompt_chunk_plan(applied_launch_profile)
    if profile_plan is None:
        return None

    profile_auto = profile_plan.get("auto")
    if not isinstance(profile_auto, dict):
        profile_auto = None
    profile_max_safe = profile_plan.get("max_safe")
    if not isinstance(profile_max_safe, dict):
        profile_max_safe = None

    profile_auto_tokens = _chunk_plan_tokens(profile_auto)
    profile_max_safe_tokens = _chunk_plan_tokens(profile_max_safe)
    profile_max_safe_prompt_tokens = _chunk_plan_prompt_tokens(profile_max_safe)
    actual_auto_tokens = _chunk_plan_tokens(actual_auto_plan)
    actual_max_safe_tokens = _chunk_plan_tokens(actual_max_safe_plan)
    actual_max_safe_prompt_tokens = _chunk_plan_prompt_tokens(actual_max_safe_plan)
    selected_tokens = (
        int(actual_prompt_chunk_tokens)
        if type(actual_prompt_chunk_tokens) is int
        else None
    )

    max_safe_match = (
        None
        if profile_max_safe_tokens is None or actual_max_safe_tokens is None
        else profile_max_safe_tokens == actual_max_safe_tokens
    )
    current_at_least_profile = (
        None
        if profile_max_safe_tokens is None or actual_max_safe_tokens is None
        else actual_max_safe_tokens >= profile_max_safe_tokens
    )
    selected_within_current = (
        None
        if selected_tokens is None or actual_max_safe_tokens is None
        else selected_tokens <= actual_max_safe_tokens
    )
    current_request_within_profile_prompt = (
        None
        if (
            profile_max_safe_prompt_tokens is None
            or actual_max_safe_prompt_tokens is None
        )
        else actual_max_safe_prompt_tokens < profile_max_safe_prompt_tokens
    )
    if profile_max_safe_tokens is None:
        status = "missing_profile_max_safe_plan"
    elif actual_max_safe_tokens is None:
        status = "missing_actual_max_safe_plan"
    elif selected_tokens is None:
        status = "missing_actual_prompt_chunk"
    elif selected_within_current is False:
        status = "selected_chunk_exceeds_current_max_safe"
    elif (
        current_at_least_profile is False
        and current_request_within_profile_prompt is True
    ):
        status = "shorter_prompt"
    elif current_at_least_profile is False:
        status = "current_max_safe_below_profile"
    elif max_safe_match is False:
        status = "changed"
    else:
        status = "ok"

    return {
        "source": "applied_launch_profile",
        "status": status,
        "profile_auto_chunk_tokens": profile_auto_tokens,
        "profile_max_safe_chunk_tokens": profile_max_safe_tokens,
        "profile_max_safe_prompt_tokens": profile_max_safe_prompt_tokens,
        "actual_prompt_chunk_tokens": selected_tokens,
        "actual_auto_chunk_tokens": actual_auto_tokens,
        "actual_max_safe_chunk_tokens": actual_max_safe_tokens,
        "actual_max_safe_prompt_tokens": actual_max_safe_prompt_tokens,
        "profile_max_safe_limiting_cap_names": (
            _chunk_plan_limiting_cap_names(profile_max_safe)
        ),
        "actual_max_safe_limiting_cap_names": (
            _chunk_plan_limiting_cap_names(actual_max_safe_plan)
        ),
        "max_safe_chunk_tokens_match": max_safe_match,
        "current_max_safe_at_least_profile": current_at_least_profile,
        "current_request_below_profile_prompt_tokens": (
            current_request_within_profile_prompt
        ),
        "selected_chunk_within_current_max_safe": selected_within_current,
    }


def _cache_max_context_tokens(cache_layout_path: str | Path) -> int:
    p = Path(cache_layout_path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TokenGeneratorError(f"failed to read cache layout {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TokenGeneratorError(f"failed to parse cache layout {p}: {exc}") from exc
    value = payload.get("max_context_tokens")
    if type(value) is not int or value <= 0:
        raise TokenGeneratorError("cache layout missing positive max_context_tokens")
    return int(value)


def _recheck_live_memory_guard(runtime_guard: GenerationRuntimeGuard | None) -> None:
    if runtime_guard is None:
        return
    budget = runtime_guard.live_memory_budget
    if budget.min_available_memory_bytes <= 0:
        return
    try:
        check_live_memory_budget(
            estimated_live_working_set_bytes=budget.estimated_live_working_set_bytes,
            max_live_working_set_bytes=budget.max_live_working_set_bytes,
            min_available_memory_bytes=budget.min_available_memory_bytes,
        )
    except GenerationGuardError as exc:
        raise TokenGeneratorError(
            f"live memory guard failed during generation: {exc}"
        ) from exc


def _stop_after_missing_dsa_debug_allow(
    runtime_guard: GenerationRuntimeGuard | None,
) -> bool:
    return bool(
        runtime_guard is not None
        and getattr(runtime_guard, "dsa_index_layers", ())
        and getattr(runtime_guard, "allow_missing_dsa_indexer", False)
        and not getattr(runtime_guard, "dsa_indexer_runtime", False)
    )


def _check_decode_routed_read_guard(
    runtime_guard: GenerationRuntimeGuard,
    *,
    max_read_gib_per_token: float,
    ssd_read_gib_per_second: float,
    max_read_seconds_per_token: float,
) -> None:
    read_bytes = int(runtime_guard.read_bytes_per_token)
    if max_read_gib_per_token > 0.0:
        max_read_bytes = int(max_read_gib_per_token * 1024**3)
        if read_bytes > max_read_bytes:
            raise TokenGeneratorError(
                "decode routed expert read "
                f"{read_bytes} bytes/token exceeds cap {max_read_bytes} "
                "bytes/token"
            )
    if max_read_seconds_per_token > 0.0:
        read_seconds = read_bytes / (ssd_read_gib_per_second * 1024**3)
        if read_seconds > max_read_seconds_per_token:
            raise TokenGeneratorError(
                "decode routed expert read time "
                f"{read_seconds:.3g}s/token exceeds cap "
                f"{max_read_seconds_per_token:.3g}s/token; "
                f"read_bytes_per_token={read_bytes} "
                f"ssd_read_gib_per_second={ssd_read_gib_per_second:.3g}"
            )


def _prefill_stage_io_hotspot_payloads(value: object | None) -> list[dict[str, object]]:
    if value is None:
        return []
    try:
        items = list(value)
    except TypeError:
        return []
    int_fields = (
        "chunk_index",
        "layer",
        "tile_index",
        "batch_tokens",
        "selected_expert_count",
        "total_assignments",
        "raw_range_count",
        "coalesced_range_count",
        "planned_read_bytes",
        "staged_bytes",
        "unique_requested_bytes",
        "waste_bytes",
        "copy_chunk_bytes",
        "copy_read_calls",
        "copy_write_calls",
    )
    float_fields = (
        "copy_elapsed_seconds",
        "copy_throughput_gib_per_second",
        "copy_average_read_bytes",
        "stage_budget_utilization",
        "unique_read_amplification",
    )
    rows: list[dict[str, object]] = []
    for item in items:
        row: dict[str, object] = {}
        for field in int_fields:
            raw = getattr(item, field, None)
            if type(raw) is int and raw >= 0:
                row[field] = raw
        selected_experts = getattr(item, "selected_experts", None)
        if isinstance(selected_experts, (list, tuple)):
            parsed_experts = [
                int(expert)
                for expert in selected_experts
                if type(expert) is int and expert >= 0
            ]
            if len(parsed_experts) == len(selected_experts):
                row["selected_experts"] = parsed_experts
        for field in float_fields:
            raw = getattr(item, field, None)
            if (
                isinstance(raw, (int, float))
                and not isinstance(raw, bool)
                and math.isfinite(float(raw))
            ):
                row[field] = float(raw)
        if row:
            rows.append(row)
    return rows


def _prefill_actual_read_time_summary(
    prompt_prefill: object | None,
) -> dict[str, object] | None:
    if prompt_prefill is None:
        return None
    raw_seconds = getattr(
        prompt_prefill,
        "total_expert_stage_planned_read_seconds",
        None,
    )
    if raw_seconds is None:
        return None
    read_seconds_ok = getattr(
        prompt_prefill,
        "total_expert_stage_read_seconds_ok",
        None,
    )
    copy_seconds_ok = getattr(
        prompt_prefill,
        "total_expert_stage_copy_seconds_ok",
        None,
    )
    summary: dict[str, object] = {
        "source": "generation_actual_prefill",
        "total_expert_stage_serial_read_bytes": int(
            getattr(prompt_prefill, "total_expert_stage_serial_read_bytes", 0) or 0
        ),
        "total_expert_stage_unique_requested_bytes": int(
            getattr(prompt_prefill, "total_expert_stage_unique_requested_bytes", 0)
            or 0
        ),
        "total_expert_stage_planned_read_bytes": int(
            getattr(prompt_prefill, "total_expert_stage_planned_read_bytes", 0) or 0
        ),
        "total_expert_stage_waste_bytes": int(
            getattr(prompt_prefill, "total_expert_stage_waste_bytes", 0) or 0
        ),
        "total_expert_stage_coalesced_savings_bytes": int(
            getattr(
                prompt_prefill,
                "total_expert_stage_coalesced_savings_bytes",
                0,
            )
            or 0
        ),
        "total_expert_stage_planned_read_seconds": float(raw_seconds),
        "prefill_ssd_read_gib_per_second": float(
            getattr(prompt_prefill, "prefill_ssd_read_gib_per_second", 0.0) or 0.0
        ),
        "prefill_max_routed_read_seconds": float(
            getattr(prompt_prefill, "prefill_max_routed_read_seconds", 0.0) or 0.0
        ),
        "total_expert_stage_read_seconds_ok": (
            bool(read_seconds_ok) if read_seconds_ok is not None else None
        ),
        "total_expert_stage_copy_seconds_ok": (
            bool(copy_seconds_ok) if copy_seconds_ok is not None else None
        ),
    }
    for field in (
        "prefill_max_stage_raw_ranges",
        "prefill_max_stage_coalesced_ranges",
        "total_expert_stage_raw_ranges",
        "total_expert_stage_coalesced_ranges",
        "max_expert_stage_raw_ranges",
        "max_expert_stage_coalesced_ranges",
    ):
        value = getattr(prompt_prefill, field, None)
        if type(value) is int:
            summary[field] = value
    for field in (
        "total_expert_stage_raw_ranges_ok",
        "total_expert_stage_coalesced_ranges_ok",
    ):
        value = getattr(prompt_prefill, field, None)
        if value is not None:
            summary[field] = bool(value)
    for field in (
        "total_expert_stage_read_advice_attempted_ranges",
        "total_expert_stage_read_advice_calls",
        "total_expert_stage_read_advice_bytes",
        "total_expert_stage_read_advice_failures",
        "total_expert_stage_copy_read_calls",
        "total_expert_stage_copy_write_calls",
    ):
        value = getattr(prompt_prefill, field, None)
        if type(value) is int:
            summary[field] = value
    for field in (
        "total_expert_stage_assignment_read_amplification",
        "total_expert_stage_unique_read_amplification",
        "max_expert_stage_unique_read_amplification",
        "max_expert_stage_stage_budget_utilization",
        "total_expert_stage_copy_average_read_bytes",
        "total_expert_stage_copy_average_write_bytes",
    ):
        value = getattr(prompt_prefill, field, None)
        if value is not None:
            summary[field] = float(value)
    copy_elapsed = getattr(
        prompt_prefill,
        "total_expert_stage_copy_elapsed_seconds",
        None,
    )
    if copy_elapsed is not None:
        summary["total_expert_stage_copy_elapsed_seconds"] = float(copy_elapsed)
    copy_throughput = getattr(
        prompt_prefill,
        "total_expert_stage_copy_throughput_gib_per_second",
        None,
    )
    if copy_throughput is not None:
        summary["total_expert_stage_copy_throughput_gib_per_second"] = float(
            copy_throughput
        )
    copy_call_counterfactuals = getattr(
        prompt_prefill,
        "total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib",
        None,
    )
    if isinstance(copy_call_counterfactuals, dict):
        parsed: dict[str, int] = {}
        for key, value in copy_call_counterfactuals.items():
            if type(value) is int:
                parsed[str(key)] = value
        if parsed:
            summary[
                "total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib"
            ] = dict(sorted(parsed.items(), key=lambda item: int(item[0])))
    stage_count = getattr(prompt_prefill, "expert_stage_io_stage_count", None)
    if type(stage_count) is int:
        summary["expert_stage_io_stage_count"] = stage_count
    copy_hotspots = _prefill_stage_io_hotspot_payloads(
        getattr(prompt_prefill, "expert_stage_copy_hotspots", None)
    )
    if copy_hotspots:
        summary["expert_stage_copy_hotspots"] = copy_hotspots
    range_hotspots = _prefill_stage_io_hotspot_payloads(
        getattr(prompt_prefill, "expert_stage_range_hotspots", None)
    )
    if range_hotspots:
        summary["expert_stage_range_hotspots"] = range_hotspots
    return summary


def _prefill_actual_acceleration_coverage_summary(
    prompt_prefill: object | None,
    *,
    require_prefill_acceleration: bool = False,
    min_accelerated_flop_fraction: float = 0.0,
    allow_router_gate_only_acceleration: bool = False,
) -> dict[str, object] | None:
    if prompt_prefill is None:
        return None
    coverage = getattr(prompt_prefill, "prefill_acceleration_coverage", None)
    if not isinstance(coverage, dict):
        return None
    summary = dict(coverage)
    summary.setdefault("source", "generation_actual_prefill")
    summary["required"] = bool(
        require_prefill_acceleration or min_accelerated_flop_fraction > 0.0
    )
    summary["allow_router_gate_only_acceleration"] = bool(
        allow_router_gate_only_acceleration
    )
    if min_accelerated_flop_fraction > 0.0:
        summary["min_accelerated_flop_fraction"] = float(
            min_accelerated_flop_fraction
        )
    else:
        summary.setdefault("min_accelerated_flop_fraction", 0.0)
    return summary


def _prefill_actual_acceleration_frontier_summary(
    prompt_prefill: object | None,
) -> dict[str, object] | None:
    if prompt_prefill is None:
        return None
    frontier = getattr(prompt_prefill, "prefill_acceleration_frontier", None)
    if not isinstance(frontier, dict):
        return None
    summary = dict(frontier)
    summary.setdefault("source", "generation_actual_prefill")
    return summary


def _float_mapping(value: object | None) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, (int, float)) and not isinstance(
            item,
            bool,
        ):
            result[key] = float(item)
    return result


def _int_mapping(value: object | None) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(key, str) and type(item) is int:
            result[key] = int(item)
    return result


def _linear_component_stats_summary(value: object | None) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {}
    for component, raw_payload in sorted(value.items()):
        if not isinstance(component, str) or not isinstance(raw_payload, dict):
            continue
        counts = _int_mapping(raw_payload.get("linear_backend_counts"))
        flops = _int_mapping(raw_payload.get("linear_backend_flops"))
        elapsed = _float_mapping(raw_payload.get("linear_backend_elapsed_seconds"))
        if not counts and not flops and not elapsed:
            continue
        payload: dict[str, object] = {}
        if counts:
            payload["linear_backend_counts"] = counts
        if flops:
            payload["linear_backend_flops"] = flops
        if elapsed:
            payload["linear_backend_elapsed_seconds"] = elapsed
        tflops = _float_mapping(raw_payload.get("linear_backend_estimated_tflops"))
        if tflops:
            payload["linear_backend_estimated_tflops"] = tflops
        result[component] = payload
    return result


def _prefill_actual_linear_backend_summary(
    prompt_prefill: object | None,
    *,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int,
    prefill_mpsgraph_min_matrix_dim: int,
) -> dict[str, object] | None:
    if prompt_prefill is None:
        return None
    counts = _int_mapping(getattr(prompt_prefill, "linear_backend_counts", None))
    flops = _int_mapping(getattr(prompt_prefill, "linear_backend_flops", None))
    if not counts and not flops:
        return None
    elapsed = _float_mapping(
        getattr(prompt_prefill, "linear_backend_elapsed_seconds", None)
    )
    tflops = _float_mapping(
        getattr(prompt_prefill, "linear_backend_estimated_tflops", None)
    )
    summary: dict[str, object] = {
        "source": "generation_actual_prefill",
        "configured_backend": str(prefill_linear_backend),
        "auto_policy": {
            "mpsgraph_min_batch_tokens": int(prefill_mpsgraph_min_batch_tokens),
            "mpsgraph_min_matrix_dim": int(prefill_mpsgraph_min_matrix_dim),
        },
        "linear_backend_counts": counts,
        "linear_backend_flops": flops,
        "linear_backend_elapsed_seconds": elapsed,
        "linear_backend_estimated_tflops": tflops,
        "total_linear_estimated_flops": int(
            getattr(prompt_prefill, "total_linear_estimated_flops", 0) or 0
        ),
        "accelerated_linear_estimated_flops": int(
            getattr(prompt_prefill, "accelerated_linear_estimated_flops", 0) or 0
        ),
        "custom_linear_estimated_flops": int(
            getattr(prompt_prefill, "custom_linear_estimated_flops", 0) or 0
        ),
        "unsupported_linear_estimated_flops": int(
            getattr(prompt_prefill, "unsupported_linear_estimated_flops", 0) or 0
        ),
        "accelerated_linear_flop_fraction": float(
            getattr(prompt_prefill, "accelerated_linear_flop_fraction", 0.0) or 0.0
        ),
    }
    component_stats = _linear_component_stats_summary(
        getattr(prompt_prefill, "linear_backend_component_stats", None)
    )
    if component_stats:
        summary["linear_backend_component_stats"] = component_stats
    return summary


def _decode_actual_read_time_summary(
    *,
    steps: tuple[GeneratedStep, ...],
    runtime_guard: GenerationRuntimeGuard | None,
    ssd_read_gib_per_second: float,
    max_read_seconds_per_token: float,
) -> dict[str, object] | None:
    if runtime_guard is None or ssd_read_gib_per_second <= 0.0:
        return None
    decode_steps = tuple(step for step in steps if step.decode_layers)
    if not decode_steps:
        return None
    read_bytes_per_token = int(runtime_guard.read_bytes_per_token)
    if read_bytes_per_token <= 0:
        return None
    actual_read_bytes = sum(int(step.expert_read_bytes) for step in decode_steps)
    planned_read_bytes = read_bytes_per_token * len(decode_steps)
    actual_bytes_ok = actual_read_bytes <= planned_read_bytes
    planned_seconds = planned_read_bytes / (ssd_read_gib_per_second * 1024**3)
    actual_seconds = actual_read_bytes / (ssd_read_gib_per_second * 1024**3)
    total_max_seconds = (
        max_read_seconds_per_token * len(decode_steps)
        if max_read_seconds_per_token > 0.0
        else None
    )
    seconds_ok = (
        actual_seconds <= total_max_seconds
        if total_max_seconds is not None
        else None
    )
    return {
        "source": "generation_actual_decode",
        "decode_step_count": len(decode_steps),
        "decode_read_bytes_per_token": read_bytes_per_token,
        "planned_decode_routed_read_bytes": planned_read_bytes,
        "actual_decode_routed_read_bytes": actual_read_bytes,
        "actual_decode_routed_read_bytes_ok": actual_bytes_ok,
        "planned_decode_routed_read_seconds": planned_seconds,
        "actual_decode_routed_read_seconds": actual_seconds,
        "prefill_ssd_read_gib_per_second": ssd_read_gib_per_second,
        "decode_max_routed_read_seconds_per_token": (
            max_read_seconds_per_token
            if max_read_seconds_per_token > 0.0
            else None
        ),
        "total_decode_max_routed_read_seconds": total_max_seconds,
        "total_decode_routed_read_seconds_ok": seconds_ok,
    }


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _resident_dtype_bytes(dtype: str) -> int:
    normalized = dtype.upper()
    if normalized in {"F32", "FLOAT32"}:
        return 4
    if normalized in {"BF16", "BFLOAT16", "F16", "FLOAT16"}:
        return 2
    return 0


def _resident_affine_companion_weight(
    *,
    tensor_name: str,
    tensors_by_name: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    for suffix in (".scales", ".biases"):
        if tensor_name.endswith(suffix):
            base = tensor_name[: -len(suffix)]
            weight = tensors_by_name.get(f"{base}.weight")
            if isinstance(weight, dict) and is_affine_int4_weight_dtype(
                weight.get("dtype")
            ):
                return weight
    return None


def _resident_mxfp4_scales_companion(
    *,
    weight: dict[str, object],
    tensors_by_name: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if not is_affine_int4_weight_dtype(weight.get("dtype")):
        return None
    weight_name = weight.get("name")
    if not isinstance(weight_name, str) or not weight_name.endswith(".weight"):
        return None
    base = weight_name[: -len(".weight")]
    scales = tensors_by_name.get(f"{base}.scales")
    biases = tensors_by_name.get(f"{base}.biases")
    if (
        isinstance(scales, dict)
        and biases is None
        and is_mxfp4_scale_dtype(scales.get("dtype"))
    ):
        return scales
    return None


def _resident_shape_rank(tensor: dict[str, object]) -> int | None:
    shape = tensor.get("shape")
    if isinstance(shape, (list, tuple)):
        return len(shape)
    return None


def _resident_mxfp4_2d_layout_info(
    resident_layout: dict[str, object],
    weight: dict[str, object],
    *,
    tensors_by_name: dict[str, dict[str, object]],
):
    scales = _resident_mxfp4_scales_companion(
        weight=weight,
        tensors_by_name=tensors_by_name,
    )
    if scales is None:
        return None
    if _resident_shape_rank(weight) != 2 or _resident_shape_rank(scales) != 2:
        return None
    return resident_mxfp4_layout_info(
        resident_layout,
        weight,
        tensors_by_name=tensors_by_name,
    )


def _auto_prefill_layer_count(
    *,
    expert_layout: dict | None,
    layers: Iterable[int] | None,
    dense_layers: Iterable[int] | None,
) -> int:
    if layers is not None:
        selected = {_nonnegative_integer_value(layer, label="layer") for layer in layers}
        return max(1, len(selected))
    selected: set[int] = {
        _nonnegative_integer_value(layer, label="dense layer")
        for layer in dense_layers or ()
    }
    if isinstance(expert_layout, dict):
        raw_layers = expert_layout.get("layers")
        if isinstance(raw_layers, list):
            for item in raw_layers:
                if not isinstance(item, dict):
                    continue
                layer = item.get("layer")
                if type(layer) is not int:
                    raise TokenGeneratorError("expert layout layer must be an integer")
                selected.add(int(layer))
    return max(1, len(selected))


def _auto_prefill_matrix_scratch_bytes(
    *,
    dtype: str,
    rows: int,
    cols: int,
    size: int,
    batch_tokens: int,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
) -> int:
    use_f32_conversion_backend = (
        prefill_linear_backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS
    )
    if prefill_linear_backend == "auto":
        use_f32_conversion_backend = (
            dtype in PREFILL_LINEAR_MPSGRAPH_DTYPES
            and batch_tokens >= prefill_mpsgraph_min_batch_tokens
            and min(rows, cols) >= prefill_mpsgraph_min_matrix_dim
        )
    if use_f32_conversion_backend and dtype in PREFILL_LINEAR_MPSGRAPH_DTYPES:
        converted = rows * cols * 4
        extra_raw = 0 if dtype.upper() in {"F32", "FLOAT32"} else size
        return _align_up(converted, 2 * 1024 * 1024) + extra_raw
    return _align_up(size, 2 * 1024 * 1024)


def _auto_prefill_scratch_cap_tokens(
    *,
    matrix_specs: list[tuple[str, int, int, int]],
    per_token_activation_bytes: int,
    max_runner_scratch_bytes: int,
    prompt_tokens: int,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int,
    prefill_mpsgraph_min_matrix_dim: int,
) -> int:
    if per_token_activation_bytes <= 0:
        return prompt_tokens

    def fits(tokens: int) -> bool:
        max_matrix_scratch = 0
        for dtype, rows, cols, size in matrix_specs:
            max_matrix_scratch = max(
                max_matrix_scratch,
                _auto_prefill_matrix_scratch_bytes(
                    dtype=dtype,
                    rows=rows,
                    cols=cols,
                    size=size,
                    batch_tokens=tokens,
                    prefill_linear_backend=prefill_linear_backend,
                    prefill_mpsgraph_min_batch_tokens=(
                        prefill_mpsgraph_min_batch_tokens
                    ),
                    prefill_mpsgraph_min_matrix_dim=(
                        prefill_mpsgraph_min_matrix_dim
                    ),
                ),
            )
        return (
            max_matrix_scratch + tokens * per_token_activation_bytes
            <= max_runner_scratch_bytes
        )

    lo = 0
    hi = prompt_tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _mpp_candidate_blocking_cap_category(name: str) -> str:
    if name.startswith("expert_stage_layer_"):
        return "expert_stage"
    if name.startswith("cache_read_layer_"):
        return "cache_read"
    if name.startswith("cache_write_layer_"):
        return "cache_write"
    return name


def _mpp_candidate_blocking_cap_summary(
    caps: Iterable[AutoPrefillChunkCap],
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for cap in caps:
        category = _mpp_candidate_blocking_cap_category(cap.name)
        summary[category] = summary.get(category, 0) + 1
    return dict(sorted(summary.items()))


def _mpp_candidate_expert_stage_tiling_plan(
    *,
    expert_layer_specs: Iterable[tuple[int, int, int]],
    top_k: int,
    target_chunk_tokens: int,
    max_stage_bytes: int,
    max_compact_stage_bytes: int,
    expert_stage_align_bytes: int,
) -> AutoPrefillExpertStageTilingPlan | None:
    specs = tuple(expert_layer_specs)
    if not specs:
        return None
    target_assignments = target_chunk_tokens * top_k
    layer_count = 0
    layers_requiring_tiling = 0
    max_target_unique = 0
    max_tile_experts = 0
    max_tile_count = 0
    total_tile_count = 0
    max_stage_tile_bytes = 0
    max_compact_tile_bytes = 0
    blocker: str | None = None
    for _layer, num_experts, slot_bytes in specs:
        layer_count += 1
        stage_bytes_per_unique = slot_bytes + expert_stage_align_bytes
        if stage_bytes_per_unique <= 0 or slot_bytes <= 0:
            blocker = "invalid_expert_stage_bytes"
            continue
        stage_capacity = max_stage_bytes // stage_bytes_per_unique
        compact_capacity = max_compact_stage_bytes // slot_bytes
        tile_capacity = min(num_experts, stage_capacity, compact_capacity)
        target_unique = min(num_experts, target_assignments)
        max_target_unique = max(max_target_unique, target_unique)
        max_tile_experts = max(max_tile_experts, tile_capacity)
        if target_unique <= 0:
            tile_count = 0
        elif tile_capacity <= 0:
            tile_count = 0
            blocker = blocker or "expert_stage_tile_capacity"
        else:
            tile_count = math.ceil(target_unique / tile_capacity)
            if tile_count > 1:
                layers_requiring_tiling += 1
            tile_experts = min(tile_capacity, target_unique)
            max_stage_tile_bytes = max(
                max_stage_tile_bytes,
                tile_experts * stage_bytes_per_unique,
            )
            max_compact_tile_bytes = max(
                max_compact_tile_bytes,
                tile_experts * slot_bytes,
            )
        max_tile_count = max(max_tile_count, tile_count)
        total_tile_count += tile_count
    return AutoPrefillExpertStageTilingPlan(
        target_chunk_tokens=target_chunk_tokens,
        target_assignments_per_layer=target_assignments,
        layer_count=layer_count,
        layers_requiring_tiling=layers_requiring_tiling,
        max_target_unique_experts_per_layer=max_target_unique,
        max_experts_per_stage_tile=max_tile_experts,
        max_stage_tile_count_per_layer=max_tile_count,
        total_stage_tile_count=total_tile_count,
        max_stage_tile_bytes=max_stage_tile_bytes,
        max_compact_stage_tile_bytes=max_compact_tile_bytes,
        max_stage_plus_compact_tile_bytes=(
            max_stage_tile_bytes + max_compact_tile_bytes
        ),
        can_tile_all_layers=blocker is None,
        blocker=blocker,
    )


def _auto_prefill_prompt_chunk_plan(
    *,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    prompt_tokens: int,
    start_position: int,
    layers: Iterable[int] | None,
    dense_layers: Iterable[int] | None,
    work_dir: str | Path | None,
    top_k: int,
    max_prompt_batch_mib: float,
    max_cache_read_mib: float,
    max_cache_write_mib: float,
    max_runner_scratch_mib: float,
    prefill_max_stage_mib: float,
    prefill_max_compact_stage_mib: float,
    prefill_expert_stage_align_kib: float,
    prefill_stage_disk_margin_mib: float,
    dsa_indexer_types: Iterable[str] | None,
    dsa_index_topk: int | None,
    prefill_expert_stage_tiling: bool = False,
    prefill_linear_backend: str = "auto",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    prefill_work_dir: str | Path | None = None,
    tile_tokens: int = 64,
) -> AutoPrefillPromptChunkPlan:
    if prompt_tokens <= 0:
        raise TokenGeneratorError("prompt token count must be positive")
    if start_position < 0:
        raise TokenGeneratorError("start_position must be non-negative")
    context_tokens = start_position + prompt_tokens
    if (
        max_prompt_batch_mib <= 0
        or max_cache_read_mib <= 0
        or max_cache_write_mib <= 0
        or max_runner_scratch_mib <= 0
        or prefill_max_stage_mib <= 0
        or prefill_max_compact_stage_mib <= 0
        or prefill_expert_stage_align_kib <= 0
    ):
        raise TokenGeneratorError("prefill auto chunk memory limits must be positive")
    if prefill_stage_disk_margin_mib < 0:
        raise TokenGeneratorError("prefill_stage_disk_margin_mib must be non-negative")
    if type(prefill_expert_stage_tiling) is not bool:
        raise TokenGeneratorError("prefill_expert_stage_tiling must be a boolean")
    if top_k <= 0:
        raise TokenGeneratorError("top_k must be positive for prefill auto chunk sizing")
    if prefill_linear_backend not in PREFILL_LINEAR_BACKENDS:
        raise TokenGeneratorError(
            "prefill_linear_backend must be custom-metal, mpsgraph-f32, "
            "mps-matrix-f32, or auto"
        )
    prefill_mpsgraph_min_batch_tokens = _positive_integer_value(
        prefill_mpsgraph_min_batch_tokens,
        label="prefill_mpsgraph_min_batch_tokens",
    )
    prefill_mpsgraph_min_matrix_dim = _positive_integer_value(
        prefill_mpsgraph_min_matrix_dim,
        label="prefill_mpsgraph_min_matrix_dim",
    )

    try:
        layout = json.loads(Path(resident_layout_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise TokenGeneratorError(
            f"failed to read resident layout for prefill auto chunk sizing: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TokenGeneratorError(
            f"failed to parse resident layout for prefill auto chunk sizing: {exc}"
        ) from exc
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise TokenGeneratorError(
            "resident layout missing tensors array for prefill auto chunk sizing"
        )
    try:
        tensors_by_name = resident_layout_tensors_by_name(layout)
    except ResidentAffineLayoutError as exc:
        raise TokenGeneratorError(str(exc)) from exc

    hidden_dim = 0
    matrix_specs: list[tuple[str, int, int, int]] = []
    per_token_activation_bytes = 0
    affine_companion_names: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        if not isinstance(name, str):
            continue
        if ".layers." not in name:
            mxfp4 = _resident_mxfp4_2d_layout_info(
                layout,
                tensor,
                tensors_by_name=tensors_by_name,
            )
            if mxfp4 is not None:
                rows, cols = mxfp4.out_dim, mxfp4.in_dim
            else:
                shape = tensor.get("shape")
                if not isinstance(shape, list) or len(shape) < 2:
                    continue
                if type(shape[0]) is not int or type(shape[1]) is not int:
                    raise TokenGeneratorError(
                        f"resident tensor {name} shape must use integer rows and cols"
                    )
                rows, cols = int(shape[0]), int(shape[1])
            if rows > 0 and cols > 0 and name.endswith(".embed_tokens.weight"):
                hidden_dim = cols
                continue
            continue
        if name in affine_companion_names:
            continue
        companion_weight = _resident_affine_companion_weight(
            tensor_name=name,
            tensors_by_name=tensors_by_name,
        )
        if companion_weight is not None:
            if _resident_mxfp4_scales_companion(
                weight=companion_weight,
                tensors_by_name=tensors_by_name,
            ) is not None:
                affine_companion_names.add(name)
                continue
            try:
                affine = resident_affine_int4_layout_info(
                    layout,
                    companion_weight,
                    tensors_by_name=tensors_by_name,
                )
            except ResidentAffineLayoutError as exc:
                raise TokenGeneratorError(str(exc)) from exc
            if affine is not None:
                affine_companion_names.add(name)
                continue

        try:
            mxfp4 = _resident_mxfp4_2d_layout_info(
                layout,
                tensor,
                tensors_by_name=tensors_by_name,
            )
        except ResidentAffineLayoutError as exc:
            raise TokenGeneratorError(str(exc)) from exc
        if mxfp4 is not None:
            rows, cols = mxfp4.out_dim, mxfp4.in_dim
            size = mxfp4.total_bytes
            dtype = "mlx-mxfp4"
            scale_name = mxfp4.scales.get("name")
            if isinstance(scale_name, str):
                affine_companion_names.add(scale_name)
        elif _resident_mxfp4_scales_companion(
            weight=tensor,
            tensors_by_name=tensors_by_name,
        ) is not None:
            continue
        else:
            try:
                affine = resident_affine_int4_layout_info(
                    layout,
                    tensor,
                    tensors_by_name=tensors_by_name,
                )
            except ResidentAffineLayoutError as exc:
                raise TokenGeneratorError(str(exc)) from exc
            if affine is not None:
                rows, cols = affine.out_dim, affine.in_dim
                size = affine.total_bytes
                dtype = "affine-int4"
                for companion in (affine.scales, affine.biases):
                    companion_name = companion.get("name")
                    if isinstance(companion_name, str):
                        affine_companion_names.add(companion_name)
            else:
                shape = tensor.get("shape")
                if not isinstance(shape, list) or len(shape) < 2:
                    continue
                if type(shape[0]) is not int or type(shape[1]) is not int:
                    raise TokenGeneratorError(
                        f"resident tensor {name} shape must use integer rows and cols"
                    )
                rows, cols = int(shape[0]), int(shape[1])
                dtype = str(tensor.get("dtype") or "")
                dtype_bytes = _resident_dtype_bytes(dtype)
                if dtype_bytes == 0:
                    continue
                raw_size = tensor.get("size")
                size = (
                    rows * cols * dtype_bytes
                    if raw_size is None
                    else _positive_integer_value(
                        raw_size,
                        label=f"resident tensor {name} size",
                    )
                )
        if rows <= 0 or cols <= 0:
            continue
        matrix_specs.append((dtype, rows, cols, size))
        per_token = (rows + cols) * 4
        if ".mlp." in name:
            per_token = max(per_token, cols * 4 + 2 * rows * 4)
        per_token_activation_bytes = max(per_token_activation_bytes, per_token)

    if hidden_dim <= 0:
        raise TokenGeneratorError(
            "resident layout missing embed_tokens hidden dim for prefill auto chunk sizing"
        )
    per_token_activation_bytes = max(per_token_activation_bytes, hidden_dim * 4)
    mpp_dimension_candidate_matrix_count = sum(
        1
        for _dtype, rows, cols, _size in matrix_specs
        if rows >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
        and cols >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
    )

    max_prompt_batch_bytes = int(max_prompt_batch_mib * 1024**2)
    max_cache_read_bytes = int(max_cache_read_mib * 1024**2)
    max_cache_write_bytes = int(max_cache_write_mib * 1024**2)
    max_runner_scratch_bytes = int(max_runner_scratch_mib * 1024**2)
    max_stage_bytes = int(prefill_max_stage_mib * 1024**2)
    max_compact_stage_bytes = int(prefill_max_compact_stage_mib * 1024**2)
    expert_stage_align_bytes = int(prefill_expert_stage_align_kib * 1024)
    disk_margin_bytes = int(prefill_stage_disk_margin_mib * 1024**2)

    cap_items: list[AutoPrefillChunkCap] = [
        AutoPrefillChunkCap(name="prompt_tokens", tokens=prompt_tokens)
    ]
    cap_items.append(
        AutoPrefillChunkCap(
            name="prompt_batch_bytes",
            tokens=max_prompt_batch_bytes // (hidden_dim * 4),
            bytes_available=max_prompt_batch_bytes,
            bytes_per_token=hidden_dim * 4,
        )
    )
    scratch_cap_tokens = _auto_prefill_scratch_cap_tokens(
        matrix_specs=matrix_specs,
        per_token_activation_bytes=per_token_activation_bytes,
        max_runner_scratch_bytes=max_runner_scratch_bytes,
        prompt_tokens=prompt_tokens,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
    )

    def matrix_scratch_for_tokens(tokens: int) -> int:
        max_scratch = 0
        for dtype, rows, cols, size in matrix_specs:
            max_scratch = max(
                max_scratch,
                _auto_prefill_matrix_scratch_bytes(
                    dtype=dtype,
                    rows=rows,
                    cols=cols,
                    size=size,
                    batch_tokens=max(1, tokens),
                    prefill_linear_backend=prefill_linear_backend,
                    prefill_mpsgraph_min_batch_tokens=(
                        prefill_mpsgraph_min_batch_tokens
                    ),
                    prefill_mpsgraph_min_matrix_dim=(
                        prefill_mpsgraph_min_matrix_dim
                    ),
                ),
            )
        return max_scratch

    cap_items.append(
        AutoPrefillChunkCap(
            name="runner_scratch_bytes",
            tokens=scratch_cap_tokens,
            bytes_available=max_runner_scratch_bytes,
            bytes_per_token=per_token_activation_bytes,
            detail="includes resident matrix scratch",
        )
    )

    selected_layers = (
        {_nonnegative_integer_value(layer, label="layer") for layer in layers}
        if layers is not None
        else None
    )
    try:
        expert_layout = json.loads(Path(expert_layout_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        expert_layout = None
    layer_count = _auto_prefill_layer_count(
        expert_layout=expert_layout,
        layers=layers,
        dense_layers=dense_layers,
    )
    expert_layer_specs: list[tuple[int, int, int]] = []
    if isinstance(expert_layout, dict):
        raw_layers = expert_layout.get("layers")
        if isinstance(raw_layers, list):
            for item in raw_layers:
                if not isinstance(item, dict):
                    continue
                layer = item.get("layer")
                if type(layer) is not int:
                    raise TokenGeneratorError("expert layout layer must be an integer")
                if selected_layers is not None and layer not in selected_layers:
                    continue
                num_experts = _positive_integer_value(
                    item.get("num_experts"),
                    label="expert layout num_experts",
                )
                slot_bytes = _positive_integer_value(
                    item.get("expert_slot_bytes"),
                    label="expert layout expert_slot_bytes",
                )
                if num_experts > 0 and slot_bytes > 0:
                    expert_layer_specs.append((layer, num_experts, slot_bytes))
    routed_stage_and_compact_disk_bytes_per_token = sum(
        min(top_k, num_experts) * (2 * slot_bytes + expert_stage_align_bytes)
        for _layer, num_experts, slot_bytes in expert_layer_specs
    )
    per_token_work_bytes = hidden_dim * 4 * (1 + 2 * layer_count)
    resolved_prefill_work_dir = (
        Path(prefill_work_dir)
        if prefill_work_dir is not None
        else Path(work_dir) / "prefill_prompt"
        if work_dir is not None
        else Path("/private/tmp")
    )
    budget = disk_budget(
        resolved_prefill_work_dir,
        0,
        safety_margin_bytes=disk_margin_bytes,
    )
    usable_disk_bytes = budget.available_bytes - disk_margin_bytes
    worst_case_stage_tiling_plan = (
        _mpp_candidate_expert_stage_tiling_plan(
            expert_layer_specs=expert_layer_specs,
            top_k=top_k,
            target_chunk_tokens=prompt_tokens,
            max_stage_bytes=max_stage_bytes,
            max_compact_stage_bytes=max_compact_stage_bytes,
            expert_stage_align_bytes=expert_stage_align_bytes,
        )
        if prefill_expert_stage_tiling
        else None
    )
    max_tiled_stage_plus_compact_disk_bytes = (
        worst_case_stage_tiling_plan.max_stage_plus_compact_tile_bytes
        if worst_case_stage_tiling_plan is not None
        and worst_case_stage_tiling_plan.can_tile_all_layers
        else 0
    )
    effective_stage_disk_bytes_per_token = (
        0
        if prefill_expert_stage_tiling
        and max_tiled_stage_plus_compact_disk_bytes > 0
        else routed_stage_and_compact_disk_bytes_per_token
    )
    per_token_disk_bytes = per_token_work_bytes + effective_stage_disk_bytes_per_token
    disk_tokens = usable_disk_bytes // per_token_disk_bytes
    if max_tiled_stage_plus_compact_disk_bytes > 0:
        if usable_disk_bytes < max_tiled_stage_plus_compact_disk_bytes:
            disk_tokens = 0
        else:
            disk_tokens = (
                (usable_disk_bytes - max_tiled_stage_plus_compact_disk_bytes)
                // per_token_work_bytes
                if per_token_work_bytes > 0
                else prompt_tokens
            )
    cap_items.append(
        AutoPrefillChunkCap(
            name="work_dir_disk_bytes",
            tokens=disk_tokens,
            bytes_available=usable_disk_bytes,
            bytes_per_token=per_token_disk_bytes,
            detail=(
                "streamed_tiled_stage"
                if max_tiled_stage_plus_compact_disk_bytes > 0
                else None
            ),
        )
    )

    dsa_types = tuple(str(item).lower() for item in dsa_indexer_types or ())
    try:
        cache_layout = load_decode_cache_layout(cache_layout_path)
    except DecodeCacheError as exc:
        raise TokenGeneratorError(
            f"failed to load cache layout for prefill auto chunk sizing: {exc}"
        ) from exc
    for segment in cache_layout.segments:
        if selected_layers is not None and segment.layer not in selected_layers:
            continue
        if segment.kind == "mla_kv":
            dsa_mode = (
                dsa_types[segment.layer]
                if segment.layer < len(dsa_types)
                else "none"
            )
            indexed = dsa_mode in {"full", "shared"} and dsa_index_topk is not None
            read_rows = (
                min(int(dsa_index_topk or 0), context_tokens)
                if indexed
                else context_tokens
            )
            if read_rows > 0:
                cap_items.append(
                    AutoPrefillChunkCap(
                        name=f"cache_read_layer_{segment.layer}_{segment.kind}",
                        tokens=(
                            max_cache_read_bytes
                            // (read_rows * segment.token_stride_bytes)
                        ),
                        bytes_available=max_cache_read_bytes,
                        bytes_per_token=read_rows * segment.token_stride_bytes,
                    )
                )
            cap_items.append(
                AutoPrefillChunkCap(
                    name=f"cache_write_layer_{segment.layer}_{segment.kind}",
                    tokens=max_cache_write_bytes // segment.token_stride_bytes,
                    bytes_available=max_cache_write_bytes,
                    bytes_per_token=segment.token_stride_bytes,
                )
            )
        elif segment.kind == "dsa_index":
            dsa_mode = (
                dsa_types[segment.layer]
                if segment.layer < len(dsa_types)
                else "none"
            )
            if dsa_mode == "full":
                cap_items.append(
                    AutoPrefillChunkCap(
                        name=f"cache_read_layer_{segment.layer}_{segment.kind}",
                        tokens=(
                            max_cache_read_bytes
                            // (context_tokens * segment.token_stride_bytes)
                        ),
                        bytes_available=max_cache_read_bytes,
                        bytes_per_token=context_tokens * segment.token_stride_bytes,
                    )
                )
                cap_items.append(
                    AutoPrefillChunkCap(
                        name=f"cache_write_layer_{segment.layer}_{segment.kind}",
                        tokens=max_cache_write_bytes // segment.token_stride_bytes,
                        bytes_available=max_cache_write_bytes,
                        bytes_per_token=segment.token_stride_bytes,
                    )
                )

    for layer, num_experts, slot_bytes in expert_layer_specs:
        stage_bytes_per_unique = slot_bytes + expert_stage_align_bytes
        max_stage_unique = max_stage_bytes // stage_bytes_per_unique
        max_compact_unique = max_compact_stage_bytes // slot_bytes
        max_unique = min(num_experts, max_stage_unique, max_compact_unique)
        if prefill_expert_stage_tiling:
            if max_unique <= 0:
                cap_items.append(
                    AutoPrefillChunkCap(
                        name=f"expert_stage_layer_{layer}",
                        tokens=0,
                        bytes_available=min(max_stage_bytes, max_compact_stage_bytes),
                        bytes_per_token=top_k * slot_bytes,
                        detail="stage_tile_capacity",
                    )
                )
            continue
        if max_unique < num_experts:
            cap_items.append(
                AutoPrefillChunkCap(
                    name=f"expert_stage_layer_{layer}",
                    tokens=max_unique // top_k,
                    bytes_available=min(max_stage_bytes, max_compact_stage_bytes),
                    bytes_per_token=top_k * slot_bytes,
                    detail=(
                        "stage"
                        if max_stage_unique <= max_compact_unique
                        else "compact_stage"
                    ),
                )
            )

    raw_tokens = min(item.tokens for item in cap_items)
    if raw_tokens <= 0:
        raise TokenGeneratorError(
            "auto prefill prompt chunk cannot fit one token under the configured "
            "prompt/cache/scratch/disk limits"
        )
    chunk_tokens = min(prompt_tokens, raw_tokens)
    if chunk_tokens >= tile_tokens:
        tiled_chunk_tokens = (chunk_tokens // tile_tokens) * tile_tokens
        if tiled_chunk_tokens * 4 >= chunk_tokens * 3:
            chunk_tokens = tiled_chunk_tokens
    chunk_tokens = max(1, chunk_tokens)
    max_matrix_scratch_bytes = matrix_scratch_for_tokens(chunk_tokens)
    next_token_matrix_scratch_bytes = (
        matrix_scratch_for_tokens(chunk_tokens + 1)
        if chunk_tokens < prompt_tokens
        else None
    )
    limiting_caps = tuple(item for item in cap_items if item.tokens == raw_tokens)
    mpp_candidate_reachable = (
        mpp_dimension_candidate_matrix_count > 0
        and chunk_tokens >= DEFAULT_MPP_MIN_TOKENS
    )
    mpp_candidate_blocking_caps = (
        tuple(item for item in cap_items if item.tokens < DEFAULT_MPP_MIN_TOKENS)
        if mpp_dimension_candidate_matrix_count > 0 and not mpp_candidate_reachable
        else ()
    )
    mpp_non_expert_blocking_caps = tuple(
        item
        for item in mpp_candidate_blocking_caps
        if _mpp_candidate_blocking_cap_category(item.name) != "expert_stage"
    )
    streamed_stage_disk_cap_tokens = (
        (
            (usable_disk_bytes - max_tiled_stage_plus_compact_disk_bytes)
            // per_token_work_bytes
        )
        if max_tiled_stage_plus_compact_disk_bytes > 0
        and usable_disk_bytes >= max_tiled_stage_plus_compact_disk_bytes
        else usable_disk_bytes // per_token_work_bytes
        if per_token_work_bytes > 0
        else None
    )
    mpp_stage_tiling_plan = (
        _mpp_candidate_expert_stage_tiling_plan(
            expert_layer_specs=expert_layer_specs,
            top_k=top_k,
            target_chunk_tokens=DEFAULT_MPP_MIN_TOKENS,
            max_stage_bytes=max_stage_bytes,
            max_compact_stage_bytes=max_compact_stage_bytes,
            expert_stage_align_bytes=expert_stage_align_bytes,
        )
        if mpp_dimension_candidate_matrix_count > 0
        else None
    )
    stage_tiling_blockers: list[str] = []
    if mpp_dimension_candidate_matrix_count <= 0:
        stage_tiling_blockers.append("mpp_tensor_ops_dimension_candidate")
    if prompt_tokens < DEFAULT_MPP_MIN_TOKENS:
        stage_tiling_blockers.append("prompt_tokens")
    if (
        mpp_stage_tiling_plan is not None
        and not mpp_stage_tiling_plan.can_tile_all_layers
    ):
        stage_tiling_blockers.append("expert_stage_tiling")
    for cap in mpp_non_expert_blocking_caps:
        if cap.name == "work_dir_disk_bytes":
            if (
                streamed_stage_disk_cap_tokens is None
                or streamed_stage_disk_cap_tokens < DEFAULT_MPP_MIN_TOKENS
            ):
                stage_tiling_blockers.append("work_dir_streamed_stage_disk_bytes")
            continue
        stage_tiling_blockers.append(cap.name)
    mpp_stage_tiling_counterfactual_reachable = (
        mpp_dimension_candidate_matrix_count > 0
        and not stage_tiling_blockers
    )
    return AutoPrefillPromptChunkPlan(
        prompt_tokens=prompt_tokens,
        start_position=start_position,
        raw_tokens=raw_tokens,
        chunk_tokens=chunk_tokens,
        tile_tokens=tile_tokens,
        limiting_cap_tokens=raw_tokens,
        limiting_caps=limiting_caps,
        caps=tuple(cap_items),
        hidden_dim=hidden_dim,
        per_token_activation_bytes=per_token_activation_bytes,
        max_matrix_scratch_bytes=max_matrix_scratch_bytes,
        next_token_matrix_scratch_bytes=next_token_matrix_scratch_bytes,
        usable_disk_bytes=usable_disk_bytes,
        per_token_disk_bytes=per_token_disk_bytes,
        expert_stage_tiling=prefill_expert_stage_tiling,
        max_tiled_stage_plus_compact_disk_bytes=(
            max_tiled_stage_plus_compact_disk_bytes
        ),
        mpp_tensor_ops_candidate_min_batch_tokens=DEFAULT_MPP_MIN_TOKENS,
        mpp_tensor_ops_candidate_min_matrix_dim=MPP_TENSOR_OPS_MIN_MATRIX_DIM,
        mpp_tensor_ops_dimension_candidate_matrix_count=(
            mpp_dimension_candidate_matrix_count
        ),
        mpp_tensor_ops_candidate_reachable_under_caps=mpp_candidate_reachable,
        mpp_tensor_ops_candidate_blocking_cap_names=tuple(
            item.name for item in mpp_candidate_blocking_caps
        ),
        mpp_tensor_ops_candidate_blocking_caps=mpp_candidate_blocking_caps,
        mpp_tensor_ops_candidate_blocking_cap_summary=(
            _mpp_candidate_blocking_cap_summary(mpp_candidate_blocking_caps)
        ),
        mpp_tensor_ops_candidate_non_expert_blocking_cap_names=tuple(
            item.name for item in mpp_non_expert_blocking_caps
        ),
        mpp_tensor_ops_candidate_non_expert_blocking_cap_summary=(
            _mpp_candidate_blocking_cap_summary(mpp_non_expert_blocking_caps)
        ),
        mpp_tensor_ops_candidate_streamed_stage_disk_cap_tokens=(
            streamed_stage_disk_cap_tokens
        ),
        mpp_tensor_ops_candidate_stage_tiling_plan=mpp_stage_tiling_plan,
        mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable=(
            mpp_stage_tiling_counterfactual_reachable
        ),
        mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers=tuple(
            dict.fromkeys(stage_tiling_blockers)
        ),
    )


def _auto_prefill_prompt_chunk_tokens(**kwargs: object) -> int:
    return _auto_prefill_prompt_chunk_plan(**kwargs).chunk_tokens


def _metal_final_logits_result_from_topk(
    *,
    resident_layout_path: str | Path,
    input_f32_path: str | Path,
    topk_json_path: str | Path,
    runtime_guard: GenerationRuntimeGuard,
    top_k: int,
) -> FinalLogitsResult:
    budget = runtime_guard.final_logits_budget
    return FinalLogitsResult(
        resident_layout_path=Path(resident_layout_path),
        input_path=Path(input_f32_path),
        output_logits_path=None,
        output_topk_path=None,
        norm_tensor=budget.norm_tensor,
        head_tensor=budget.head_tensor,
        hidden_dim=budget.hidden_dim,
        vocab_size=budget.vocab_size,
        dtype=budget.dtype,
        chunk_rows=budget.chunk_rows,
        chunks=budget.chunks,
        read_bytes=budget.read_bytes,
        topk=load_final_logits_topk_json(topk_json_path, top_k=top_k),
    )


def _metal_final_logits_elapsed_from_topk(path: str | Path) -> float | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("elapsed_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    elapsed = float(value)
    if not math.isfinite(elapsed) or elapsed < 0:
        return None
    return elapsed


def _select_token(
    records: tuple[LogitRecord, ...],
    *,
    temperature: float,
    top_p: float,
    rng: random.Random | None,
) -> int:
    if not records:
        raise TokenGeneratorError("logits top-k result is empty")
    if temperature < 0 or not math.isfinite(temperature):
        raise TokenGeneratorError("temperature must be finite and non-negative")
    if top_p <= 0.0 or top_p > 1.0 or not math.isfinite(top_p):
        raise TokenGeneratorError("top_p must be in (0, 1]")
    if temperature == 0.0 or len(records) == 1:
        return records[0].token_id
    if rng is None:
        rng = random.Random()

    max_logit = max(record.logit for record in records)
    weighted: list[tuple[int, float]] = []
    for record in records:
        weight = math.exp((record.logit - max_logit) / temperature)
        if weight > 0.0 and math.isfinite(weight):
            weighted.append((record.token_id, weight))
    if not weighted:
        return records[0].token_id

    total = sum(weight for _, weight in weighted)
    if top_p < 1.0:
        kept: list[tuple[int, float]] = []
        cumulative = 0.0
        for token_id, weight in weighted:
            kept.append((token_id, weight))
            cumulative += weight / total
            if cumulative >= top_p:
                break
        weighted = kept
        total = sum(weight for _, weight in weighted)

    draw = rng.random() * total
    cumulative_weight = 0.0
    for token_id, weight in weighted:
        cumulative_weight += weight
        if draw <= cumulative_weight:
            return token_id
    return weighted[-1][0]


def _prompt_prefill_read_estimates(result: PromptPrefillResult) -> tuple[int, int]:
    expert_read = 0
    cache_read = 0
    for chunk in result.chunks:
        for layer in chunk.layers:
            cache_read += layer.attention.mla_attention.cache_read_bytes
            if layer.staged_mlp is not None:
                expert_read += layer.staged_mlp.stage_result.planned_read_bytes
                expert_read += layer.staged_mlp.staged_moe.compact_stage_bytes
    return expert_read, cache_read


def _routed_read_chunk_suggestion(
    *,
    expert_layout_path: str | Path,
    prompt_token_count: int,
    top_k: int,
    layers: Iterable[int] | None,
    max_read_amplification: float,
    max_planned_read_bytes: int,
) -> str:
    try:
        minimum_chunk = minimum_prompt_chunk_tokens_for_routed_read_limits(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            top_k=top_k,
            layers=layers,
            max_read_amplification=max_read_amplification,
            max_planned_read_bytes=max_planned_read_bytes,
        )
    except RoutedExpertReadError:
        return ""
    if minimum_chunk is None:
        return "; no prompt chunk size can satisfy these routed-read caps"
    return f"; use prefill_prompt_chunk_tokens>={minimum_chunk} or relax the cap"


def generate_token_ids(
    *,
    runner_path: str | Path,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    prompt_token_ids: Iterable[int],
    max_new_tokens: int,
    layers: Iterable[int] | None = None,
    dense_layers: Iterable[int] | None = None,
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
    eos_token_id: int | None = None,
    eos_token_ids: Iterable[int] | None = None,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    prefill_mla_kv_b_cache_dir: str | Path | None = None,
    prefill_mla_key_cache: bool = False,
    decode_mla_key_cache: bool = False,
    kv_lora_dim: int | None = None,
    cache_position_offset: int = 0,
    attention_scale: float | None = None,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    top_k: int = 8,
    max_k: int = 8,
    router_score: str = "sigmoid",
    routed_scaling_factor: float | None = None,
    norm_topk_prob: bool = False,
    no_norm_topk_prob: bool = False,
    router_n_group: int | None = None,
    router_topk_group: int | None = None,
    ignore_router_bias: bool = False,
    include_shared_expert: bool = False,
    rms_norm_eps: float = 1e-5,
    logits_top_k: int = 1,
    logits_chunk_rows: int | None = None,
    logits_max_chunk_mib: float = 64.0,
    max_slot_mib: float = 256.0,
    max_router_mib: float = 64.0,
    max_resident_matrix_mib: float = 512.0,
    max_cache_file_mib: float = 32768.0,
    max_cache_read_mib: float = 256.0,
    decode_max_routed_read_gib_per_token: float = 0.0,
    decode_max_routed_read_seconds_per_token: float = 0.0,
    max_runner_scratch_mib: float = 4096.0,
    max_live_working_set_mib: float | None = 8192.0,
    min_free_unified_memory_gib: float = 0.0,
    expert_read_advise_merge_gap_kib: int = 0,
    expert_read_advise_align_kib: int = 0,
    cache_dtype_bytes: int = 2,
    batch_prefill_prompt: bool = False,
    prefill_prompt_chunk_tokens: int = 64,
    prefill_max_prompt_batch_mib: float = 1024.0,
    prefill_max_cache_write_mib: float = 4096.0,
    prefill_expert_stage_merge_gap_kib: float = 0.0,
    prefill_expert_stage_align_kib: float = 4.0,
    prefill_max_stage_mib: float = 4096.0,
    prefill_max_compact_stage_mib: float = 4096.0,
    prefill_max_stage_raw_ranges: int = 0,
    prefill_max_stage_coalesced_ranges: int = 0,
    prefill_expert_stage_tiling: bool = False,
    prefill_persistent_moe_plan_server: bool = False,
    prefill_persistent_resident_linear_server: bool = False,
    prefill_persistent_attention_projection_server: bool = False,
    prefill_persistent_attention_output_server: bool = False,
    prefill_persistent_shared_expert_server: bool = False,
    prefill_persistent_rope_split_server: bool = False,
    prefill_persistent_mla_attention_server: bool = False,
    prefill_persistent_rmsnorm_server: bool = False,
    prefill_copy_chunk_mib: float = 8.0,
    prefill_stage_disk_margin_mib: float = 0.0,
    prefill_max_routed_read_amplification: float = 0.0,
    prefill_max_routed_read_gib: float = 0.0,
    prefill_ssd_read_gib_per_second: float = 0.0,
    prefill_max_routed_read_seconds: float = 0.0,
    prefill_moe_token_block: MoETokenBlock = "auto",
    prefill_moe_output_accumulator: MoEOutputAccumulator = "env",
    prefill_static_capacity_per_expert: StaticCapacityPerExpert = None,
    prefill_allow_static_capacity_overflow: bool = False,
    prefill_linear_backend: str = "auto",
    require_prefill_acceleration: bool = False,
    allow_router_gate_only_prefill_acceleration: bool = False,
    prefill_min_accelerated_flop_fraction: float = 0.0,
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    prefill_router_hybrid_margin_threshold: float = 0.0,
    dsa_indexer_types: Iterable[str] | None = None,
    dsa_index_topk: int | None = None,
    dsa_index_n_heads: int | None = None,
    dsa_index_head_dim: int | None = None,
    dsa_qk_rope_dim: int | None = None,
    dsa_rope_interleave: bool = False,
    dsa_layer_norm_eps: float = 1e-6,
    max_embedding_row_mib: float = 64.0,
    echo_runner_output: bool = True,
    sampling_temperature: float = 0.0,
    sampling_top_p: float = 1.0,
    sampling_seed: int | None = None,
    metal_final_logits: bool = False,
    allow_tied_embeddings: bool = True,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
    preflight_runtime: bool = True,
    allow_missing_dsa_indexer: bool = False,
) -> TokenGenerationResult:
    prompt = _normalize_token_ids(prompt_token_ids, label="prompt_token_ids")
    max_new_tokens = _nonnegative_integer_value(
        max_new_tokens,
        label="max_new_tokens",
    )
    num_heads = _positive_integer_value(num_heads, label="num_heads")
    qk_nope_dim = _positive_integer_value(qk_nope_dim, label="qk_nope_dim")
    rope_dim = _positive_integer_value(rope_dim, label="rope_dim")
    v_head_dim = _positive_integer_value(v_head_dim, label="v_head_dim")
    if type(decode_mla_key_cache) is not bool:
        raise TokenGeneratorError("decode_mla_key_cache must be a boolean")
    if kv_lora_dim is not None:
        kv_lora_dim = _positive_integer_value(kv_lora_dim, label="kv_lora_dim")
    if (
        isinstance(prefill_min_accelerated_flop_fraction, bool)
        or not isinstance(prefill_min_accelerated_flop_fraction, (int, float))
        or not 0.0 <= float(prefill_min_accelerated_flop_fraction) <= 1.0
    ):
        raise TokenGeneratorError(
            "prefill_min_accelerated_flop_fraction must be 0..1"
        )
    prefill_min_accelerated_flop_fraction = float(
        prefill_min_accelerated_flop_fraction
    )
    if (
        isinstance(prefill_router_hybrid_margin_threshold, bool)
        or not isinstance(prefill_router_hybrid_margin_threshold, (int, float))
        or not math.isfinite(float(prefill_router_hybrid_margin_threshold))
        or float(prefill_router_hybrid_margin_threshold) < 0.0
    ):
        raise TokenGeneratorError(
            "prefill_router_hybrid_margin_threshold must be a non-negative finite number"
        )
    prefill_router_hybrid_margin_threshold = float(
        prefill_router_hybrid_margin_threshold
    )
    if type(require_prefill_acceleration) is not bool:
        raise TokenGeneratorError("require_prefill_acceleration must be a boolean")
    if type(allow_router_gate_only_prefill_acceleration) is not bool:
        raise TokenGeneratorError(
            "allow_router_gate_only_prefill_acceleration must be a boolean"
        )
    cache_position_offset = _nonnegative_integer_value(
        cache_position_offset,
        label="cache_position_offset",
    )
    if attention_scale is not None:
        attention_scale = _positive_float_value(
            attention_scale,
            label="attention_scale",
        )
    rope_theta = _positive_float_value(rope_theta, label="rope_theta")
    top_k = _positive_integer_value(top_k, label="top_k")
    max_k = _positive_integer_value(max_k, label="max_k")
    if top_k > max_k or max_k > 64:
        raise TokenGeneratorError("top_k/max_k must satisfy 1 <= top_k <= max_k <= 64")
    if routed_scaling_factor is not None:
        routed_scaling_factor = _positive_float_value(
            routed_scaling_factor,
            label="routed_scaling_factor",
        )
    if router_n_group is not None:
        router_n_group = _positive_integer_value(
            router_n_group,
            label="router_n_group",
        )
    if router_topk_group is not None:
        router_topk_group = _positive_integer_value(
            router_topk_group,
            label="router_topk_group",
        )
    rms_norm_eps = _nonnegative_float_value(rms_norm_eps, label="rms_norm_eps")
    logits_top_k = _positive_integer_value(logits_top_k, label="logits_top_k")
    if logits_chunk_rows is not None:
        logits_chunk_rows = _positive_integer_value(
            logits_chunk_rows,
            label="logits_chunk_rows",
        )
    expert_read_advise_merge_gap_kib = _nonnegative_integer_value(
        expert_read_advise_merge_gap_kib,
        label="expert_read_advise_merge_gap_kib",
    )
    expert_read_advise_align_kib = _nonnegative_integer_value(
        expert_read_advise_align_kib,
        label="expert_read_advise_align_kib",
    )
    cache_dtype_bytes = _positive_integer_value(
        cache_dtype_bytes,
        label="cache_dtype_bytes",
    )
    if cache_dtype_bytes not in {2, 4}:
        raise TokenGeneratorError("cache_dtype_bytes must be 2 or 4")
    prefill_expert_stage_merge_gap_kib = _nonnegative_float_value(
        prefill_expert_stage_merge_gap_kib,
        label="prefill_expert_stage_merge_gap_kib",
    )
    prefill_expert_stage_align_kib = _positive_float_value(
        prefill_expert_stage_align_kib,
        label="prefill_expert_stage_align_kib",
    )
    prefill_max_stage_mib = _positive_float_value(
        prefill_max_stage_mib,
        label="prefill_max_stage_mib",
    )
    prefill_max_compact_stage_mib = _positive_float_value(
        prefill_max_compact_stage_mib,
        label="prefill_max_compact_stage_mib",
    )
    prefill_max_stage_raw_ranges = _nonnegative_integer_value(
        prefill_max_stage_raw_ranges,
        label="prefill_max_stage_raw_ranges",
    )
    prefill_max_stage_coalesced_ranges = _nonnegative_integer_value(
        prefill_max_stage_coalesced_ranges,
        label="prefill_max_stage_coalesced_ranges",
    )
    if type(prefill_expert_stage_tiling) is not bool:
        raise TokenGeneratorError("prefill_expert_stage_tiling must be a boolean")
    if type(prefill_persistent_moe_plan_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_moe_plan_server must be a boolean"
        )
    if type(prefill_persistent_resident_linear_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_resident_linear_server must be a boolean"
        )
    if type(prefill_persistent_attention_projection_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_attention_projection_server must be a boolean"
        )
    if type(prefill_persistent_attention_output_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_attention_output_server must be a boolean"
        )
    if type(prefill_persistent_shared_expert_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_shared_expert_server must be a boolean"
        )
    if type(prefill_persistent_rope_split_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_rope_split_server must be a boolean"
        )
    if type(prefill_persistent_mla_attention_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_mla_attention_server must be a boolean"
        )
    if type(prefill_persistent_rmsnorm_server) is not bool:
        raise TokenGeneratorError(
            "prefill_persistent_rmsnorm_server must be a boolean"
        )
    try:
        prefill_moe_output_accumulator = _normalize_moe_output_accumulator(
            prefill_moe_output_accumulator
        )
    except StagedMoEError as exc:
        raise TokenGeneratorError(str(exc)) from exc
    prefill_copy_chunk_mib = _positive_float_value(
        prefill_copy_chunk_mib,
        label="prefill_copy_chunk_mib",
    )
    prefill_stage_disk_margin_mib = _nonnegative_float_value(
        prefill_stage_disk_margin_mib,
        label="prefill_stage_disk_margin_mib",
    )
    prefill_max_routed_read_amplification = _nonnegative_float_value(
        prefill_max_routed_read_amplification,
        label="prefill_max_routed_read_amplification",
    )
    prefill_max_routed_read_gib = _nonnegative_float_value(
        prefill_max_routed_read_gib,
        label="prefill_max_routed_read_gib",
    )
    prefill_ssd_read_gib_per_second = _nonnegative_float_value(
        prefill_ssd_read_gib_per_second,
        label="prefill_ssd_read_gib_per_second",
    )
    prefill_max_routed_read_seconds = _nonnegative_float_value(
        prefill_max_routed_read_seconds,
        label="prefill_max_routed_read_seconds",
    )
    decode_max_routed_read_gib_per_token = _nonnegative_float_value(
        decode_max_routed_read_gib_per_token,
        label="decode_max_routed_read_gib_per_token",
    )
    decode_max_routed_read_seconds_per_token = _nonnegative_float_value(
        decode_max_routed_read_seconds_per_token,
        label="decode_max_routed_read_seconds_per_token",
    )
    if prefill_max_routed_read_seconds > 0 and prefill_ssd_read_gib_per_second <= 0:
        raise TokenGeneratorError(
            "prefill_ssd_read_gib_per_second must be positive when "
            "prefill_max_routed_read_seconds is set"
        )
    if (
        decode_max_routed_read_seconds_per_token > 0
        and prefill_ssd_read_gib_per_second <= 0
    ):
        raise TokenGeneratorError(
            "prefill_ssd_read_gib_per_second must be positive when "
            "decode_max_routed_read_seconds_per_token is set"
        )
    if sampling_seed is not None:
        sampling_seed = _integer_value(sampling_seed, label="sampling_seed")
    sampling_temperature = _nonnegative_float_value(
        sampling_temperature,
        label="temperature",
    )
    sampling_top_p = _positive_float_value(sampling_top_p, label="top_p")
    if sampling_top_p > 1.0:
        raise TokenGeneratorError("top_p must be in (0, 1]")
    if sampling_temperature > 0.0 and logits_top_k < 2:
        raise TokenGeneratorError("sampling requires --logits-top-k of at least 2")
    prefill_mpsgraph_min_batch_tokens = _positive_integer_value(
        prefill_mpsgraph_min_batch_tokens,
        label="prefill_mpsgraph_min_batch_tokens",
    )
    prefill_mpsgraph_min_matrix_dim = _positive_integer_value(
        prefill_mpsgraph_min_matrix_dim,
        label="prefill_mpsgraph_min_matrix_dim",
    )
    resolved_prefill_prompt_chunk_tokens = _nonnegative_integer_value(
        prefill_prompt_chunk_tokens,
        label="prefill_prompt_chunk_tokens",
    )
    auto_prefill_prompt_chunk_plan: AutoPrefillPromptChunkPlan | None = None
    max_safe_prefill_prompt_chunk_plan: AutoPrefillPromptChunkPlan | None = None
    if batch_prefill_prompt:
        if resolved_prefill_prompt_chunk_tokens <= 0:
            auto_prefill_prompt_chunk_plan = _auto_prefill_prompt_chunk_plan(
                expert_layout_path=expert_layout_path,
                resident_layout_path=resident_layout_path,
                cache_layout_path=cache_layout_path,
                prompt_tokens=len(prompt),
                start_position=0,
                layers=layers,
                dense_layers=dense_layers,
                work_dir=work_dir,
                top_k=top_k,
                max_prompt_batch_mib=prefill_max_prompt_batch_mib,
                max_cache_read_mib=max_cache_read_mib,
                max_cache_write_mib=prefill_max_cache_write_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                prefill_max_stage_mib=prefill_max_stage_mib,
                prefill_max_compact_stage_mib=prefill_max_compact_stage_mib,
                prefill_expert_stage_align_kib=prefill_expert_stage_align_kib,
                prefill_stage_disk_margin_mib=prefill_stage_disk_margin_mib,
                dsa_indexer_types=dsa_indexer_types,
                dsa_index_topk=dsa_index_topk,
                prefill_expert_stage_tiling=prefill_expert_stage_tiling,
                prefill_linear_backend=prefill_linear_backend,
                prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
                prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            )
            resolved_prefill_prompt_chunk_tokens = (
                auto_prefill_prompt_chunk_plan.chunk_tokens
            )
        if resolved_prefill_prompt_chunk_tokens <= 0:
            raise TokenGeneratorError("prefill_prompt_chunk_tokens must be positive")
        max_safe_prefill_prompt_chunk_plan = _auto_prefill_prompt_chunk_plan(
            expert_layout_path=expert_layout_path,
            resident_layout_path=resident_layout_path,
            cache_layout_path=cache_layout_path,
            prompt_tokens=len(prompt),
            start_position=0,
            layers=layers,
            dense_layers=dense_layers,
            work_dir=work_dir,
            top_k=top_k,
            max_prompt_batch_mib=prefill_max_prompt_batch_mib,
            max_cache_read_mib=max_cache_read_mib,
            max_cache_write_mib=prefill_max_cache_write_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            prefill_max_stage_mib=prefill_max_stage_mib,
            prefill_max_compact_stage_mib=prefill_max_compact_stage_mib,
            prefill_expert_stage_align_kib=prefill_expert_stage_align_kib,
            prefill_stage_disk_margin_mib=prefill_stage_disk_margin_mib,
            dsa_indexer_types=dsa_indexer_types,
            dsa_index_topk=dsa_index_topk,
            prefill_expert_stage_tiling=prefill_expert_stage_tiling,
            prefill_linear_backend=prefill_linear_backend,
            prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            tile_tokens=1,
        )
        max_safe_prefill_chunk_tokens = (
            max_safe_prefill_prompt_chunk_plan.chunk_tokens
        )
        if resolved_prefill_prompt_chunk_tokens > max_safe_prefill_chunk_tokens:
            raise TokenGeneratorError(
                "prefill_prompt_chunk_tokens "
                f"{resolved_prefill_prompt_chunk_tokens} exceeds safety-capped "
                f"maximum {max_safe_prefill_chunk_tokens}; use auto or lower the "
                "configured prompt/cache/scratch/disk limits"
            )
        try:
            static_capacity_request = _normalize_static_capacity_per_expert(
                prefill_static_capacity_per_expert
            )
            _reject_unsafe_strict_static_capacity(
                static_capacity_request=static_capacity_request,
                max_batch_tokens=min(resolved_prefill_prompt_chunk_tokens, len(prompt)),
                allow_static_capacity_overflow=prefill_allow_static_capacity_overflow,
                label="prefill_static_capacity_per_expert",
            )
        except PromptPrefillError as exc:
            raise TokenGeneratorError(str(exc)) from exc
        if (
            prefill_max_routed_read_amplification > 0
            or prefill_max_routed_read_gib > 0
            or prefill_max_routed_read_seconds > 0
        ):
            max_routed_read_bytes = (
                int(prefill_max_routed_read_gib * 1024**3)
                if prefill_max_routed_read_gib > 0
                else 0
            )
            max_routed_read_seconds_bytes = (
                int(
                    prefill_max_routed_read_seconds
                    * prefill_ssd_read_gib_per_second
                    * 1024**3
                )
                if prefill_max_routed_read_seconds > 0
                else 0
            )
            effective_max_routed_read_bytes = max(
                0,
                min(
                    cap
                    for cap in (
                        max_routed_read_bytes,
                        max_routed_read_seconds_bytes,
                    )
                    if cap > 0
                )
                if max_routed_read_bytes > 0 or max_routed_read_seconds_bytes > 0
                else 0,
            )
            try:
                routed_read = estimate_routed_expert_read(
                    expert_layout_path=expert_layout_path,
                    prompt_token_count=len(prompt),
                    prompt_chunk_tokens=resolved_prefill_prompt_chunk_tokens,
                    top_k=top_k,
                    layers=layers,
                )
            except RoutedExpertReadError as exc:
                raise TokenGeneratorError(str(exc)) from exc
            if (
                prefill_max_routed_read_amplification > 0
                and routed_read.read_amplification
                > prefill_max_routed_read_amplification
            ):
                suggestion = _routed_read_chunk_suggestion(
                    expert_layout_path=expert_layout_path,
                    prompt_token_count=len(prompt),
                    top_k=top_k,
                    layers=layers,
                    max_read_amplification=prefill_max_routed_read_amplification,
                    max_planned_read_bytes=effective_max_routed_read_bytes,
                )
                raise TokenGeneratorError(
                    "prefill routed expert read amplification "
                    f"{routed_read.read_amplification:.3g} exceeds cap "
                    f"{prefill_max_routed_read_amplification:.3g}; "
                    f"planned_read_bytes={routed_read.planned_read_bytes} "
                    f"baseline_read_bytes={routed_read.baseline_read_bytes} "
                    f"extra_read_bytes={routed_read.extra_read_bytes}"
                    f"{suggestion}"
                )
            if prefill_max_routed_read_gib > 0:
                if routed_read.planned_read_bytes > max_routed_read_bytes:
                    suggestion = _routed_read_chunk_suggestion(
                        expert_layout_path=expert_layout_path,
                        prompt_token_count=len(prompt),
                        top_k=top_k,
                        layers=layers,
                        max_read_amplification=prefill_max_routed_read_amplification,
                        max_planned_read_bytes=effective_max_routed_read_bytes,
                    )
                    raise TokenGeneratorError(
                        "prefill routed expert planned read "
                        f"{routed_read.planned_read_bytes} bytes exceeds cap "
                        f"{max_routed_read_bytes} bytes; "
                        f"read_amplification={routed_read.read_amplification:.3g} "
                        f"baseline_read_bytes={routed_read.baseline_read_bytes} "
                        f"extra_read_bytes={routed_read.extra_read_bytes}"
                        f"{suggestion}"
                    )
            if prefill_max_routed_read_seconds > 0:
                planned_read_seconds = (
                    routed_read.planned_read_bytes
                    / (prefill_ssd_read_gib_per_second * 1024**3)
                )
                if planned_read_seconds > prefill_max_routed_read_seconds:
                    suggestion = _routed_read_chunk_suggestion(
                        expert_layout_path=expert_layout_path,
                        prompt_token_count=len(prompt),
                        top_k=top_k,
                        layers=layers,
                        max_read_amplification=prefill_max_routed_read_amplification,
                        max_planned_read_bytes=effective_max_routed_read_bytes,
                    )
                    raise TokenGeneratorError(
                        "prefill routed expert planned read time "
                        f"{planned_read_seconds:.3g}s exceeds cap "
                        f"{prefill_max_routed_read_seconds:.3g}s; "
                        f"planned_read_bytes={routed_read.planned_read_bytes} "
                        f"ssd_read_gib_per_second={prefill_ssd_read_gib_per_second:.3g} "
                        f"read_amplification={routed_read.read_amplification:.3g}"
                        f"{suggestion}"
                    )
    eos_set = {
        _nonnegative_integer_value(token, label="eos token ids")
        for token in eos_token_ids or ()
    }
    if eos_token_id is not None:
        eos_set.add(_nonnegative_integer_value(eos_token_id, label="eos token ids"))
    eos_tokens = frozenset(eos_set)

    max_context = _cache_max_context_tokens(cache_layout_path)
    required_steps = len(prompt) + max_new_tokens
    if required_steps > max_context:
        raise TokenGeneratorError(
            f"prompt+generation length {required_steps} exceeds cache context "
            f"{max_context}"
        )
    request_fits_dsa_visible_topk = (
        dsa_index_topk is not None and required_steps <= dsa_index_topk
    )
    if max_new_tokens == 0:
        return TokenGenerationResult(
            prompt_token_ids=prompt,
            generated_token_ids=(),
            steps=(),
            work_dir=Path(work_dir or ""),
            kept_work_dir=bool(work_dir),
            max_context_tokens=max_context,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
            runtime_guard=None,
        )

    runtime_guard: GenerationRuntimeGuard | None = None
    needs_runtime_guard = (
        preflight_runtime
        or min_free_unified_memory_gib > 0.0
        or decode_max_routed_read_gib_per_token > 0.0
        or decode_max_routed_read_seconds_per_token > 0.0
    )
    if needs_runtime_guard:
        extra_live_working_set_bytes = 0
        if batch_prefill_prompt:
            extra_live_working_set_bytes = (
                estimate_prompt_prefill_live_memory(
                    max_prompt_batch_mib=prefill_max_prompt_batch_mib,
                    max_runner_scratch_mib=max_runner_scratch_mib,
                    max_cache_read_mib=max_cache_read_mib,
                    max_cache_write_mib=prefill_max_cache_write_mib,
                    copy_chunk_mib=prefill_copy_chunk_mib,
                ).estimated_live_working_set_bytes
            )
        try:
            runtime_guard = check_generation_runtime(
                expert_layout_path=expert_layout_path,
                resident_layout_path=resident_layout_path,
                cache_layout_path=cache_layout_path,
                cache_file_path=cache_file_path,
                requested_context_tokens=required_steps,
                layers=layers,
                dense_layers=dense_layers,
                top_k=top_k,
                max_k=max_k,
                num_heads=num_heads,
                qk_nope_dim=qk_nope_dim,
                rope_dim=rope_dim,
                v_head_dim=v_head_dim,
                include_shared_expert=include_shared_expert,
                logits_top_k=logits_top_k,
                logits_chunk_rows=logits_chunk_rows,
                logits_max_chunk_mib=logits_max_chunk_mib,
                rms_norm_eps=rms_norm_eps,
                max_slot_mib=max_slot_mib,
                max_router_mib=max_router_mib,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_cache_file_mib=max_cache_file_mib,
                max_cache_read_mib=max_cache_read_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                cache_dtype_bytes=cache_dtype_bytes,
                metal_final_logits=metal_final_logits,
                decode_mla_key_cache=decode_mla_key_cache,
                allow_tied_embeddings=allow_tied_embeddings,
                expected_vocab_size=expected_vocab_size,
                expected_hidden_size=expected_hidden_size,
                max_embedding_row_mib=max_embedding_row_mib,
                max_live_working_set_mib=max_live_working_set_mib,
                min_free_unified_memory_mib=min_free_unified_memory_gib * 1024,
                extra_live_working_set_bytes=extra_live_working_set_bytes,
                dsa_indexer_runtime=bool(dsa_indexer_types),
                dsa_indexer_types=dsa_indexer_types,
                dsa_index_topk=dsa_index_topk,
                dsa_index_head_dim=dsa_index_head_dim,
                allow_missing_dsa_indexer=allow_missing_dsa_indexer,
            )
        except RuntimeError as exc:
            raise TokenGeneratorError(str(exc)) from exc
        _check_decode_routed_read_guard(
            runtime_guard,
            max_read_gib_per_token=decode_max_routed_read_gib_per_token,
            ssd_read_gib_per_second=prefill_ssd_read_gib_per_second,
            max_read_seconds_per_token=decode_max_routed_read_seconds_per_token,
        )

    created_work_dir = False
    if work_dir is None:
        root = Path(tempfile.mkdtemp(prefix="largerlm-generate-token-ids-", dir="/private/tmp"))
        created_work_dir = True
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)

    generated: list[int] = []
    steps: list[GeneratedStep] = []
    token_to_feed = prompt[0]
    decode_start_position = 0
    prompt_prefill_result: PromptPrefillResult | None = None
    rng = random.Random(sampling_seed) if sampling_seed is not None else None
    total_started = time.perf_counter()
    total_embedding_read = 0
    total_expert_read = 0
    total_cache_read = 0
    total_logits_read = 0
    try:
        _recheck_live_memory_guard(runtime_guard)
        if batch_prefill_prompt:
            step_started = time.perf_counter()
            prefill_last_hidden = root / "prefill_last_hidden.f32"
            prefill_final_chunk = root / "prefill_final_chunk.f32" if keep_work_dir else None
            try:
                prompt_prefill_result = run_prompt_prefill(
                    runner_path=runner_path,
                    expert_layout_path=expert_layout_path,
                    resident_layout_path=resident_layout_path,
                    cache_layout_path=cache_layout_path,
                    cache_file_path=cache_file_path,
                    prompt_token_ids=prompt,
                    output_last_hidden_f32_path=prefill_last_hidden,
                    output_final_chunk_f32_path=prefill_final_chunk,
                    layers=layers,
                    dense_layers=dense_layers,
                    work_dir=root / "prefill_prompt",
                    keep_work_dir=keep_work_dir,
                    start_position=0,
                    prompt_chunk_tokens=resolved_prefill_prompt_chunk_tokens,
                    max_prompt_batch_mib=prefill_max_prompt_batch_mib,
                    num_heads=num_heads,
                    qk_nope_dim=qk_nope_dim,
                    rope_dim=rope_dim,
                    v_head_dim=v_head_dim,
                    mla_kv_b_cache_dir=prefill_mla_kv_b_cache_dir,
                    mla_key_cache=prefill_mla_key_cache,
                    kv_lora_dim=kv_lora_dim,
                    cache_position_offset=cache_position_offset,
                    attention_scale=attention_scale,
                    rope_theta=rope_theta,
                    rope_interleave=rope_interleave,
                    top_k=top_k,
                    max_k=max_k,
                    router_score=router_score,
                    routed_scaling_factor=routed_scaling_factor,
                    norm_topk_prob=norm_topk_prob,
                    no_norm_topk_prob=no_norm_topk_prob,
                    router_n_group=router_n_group,
                    router_topk_group=router_topk_group,
                    ignore_router_bias=ignore_router_bias,
                    include_shared_expert=include_shared_expert,
                    rms_norm_eps=rms_norm_eps,
                    max_embedding_row_mib=max_embedding_row_mib,
                    max_slot_mib=max_slot_mib,
                    max_router_mib=max_router_mib,
                    max_resident_matrix_mib=max_resident_matrix_mib,
                    max_cache_file_mib=max_cache_file_mib,
                    max_cache_write_mib=prefill_max_cache_write_mib,
                    max_cache_read_mib=max_cache_read_mib,
                    max_runner_scratch_mib=max_runner_scratch_mib,
                    max_live_working_set_mib=max_live_working_set_mib,
                    min_free_unified_memory_gib=min_free_unified_memory_gib,
                    moe_token_block=prefill_moe_token_block,
                    moe_output_accumulator=prefill_moe_output_accumulator,
                    static_capacity_per_expert=prefill_static_capacity_per_expert,
                    allow_static_capacity_overflow=prefill_allow_static_capacity_overflow,
                    expert_stage_merge_gap_kib=prefill_expert_stage_merge_gap_kib,
                    expert_stage_align_kib=prefill_expert_stage_align_kib,
                    max_stage_mib=prefill_max_stage_mib,
                    max_compact_stage_mib=prefill_max_compact_stage_mib,
                    copy_chunk_mib=prefill_copy_chunk_mib,
                    stage_disk_safety_margin_bytes=int(
                        prefill_stage_disk_margin_mib * 1024**2
                    ),
                    prefill_ssd_read_gib_per_second=(
                        prefill_ssd_read_gib_per_second
                    ),
                    prefill_max_routed_read_seconds=(
                        prefill_max_routed_read_seconds
                    ),
                    expert_stage_max_raw_ranges=prefill_max_stage_raw_ranges,
                    expert_stage_max_coalesced_ranges=(
                        prefill_max_stage_coalesced_ranges
                    ),
                    expert_stage_tiling=prefill_expert_stage_tiling,
                    persistent_moe_plan_server=prefill_persistent_moe_plan_server,
                    persistent_resident_linear_server=(
                        prefill_persistent_resident_linear_server
                    ),
                    persistent_attention_projection_server=(
                        prefill_persistent_attention_projection_server
                    ),
                    persistent_attention_output_server=(
                        prefill_persistent_attention_output_server
                    ),
                    persistent_shared_expert_server=(
                        prefill_persistent_shared_expert_server
                    ),
                    persistent_rope_split_server=(
                        prefill_persistent_rope_split_server
                    ),
                    persistent_mla_attention_server=(
                        prefill_persistent_mla_attention_server
                    ),
                    persistent_rmsnorm_server=prefill_persistent_rmsnorm_server,
                    prefill_linear_backend=prefill_linear_backend,
                    prefill_min_accelerated_flop_fraction=(
                        prefill_min_accelerated_flop_fraction
                    ),
                    prefill_mpsgraph_min_batch_tokens=(
                        prefill_mpsgraph_min_batch_tokens
                    ),
                    prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
                    router_hybrid_margin_threshold=(
                        prefill_router_hybrid_margin_threshold
                    ),
                    dsa_indexer_types=dsa_indexer_types,
                    dsa_index_topk=dsa_index_topk,
                    dsa_index_n_heads=dsa_index_n_heads,
                    dsa_qk_rope_dim=dsa_qk_rope_dim,
                    dsa_rope_interleave=dsa_rope_interleave,
                    dsa_layer_norm_eps=dsa_layer_norm_eps,
                    write_dsa_future_cache=not request_fits_dsa_visible_topk,
                    expected_vocab_size=expected_vocab_size,
                    expected_hidden_size=expected_hidden_size,
                    echo_runner_output=echo_runner_output,
                )
            except PromptPrefillError as exc:
                raise TokenGeneratorError(str(exc)) from exc
            prefill_expert_read, prefill_cache_read = _prompt_prefill_read_estimates(
                prompt_prefill_result
            )
            _recheck_live_memory_guard(runtime_guard)
            logits_started = time.perf_counter()
            if metal_final_logits:
                logits = compute_final_logits_metal(
                    runner_path=runner_path,
                    resident_layout_path=resident_layout_path,
                    input_f32_path=prefill_last_hidden,
                    top_k=logits_top_k,
                    rms_norm_eps=rms_norm_eps,
                    chunk_rows=logits_chunk_rows,
                    max_chunk_bytes=int(logits_max_chunk_mib * 1024**2),
                    max_runner_scratch_bytes=int(max_runner_scratch_mib * 1024**2),
                    allow_tied_embeddings=allow_tied_embeddings,
                    expected_vocab_size=expected_vocab_size,
                    expected_hidden_size=expected_hidden_size,
                    echo_runner_output=echo_runner_output,
                )
            else:
                logits = compute_final_logits(
                    resident_layout_path,
                    prefill_last_hidden,
                    top_k=logits_top_k,
                    rms_norm_eps=rms_norm_eps,
                    chunk_rows=logits_chunk_rows,
                    max_chunk_bytes=int(logits_max_chunk_mib * 1024**2),
                    allow_tied_embeddings=allow_tied_embeddings,
                    expected_vocab_size=expected_vocab_size,
                    expected_hidden_size=expected_hidden_size,
                )
            logits_elapsed = time.perf_counter() - logits_started
            selected = _select_token(
                logits.topk,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                rng=rng,
            )
            step_elapsed = time.perf_counter() - step_started
            total_embedding_read += prompt_prefill_result.total_embedding_read_bytes
            total_expert_read += prefill_expert_read
            total_cache_read += prefill_cache_read
            total_logits_read += logits.read_bytes
            generated.append(selected)
            steps.append(
                GeneratedStep(
                    position=len(prompt) - 1,
                    input_token_id=prompt[-1],
                    selected_token_id=selected,
                    topk=logits.topk,
                    elapsed_seconds=step_elapsed,
                    embedding_read_bytes=prompt_prefill_result.total_embedding_read_bytes,
                    expert_read_bytes=prefill_expert_read,
                    cache_read_bytes=prefill_cache_read,
                    logits_read_bytes=logits.read_bytes,
                    logits_elapsed_seconds=logits_elapsed,
                )
            )
            token_to_feed = selected
            decode_start_position = len(prompt)
            if selected in eos_tokens:
                decode_start_position = required_steps
            elif _stop_after_missing_dsa_debug_allow(runtime_guard):
                decode_start_position = required_steps
            if not keep_work_dir:
                prefill_last_hidden.unlink(missing_ok=True)

        for position in range(decode_start_position, required_steps):
            if len(generated) >= max_new_tokens:
                break
            _recheck_live_memory_guard(runtime_guard)
            step_started = time.perf_counter()
            input_hidden = root / f"pos_{position:04d}_input.f32"
            output_hidden = root / f"pos_{position:04d}_hidden.f32"
            embedding = embed_token(
                resident_layout_path,
                token_id=token_to_feed,
                output_f32_path=input_hidden,
                max_row_bytes=int(max_embedding_row_mib * 1024**2),
                expected_vocab_size=expected_vocab_size,
                expected_hidden_size=expected_hidden_size,
            )
            is_generation_step = position >= len(prompt) - 1
            final_generation_step = (
                is_generation_step and len(generated) + 1 >= max_new_tokens
            )
            write_dsa_future_cache = not (
                request_fits_dsa_visible_topk or final_generation_step
            )
            fused_final_topk_path: Path | None = None
            if (
                metal_final_logits
                and runtime_guard is not None
                and not keep_work_dir
                and Path(runner_path).name == "largerlm-runner"
                and not write_dsa_future_cache
                and (
                    position + 1 == 1
                    or (
                        dsa_index_topk is not None
                        and position + 1 <= dsa_index_topk
                    )
                )
            ):
                fused_final_topk_path = root / f"pos_{position:04d}_topk.json"
            layers_result = run_decode_layers(
                runner_path=runner_path,
                expert_layout_path=expert_layout_path,
                resident_layout_path=resident_layout_path,
                cache_layout_path=cache_layout_path,
                cache_file_path=cache_file_path,
                input_path=input_hidden,
                output_path=output_hidden,
                layers=layers,
                dense_layers=dense_layers,
                work_dir=(
                    root / f"pos_{position:04d}_layers"
                    if keep_work_dir
                    else None
                ),
                keep_work_dir=keep_work_dir,
                position=position,
                context_length=position + 1,
                num_heads=num_heads,
                qk_nope_dim=qk_nope_dim,
                rope_dim=rope_dim,
                v_head_dim=v_head_dim,
                kv_lora_dim=kv_lora_dim,
                mla_kv_b_cache_dir=prefill_mla_kv_b_cache_dir,
                mla_key_cache=decode_mla_key_cache,
                cache_position_offset=cache_position_offset,
                attention_scale=attention_scale,
                rope_theta=rope_theta,
                rope_interleave=rope_interleave,
                top_k=top_k,
                max_k=max_k,
                router_score=router_score,
                routed_scaling_factor=routed_scaling_factor,
                norm_topk_prob=norm_topk_prob,
                no_norm_topk_prob=no_norm_topk_prob,
                router_n_group=router_n_group,
                router_topk_group=router_topk_group,
                ignore_router_bias=ignore_router_bias,
                include_shared_expert=include_shared_expert,
                rms_norm_eps=rms_norm_eps,
                max_slot_mib=max_slot_mib,
                max_router_mib=max_router_mib,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_cache_file_mib=max_cache_file_mib,
                max_cache_read_mib=max_cache_read_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                expert_read_advise_merge_gap_kib=expert_read_advise_merge_gap_kib,
                expert_read_advise_align_kib=expert_read_advise_align_kib,
                cache_dtype_bytes=cache_dtype_bytes,
                dsa_indexer_types=dsa_indexer_types,
                dsa_index_topk=dsa_index_topk,
                dsa_index_n_heads=dsa_index_n_heads,
                dsa_index_head_dim=dsa_index_head_dim,
                dsa_qk_rope_dim=dsa_qk_rope_dim,
                dsa_rope_interleave=dsa_rope_interleave,
                dsa_layer_norm_eps=dsa_layer_norm_eps,
                write_dsa_future_cache=write_dsa_future_cache,
                final_logits_topk_path=fused_final_topk_path,
                final_logits_top_k=logits_top_k,
                final_logits_chunk_rows=logits_chunk_rows,
                final_logits_max_chunk_mib=logits_max_chunk_mib,
                final_logits_rms_norm_eps=rms_norm_eps,
                final_logits_allow_tied_embeddings=allow_tied_embeddings,
                echo_runner_output=echo_runner_output,
            )
            step_expert_read = sum(
                budget.read_bytes_per_token for budget in layers_result.budgets
            )
            step_cache_read = sum(
                budget.decoder_cache_read_bytes for budget in layers_result.budgets
            )
            _recheck_live_memory_guard(runtime_guard)
            logits_started = time.perf_counter()
            if fused_final_topk_path is not None:
                assert runtime_guard is not None
                logits = _metal_final_logits_result_from_topk(
                    resident_layout_path=resident_layout_path,
                    input_f32_path=output_hidden,
                    topk_json_path=fused_final_topk_path,
                    runtime_guard=runtime_guard,
                    top_k=logits_top_k,
                )
                fused_logits_elapsed = _metal_final_logits_elapsed_from_topk(
                    fused_final_topk_path
                )
            elif metal_final_logits:
                logits = compute_final_logits_metal(
                    runner_path=runner_path,
                    resident_layout_path=resident_layout_path,
                    input_f32_path=output_hidden,
                    top_k=logits_top_k,
                    rms_norm_eps=rms_norm_eps,
                    chunk_rows=logits_chunk_rows,
                    max_chunk_bytes=int(logits_max_chunk_mib * 1024**2),
                    max_runner_scratch_bytes=int(max_runner_scratch_mib * 1024**2),
                    allow_tied_embeddings=allow_tied_embeddings,
                    expected_vocab_size=expected_vocab_size,
                    expected_hidden_size=expected_hidden_size,
                    echo_runner_output=echo_runner_output,
                )
                fused_logits_elapsed = None
            else:
                logits = compute_final_logits(
                    resident_layout_path,
                    output_hidden,
                    top_k=logits_top_k,
                    rms_norm_eps=rms_norm_eps,
                    chunk_rows=logits_chunk_rows,
                    max_chunk_bytes=int(logits_max_chunk_mib * 1024**2),
                    allow_tied_embeddings=allow_tied_embeddings,
                    expected_vocab_size=expected_vocab_size,
                    expected_hidden_size=expected_hidden_size,
                )
                fused_logits_elapsed = None
            logits_elapsed = (
                fused_logits_elapsed
                if fused_logits_elapsed is not None
                else time.perf_counter() - logits_started
            )
            step_elapsed = time.perf_counter() - step_started
            total_embedding_read += embedding.read_bytes
            total_expert_read += step_expert_read
            total_cache_read += step_cache_read
            total_logits_read += logits.read_bytes
            selected = _select_token(
                logits.topk,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                rng=rng,
            )
            stop = False
            if position >= len(prompt) - 1:
                generated.append(selected)
                steps.append(
                    GeneratedStep(
                        position=position,
                        input_token_id=token_to_feed,
                        selected_token_id=selected,
                        topk=logits.topk,
                        elapsed_seconds=step_elapsed,
                        embedding_read_bytes=embedding.read_bytes,
                        expert_read_bytes=step_expert_read,
                        cache_read_bytes=step_cache_read,
                        logits_read_bytes=logits.read_bytes,
                        logits_elapsed_seconds=logits_elapsed,
                        decode_layers=layers_result.records,
                    )
                )
                if selected in eos_tokens:
                    stop = True
                elif _stop_after_missing_dsa_debug_allow(runtime_guard):
                    stop = True
                elif len(generated) >= max_new_tokens:
                    stop = True
            next_prompt_index = position + 1
            token_to_feed = (
                prompt[next_prompt_index]
                if next_prompt_index < len(prompt)
                else selected
            )
            if not keep_work_dir:
                input_hidden.unlink(missing_ok=True)
                output_hidden.unlink(missing_ok=True)
            if stop:
                break
    except (EmbeddingError, DecodeDriverError, FinalLogitsError, OSError) as exc:
        if keep_work_dir or not created_work_dir:
            keep_work_dir = True
        raise TokenGeneratorError(str(exc)) from exc
    except Exception:
        if keep_work_dir or not created_work_dir:
            keep_work_dir = True
        raise
    finally:
        if created_work_dir and not keep_work_dir:
            shutil.rmtree(root, ignore_errors=True)

    return TokenGenerationResult(
        prompt_token_ids=prompt,
        generated_token_ids=tuple(generated),
        steps=tuple(steps),
        work_dir=root,
        kept_work_dir=keep_work_dir or not created_work_dir,
        max_context_tokens=max_context,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
        elapsed_seconds=time.perf_counter() - total_started,
        estimated_read_bytes=(
            total_embedding_read
            + total_expert_read
            + total_cache_read
            + total_logits_read
        ),
        estimated_embedding_read_bytes=total_embedding_read,
        estimated_expert_read_bytes=total_expert_read,
        estimated_cache_read_bytes=total_cache_read,
        estimated_logits_read_bytes=total_logits_read,
        runtime_guard=runtime_guard,
        prompt_prefill=prompt_prefill_result,
        auto_prefill_prompt_chunk_plan=auto_prefill_prompt_chunk_plan,
        max_safe_prefill_prompt_chunk_plan=max_safe_prefill_prompt_chunk_plan,
        prefill_actual_read_time=_prefill_actual_read_time_summary(
            prompt_prefill_result
        ),
        prefill_actual_acceleration_coverage=(
            _prefill_actual_acceleration_coverage_summary(
                prompt_prefill_result,
                require_prefill_acceleration=require_prefill_acceleration,
                min_accelerated_flop_fraction=(
                    prefill_min_accelerated_flop_fraction
                ),
                allow_router_gate_only_acceleration=(
                    allow_router_gate_only_prefill_acceleration
                ),
            )
        ),
        prefill_actual_acceleration_frontier=(
            _prefill_actual_acceleration_frontier_summary(prompt_prefill_result)
        ),
        prefill_actual_linear_backend=_prefill_actual_linear_backend_summary(
            prompt_prefill_result,
            prefill_linear_backend=prefill_linear_backend,
            prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        ),
        decode_actual_read_time=_decode_actual_read_time_summary(
            steps=tuple(steps),
            runtime_guard=runtime_guard,
            ssd_read_gib_per_second=prefill_ssd_read_gib_per_second,
            max_read_seconds_per_token=decode_max_routed_read_seconds_per_token,
        ),
    )
