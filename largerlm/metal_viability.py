from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .latent_value_collapse import (
    LatentValueCollapseError,
    build_latent_value_collapse_plan,
)


class MetalViabilityError(RuntimeError):
    """Raised when a Metal runtime viability report cannot be built."""


@dataclass(frozen=True)
class MetalViabilityReport:
    prepared_dir: Path
    expert_layer_count: int
    top_k: int
    expert_slot_bytes_min: int
    expert_slot_bytes_max: int
    routed_expert_read_bytes_per_token: int
    routed_expert_read_gib_per_token: float
    qwen_flash_moe_reference_read_bytes_per_token: int
    glm_to_qwen_flash_moe_read_ratio: float
    ssd_read_gib_per_second: float | None
    routed_read_lower_bound_seconds_per_token: float | None
    observed_decode_seconds_per_token: float | None
    observed_tokens_per_second: float | None
    observed_generated_token_ids: tuple[int, ...]
    observed_estimated_read_bytes: int | None
    observed_estimated_logits_read_bytes: int | None
    observed_estimated_expert_read_bytes: int | None
    decision: str
    next_runtime_step: str
    context1_collapse_plan_path: Path | None = None
    context1_layers_supported: int | None = None
    context1_cache_bytes: int | None = None
    context1_current_o_proj_bytes_per_token: int | None = None
    context1_cache_read_ratio: float | None = None
    context1_build_fma_per_layer: int | None = None
    context1_build_fma_total: int | None = None
    observed_decode_token_count: int | None = None
    observed_attn_output_bytes_per_token: float | None = None
    observed_attn_output_read_seconds_per_token: float | None = None
    observed_attn_output_projection_seconds_per_token: float | None = None
    projected_context1_only_total_with_logits_seconds: float | None = None
    projected_context1_only_tokens_per_second: float | None = None
    projected_context1_only_speedup: float | None = None
    exact_all_context_per_head_cache_bytes: int | None = None
    exact_all_context_per_head_cache_read_ratio: float | None = None
    exact_all_context_per_head_int4_floor_bytes: int | None = None
    exact_all_context_per_head_int4_floor_read_ratio: float | None = None
    exact_all_context_per_head_runtime_fma_ratio: float | None = None
    exact_all_context_break_even_shared_head_groups: int | None = None
    exact_all_context_independent_attention_head_groups: int | None = None
    projected_exact_all_context_per_head_total_with_logits_seconds: float | None = None
    projected_exact_all_context_per_head_tokens_per_second: float | None = None
    projected_exact_all_context_per_head_speedup: float | None = None
    efficiency_rewrite_decision: str | None = None
    target_tokens_per_second: float | None = None
    target_reference_tokens_per_second: float | None = None
    target_reference_source: str | None = None
    target_met: bool | None = None
    target_gap_multiplier: float | None = None
    target_decision: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prepared_dir"] = str(self.prepared_dir)
        if self.context1_collapse_plan_path is not None:
            payload["context1_collapse_plan_path"] = str(
                self.context1_collapse_plan_path
            )
        payload["observed_generated_token_ids"] = list(
            self.observed_generated_token_ids
        )
        return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetalViabilityError(f"failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MetalViabilityError(f"{path} must contain a JSON object")
    return payload


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _extract_token_result(payload: dict[str, Any]) -> dict[str, Any]:
    token_result = payload.get("token_result")
    if isinstance(token_result, dict):
        return token_result
    return payload


def _extract_generated_token_ids(token_result: dict[str, Any]) -> tuple[int, ...]:
    raw = token_result.get("generated_token_ids")
    if not isinstance(raw, list):
        return ()
    ids: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            return ()
        ids.append(int(item))
    return tuple(ids)


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _number_sum(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        total = 0.0
        seen = False
        for item in value:
            parsed = _number(item)
            if parsed is None:
                continue
            total += parsed
            seen = True
        return total if seen else None
    return _number(value)


def _extract_timing_component(
    components: object,
    label: str,
) -> float | None:
    if not isinstance(components, list):
        return None
    for item in components:
        if not isinstance(item, dict) or item.get("label") != label:
            continue
        return _number(item.get("seconds"))
    return None


def _extract_context1_plan(
    prepared: Path,
    explicit_path: str | Path | None,
) -> tuple[Path | None, dict[str, Any]]:
    path = Path(explicit_path) if explicit_path is not None else None
    if path is None:
        default_path = prepared / "context1-o-proj-collapse-plan-latest.json"
        if default_path.exists():
            path = default_path
    if path is None:
        return None, {}
    return path, _load_json_object(path)


def _extract_decode_telemetry(
    explicit_path: str | Path | None,
) -> dict[str, float | int | None]:
    if explicit_path is None:
        return {}
    payload = _load_json_object(Path(explicit_path))
    if payload.get("schema") == "largerlm.decode_telemetry_report.v1":
        bytes_payload = payload.get("bytes")
        if not isinstance(bytes_payload, dict):
            bytes_payload = {}
        timing = payload.get("timing")
        nested = timing.get("nested_components") if isinstance(timing, dict) else None
        token_count = _optional_int(payload.get("token_count"))
        elapsed = _optional_float(payload.get("elapsed_seconds"))
        total_with_logits = _optional_float(
            payload.get("total_with_final_logits_seconds")
        )
        attn_output_bytes = _optional_int(bytes_payload.get("attn_output_bytes_read"))
        attn_read = _extract_timing_component(nested, "attn_output_read")
        attn_projection = _extract_timing_component(
            nested,
            "attn_output_projection_kernel",
        )
        return {
            "token_count": token_count,
            "decode_seconds": elapsed,
            "total_with_logits_seconds": total_with_logits,
            "attn_output_bytes": attn_output_bytes,
            "attn_output_read_seconds": attn_read,
            "attn_output_projection_seconds": attn_projection,
        }

    generated = payload.get("generated_token_ids")
    generated_count = len(generated) if isinstance(generated, list) else None
    decode_elapsed = _number_sum(payload, "decode_elapsed_seconds")
    final_logits = _number_sum(payload, "final_logits_elapsed_seconds")
    token_count = generated_count
    if token_count is None:
        raw_decode = payload.get("decode_elapsed_seconds")
        if isinstance(raw_decode, list):
            token_count = len(raw_decode)
    total_with_logits = None
    if decode_elapsed is not None:
        total_with_logits = decode_elapsed + (final_logits or 0.0)
    return {
        "token_count": token_count,
        "decode_seconds": decode_elapsed,
        "total_with_logits_seconds": total_with_logits,
        "attn_output_bytes": _number_sum(payload, "decode_attn_output_bytes_read"),
        "attn_output_read_seconds": _number_sum(
            payload,
            "decode_attn_output_read_seconds",
        ),
        "attn_output_projection_seconds": _number_sum(
            payload,
            "decode_attn_output_projection_kernel_seconds",
        ),
    }


def _collapse_efficiency_decision(
    *,
    context1_speedup: float | None,
    exact_per_head_speedup: float | None,
    independent_head_groups: int | None,
    break_even_head_groups: int | None,
) -> str | None:
    if context1_speedup is None and exact_per_head_speedup is None:
        return None
    if context1_speedup is not None and context1_speedup < 1.10:
        prefix = (
            "context1_only_not_enough: current cache proves the GLM linear "
            "collapse but cannot make general decode usable by itself"
        )
    else:
        prefix = (
            "context1_measurement_needed: build one guarded cache layer and "
            "measure the narrow context=1 branch before promotion"
        )
    if exact_per_head_speedup is not None and exact_per_head_speedup < 1.0:
        return (
            f"{prefix}; exact_all_context_per_head_cache_not_viable: "
            f"{independent_head_groups} independent attention heads exceed the "
            f"{break_even_head_groups} break-even shared-head groups, so do not "
            "replace general decode with a per-head collapsed cache"
        )
    return (
        f"{prefix}; exact_all_context_cache_requires_measurement: only proceed "
        "if a shape-specific plan proves the head grouping/cache traffic is a win"
    )


def _decision(
    *,
    observed_decode_seconds: float | None,
    lower_bound_seconds: float | None,
) -> str:
    if observed_decode_seconds is None:
        return (
            "estimate_only: no observed decode smoke was supplied; keep using "
            "the single-process C/Metal rewrite path and measure before promotion"
        )
    if observed_decode_seconds > 2.0:
        return (
            "runnable_but_not_usable_speed: do not continue optimizing the old "
            "Python/file orchestration path; continue the Flash-MoE-shaped "
            "persistent C/Metal runtime"
        )
    if (
        lower_bound_seconds is not None
        and observed_decode_seconds < lower_bound_seconds
    ):
        return (
            "measurement_suspicious: observed decode is faster than the modeled "
            "routed-read lower bound; recheck smoke inputs and cached-read assumptions"
        )
    return (
        "candidate_usable_decode: decode is close enough to interactive speed "
        "to prioritize prompt prefill and serving integration"
    )


def _target_decision(
    *,
    target_tokens_per_second: float | None,
    candidates: tuple[tuple[str, float | None], ...],
) -> tuple[float | None, str | None, bool | None, float | None, str | None]:
    if target_tokens_per_second is None:
        return None, None, None, None, None
    if target_tokens_per_second <= 0.0:
        raise MetalViabilityError("target_tokens_per_second must be positive")
    valid = tuple(
        (source, tokens_per_second)
        for source, tokens_per_second in candidates
        if tokens_per_second is not None and tokens_per_second > 0.0
    )
    if not valid:
        return (
            None,
            None,
            None,
            None,
            (
                "target_unknown: no observed or projected throughput is available "
                "for the requested target"
            ),
        )
    source, best_tokens_per_second = max(valid, key=lambda item: item[1])
    target_met = best_tokens_per_second >= target_tokens_per_second
    gap = (
        target_tokens_per_second / best_tokens_per_second
        if best_tokens_per_second > 0.0
        else None
    )
    if target_met:
        decision = (
            "continue_above_target: best observed/projected throughput meets "
            "the requested target"
        )
    else:
        decision = (
            "stop_at_minimal_usable: best observed/projected throughput is "
            "below the requested target; finish the safe runnable path and "
            "do not continue speculative optimization without a model/layout "
            "change"
        )
    return best_tokens_per_second, source, target_met, gap, decision


def build_metal_viability_report(
    prepared_dir: str | Path,
    *,
    top_k: int = 8,
    smoke_result_path: str | Path | None = None,
    ssd_read_gib_per_second: float | None = None,
    context1_collapse_plan_path: str | Path | None = None,
    decode_telemetry_path: str | Path | None = None,
    target_tokens_per_second: float | None = None,
    flash_moe_reference_layers: int = 60,
    flash_moe_reference_top_k: int = 4,
    flash_moe_reference_expert_slot_bytes: int = 7_080_000,
) -> MetalViabilityReport:
    prepared = Path(prepared_dir)
    if top_k <= 0:
        raise MetalViabilityError("top_k must be positive")
    layout_path = prepared / "experts" / "layout.json"
    layout = _load_json_object(layout_path)
    raw_layers = layout.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise MetalViabilityError(f"{layout_path} must contain non-empty layers")
    slot_bytes: list[int] = []
    for item in raw_layers:
        if not isinstance(item, dict):
            raise MetalViabilityError("expert layout layers must be objects")
        raw_slot = item.get("expert_slot_bytes")
        if (
            isinstance(raw_slot, bool)
            or not isinstance(raw_slot, int)
            or raw_slot <= 0
        ):
            raise MetalViabilityError("expert_slot_bytes must be positive integers")
        slot_bytes.append(int(raw_slot))

    routed_read_bytes = sum(slot * top_k for slot in slot_bytes)
    routed_read_gib = routed_read_bytes / 1024**3
    reference_read_bytes = (
        int(flash_moe_reference_layers)
        * int(flash_moe_reference_top_k)
        * int(flash_moe_reference_expert_slot_bytes)
    )
    read_ratio = (
        routed_read_bytes / reference_read_bytes
        if reference_read_bytes > 0
        else float("inf")
    )

    token_result: dict[str, Any] = {}
    resolved_smoke = Path(smoke_result_path) if smoke_result_path is not None else None
    if resolved_smoke is None:
        default_smoke = prepared / "smoke-decode-1tok-metal-logits-result.json"
        if default_smoke.exists():
            resolved_smoke = default_smoke
    if resolved_smoke is not None:
        token_result = _extract_token_result(_load_json_object(resolved_smoke))

    decode_actual = token_result.get("decode_actual_read_time")
    if not isinstance(decode_actual, dict):
        decode_actual = {}
    observed_decode_seconds = _optional_float(token_result.get("elapsed_seconds"))
    generated_ids = _extract_generated_token_ids(token_result)
    if observed_decode_seconds is not None and generated_ids:
        observed_decode_seconds = observed_decode_seconds / len(generated_ids)
    observed_tps = (
        1.0 / observed_decode_seconds
        if observed_decode_seconds is not None and observed_decode_seconds > 0.0
        else None
    )
    resolved_ssd = (
        float(ssd_read_gib_per_second)
        if ssd_read_gib_per_second is not None
        else _optional_float(decode_actual.get("prefill_ssd_read_gib_per_second"))
    )
    lower_bound = (
        routed_read_gib / resolved_ssd
        if resolved_ssd is not None and resolved_ssd > 0.0
        else None
    )

    context1_path, context1_plan = _extract_context1_plan(
        prepared,
        context1_collapse_plan_path,
    )
    context1_bytes = context1_plan.get("bytes")
    if not isinstance(context1_bytes, dict):
        context1_bytes = {}
    context1_build_work = context1_plan.get("build_work")
    if not isinstance(context1_build_work, dict):
        context1_build_work = {}
    context1_layers_supported = _optional_int(context1_plan.get("layers_supported"))
    context1_cache_bytes = _optional_int(context1_bytes.get("chosen_cache_total"))
    context1_current_o_proj_bytes = _optional_int(
        context1_bytes.get("current_o_proj_storage_per_token")
    )
    context1_ratio = (
        context1_cache_bytes / context1_current_o_proj_bytes
        if context1_cache_bytes is not None
        and context1_current_o_proj_bytes is not None
        and context1_current_o_proj_bytes > 0
        else None
    )
    context1_build_fma_per_layer = _optional_int(
        context1_build_work.get("fma_per_layer")
    )
    context1_build_fma_total = _optional_int(context1_build_work.get("fma_total"))
    latent_plan = None
    try:
        latent_plan = build_latent_value_collapse_plan(prepared, dtype="BF16")
    except LatentValueCollapseError:
        latent_plan = None

    telemetry = _extract_decode_telemetry(decode_telemetry_path)
    observed_token_count = _optional_int(telemetry.get("token_count"))
    observed_total_with_logits = _optional_float(
        telemetry.get("total_with_logits_seconds")
    )
    observed_attn_output_bytes = _optional_float(telemetry.get("attn_output_bytes"))
    observed_attn_output_read = _optional_float(
        telemetry.get("attn_output_read_seconds")
    )
    observed_attn_output_projection = _optional_float(
        telemetry.get("attn_output_projection_seconds")
    )
    attn_output_bytes_per_token = (
        observed_attn_output_bytes / observed_token_count
        if observed_attn_output_bytes is not None
        and observed_token_count is not None
        and observed_token_count > 0
        else None
    )
    attn_output_read_per_token = (
        observed_attn_output_read / observed_token_count
        if observed_attn_output_read is not None
        and observed_token_count is not None
        and observed_token_count > 0
        else None
    )
    attn_output_projection_per_token = (
        observed_attn_output_projection / observed_token_count
        if observed_attn_output_projection is not None
        and observed_token_count is not None
        and observed_token_count > 0
        else None
    )
    observed_baseline_tps = (
        observed_token_count / observed_total_with_logits
        if observed_token_count is not None
        and observed_token_count > 0
        and observed_total_with_logits is not None
        and observed_total_with_logits > 0.0
        else None
    )
    removable_attn_seconds = None
    if observed_attn_output_read is not None or observed_attn_output_projection is not None:
        removable_attn_seconds = (observed_attn_output_read or 0.0) + (
            observed_attn_output_projection or 0.0
        )
    projected_context1_total = None
    projected_context1_tps = None
    projected_context1_speedup = None
    projected_exact_per_head_total = None
    projected_exact_per_head_tps = None
    projected_exact_per_head_speedup = None
    if (
        observed_total_with_logits is not None
        and observed_token_count is not None
        and observed_token_count > 0
        and removable_attn_seconds is not None
        and context1_ratio is not None
    ):
        removable_per_token = removable_attn_seconds / observed_token_count
        replacement_per_token = removable_per_token * context1_ratio
        saving_per_token = max(0.0, removable_per_token - replacement_per_token)
        projected_context1_total = max(
            0.0,
            observed_total_with_logits - saving_per_token * observed_token_count,
        )
        if projected_context1_total > 0.0:
            projected_context1_tps = observed_token_count / projected_context1_total
        if observed_baseline_tps is not None and observed_baseline_tps > 0.0:
            if projected_context1_tps is not None:
                projected_context1_speedup = (
                    projected_context1_tps / observed_baseline_tps
                )
    if (
        observed_total_with_logits is not None
        and observed_token_count is not None
        and observed_token_count > 0
        and latent_plan is not None
        and observed_attn_output_read is not None
        and observed_attn_output_projection is not None
        and latent_plan.exact_per_head_cache_read_ratio is not None
        and latent_plan.exact_per_head_runtime_fma_ratio is not None
    ):
        replacement = (
            observed_attn_output_read * latent_plan.exact_per_head_cache_read_ratio
            + observed_attn_output_projection
            * latent_plan.exact_per_head_runtime_fma_ratio
        )
        projected_exact_per_head_total = max(
            0.0,
            observed_total_with_logits
            - observed_attn_output_read
            - observed_attn_output_projection
            + replacement,
        )
        if projected_exact_per_head_total > 0.0:
            projected_exact_per_head_tps = (
                observed_token_count / projected_exact_per_head_total
            )
        if (
            observed_baseline_tps is not None
            and observed_baseline_tps > 0.0
            and projected_exact_per_head_tps is not None
        ):
            projected_exact_per_head_speedup = (
                projected_exact_per_head_tps / observed_baseline_tps
            )
    collapse_decision = _collapse_efficiency_decision(
        context1_speedup=projected_context1_speedup,
        exact_per_head_speedup=projected_exact_per_head_speedup,
        independent_head_groups=(
            latent_plan.independent_attention_head_groups
            if latent_plan is not None
            else None
        ),
        break_even_head_groups=(
            latent_plan.break_even_shared_head_groups
            if latent_plan is not None
            else None
        ),
    )
    if collapse_decision and "exact_all_context_per_head_cache_not_viable" in collapse_decision:
        next_runtime_step = (
            "keep the context=1 cache as a narrow measured fast path, do not "
            "build a general per-head collapsed cache, and continue the "
            "Flash-MoE-shaped persistent C/Metal runtime with fused "
            "MLA/attention-output scheduling and prompt prefill integration"
        )
    else:
        next_runtime_step = (
            "keep decode inside the persistent JSONL runtime, make optional "
            "debug intermediates in-process state, then fold prompt prefill "
            "into that same process"
        )
    (
        target_reference_tps,
        target_reference_source,
        target_met,
        target_gap_multiplier,
        target_decision,
    ) = _target_decision(
        target_tokens_per_second=target_tokens_per_second,
        candidates=(
            ("observed_smoke", observed_tps),
            ("observed_decode_telemetry_with_logits", observed_baseline_tps),
            ("projected_context1_only_with_logits", projected_context1_tps),
            (
                "projected_exact_all_context_per_head_with_logits",
                projected_exact_per_head_tps,
            ),
        ),
    )

    return MetalViabilityReport(
        prepared_dir=prepared,
        expert_layer_count=len(slot_bytes),
        top_k=top_k,
        expert_slot_bytes_min=min(slot_bytes),
        expert_slot_bytes_max=max(slot_bytes),
        routed_expert_read_bytes_per_token=routed_read_bytes,
        routed_expert_read_gib_per_token=routed_read_gib,
        qwen_flash_moe_reference_read_bytes_per_token=reference_read_bytes,
        glm_to_qwen_flash_moe_read_ratio=read_ratio,
        ssd_read_gib_per_second=resolved_ssd,
        routed_read_lower_bound_seconds_per_token=lower_bound,
        observed_decode_seconds_per_token=observed_decode_seconds,
        observed_tokens_per_second=observed_tps,
        observed_generated_token_ids=generated_ids,
        observed_estimated_read_bytes=_optional_int(
            token_result.get("estimated_read_bytes")
        ),
        observed_estimated_logits_read_bytes=_optional_int(
            token_result.get("estimated_logits_read_bytes")
        ),
        observed_estimated_expert_read_bytes=_optional_int(
            token_result.get("estimated_expert_read_bytes")
        ),
        decision=_decision(
            observed_decode_seconds=observed_decode_seconds,
            lower_bound_seconds=lower_bound,
        ),
        next_runtime_step=next_runtime_step,
        context1_collapse_plan_path=context1_path,
        context1_layers_supported=context1_layers_supported,
        context1_cache_bytes=context1_cache_bytes,
        context1_current_o_proj_bytes_per_token=context1_current_o_proj_bytes,
        context1_cache_read_ratio=context1_ratio,
        context1_build_fma_per_layer=context1_build_fma_per_layer,
        context1_build_fma_total=context1_build_fma_total,
        observed_decode_token_count=observed_token_count,
        observed_attn_output_bytes_per_token=attn_output_bytes_per_token,
        observed_attn_output_read_seconds_per_token=attn_output_read_per_token,
        observed_attn_output_projection_seconds_per_token=(
            attn_output_projection_per_token
        ),
        projected_context1_only_total_with_logits_seconds=projected_context1_total,
        projected_context1_only_tokens_per_second=projected_context1_tps,
        projected_context1_only_speedup=projected_context1_speedup,
        exact_all_context_per_head_cache_bytes=(
            latent_plan.exact_per_head_cache_bytes if latent_plan is not None else None
        ),
        exact_all_context_per_head_cache_read_ratio=(
            latent_plan.exact_per_head_cache_read_ratio
            if latent_plan is not None
            else None
        ),
        exact_all_context_per_head_int4_floor_bytes=(
            latent_plan.exact_per_head_int4_floor_bytes
            if latent_plan is not None
            else None
        ),
        exact_all_context_per_head_int4_floor_read_ratio=(
            latent_plan.exact_per_head_int4_floor_read_ratio
            if latent_plan is not None
            else None
        ),
        exact_all_context_per_head_runtime_fma_ratio=(
            latent_plan.exact_per_head_runtime_fma_ratio
            if latent_plan is not None
            else None
        ),
        exact_all_context_break_even_shared_head_groups=(
            latent_plan.break_even_shared_head_groups
            if latent_plan is not None
            else None
        ),
        exact_all_context_independent_attention_head_groups=(
            latent_plan.independent_attention_head_groups
            if latent_plan is not None
            else None
        ),
        projected_exact_all_context_per_head_total_with_logits_seconds=(
            projected_exact_per_head_total
        ),
        projected_exact_all_context_per_head_tokens_per_second=(
            projected_exact_per_head_tps
        ),
        projected_exact_all_context_per_head_speedup=(
            projected_exact_per_head_speedup
        ),
        efficiency_rewrite_decision=collapse_decision,
        target_tokens_per_second=target_tokens_per_second,
        target_reference_tokens_per_second=target_reference_tps,
        target_reference_source=target_reference_source,
        target_met=target_met,
        target_gap_multiplier=target_gap_multiplier,
        target_decision=target_decision,
    )
