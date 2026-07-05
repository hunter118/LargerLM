#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


GIB = 1024**3

TOP_LEVEL_TIMING_FIELDS = {
    "attn_projection": "attn_projection_elapsed_seconds",
    "mla_attention": "mla_attention_elapsed_seconds",
    "mlp": "mlp_elapsed_seconds",
    "final_logits": "final_logits_elapsed_seconds",
}

NESTED_TIMING_FIELDS = {
    "mla_cache_read": "mla_attention_cache_read_seconds",
    "mla_value_read": "mla_attention_value_read_seconds",
    "mla_kernel": "mla_attention_kernel_seconds",
    "mla_output_write": "mla_attention_output_write_seconds",
    "attn_output_read": "attn_output_read_seconds",
    "attn_output_projection_kernel": "attn_output_projection_kernel_seconds",
    "post_attn_norm_weight_read": "post_attn_norm_weight_read_seconds",
    "router_read": "router_read_seconds",
    "router_kernel": "router_kernel_seconds",
    "expert_read": "expert_read_seconds",
    "shared_read": "shared_read_seconds",
    "shared_prefetch": "shared_prefetch_seconds",
    "moe_mlp_kernel": "moe_mlp_kernel_seconds",
    "moe_mlp_output_write": "moe_mlp_output_write_seconds",
    "moe_mlp_overhead": "moe_mlp_overhead_seconds",
    "layer_overhead": "layer_overhead_seconds",
    "dense_mlp": "dense_mlp_elapsed_seconds",
    "moe_mlp": "moe_mlp_elapsed_seconds",
}

SUM_FIELDS = {
    "elapsed_seconds",
    "layer_elapsed_seconds",
    "final_logits_elapsed_seconds",
    "final_logits_bytes_read",
    "final_logits_lm_head_bytes_read",
    "final_logits_read_seconds",
    "final_logits_kernel_seconds",
    "attn_projection_elapsed_seconds",
    "attn_projection_synchronous_wait_count",
    "attn_projection_async_submitted_count",
    "mla_attention_elapsed_seconds",
    "mla_attention_cache_read_seconds",
    "mla_attention_value_read_seconds",
    "mla_attention_kernel_seconds",
    "mla_attention_output_write_seconds",
    "attn_output_elapsed_seconds",
    "attn_output_read_seconds",
    "attn_output_projection_kernel_seconds",
    "post_attn_norm_weight_read_seconds",
    "router_read_seconds",
    "router_kernel_seconds",
    "mlp_elapsed_seconds",
    "dense_mlp_elapsed_seconds",
    "moe_mlp_elapsed_seconds",
    "expert_read_seconds",
    "shared_read_seconds",
    "shared_prefetch_seconds",
    "moe_mlp_kernel_seconds",
    "moe_mlp_output_write_seconds",
    "moe_mlp_overhead_seconds",
    "layer_overhead_seconds",
    "attn_projection_command_buffer_count",
    "rope_mla_command_buffer_count",
    "attn_output_command_buffer_count",
    "attn_output_context1_o_proj_cache_count",
    "attn_output_resident_mmap_backed_count",
    "post_attn_norm_command_buffer_count",
    "router_command_buffer_count",
    "post_attn_norm_router_command_buffer_count",
    "dense_mlp_command_buffer_count",
    "dense_mlp_synchronous_wait_count",
    "dense_mlp_async_submitted_count",
    "moe_mlp_command_buffer_count",
    "moe_mlp_synchronous_wait_count",
    "attn_output_norm_router_fused_count",
    "rope_mla_attn_output_norm_router_fused_count",
    "rope_mla_input_buffer_direct_count",
    "attn_output_buffer_direct_count",
    "moe_mlp_input_buffer_direct_count",
    "layer_input_buffer_direct_count",
    "command_buffer_count",
    "synchronous_wait_count_estimate",
    "expert_bytes_read",
    "dense_mlp_bytes_read",
    "shared_bytes_read",
    "shared_prefetch_used_count",
    "attn_output_bytes_read",
    "post_attn_norm_weight_bytes_read",
    "router_bytes_read",
    "router_correction_bias_bytes_read",
    "expert_read_dispatch_count",
    "expert_read_task_count",
    "expert_read_pool_dispatch_count",
    "expert_read_serial_dispatch_count",
}

MAX_FIELDS = {
    "layer_count",
    "dense_layer_count",
    "moe_layer_count",
    "expert_read_max_task_count",
    "expert_read_max_worker_count",
}


class DecodeTelemetryReportError(RuntimeError):
    """Raised when a decode telemetry report cannot be built."""


def _load_json(path: str | Path) -> dict[str, Any]:
    if str(path) == "-":
        raw = sys.stdin.read()
        label = "stdin"
    else:
        label = str(path)
        raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecodeTelemetryReportError(f"failed to parse {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecodeTelemetryReportError(f"{label} must contain a JSON object")
    return payload


def _number(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DecodeTelemetryReportError(f"{key} must be numeric") from exc


def _integer(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DecodeTelemetryReportError(f"{key} must be an integer") from exc


def _sum_step_payloads(steps: list[Any]) -> dict[str, Any]:
    step_dicts = [step for step in steps if isinstance(step, dict)]
    aggregate: dict[str, Any] = {"generated_step_count": len(step_dicts)}
    aggregate["_present_moe_mlp_synchronous_wait_count"] = any(
        "moe_mlp_synchronous_wait_count" in step for step in step_dicts
    )
    aggregate["_present_dense_mlp_synchronous_wait_count"] = any(
        "dense_mlp_synchronous_wait_count" in step for step in step_dicts
    )
    for key in SUM_FIELDS:
        total = sum(_number(step, key) for step in step_dicts)
        if key.endswith("_count") or key.endswith("_bytes_read"):
            aggregate[key] = int(total)
        else:
            aggregate[key] = total
    aggregate["elapsed_seconds"] = sum(
        _number(step, "decode_elapsed_seconds") for step in step_dicts
    )
    aggregate["layer_elapsed_seconds"] = aggregate["elapsed_seconds"]
    for key in MAX_FIELDS:
        aggregate[key] = max((_integer(step, key) for step in step_dicts), default=0)
    return aggregate


def _numeric_values(payload: dict[str, Any], key: str) -> list[float]:
    value = payload.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    else:
        values = [value]
    numbers: list[float] = []
    for index, item in enumerate(values):
        if item is None:
            numbers.append(0.0)
            continue
        try:
            numbers.append(float(item))
        except (TypeError, ValueError) as exc:
            raise DecodeTelemetryReportError(
                f"{key}[{index}] must be numeric"
            ) from exc
    return numbers


def _flat_generation_values(payload: dict[str, Any], key: str) -> list[float]:
    if key == "elapsed_seconds":
        return _numeric_values(payload, "decode_elapsed_seconds")
    if key == "final_logits_elapsed_seconds":
        return _numeric_values(payload, "final_logits_elapsed_seconds")
    if key == "layer_elapsed_seconds":
        values = _numeric_values(payload, "decode_layer_elapsed_seconds")
        return values if values else _numeric_values(payload, "decode_elapsed_seconds")
    values = _numeric_values(payload, f"decode_{key}")
    if values:
        return values
    return _numeric_values(payload, key)


def _sum_flat_generation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decode_elapsed = _numeric_values(payload, "decode_elapsed_seconds")
    generated_ids = payload.get("generated_token_ids")
    generated_count = len(generated_ids) if isinstance(generated_ids, list) else 0
    step_count = max(len(decode_elapsed), generated_count)
    aggregate: dict[str, Any] = {"generated_step_count": step_count}
    aggregate["_present_moe_mlp_synchronous_wait_count"] = bool(
        _numeric_values(payload, "decode_moe_mlp_synchronous_wait_count")
    )
    aggregate["_present_dense_mlp_synchronous_wait_count"] = bool(
        _numeric_values(payload, "decode_dense_mlp_synchronous_wait_count")
    )
    for key in SUM_FIELDS:
        total = sum(_flat_generation_values(payload, key))
        if key.endswith("_count") or key.endswith("_bytes_read"):
            aggregate[key] = int(total)
        else:
            aggregate[key] = total
    for key in MAX_FIELDS:
        values = _flat_generation_values(payload, key)
        aggregate[key] = int(max(values, default=0.0))
    return aggregate


def _is_flat_generation_payload(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("decode_elapsed_seconds"), list)
        or isinstance(payload.get("generated_token_ids"), list)
        and isinstance(payload.get("final_logits_elapsed_seconds"), list)
    )


def _select_decode_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    decode = payload.get("probe_decode_layers")
    if isinstance(decode, dict):
        return "probe_decode_layers", decode
    dims = payload.get("dims")
    if isinstance(dims, dict):
        return "smoke_dims", dims
    generate = payload.get("probe_generate")
    if isinstance(generate, dict) and isinstance(generate.get("steps"), list):
        return "probe_generate_steps", _sum_step_payloads(generate["steps"])
    steps = payload.get("steps")
    if isinstance(steps, list):
        return "steps", _sum_step_payloads(steps)
    response_payload = payload.get("payload")
    if isinstance(response_payload, dict) and isinstance(
        response_payload.get("steps"),
        list,
    ):
        return "payload_steps", _sum_step_payloads(response_payload["steps"])
    requests = payload.get("requests")
    if isinstance(requests, list):
        for request in reversed(requests):
            if not isinstance(request, dict):
                continue
            response_payload = request.get("payload")
            if not isinstance(response_payload, dict):
                continue
            response_steps = response_payload.get("steps")
            if isinstance(response_steps, list):
                return "request_last_steps", _sum_step_payloads(response_steps)
    if _is_flat_generation_payload(payload):
        return "metal_generate_result", _sum_flat_generation_payload(payload)
    return "payload", payload


def _timing_entry(label: str, seconds: float, denominator: float) -> dict[str, Any]:
    return {
        "label": label,
        "seconds": seconds,
        "share_of_elapsed": seconds / denominator if denominator > 0.0 else None,
    }


def _speedup_if_removed(elapsed: float, seconds: float) -> float | None:
    if elapsed <= 0.0 or seconds <= 0.0 or seconds >= elapsed:
        return None
    return elapsed / (elapsed - seconds)


def _frontier_entry(
    *,
    label: str,
    seconds: float,
    elapsed: float,
    token_count: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "seconds": seconds,
        "seconds_per_token": seconds / token_count if token_count > 0 else None,
        "share_of_decode": seconds / elapsed if elapsed > 0.0 else None,
        "speedup_if_removed_upper_bound": _speedup_if_removed(elapsed, seconds),
    }


def _optimization_frontier(
    *,
    decode: dict[str, Any],
    elapsed: float,
    token_count: int,
    command_buffer_count: int,
    wait_count_estimate: int,
    moe_wait_count_available: bool,
    pooled_read_ok: bool,
) -> dict[str, Any]:
    components = {
        "attention_output_combined": _number(decode, "attn_output_read_seconds")
        + _number(decode, "attn_output_projection_kernel_seconds"),
        "attn_output_projection_kernel": _number(
            decode,
            "attn_output_projection_kernel_seconds",
        ),
        "attn_output_read": _number(decode, "attn_output_read_seconds"),
        "expert_read": _number(decode, "expert_read_seconds"),
        "moe_mlp_overhead": _number(decode, "moe_mlp_overhead_seconds"),
        "layer_overhead": _number(decode, "layer_overhead_seconds"),
        "mla_value_read": _number(decode, "mla_attention_value_read_seconds"),
        "mla_kernel": _number(decode, "mla_attention_kernel_seconds"),
    }
    ranked = sorted(
        (
            _frontier_entry(
                label=label,
                seconds=seconds,
                elapsed=elapsed,
                token_count=token_count,
            )
            for label, seconds in components.items()
            if seconds > 0.0
        ),
        key=lambda item: float(item["seconds"] or 0.0),
        reverse=True,
    )
    seconds_per_token = {
        label: (seconds / token_count if token_count > 0 else None)
        for label, seconds in components.items()
        if seconds > 0.0
    }
    counts_per_token = {
        "command_buffers": (
            command_buffer_count / token_count if token_count > 0 else None
        ),
        "synchronous_waits_estimate": (
            wait_count_estimate / token_count if token_count > 0 else None
        ),
        "attn_projection_synchronous_waits": (
            _integer(decode, "attn_projection_synchronous_wait_count") / token_count
            if token_count > 0
            else None
        ),
        "moe_mlp_synchronous_waits": (
            _integer(decode, "moe_mlp_synchronous_wait_count") / token_count
            if token_count > 0 and moe_wait_count_available
            else None
        ),
    }
    if "_present_dense_mlp_synchronous_wait_count" in decode:
        dense_wait_count_available = bool(
            decode.get("_present_dense_mlp_synchronous_wait_count")
        )
    else:
        dense_wait_count_available = "dense_mlp_synchronous_wait_count" in decode
    dense_wait_count = (
        _integer(decode, "dense_mlp_synchronous_wait_count")
        if dense_wait_count_available
        else _integer(decode, "dense_mlp_command_buffer_count")
    )
    wait_counts = {
        "attn_projection": _integer(
            decode,
            "attn_projection_synchronous_wait_count",
        ),
        "rope_mla": _integer(decode, "rope_mla_command_buffer_count"),
        "attn_output": _integer(decode, "attn_output_command_buffer_count"),
        "post_attn_norm": _integer(
            decode,
            "post_attn_norm_command_buffer_count",
        ),
        "router": _integer(decode, "router_command_buffer_count"),
        "post_attn_norm_router": _integer(
            decode,
            "post_attn_norm_router_command_buffer_count",
        ),
        "dense_mlp": dense_wait_count,
        "moe_mlp": (
            _integer(decode, "moe_mlp_synchronous_wait_count")
            if moe_wait_count_available
            else 0
        ),
    }
    sync_waits_by_stage = [
        {
            "label": label,
            "count": count,
            "per_token": count / token_count if token_count > 0 else None,
        }
        for label, count in sorted(
            wait_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if count > 0
    ]

    recommendations: list[dict[str, Any]] = []
    attn_combined = components["attention_output_combined"]
    expert_read = components["expert_read"]
    if attn_combined > 0.0 and attn_combined >= expert_read:
        recommendations.append(
            {
                "code": "attention_output_stage_first",
                "reason": (
                    "attention output read+projection is at least as expensive "
                    "as routed expert reads"
                ),
                "seconds_per_token": (
                    attn_combined / token_count if token_count > 0 else None
                ),
            }
        )
    if (
        counts_per_token["command_buffers"] is not None
        and counts_per_token["command_buffers"] >= 150.0
    ) or (
        counts_per_token["synchronous_waits_estimate"] is not None
        and counts_per_token["synchronous_waits_estimate"] >= 50.0
    ):
        recommendations.append(
            {
                "code": "reduce_command_buffer_and_wait_count",
                "reason": (
                    "decode still launches or waits on many per-layer command "
                    "buffers per token"
                ),
                "command_buffers_per_token": counts_per_token["command_buffers"],
                "sync_waits_per_token": counts_per_token[
                    "synchronous_waits_estimate"
                ],
            }
        )
    if not pooled_read_ok:
        recommendations.append(
            {
                "code": "fix_expert_read_pooling",
                "reason": "expert reads fell back to serial dispatch",
            }
        )
    if expert_read > 0.0 and elapsed > 0.0 and expert_read / elapsed >= 0.25:
        recommendations.append(
            {
                "code": "expert_io_bandwidth",
                "reason": "expert read time is at least 25% of decode elapsed",
                "share_of_decode": expert_read / elapsed,
            }
        )

    return {
        "ranked_seconds": ranked,
        "seconds_per_token": seconds_per_token,
        "counts_per_token": counts_per_token,
        "sync_waits_by_stage": sync_waits_by_stage,
        "primary_recommendation": recommendations[0] if recommendations else None,
        "recommendations": recommendations,
    }


def _layer_command_buffer_count(layer: dict[str, Any]) -> int:
    return sum(
        _integer(layer, key)
        for key in (
            "attn_projection_command_buffer_count",
            "rope_mla_command_buffer_count",
            "attn_output_command_buffer_count",
            "post_attn_norm_command_buffer_count",
            "router_command_buffer_count",
            "post_attn_norm_router_command_buffer_count",
            "dense_mlp_command_buffer_count",
            "moe_mlp_command_buffer_count",
        )
    )


def _layer_wait_count_estimate(layer: dict[str, Any]) -> int:
    return (
        _integer(layer, "attn_projection_synchronous_wait_count")
        + _integer(layer, "rope_mla_command_buffer_count")
        + _integer(layer, "attn_output_command_buffer_count")
        + _integer(layer, "post_attn_norm_command_buffer_count")
        + _integer(layer, "router_command_buffer_count")
        + _integer(layer, "post_attn_norm_router_command_buffer_count")
        + _integer(
            layer,
            "dense_mlp_synchronous_wait_count",
            _integer(layer, "dense_mlp_command_buffer_count"),
        )
        + _integer(layer, "moe_mlp_synchronous_wait_count")
    )


def _layer_frontier_entry(layer: dict[str, Any]) -> dict[str, Any]:
    attn_output_read = _number(layer, "attn_output_read_seconds")
    attn_output_kernel = _number(layer, "attn_output_projection_kernel_seconds")
    attn_output_combined = attn_output_read + attn_output_kernel
    return {
        "layer": _integer(layer, "layer"),
        "kind": layer.get("kind") if isinstance(layer.get("kind"), str) else None,
        "elapsed_seconds": _number(layer, "elapsed_seconds"),
        "attention_output_combined_seconds": attn_output_combined,
        "attn_output_read_seconds": attn_output_read,
        "attn_output_projection_kernel_seconds": attn_output_kernel,
        "expert_read_seconds": _number(layer, "expert_read_seconds"),
        "moe_mlp_overhead_seconds": _number(layer, "moe_mlp_overhead_seconds"),
        "layer_overhead_seconds": _number(layer, "layer_overhead_seconds"),
        "command_buffer_count": _layer_command_buffer_count(layer),
        "synchronous_wait_count_estimate": _layer_wait_count_estimate(layer),
        "attn_projection_synchronous_wait_count": _integer(
            layer,
            "attn_projection_synchronous_wait_count",
        ),
        "attn_projection_async_submitted": bool(
            layer.get("attn_projection_async_submitted")
        ),
        "attn_output_bytes_read": _integer(layer, "attn_output_bytes_read"),
        "expert_bytes_read": _integer(layer, "expert_bytes_read"),
        "router_topk_backend": (
            layer.get("router_topk_backend")
            if isinstance(layer.get("router_topk_backend"), str)
            else None
        ),
        "rope_mla_attn_output_norm_router_fused": bool(
            layer.get("rope_mla_attn_output_norm_router_fused")
        ),
        "moe_mlp_input_buffer_direct": bool(layer.get("moe_mlp_input_buffer_direct")),
        "dense_mlp_synchronous_wait_count": _integer(
            layer,
            "dense_mlp_synchronous_wait_count",
            _integer(layer, "dense_mlp_command_buffer_count"),
        ),
        "dense_mlp_async_submitted": bool(layer.get("dense_mlp_async_submitted")),
    }


def _top_layer_entries(
    entries: list[dict[str, Any]],
    key: str,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    return [
        item
        for item in sorted(
            entries,
            key=lambda entry: float(entry.get(key) or 0.0),
            reverse=True,
        )[:limit]
        if float(item.get(key) or 0.0) > 0.0
    ]


def _layer_frontier(decode: dict[str, Any]) -> dict[str, Any] | None:
    raw_layers = decode.get("layers")
    if not isinstance(raw_layers, list):
        return None
    layer_dicts = [item for item in raw_layers if isinstance(item, dict)]
    if not layer_dicts:
        return None
    entries = [_layer_frontier_entry(layer) for layer in layer_dicts]
    return {
        "layer_count": len(entries),
        "top_attention_output_layers": _top_layer_entries(
            entries,
            "attention_output_combined_seconds",
        ),
        "top_layer_overhead_layers": _top_layer_entries(
            entries,
            "layer_overhead_seconds",
        ),
        "top_moe_overhead_layers": _top_layer_entries(
            entries,
            "moe_mlp_overhead_seconds",
        ),
        "top_command_buffer_layers": _top_layer_entries(
            entries,
            "command_buffer_count",
        ),
        "top_sync_wait_layers": _top_layer_entries(
            entries,
            "synchronous_wait_count_estimate",
        ),
    }


def _attn_output_timing_label(decode: dict[str, Any]) -> str:
    if _integer(decode, "rope_mla_attn_output_norm_router_fused_count") > 0:
        return "rope_mla_attn_output_norm_router_fused"
    if _integer(decode, "attn_output_norm_router_fused_count") > 0:
        return "attn_output_norm_router_fused"
    return "attn_output"


def build_decode_telemetry_report(
    payload: dict[str, Any],
    *,
    reference_cold_read_gib_per_second: float | None = None,
) -> dict[str, Any]:
    source_kind, decode = _select_decode_payload(payload)
    elapsed = _number(decode, "elapsed_seconds")
    final_logits = _number(decode, "final_logits_elapsed_seconds")
    layer_elapsed = _number(decode, "layer_elapsed_seconds", elapsed)
    total_with_logits = elapsed + final_logits
    token_count = max(1, _integer(decode, "generated_step_count", 1))

    top_level_fields = {
        **TOP_LEVEL_TIMING_FIELDS,
        _attn_output_timing_label(decode): "attn_output_elapsed_seconds",
    }
    top_level = [
        _timing_entry(label, _number(decode, key), elapsed)
        for label, key in top_level_fields.items()
        if _number(decode, key) > 0.0
    ]
    nested = [
        _timing_entry(label, _number(decode, key), elapsed)
        for label, key in NESTED_TIMING_FIELDS.items()
        if _number(decode, key) > 0.0
    ]
    primary = max(top_level, key=lambda item: item["seconds"], default=None)

    expert_bytes = _integer(decode, "expert_bytes_read")
    expert_read_seconds = _number(decode, "expert_read_seconds")
    observed_read_gib_per_second = (
        (expert_bytes / GIB) / expert_read_seconds
        if expert_bytes > 0 and expert_read_seconds > 0.0
        else None
    )
    cold_floor_seconds = None
    if reference_cold_read_gib_per_second:
        cold_floor_seconds = (expert_bytes / GIB) / reference_cold_read_gib_per_second

    serial_dispatches = _integer(decode, "expert_read_serial_dispatch_count")
    pool_dispatches = _integer(decode, "expert_read_pool_dispatch_count")
    task_count = _integer(decode, "expert_read_task_count")
    pooled_read_ok = task_count == 0 or (pool_dispatches > 0 and serial_dispatches == 0)

    top_level_known = sum(
        _number(decode, key)
        for key in (
            "attn_projection_elapsed_seconds",
            "mla_attention_elapsed_seconds",
            "attn_output_elapsed_seconds",
            "mlp_elapsed_seconds",
        )
    )
    unattributed_layer_seconds = layer_elapsed - top_level_known
    command_buffer_count = _integer(decode, "command_buffer_count")
    wait_count_estimate = _integer(decode, "synchronous_wait_count_estimate")
    if "_present_moe_mlp_synchronous_wait_count" in decode:
        moe_wait_count_available = bool(
            decode.get("_present_moe_mlp_synchronous_wait_count")
        )
    else:
        moe_wait_count_available = "moe_mlp_synchronous_wait_count" in decode
    if "_present_dense_mlp_synchronous_wait_count" in decode:
        dense_wait_count_available = bool(
            decode.get("_present_dense_mlp_synchronous_wait_count")
        )
    else:
        dense_wait_count_available = "dense_mlp_synchronous_wait_count" in decode
    dense_mlp_wait_count = (
        _integer(decode, "dense_mlp_synchronous_wait_count")
        if dense_wait_count_available
        else _integer(decode, "dense_mlp_command_buffer_count")
    )

    return {
        "schema": "largerlm.decode_telemetry_report.v1",
        "source_kind": source_kind,
        "token_count": token_count,
        "elapsed_seconds": elapsed,
        "final_logits_elapsed_seconds": final_logits,
        "total_with_final_logits_seconds": total_with_logits,
        "decode_tokens_per_second": (
            (token_count / elapsed) if elapsed > 0.0 else None
        ),
        "with_final_logits_tokens_per_second": (
            (token_count / total_with_logits) if total_with_logits > 0.0 else None
        ),
        "layer_count": _integer(decode, "layer_count", len(decode.get("layers", []))),
        "dense_layer_count": _integer(
            decode,
            "dense_layer_count",
            len(decode.get("dense_layers", [])),
        ),
        "moe_layer_count": _integer(
            decode,
            "moe_layer_count",
            _integer(decode, "expert_read_dispatch_count"),
        ),
        "bytes": {
            "expert_bytes_read": expert_bytes,
            "expert_gib_read": expert_bytes / GIB,
            "dense_mlp_bytes_read": _integer(decode, "dense_mlp_bytes_read"),
            "shared_bytes_read": _integer(decode, "shared_bytes_read"),
            "attn_output_bytes_read": _integer(
                decode,
                "attn_output_bytes_read",
            ),
            "post_attn_norm_weight_bytes_read": _integer(
                decode,
                "post_attn_norm_weight_bytes_read",
            ),
            "router_bytes_read": _integer(decode, "router_bytes_read"),
            "router_correction_bias_bytes_read": _integer(
                decode,
                "router_correction_bias_bytes_read",
            ),
            "final_logits_bytes_read": _integer(
                decode,
                "final_logits_bytes_read",
            ),
            "final_logits_lm_head_bytes_read": _integer(
                decode,
                "final_logits_lm_head_bytes_read",
            ),
        },
        "expert_read": {
            "seconds": expert_read_seconds,
            "observed_gib_per_second": observed_read_gib_per_second,
            "reference_cold_read_gib_per_second": reference_cold_read_gib_per_second,
            "reference_cold_read_floor_seconds": cold_floor_seconds,
            "dispatch_count": _integer(decode, "expert_read_dispatch_count"),
            "task_count": task_count,
            "max_task_count": _integer(decode, "expert_read_max_task_count"),
            "max_worker_count": _integer(decode, "expert_read_max_worker_count"),
            "pool_dispatch_count": pool_dispatches,
            "serial_dispatch_count": serial_dispatches,
            "pooled_read_ok": pooled_read_ok,
        },
        "shared_read": {
            "bytes_read": _integer(decode, "shared_bytes_read"),
            "seconds": _number(decode, "shared_read_seconds"),
            "prefetch_seconds": _number(decode, "shared_prefetch_seconds"),
            "prefetch_used_count": _integer(decode, "shared_prefetch_used_count"),
        },
        "command_buffers": {
            "count": command_buffer_count,
            "synchronous_wait_count_estimate": wait_count_estimate,
            "attn_projection_synchronous_wait_count": _integer(
                decode,
                "attn_projection_synchronous_wait_count",
            ),
            "attn_projection_async_submitted_count": _integer(
                decode,
                "attn_projection_async_submitted_count",
            ),
            "attn_projection_count": _integer(
                decode,
                "attn_projection_command_buffer_count",
            ),
            "rope_mla_count": _integer(decode, "rope_mla_command_buffer_count"),
            "attn_output_count": _integer(
                decode,
                "attn_output_command_buffer_count",
            ),
            "attn_output_context1_o_proj_cache_count": _integer(
                decode,
                "attn_output_context1_o_proj_cache_count",
            ),
            "attn_output_resident_mmap_backed_count": _integer(
                decode,
                "attn_output_resident_mmap_backed_count",
            ),
            "post_attn_norm_count": _integer(
                decode,
                "post_attn_norm_command_buffer_count",
            ),
            "router_count": _integer(decode, "router_command_buffer_count"),
            "post_attn_norm_router_count": _integer(
                decode,
                "post_attn_norm_router_command_buffer_count",
            ),
            "dense_mlp_count": _integer(
                decode,
                "dense_mlp_command_buffer_count",
            ),
            "dense_mlp_synchronous_wait_count": dense_mlp_wait_count,
            "dense_mlp_async_submitted_count": _integer(
                decode,
                "dense_mlp_async_submitted_count",
            ),
            "moe_mlp_count": _integer(decode, "moe_mlp_command_buffer_count"),
            "moe_mlp_synchronous_wait_count": _integer(
                decode,
                "moe_mlp_synchronous_wait_count",
            ),
            "moe_mlp_synchronous_wait_count_available": moe_wait_count_available,
            "attn_output_norm_router_fused_count": _integer(
                decode,
                "attn_output_norm_router_fused_count",
            ),
            "rope_mla_attn_output_norm_router_fused_count": _integer(
                decode,
                "rope_mla_attn_output_norm_router_fused_count",
            ),
            "rope_mla_input_buffer_direct_count": _integer(
                decode,
                "rope_mla_input_buffer_direct_count",
            ),
            "attn_output_buffer_direct_count": _integer(
                decode,
                "attn_output_buffer_direct_count",
            ),
            "moe_mlp_input_buffer_direct_count": _integer(
                decode,
                "moe_mlp_input_buffer_direct_count",
            ),
            "layer_input_buffer_direct_count": _integer(
                decode,
                "layer_input_buffer_direct_count",
            ),
        },
        "timing": {
            "top_level_components": top_level,
            "nested_components": nested,
            "primary_top_level_bottleneck": primary,
            "unattributed_layer_seconds": unattributed_layer_seconds,
        },
        "optimization_frontier": _optimization_frontier(
            decode=decode,
            elapsed=elapsed,
            token_count=token_count,
            command_buffer_count=command_buffer_count,
            wait_count_estimate=wait_count_estimate,
            moe_wait_count_available=moe_wait_count_available,
            pooled_read_ok=pooled_read_ok,
        ),
        "layer_frontier": _layer_frontier(decode),
    }


def _print_text(report: dict[str, Any]) -> None:
    print("LargerLM decode telemetry report")
    print(f"  source:                {report['source_kind']}")
    print(f"  decode elapsed:        {report['elapsed_seconds']:.6f} s")
    print(
        "  final logits:          "
        f"{report['final_logits_elapsed_seconds']:.6f} s"
    )
    if report["with_final_logits_tokens_per_second"]:
        print(
            "  throughput:            "
            f"{report['with_final_logits_tokens_per_second']:.3f} tok/s incl logits"
        )
    bytes_payload = report["bytes"]
    print(
        "  expert read:           "
        f"{bytes_payload['expert_gib_read']:.3f} GiB in "
        f"{report['expert_read']['seconds']:.6f} s"
    )
    observed = report["expert_read"]["observed_gib_per_second"]
    if observed is not None:
        print(f"  expert read bandwidth: {observed:.3f} GiB/s")
    attn_output_bytes = bytes_payload.get("attn_output_bytes_read", 0)
    if attn_output_bytes:
        nested = {
            item["label"]: item["seconds"]
            for item in report["timing"]["nested_components"]
        }
        print(
            "  attn output read:      "
            f"{attn_output_bytes / GIB:.3f} GiB in "
            f"{nested.get('attn_output_read', 0.0):.6f} s"
        )
    final_logits_bytes = bytes_payload.get("final_logits_bytes_read", 0)
    final_logits_lm_head_bytes = bytes_payload.get(
        "final_logits_lm_head_bytes_read",
        0,
    )
    if final_logits_bytes or final_logits_lm_head_bytes:
        print(
            "  final logits read:     "
            f"{final_logits_bytes / GIB:.3f} GiB total, "
            f"{final_logits_lm_head_bytes / GIB:.3f} GiB lm_head"
        )
    router_bytes = bytes_payload.get("router_bytes_read", 0)
    if router_bytes:
        nested = {
            item["label"]: item["seconds"]
            for item in report["timing"]["nested_components"]
        }
        print(
            "  router read:           "
            f"{router_bytes / GIB:.3f} GiB in "
            f"{nested.get('router_read', 0.0):.6f} s"
        )
    shared = report.get("shared_read", {})
    shared_bytes = int(shared.get("bytes_read") or 0)
    if shared_bytes:
        print(
            "  shared read:           "
            f"{shared_bytes / GIB:.3f} GiB in "
            f"{float(shared.get('seconds') or 0.0):.6f} s, "
            f"{int(shared.get('prefetch_used_count') or 0)} prefetched"
        )
    print(
        "  read dispatch:         "
        f"{report['expert_read']['task_count']} tasks, "
        f"{report['expert_read']['pool_dispatch_count']} pooled, "
        f"{report['expert_read']['serial_dispatch_count']} serial"
    )
    command_buffers = report["command_buffers"]
    if command_buffers["count"]:
        print(
            "  command buffers:       "
            f"{command_buffers['count']} total, "
            f"{command_buffers['synchronous_wait_count_estimate']} sync waits est"
        )
    attn_proj_waits = command_buffers.get("attn_projection_synchronous_wait_count")
    attn_proj_async = command_buffers.get("attn_projection_async_submitted_count")
    attn_proj_cmds = command_buffers.get("attn_projection_count")
    if attn_proj_cmds and (attn_proj_waits or attn_proj_async):
        print(
            "  Attn proj waits:       "
            f"{attn_proj_waits} waits, {attn_proj_async} async / {attn_proj_cmds}"
        )
    moe_waits = command_buffers.get("moe_mlp_synchronous_wait_count")
    moe_cmds = command_buffers.get("moe_mlp_count")
    if moe_cmds and command_buffers.get("moe_mlp_synchronous_wait_count_available"):
        print(f"  MoE command waits:    {moe_waits} / {moe_cmds}")
    direct_count = command_buffers.get("moe_mlp_input_buffer_direct_count")
    if direct_count:
        print(f"  MoE direct input:      {direct_count} layer buffers")
    dense_waits = command_buffers.get("dense_mlp_synchronous_wait_count")
    dense_async = command_buffers.get("dense_mlp_async_submitted_count")
    dense_cmds = command_buffers.get("dense_mlp_count")
    if dense_cmds and (dense_waits or dense_async):
        print(
            "  Dense MLP waits:       "
            f"{dense_waits} waits, {dense_async} async / {dense_cmds}"
        )
    attn_direct_count = command_buffers.get("attn_output_buffer_direct_count")
    if attn_direct_count:
        print(f"  Attn output direct:    {attn_direct_count} layer buffers")
    context1_count = command_buffers.get("attn_output_context1_o_proj_cache_count")
    if context1_count:
        print(f"  Context1 o_proj cache: {context1_count} layer hits")
    resident_mmap_count = command_buffers.get(
        "attn_output_resident_mmap_backed_count"
    )
    if resident_mmap_count:
        print(f"  Attn output mmap:      {resident_mmap_count} layer hits")
    rope_direct_count = command_buffers.get("rope_mla_input_buffer_direct_count")
    if rope_direct_count:
        print(f"  RoPE/MLA direct:      {rope_direct_count} layer buffers")
    layer_direct_count = command_buffers.get("layer_input_buffer_direct_count")
    if layer_direct_count:
        print(f"  Layer input direct:   {layer_direct_count} layer buffers")
    attn_norm_router_fused = command_buffers.get(
        "attn_output_norm_router_fused_count"
    )
    if attn_norm_router_fused:
        print(
            "  Attn+norm/router fuse: "
            f"{attn_norm_router_fused} layer command buffers"
        )
    rope_mla_attn_fused = command_buffers.get(
        "rope_mla_attn_output_norm_router_fused_count"
    )
    if rope_mla_attn_fused:
        print(
            "  RoPE+MLA+attn fuse:   "
            f"{rope_mla_attn_fused} layer command buffers"
        )
    primary = report["timing"]["primary_top_level_bottleneck"]
    if primary:
        share = primary["share_of_elapsed"]
        print(
            "  primary bottleneck:    "
            f"{primary['label']} {primary['seconds']:.6f} s"
            + (f" ({share:.1%})" if share is not None else "")
        )
    frontier = report.get("optimization_frontier")
    if isinstance(frontier, dict):
        recommendation = frontier.get("primary_recommendation")
        if isinstance(recommendation, dict):
            print(
                "  next frontier:         "
                f"{recommendation.get('code')} - {recommendation.get('reason')}"
            )
    layer_frontier = report.get("layer_frontier")
    if isinstance(layer_frontier, dict):
        hot_layers = layer_frontier.get("top_attention_output_layers")
        if isinstance(hot_layers, list) and hot_layers:
            first = hot_layers[0]
            print(
                "  hottest attn layer:    "
                f"{first.get('layer')} "
                f"{float(first.get('attention_output_combined_seconds') or 0.0):.6f} s"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize glm_moe_infer decode telemetry JSON."
    )
    parser.add_argument("json_path", help="Telemetry JSON path, or '-' for stdin.")
    parser.add_argument(
        "--reference-cold-read-gib-per-second",
        type=float,
        default=None,
        help="Optional cold-read bandwidth for an expert-I/O floor estimate.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine JSON.")
    args = parser.parse_args(argv)

    try:
        payload = _load_json(args.json_path)
        report = build_decode_telemetry_report(
            payload,
            reference_cold_read_gib_per_second=(
                args.reference_cold_read_gib_per_second
            ),
        )
    except DecodeTelemetryReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
