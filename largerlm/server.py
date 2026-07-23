from __future__ import annotations

import json
import math
import shutil
import threading
import time
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .config import ConfigError, ModelConfig, load_config
from .context1_o_proj_cache import (
    Context1OProjCacheError,
    load_context1_o_proj_cache_layout,
    load_context1_o_proj_cache_progress,
)
from .decode_cache import DecodeCacheError, load_decode_cache_layout
from .generation_guard import (
    GenerationGuardError,
    check_generation_runtime,
    estimate_prompt_prefill_live_memory,
    system_memory_snapshot,
)
from .layout import DEFAULT_EXPERT_COMPONENTS, MXFP4_EXPERT_COMPONENTS
from .prepared import (
    PreparedManifest,
    load_prepared_manifest,
    validate_layout_model_config_sha256,
)
from .prepared_lock import (
    PreparedRunLockError,
    acquire_prepared_run_lock_path,
    inspect_prepared_run_lock_path,
    prepared_run_lock_path_for_manifest,
)
from .metal_generate import (
    MetalGenerateError,
    MetalGenerateServerSession,
    MetalTokenGenerationResult,
    generate_metal_token_ids,
)
from .metal_text_generator import (
    MetalTextGenerationError,
    MetalTextGenerationResult,
    generate_metal_text,
)
from .prefill_execute import (
    AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    AUTO_MPSGRAPH_MIN_DIM,
    PREFILL_LINEAR_AUTO_MPP_BACKEND,
    PREFILL_LINEAR_ACCELERATED_BACKENDS,
    PREFILL_LINEAR_F32_CONVERSION_BACKENDS,
    PREFILL_LINEAR_MPSGRAPH_DTYPES,
    _resident_linear_matrix_scratch,
    is_prompt_prefill_resident_linear_backend_tensor,
    is_prompt_prefill_router_gate_tensor,
)
from .prefill_backend import (
    DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
    default_prefill_backend_probe_path,
    evaluate_prefill_acceleration_requirement,
    inspect_prefill_backend,
    prefill_acceleration_runtime_gaps,
    prefill_acceleration_runtimes,
    prefill_neural_accelerator_status,
    selectable_accelerated_prefill_backends,
    suggested_prefill_acceleration_flags,
    validated_accelerated_prefill_backends,
)
from .prefill_plan import (
    DEFAULT_MPP_MIN_TOKENS,
    MPP_TENSOR_OPS_MIN_MATRIX_DIM,
    build_prefill_cache_io_plan,
)
from .prompt_prefill import (
    _streamed_routed_expert_linear_accounting_for_layers,
    prompt_prefill_acceleration_failure_reason,
)
from .routed_read import (
    RoutedExpertReadError,
    combine_prefill_guard_flags,
    estimate_routed_expert_read,
    estimate_routed_prefill_chunk_frontier,
    estimate_routed_stage_temp,
    format_routed_read_guard_flag_float,
    minimum_prompt_chunk_tokens_for_routed_read_limits,
    suggest_decode_routed_read_guard_flags,
    suggest_routed_read_guard_flags,
    suggest_routed_stage_temp_guard_flags,
    _load_routed_expert_layer_specs,
)
from .resident_affine import (
    ResidentAffineLayoutError,
    resident_affine_int4_layout_info,
)
from .runtime_check import RuntimeCheckError
from .safetensors import is_routed_expert_tensor_for_moe_layers
from .text_generator import TextGenerationError, TextGenerationResult, generate_text
from .token_generator import (
    TokenGenerationResult,
    TokenGeneratorError,
    _auto_prefill_prompt_chunk_plan,
    generate_token_ids,
    prefill_prompt_chunk_plan_drift_summary,
)
from .tokenizer import TokenizerError, load_tokenizer, render_chat_prompt


class PreparedServerError(RuntimeError):
    """Raised when the prepared HTTP generation server cannot handle a request."""


class PreparedRequestCheckError(PreparedServerError):
    """Raised when a request check fails with structured diagnostics."""

    def __init__(self, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


AFFINE_INT4_WEIGHT_DTYPES = ("U32", "UINT32")
AFFINE_INT4_META_DTYPES = ("BF16", "BFLOAT16", "F16", "FLOAT16")
MXFP4_SCALE_DTYPES = ("U8", "UINT8")
MXFP4_GROUP_SIZE = 32
AFFINE_INT4_EXPERT_QUANTIZATIONS = frozenset(
    {"largerlm-affine-int4", "mlx-affine-int4"}
)
MXFP4_EXPERT_QUANTIZATIONS = frozenset({"mlx-mxfp4"})


def prepared_request_check_failure_reason(check: object) -> str | None:
    if not isinstance(check, dict):
        return "request check did not return structured diagnostics"
    if check.get("ok") is True:
        return None
    for key in ("reason", "error", "message"):
        value = check.get(key)
        if isinstance(value, str) and value:
            return value
    code = check.get("code")
    if isinstance(code, str) and code:
        return f"request check {code} did not pass"
    return f"request check returned ok={check.get('ok')!r}"


def require_prepared_request_check_ok(check: object) -> None:
    reason = prepared_request_check_failure_reason(check)
    if reason is None:
        return
    payload = check if isinstance(check, dict) else {"request_check": check}
    raise PreparedRequestCheckError(reason, payload=payload)


def _mps_graph_runtime_available_from_backend(backend: object) -> bool:
    value = getattr(backend, "mps_graph_runtime_available", None)
    if isinstance(value, bool):
        return value
    return bool(getattr(backend, "mps_graph_matmul_declared", False))


def _numeric_config_value(config: object, name: str) -> float:
    value = getattr(config, name)
    if isinstance(value, bool):
        raise PreparedServerError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PreparedServerError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise PreparedServerError(f"{name} must be finite")
    return parsed


def _require_positive_config_value(config: object, name: str) -> None:
    if _numeric_config_value(config, name) <= 0:
        raise PreparedServerError(f"{name} must be positive")


def _require_nonnegative_config_value(config: object, name: str) -> None:
    if _numeric_config_value(config, name) < 0:
        raise PreparedServerError(f"{name} must be non-negative")


def _system_memory_health() -> dict[str, Any] | None:
    try:
        snapshot = system_memory_snapshot()
    except Exception:
        return None
    if snapshot is None:
        return None
    return {
        "total_bytes": snapshot.total_bytes,
        "available_bytes": snapshot.available_bytes,
        "page_size": snapshot.page_size,
        "source": snapshot.source,
    }


def _system_disk_usage(path: Path) -> dict[str, Any] | None:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _sum_known_bytes(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    return int(sum(value for value in values if value is not None))


def _prepared_storage_health(prepared: PreparedManifest) -> dict[str, Any]:
    decode_cache_extra = None
    if (
        prepared.decode_cache_file_bytes is not None
        and prepared.decode_cache_layout_bytes is not None
    ):
        decode_cache_extra = (
            prepared.decode_cache_file_bytes - prepared.decode_cache_layout_bytes
        )
    expert_layout_backing_validated = prepared.expert_layout_bytes is not None
    resident_layout_backing_validated = prepared.resident_layout_bytes is not None
    decode_cache_file_exact_size = decode_cache_extra == 0
    prepared_storage_validated = (
        expert_layout_backing_validated
        and resident_layout_backing_validated
        and decode_cache_file_exact_size
    )
    return {
        "prepared_storage_validated": prepared_storage_validated,
        "expert_layout_backing_validated": expert_layout_backing_validated,
        "resident_layout_backing_validated": resident_layout_backing_validated,
        "decode_cache_file_exact_size": decode_cache_file_exact_size,
        "expert_layout_bytes": prepared.expert_layout_bytes,
        "resident_layout_bytes": prepared.resident_layout_bytes,
        "decode_cache_layout_bytes": prepared.decode_cache_layout_bytes,
        "decode_cache_file_bytes": prepared.decode_cache_file_bytes,
        "decode_cache_file_extra_bytes": decode_cache_extra,
        "prepared_total_layout_bytes": _sum_known_bytes(
            prepared.expert_layout_bytes,
            prepared.resident_layout_bytes,
            prepared.decode_cache_layout_bytes,
        ),
        "prepared_total_file_bytes": _sum_known_bytes(
            prepared.expert_layout_bytes,
            prepared.resident_layout_bytes,
            prepared.decode_cache_file_bytes,
        ),
        "resident_and_cache_layout_bytes": _sum_known_bytes(
            prepared.resident_layout_bytes,
            prepared.decode_cache_layout_bytes,
        ),
        "recommended_max_live_working_set_bytes": (
            prepared.recommended_max_live_working_set_bytes
        ),
        "recommended_min_free_unified_memory_bytes": (
            prepared.recommended_min_free_unified_memory_bytes
        ),
        "recommended_required_available_memory_bytes": _sum_known_bytes(
            prepared.recommended_max_live_working_set_bytes,
            prepared.recommended_min_free_unified_memory_bytes,
        ),
        "expert_quantization": prepared.expert_quantization,
        "expert_group_size": prepared.expert_group_size,
        "expert_layout_quantization": prepared.expert_layout_quantization,
        "expert_layout_group_size": prepared.expert_layout_group_size,
        "prepare_hardware_chip_name": prepared.prepare_hardware_chip_name,
        "prepare_hardware_unified_memory_bytes": (
            prepared.prepare_hardware_unified_memory_bytes
        ),
        "prepare_hardware_gpu_cores": prepared.prepare_hardware_gpu_cores,
        "prepare_hardware_apple_silicon_generation": (
            prepared.prepare_hardware_apple_silicon_generation
        ),
        "prepare_hardware_apple_silicon_tier": (
            prepared.prepare_hardware_apple_silicon_tier
        ),
        "prepare_effective_unified_memory_bytes": (
            prepared.prepare_effective_unified_memory_bytes
        ),
        "prepare_effective_unified_memory_source": (
            prepared.prepare_effective_unified_memory_source
        ),
        "prepare_system_reserve_bytes": prepared.prepare_system_reserve_bytes,
        "prepare_auto_context_from_budget": (
            prepared.prepare_auto_context_from_budget
        ),
        "prepare_requested_max_context_tokens": (
            prepared.prepare_requested_max_context_tokens
        ),
        "prepare_resolved_max_context_tokens": (
            prepared.prepare_resolved_max_context_tokens
        ),
        "prepare_decode_cache_budget_bytes": (
            prepared.prepare_decode_cache_budget_bytes
        ),
        "prepare_decode_cache_safe_context_tokens": (
            prepared.prepare_decode_cache_safe_context_tokens
        ),
        "prepare_effective_max_cache_bytes": (
            prepared.prepare_effective_max_cache_bytes
        ),
        "prepare_model_max_position_embeddings": (
            prepared.prepare_model_max_position_embeddings
        ),
        "prepare_cache_dtype": prepared.prepare_cache_dtype,
        "prepare_cache_alignment": prepared.prepare_cache_alignment,
        "prepare_flags_applied": prepared.prepare_flags_applied,
        "prepare_flags_source": prepared.prepare_flags_source,
        "prepare_flags_path": prepared.prepare_flags_path,
        "prepare_flags_sha256": prepared.prepare_flags_sha256,
        "prepare_expert_pack_chunk_size_bytes": (
            prepared.prepare_expert_pack_chunk_size_bytes
        ),
        "prepare_expert_pack_estimated_peak_heap_bytes": (
            prepared.prepare_expert_pack_estimated_peak_heap_bytes
        ),
        "prepare_expert_pack_max_heap_bytes": (
            prepared.prepare_expert_pack_max_heap_bytes
        ),
        "prepare_raw_quantization_extra_heap_bytes": (
            prepared.prepare_raw_quantization_extra_heap_bytes
        ),
        "prepare_raw_quantization_max_source_block_bytes": (
            prepared.prepare_raw_quantization_max_source_block_bytes
        ),
        "prepare_raw_quantization_max_output_block_bytes": (
            prepared.prepare_raw_quantization_max_output_block_bytes
        ),
        "prepare_raw_quantization_max_rows_per_block": (
            prepared.prepare_raw_quantization_max_rows_per_block
        ),
        "prepare_resident_component_alias_source_tensor_count": (
            prepared.prepare_resident_component_alias_source_tensor_count
        ),
        "prepare_resident_component_alias_renamed_tensor_count": (
            prepared.prepare_resident_component_alias_renamed_tensor_count
        ),
        "prepare_resident_component_alias_bytes": (
            prepared.prepare_resident_component_alias_bytes
        ),
        "prepare_resident_fused_gate_up_source_tensor_count": (
            prepared.prepare_resident_fused_gate_up_source_tensor_count
        ),
        "prepare_resident_fused_gate_up_expanded_tensor_count": (
            prepared.prepare_resident_fused_gate_up_expanded_tensor_count
        ),
        "prepare_resident_fused_gate_up_expanded_bytes": (
            prepared.prepare_resident_fused_gate_up_expanded_bytes
        ),
        "prepare_public_glm_5_2_shape_required": (
            prepared.prepare_public_glm_5_2_shape_required
        ),
        "prepare_public_glm_5_2_shape_matches": (
            prepared.prepare_public_glm_5_2_shape_matches
        ),
        "prepare_public_glm_5_2_shape_mismatched_fields": (
            prepared.prepare_public_glm_5_2_shape_mismatched_fields
        ),
        "prepare_cold_read_gib_per_second": (
            prepared.prepare_cold_read_gib_per_second
        ),
        "prepare_cold_read_source": prepared.prepare_cold_read_source,
        "prepare_cold_read_benchmark_path": (
            prepared.prepare_cold_read_benchmark_path
        ),
        "prepare_cold_read_benchmark_requested_bytes": (
            prepared.prepare_cold_read_benchmark_requested_bytes
        ),
        "prepare_cold_read_benchmark_measured_bytes": (
            prepared.prepare_cold_read_benchmark_measured_bytes
        ),
        "prepare_cold_read_benchmark_elapsed_seconds": (
            prepared.prepare_cold_read_benchmark_elapsed_seconds
        ),
        "model_config_sha256": prepared.model_config_sha256,
    }


def _prepare_live_memory_health(
    prepared: PreparedManifest,
    system_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    required = prepared.prepare_live_memory_required_available_memory_bytes
    current_available = (
        system_memory.get("available_bytes") if system_memory is not None else None
    )
    current_ok = (
        int(current_available) >= required
        if isinstance(current_available, int) and required is not None
        else None
    )
    return {
        "recorded": required is not None,
        "estimated_live_working_set_bytes": (
            prepared.prepare_live_memory_estimated_live_working_set_bytes
        ),
        "min_available_memory_bytes": (
            prepared.prepare_live_memory_min_available_memory_bytes
        ),
        "required_available_memory_bytes": required,
        "prepare_system_available_memory_bytes": (
            prepared.prepare_live_memory_system_available_memory_bytes
        ),
        "prepare_system_total_memory_bytes": (
            prepared.prepare_live_memory_system_total_bytes
        ),
        "prepare_system_memory_source": (
            prepared.prepare_live_memory_system_source
        ),
        "current_system_available_memory_bytes": current_available,
        "current_system_memory_source": (
            system_memory.get("source") if system_memory is not None else None
        ),
        "current_available_meets_prepare_live_requirement": current_ok,
    }


def _suggest_prepared_launch_guard_flags(
    prepared: PreparedManifest,
    *,
    require_memory_profile: bool = False,
) -> dict[str, object] | None:
    argv: list[str] = []
    suggested: dict[str, object] = {"source": "prepared_manifest"}
    if require_memory_profile:
        suggested["require_prepared_memory_profile"] = True
        argv.append("--require-prepared-memory-profile")
    max_live_bytes = prepared.recommended_max_live_working_set_bytes
    if max_live_bytes is not None:
        max_live_mib = max_live_bytes / 1024**2
        suggested["recommended_max_live_working_set_bytes"] = max_live_bytes
        suggested["max_live_working_set_mib"] = max_live_mib
        argv.extend(
            [
                "--max-live-working-set-mib",
                format_routed_read_guard_flag_float(max_live_mib),
            ]
        )
    min_free_bytes = prepared.recommended_min_free_unified_memory_bytes
    if min_free_bytes is not None:
        min_free_gib = min_free_bytes / 1024**3
        suggested["recommended_min_free_unified_memory_bytes"] = min_free_bytes
        suggested["min_free_unified_memory_gib"] = min_free_gib
        argv.extend(
            [
                "--min-free-unified-memory-gib",
                format_routed_read_guard_flag_float(min_free_gib),
            ]
        )
    required_available = _sum_known_bytes(max_live_bytes, min_free_bytes)
    if required_available is not None:
        suggested["recommended_required_available_memory_bytes"] = required_available
    if not argv:
        return None
    suggested["argv"] = tuple(argv)
    return suggested


def _with_decode_mla_key_cache_guard_flag(
    payload: dict[str, object] | None,
    *,
    source: str,
    decode_mla_key_cache: bool = False,
) -> dict[str, object] | None:
    if not decode_mla_key_cache:
        return payload
    if payload is None:
        payload = {"source": source, "argv": ()}
    argv = list(payload.get("argv") or ())
    if "--decode-mla-key-cache" not in argv:
        argv.append("--decode-mla-key-cache")
    payload = dict(payload)
    payload["decode_mla_key_cache"] = True
    payload["argv"] = tuple(argv)
    return payload


def _suggest_prepared_decode_guard_flags(
    readiness: dict[str, Any],
    *,
    ssd_read_gib_per_second: float,
    decode_mla_key_cache: bool = False,
) -> dict[str, object] | None:
    if readiness.get("ok") is not True:
        return None
    payload = suggest_decode_routed_read_guard_flags(
        read_bytes_per_token=readiness.get(
            "expected_decode_token_routed_expert_read_bytes"
        ),
        ssd_read_gib_per_second=ssd_read_gib_per_second,
        source="prepared_health",
    )
    return _with_decode_mla_key_cache_guard_flag(
        payload,
        source="prepared_health",
        decode_mla_key_cache=decode_mla_key_cache,
    )


def _suggest_prepared_ssd_read_flags(
    prepared: PreparedManifest,
    *,
    ssd_read_gib_per_second: float,
) -> dict[str, object] | None:
    if ssd_read_gib_per_second <= 0:
        return None
    manifest_speed = prepared.prepare_cold_read_gib_per_second
    source = (
        "prepared_manifest"
        if manifest_speed == ssd_read_gib_per_second
        else "configured_runtime"
    )
    return {
        "source": source,
        "prefill_ssd_read_gib_per_second": ssd_read_gib_per_second,
        "prepare_cold_read_gib_per_second": manifest_speed,
        "prepare_cold_read_source": prepared.prepare_cold_read_source,
        "matches_prepare_cold_read": manifest_speed == ssd_read_gib_per_second,
        "argv": (
            "--prefill-ssd-read-gib-s",
            format_routed_read_guard_flag_float(ssd_read_gib_per_second),
        ),
    }


def _suggest_prefill_backend_probe_flags(
    *,
    compile_mpp_probe: bool,
    run_mpp_probe: bool,
    run_mpsgraph_probe: bool,
    source: str,
) -> dict[str, object] | None:
    if not compile_mpp_probe and not run_mpp_probe and not run_mpsgraph_probe:
        return None
    argv: list[str] = []
    payload: dict[str, object] = {"source": source}
    if compile_mpp_probe:
        payload["compile_mpp_probe"] = True
        argv.append("--compile-mpp-probe")
    if run_mpp_probe:
        payload["run_mpp_probe"] = True
        argv.append("--run-mpp-probe")
    if run_mpsgraph_probe:
        payload["run_mpsgraph_probe"] = True
        argv.append("--run-mpsgraph-probe")
    payload["argv"] = tuple(argv)
    return payload


def _suggest_prefill_runtime_policy_flags(
    config: "PreparedServerConfig",
    *,
    source: str,
) -> dict[str, object] | None:
    if (
        config.prefill_linear_backend == "auto"
        and config.prefill_mpsgraph_min_batch_tokens == AUTO_MPSGRAPH_MIN_BATCH_TOKENS
        and config.prefill_mpsgraph_min_matrix_dim == AUTO_MPSGRAPH_MIN_DIM
        and config.prefill_min_accelerated_flop_fraction == 0.0
        and config.prefill_router_hybrid_margin_threshold == 0.0
        and not config.require_prefill_acceleration
        and not config.allow_router_gate_only_prefill_acceleration
        and not config.prefill_mla_key_cache
    ):
        return None
    argv: list[str] = []
    payload: dict[str, object] = {"source": source}
    if config.prefill_linear_backend != "auto":
        payload["prefill_linear_backend"] = config.prefill_linear_backend
        argv.extend(["--prefill-linear-backend", config.prefill_linear_backend])
    if (
        config.prefill_mpsgraph_min_batch_tokens != AUTO_MPSGRAPH_MIN_BATCH_TOKENS
        or config.prefill_mpsgraph_min_matrix_dim != AUTO_MPSGRAPH_MIN_DIM
    ):
        payload["prefill_mpsgraph_min_batch_tokens"] = (
            config.prefill_mpsgraph_min_batch_tokens
        )
        payload["prefill_mpsgraph_min_matrix_dim"] = (
            config.prefill_mpsgraph_min_matrix_dim
        )
        argv.extend(
            [
                "--prefill-mpsgraph-min-batch-tokens",
                str(config.prefill_mpsgraph_min_batch_tokens),
                "--prefill-mpsgraph-min-matrix-dim",
                str(config.prefill_mpsgraph_min_matrix_dim),
            ]
        )
    if config.prefill_min_accelerated_flop_fraction > 0.0:
        payload["prefill_min_accelerated_flop_fraction"] = (
            config.prefill_min_accelerated_flop_fraction
        )
        argv.extend(
            [
                "--prefill-min-accelerated-flop-fraction",
                format_routed_read_guard_flag_float(
                    config.prefill_min_accelerated_flop_fraction
                ),
            ]
        )
    if config.prefill_router_hybrid_margin_threshold > 0.0:
        payload["prefill_router_hybrid_margin_threshold"] = (
            config.prefill_router_hybrid_margin_threshold
        )
        argv.extend(
            [
                "--prefill-router-hybrid-margin-threshold",
                format_routed_read_guard_flag_float(
                    config.prefill_router_hybrid_margin_threshold
                ),
            ]
        )
    if config.require_prefill_acceleration:
        payload["require_prefill_acceleration"] = True
        argv.append("--require-prefill-acceleration")
    if config.allow_router_gate_only_prefill_acceleration:
        payload["allow_router_gate_only_prefill_acceleration"] = True
        argv.append("--allow-router-gate-only-prefill-acceleration")
    if config.prefill_mla_key_cache:
        payload["prefill_mla_key_cache"] = True
        argv.append("--prefill-mla-key-cache")
    payload["argv"] = tuple(argv)
    return payload


def _suggest_prefill_copy_policy_flags(
    config: "PreparedServerConfig",
    *,
    source: str,
) -> dict[str, object] | None:
    if config.prefill_copy_chunk_mib == 8.0:
        return None
    return {
        "source": source,
        "prefill_copy_chunk_mib": config.prefill_copy_chunk_mib,
        "argv": (
            "--prefill-copy-chunk-mib",
            format_routed_read_guard_flag_float(config.prefill_copy_chunk_mib),
        ),
    }


def _suggest_glm_4bit_guard_flags(
    readiness: dict[str, Any],
    *,
    source: str,
) -> dict[str, object] | None:
    if readiness.get("ok") is not True:
        return None
    return {
        "source": source,
        "require_glm_4bit": True,
        "argv": ("--require-glm-4bit",),
    }


def _suggest_public_glm_5_2_shape_guard_flags(
    readiness: dict[str, Any],
    *,
    source: str,
) -> dict[str, object] | None:
    if (
        readiness.get("ok") is not True
        or readiness.get("matches_public_glm_5_2_shape") is not True
    ):
        return None
    return {
        "source": source,
        "require_public_glm_5_2_shape": True,
        "argv": ("--require-public-glm-5-2-shape",),
    }


def suggest_final_logits_flags(
    *,
    metal_final_logits: bool,
    source: str,
) -> dict[str, object] | None:
    if not metal_final_logits:
        return None
    return {
        "source": source,
        "metal_final_logits": True,
        "argv": ("--metal-final-logits",),
    }


def suggest_metal_runtime_mmap_final_logits_flags(
    *,
    enabled: bool,
    metal_runtime_generation: bool,
    source: str,
) -> dict[str, object] | None:
    if not metal_runtime_generation:
        return None
    payload: dict[str, object] = {
        "source": source,
        "metal_runtime_mmap_final_logits": True,
        "argv": ("--metal-runtime-mmap-final-logits",),
    }
    if not enabled:
        payload["recommended"] = True
    return payload


def suggest_metal_runtime_context1_o_proj_cache_flags(
    *,
    layout_path: Path | None,
    cache_file_path: Path | None,
    metal_runtime_generation: bool,
    source: str,
    validated_cache: dict[str, Any] | None = None,
) -> dict[str, object] | None:
    if not metal_runtime_generation or layout_path is None:
        return None
    argv: list[str] = [
        "--metal-runtime-context1-o-proj-cache-layout",
        str(layout_path),
    ]
    payload: dict[str, object] = {
        "source": source,
        "metal_runtime_context1_o_proj_cache": True,
        "metal_runtime_context1_o_proj_cache_layout": str(layout_path),
        "argv": tuple(argv),
    }
    if cache_file_path is not None:
        argv.extend(
            [
                "--metal-runtime-context1-o-proj-cache-file",
                str(cache_file_path),
            ]
        )
        payload["metal_runtime_context1_o_proj_cache_file"] = str(cache_file_path)
        payload["argv"] = tuple(argv)
    if isinstance(validated_cache, dict):
        payload["validated"] = validated_cache.get("ok") is True
        payload["cache_file_override"] = validated_cache.get("cache_file_override") is True
        runtime_file = validated_cache.get("runtime_cache_file")
        if isinstance(runtime_file, str):
            payload["runtime_cache_file"] = runtime_file
        runtime_bytes = validated_cache.get("runtime_cache_file_bytes")
        if isinstance(runtime_bytes, int):
            payload["runtime_cache_file_bytes"] = runtime_bytes
    return payload


def _recommended_mla_kv_b_cache_mib(total_bytes: int) -> float | None:
    if total_bytes <= 0:
        return None
    mib = max(1, math.ceil(total_bytes / 1024**2))
    if mib >= 1024:
        mib = ((mib + 255) // 256) * 256
    return float(mib)


def _estimate_mla_kv_b_f32_cache_plan(
    config: ModelConfig,
) -> dict[str, object] | None:
    if config.kv_lora_rank is None:
        return None
    kv_b_out = config.attention_kv_b_output_dim
    if kv_b_out is None:
        return None
    try:
        layer_count = int(config.num_hidden_layers)
        kv_lora = int(config.kv_lora_rank)
        kv_b_out = int(kv_b_out)
    except (TypeError, ValueError):
        return None
    if layer_count <= 0 or kv_lora <= 0 or kv_b_out <= 0:
        return None
    per_layer_bytes = kv_b_out * kv_lora * 4
    total_bytes = per_layer_bytes * layer_count
    recommended_mib = _recommended_mla_kv_b_cache_mib(total_bytes)
    if recommended_mib is None:
        return None
    return {
        "source": "model_config",
        "num_hidden_layers": layer_count,
        "kv_lora_rank": kv_lora,
        "attention_kv_b_output_dim": kv_b_out,
        "per_layer_bytes": per_layer_bytes,
        "estimated_full_cache_bytes": total_bytes,
        "estimated_full_cache_mib": total_bytes / 1024**2,
        "recommended_max_cache_mib": recommended_mib,
    }


def suggest_metal_runtime_mla_kv_b_cache_flags(
    *,
    enabled: bool,
    max_cache_mib: float,
    source: str,
    recommended_cache_mib: float | None = None,
    estimated_cache_bytes: int | None = None,
) -> dict[str, object] | None:
    if not enabled:
        if recommended_cache_mib is None or recommended_cache_mib <= 0.0:
            return None
        max_cache_mib = float(recommended_cache_mib)
        value = format_routed_read_guard_flag_float(max_cache_mib)
        payload: dict[str, object] = {
            "source": source,
            "recommended": True,
            "metal_runtime_cache_mla_kv_b_f32": True,
            "metal_runtime_max_mla_kv_b_cache_mib": max_cache_mib,
            "argv": (
                "--metal-runtime-cache-mla-kv-b-f32",
                "--metal-runtime-max-mla-kv-b-cache-mib",
                value,
            ),
        }
        if estimated_cache_bytes is not None:
            payload["estimated_full_cache_bytes"] = int(estimated_cache_bytes)
            payload["estimated_full_cache_mib"] = int(estimated_cache_bytes) / 1024**2
        return payload
    value = format_routed_read_guard_flag_float(max_cache_mib)
    return {
        "source": source,
        "metal_runtime_cache_mla_kv_b_f32": True,
        "metal_runtime_max_mla_kv_b_cache_mib": float(max_cache_mib),
        "argv": (
            "--metal-runtime-cache-mla-kv-b-f32",
            "--metal-runtime-max-mla-kv-b-cache-mib",
            value,
        ),
    }


_LAUNCH_PROFILE_VALUELESS_FLAGS = frozenset(
    {
        "--require-prefill-acceleration",
        "--allow-non-accelerated-prefill-launch-audit",
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--compile-mpp-probe",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
        "--no-runtime-preflight",
        "--prefill-mla-key-cache",
        "--prefill-expert-stage-tiling",
        "--prefill-persistent-moe-plan-server",
        "--prefill-persistent-resident-linear-server",
        "--prefill-persistent-attention-projection-server",
        "--prefill-persistent-attention-output-server",
        "--prefill-persistent-shared-expert-server",
        "--prefill-persistent-rope-split-server",
        "--prefill-persistent-mla-attention-server",
        "--prefill-persistent-rmsnorm-server",
        "--metal-final-logits",
        "--metal-runtime-cache-mla-kv-b-f32",
        "--metal-runtime-mmap-final-logits",
    }
)


def _without_launch_profile_flag(
    argv: object,
    flag_to_remove: str,
) -> tuple[str, ...]:
    if not isinstance(argv, (list, tuple)):
        return ()
    items = [str(item) for item in argv]
    out: list[str] = []
    index = 0
    while index < len(items):
        flag = items[index]
        has_value = flag not in _LAUNCH_PROFILE_VALUELESS_FLAGS
        if flag == flag_to_remove:
            index += 2 if has_value and index + 1 < len(items) else 1
            continue
        out.append(flag)
        if has_value and index + 1 < len(items):
            out.append(items[index + 1])
            index += 2
        else:
            index += 1
    return tuple(out)


def _prefill_acceleration_launch_profile_flags(
    config: "PreparedServerConfig",
    acceleration_flags: object,
) -> dict[str, object] | None:
    if not isinstance(acceleration_flags, dict):
        return None
    if config.prefill_linear_backend != "auto":
        return acceleration_flags

    argv = _without_launch_profile_flag(
        acceleration_flags.get("argv"),
        "--prefill-linear-backend",
    )
    if not argv:
        return None
    filtered = dict(acceleration_flags)
    filtered.pop("prefill_linear_backend", None)
    filtered["argv"] = argv
    filtered["prefill_linear_backend_policy"] = "auto"
    return filtered


def _suggest_prefill_persistent_moe_plan_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_moe_plan_server": True,
        "argv": ("--prefill-persistent-moe-plan-server",),
    }


def _suggest_prefill_persistent_resident_linear_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_resident_linear_server": True,
        "argv": ("--prefill-persistent-resident-linear-server",),
    }


def _suggest_prefill_persistent_attention_projection_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_attention_projection_server": True,
        "argv": ("--prefill-persistent-attention-projection-server",),
    }


def _suggest_prefill_persistent_attention_output_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_attention_output_server": True,
        "argv": ("--prefill-persistent-attention-output-server",),
    }


def _suggest_prefill_persistent_shared_expert_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_shared_expert_server": True,
        "argv": ("--prefill-persistent-shared-expert-server",),
    }


def _suggest_prefill_persistent_rope_split_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_rope_split_server": True,
        "argv": ("--prefill-persistent-rope-split-server",),
    }


def _suggest_prefill_persistent_mla_attention_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_mla_attention_server": True,
        "argv": ("--prefill-persistent-mla-attention-server",),
    }


def _suggest_prefill_persistent_rmsnorm_server_flags(
    *,
    enabled: bool,
    source: str,
) -> dict[str, object] | None:
    if not enabled:
        return None
    return {
        "source": source,
        "prefill_persistent_rmsnorm_server": True,
        "argv": ("--prefill-persistent-rmsnorm-server",),
    }


def _suggest_prefill_moe_output_accumulator_flags(
    *,
    mode: str,
    source: str,
) -> dict[str, object] | None:
    if mode == "env":
        return None
    return {
        "source": source,
        "prefill_moe_output_accumulator": mode,
        "argv": ("--prefill-moe-output-accumulator", mode),
    }


def prepared_launch_profile_target(prepared: PreparedManifest) -> dict[str, object]:
    warnings: list[str] = []
    if prepared.model_config_sha256 is None:
        warnings.append(
            "model_config_sha256 is unavailable; profile matching falls back to layout bytes"
        )
    for field, value in (
        ("expert_layout_bytes", prepared.expert_layout_bytes),
        ("resident_layout_bytes", prepared.resident_layout_bytes),
        ("decode_cache_layout_bytes", prepared.decode_cache_layout_bytes),
        ("decode_cache_file_bytes", prepared.decode_cache_file_bytes),
        ("max_context_tokens", prepared.max_context_tokens),
        ("expert_quantization", prepared.expert_quantization),
        ("expert_group_size", prepared.expert_group_size),
    ):
        if value is None:
            warnings.append(f"{field} is unavailable for profile matching")
    prepare_flags_applied = prepared.prepare_flags_applied is True
    prepare_flags_source = prepared.prepare_flags_source if prepare_flags_applied else None
    prepare_flags_sha256 = prepared.prepare_flags_sha256 if prepare_flags_applied else None
    if prepare_flags_applied:
        if prepare_flags_source is None:
            warnings.append("prepare_flags_source is unavailable for profile matching")
        if prepare_flags_sha256 is None:
            warnings.append("prepare_flags_sha256 is unavailable for profile matching")
    strength = "strong" if not warnings else "weak"
    return {
        "prepared_manifest": str(prepared.manifest_path),
        "model_dir": str(prepared.model_dir),
        "identity_strength": strength,
        "identity_warnings": tuple(warnings),
        "model_config_sha256": prepared.model_config_sha256,
        "expert_layout_bytes": prepared.expert_layout_bytes,
        "resident_layout_bytes": prepared.resident_layout_bytes,
        "decode_cache_layout_bytes": prepared.decode_cache_layout_bytes,
        "decode_cache_file_bytes": prepared.decode_cache_file_bytes,
        "max_context_tokens": prepared.max_context_tokens,
        "expert_quantization": prepared.expert_quantization,
        "expert_group_size": prepared.expert_group_size,
        "prepare_expert_pack_chunk_size_bytes": (
            prepared.prepare_expert_pack_chunk_size_bytes
        ),
        "prepare_expert_pack_estimated_peak_heap_bytes": (
            prepared.prepare_expert_pack_estimated_peak_heap_bytes
        ),
        "prepare_expert_pack_max_heap_bytes": (
            prepared.prepare_expert_pack_max_heap_bytes
        ),
        "prepare_raw_quantization_extra_heap_bytes": (
            prepared.prepare_raw_quantization_extra_heap_bytes
        ),
        "prepare_raw_quantization_max_source_block_bytes": (
            prepared.prepare_raw_quantization_max_source_block_bytes
        ),
        "prepare_raw_quantization_max_output_block_bytes": (
            prepared.prepare_raw_quantization_max_output_block_bytes
        ),
        "prepare_raw_quantization_max_rows_per_block": (
            prepared.prepare_raw_quantization_max_rows_per_block
        ),
        "prepare_resident_component_alias_source_tensor_count": (
            prepared.prepare_resident_component_alias_source_tensor_count
        ),
        "prepare_resident_component_alias_renamed_tensor_count": (
            prepared.prepare_resident_component_alias_renamed_tensor_count
        ),
        "prepare_resident_component_alias_bytes": (
            prepared.prepare_resident_component_alias_bytes
        ),
        "prepare_resident_fused_gate_up_source_tensor_count": (
            prepared.prepare_resident_fused_gate_up_source_tensor_count
        ),
        "prepare_resident_fused_gate_up_expanded_tensor_count": (
            prepared.prepare_resident_fused_gate_up_expanded_tensor_count
        ),
        "prepare_resident_fused_gate_up_expanded_bytes": (
            prepared.prepare_resident_fused_gate_up_expanded_bytes
        ),
        "prepare_hardware_chip_name": prepared.prepare_hardware_chip_name,
        "prepare_hardware_unified_memory_bytes": (
            prepared.prepare_hardware_unified_memory_bytes
        ),
        "prepare_hardware_gpu_cores": prepared.prepare_hardware_gpu_cores,
        "prepare_hardware_apple_silicon_generation": (
            prepared.prepare_hardware_apple_silicon_generation
        ),
        "prepare_hardware_apple_silicon_tier": (
            prepared.prepare_hardware_apple_silicon_tier
        ),
        "prepare_flags_applied": prepare_flags_applied,
        "prepare_flags_source": prepare_flags_source,
        "prepare_flags_sha256": prepare_flags_sha256,
        "prepare_public_glm_5_2_shape_required": (
            prepared.prepare_public_glm_5_2_shape_required
        ),
        "prepare_public_glm_5_2_shape_matches": (
            prepared.prepare_public_glm_5_2_shape_matches
        ),
        "prepare_public_glm_5_2_shape_mismatched_fields": (
            prepared.prepare_public_glm_5_2_shape_mismatched_fields
        ),
    }


def _iter_profile_argv(argv: object) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(argv, (list, tuple)):
        return ()
    items = [str(item) for item in argv]
    parsed: list[tuple[str, str | None]] = []
    index = 0
    while index < len(items):
        flag = items[index]
        if not flag.startswith("--"):
            index += 1
            continue
        if flag in _LAUNCH_PROFILE_VALUELESS_FLAGS:
            parsed.append((flag, None))
            index += 1
            continue
        if index + 1 < len(items) and not items[index + 1].startswith("--"):
            parsed.append((flag, items[index + 1]))
            index += 2
            continue
        parsed.append((flag, None))
        index += 1
    return tuple(parsed)


def combine_suggested_launch_profile(
    *,
    launch_guard_flags: dict[str, object] | None = None,
    prepared_ssd_read_flags: dict[str, object] | None = None,
    prefill_backend_probe_flags: dict[str, object] | None = None,
    prefill_backend_policy_flags: dict[str, object] | None = None,
    prefill_runtime_policy_flags: dict[str, object] | None = None,
    prefill_copy_policy_flags: dict[str, object] | None = None,
    glm_4bit_guard_flags: dict[str, object] | None = None,
    public_glm_5_2_shape_guard_flags: dict[str, object] | None = None,
    prefill_acceleration_flags: dict[str, object] | None = None,
    prefill_guard_flags: dict[str, object] | None = None,
    prefill_persistent_moe_plan_server_flags: dict[str, object] | None = None,
    prefill_persistent_resident_linear_server_flags: dict[str, object] | None = None,
    prefill_persistent_attention_projection_server_flags: (
        dict[str, object] | None
    ) = None,
    prefill_persistent_attention_output_server_flags: (
        dict[str, object] | None
    ) = None,
    prefill_persistent_shared_expert_server_flags: (
        dict[str, object] | None
    ) = None,
    prefill_persistent_rope_split_server_flags: dict[str, object] | None = None,
    prefill_persistent_mla_attention_server_flags: dict[str, object] | None = None,
    prefill_persistent_rmsnorm_server_flags: dict[str, object] | None = None,
    prefill_moe_output_accumulator_flags: dict[str, object] | None = None,
    decode_guard_flags: dict[str, object] | None = None,
    metal_runtime_mla_kv_b_cache_flags: dict[str, object] | None = None,
    metal_runtime_mmap_final_logits_flags: dict[str, object] | None = None,
    metal_runtime_context1_o_proj_cache_flags: dict[str, object] | None = None,
    final_logits_flags: dict[str, object] | None = None,
    prepared_target: dict[str, object] | None = None,
    source: str,
) -> dict[str, object] | None:
    sections = (
        ("launch_guard_flags", launch_guard_flags),
        ("prepared_ssd_read_flags", prepared_ssd_read_flags),
        ("prefill_backend_probe_flags", prefill_backend_probe_flags),
        ("prefill_backend_policy_flags", prefill_backend_policy_flags),
        ("prefill_runtime_policy_flags", prefill_runtime_policy_flags),
        ("prefill_copy_policy_flags", prefill_copy_policy_flags),
        ("glm_4bit_guard_flags", glm_4bit_guard_flags),
        ("public_glm_5_2_shape_guard_flags", public_glm_5_2_shape_guard_flags),
        ("prefill_acceleration_flags", prefill_acceleration_flags),
        ("prefill_guard_flags", prefill_guard_flags),
        (
            "prefill_persistent_moe_plan_server_flags",
            prefill_persistent_moe_plan_server_flags,
        ),
        (
            "prefill_persistent_resident_linear_server_flags",
            prefill_persistent_resident_linear_server_flags,
        ),
        (
            "prefill_persistent_attention_projection_server_flags",
            prefill_persistent_attention_projection_server_flags,
        ),
        (
            "prefill_persistent_attention_output_server_flags",
            prefill_persistent_attention_output_server_flags,
        ),
        (
            "prefill_persistent_shared_expert_server_flags",
            prefill_persistent_shared_expert_server_flags,
        ),
        (
            "prefill_persistent_rope_split_server_flags",
            prefill_persistent_rope_split_server_flags,
        ),
        (
            "prefill_persistent_mla_attention_server_flags",
            prefill_persistent_mla_attention_server_flags,
        ),
        (
            "prefill_persistent_rmsnorm_server_flags",
            prefill_persistent_rmsnorm_server_flags,
        ),
        (
            "prefill_moe_output_accumulator_flags",
            prefill_moe_output_accumulator_flags,
        ),
        ("decode_guard_flags", decode_guard_flags),
        ("metal_runtime_mla_kv_b_cache_flags", metal_runtime_mla_kv_b_cache_flags),
        (
            "metal_runtime_mmap_final_logits_flags",
            metal_runtime_mmap_final_logits_flags,
        ),
        (
            "metal_runtime_context1_o_proj_cache_flags",
            metal_runtime_context1_o_proj_cache_flags,
        ),
        ("final_logits_flags", final_logits_flags),
    )
    payload_sections: dict[str, dict[str, object]] = {
        name: value
        for name, value in sections
        if isinstance(value, dict)
    }
    if not payload_sections:
        return None

    argv: list[str] = []
    seen: dict[str, str | None] = {}
    conflicts: list[dict[str, object]] = []
    for section_name, section in sections:
        if not isinstance(section, dict):
            continue
        for flag, value in _iter_profile_argv(section.get("argv")):
            if flag in seen:
                if seen[flag] != value:
                    conflicts.append(
                        {
                            "section": section_name,
                            "flag": flag,
                            "kept_value": seen[flag],
                            "dropped_value": value,
                        }
                    )
                continue
            seen[flag] = value
            argv.append(flag)
            if value is not None:
                argv.append(value)

    if not argv:
        return None
    profile: dict[str, object] = {
        "source": source,
        "argv": tuple(argv),
        "sections": payload_sections,
        "argv_safe_to_replay": not conflicts,
    }
    if prepared_target is not None:
        profile["prepared"] = prepared_target
    if conflicts:
        profile["argv_conflicts"] = tuple(conflicts)
    return profile


def _memory_guard_health(
    config: "PreparedServerConfig",
    system_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    max_live_bytes = (
        int(config.max_live_working_set_mib * 1024**2)
        if config.max_live_working_set_mib is not None
        else None
    )
    min_free_bytes = (
        int(config.min_free_unified_memory_gib * 1024**3)
        if config.min_free_unified_memory_gib is not None
        else 0
    )
    required_available = _sum_known_bytes(max_live_bytes, min_free_bytes)
    available = system_memory.get("available_bytes") if system_memory is not None else None
    total = system_memory.get("total_bytes") if system_memory is not None else None
    source = system_memory.get("source") if system_memory is not None else None
    available_ok = (
        available >= required_available
        if available is not None and required_available is not None
        else None
    )
    return {
        "configured_max_live_working_set_bytes": max_live_bytes,
        "configured_min_free_unified_memory_bytes": min_free_bytes,
        "configured_required_available_memory_bytes": required_available,
        "system_available_memory_bytes": available,
        "system_total_memory_bytes": total,
        "system_memory_source": source,
        "available_ok": available_ok,
    }


def _prepared_runtime_profile_health(
    prepared: PreparedManifest,
    system_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    prepared_effective = prepared.prepare_effective_unified_memory_bytes
    prepared_reserve = prepared.prepare_system_reserve_bytes
    prepared_required_available = _sum_known_bytes(
        prepared.recommended_max_live_working_set_bytes,
        prepared.recommended_min_free_unified_memory_bytes,
    )
    total = system_memory.get("total_bytes") if system_memory is not None else None
    available = (
        system_memory.get("available_bytes") if system_memory is not None else None
    )
    source = system_memory.get("source") if system_memory is not None else None
    total_ok = None
    reserve_ok = None
    required_available_ok = None
    warnings: list[str] = []
    if prepared_effective is not None:
        if total is None:
            warnings.append(
                "could not verify current system total memory against prepared "
                "effective unified-memory budget"
            )
        else:
            total_ok = int(total) >= prepared_effective
            if not total_ok:
                warnings.append(
                    "current system total memory is below the prepared "
                    "effective unified-memory budget"
                )
    if prepared_reserve is not None:
        if available is None:
            warnings.append(
                "could not verify current available memory against prepared "
                "system reserve"
            )
        else:
            reserve_ok = int(available) >= prepared_reserve
            if not reserve_ok:
                warnings.append(
                    "current available memory is below the prepared system reserve"
                )
    if prepared_required_available is not None:
        if available is None:
            warnings.append(
                "could not verify current available memory against prepared "
                "recommended required available memory"
            )
        else:
            required_available_ok = int(available) >= prepared_required_available
            if not required_available_ok:
                warnings.append(
                    "current available memory is below the prepared recommended "
                    "required available memory"
                )
    known_checks = [
        value
        for value in (total_ok, reserve_ok, required_available_ok)
        if value is not None
    ]
    profile_ok = all(known_checks) if known_checks else None
    return {
        "prepare_effective_unified_memory_bytes": prepared_effective,
        "prepare_effective_unified_memory_source": (
            prepared.prepare_effective_unified_memory_source
        ),
        "prepare_system_reserve_bytes": prepared_reserve,
        "prepared_recommended_max_live_working_set_bytes": (
            prepared.recommended_max_live_working_set_bytes
        ),
        "prepared_recommended_min_free_unified_memory_bytes": (
            prepared.recommended_min_free_unified_memory_bytes
        ),
        "prepared_recommended_required_available_memory_bytes": (
            prepared_required_available
        ),
        "system_total_memory_bytes": total,
        "system_available_memory_bytes": available,
        "system_memory_source": source,
        "system_total_meets_prepare_effective_unified_memory": total_ok,
        "system_available_meets_prepare_system_reserve": reserve_ok,
        "system_available_meets_prepared_recommended_required_available": (
            required_available_ok
        ),
        "profile_ok": profile_ok,
        "warnings": warnings,
    }


def prepared_memory_profile_missing_fields(
    prepared: PreparedManifest,
) -> tuple[str, ...]:
    missing: list[str] = []
    for field in (
        "prepare_effective_unified_memory_bytes",
        "prepare_system_reserve_bytes",
        "recommended_max_live_working_set_bytes",
        "recommended_min_free_unified_memory_bytes",
    ):
        if getattr(prepared, field) is None:
            missing.append(field)
    return tuple(missing)


def prepared_memory_profile_failure_reason(
    prepared: PreparedManifest,
) -> str | None:
    missing = prepared_memory_profile_missing_fields(prepared)
    if not missing:
        return None
    return (
        "prepared manifest is missing required memory profile fields: "
        + ", ".join(missing)
    )


def prepared_runtime_profile_health(prepared: PreparedManifest) -> dict[str, Any]:
    return _prepared_runtime_profile_health(prepared, _system_memory_health())


def prepared_runtime_profile_failure_from_health(
    profile: dict[str, Any],
) -> str | None:
    warnings = profile.get("warnings")
    warning_lines = (
        [str(warning) for warning in warnings[:5]]
        if isinstance(warnings, list) and warnings
        else []
    )
    if profile.get("profile_ok") is False:
        if warning_lines:
            return "; ".join(warning_lines)
        return "current runtime profile is below the prepared profile"
    if profile.get("profile_ok") is None and warning_lines:
        return "could not verify prepared runtime profile: " + "; ".join(
            warning_lines
        )
    return None


def prepared_runtime_profile_failure_reason(
    prepared: PreparedManifest,
    *,
    require_memory_profile: bool = False,
) -> str | None:
    if require_memory_profile:
        reason = prepared_memory_profile_failure_reason(prepared)
        if reason is not None:
            return reason
    profile = prepared_runtime_profile_health(prepared)
    reason = prepared_runtime_profile_failure_from_health(profile)
    if reason is None:
        return None
    return reason


def _prepared_runtime_profile_failure_reason(
    prepared: PreparedManifest,
) -> str | None:
    return prepared_runtime_profile_failure_reason(prepared)


def _require_prepared_runtime_profile_for_generation(
    prepared: PreparedManifest,
    *,
    require_memory_profile: bool = False,
) -> None:
    reason = prepared_runtime_profile_failure_reason(
        prepared,
        require_memory_profile=require_memory_profile,
    )
    if reason is not None:
        raise PreparedServerError(
            "prepared runtime profile check failed: "
            f"{reason}"
        )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreparedServerError(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PreparedServerError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreparedServerError(f"{label} must be a JSON object")
    return payload


def _layout_relative_file_path(
    layout_path: Path,
    filename: str,
    *,
    label: str,
) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PreparedServerError(
            f"{label} must be a relative path inside the layout directory"
        )
    return layout_path.parent / path


def _dense_prefix_layer_count(config: ModelConfig) -> int:
    if config.mlp_layer_types is not None:
        count = 0
        for layer_type in config.mlp_layer_types:
            if layer_type != "dense":
                break
            count += 1
        return count
    return int(config.first_k_dense_replace or 0)


def _public_glm_5_2_indexer_types() -> tuple[str, ...]:
    return tuple(
        "full" if (max(layer - 3 + 1, 0) % 4) == 0 else "shared"
        for layer in range(78)
    )


def _shape_check(
    name: str,
    actual: object,
    expected: object,
) -> tuple[str, dict[str, object]]:
    return (
        name,
        {
            "actual": actual,
            "expected": expected,
            "matches": actual == expected,
        },
    )


def _float_shape_check(
    name: str,
    actual: object,
    expected: float,
    *,
    tolerance: float = 1e-9,
) -> tuple[str, dict[str, object]]:
    matches = isinstance(actual, (int, float)) and math.isfinite(float(actual))
    if matches:
        matches = abs(float(actual) - expected) <= tolerance
    return (
        name,
        {
            "actual": actual,
            "expected": expected,
            "matches": matches,
        },
    )


def _raw_config_value(config: ModelConfig, key: str) -> object:
    raw = config.raw
    if not isinstance(raw, dict):
        return None
    return raw.get(key)


def _public_glm_5_2_shape_report(config: ModelConfig) -> dict[str, object]:
    try:
        moe_layers = tuple(config.moe_layers)
        routed = config.routed_experts
        top_k = config.experts_per_token
        moe_hidden = config.moe_hidden_size
    except ConfigError as exc:
        return {
            "matches": False,
            "error": str(exc),
            "mismatched_fields": ("config",),
            "checks": {},
        }

    expected_moe_layers = tuple(range(3, 78))
    expected_indexer_types = _public_glm_5_2_indexer_types()
    full_indexer_layers = tuple(
        layer
        for layer, mode in enumerate(config.indexer_types or ())
        if mode == "full"
    )
    expected_full_indexer_layers = tuple(
        layer for layer, mode in enumerate(expected_indexer_types) if mode == "full"
    )
    checks = dict(
        (
            _shape_check("model_type", config.model_type, "glm_moe_dsa"),
            _shape_check("weight_dtype", config.weight_dtype, "bfloat16"),
            _shape_check("hidden_size", config.hidden_size, 6144),
            _shape_check("num_hidden_layers", config.num_hidden_layers, 78),
            _shape_check("vocab_size", config.vocab_size, 154880),
            _shape_check("hidden_act", _raw_config_value(config, "hidden_act"), "silu"),
            _shape_check(
                "attention_bias",
                _raw_config_value(config, "attention_bias"),
                False,
            ),
            _float_shape_check(
                "attention_dropout",
                _raw_config_value(config, "attention_dropout"),
                0.0,
            ),
            _shape_check("intermediate_size", config.intermediate_size, 12288),
            _shape_check("moe_intermediate_size", moe_hidden, 2048),
            _shape_check("n_routed_experts", routed, 256),
            _shape_check("n_shared_experts", config.n_shared_experts, 1),
            _shape_check("num_experts_per_tok", top_k, 8),
            _shape_check("dense_prefix_layers", _dense_prefix_layer_count(config), 3),
            _shape_check("moe_layers", moe_layers, expected_moe_layers),
            _shape_check(
                "max_position_embeddings",
                config.max_position_embeddings,
                1048576,
            ),
            _shape_check("num_attention_heads", config.num_attention_heads, 64),
            _shape_check("num_key_value_heads", config.num_key_value_heads, 64),
            _shape_check("q_lora_rank", config.q_lora_rank, 2048),
            _shape_check("kv_lora_rank", config.kv_lora_rank, 512),
            _shape_check("qk_nope_head_dim", config.qk_nope_head_dim, 192),
            _shape_check("qk_rope_head_dim", config.qk_rope_head_dim, 64),
            _shape_check("v_head_dim", config.v_head_dim, 256),
            _shape_check("attention_q_head_dim", config.attention_q_head_dim, 256),
            _shape_check(
                "attention_q_projection_output_dim",
                config.attention_q_projection_output_dim,
                16384,
            ),
            _shape_check(
                "attention_kv_a_output_dim",
                config.attention_kv_a_output_dim,
                576,
            ),
            _shape_check(
                "attention_kv_b_output_dim",
                config.attention_kv_b_output_dim,
                28672,
            ),
            _shape_check(
                "attention_value_output_dim",
                config.attention_value_output_dim,
                16384,
            ),
            _float_shape_check("rms_norm_eps", config.rms_norm_eps, 1e-5),
            _float_shape_check("rope_theta", config.rope_theta, 8_000_000.0),
            _shape_check("index_head_dim", config.index_head_dim, 128),
            _shape_check("index_n_heads", config.index_n_heads, 32),
            _shape_check("index_topk", config.index_topk, 2048),
            _shape_check(
                "index_topk_freq",
                _raw_config_value(config, "index_topk_freq"),
                4,
            ),
            _shape_check(
                "index_skip_topk_offset",
                _raw_config_value(config, "index_skip_topk_offset"),
                3,
            ),
            _shape_check(
                "num_nextn_predict_layers",
                _raw_config_value(config, "num_nextn_predict_layers"),
                1,
            ),
            _shape_check(
                "indexer_rope_interleave",
                config.indexer_rope_interleave,
                True,
            ),
            _shape_check("indexer_types", config.indexer_types, expected_indexer_types),
            _shape_check(
                "full_indexer_layers",
                full_indexer_layers,
                expected_full_indexer_layers,
            ),
            _shape_check("scoring_func", config.scoring_func, "sigmoid"),
            _shape_check("topk_method", config.topk_method, "noaux_tc"),
            _shape_check("norm_topk_prob", config.norm_topk_prob, True),
            _float_shape_check(
                "routed_scaling_factor",
                config.routed_scaling_factor,
                2.5,
            ),
            _shape_check("n_group", config.n_group, 1),
            _shape_check("topk_group", config.topk_group, 1),
            _shape_check("rope_interleave", config.rope_interleave, True),
            _shape_check("tie_word_embeddings", config.tie_word_embeddings, False),
        )
    )
    mismatched = tuple(
        name for name, payload in checks.items() if payload.get("matches") is not True
    )
    return {
        "matches": not mismatched,
        "mismatched_fields": mismatched,
        "dsa_full_indexer_layer_count": len(full_indexer_layers),
        "dsa_full_indexer_layers": full_indexer_layers,
        "expected_dsa_full_indexer_layer_count": len(expected_full_indexer_layers),
        "expected_dsa_full_indexer_layers": expected_full_indexer_layers,
        "dsa_schedule": {
            "index_topk_freq": _raw_config_value(config, "index_topk_freq"),
            "index_skip_topk_offset": _raw_config_value(
                config,
                "index_skip_topk_offset",
            ),
            "full_indexer_layer_count": len(full_indexer_layers),
            "first_full_indexer_layers": full_indexer_layers[:8],
            "last_full_indexer_layers": full_indexer_layers[-4:],
        },
        "checks": checks,
    }


def _is_public_glm_5_2_shape(config: ModelConfig) -> bool:
    return bool(_public_glm_5_2_shape_report(config).get("matches"))


def _public_glm_5_2_shape_failure_detail(
    readiness: dict[str, Any],
) -> str | None:
    report = readiness.get("public_glm_5_2_shape")
    if not isinstance(report, dict):
        return None
    fields = report.get("mismatched_fields")
    if not isinstance(fields, (list, tuple)) or not fields:
        return None
    preview = ", ".join(str(field) for field in tuple(fields)[:6])
    if len(fields) > 6:
        preview += f", +{len(fields) - 6} more"
    return preview


_GLM_GLOBAL_EMBED_SUFFIXES = (
    "model.embed_tokens.weight",
    ".embed_tokens.weight",
    "transformer.word_embeddings.weight",
    ".word_embeddings.weight",
)
_GLM_GLOBAL_NORM_SUFFIXES = (
    "model.norm.weight",
    ".model.norm.weight",
    "transformer.norm.weight",
    ".transformer.norm.weight",
    "norm.weight",
    ".norm.weight",
)
_GLM_GLOBAL_LM_HEAD_SUFFIXES = ("lm_head.weight", ".lm_head.weight")
_GLM_ATTENTION_SUFFIXES = (
    ".input_layernorm.weight",
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_a_layernorm.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.kv_a_layernorm.weight",
    ".self_attn.kv_b_proj.weight",
    ".self_attn.o_proj.weight",
    ".post_attention_layernorm.weight",
)
_GLM_ATTENTION_KV_B_ALTERNATIVE_SUFFIXES = (
    ".self_attn.embed_q.weight",
    ".self_attn.unembed_out.weight",
)
_GLM_MLP_COMPONENTS = ("gate_proj", "up_proj", "down_proj")
_GLM_INDEXER_SUFFIXES = (
    ".self_attn.indexer.wk.weight",
    ".self_attn.indexer.wq_b.weight",
    ".self_attn.indexer.weights_proj.weight",
    ".self_attn.indexer.k_norm.weight",
    ".self_attn.indexer.k_norm.bias",
)


def _resident_tensor_name(tensor: dict[str, Any]) -> str | None:
    name = tensor.get("name")
    return name if isinstance(name, str) and name else None


def _resident_tensor_shape(tensor: dict[str, Any]) -> tuple[int, ...] | None:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    out: list[int] = []
    for dim in shape:
        if type(dim) is not int or dim <= 0:
            return None
        out.append(int(dim))
    return tuple(out)


def _resident_tensor_size(tensor: dict[str, Any]) -> int | None:
    size = tensor.get("size")
    if type(size) is not int or size < 0:
        return None
    return int(size)


def _resident_tensors_by_name(
    resident_layout: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_tensors = resident_layout.get("tensors")
    if not isinstance(raw_tensors, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in raw_tensors:
        if not isinstance(item, dict):
            continue
        name = _resident_tensor_name(item)
        if name is not None:
            out[name] = item
    return out


def _resident_mxfp4_logical_shape(
    resident_layout: dict[str, Any],
    tensor: dict[str, Any],
) -> tuple[int, ...] | None:
    dtype = tensor.get("dtype")
    if not isinstance(dtype, str) or dtype.upper() not in AFFINE_INT4_WEIGHT_DTYPES:
        return None
    weight_name = _resident_tensor_name(tensor)
    if weight_name is None or not weight_name.endswith(".weight"):
        return None
    by_name = _resident_tensors_by_name(resident_layout)
    base = weight_name[: -len(".weight")]
    scales = by_name.get(f"{base}.scales")
    biases = by_name.get(f"{base}.biases")
    if scales is None or biases is not None:
        return None
    scale_dtype = scales.get("dtype")
    if (
        not isinstance(scale_dtype, str)
        or scale_dtype.upper() not in MXFP4_SCALE_DTYPES
    ):
        return None
    weight_shape = _resident_tensor_shape(tensor)
    scale_shape = _resident_tensor_shape(scales)
    if weight_shape is None:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 weight {weight_name} must have a positive integer shape"
        )
    if len(weight_shape) < 2:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 weight {weight_name} must be at least 2-D"
        )
    if scale_shape is None:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 scales for {weight_name} must have a positive integer shape"
        )
    if len(scale_shape) != len(weight_shape):
        raise ResidentAffineLayoutError(
            f"resident MXFP4 scales for {weight_name} must match weight rank"
        )
    if scale_shape[:-1] != weight_shape[:-1]:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 scales for {weight_name} must match weight prefix shape"
        )
    packed_cols = weight_shape[-1]
    groups = scale_shape[-1]
    in_dim = packed_cols * 8
    if in_dim % groups != 0:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 groups for {weight_name} do not divide logical input dim "
            f"{in_dim}"
        )
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 group size for {weight_name} must be a positive multiple "
            "of 8"
        )
    expected_weight_bytes = 4
    for dim in weight_shape:
        expected_weight_bytes *= dim
    expected_scale_bytes = 1
    for dim in scale_shape:
        expected_scale_bytes *= dim
    if _resident_tensor_size(tensor) != expected_weight_bytes:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 weight {weight_name} size does not match packed shape"
        )
    if _resident_tensor_size(scales) != expected_scale_bytes:
        raise ResidentAffineLayoutError(
            f"resident MXFP4 scales for {weight_name} size does not match metadata shape"
        )
    return weight_shape[:-1] + (in_dim,)


def _resident_dtype_bytes(dtype: str) -> int:
    if dtype in {"F32", "float32", "FLOAT32"}:
        return 4
    if dtype in {"BF16", "bfloat16", "BFLOAT16", "F16", "float16", "FLOAT16"}:
        return 2
    return 0


def _find_global_resident_tensor(
    tensors: tuple[dict[str, Any], ...],
    suffixes: tuple[str, ...],
) -> dict[str, Any] | None:
    for tensor in tensors:
        name = _resident_tensor_name(tensor)
        if name is not None and any(name.endswith(suffix) for suffix in suffixes):
            return tensor
    return None


def _find_layer_resident_tensor(
    tensors: tuple[dict[str, Any], ...],
    layer: int,
    suffix: str,
) -> dict[str, Any] | None:
    layer_marker = f".layers.{layer}."
    for tensor in tensors:
        name = _resident_tensor_name(tensor)
        if name is not None and layer_marker in name and name.endswith(suffix):
            return tensor
    return None


def _find_router_resident_tensor(
    tensors: tuple[dict[str, Any], ...],
    layer: int,
) -> dict[str, Any] | None:
    layer_marker = f".layers.{layer}."
    for tensor in tensors:
        name = _resident_tensor_name(tensor)
        if name is None or layer_marker not in name:
            continue
        if name.endswith(".gate.weight") or ".mlp.gate.weight" in name:
            return tensor
    return None


def _find_dense_mlp_resident_tensor(
    tensors: tuple[dict[str, Any], ...],
    layer: int,
    component: str,
) -> dict[str, Any] | None:
    for suffix in (
        f".mlp.{component}.weight",
        f".mlp.switch_mlp.{component}.weight",
        f".switch_mlp.{component}.weight",
    ):
        tensor = _find_layer_resident_tensor(tensors, layer, suffix)
        if tensor is not None:
            return tensor
    return None


def _find_shared_resident_tensor(
    tensors: tuple[dict[str, Any], ...],
    layer: int,
    component: str,
) -> dict[str, Any] | None:
    layer_marker = f".layers.{layer}."
    suffix = f".{component}.weight"
    for tensor in tensors:
        name = _resident_tensor_name(tensor)
        if name is None or layer_marker not in name or not name.endswith(suffix):
            continue
        if ".shared_experts." in name or ".shared_expert." in name:
            return tensor
    return None


def _check_resident_tensor_shape_size(
    tensor: dict[str, Any],
    *,
    label: str,
    add_issue: Callable[[str], None],
    expected_shape: tuple[int, ...] | None = None,
    resident_layout: dict[str, Any] | None = None,
) -> tuple[int, ...] | None:
    if resident_layout is not None:
        try:
            mxfp4_shape = _resident_mxfp4_logical_shape(resident_layout, tensor)
            if mxfp4_shape is not None:
                shape = mxfp4_shape
                if expected_shape is not None and shape != expected_shape:
                    add_issue(
                        f"resident tensor {label} shape {list(shape)} does not match "
                        f"expected {list(expected_shape)}"
                    )
                return shape
            affine = resident_affine_int4_layout_info(resident_layout, tensor)
        except ResidentAffineLayoutError as exc:
            add_issue(str(exc))
            return _resident_tensor_shape(tensor)
        if affine is not None:
            shape = (affine.out_dim, affine.in_dim)
            if expected_shape is not None and shape != expected_shape:
                add_issue(
                    f"resident tensor {label} shape {list(shape)} does not match "
                    f"expected {list(expected_shape)}"
                )
            return shape
    shape = _resident_tensor_shape(tensor)
    if shape is None:
        add_issue(f"resident tensor {label} must have a positive integer shape")
        return None
    if expected_shape is not None and shape != expected_shape:
        add_issue(
            f"resident tensor {label} shape {list(shape)} does not match "
            f"expected {list(expected_shape)}"
        )
    dtype = str(tensor.get("dtype") or "")
    dtype_bytes = _resident_dtype_bytes(dtype)
    if dtype_bytes <= 0:
        add_issue(f"resident tensor {label} dtype {dtype!r} is not F32/BF16/F16")
        return shape
    size = tensor.get("size")
    if type(size) is not int or size <= 0:
        add_issue(f"resident tensor {label} size must be a positive integer")
        return shape
    expected_size = dtype_bytes
    for dim in shape:
        expected_size *= dim
    if int(size) != expected_size:
        add_issue(
            f"resident tensor {label} size {size} does not match expected "
            f"{expected_size}"
        )
    return shape


def _expected_affine_int4_components(
    config: ModelConfig,
    group_size: int,
) -> dict[str, tuple[tuple[str, ...], int, tuple[int, int]]]:
    if group_size <= 0:
        raise PreparedServerError("expert layout group_size must be positive")
    if group_size % 8 != 0:
        raise PreparedServerError("expert layout group_size must be divisible by 8")
    raw_dims = {
        "gate_proj": (config.moe_hidden_size, config.hidden_size),
        "up_proj": (config.moe_hidden_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.moe_hidden_size),
    }
    expected: dict[str, tuple[tuple[str, ...], int, tuple[int, int]]] = {}
    for base, (out_dim, in_dim) in raw_dims.items():
        if in_dim % 8 != 0:
            raise PreparedServerError(
                f"{base}.weight input dim {in_dim} is not divisible by 8"
            )
        if in_dim % group_size != 0:
            raise PreparedServerError(
                f"{base}.weight input dim {in_dim} is not divisible by "
                f"group size {group_size}"
            )
        packed_cols = in_dim // 8
        groups_per_row = in_dim // group_size
        expected[f"{base}.weight"] = (
            AFFINE_INT4_WEIGHT_DTYPES,
            out_dim * packed_cols * 4,
            (out_dim, packed_cols),
        )
        expected[f"{base}.scales"] = (
            AFFINE_INT4_META_DTYPES,
            out_dim * groups_per_row * 2,
            (out_dim, groups_per_row),
        )
        expected[f"{base}.biases"] = (
            AFFINE_INT4_META_DTYPES,
            out_dim * groups_per_row * 2,
            (out_dim, groups_per_row),
        )
    return expected


def _expected_mxfp4_components(
    config: ModelConfig,
    group_size: int,
) -> dict[str, tuple[tuple[str, ...], int, tuple[int, int]]]:
    if group_size != MXFP4_GROUP_SIZE:
        raise PreparedServerError(
            f"mlx-mxfp4 expert layout group_size must be {MXFP4_GROUP_SIZE}"
        )
    raw_dims = {
        "gate_proj": (config.moe_hidden_size, config.hidden_size),
        "up_proj": (config.moe_hidden_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.moe_hidden_size),
    }
    expected: dict[str, tuple[tuple[str, ...], int, tuple[int, int]]] = {}
    for base, (out_dim, in_dim) in raw_dims.items():
        if in_dim % 8 != 0:
            raise PreparedServerError(
                f"{base}.weight input dim {in_dim} is not divisible by 8"
            )
        if in_dim % group_size != 0:
            raise PreparedServerError(
                f"{base}.weight input dim {in_dim} is not divisible by "
                f"MXFP4 group size {group_size}"
            )
        packed_cols = in_dim // 8
        groups_per_row = in_dim // group_size
        expected[f"{base}.weight"] = (
            AFFINE_INT4_WEIGHT_DTYPES,
            out_dim * packed_cols * 4,
            (out_dim, packed_cols),
        )
        expected[f"{base}.scales"] = (
            MXFP4_SCALE_DTYPES,
            out_dim * groups_per_row,
            (out_dim, groups_per_row),
        )
    return expected


def _shape2(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if type(value[0]) is not int or type(value[1]) is not int:
        return None
    return int(value[0]), int(value[1])


def _expected_resident_router_metadata(config: ModelConfig) -> dict[str, object]:
    expected: dict[str, object] = {}
    if config.scoring_func is not None:
        expected["scoring_func"] = config.scoring_func
    if config.norm_topk_prob is not None:
        expected["norm_topk_prob"] = config.norm_topk_prob
    if config.routed_scaling_factor is not None:
        expected["routed_scaling_factor"] = config.routed_scaling_factor
    if config.n_group is not None:
        expected["n_group"] = int(config.n_group)
    if config.topk_group is not None:
        expected["topk_group"] = int(config.topk_group)
    if config.topk_method is not None:
        expected["topk_method"] = config.topk_method
    if config.num_experts_per_tok is not None:
        expected["num_experts_per_tok"] = int(config.num_experts_per_tok)
    return expected


def _router_metadata_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    return actual == expected


def _resident_router_metadata_readiness(
    resident: dict[str, Any],
    config: ModelConfig,
    *,
    add_issue: Callable[[str], None],
) -> dict[str, Any]:
    expected = _expected_resident_router_metadata(config)
    router = resident.get("router")
    present = isinstance(router, dict)
    ok = True
    if expected and router is None:
        add_issue("resident layout missing router metadata")
        ok = False
        router = None
    elif expected and not isinstance(router, dict):
        add_issue("resident layout router metadata must be an object")
        ok = False
        router = None
    if isinstance(router, dict):
        for field, expected_value in expected.items():
            if field not in router:
                add_issue(f"resident layout router metadata missing {field}")
                ok = False
                continue
            actual_value = router[field]
            if not _router_metadata_value_matches(actual_value, expected_value):
                add_issue(
                    "resident layout router metadata "
                    f"{field} {actual_value!r} does not match config "
                    f"{expected_value!r}"
                )
                ok = False
    return {
        "resident_router_metadata_present": present,
        "resident_router_metadata_ok": ok,
        "resident_router_metadata_checked_fields": tuple(expected),
    }


def _layout_config_sha256_readiness(
    layout: dict[str, Any],
    *,
    label: str,
    prepared: PreparedManifest,
    add_issue: Callable[[str], None],
) -> str | None:
    value = layout.get("config_sha256")
    if value is None:
        add_issue(f"{label} missing config_sha256")
        return None
    if not isinstance(value, str) or not value:
        add_issue(f"{label} config_sha256 must be a non-empty string")
        return None
    if prepared.model_config_sha256 is None:
        add_issue("prepared model_config_sha256 is unavailable")
    elif value != prepared.model_config_sha256:
        add_issue(f"{label} config_sha256 does not match prepared model_config_sha256")
    return value


def _glm_resident_layout_readiness(
    prepared: PreparedManifest,
    config: ModelConfig,
    *,
    moe_layers: tuple[int, ...],
    add_issue: Callable[[str], None],
) -> dict[str, Any]:
    try:
        resident = _read_json_object(prepared.resident_layout, "resident layout")
    except PreparedServerError as exc:
        add_issue(str(exc))
        return {
            "resident_layout_tensor_count": 0,
            "resident_embedding": False,
            "resident_final_norm": False,
            "resident_lm_head": False,
            "resident_tied_lm_head": False,
            "resident_attention_layers_checked": config.num_hidden_layers,
            "resident_attention_layers_ok": 0,
            "resident_router_layers_checked": len(moe_layers),
            "resident_router_layers_ok": 0,
            "resident_dense_layers_checked": 0,
            "resident_dense_layers_ok": 0,
            "resident_shared_layers_checked": 0,
            "resident_shared_layers_ok": 0,
            "resident_indexer_layers_checked": 0,
            "resident_indexer_layers_ok": 0,
            "resident_routed_expert_tensor_count": 0,
            "resident_layout_model_type": None,
            "resident_layout_config_sha256": None,
            "resident_layout_total_bytes": None,
            "resident_weight_file_bytes": None,
            "resident_weight_file_exact_size": False,
            "resident_router_metadata_present": False,
            "resident_router_metadata_ok": False,
            "resident_router_metadata_checked_fields": (),
        }
    raw_total_bytes = resident.get("total_bytes")
    resident_layout_total_bytes: int | None
    if type(raw_total_bytes) is int and raw_total_bytes >= 0:
        resident_layout_total_bytes = int(raw_total_bytes)
    else:
        resident_layout_total_bytes = None
        add_issue("resident layout total_bytes must be a non-negative integer")
    resident_weight_file_bytes: int | None = None
    resident_weight_file_exact_size = False
    weight_file = resident.get("weight_file")
    if not isinstance(weight_file, str) or not weight_file:
        add_issue("resident layout missing weight_file")
    else:
        try:
            weight_path = _layout_relative_file_path(
                prepared.resident_layout,
                weight_file,
                label="resident weight_file",
            )
            resident_weight_file_bytes = int(weight_path.stat().st_size)
        except (OSError, PreparedServerError) as exc:
            add_issue(f"failed to stat resident weight file {weight_file}: {exc}")
        else:
            if resident_layout_total_bytes is not None:
                resident_weight_file_exact_size = (
                    resident_weight_file_bytes == resident_layout_total_bytes
                )
                if not resident_weight_file_exact_size:
                    add_issue(
                        f"resident weight file bytes {resident_weight_file_bytes} "
                        "do not match layout total_bytes "
                        f"{resident_layout_total_bytes}"
                    )
    resident_config_sha256 = _layout_config_sha256_readiness(
        resident,
        label="resident layout",
        prepared=prepared,
        add_issue=add_issue,
    )
    resident_model_type = resident.get("model_type")
    if resident_model_type != config.model_type:
        add_issue(
            "resident layout model_type "
            f"{resident_model_type!r} does not match config {config.model_type!r}"
        )
    router_metadata_summary = _resident_router_metadata_readiness(
        resident,
        config,
        add_issue=add_issue,
    )
    raw_tensors = resident.get("tensors")
    if not isinstance(raw_tensors, list):
        add_issue("resident layout missing tensors array")
        tensors: tuple[dict[str, Any], ...] = ()
    else:
        tensor_items: list[dict[str, Any]] = []
        for tensor in raw_tensors:
            if isinstance(tensor, dict):
                tensor_items.append(tensor)
            else:
                add_issue("resident layout tensors must be objects")
        tensors = tuple(tensor_items)

    moe_set = set(moe_layers)
    resident_routed_expert_names = tuple(
        name
        for tensor in tensors
        for name in (_resident_tensor_name(tensor),)
        if name is not None
        and is_routed_expert_tensor_for_moe_layers(
            name,
            moe_set,
            num_hidden_layers=config.num_hidden_layers,
        )
    )
    if resident_routed_expert_names:
        preview = ", ".join(resident_routed_expert_names[:3])
        more = (
            ""
            if len(resident_routed_expert_names) <= 3
            else f", +{len(resident_routed_expert_names) - 3} more"
        )
        add_issue(
            "resident layout contains routed expert tensors that must remain "
            f"SSD-backed: {preview}{more}"
        )

    hidden = int(config.hidden_size)
    vocab = config.vocab_size
    embedding = _find_global_resident_tensor(tensors, _GLM_GLOBAL_EMBED_SUFFIXES)
    embedding_ok = embedding is not None
    if embedding is None:
        add_issue("resident layout missing global embed_tokens.weight")
    else:
        shape = _check_resident_tensor_shape_size(
            embedding,
            label="embed_tokens.weight",
            add_issue=add_issue,
            resident_layout=resident,
        )
        if shape is not None:
            if len(shape) != 2:
                add_issue("resident tensor embed_tokens.weight must be 2-D")
            elif vocab is not None and shape[0] != vocab:
                add_issue(
                    f"resident tensor embed_tokens.weight vocab rows {shape[0]} "
                    f"does not match config vocab_size {vocab}"
                )
            elif shape[1] != hidden:
                add_issue(
                    f"resident tensor embed_tokens.weight hidden dim {shape[1]} "
                    f"does not match config hidden_size {hidden}"
                )

    final_norm = _find_global_resident_tensor(tensors, _GLM_GLOBAL_NORM_SUFFIXES)
    final_norm_ok = final_norm is not None
    if final_norm is None:
        add_issue("resident layout missing global norm.weight")
    else:
        _check_resident_tensor_shape_size(
            final_norm,
            label="norm.weight",
            add_issue=add_issue,
            expected_shape=(hidden,),
            resident_layout=resident,
        )

    lm_head = _find_global_resident_tensor(tensors, _GLM_GLOBAL_LM_HEAD_SUFFIXES)
    if lm_head is not None:
        shape = _check_resident_tensor_shape_size(
            lm_head,
            label="lm_head.weight",
            add_issue=add_issue,
            resident_layout=resident,
        )
        if shape is not None:
            if len(shape) != 2:
                add_issue("resident tensor lm_head.weight must be 2-D")
            elif vocab is not None and shape[0] != vocab:
                add_issue(
                    f"resident tensor lm_head.weight vocab rows {shape[0]} "
                    f"does not match config vocab_size {vocab}"
                )
            elif shape[1] != hidden:
                add_issue(
                    f"resident tensor lm_head.weight hidden dim {shape[1]} "
                    f"does not match config hidden_size {hidden}"
                )
    elif embedding is None:
        add_issue(
            "resident layout missing lm_head.weight and tied embeddings are unavailable"
        )
    elif config.tie_word_embeddings is False:
        add_issue(
            "resident layout missing lm_head.weight and config tie_word_embeddings=false"
        )

    expected_attention_shapes: dict[str, tuple[int, ...]] = {
        ".input_layernorm.weight": (hidden,),
        ".post_attention_layernorm.weight": (hidden,),
    }
    if (
        config.kv_lora_rank is not None
        and config.qk_rope_head_dim is not None
        and config.num_attention_heads is not None
        and config.qk_nope_head_dim is not None
        and config.v_head_dim is not None
    ):
        kv_lora = int(config.kv_lora_rank)
        rope = int(config.qk_rope_head_dim)
        heads = int(config.num_attention_heads)
        nope = int(config.qk_nope_head_dim)
        value = int(config.v_head_dim)
        expected_attention_shapes.update(
            {
                ".self_attn.kv_a_proj_with_mqa.weight": (kv_lora + rope, hidden),
                ".self_attn.kv_a_layernorm.weight": (kv_lora,),
                ".self_attn.kv_b_proj.weight": (heads * (nope + value), kv_lora),
                ".self_attn.embed_q.weight": (heads, kv_lora, nope),
                ".self_attn.unembed_out.weight": (heads, value, kv_lora),
                ".self_attn.o_proj.weight": (hidden, heads * value),
            }
        )
        if config.q_lora_rank is not None:
            q_lora = int(config.q_lora_rank)
            expected_attention_shapes.update(
                {
                    ".self_attn.q_a_proj.weight": (q_lora, hidden),
                    ".self_attn.q_a_layernorm.weight": (q_lora,),
                    ".self_attn.q_b_proj.weight": (heads * (nope + rope), q_lora),
                }
            )

    attention_ok = 0
    for layer in range(config.num_hidden_layers):
        missing = []
        for suffix in _GLM_ATTENTION_SUFFIXES:
            tensor = _find_layer_resident_tensor(tensors, layer, suffix)
            if tensor is None:
                if (
                    suffix == ".self_attn.kv_b_proj.weight"
                    and all(
                        _find_layer_resident_tensor(tensors, layer, alternative)
                        is not None
                        for alternative in _GLM_ATTENTION_KV_B_ALTERNATIVE_SUFFIXES
                    )
                ):
                    continue
                missing.append(suffix)
                continue
            _check_resident_tensor_shape_size(
                tensor,
                label=f"layer {layer} {suffix}",
                add_issue=add_issue,
                expected_shape=expected_attention_shapes.get(suffix),
                resident_layout=resident,
            )
        if ".self_attn.kv_b_proj.weight" not in missing:
            for suffix in _GLM_ATTENTION_KV_B_ALTERNATIVE_SUFFIXES:
                tensor = _find_layer_resident_tensor(tensors, layer, suffix)
                if tensor is None:
                    continue
                _check_resident_tensor_shape_size(
                    tensor,
                    label=f"layer {layer} {suffix}",
                    add_issue=add_issue,
                    expected_shape=expected_attention_shapes.get(suffix),
                    resident_layout=resident,
                )
        if missing:
            for suffix in missing:
                add_issue(f"resident layout layer {layer} missing {suffix}")
        else:
            attention_ok += 1

    router_ok = 0
    routed_experts = config.n_routed_experts
    for layer in moe_layers:
        router = _find_router_resident_tensor(tensors, layer)
        if router is None:
            add_issue(f"resident layout layer {layer} missing router gate.weight")
            continue
        expected = (
            (int(routed_experts), hidden)
            if routed_experts is not None
            else None
        )
        _check_resident_tensor_shape_size(
            router,
            label=f"layer {layer} router gate.weight",
            add_issue=add_issue,
            expected_shape=expected,
            resident_layout=resident,
        )
        router_ok += 1

    dense_layers = tuple(
        layer for layer in range(config.num_hidden_layers) if layer not in moe_set
    )
    dense_ok = 0
    expected_dense_shapes: dict[str, tuple[int, ...]] = {}
    if config.intermediate_size is not None:
        intermediate = int(config.intermediate_size)
        expected_dense_shapes = {
            "gate_proj": (intermediate, hidden),
            "up_proj": (intermediate, hidden),
            "down_proj": (hidden, intermediate),
        }
    for layer in dense_layers:
        missing = []
        for component in _GLM_MLP_COMPONENTS:
            tensor = _find_dense_mlp_resident_tensor(tensors, layer, component)
            if tensor is None:
                missing.append(component)
                continue
            _check_resident_tensor_shape_size(
                tensor,
                label=f"layer {layer} dense MLP {component}.weight",
                add_issue=add_issue,
                expected_shape=expected_dense_shapes.get(component),
                resident_layout=resident,
            )
        if missing:
            add_issue(
                "resident layout layer "
                f"{layer} missing dense MLP {','.join(missing)}"
            )
        else:
            dense_ok += 1

    shared_layers_checked = 0
    shared_ok = 0
    if (config.n_shared_experts or 0) > 0:
        shared_layers_checked = len(moe_layers)
        try:
            shared_hidden = config.moe_hidden_size
        except ConfigError:
            shared_hidden = None
        expected_shared_shapes: dict[str, tuple[int, ...]] = {}
        if shared_hidden is not None:
            shared_width = int(config.n_shared_experts or 0) * int(shared_hidden)
            expected_shared_shapes = {
                "gate_proj": (shared_width, hidden),
                "up_proj": (shared_width, hidden),
                "down_proj": (hidden, shared_width),
            }
        for layer in moe_layers:
            missing = []
            for component in _GLM_MLP_COMPONENTS:
                tensor = _find_shared_resident_tensor(tensors, layer, component)
                if tensor is None:
                    missing.append(component)
                    continue
                _check_resident_tensor_shape_size(
                    tensor,
                    label=f"layer {layer} shared expert {component}.weight",
                    add_issue=add_issue,
                    expected_shape=expected_shared_shapes.get(component),
                    resident_layout=resident,
                )
            if missing:
                add_issue(
                    "resident layout layer "
                    f"{layer} missing shared expert {','.join(missing)}"
                )
            else:
                shared_ok += 1

    indexer_layers = tuple(
        layer
        for layer, indexer_type in enumerate(config.indexer_types or ())
        if layer < config.num_hidden_layers and str(indexer_type).lower() == "full"
    )
    indexer_ok = 0
    expected_indexer_shapes: dict[str, tuple[int, ...]] = {}
    if indexer_layers:
        missing_config = []
        if config.index_head_dim is None:
            missing_config.append("index_head_dim")
        if config.index_n_heads is None:
            missing_config.append("index_n_heads")
        if config.q_lora_rank is None:
            missing_config.append("q_lora_rank")
        if missing_config:
            add_issue(
                "full DSA indexer readiness requires config fields: "
                + ",".join(missing_config)
            )
        else:
            index_head_dim = int(config.index_head_dim)
            index_n_heads = int(config.index_n_heads)
            q_lora = int(config.q_lora_rank)
            expected_indexer_shapes = {
                ".self_attn.indexer.wk.weight": (index_head_dim, hidden),
                ".self_attn.indexer.wq_b.weight": (
                    index_n_heads * index_head_dim,
                    q_lora,
                ),
                ".self_attn.indexer.weights_proj.weight": (index_n_heads, hidden),
                ".self_attn.indexer.k_norm.weight": (index_head_dim,),
                ".self_attn.indexer.k_norm.bias": (index_head_dim,),
            }
    for layer in indexer_layers:
        missing = []
        for suffix in _GLM_INDEXER_SUFFIXES:
            tensor = _find_layer_resident_tensor(tensors, layer, suffix)
            if tensor is None:
                missing.append(suffix)
                continue
            _check_resident_tensor_shape_size(
                tensor,
                label=f"layer {layer} indexer {suffix}",
                add_issue=add_issue,
                expected_shape=expected_indexer_shapes.get(suffix),
                resident_layout=resident,
            )
        if missing:
            for suffix in missing:
                add_issue(f"resident layout layer {layer} missing indexer {suffix}")
        else:
            indexer_ok += 1

    return {
        "resident_layout_tensor_count": len(tensors),
        "resident_embedding": embedding_ok,
        "resident_final_norm": final_norm_ok,
        "resident_lm_head": lm_head is not None,
        "resident_tied_lm_head": (
            embedding is not None
            and lm_head is None
            and config.tie_word_embeddings is not False
        ),
        "resident_attention_layers_checked": config.num_hidden_layers,
        "resident_attention_layers_ok": attention_ok,
        "resident_router_layers_checked": len(moe_layers),
        "resident_router_layers_ok": router_ok,
        "resident_dense_layers_checked": len(dense_layers),
        "resident_dense_layers_ok": dense_ok,
        "resident_shared_layers_checked": shared_layers_checked,
        "resident_shared_layers_ok": shared_ok,
        "resident_indexer_layers_checked": len(indexer_layers),
        "resident_indexer_layers_ok": indexer_ok,
        "resident_routed_expert_tensor_count": len(resident_routed_expert_names),
        "resident_layout_model_type": resident_model_type,
        "resident_layout_config_sha256": resident_config_sha256,
        "resident_layout_total_bytes": resident_layout_total_bytes,
        "resident_weight_file_bytes": resident_weight_file_bytes,
        "resident_weight_file_exact_size": resident_weight_file_exact_size,
    } | router_metadata_summary


def _glm_decode_cache_readiness(
    prepared: PreparedManifest,
    config: ModelConfig,
    *,
    add_issue: Callable[[str], None],
) -> dict[str, Any]:
    try:
        layout = load_decode_cache_layout(prepared.decode_cache_layout)
    except DecodeCacheError as exc:
        add_issue(str(exc))
        return {
            "decode_cache_layout_ok": False,
            "decode_cache_context_tokens": None,
            "decode_cache_model_type": None,
            "decode_cache_dtype": None,
            "decode_cache_layout_total_bytes": None,
            "decode_cache_segment_count": 0,
            "decode_cache_mla_kv_segments_checked": config.num_hidden_layers,
            "decode_cache_mla_kv_segments_ok": 0,
            "decode_cache_dsa_index_segments_checked": 0,
            "decode_cache_dsa_index_segments_ok": 0,
        }

    ok = True
    if layout.model_type != config.model_type:
        add_issue(
            "decode cache layout model_type "
            f"{layout.model_type!r} does not match config {config.model_type!r}"
        )
        ok = False
    if (
        prepared.max_context_tokens is not None
        and layout.max_context_tokens != int(prepared.max_context_tokens)
    ):
        add_issue("decode cache layout max_context_tokens does not match manifest")
        ok = False
    if (
        config.max_position_embeddings is not None
        and layout.max_context_tokens > int(config.max_position_embeddings)
    ):
        add_issue(
            "decode cache layout max_context_tokens exceeds "
            "config max_position_embeddings"
        )
        ok = False
    if (
        prepared.decode_cache_file_bytes is not None
        and prepared.decode_cache_file_bytes != layout.total_bytes
    ):
        add_issue("decode cache file bytes do not match layout total_bytes")
        ok = False

    try:
        mla_width = config.mla_cache_width
    except ConfigError as exc:
        add_issue(str(exc))
        mla_width = None
        ok = False
    if mla_width is None:
        add_issue("config is missing MLA cache width")
        ok = False

    expected_dsa_layers: tuple[int, ...] = ()
    if config.indexer_types is not None:
        expected_dsa_layers = tuple(
            layer
            for layer, indexer_type in enumerate(config.indexer_types)
            if layer < config.num_hidden_layers and str(indexer_type).lower() == "full"
        )
    dsa_width = config.index_head_dim if expected_dsa_layers else None
    if expected_dsa_layers and dsa_width is None:
        add_issue("full DSA indexer decode cache requires config index_head_dim")
        ok = False

    by_key: dict[tuple[str, int], list[Any]] = {}
    for segment in layout.segments:
        by_key.setdefault((segment.kind, segment.layer), []).append(segment)
        if segment.layer < 0 or segment.layer >= config.num_hidden_layers:
            add_issue(
                f"decode cache segment {segment.kind} layer {segment.layer} "
                "is outside config layers"
            )
            ok = False
        if segment.max_context_tokens != layout.max_context_tokens:
            add_issue(
                f"decode cache segment {segment.kind} layer {segment.layer} "
                "max_context_tokens does not match layout"
            )
            ok = False
        if segment.kind not in {"mla_kv", "dsa_index"}:
            add_issue(f"decode cache segment kind {segment.kind!r} is unsupported")
            ok = False

    mla_ok = 0
    for layer in range(config.num_hidden_layers):
        segments = by_key.get(("mla_kv", layer), [])
        if len(segments) != 1:
            add_issue(f"decode cache layout layer {layer} missing unique mla_kv segment")
            ok = False
            continue
        segment = segments[0]
        if mla_width is not None and segment.width != int(mla_width):
            add_issue(
                f"decode cache layout layer {layer} mla_kv width "
                "does not match config"
            )
            ok = False
            continue
        mla_ok += 1

    dsa_ok = 0
    expected_dsa_set = set(expected_dsa_layers)
    for layer in expected_dsa_layers:
        segments = by_key.get(("dsa_index", layer), [])
        if len(segments) != 1:
            add_issue(f"decode cache layout layer {layer} missing unique dsa_index segment")
            ok = False
            continue
        segment = segments[0]
        if dsa_width is not None and segment.width != int(dsa_width):
            add_issue(
                f"decode cache layout layer {layer} dsa_index width "
                "does not match config"
            )
            ok = False
            continue
        dsa_ok += 1
    for (kind, layer), segments in by_key.items():
        if len(segments) > 1:
            add_issue(
                f"decode cache layout has duplicate {kind} segment for layer {layer}"
            )
            ok = False
        if kind == "dsa_index" and layer not in expected_dsa_set:
            add_issue(f"decode cache layout layer {layer} has unexpected dsa_index segment")
            ok = False

    return {
        "decode_cache_layout_ok": ok,
        "decode_cache_context_tokens": layout.max_context_tokens,
        "decode_cache_model_type": layout.model_type,
        "decode_cache_dtype": layout.dtype,
        "decode_cache_layout_total_bytes": layout.total_bytes,
        "decode_cache_segment_count": len(layout.segments),
        "decode_cache_mla_kv_segments_checked": config.num_hidden_layers,
        "decode_cache_mla_kv_segments_ok": mla_ok,
        "decode_cache_dsa_index_segments_checked": len(expected_dsa_layers),
        "decode_cache_dsa_index_segments_ok": dsa_ok,
    }


def _glm_4bit_readiness(
    prepared: PreparedManifest,
    config: ModelConfig,
) -> dict[str, Any]:
    issues: list[str] = []

    def add_issue(message: str) -> None:
        if len(issues) < 24:
            issues.append(message)

    try:
        layout = _read_json_object(prepared.experts_layout, "expert layout")
    except PreparedServerError as exc:
        return {
            "analyzed": False,
            "ok": False,
            "issues": [str(exc)],
        }

    layout_quantization = layout.get("quantization")
    layout_group_size = layout.get("group_size")
    layout_model_type = layout.get("model_type")
    expert_layout_config_sha256 = _layout_config_sha256_readiness(
        layout,
        label="expert layout",
        prepared=prepared,
        add_issue=add_issue,
    )
    layout_layers = layout.get("layers")
    try:
        moe_layers = tuple(config.moe_layers)
        routed_experts = config.routed_experts
        experts_per_token = config.experts_per_token
        moe_hidden = config.moe_hidden_size
    except ConfigError as exc:
        moe_layers = ()
        routed_experts = None
        experts_per_token = None
        moe_hidden = None
        add_issue(str(exc))

    if config.model_type != "glm_moe_dsa":
        add_issue(f"model_type {config.model_type!r} is not glm_moe_dsa")
    raw_attention_bias = _raw_config_value(config, "attention_bias")
    if raw_attention_bias not in (None, False):
        add_issue("config attention_bias=true is not supported by the GLM runner")
    raw_hidden_act = _raw_config_value(config, "hidden_act")
    if raw_hidden_act is not None and raw_hidden_act != "silu":
        add_issue(
            f"config hidden_act {raw_hidden_act!r} is not supported by the GLM runner"
        )
    if layout_model_type != config.model_type:
        add_issue(
            "expert layout model_type "
            f"{layout_model_type!r} does not match config {config.model_type!r}"
        )
    if layout_quantization in AFFINE_INT4_EXPERT_QUANTIZATIONS:
        expert_quantization_family = "affine-int4"
        expected_component_order = DEFAULT_EXPERT_COMPONENTS
    elif layout_quantization in MXFP4_EXPERT_QUANTIZATIONS:
        expert_quantization_family = "mlx-mxfp4"
        expected_component_order = MXFP4_EXPERT_COMPONENTS
    else:
        expert_quantization_family = None
        expected_component_order = ()
        add_issue(
            f"expert layout quantization {layout_quantization!r} is not a "
            "supported GLM 4bit expert format"
        )
    expert_quantization_label = expert_quantization_family or "GLM 4bit"
    if prepared.expert_quantization is None:
        add_issue("manifest missing expert_quantization")
    elif prepared.expert_quantization != layout_quantization:
        add_issue("manifest expert_quantization does not match expert layout")
    if type(layout_group_size) is not int or layout_group_size <= 0:
        add_issue("expert layout group_size must be a positive integer")
        expected_components: dict[
            str,
            tuple[tuple[str, ...], int, tuple[int, int]],
        ] = {}
    else:
        if prepared.expert_group_size is None:
            add_issue("manifest missing expert_group_size")
        elif prepared.expert_group_size != layout_group_size:
            add_issue("manifest expert_group_size does not match expert layout")
        try:
            if expert_quantization_family == "affine-int4":
                expected_components = _expected_affine_int4_components(
                    config,
                    int(layout_group_size),
                )
            elif expert_quantization_family == "mlx-mxfp4":
                expected_components = _expected_mxfp4_components(
                    config,
                    int(layout_group_size),
                )
            else:
                expected_components = {}
        except (ConfigError, PreparedServerError) as exc:
            expected_components = {}
            add_issue(str(exc))

    if not isinstance(layout_layers, list):
        add_issue("expert layout missing layers array")
        layer_items: list[dict[str, Any]] = []
    else:
        layer_items = [item for item in layout_layers if isinstance(item, dict)]
        if len(layer_items) != len(layout_layers):
            add_issue("expert layout layers must be objects")

    layer_ids = tuple(
        int(item["layer"])
        for item in layer_items
        if type(item.get("layer")) is int
    )
    layout_num_layers = layout.get("num_layers")
    if moe_layers and layer_ids != moe_layers:
        add_issue("expert layout layers do not match config MoE layers")
    if type(layout_num_layers) is not int or layout_num_layers <= 0:
        add_issue("expert layout num_layers must match config num_hidden_layers")
        layout_model_layer_count = None
    else:
        layout_model_layer_count = int(layout_num_layers)
        if layout_model_layer_count != config.num_hidden_layers:
            add_issue("expert layout num_layers does not match config num_hidden_layers")
    layout_num_experts = layout.get("num_experts")
    if routed_experts is not None and layout_num_experts != routed_experts:
        add_issue("expert layout num_experts does not match config")
    expected_slot_bytes = sum(item[1] for item in expected_components.values())
    expected_expert_layer_bytes = (
        expected_slot_bytes * int(routed_experts)
        if expected_slot_bytes and routed_experts is not None
        else None
    )
    expected_total_expert_bytes = (
        expected_expert_layer_bytes * len(moe_layers)
        if expected_expert_layer_bytes is not None
        else None
    )
    expected_decode_token_routed_expert_read_bytes = (
        expected_slot_bytes * int(experts_per_token) * len(moe_layers)
        if expected_slot_bytes and experts_per_token is not None
        else None
    )
    expected_full_prompt_routed_expert_sweep_bytes = expected_total_expert_bytes
    if (
        expected_total_expert_bytes is not None
        and prepared.expert_layout_bytes is not None
        and prepared.expert_layout_bytes != expected_total_expert_bytes
    ):
        add_issue(
            "prepared expert bytes do not match config-derived "
            f"{expert_quantization_label} total"
        )
    component_order = layout.get("component_order")
    if (
        not isinstance(component_order, list)
        or not all(isinstance(name, str) for name in component_order)
    ):
        add_issue(
            "expert layout component_order must be supported GLM 4bit components"
        )
    elif expected_component_order and tuple(component_order) != expected_component_order:
        add_issue(
            "expert layout component_order does not match "
            f"{expert_quantization_label} slot order"
        )

    expert_layer_file_count = 0
    seen_layer_files: set[str] = set()
    prepared_expert_layer_file_bytes: int | None = 0
    expert_layer_files_exact_size = True
    for item in layer_items:
        layer_id = item.get("layer")
        layer_label = f"layer {layer_id}" if type(layer_id) is int else "layer"
        item_num_experts = item.get("num_experts")
        if routed_experts is not None and item_num_experts != routed_experts:
            add_issue(f"expert layout {layer_label} num_experts does not match config")
        layer_file = item.get("layer_file")
        layer_file_actual_bytes: int | None = None
        if not isinstance(layer_file, str) or not layer_file:
            expert_layer_files_exact_size = False
            add_issue(f"expert layout {layer_label} missing layer_file")
        else:
            try:
                layer_path = _layout_relative_file_path(
                    prepared.experts_layout,
                    layer_file,
                    label=f"expert layout {layer_label} layer_file",
                )
            except PreparedServerError as exc:
                expert_layer_files_exact_size = False
                add_issue(str(exc))
            else:
                expert_layer_file_count += 1
                layer_key = str(layer_path)
                if layer_key in seen_layer_files:
                    expert_layer_files_exact_size = False
                    add_issue(
                        f"expert layout {layer_label} layer_file {layer_file} is duplicated"
                    )
                seen_layer_files.add(layer_key)
                try:
                    layer_file_actual_bytes = layer_path.stat().st_size
                except OSError as exc:
                    expert_layer_files_exact_size = False
                    prepared_expert_layer_file_bytes = None
                    add_issue(
                        f"failed to stat expert layout {layer_label} file "
                        f"{layer_file}: {exc}"
                    )
                else:
                    if prepared_expert_layer_file_bytes is not None:
                        prepared_expert_layer_file_bytes += layer_file_actual_bytes
        slot_bytes = item.get("expert_slot_bytes")
        if type(slot_bytes) is not int or slot_bytes <= 0:
            add_issue(f"expert layout {layer_label} expert_slot_bytes must be positive")
            slot_bytes = 0
        if expected_slot_bytes and slot_bytes != expected_slot_bytes:
            add_issue(
                f"expert layout {layer_label} expert_slot_bytes does not match "
                f"config-derived {expert_quantization_label} slot size"
            )
        expected_layer_file_bytes = (
            int(item_num_experts) * slot_bytes
            if type(item_num_experts) is int and item_num_experts > 0 and slot_bytes
            else None
        )
        if (
            layer_file_actual_bytes is not None
            and expected_layer_file_bytes is not None
            and layer_file_actual_bytes != expected_layer_file_bytes
        ):
            expert_layer_files_exact_size = False
            add_issue(
                f"expert layout {layer_label} file bytes {layer_file_actual_bytes} "
                f"do not match expected packed layer bytes {expected_layer_file_bytes}"
            )
        components = item.get("components")
        if not isinstance(components, list):
            add_issue(f"expert layout {layer_label} missing components array")
            continue
        by_name: dict[str, dict[str, Any]] = {}
        for component in components:
            if not isinstance(component, dict):
                add_issue(f"expert layout {layer_label} components must be objects")
                continue
            name = component.get("name")
            if not isinstance(name, str) or not name:
                add_issue(f"expert layout {layer_label} component missing name")
                continue
            if name in by_name:
                add_issue(f"expert layout {layer_label} component {name} is duplicated")
            by_name[name] = component
        expected_component_offset = 0
        for name in expected_component_order:
            expected = expected_components.get(name)
            expected_size = expected[1] if expected is not None else None
            component = by_name.get(name)
            if component is None:
                add_issue(f"expert layout {layer_label} missing component {name}")
                if expected_size is not None:
                    expected_component_offset += expected_size
                continue
            if expected is None:
                continue
            expected_dtypes, expected_size, expected_shape = expected
            dtype = component.get("dtype")
            if not isinstance(dtype, str) or dtype.upper() not in expected_dtypes:
                add_issue(
                    f"expert layout {layer_label} component {name} dtype "
                    f"does not match {expert_quantization_label} expectation"
                )
            if component.get("size") != expected_size:
                add_issue(
                    f"expert layout {layer_label} component {name} size "
                    f"does not match config-derived {expert_quantization_label} size"
                )
            if _shape2(component.get("shape")) != expected_shape:
                add_issue(
                    f"expert layout {layer_label} component {name} shape "
                    f"does not match config-derived {expert_quantization_label} shape"
                )
            offset = component.get("offset")
            size = component.get("size")
            if type(offset) is not int or offset < 0 or type(size) is not int or size < 0:
                add_issue(f"expert layout {layer_label} component {name} span is invalid")
            else:
                if offset != expected_component_offset:
                    add_issue(
                        f"expert layout {layer_label} component {name} offset "
                        f"does not match {expert_quantization_label} slot order"
                    )
                if slot_bytes and offset + size > slot_bytes:
                    add_issue(
                        f"expert layout {layer_label} component {name} "
                        "exceeds expert slot"
                    )
            expected_component_offset += expected_size
        if (
            slot_bytes
            and expected_component_offset
            and expected_component_offset != slot_bytes
        ):
            add_issue(
                f"expert layout {layer_label} {expert_quantization_label} "
                "component span total "
                "does not match expert slot bytes"
            )

    resident_summary = _glm_resident_layout_readiness(
        prepared,
        config,
        moe_layers=moe_layers,
        add_issue=add_issue,
    )
    decode_cache_summary = _glm_decode_cache_readiness(
        prepared,
        config,
        add_issue=add_issue,
    )
    public_glm_5_2_shape = _public_glm_5_2_shape_report(config)

    return {
        "analyzed": True,
        "ok": not issues,
        "issues": issues,
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "moe_intermediate_size": moe_hidden,
        "num_hidden_layers": config.num_hidden_layers,
        "moe_layer_count": len(moe_layers),
        "dense_prefix_layers": _dense_prefix_layer_count(config),
        "routed_experts": routed_experts,
        "experts_per_token": experts_per_token,
        "tie_word_embeddings": config.tie_word_embeddings,
        "scoring_func": config.scoring_func,
        "topk_method": config.topk_method,
        "norm_topk_prob": config.norm_topk_prob,
        "routed_scaling_factor": config.routed_scaling_factor,
        "n_group": config.n_group,
        "topk_group": config.topk_group,
        "rope_interleave": config.rope_interleave,
        "model_config_sha256": prepared.model_config_sha256,
        "expert_layout_config_sha256": expert_layout_config_sha256,
        "expert_layout_model_type": layout_model_type,
        "expert_layout_quantization": layout_quantization,
        "expert_layout_group_size": layout_group_size,
        "expert_layout_model_layer_count": layout_model_layer_count,
        "expert_layout_moe_layer_count": len(layer_items),
        "expert_layout_layer_count": len(layer_items),
        "expected_expert_slot_bytes": expected_slot_bytes or None,
        "expected_expert_layer_bytes": expected_expert_layer_bytes,
        "expected_total_expert_bytes": expected_total_expert_bytes,
        "expected_decode_token_routed_expert_read_bytes": (
            expected_decode_token_routed_expert_read_bytes
        ),
        "expected_full_prompt_routed_expert_sweep_bytes": (
            expected_full_prompt_routed_expert_sweep_bytes
        ),
        "prepared_expert_layout_bytes": prepared.expert_layout_bytes,
        "prepared_expert_layer_file_bytes": prepared_expert_layer_file_bytes,
        "expert_layer_file_count": expert_layer_file_count,
        "unique_expert_layer_file_count": len(seen_layer_files),
        "expert_layer_files_exact_size": expert_layer_files_exact_size,
        "prepared_resident_layout_bytes": prepared.resident_layout_bytes,
        "prepared_decode_cache_file_bytes": prepared.decode_cache_file_bytes,
        "matches_public_glm_5_2_shape": _is_public_glm_5_2_shape(config),
        "public_glm_5_2_shape": public_glm_5_2_shape,
    } | resident_summary | decode_cache_summary


def prepared_glm_4bit_readiness(
    prepared: PreparedManifest,
    model_config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(model_config_path or prepared.model_dir)
    return _glm_4bit_readiness(prepared, config)


def _glm_4bit_readiness_failure_reason(readiness: dict[str, Any]) -> str | None:
    if readiness.get("ok") is True:
        return None
    issues = readiness.get("issues")
    if isinstance(issues, list) and issues:
        return "; ".join(str(issue) for issue in issues[:5])
    return "readiness check did not pass"


def _public_glm_5_2_shape_failure_reason(
    readiness: dict[str, Any],
) -> str | None:
    reason = _glm_4bit_readiness_failure_reason(readiness)
    if reason is not None:
        return reason
    if readiness.get("matches_public_glm_5_2_shape") is not True:
        detail = _public_glm_5_2_shape_failure_detail(readiness)
        if detail:
            return (
                "prepared config does not match the public GLM-5.2 shape "
                f"({detail})"
            )
        return "prepared config does not match the public GLM-5.2 shape"
    return None


def _prefill_backend_health(
    configured_backend: str,
    *,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
    compile_mpp_probe: bool,
    run_mpp_probe: bool,
    run_mpsgraph_probe: bool,
    probe_timeout_seconds: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    capability: dict[str, Any] | None = None
    effective_backend = configured_backend
    try:
        backend = inspect_prefill_backend(
            run_host_probe=True,
            compile_mpp_probe=compile_mpp_probe,
            run_mpp_probe=run_mpp_probe,
            run_mpsgraph_probe=run_mpsgraph_probe,
            probe_timeout_seconds=probe_timeout_seconds,
        )
    except Exception as exc:
        warnings.append(f"prefill backend inspection failed: {exc}")
        if configured_backend == "mpp-f32":
            warnings.append("mpp-f32 was forced but MPP TensorOps support was not verified")
        if configured_backend == "mpsgraph-f32":
            warnings.append("mpsgraph-f32 was forced but MPSGraph support was not verified")
        if configured_backend == "auto":
            effective_backend = "custom-metal"
    else:
        mpsgraph_runtime_available = _mps_graph_runtime_available_from_backend(backend)
        acceleration_runtimes = prefill_acceleration_runtimes(backend)
        selectable_accelerated = selectable_accelerated_prefill_backends(backend)
        validated_accelerated = validated_accelerated_prefill_backends(backend)
        if (
            not validated_accelerated
            and run_mpsgraph_probe
            and mpsgraph_runtime_available
            and getattr(backend, "mps_graph_probe_ran", False) is True
            and getattr(backend, "mps_graph_probe_ok", None) is True
        ):
            validated_accelerated = ("mpsgraph-f32",)
        acceleration_runtime_gaps = prefill_acceleration_runtime_gaps(backend)
        capability = {
            "sdk_path": str(backend.sdk_path) if backend.sdk_path is not None else None,
            "recommended_backend": backend.recommended_backend,
            "mps_graph_matmul_declared": backend.mps_graph_matmul_declared,
            "mps_graph_runtime_available": mpsgraph_runtime_available,
            "mps_graph_probe_requested": run_mpsgraph_probe,
            "mps_graph_probe_ran": getattr(backend, "mps_graph_probe_ran", False),
            "mps_graph_probe_ok": getattr(backend, "mps_graph_probe_ok", None),
            "mps_graph_probe_error": getattr(
                backend,
                "mps_graph_probe_error",
                None,
            ),
            "metal4_ml_runtime_available": backend.metal4_ml_runtime_available,
            "mpp_runtime_available": backend.mpp_runtime_available,
            "mpp_compile_probe_requested": compile_mpp_probe,
            "mpp_compile_probe_ran": getattr(backend, "mpp_compile_probe_ran", False),
            "mpp_compile_probe_ok": getattr(backend, "mpp_compile_probe_ok", None),
            "mpp_compile_variant": getattr(backend, "mpp_compile_variant", None),
            "mpp_compile_error": getattr(backend, "mpp_compile_error", None),
            "mpp_run_probe_requested": run_mpp_probe,
            "mpp_run_probe_ran": getattr(backend, "mpp_run_probe_ran", False),
            "mpp_run_probe_ok": getattr(backend, "mpp_run_probe_ok", None),
            "mpp_run_probe_error": getattr(backend, "mpp_run_probe_error", None),
            "mpp_run_probe_max_abs_error": getattr(
                backend,
                "mpp_run_probe_max_abs_error",
                None,
            ),
            "mpp_run_probe_kernel_variant": getattr(
                backend,
                "mpp_run_probe_kernel_variant",
                None,
            ),
            "mpp_run_probe_shape": getattr(backend, "mpp_run_probe_shape", None),
            "mpp_run_probe_dtype": getattr(backend, "mpp_run_probe_dtype", None),
            "mpp_run_probe_execution_path": getattr(
                backend,
                "mpp_run_probe_execution_path",
                None,
            ),
            "prefill_acceleration_runtimes": acceleration_runtimes,
            "selectable_accelerated_prefill_backends": selectable_accelerated,
            "validated_accelerated_prefill_backends": validated_accelerated,
            "prefill_acceleration_runtime_gaps": acceleration_runtime_gaps,
            "prefill_neural_accelerator_status": (
                prefill_neural_accelerator_status(backend)
            ),
            "selectable_prefill_acceleration_available": bool(
                selectable_accelerated
            ),
            "validated_prefill_acceleration_available": bool(
                validated_accelerated
            ),
            "suggested_prefill_acceleration_flags": (
                suggested_prefill_acceleration_flags(
                    backend,
                    source="prepared_health",
                )
            ),
            "reasons": backend.reasons,
        }
        host_probe_requested = bool(getattr(backend, "host_probe_requested", True))
        host_probe_path = getattr(backend, "host_probe_path", None)
        if host_probe_requested and host_probe_path is None:
            host_probe_path = default_prefill_backend_probe_path()
        host_probe_ran = getattr(backend, "host_probe_ran", None)
        if not isinstance(host_probe_ran, bool):
            host_probe_ran = bool(
                getattr(backend, "mps_graph_probe_ran", False)
                or mpsgraph_runtime_available
            )
        host_probe_ok = getattr(backend, "host_probe_ok", None)
        if not isinstance(host_probe_ok, bool):
            host_probe_ok = bool(host_probe_ran and mpsgraph_runtime_available)
        capability.update(
            {
                "host_probe_requested": host_probe_requested,
                "host_probe_path": (
                    str(host_probe_path) if host_probe_path is not None else None
                ),
                "host_probe_ran": host_probe_ran,
                "host_probe_ok": host_probe_ok,
                "host_probe_error": getattr(backend, "host_probe_error", None),
                "prefill_backend_probe_timeout_seconds": float(
                    getattr(
                        backend,
                        "probe_timeout_seconds",
                        probe_timeout_seconds,
                    )
                ),
            }
        )
        if configured_backend == "mpsgraph-f32" and not mpsgraph_runtime_available:
            if not backend.mps_graph_matmul_declared:
                warnings.append(
                    "mpsgraph-f32 was forced but MPSGraph matmul headers were not found"
                )
            else:
                warnings.append(
                    "mpsgraph-f32 was forced but MPSGraph runtime was not verified"
                )
        elif configured_backend == "mpp-f32" and not backend.mpp_runtime_available:
            warnings.append(
                "mpp-f32 was forced but MPP TensorOps runtime was not verified"
            )
        elif configured_backend == "auto" and not selectable_accelerated:
            if not backend.mps_graph_matmul_declared:
                warnings.append(
                    "auto prefill will fall back to custom-metal resident GEMMs because "
                    "MPSGraph matmul headers were not found"
                )
            else:
                warnings.append(
                    "auto prefill will fall back to custom-metal resident GEMMs because "
                    "the MPSGraph runtime probe did not succeed"
                )
            effective_backend = "custom-metal"
        elif configured_backend == "auto":
            effective_backend = "auto"
    return {
        "configured_backend": configured_backend,
        "effective_backend": effective_backend,
        "capability": capability,
        "auto_policy": {
            "mpsgraph_min_batch_tokens": mpsgraph_min_batch_tokens,
            "mpsgraph_min_matrix_dim": mpsgraph_min_matrix_dim,
            "default_mpsgraph_min_batch_tokens": AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
            "default_mpsgraph_min_matrix_dim": AUTO_MPSGRAPH_MIN_DIM,
            "mpsgraph_dtypes": tuple(sorted(PREFILL_LINEAR_MPSGRAPH_DTYPES)),
            "compile_mpp_probe": compile_mpp_probe,
            "run_mpp_probe": run_mpp_probe,
            "run_mpsgraph_probe": run_mpsgraph_probe,
        },
        "warnings": tuple(warnings),
    }


def _resolve_server_prefill_linear_backend(
    configured_backend: str,
    *,
    compile_mpp_probe: bool = False,
    run_mpp_probe: bool = False,
    run_mpsgraph_probe: bool = False,
    probe_timeout_seconds: float = DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
) -> str:
    if configured_backend != "auto":
        return configured_backend
    try:
        backend = inspect_prefill_backend(
            run_host_probe=True,
            compile_mpp_probe=compile_mpp_probe,
            run_mpp_probe=run_mpp_probe,
            run_mpsgraph_probe=run_mpsgraph_probe,
            probe_timeout_seconds=probe_timeout_seconds,
        )
    except Exception:
        return "custom-metal"
    validated = validated_accelerated_prefill_backends(backend)
    if "mpp-f32" in validated:
        return PREFILL_LINEAR_AUTO_MPP_BACKEND
    if validated:
        return "auto"
    selectable = selectable_accelerated_prefill_backends(backend)
    return "auto" if "mpsgraph-f32" in selectable else "custom-metal"


def _server_prefill_acceleration_required(config: "PreparedServerConfig") -> bool:
    return (
        config.require_prefill_acceleration
        or config.prefill_min_accelerated_flop_fraction > 0.0
    )


def _require_server_prefill_acceleration_backend(
    config: "PreparedServerConfig",
) -> None:
    if not _server_prefill_acceleration_required(config):
        return
    if config.prefill_linear_backend == "custom-metal":
        raise PreparedServerError(
            "prefill acceleration requirement failed: custom-metal was configured"
        )
    try:
        backend = inspect_prefill_backend(
            run_host_probe=True,
            compile_mpp_probe=config.prefill_compile_mpp_probe,
            run_mpp_probe=config.prefill_run_mpp_probe,
            run_mpsgraph_probe=config.prefill_run_mpsgraph_probe,
            probe_timeout_seconds=config.prefill_backend_probe_timeout_seconds,
        )
    except Exception as exc:
        raise PreparedServerError(
            f"prefill backend inspection failed: {exc}"
        ) from exc
    gate = evaluate_prefill_acceleration_requirement(
        configured_backend=config.prefill_linear_backend,
        mps_graph_runtime_available=backend.mps_graph_runtime_available,
        mpp_runtime_available=backend.mpp_runtime_available,
        mps_graph_probe_requested=getattr(
            backend,
            "mps_graph_probe_requested",
            None,
        ),
        mps_graph_probe_ran=getattr(backend, "mps_graph_probe_ran", None),
        mps_graph_probe_ok=getattr(backend, "mps_graph_probe_ok", None),
        mpp_run_probe_requested=getattr(
            backend,
            "mpp_run_probe_requested",
            None,
        ),
        mpp_run_probe_ran=getattr(backend, "mpp_run_probe_ran", None),
        mpp_run_probe_ok=getattr(backend, "mpp_run_probe_ok", None),
        selectable_backends=selectable_accelerated_prefill_backends(backend),
        acceleration_runtimes=prefill_acceleration_runtimes(backend),
    )
    if gate.ok is not True:
        raise PreparedServerError(
            "prefill acceleration requirement failed: "
            f"{gate.reason}"
        )


def _prefill_acceleration_requirement_health(
    config: "PreparedServerConfig",
    prefill_backend: dict[str, Any],
) -> dict[str, object] | None:
    if not _server_prefill_acceleration_required(config):
        return None
    capability = prefill_backend.get("capability")
    if not isinstance(capability, dict):
        capability = {}
    return evaluate_prefill_acceleration_requirement(
        configured_backend=config.prefill_linear_backend,
        mps_graph_runtime_available=capability.get("mps_graph_runtime_available"),
        mpp_runtime_available=capability.get("mpp_runtime_available"),
        mps_graph_probe_requested=capability.get("mps_graph_probe_requested"),
        mps_graph_probe_ran=capability.get("mps_graph_probe_ran"),
        mps_graph_probe_ok=capability.get("mps_graph_probe_ok"),
        mpp_run_probe_requested=capability.get("mpp_run_probe_requested"),
        mpp_run_probe_ran=capability.get("mpp_run_probe_ran"),
        mpp_run_probe_ok=capability.get("mpp_run_probe_ok"),
        selectable_backends=tuple(
            capability.get("selectable_accelerated_prefill_backends") or ()
        ),
        acceleration_runtimes=tuple(
            capability.get("prefill_acceleration_runtimes") or ()
        ),
    ).to_json()


def _sorted_positive_backend_ints(values: Any) -> dict[str, int]:
    if not isinstance(values, dict):
        return {}
    out: dict[str, int] = {}
    for backend, value in values.items():
        if not isinstance(backend, str) or not backend:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            out[backend] = out.get(backend, 0) + parsed
    return dict(sorted(out.items()))


def _prefill_linear_backend_request_summary(
    *,
    resident_layout_path: Path,
    configured_backend: str,
    effective_backend: str,
    prompt_chunk_tokens: int | None,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
    streamed_routed_expert_hidden_dim: int = 0,
    streamed_routed_expert_top_k: int = 0,
    streamed_routed_expert_moe_hidden_dims: tuple[int, ...] = (),
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "configured": configured_backend,
        "effective": effective_backend,
        "analyzed": False,
        "prompt_chunk_tokens": prompt_chunk_tokens,
        "auto_policy": {
            "mpsgraph_min_batch_tokens": mpsgraph_min_batch_tokens,
            "mpsgraph_min_matrix_dim": mpsgraph_min_matrix_dim,
        },
            "mpp_candidate_policy": {
            "candidate_backend": "mpp_tensor_ops_prefill",
            "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
            "mpp_tensor_ops_min_batch_tokens": DEFAULT_MPP_MIN_TOKENS,
            "mpp_tensor_ops_min_matrix_dim": MPP_TENSOR_OPS_MIN_MATRIX_DIM,
            "selectable_prefill_backend": effective_backend
            in {"mpp-f32", PREFILL_LINEAR_AUTO_MPP_BACKEND},
        },
    }
    if prompt_chunk_tokens is None or prompt_chunk_tokens <= 0:
        summary["reason"] = "batch prompt prefill disabled"
        return summary
    try:
        payload = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparedServerError(
            f"failed to inspect resident layout for prefill backend summary: {exc}"
        ) from exc
    tensors = payload.get("tensors")
    if not isinstance(tensors, list):
        raise PreparedServerError(
            "resident layout missing tensors array for prefill backend summary"
        )

    matrix_count = 0
    mpp_count = 0
    mpsgraph_count = 0
    mps_matrix_count = 0
    custom_count = 0
    unsupported_count = 0
    total_estimated_flops = 0
    mpp_estimated_flops = 0
    mpsgraph_estimated_flops = 0
    mps_matrix_estimated_flops = 0
    custom_estimated_flops = 0
    unsupported_estimated_flops = 0
    mpp_candidate_count = 0
    mpp_candidate_estimated_flops = 0
    mpp_candidate_backend_counts: dict[str, int] = {}
    mpp_candidate_backend_flops: dict[str, int] = {}
    max_scratch = 0
    total_scratch = 0
    total_raw_conversion = 0
    max_raw_conversion = 0
    router_gate_matrix_count = 0
    router_gate_estimated_flops = 0
    router_gate_accelerated_matrix_count = 0
    router_gate_accelerated_estimated_flops = 0
    top_matrices: list[dict[str, Any]] = []
    streamed_totals: dict[str, int] = {}
    tensors_by_name = _resident_tensors_by_name(payload)
    mxfp4_companion_names: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        router_gate_tensor = is_prompt_prefill_router_gate_tensor(name)
        if (
            not is_prompt_prefill_resident_linear_backend_tensor(name)
            and not router_gate_tensor
        ):
            continue
        if name in mxfp4_companion_names:
            continue
        if name.endswith(".scales"):
            base = name[: -len(".scales")]
            companion_weight = tensors_by_name.get(f"{base}.weight")
            if companion_weight is not None:
                mxfp4_shape = _resident_mxfp4_logical_shape(
                    payload,
                    companion_weight,
                )
                if mxfp4_shape is not None:
                    continue
        mxfp4_shape = _resident_mxfp4_logical_shape(payload, tensor)
        if mxfp4_shape is not None:
            base = name[: -len(".weight")]
            scales = tensors_by_name.get(f"{base}.scales")
            if scales is not None:
                scale_name = _resident_tensor_name(scales)
                if scale_name is not None:
                    mxfp4_companion_names.add(scale_name)
            if len(mxfp4_shape) != 2:
                continue
            rows, cols = int(mxfp4_shape[0]), int(mxfp4_shape[1])
            weight_size = _resident_tensor_size(tensor)
            scale_size = _resident_tensor_size(scales) if scales is not None else None
            if weight_size is None or scale_size is None:
                raise PreparedServerError(
                    f"resident MXFP4 tensor {name} size metadata is invalid"
                )
            size = weight_size + scale_size
            dtype = "mlx-mxfp4"
        else:
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) < 2:
                continue
            if type(shape[0]) is not int or type(shape[1]) is not int:
                raise PreparedServerError(
                    f"resident tensor {name} shape must use integer rows and cols"
                )
            rows, cols = int(shape[0]), int(shape[1])
            dtype = str(tensor.get("dtype") or "")
            raw_size = tensor.get("size")
            if type(raw_size) is not int:
                raise PreparedServerError(
                    f"resident tensor {name} size must be an integer"
                )
            size = int(raw_size)
        if rows <= 0 or cols <= 0:
            continue
        estimated_flops = 2 * int(prompt_chunk_tokens) * rows * cols
        if effective_backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS:
            if dtype not in PREFILL_LINEAR_MPSGRAPH_DTYPES:
                resolved_backend = "unsupported-mpsgraph"
            else:
                resolved_backend = effective_backend
        elif effective_backend in {"auto", PREFILL_LINEAR_AUTO_MPP_BACKEND}:
            resolved_backend = (
                (
                    "mpp-f32"
                    if effective_backend == PREFILL_LINEAR_AUTO_MPP_BACKEND
                    else "mpsgraph-f32"
                )
                if dtype in PREFILL_LINEAR_MPSGRAPH_DTYPES
                and prompt_chunk_tokens >= mpsgraph_min_batch_tokens
                and min(rows, cols) >= mpsgraph_min_matrix_dim
                else "custom-metal"
            )
        else:
            resolved_backend = "custom-metal"
        if router_gate_tensor and resolved_backend == "custom-metal":
            continue
        if router_gate_tensor:
            router_gate_matrix_count += 1
            router_gate_estimated_flops += estimated_flops
            if resolved_backend in PREFILL_LINEAR_ACCELERATED_BACKENDS:
                router_gate_accelerated_matrix_count += 1
                router_gate_accelerated_estimated_flops += estimated_flops
        total_estimated_flops += estimated_flops
        matrix_count += 1
        mpp_tensor_ops_candidate = (
            prompt_chunk_tokens >= DEFAULT_MPP_MIN_TOKENS
            and cols >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
            and rows >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
            )
        if mpp_tensor_ops_candidate:
            mpp_candidate_count += 1
            mpp_candidate_estimated_flops += estimated_flops
            mpp_candidate_backend_counts[resolved_backend] = (
                mpp_candidate_backend_counts.get(resolved_backend, 0) + 1
            )
            mpp_candidate_backend_flops[resolved_backend] = (
                mpp_candidate_backend_flops.get(resolved_backend, 0)
                + estimated_flops
            )
        if resolved_backend == "unsupported-mpsgraph":
            unsupported_count += 1
            unsupported_estimated_flops += estimated_flops
            continue
        if resolved_backend == "mpp-f32":
            mpp_count += 1
            mpp_estimated_flops += estimated_flops
        elif resolved_backend == "mpsgraph-f32":
            mpsgraph_count += 1
            mpsgraph_estimated_flops += estimated_flops
        elif resolved_backend == "mps-matrix-f32":
            mps_matrix_count += 1
            mps_matrix_estimated_flops += estimated_flops
        else:
            custom_count += 1
            custom_estimated_flops += estimated_flops
        scratch = _resident_linear_matrix_scratch(
            matrix_bytes=size,
            dtype=dtype,
            in_dim=cols,
            out_dim=rows,
            backend=resolved_backend,
        )
        top_matrices.append(
            {
                "rank": 0,
                "name": name,
                "rows": rows,
                "cols": cols,
                "dtype": dtype,
                "size_bytes": size,
                "resolved_backend": resolved_backend,
                "estimated_flops": estimated_flops,
                "mpp_tensor_ops_candidate": mpp_tensor_ops_candidate,
                "matrix_scratch_bytes": scratch.matrix_scratch_bytes,
                "matrix_raw_conversion_bytes": scratch.matrix_raw_conversion_bytes,
            }
        )
        max_scratch = max(max_scratch, scratch.matrix_scratch_bytes)
        total_scratch += scratch.matrix_scratch_bytes
        total_raw_conversion += scratch.matrix_raw_conversion_bytes
        max_raw_conversion = max(
            max_raw_conversion,
            scratch.matrix_raw_conversion_bytes,
        )
    streamed_accounting = _streamed_routed_expert_linear_accounting_for_layers(
        batch_tokens=int(prompt_chunk_tokens),
        top_k=streamed_routed_expert_top_k,
        hidden_dim=streamed_routed_expert_hidden_dim,
        moe_hidden_dims=streamed_routed_expert_moe_hidden_dims,
    )
    streamed_flops = int(streamed_accounting.get("estimated_flops") or 0)
    if streamed_flops > 0:
        streamed_matrix_count = int(streamed_accounting.get("matrix_count") or 0)
        streamed_assignments = int(streamed_accounting.get("assignments") or 0)
        streamed_layer_count = int(streamed_accounting.get("layer_count") or 0)
        streamed_mpp_count = int(
            streamed_accounting.get("mpp_candidate_matrix_count") or 0
        )
        streamed_mpp_flops = int(
            streamed_accounting.get("mpp_candidate_estimated_flops") or 0
        )
        matrix_count += streamed_matrix_count
        custom_count += streamed_matrix_count
        custom_estimated_flops += streamed_flops
        total_estimated_flops += streamed_flops
        mpp_candidate_count += streamed_mpp_count
        mpp_candidate_estimated_flops += streamed_mpp_flops
        if streamed_mpp_count > 0:
            mpp_candidate_backend_counts["custom-metal"] = (
                mpp_candidate_backend_counts.get("custom-metal", 0)
                + streamed_mpp_count
            )
        if streamed_mpp_flops > 0:
            mpp_candidate_backend_flops["custom-metal"] = (
                mpp_candidate_backend_flops.get("custom-metal", 0)
                + streamed_mpp_flops
            )
        streamed_totals = {
            "layer_count": streamed_layer_count,
            "matrix_count": streamed_matrix_count,
            "assignments": streamed_assignments,
            "estimated_flops": streamed_flops,
            "mpp_candidate_matrix_count": streamed_mpp_count,
            "mpp_candidate_estimated_flops": streamed_mpp_flops,
        }
        top_matrices.append(
            {
                "rank": 0,
                "name": "streamed_routed_expert_mlp",
                "rows": streamed_routed_expert_moe_hidden_dims[0],
                "cols": streamed_routed_expert_hidden_dim,
                "dtype": "streamed-expert",
                "size_bytes": 0,
                "resolved_backend": "custom-metal",
                "estimated_flops": streamed_flops,
                "mpp_tensor_ops_candidate": streamed_mpp_count > 0,
                "matrix_scratch_bytes": 0,
                "matrix_raw_conversion_bytes": 0,
                "streamed_routed_expert_layer_count": streamed_layer_count,
                "streamed_routed_expert_assignments": streamed_assignments,
            }
        )
    top_matrices = sorted(
        top_matrices,
        key=lambda item: (
            -int(item["estimated_flops"]),
            -int(item["size_bytes"]),
            str(item["name"]),
        ),
    )[:8]
    for rank, item in enumerate(top_matrices, start=1):
        item["rank"] = rank
    summary.update(
        {
            "analyzed": True,
            "matrix_count": matrix_count,
            "accelerated_matrix_count": mpp_count + mpsgraph_count + mps_matrix_count,
            "mpp_matrix_count": mpp_count,
            "mpsgraph_matrix_count": mpsgraph_count,
            "mps_matrix_matrix_count": mps_matrix_count,
            "custom_metal_matrix_count": custom_count,
            "unsupported_mpsgraph_matrix_count": unsupported_count,
            "total_estimated_flops": total_estimated_flops,
            "accelerated_estimated_flops": (
                mpp_estimated_flops
                + mpsgraph_estimated_flops
                + mps_matrix_estimated_flops
            ),
            "mpp_estimated_flops": mpp_estimated_flops,
            "mpsgraph_estimated_flops": mpsgraph_estimated_flops,
            "mps_matrix_estimated_flops": mps_matrix_estimated_flops,
            "custom_metal_estimated_flops": custom_estimated_flops,
            "unsupported_mpsgraph_estimated_flops": unsupported_estimated_flops,
            "router_gate_matrix_count": router_gate_matrix_count,
            "router_gate_estimated_flops": router_gate_estimated_flops,
            "router_gate_accelerated_matrix_count": (
                router_gate_accelerated_matrix_count
            ),
            "router_gate_accelerated_estimated_flops": (
                router_gate_accelerated_estimated_flops
            ),
            "mpp_tensor_ops_candidate_matrix_count": mpp_candidate_count,
            "mpp_tensor_ops_candidate_estimated_flops": (
                mpp_candidate_estimated_flops
            ),
            "mpp_tensor_ops_candidate_flop_fraction": (
                mpp_candidate_estimated_flops / total_estimated_flops
                if total_estimated_flops > 0
                else 0.0
            ),
            "mpp_tensor_ops_candidate_backend_counts": dict(
                sorted(mpp_candidate_backend_counts.items())
            ),
            "mpp_tensor_ops_candidate_backend_flops": dict(
                sorted(mpp_candidate_backend_flops.items())
            ),
            "streamed_routed_expert_layer_count": streamed_totals.get(
                "layer_count",
                0,
            ),
            "streamed_routed_expert_matrix_count": streamed_totals.get(
                "matrix_count",
                0,
            ),
            "streamed_routed_expert_assignments": streamed_totals.get(
                "assignments",
                0,
            ),
            "streamed_routed_expert_estimated_flops": streamed_totals.get(
                "estimated_flops",
                0,
            ),
            "streamed_routed_expert_mpp_candidate_matrix_count": (
                streamed_totals.get("mpp_candidate_matrix_count", 0)
            ),
            "streamed_routed_expert_mpp_candidate_estimated_flops": (
                streamed_totals.get("mpp_candidate_estimated_flops", 0)
            ),
            "max_matrix_scratch_bytes": max_scratch,
            "total_matrix_scratch_bytes": total_scratch,
            "total_matrix_raw_conversion_bytes": total_raw_conversion,
            "max_matrix_raw_conversion_bytes": max_raw_conversion,
            "top_matrices": tuple(top_matrices),
        }
    )
    return summary


def _prefill_acceleration_coverage_summary(
    linear_summary: dict[str, Any],
    *,
    required: bool,
    min_accelerated_flop_fraction: float = 0.0,
    allow_router_gate_only_acceleration: bool = False,
) -> dict[str, Any]:
    analyzed = bool(linear_summary.get("analyzed"))
    matrix_count = int(linear_summary.get("matrix_count") or 0)
    mpp_count = int(linear_summary.get("mpp_matrix_count") or 0)
    mpsgraph_count = int(linear_summary.get("mpsgraph_matrix_count") or 0)
    mps_matrix_count = int(linear_summary.get("mps_matrix_matrix_count") or 0)
    custom_count = int(linear_summary.get("custom_metal_matrix_count") or 0)
    unsupported_count = int(
        linear_summary.get("unsupported_mpsgraph_matrix_count") or 0
    )
    total_estimated_flops = int(linear_summary.get("total_estimated_flops") or 0)
    accelerated_estimated_flops = int(
        linear_summary.get("accelerated_estimated_flops") or 0
    )
    custom_estimated_flops = int(
        linear_summary.get("custom_metal_estimated_flops") or 0
    )
    unsupported_estimated_flops = int(
        linear_summary.get("unsupported_mpsgraph_estimated_flops") or 0
    )
    router_gate_count = min(
        max(0, int(linear_summary.get("router_gate_matrix_count") or 0)),
        matrix_count,
    )
    router_gate_estimated_flops = min(
        max(0, int(linear_summary.get("router_gate_estimated_flops") or 0)),
        total_estimated_flops,
    )
    router_gate_accelerated_count = min(
        max(
            0,
            int(linear_summary.get("router_gate_accelerated_matrix_count") or 0),
        ),
        router_gate_count,
    )
    router_gate_accelerated_estimated_flops = min(
        max(
            0,
            int(
                linear_summary.get(
                    "router_gate_accelerated_estimated_flops",
                )
                or 0
            ),
        ),
        router_gate_estimated_flops,
        accelerated_estimated_flops,
    )
    mpp_candidate_count = int(
        linear_summary.get("mpp_tensor_ops_candidate_matrix_count") or 0
    )
    mpp_candidate_estimated_flops = int(
        linear_summary.get("mpp_tensor_ops_candidate_estimated_flops") or 0
    )
    mpp_candidate_backend_counts = _sorted_positive_backend_ints(
        linear_summary.get("mpp_tensor_ops_candidate_backend_counts")
    )
    mpp_candidate_backend_flops = _sorted_positive_backend_ints(
        linear_summary.get("mpp_tensor_ops_candidate_backend_flops")
    )
    streamed_layer_count = int(
        linear_summary.get("streamed_routed_expert_layer_count") or 0
    )
    streamed_matrix_count = int(
        linear_summary.get("streamed_routed_expert_matrix_count") or 0
    )
    streamed_assignments = int(
        linear_summary.get("streamed_routed_expert_assignments") or 0
    )
    streamed_estimated_flops = int(
        linear_summary.get("streamed_routed_expert_estimated_flops") or 0
    )
    streamed_mpp_count = int(
        linear_summary.get(
            "streamed_routed_expert_mpp_candidate_matrix_count",
        )
        or 0
    )
    streamed_mpp_flops = int(
        linear_summary.get(
            "streamed_routed_expert_mpp_candidate_estimated_flops",
        )
        or 0
    )
    other_estimated_flops = max(
        0,
        total_estimated_flops
        - accelerated_estimated_flops
        - custom_estimated_flops
        - unsupported_estimated_flops,
    )
    accelerated_flop_fraction = (
        accelerated_estimated_flops / total_estimated_flops
        if total_estimated_flops > 0
        else 0.0
    )
    mpp_candidate_flop_fraction = (
        mpp_candidate_estimated_flops / total_estimated_flops
        if total_estimated_flops > 0
        else 0.0
    )
    accelerated_count = int(
        linear_summary.get(
            "accelerated_matrix_count",
            mpp_count + mpsgraph_count + mps_matrix_count,
        )
        or 0
    )
    other_count = max(
        0,
        matrix_count - accelerated_count - custom_count - unsupported_count,
    )
    router_gate_accelerated_count = min(
        router_gate_accelerated_count,
        accelerated_count,
    )
    non_router_accelerated_count = max(
        0,
        accelerated_count - router_gate_accelerated_count,
    )
    non_router_accelerated_estimated_flops = max(
        0,
        accelerated_estimated_flops - router_gate_accelerated_estimated_flops,
    )
    non_router_matrix_count = max(0, matrix_count - router_gate_count)
    non_router_estimated_flops = max(
        0,
        total_estimated_flops - router_gate_estimated_flops,
    )
    non_router_unaccelerated_matrix_count = max(
        0,
        non_router_matrix_count - non_router_accelerated_count,
    )
    non_router_unaccelerated_estimated_flops = max(
        0,
        non_router_estimated_flops - non_router_accelerated_estimated_flops,
    )
    non_router_unaccelerated_flop_fraction = (
        non_router_unaccelerated_estimated_flops / total_estimated_flops
        if total_estimated_flops > 0
        else 0.0
    )
    streamed_unaccelerated_count = min(
        streamed_matrix_count,
        non_router_unaccelerated_matrix_count,
    )
    streamed_unaccelerated_flops = min(
        streamed_estimated_flops,
        non_router_unaccelerated_estimated_flops,
    )
    non_streamed_unaccelerated_count = max(
        0,
        non_router_unaccelerated_matrix_count - streamed_unaccelerated_count,
    )
    non_streamed_unaccelerated_flops = max(
        0,
        non_router_unaccelerated_estimated_flops - streamed_unaccelerated_flops,
    )
    unaccelerated_backend_matrix_counts = _sorted_positive_backend_ints(
        {
            "custom-metal": custom_count,
            "unsupported-mpsgraph": unsupported_count,
            "other": other_count,
        }
    )
    unaccelerated_backend_estimated_flops = _sorted_positive_backend_ints(
        {
            "custom-metal": custom_estimated_flops,
            "unsupported-mpsgraph": unsupported_estimated_flops,
            "other": other_estimated_flops,
        }
    )
    accelerated_router_gate_flop_share = (
        router_gate_accelerated_estimated_flops / accelerated_estimated_flops
        if accelerated_estimated_flops > 0
        else 0.0
    )
    accelerated_router_gate_only = (
        accelerated_count > 0
        and router_gate_accelerated_count == accelerated_count
        and router_gate_accelerated_estimated_flops
        == accelerated_estimated_flops
    )
    accelerated_backends = tuple(
        backend
        for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        if (
            mpp_count
            if backend == "mpp-f32"
            else (
                mpsgraph_count
                if backend == "mpsgraph-f32"
                else mps_matrix_count
            )
        )
        > 0
    )
    any_accelerated = accelerated_count > 0
    all_resident_accelerated = (
        matrix_count > 0
        and accelerated_count == matrix_count
        and custom_count == 0
        and unsupported_count == 0
    )
    effective_backend = str(linear_summary.get("effective") or "")
    f32_backend_has_unsupported_matrices = (
        effective_backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS
        and unsupported_count > 0
    )
    if f32_backend_has_unsupported_matrices:
        reason = (
            f"{effective_backend} cannot run {unsupported_count} resident "
            "prefill matrices with unsupported dtypes"
        )
    elif any_accelerated:
        reason = ""
    elif not analyzed:
        reason = str(linear_summary.get("reason") or "prefill backend not analyzed")
    elif matrix_count <= 0:
        reason = "no resident prefill matrices were found"
    else:
        reason = "no resident prefill matrices resolved to an accelerated backend"
    ok = True
    if required:
        ok = (
            any_accelerated
            and not f32_backend_has_unsupported_matrices
            and (
                allow_router_gate_only_acceleration
                or not accelerated_router_gate_only
            )
            and accelerated_flop_fraction >= min_accelerated_flop_fraction
        )
        if (
            any_accelerated
            and accelerated_router_gate_only
            and not allow_router_gate_only_acceleration
        ):
            reason = (
                "accelerated prefill coverage comes only from MoE router gates; "
                "pass --allow-router-gate-only-prefill-acceleration only for "
                "explicit routing-drift experiments"
            )
        elif any_accelerated and not ok and not f32_backend_has_unsupported_matrices:
            reason = (
                f"accelerated prefill FLOP fraction "
                f"{accelerated_flop_fraction:.3g} is below required "
                f"{min_accelerated_flop_fraction:.3g}"
            )
    return {
        "required": required,
        "analyzed": analyzed,
        "ok": ok,
        "min_accelerated_flop_fraction": min_accelerated_flop_fraction,
        "matrix_count": matrix_count,
        "accelerated_matrix_count": accelerated_count,
        "mpsgraph_matrix_count": mpsgraph_count,
        "custom_metal_matrix_count": custom_count,
        "unsupported_mpsgraph_matrix_count": unsupported_count,
        "other_matrix_count": other_count,
        "total_estimated_flops": total_estimated_flops,
        "accelerated_estimated_flops": accelerated_estimated_flops,
        "custom_metal_estimated_flops": custom_estimated_flops,
        "unsupported_mpsgraph_estimated_flops": unsupported_estimated_flops,
        "other_estimated_flops": other_estimated_flops,
        "router_gate_matrix_count": router_gate_count,
        "router_gate_estimated_flops": router_gate_estimated_flops,
        "router_gate_accelerated_matrix_count": router_gate_accelerated_count,
        "router_gate_accelerated_estimated_flops": (
            router_gate_accelerated_estimated_flops
        ),
        "non_router_matrix_count": non_router_matrix_count,
        "non_router_estimated_flops": non_router_estimated_flops,
        "non_router_accelerated_matrix_count": non_router_accelerated_count,
        "non_router_accelerated_estimated_flops": (
            non_router_accelerated_estimated_flops
        ),
        "non_router_unaccelerated_matrix_count": (
            non_router_unaccelerated_matrix_count
        ),
        "non_router_unaccelerated_estimated_flops": (
            non_router_unaccelerated_estimated_flops
        ),
        "non_router_unaccelerated_flop_fraction": (
            non_router_unaccelerated_flop_fraction
        ),
        "non_router_unaccelerated_streamed_routed_expert_matrix_count": (
            streamed_unaccelerated_count
        ),
        "non_router_unaccelerated_streamed_routed_expert_estimated_flops": (
            streamed_unaccelerated_flops
        ),
        "non_router_unaccelerated_non_streamed_matrix_count": (
            non_streamed_unaccelerated_count
        ),
        "non_router_unaccelerated_non_streamed_estimated_flops": (
            non_streamed_unaccelerated_flops
        ),
        "unaccelerated_backend_matrix_counts": unaccelerated_backend_matrix_counts,
        "unaccelerated_backend_estimated_flops": (
            unaccelerated_backend_estimated_flops
        ),
        "accelerated_router_gate_flop_share": accelerated_router_gate_flop_share,
        "accelerated_router_gate_only": accelerated_router_gate_only,
        "allow_router_gate_only_acceleration": (
            allow_router_gate_only_acceleration
        ),
        "mpp_candidate_policy": linear_summary.get("mpp_candidate_policy"),
        "mpp_tensor_ops_candidate_matrix_count": mpp_candidate_count,
        "mpp_tensor_ops_candidate_estimated_flops": mpp_candidate_estimated_flops,
        "mpp_tensor_ops_candidate_flop_fraction": mpp_candidate_flop_fraction,
        "mpp_tensor_ops_candidate_backend_counts": mpp_candidate_backend_counts,
        "mpp_tensor_ops_candidate_backend_flops": mpp_candidate_backend_flops,
        "streamed_routed_expert_layer_count": streamed_layer_count,
        "streamed_routed_expert_matrix_count": streamed_matrix_count,
        "streamed_routed_expert_assignments": streamed_assignments,
        "streamed_routed_expert_estimated_flops": streamed_estimated_flops,
        "streamed_routed_expert_mpp_candidate_matrix_count": streamed_mpp_count,
        "streamed_routed_expert_mpp_candidate_estimated_flops": streamed_mpp_flops,
        "accelerated_flop_fraction": accelerated_flop_fraction,
        "dominant_resident_flops_accelerated": (
            total_estimated_flops > 0
            and accelerated_estimated_flops * 2 >= total_estimated_flops
        ),
        "accelerated_backends": accelerated_backends,
        "any_resident_matrix_accelerated": any_accelerated,
        "all_resident_matrices_accelerated": all_resident_accelerated,
        "f32_backend_has_unsupported_matrices": (
            f32_backend_has_unsupported_matrices
        ),
        "reason": reason,
    }


def _prefill_acceleration_frontier_request_summary(
    *,
    resident_layout_path: Path,
    configured_backend: str,
    effective_backend: str,
    prompt_token_count: int,
    prompt_chunk_tokens: int | None,
    max_safe_chunk_tokens: int | None,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
    require_prefill_acceleration: bool = False,
    min_accelerated_flop_fraction: float = 0.0,
    streamed_routed_expert_hidden_dim: int = 0,
    streamed_routed_expert_top_k: int = 0,
    streamed_routed_expert_moe_hidden_dims: tuple[int, ...] = (),
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analyzed": False,
        "prompt_token_count": prompt_token_count,
        "resolved_prompt_chunk_tokens": prompt_chunk_tokens,
        "max_safe_prompt_chunk_tokens": max_safe_chunk_tokens,
        "configured_backend": configured_backend,
        "effective_backend": effective_backend,
        "auto_policy": {
            "mpsgraph_min_batch_tokens": mpsgraph_min_batch_tokens,
            "mpsgraph_min_matrix_dim": mpsgraph_min_matrix_dim,
        },
        "require_prefill_acceleration": require_prefill_acceleration,
        "min_accelerated_flop_fraction": min_accelerated_flop_fraction,
    }
    if prompt_chunk_tokens is None or prompt_chunk_tokens <= 0:
        summary["reason"] = "batch prompt prefill disabled"
        return summary

    candidate_values = {
        1,
        int(prompt_chunk_tokens),
        int(prompt_token_count),
        int(mpsgraph_min_batch_tokens),
    }
    if max_safe_chunk_tokens is not None:
        candidate_values.add(int(max_safe_chunk_tokens))
    candidate_chunks = tuple(sorted(value for value in candidate_values if value > 0))

    candidates: list[dict[str, Any]] = []
    minimum_accelerated: int | None = None
    for chunk_tokens in candidate_chunks:
        linear = _prefill_linear_backend_request_summary(
            resident_layout_path=resident_layout_path,
            configured_backend=configured_backend,
            effective_backend=effective_backend,
            prompt_chunk_tokens=chunk_tokens,
            mpsgraph_min_batch_tokens=mpsgraph_min_batch_tokens,
            mpsgraph_min_matrix_dim=mpsgraph_min_matrix_dim,
            streamed_routed_expert_hidden_dim=streamed_routed_expert_hidden_dim,
            streamed_routed_expert_top_k=streamed_routed_expert_top_k,
            streamed_routed_expert_moe_hidden_dims=(
                streamed_routed_expert_moe_hidden_dims
            ),
        )
        coverage = _prefill_acceleration_coverage_summary(
            linear,
            required=False,
            min_accelerated_flop_fraction=min_accelerated_flop_fraction,
            allow_router_gate_only_acceleration=True,
        )
        exceeds_prompt = chunk_tokens > prompt_token_count
        exceeds_safe = (
            max_safe_chunk_tokens is not None
            and chunk_tokens > max_safe_chunk_tokens
        )
        viable = not exceeds_prompt and not exceeds_safe
        accelerated = coverage.get("any_resident_matrix_accelerated") is True
        meets_fraction = (
            float(coverage.get("accelerated_flop_fraction") or 0.0)
            >= min_accelerated_flop_fraction
        )
        if viable and accelerated and meets_fraction and minimum_accelerated is None:
            minimum_accelerated = chunk_tokens
        candidates.append(
            {
                "prompt_chunk_tokens": chunk_tokens,
                "is_resolved": chunk_tokens == prompt_chunk_tokens,
                "is_max_safe": chunk_tokens == max_safe_chunk_tokens,
                "is_auto_mpsgraph_threshold": (
                    chunk_tokens == mpsgraph_min_batch_tokens
                ),
                "viable_for_request": viable,
                "exceeds_prompt_tokens": exceeds_prompt,
                "exceeds_max_safe_prompt_chunk_tokens": exceeds_safe,
                "matrix_count": coverage["matrix_count"],
                "accelerated_matrix_count": coverage["accelerated_matrix_count"],
                "mpsgraph_matrix_count": coverage["mpsgraph_matrix_count"],
                "custom_metal_matrix_count": coverage["custom_metal_matrix_count"],
                "unsupported_mpsgraph_matrix_count": (
                    coverage["unsupported_mpsgraph_matrix_count"]
                ),
                "other_matrix_count": coverage.get("other_matrix_count", 0),
                "total_estimated_flops": coverage["total_estimated_flops"],
                "accelerated_estimated_flops": (
                    coverage["accelerated_estimated_flops"]
                ),
                "custom_metal_estimated_flops": (
                    coverage["custom_metal_estimated_flops"]
                ),
                "unsupported_mpsgraph_estimated_flops": (
                    coverage["unsupported_mpsgraph_estimated_flops"]
                ),
                "router_gate_matrix_count": coverage.get(
                    "router_gate_matrix_count",
                    0,
                ),
                "router_gate_estimated_flops": coverage.get(
                    "router_gate_estimated_flops",
                    0,
                ),
                "router_gate_accelerated_matrix_count": coverage.get(
                    "router_gate_accelerated_matrix_count",
                    0,
                ),
                "router_gate_accelerated_estimated_flops": coverage.get(
                    "router_gate_accelerated_estimated_flops",
                    0,
                ),
                "non_router_accelerated_matrix_count": coverage.get(
                    "non_router_accelerated_matrix_count",
                    0,
                ),
                "non_router_accelerated_estimated_flops": coverage.get(
                    "non_router_accelerated_estimated_flops",
                    0,
                ),
                "non_router_matrix_count": coverage.get(
                    "non_router_matrix_count",
                    0,
                ),
                "non_router_estimated_flops": coverage.get(
                    "non_router_estimated_flops",
                    0,
                ),
                "non_router_unaccelerated_matrix_count": coverage.get(
                    "non_router_unaccelerated_matrix_count",
                    0,
                ),
                "non_router_unaccelerated_estimated_flops": coverage.get(
                    "non_router_unaccelerated_estimated_flops",
                    0,
                ),
                "non_router_unaccelerated_flop_fraction": coverage.get(
                    "non_router_unaccelerated_flop_fraction",
                    0.0,
                ),
                "non_router_unaccelerated_streamed_routed_expert_matrix_count": (
                    coverage.get(
                        "non_router_unaccelerated_streamed_routed_expert_matrix_count",
                        0,
                    )
                ),
                "non_router_unaccelerated_streamed_routed_expert_estimated_flops": (
                    coverage.get(
                        "non_router_unaccelerated_streamed_routed_expert_estimated_flops",
                        0,
                    )
                ),
                "non_router_unaccelerated_non_streamed_matrix_count": coverage.get(
                    "non_router_unaccelerated_non_streamed_matrix_count",
                    0,
                ),
                "non_router_unaccelerated_non_streamed_estimated_flops": coverage.get(
                    "non_router_unaccelerated_non_streamed_estimated_flops",
                    0,
                ),
                "unaccelerated_backend_matrix_counts": coverage.get(
                    "unaccelerated_backend_matrix_counts",
                    {},
                ),
                "unaccelerated_backend_estimated_flops": coverage.get(
                    "unaccelerated_backend_estimated_flops",
                    {},
                ),
                "accelerated_router_gate_flop_share": coverage.get(
                    "accelerated_router_gate_flop_share",
                    0.0,
                ),
                "accelerated_router_gate_only": coverage.get(
                    "accelerated_router_gate_only",
                    False,
                ),
                "mpp_tensor_ops_candidate_matrix_count": linear.get(
                    "mpp_tensor_ops_candidate_matrix_count",
                    0,
                ),
                "mpp_tensor_ops_candidate_estimated_flops": linear.get(
                    "mpp_tensor_ops_candidate_estimated_flops",
                    0,
                ),
                "mpp_tensor_ops_candidate_flop_fraction": linear.get(
                    "mpp_tensor_ops_candidate_flop_fraction",
                    0.0,
                ),
                "mpp_tensor_ops_candidate_backend_counts": linear.get(
                    "mpp_tensor_ops_candidate_backend_counts",
                    {},
                ),
                "mpp_tensor_ops_candidate_backend_flops": linear.get(
                    "mpp_tensor_ops_candidate_backend_flops",
                    {},
                ),
                "streamed_routed_expert_layer_count": linear.get(
                    "streamed_routed_expert_layer_count",
                    0,
                ),
                "streamed_routed_expert_matrix_count": linear.get(
                    "streamed_routed_expert_matrix_count",
                    0,
                ),
                "streamed_routed_expert_assignments": linear.get(
                    "streamed_routed_expert_assignments",
                    0,
                ),
                "streamed_routed_expert_estimated_flops": linear.get(
                    "streamed_routed_expert_estimated_flops",
                    0,
                ),
                "accelerated_flop_fraction": coverage["accelerated_flop_fraction"],
                "dominant_resident_flops_accelerated": (
                    coverage["dominant_resident_flops_accelerated"]
                ),
                "any_resident_matrix_accelerated": accelerated,
                "all_resident_matrices_accelerated": (
                    coverage["all_resident_matrices_accelerated"]
                ),
                "max_matrix_scratch_bytes": linear.get("max_matrix_scratch_bytes", 0),
                "total_matrix_scratch_bytes": linear.get(
                    "total_matrix_scratch_bytes",
                    0,
                ),
                "total_matrix_raw_conversion_bytes": linear.get(
                    "total_matrix_raw_conversion_bytes",
                    0,
                ),
            }
        )

    if minimum_accelerated is not None:
        reason = ""
        suggested_argv: list[str] = [
            "--prefill-prompt-chunk-tokens",
            str(minimum_accelerated),
        ]
        suggested: dict[str, Any] = {
            "prefill_prompt_chunk_tokens": minimum_accelerated,
        }
        if require_prefill_acceleration:
            suggested["require_prefill_acceleration"] = True
            suggested_argv.append("--require-prefill-acceleration")
        if min_accelerated_flop_fraction > 0.0:
            suggested["prefill_min_accelerated_flop_fraction"] = (
                min_accelerated_flop_fraction
            )
            suggested_argv.extend(
                [
                    "--prefill-min-accelerated-flop-fraction",
                    format_routed_read_guard_flag_float(
                        min_accelerated_flop_fraction
                    ),
                ]
            )
        suggested["argv"] = tuple(suggested_argv)
    elif effective_backend == "custom-metal":
        reason = "effective prefill backend is custom-metal"
        suggested = None
    elif all(item["matrix_count"] <= 0 for item in candidates):
        reason = "no resident prefill matrices were found"
        suggested = None
    elif (
        effective_backend in {"auto", PREFILL_LINEAR_AUTO_MPP_BACKEND}
        and mpsgraph_min_batch_tokens > prompt_token_count
    ):
        reason = "prompt token count is below the MPSGraph auto threshold"
        suggested = None
    elif (
        effective_backend in {"auto", PREFILL_LINEAR_AUTO_MPP_BACKEND}
        and max_safe_chunk_tokens is not None
        and mpsgraph_min_batch_tokens > max_safe_chunk_tokens
    ):
        reason = "safety-capped prompt chunk maximum is below the MPSGraph auto threshold"
        suggested = None
    else:
        reason = "no viable prompt chunk resolves resident matrices to MPSGraph"
        suggested = None

    summary.update(
        {
            "analyzed": True,
            "minimum_accelerated_prompt_chunk_tokens": minimum_accelerated,
            "suggested_guard_flags": suggested,
            "candidates": tuple(candidates),
            "reason": reason,
        }
    )
    return summary


def _prefill_cache_io_request_summary(
    *,
    cfg: ModelConfig,
    prompt_token_count: int,
    dtype_bytes: int,
) -> dict[str, Any] | None:
    plan = build_prefill_cache_io_plan(
        cfg,
        prompt_tokens=prompt_token_count,
        dtype_bytes=dtype_bytes,
    )
    if plan is None:
        return None
    return asdict(plan)


def _prefill_cache_io_guard_flag_suggestions(
    *,
    max_cache_read_mib: float,
    prefill_max_cache_write_mib: float,
    source: str,
) -> dict[str, Any] | None:
    argv: list[str] = []
    payload: dict[str, Any] = {"source": source}
    if max_cache_read_mib != PreparedServerConfig.max_cache_read_mib:
        value = format_routed_read_guard_flag_float(max_cache_read_mib)
        argv.extend(["--max-cache-read-mib", value])
        payload["max_cache_read_mib"] = float(max_cache_read_mib)
    if prefill_max_cache_write_mib != PreparedServerConfig.prefill_max_cache_write_mib:
        value = format_routed_read_guard_flag_float(prefill_max_cache_write_mib)
        argv.extend(["--prefill-max-cache-write-mib", value])
        payload["prefill_max_cache_write_mib"] = float(prefill_max_cache_write_mib)
    if not argv:
        return None
    payload["argv"] = tuple(argv)
    return payload


def _prefill_routed_read_request_summary(
    *,
    expert_layout_path: Path,
    prompt_token_count: int,
    prompt_chunk_tokens: int | None,
    top_k: int,
    layers: set[int] | None,
    max_read_amplification: float,
    max_planned_read_bytes: int,
    ssd_read_gib_per_second: float,
    max_read_seconds: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analyzed": False,
        "prompt_chunk_tokens": prompt_chunk_tokens,
        "top_k": top_k,
        "max_read_amplification": max_read_amplification,
        "max_planned_read_bytes": max_planned_read_bytes,
        "ssd_read_gib_per_second": ssd_read_gib_per_second,
        "max_read_seconds": max_read_seconds,
    }
    if prompt_chunk_tokens is None or prompt_chunk_tokens <= 0:
        summary["reason"] = "batch prompt prefill disabled"
        return summary
    try:
        estimate = estimate_routed_expert_read(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            prompt_chunk_tokens=prompt_chunk_tokens,
            top_k=top_k,
            layers=layers,
        )
    except RoutedExpertReadError as exc:
        raise PreparedServerError(str(exc)) from exc
    baseline_seconds = (
        estimate.baseline_read_bytes / (ssd_read_gib_per_second * 1024**3)
        if ssd_read_gib_per_second > 0
        else None
    )
    planned_seconds = (
        estimate.planned_read_bytes / (ssd_read_gib_per_second * 1024**3)
        if ssd_read_gib_per_second > 0
        else None
    )
    extra_seconds = (
        estimate.extra_read_bytes / (ssd_read_gib_per_second * 1024**3)
        if ssd_read_gib_per_second > 0
        else None
    )
    seconds_limit_bytes = (
        int(max_read_seconds * ssd_read_gib_per_second * 1024**3)
        if max_read_seconds > 0 and ssd_read_gib_per_second > 0
        else 0
    )
    effective_planned_read_limit = (
        min(
            cap
            for cap in (max_planned_read_bytes, seconds_limit_bytes)
            if cap > 0
        )
        if max_planned_read_bytes > 0 or seconds_limit_bytes > 0
        else 0
    )
    minimum_chunk = None
    if max_read_amplification > 0 or effective_planned_read_limit > 0:
        try:
            minimum_chunk = minimum_prompt_chunk_tokens_for_routed_read_limits(
                expert_layout_path=expert_layout_path,
                prompt_token_count=prompt_token_count,
                top_k=top_k,
                layers=layers,
                max_read_amplification=max_read_amplification,
                max_planned_read_bytes=effective_planned_read_limit,
            )
        except RoutedExpertReadError as exc:
            raise PreparedServerError(str(exc)) from exc
    summary.update(
        {
            "analyzed": True,
            "layers": estimate.layers,
            "chunks_per_prompt": estimate.chunks_per_prompt,
            "baseline_read_bytes": estimate.baseline_read_bytes,
            "planned_read_bytes": estimate.planned_read_bytes,
            "extra_read_bytes": estimate.extra_read_bytes,
            "baseline_read_seconds": baseline_seconds,
            "planned_read_seconds": planned_seconds,
            "extra_read_seconds": extra_seconds,
            "seconds_limit_planned_read_bytes": seconds_limit_bytes,
            "effective_planned_read_limit_bytes": effective_planned_read_limit,
            "minimum_chunk_tokens_for_limits": minimum_chunk,
            "read_amplification": estimate.read_amplification,
            "max_layer_baseline_read_bytes": estimate.max_layer_baseline_read_bytes,
            "max_layer_planned_read_bytes": estimate.max_layer_planned_read_bytes,
            "within_amplification_limit": (
                True
                if max_read_amplification <= 0
                else estimate.read_amplification <= max_read_amplification
            ),
            "within_planned_read_limit": (
                True
                if max_planned_read_bytes <= 0
                else estimate.planned_read_bytes <= max_planned_read_bytes
            ),
            "within_seconds_limit": (
                True
                if max_read_seconds <= 0
                else (
                    planned_seconds is not None
                    and planned_seconds <= max_read_seconds
                )
            ),
            "within_limit": (
                (
                    max_read_amplification <= 0
                    or estimate.read_amplification <= max_read_amplification
                )
                and (
                    max_planned_read_bytes <= 0
                    or estimate.planned_read_bytes <= max_planned_read_bytes
                )
                and (
                    max_read_seconds <= 0
                    or (
                        planned_seconds is not None
                        and planned_seconds <= max_read_seconds
                    )
                )
            ),
        }
    )
    return summary


def _streamed_routed_expert_request_dims(
    cfg: ModelConfig,
    routed_read: dict[str, Any] | None,
) -> tuple[int, tuple[int, ...]]:
    if not isinstance(routed_read, dict) or routed_read.get("analyzed") is not True:
        return 0, ()
    try:
        layer_count = int(routed_read.get("layers") or 0)
        hidden_dim = int(cfg.hidden_size)
        moe_hidden_dim = int(cfg.moe_hidden_size)
    except (ConfigError, TypeError, ValueError):
        return 0, ()
    if layer_count <= 0 or hidden_dim <= 0 or moe_hidden_dim <= 0:
        return 0, ()
    return hidden_dim, tuple(moe_hidden_dim for _ in range(layer_count))


def _prefill_routed_stage_temp_request_summary(
    *,
    expert_layout_path: Path,
    prompt_token_count: int,
    prompt_chunk_tokens: int | None,
    top_k: int,
    layers: set[int] | None,
    stage_align_bytes: int,
    static_capacity_per_expert: object | None,
    allow_static_capacity_overflow: bool,
    max_stage_bytes: int,
    max_compact_stage_bytes: int,
    max_stage_raw_ranges: int = 0,
    max_stage_coalesced_ranges: int = 0,
    expert_stage_tiling: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "analyzed": False,
        "prompt_chunk_tokens": prompt_chunk_tokens,
        "top_k": top_k,
        "expert_stage_tiling": bool(expert_stage_tiling),
        "stage_align_bytes": stage_align_bytes,
        "max_stage_limit_bytes": max_stage_bytes,
        "max_compact_stage_limit_bytes": max_compact_stage_bytes,
        "max_stage_raw_range_limit": max_stage_raw_ranges,
        "max_stage_coalesced_range_limit": max_stage_coalesced_ranges,
    }
    if prompt_chunk_tokens is None or prompt_chunk_tokens <= 0:
        summary["reason"] = "batch prompt prefill disabled"
        return summary
    try:
        estimate = estimate_routed_stage_temp(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            prompt_chunk_tokens=prompt_chunk_tokens,
            top_k=top_k,
            stage_align_bytes=stage_align_bytes,
            layers=layers,
            static_capacity_per_expert=static_capacity_per_expert,
            allow_static_capacity_overflow=allow_static_capacity_overflow,
        )
    except RoutedExpertReadError as exc:
        raise PreparedServerError(str(exc)) from exc
    tiling_plan: dict[str, Any] | None = None
    effective_max_stage_bytes = estimate.max_stage_bytes
    effective_max_compact_stage_bytes = estimate.max_compact_stage_bytes
    effective_max_stage_raw_ranges = estimate.max_stage_raw_ranges
    effective_max_stage_coalesced_ranges = estimate.max_stage_coalesced_ranges
    effective_max_stage_plus_compact = estimate.max_stage_plus_compact_bytes
    effective_max_stage_plus_compact_plus_static = (
        estimate.max_stage_plus_compact_plus_static_bytes
    )
    if expert_stage_tiling:
        try:
            specs = _load_routed_expert_layer_specs(
                expert_layout_path=expert_layout_path,
                layers=layers,
                purpose="routed stage tiling estimate",
            )
        except RoutedExpertReadError as exc:
            raise PreparedServerError(str(exc)) from exc
        remaining = prompt_token_count
        total_tile_count = 0
        layers_requiring_tiling = 0
        max_tile_count_per_layer = 0
        max_tile_experts = 0
        max_tile_stage = 0
        max_tile_compact = 0
        max_target_unique = 0
        blocker: str | None = None
        while remaining > 0:
            current = min(prompt_chunk_tokens, remaining)
            target_assignments = current * top_k
            for spec in specs:
                if spec.num_experts <= 0 or spec.expert_slot_bytes <= 0:
                    continue
                unique = min(spec.num_experts, target_assignments)
                if unique <= 0:
                    continue
                stage_per_expert = spec.expert_slot_bytes + stage_align_bytes
                stage_capacity = (
                    max_stage_bytes // stage_per_expert
                    if max_stage_bytes > 0 and stage_per_expert > 0
                    else unique
                )
                compact_capacity = (
                    max_compact_stage_bytes // spec.expert_slot_bytes
                    if max_compact_stage_bytes > 0
                    else unique
                )
                tile_capacity = min(unique, stage_capacity, compact_capacity)
                max_target_unique = max(max_target_unique, unique)
                if tile_capacity <= 0:
                    blocker = blocker or "expert_stage_tile_capacity"
                    continue
                tile_count = math.ceil(unique / tile_capacity)
                if tile_count > 1:
                    layers_requiring_tiling += 1
                tile_experts = min(unique, tile_capacity)
                total_tile_count += tile_count
                max_tile_count_per_layer = max(max_tile_count_per_layer, tile_count)
                max_tile_experts = max(max_tile_experts, tile_experts)
                max_tile_stage = max(max_tile_stage, tile_experts * stage_per_expert)
                max_tile_compact = max(
                    max_tile_compact,
                    tile_experts * spec.expert_slot_bytes,
                )
            remaining -= current
        tiling_plan = {
            "can_tile_all_layers": blocker is None,
            "blocker": blocker,
            "target_prompt_chunk_tokens": prompt_chunk_tokens,
            "max_target_unique_experts_per_layer": max_target_unique,
            "max_experts_per_stage_tile": max_tile_experts,
            "max_stage_tile_count_per_layer": max_tile_count_per_layer,
            "total_stage_tile_count": total_tile_count,
            "layers_requiring_tiling": layers_requiring_tiling,
            "max_stage_tile_bytes": max_tile_stage,
            "max_compact_stage_tile_bytes": max_tile_compact,
            "max_stage_plus_compact_tile_bytes": max_tile_stage + max_tile_compact,
        }
        if blocker is None:
            effective_max_stage_bytes = max_tile_stage
            effective_max_compact_stage_bytes = max_tile_compact
            effective_max_stage_raw_ranges = max_tile_experts
            effective_max_stage_coalesced_ranges = max_tile_experts
            effective_max_stage_plus_compact = max_tile_stage + max_tile_compact
            effective_max_stage_plus_compact_plus_static = (
                effective_max_stage_plus_compact
                + estimate.max_static_capacity_binary_bytes
            )
    within_stage_limit = (
        True
        if max_stage_bytes <= 0
        else effective_max_stage_bytes <= max_stage_bytes
    )
    within_compact_stage_limit = (
        True
        if max_compact_stage_bytes <= 0
        else effective_max_compact_stage_bytes <= max_compact_stage_bytes
    )
    within_raw_range_limit = (
        True
        if max_stage_raw_ranges <= 0
        else effective_max_stage_raw_ranges <= max_stage_raw_ranges
    )
    within_coalesced_range_limit = (
        True
        if max_stage_coalesced_ranges <= 0
        else effective_max_stage_coalesced_ranges <= max_stage_coalesced_ranges
    )
    summary.update(
        asdict(estimate)
        | {
            "analyzed": True,
            "expert_stage_tiling": bool(expert_stage_tiling),
            "expert_stage_tiling_plan": tiling_plan,
            "effective_max_stage_bytes": effective_max_stage_bytes,
            "effective_max_compact_stage_bytes": effective_max_compact_stage_bytes,
            "effective_max_stage_raw_ranges": effective_max_stage_raw_ranges,
            "effective_max_stage_coalesced_ranges": (
                effective_max_stage_coalesced_ranges
            ),
            "effective_max_stage_plus_compact_bytes": (
                effective_max_stage_plus_compact
            ),
            "effective_max_stage_plus_compact_plus_static_bytes": (
                effective_max_stage_plus_compact_plus_static
            ),
            "max_stage_limit_bytes": max_stage_bytes,
            "max_compact_stage_limit_bytes": max_compact_stage_bytes,
            "max_stage_raw_range_limit": max_stage_raw_ranges,
            "max_stage_coalesced_range_limit": max_stage_coalesced_ranges,
            "within_stage_limit": within_stage_limit,
            "within_compact_stage_limit": within_compact_stage_limit,
            "within_stage_raw_range_limit": within_raw_range_limit,
            "within_stage_coalesced_range_limit": within_coalesced_range_limit,
            "within_limit": (
                within_stage_limit
                and within_compact_stage_limit
                and within_raw_range_limit
                and within_coalesced_range_limit
            ),
        }
    )
    return summary


def _prefill_stage_temp_disk_free_request_summary(
    *,
    stage_temp: dict[str, Any] | None,
    disk_safety_margin_bytes: int,
    temp_dir: Path = Path("/private/tmp"),
) -> dict[str, Any] | None:
    if not isinstance(stage_temp, dict) or not stage_temp.get("analyzed"):
        return None
    required_stage = stage_temp.get("effective_max_stage_plus_compact_plus_static_bytes")
    if type(required_stage) is not int or required_stage <= 0:
        required_stage = stage_temp.get("max_stage_plus_compact_plus_static_bytes")
    if type(required_stage) is not int or required_stage <= 0:
        required_stage = stage_temp.get("effective_max_stage_plus_compact_bytes")
    if type(required_stage) is not int or required_stage <= 0:
        required_stage = stage_temp.get("max_stage_plus_compact_bytes")
    if type(required_stage) is not int or required_stage <= 0:
        return {
            "analyzed": False,
            "reason": "stage temp estimate did not report a positive requirement",
            "path": str(temp_dir),
        }
    margin = max(0, int(disk_safety_margin_bytes))
    required_free = int(required_stage) + margin
    usage = _system_disk_usage(temp_dir)
    free_bytes = usage.get("free_bytes") if isinstance(usage, dict) else None
    within = (
        free_bytes >= required_free
        if isinstance(free_bytes, int)
        else None
    )
    summary: dict[str, Any] = {
        "analyzed": True,
        "path": str(temp_dir),
        "required_stage_temp_bytes": int(required_stage),
        "disk_safety_margin_bytes": margin,
        "required_free_bytes": required_free,
        "within_free_space": within,
    }
    if isinstance(usage, dict):
        summary.update(
            {
                "total_bytes": usage.get("total_bytes"),
                "used_bytes": usage.get("used_bytes"),
                "free_bytes": usage.get("free_bytes"),
            }
        )
    else:
        summary["reason"] = "could not inspect prompt prefill temp disk"
    return summary


def _prefill_routed_chunk_frontier_request_summary(
    *,
    expert_layout_path: Path,
    prompt_token_count: int,
    prompt_chunk_tokens: int | None,
    max_safe_chunk_tokens: int | None,
    top_k: int,
    layers: tuple[int, ...] | None,
    stage_align_bytes: int,
    ssd_read_gib_per_second: float,
    static_capacity_per_expert: object | None,
    allow_static_capacity_overflow: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"analyzed": False}
    if prompt_chunk_tokens is None:
        summary["reason"] = "batch prompt prefill disabled"
        return summary
    include: list[int] = [prompt_chunk_tokens]
    if max_safe_chunk_tokens is not None:
        include.append(max_safe_chunk_tokens)
    try:
        frontier = estimate_routed_prefill_chunk_frontier(
            expert_layout_path=expert_layout_path,
            prompt_token_count=prompt_token_count,
            top_k=top_k,
            stage_align_bytes=stage_align_bytes,
            layers=layers,
            include_chunk_tokens=include,
            ssd_read_gib_per_second=ssd_read_gib_per_second,
            static_capacity_per_expert=static_capacity_per_expert,
            allow_static_capacity_overflow=allow_static_capacity_overflow,
        )
    except RoutedExpertReadError as exc:
        raise PreparedServerError(str(exc)) from exc
    return asdict(frontier) | {
        "analyzed": True,
        "resolved_prompt_chunk_tokens": prompt_chunk_tokens,
        "max_safe_prompt_chunk_tokens": max_safe_chunk_tokens,
    }


def _prefill_routed_guard_flag_suggestions(
    *,
    routed_read: dict[str, Any] | None,
    prompt_chunk_tokens: int | None,
) -> dict[str, Any] | None:
    if (
        not isinstance(routed_read, dict)
        or not routed_read.get("analyzed")
        or prompt_chunk_tokens is None
        or prompt_chunk_tokens <= 0
    ):
        return None
    suggested = suggest_routed_read_guard_flags(
        prompt_chunk_tokens=prompt_chunk_tokens,
        planned_read_bytes=routed_read.get("planned_read_bytes"),
        read_amplification=routed_read.get("read_amplification"),
        ssd_read_gib_per_second=routed_read.get("ssd_read_gib_per_second"),
        planned_read_seconds=routed_read.get("planned_read_seconds"),
        source="prepared_request_check",
    )
    return suggested if isinstance(suggested, dict) else None


def _prefill_routed_stage_guard_flag_suggestions(
    *,
    stage_temp: dict[str, Any] | None,
    prompt_chunk_tokens: int | None,
) -> dict[str, Any] | None:
    if (
        not isinstance(stage_temp, dict)
        or not stage_temp.get("analyzed")
        or prompt_chunk_tokens is None
        or prompt_chunk_tokens <= 0
    ):
        return None
    use_tiling = stage_temp.get("expert_stage_tiling") is True
    suggested = suggest_routed_stage_temp_guard_flags(
        prompt_chunk_tokens=prompt_chunk_tokens,
        max_stage_bytes=(
            stage_temp.get("effective_max_stage_bytes")
            if use_tiling
            else stage_temp.get("max_stage_bytes")
        ),
        max_compact_stage_bytes=(
            stage_temp.get("effective_max_compact_stage_bytes")
            if use_tiling
            else stage_temp.get("max_compact_stage_bytes")
        ),
        max_stage_raw_ranges=(
            stage_temp.get("effective_max_stage_raw_ranges")
            if use_tiling
            else stage_temp.get("max_stage_raw_ranges")
        ),
        max_stage_coalesced_ranges=(
            stage_temp.get("effective_max_stage_coalesced_ranges")
            if use_tiling
            else stage_temp.get("max_stage_coalesced_ranges")
        ),
        max_stage_plus_compact_bytes=stage_temp.get(
            "effective_max_stage_plus_compact_bytes"
            if use_tiling
            else "max_stage_plus_compact_bytes"
        ),
        total_stage_plus_compact_bytes=stage_temp.get(
            "total_stage_plus_compact_bytes"
        ),
        max_static_capacity_binary_bytes=stage_temp.get(
            "max_static_capacity_binary_bytes"
        ),
        total_static_capacity_binary_bytes=stage_temp.get(
            "total_static_capacity_binary_bytes"
        ),
        max_stage_plus_compact_plus_static_bytes=stage_temp.get(
            "max_stage_plus_compact_plus_static_bytes"
        ),
        total_stage_plus_compact_plus_static_bytes=stage_temp.get(
            "total_stage_plus_compact_plus_static_bytes"
        ),
        static_capacity_per_expert=stage_temp.get("static_capacity_per_expert"),
        source="prepared_request_check",
    )
    if not isinstance(suggested, dict):
        return None
    if use_tiling:
        suggested["prefill_expert_stage_tiling"] = True
        argv = suggested.get("argv")
        if isinstance(argv, tuple):
            suggested["argv"] = argv + ("--prefill-expert-stage-tiling",)
        elif isinstance(argv, list):
            suggested["argv"] = tuple(argv + ["--prefill-expert-stage-tiling"])
    return suggested


def _decode_routed_read_request_summary(
    *,
    read_bytes_per_token: int,
    max_read_gib_per_token: float,
    ssd_read_gib_per_second: float,
    max_read_seconds_per_token: float,
) -> dict[str, Any]:
    max_read_bytes = (
        int(max_read_gib_per_token * 1024**3)
        if max_read_gib_per_token > 0
        else 0
    )
    planned_seconds = (
        read_bytes_per_token / (ssd_read_gib_per_second * 1024**3)
        if ssd_read_gib_per_second > 0
        else None
    )
    within_read_limit = (
        True if max_read_bytes <= 0 else read_bytes_per_token <= max_read_bytes
    )
    within_seconds_limit = (
        True
        if max_read_seconds_per_token <= 0
        else (
            planned_seconds is not None
            and planned_seconds <= max_read_seconds_per_token
        )
    )
    return {
        "analyzed": True,
        "read_bytes_per_token": read_bytes_per_token,
        "max_read_bytes_per_token": max_read_bytes,
        "within_read_limit": within_read_limit,
        "ssd_read_gib_per_second": ssd_read_gib_per_second,
        "planned_read_seconds_per_token": planned_seconds,
        "max_read_seconds_per_token": max_read_seconds_per_token,
        "within_seconds_limit": within_seconds_limit,
        "within_limit": within_read_limit and within_seconds_limit,
    }


@dataclass(frozen=True)
class PreparedServerConfig:
    prepared_path: Path
    runner_path: Path
    model_config_path: Path | None = None
    tokenizer_path: Path | None = None
    tokenizer_backend: str = "auto"
    trust_remote_code: bool = False
    require_prepared_memory_profile: bool = False
    require_glm_4bit: bool = False
    require_public_glm_5_2_shape: bool = False
    served_model_name: str = "largerlm-prepared"
    host: str = "127.0.0.1"
    port: int = 8000
    max_new_tokens_cap: int = 256
    max_prompt_tokens: int = 4096
    max_request_bytes: int = 1024 * 1024
    batch_prefill_prompt: bool = True
    max_cache_read_mib: float = 256.0
    max_cache_file_mib: float = 32768.0
    decode_max_routed_read_gib_per_token: float = 0.0
    decode_max_routed_read_seconds_per_token: float = 0.0
    max_runner_scratch_mib: float = 4096.0
    max_live_working_set_mib: float | None = None
    min_free_unified_memory_gib: float | None = None
    expert_read_advise_merge_gap_kib: int = 0
    expert_read_advise_align_kib: int = 0
    prefill_prompt_chunk_tokens: int = 0
    prefill_max_prompt_batch_mib: float = 1024.0
    prefill_max_cache_write_mib: float = 4096.0
    prefill_max_stage_mib: float = 4096.0
    prefill_max_compact_stage_mib: float = 4096.0
    prefill_max_stage_raw_ranges: int = 0
    prefill_max_stage_coalesced_ranges: int = 0
    prefill_expert_stage_tiling: bool = False
    prefill_persistent_moe_plan_server: bool = False
    prefill_persistent_resident_linear_server: bool = False
    prefill_persistent_attention_projection_server: bool = False
    prefill_persistent_attention_output_server: bool = False
    prefill_persistent_shared_expert_server: bool = False
    prefill_persistent_rope_split_server: bool = False
    prefill_persistent_mla_attention_server: bool = False
    prefill_persistent_rmsnorm_server: bool = False
    prefill_copy_chunk_mib: float = 8.0
    prefill_stage_disk_margin_mib: float = 0.0
    prefill_max_routed_read_amplification: float = 0.0
    prefill_max_routed_read_gib: float = 0.0
    prefill_ssd_read_gib_per_second: float = 0.0
    prefill_max_routed_read_seconds: float = 0.0
    prefill_moe_token_block: int | str = "auto"
    prefill_moe_output_accumulator: str = "env"
    prefill_static_capacity_per_expert: object | None = "auto"
    prefill_allow_static_capacity_overflow: bool = False
    prefill_mla_kv_b_cache_dir: Path | None = None
    prefill_mla_key_cache: bool = False
    decode_mla_key_cache: bool = False
    metal_runtime_cache_mla_kv_b_f32: bool = False
    metal_runtime_max_mla_kv_b_cache_mib: float = 0.0
    metal_runtime_expert_pin_plan: Path | None = None
    metal_runtime_max_adaptive_expert_cache_gib: float = 0.0
    metal_runtime_mmap_final_logits: bool = False
    metal_runtime_context1_o_proj_cache_layout: Path | None = None
    metal_runtime_context1_o_proj_cache_file: Path | None = None
    prefill_linear_backend: str = "auto"
    require_prefill_acceleration: bool = False
    allow_router_gate_only_prefill_acceleration: bool = False
    prefill_min_accelerated_flop_fraction: float = 0.0
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM
    prefill_router_hybrid_margin_threshold: float = 0.0
    prefill_compile_mpp_probe: bool = False
    prefill_run_mpp_probe: bool = False
    prefill_run_mpsgraph_probe: bool = False
    prefill_backend_probe_timeout_seconds: float = (
        DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS
    )
    enforce_prefill_acceleration_probe: bool = True
    metal_final_logits: bool = False
    metal_runtime_generation: bool = False
    metal_binary_path: Path = Path("metal/glm_moe_infer")
    logits_top_k_cap: int = 64
    echo_runner_output: bool = False
    allow_missing_dsa_indexer: bool = False
    applied_launch_profile: dict[str, Any] | None = None
    launch_audit_envelope: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedServerState:
    config: PreparedServerConfig
    prepared: PreparedManifest
    model_config: ModelConfig
    decode_cache_context_tokens: int
    model_context_tokens: int | None
    effective_context_tokens: int
    runtime_prefill_linear_backend: str
    metal_runtime_context1_o_proj_cache: dict[str, Any] | None
    lock: threading.Lock


def _config_eos_token_ids(cfg: ModelConfig) -> tuple[int, ...]:
    return tuple(cfg.eos_token_ids)


def _dense_layers(cfg: ModelConfig) -> tuple[int, ...]:
    if cfg.mlp_layer_types is None:
        return ()
    return tuple(
        layer
        for layer, kind in enumerate(cfg.mlp_layer_types)
        if str(kind).lower() not in {"sparse", "moe", "moe_sparse"}
    )


def _router_score(cfg: ModelConfig) -> str:
    score = cfg.scoring_func
    return str(score) if score in {"sigmoid", "softmax", "raw"} else "sigmoid"


def _dsa_kwargs(cfg: ModelConfig) -> dict[str, Any]:
    return {
        "dsa_indexer_types": cfg.indexer_types,
        "dsa_index_topk": cfg.index_topk,
        "dsa_index_n_heads": cfg.index_n_heads,
        "dsa_index_head_dim": cfg.index_head_dim,
        "dsa_qk_rope_dim": cfg.qk_rope_head_dim,
        "dsa_rope_interleave": cfg.indexer_rope_interleave,
        "dsa_layer_norm_eps": 1e-6,
    }


def _acquire_server_prepared_generation_lock(prepared: PreparedManifest):
    try:
        return acquire_prepared_run_lock_path(
            prepared_run_lock_path_for_manifest(prepared.manifest_path),
            busy_message=(
                "another prepared generation is already running for this "
                "prepared package"
            ),
        )
    except PreparedRunLockError as exc:
        raise PreparedServerError(str(exc)) from exc


def _auto_prefill_prompt_chunk_plan_summary(
    plan: object | None,
) -> dict[str, Any] | None:
    if plan is None:
        return None
    payload = asdict(plan)
    limiting_caps = payload.get("limiting_caps")
    if isinstance(limiting_caps, (list, tuple)):
        payload["limiting_cap_names"] = tuple(
            cap.get("name")
            for cap in limiting_caps
            if isinstance(cap, dict) and isinstance(cap.get("name"), str)
        )
    return payload


_PREFILL_CHUNK_PLAN_ADMISSION_FAILURES = frozenset(
    (
        "missing_profile_max_safe_plan",
        "missing_actual_max_safe_plan",
        "missing_actual_prompt_chunk",
        "selected_chunk_exceeds_current_max_safe",
        "current_max_safe_below_profile",
    )
)


def _require_prefill_chunk_plan_profile_admission(
    drift: dict[str, object] | None,
) -> None:
    if not isinstance(drift, dict):
        return
    status = drift.get("status")
    if status not in _PREFILL_CHUNK_PLAN_ADMISSION_FAILURES:
        return
    raise PreparedServerError(
        "prefill chunk-plan drift "
        f"{status}: "
        f"profile_max_safe={drift.get('profile_max_safe_chunk_tokens')} "
        f"current_max_safe={drift.get('actual_max_safe_chunk_tokens')} "
        f"selected={drift.get('actual_prompt_chunk_tokens')}"
    )


def _bool_payload(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise PreparedServerError(f"{key} must be a boolean")


def _int_payload(
    payload: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = payload.get(key, default)
    if value is None:
        raise PreparedServerError(f"{key} is required")
    if type(value) is not int:
        raise PreparedServerError(f"{key} must be an integer")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise PreparedServerError(f"{key} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise PreparedServerError(f"{key} must be <= {maximum}")
    return parsed


def _float_payload(
    payload: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise PreparedServerError(f"{key} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PreparedServerError(f"{key} must be numeric") from exc
    if not math.isfinite(parsed):
        raise PreparedServerError(f"{key} must be finite")
    if minimum is not None and parsed < minimum:
        raise PreparedServerError(f"{key} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise PreparedServerError(f"{key} must be <= {maximum}")
    return parsed


def _batch_prefill_requested(
    server: PreparedServerConfig,
    payload: dict[str, Any],
) -> bool:
    return server.batch_prefill_prompt and _bool_payload(
        payload,
        "batch_prefill_prompt",
        True,
    )


def _prompt_token_ids(payload: dict[str, Any], *, max_prompt_tokens: int) -> tuple[int, ...]:
    raw = payload.get("prompt_token_ids")
    if not isinstance(raw, list):
        raise PreparedServerError("prompt_token_ids must be a JSON array")
    if not raw:
        raise PreparedServerError("prompt_token_ids must be non-empty")
    if len(raw) > max_prompt_tokens:
        raise PreparedServerError(
            f"prompt_token_ids length {len(raw)} exceeds server cap {max_prompt_tokens}"
        )
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise PreparedServerError("prompt_token_ids must contain integers")
        if item < 0:
            raise PreparedServerError("prompt_token_ids must be non-negative")
        out.append(item)
    return tuple(out)


def _request_prompt_token_cap(
    state: PreparedServerState,
    *,
    max_new_tokens: int,
) -> int:
    remaining = state.effective_context_tokens - max_new_tokens
    if remaining <= 0:
        raise PreparedServerError(
            f"max_new_tokens {max_new_tokens} leaves no room for a prompt within "
            f"context limit {state.effective_context_tokens}"
        )
    return min(state.config.max_prompt_tokens, remaining)


def _launch_audit_envelope_int(
    envelope: dict[str, Any],
    field: str,
    *,
    minimum: int,
) -> int:
    value = envelope.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreparedServerError(
            "launch audit envelope is malformed: "
            f"{field}={value!r}"
        )
    return int(value)


def _launch_audit_envelope_request_summary(
    envelope: dict[str, Any] | None,
    *,
    prompt_token_count: int,
    max_new_tokens: int,
) -> dict[str, Any] | None:
    if envelope is None:
        return None
    if not isinstance(envelope, dict):
        raise PreparedServerError(
            f"launch audit envelope must be an object, got {type(envelope).__name__}"
        )
    schema = envelope.get("schema")
    if schema != "largerlm.launch_audit_server_envelope.v1":
        raise PreparedServerError(
            "launch audit envelope schema is invalid: "
            f"{schema!r}"
        )
    audited_prompt = _launch_audit_envelope_int(
        envelope,
        "audited_prompt_token_count",
        minimum=1,
    )
    audited_new = _launch_audit_envelope_int(
        envelope,
        "audited_max_new_tokens",
        minimum=0,
    )
    audited_context = envelope.get("audited_required_context_tokens")
    if audited_context is not None:
        if (
            isinstance(audited_context, bool)
            or not isinstance(audited_context, int)
            or audited_context != audited_prompt + audited_new
        ):
            raise PreparedServerError(
                "launch audit envelope is malformed: "
                f"audited_required_context_tokens={audited_context!r}"
            )
    prompt_within = prompt_token_count <= audited_prompt
    new_within = max_new_tokens <= audited_new
    summary = {
        "schema": schema,
        "artifact_path": envelope.get("artifact_path"),
        "applied_launch_profile_sha256": (
            envelope.get("applied_launch_profile_sha256")
        ),
        "audited_prompt_token_count": audited_prompt,
        "audited_max_new_tokens": audited_new,
        "audited_required_context_tokens": audited_prompt + audited_new,
        "request_prompt_token_count": prompt_token_count,
        "request_max_new_tokens": max_new_tokens,
        "request_required_context_tokens": prompt_token_count + max_new_tokens,
        "prompt_within_envelope": prompt_within,
        "max_new_within_envelope": new_within,
        "within_envelope": prompt_within and new_within,
    }
    if not prompt_within:
        raise PreparedServerError(
            "request exceeds launch audit prompt envelope: "
            f"prompt_tokens={prompt_token_count} "
            f"audited_prompt_tokens={audited_prompt}"
        )
    if not new_within:
        raise PreparedServerError(
            "request exceeds launch audit generation envelope: "
            f"max_new_tokens={max_new_tokens} "
            f"audited_max_new_tokens={audited_new}"
        )
    return summary


def _base_generation_kwargs(
    state: PreparedServerState,
    *,
    payload: dict[str, Any],
    prompt_token_count: int,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = state.model_config
    server = state.config
    top_k = cfg.experts_per_token
    batch_prefill_prompt = (
        _batch_prefill_requested(server, payload) and prompt_token_count > 1
    )
    logits_top_k = _int_payload(
        payload,
        "logits_top_k",
        default=1,
        minimum=1,
        maximum=server.logits_top_k_cap,
    )
    sampling_top_p = _float_payload(
        payload,
        "top_p",
        default=1.0,
        minimum=0.0,
        maximum=1.0,
    )
    if sampling_top_p <= 0.0:
        raise PreparedServerError("top_p must be > 0.0")
    kwargs = {
        "layers": None,
        "dense_layers": _dense_layers(cfg),
        "work_dir": None,
        "keep_work_dir": False,
        "num_heads": cfg.num_attention_heads,
        "qk_nope_dim": cfg.qk_nope_head_dim,
        "rope_dim": cfg.qk_rope_head_dim,
        "v_head_dim": cfg.v_head_dim,
        "kv_lora_dim": cfg.kv_lora_rank,
        "cache_position_offset": 0,
        "attention_scale": None,
        "rope_theta": cfg.rope_theta or 10000.0,
        "rope_interleave": cfg.rope_interleave,
        "top_k": top_k,
        "max_k": max(8, int(top_k)),
        "router_score": _router_score(cfg),
        "routed_scaling_factor": cfg.routed_scaling_factor,
        "norm_topk_prob": bool(cfg.norm_topk_prob),
        "no_norm_topk_prob": False,
        "router_n_group": cfg.n_group,
        "router_topk_group": cfg.topk_group,
        "ignore_router_bias": False,
        "include_shared_expert": (cfg.n_shared_experts or 0) > 0,
        "rms_norm_eps": cfg.rms_norm_eps if cfg.rms_norm_eps is not None else 1e-5,
        "logits_top_k": logits_top_k,
        "logits_chunk_rows": None,
        "logits_max_chunk_mib": 64.0,
        "max_slot_mib": 256.0,
        "max_router_mib": 64.0,
        "max_resident_matrix_mib": 512.0,
        "max_cache_file_mib": server.max_cache_file_mib,
        "decode_max_routed_read_gib_per_token": (
            server.decode_max_routed_read_gib_per_token
        ),
        "decode_max_routed_read_seconds_per_token": (
            server.decode_max_routed_read_seconds_per_token
        ),
        "max_cache_read_mib": server.max_cache_read_mib,
        "max_runner_scratch_mib": server.max_runner_scratch_mib,
        "max_live_working_set_mib": server.max_live_working_set_mib,
        "min_free_unified_memory_gib": server.min_free_unified_memory_gib,
        "expert_read_advise_merge_gap_kib": server.expert_read_advise_merge_gap_kib,
        "expert_read_advise_align_kib": server.expert_read_advise_align_kib,
        "cache_dtype_bytes": 2,
        "batch_prefill_prompt": batch_prefill_prompt,
        "prefill_prompt_chunk_tokens": server.prefill_prompt_chunk_tokens,
        "prefill_max_prompt_batch_mib": server.prefill_max_prompt_batch_mib,
        "prefill_max_cache_write_mib": server.prefill_max_cache_write_mib,
        "prefill_expert_stage_merge_gap_kib": 0.0,
        "prefill_expert_stage_align_kib": 4.0,
        "prefill_max_stage_mib": server.prefill_max_stage_mib,
        "prefill_max_compact_stage_mib": server.prefill_max_compact_stage_mib,
        "prefill_max_stage_raw_ranges": server.prefill_max_stage_raw_ranges,
        "prefill_max_stage_coalesced_ranges": (
            server.prefill_max_stage_coalesced_ranges
        ),
        "prefill_expert_stage_tiling": server.prefill_expert_stage_tiling,
        "prefill_persistent_moe_plan_server": (
            server.prefill_persistent_moe_plan_server
        ),
        "prefill_persistent_resident_linear_server": (
            server.prefill_persistent_resident_linear_server
        ),
        "prefill_persistent_attention_projection_server": (
            server.prefill_persistent_attention_projection_server
        ),
        "prefill_persistent_attention_output_server": (
            server.prefill_persistent_attention_output_server
        ),
        "prefill_persistent_shared_expert_server": (
            server.prefill_persistent_shared_expert_server
        ),
        "prefill_persistent_rope_split_server": (
            server.prefill_persistent_rope_split_server
        ),
        "prefill_persistent_mla_attention_server": (
            server.prefill_persistent_mla_attention_server
        ),
        "prefill_persistent_rmsnorm_server": (
            server.prefill_persistent_rmsnorm_server
        ),
        "prefill_copy_chunk_mib": server.prefill_copy_chunk_mib,
        "prefill_stage_disk_margin_mib": server.prefill_stage_disk_margin_mib,
        "prefill_max_routed_read_amplification": (
            server.prefill_max_routed_read_amplification
        ),
        "prefill_max_routed_read_gib": server.prefill_max_routed_read_gib,
        "prefill_ssd_read_gib_per_second": (
            server.prefill_ssd_read_gib_per_second
        ),
        "prefill_max_routed_read_seconds": server.prefill_max_routed_read_seconds,
        "prefill_moe_token_block": server.prefill_moe_token_block,
        "prefill_moe_output_accumulator": server.prefill_moe_output_accumulator,
        "prefill_static_capacity_per_expert": (
            server.prefill_static_capacity_per_expert
            if batch_prefill_prompt
            else None
        ),
        "prefill_allow_static_capacity_overflow": (
            server.prefill_allow_static_capacity_overflow
        ),
        "prefill_mla_kv_b_cache_dir": server.prefill_mla_kv_b_cache_dir,
        "prefill_mla_key_cache": server.prefill_mla_key_cache,
        "decode_mla_key_cache": server.decode_mla_key_cache,
        "prefill_linear_backend": state.runtime_prefill_linear_backend,
        "require_prefill_acceleration": server.require_prefill_acceleration,
        "allow_router_gate_only_prefill_acceleration": (
            server.allow_router_gate_only_prefill_acceleration
        ),
        "prefill_min_accelerated_flop_fraction": (
            server.prefill_min_accelerated_flop_fraction
        ),
        "prefill_mpsgraph_min_batch_tokens": (
            server.prefill_mpsgraph_min_batch_tokens
        ),
        "prefill_mpsgraph_min_matrix_dim": server.prefill_mpsgraph_min_matrix_dim,
        "prefill_router_hybrid_margin_threshold": (
            server.prefill_router_hybrid_margin_threshold
        ),
        "max_embedding_row_mib": 64.0,
        "echo_runner_output": server.echo_runner_output,
        "sampling_temperature": _float_payload(
            payload,
            "temperature",
            default=0.0,
            minimum=0.0,
        ),
        "sampling_top_p": sampling_top_p,
        "sampling_seed": (
            _int_payload(payload, "seed", default=None)
            if payload.get("seed") is not None
            else None
        ),
        "metal_final_logits": _bool_payload(
            payload,
            "metal_final_logits",
            server.metal_final_logits,
        ),
        "allow_tied_embeddings": cfg.tie_word_embeddings is not False,
        "expected_vocab_size": cfg.vocab_size,
        "expected_hidden_size": cfg.hidden_size,
        "preflight_runtime": True,
        "eos_token_ids": _config_eos_token_ids(cfg),
        "allow_missing_dsa_indexer": server.allow_missing_dsa_indexer,
    }
    kwargs.update(_dsa_kwargs(cfg))
    if generation_overrides is not None:
        for key, value in generation_overrides.items():
            if key in kwargs:
                kwargs[key] = value
    return kwargs


def _max_new_tokens(payload: dict[str, Any], cap: int) -> int:
    return _int_payload(
        payload,
        "max_new_tokens",
        minimum=0,
        maximum=cap,
    )


def _openai_max_tokens(payload: dict[str, Any], cap: int) -> int:
    default = min(16, cap)
    if (
        payload.get("max_tokens") is not None
        and payload.get("max_completion_tokens") is not None
        and payload["max_tokens"] != payload["max_completion_tokens"]
    ):
        raise PreparedServerError(
            "max_tokens and max_completion_tokens must match when both are set"
        )
    value = payload.get("max_completion_tokens", payload.get("max_tokens", default))
    return _int_payload(
        {"max_tokens": value},
        "max_tokens",
        default=default,
        minimum=0,
        maximum=cap,
    )


def _openai_model_name(payload: dict[str, Any], *, default: str) -> str:
    model = payload.get("model", default)
    if not isinstance(model, str) or not model:
        raise PreparedServerError("model must be a non-empty string")
    return model


def _reject_openai_completion_unsupported(payload: dict[str, Any]) -> None:
    if _bool_payload(payload, "stream", False):
        raise PreparedServerError("stream=true is not supported")
    if _int_payload(payload, "n", default=1, minimum=1, maximum=1) != 1:
        raise PreparedServerError("n must be 1")
    for key in ("best_of", "logprobs", "suffix"):
        if payload.get(key) is not None:
            raise PreparedServerError(f"{key} is not supported")
    stop = payload.get("stop")
    if stop not in (None, ""):
        if isinstance(stop, list) and len(stop) == 0:
            return
        raise PreparedServerError("stop is not supported; use tokenizer/model EOS tokens")


def _reject_openai_chat_unsupported(payload: dict[str, Any]) -> None:
    _reject_openai_completion_unsupported(payload)
    for key in (
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "response_format",
        "parallel_tool_calls",
        "stream_options",
    ):
        if payload.get(key) is not None:
            raise PreparedServerError(f"{key} is not supported")


def _openai_chat_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        raise PreparedServerError("messages must be a non-empty array")
    out: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PreparedServerError(f"messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise PreparedServerError(f"messages[{index}].role must be a non-empty string")
        if not isinstance(content, str):
            raise PreparedServerError(f"messages[{index}].content must be a string")
        out.append({"role": role, "content": content})
    return out


def _decode_layer_record_payload(record: object) -> dict[str, Any]:
    command = tuple(str(item) for item in (getattr(record, "command", ()) or ()))
    mla_kv_b_cache_dir = None
    if "--mla-kv-b-cache-dir" in command:
        index = command.index("--mla-kv-b-cache-dir")
        if index + 1 < len(command):
            mla_kv_b_cache_dir = command[index + 1]
    return {
        "layer": getattr(record, "layer", None),
        "kind": getattr(record, "kind", None),
        "composed": bool(getattr(record, "composed", False)),
        "expert_read_bytes": int(getattr(record, "expert_read_bytes", 0) or 0),
        "attention_read_bytes": int(
            getattr(record, "attention_read_bytes", 0) or 0
        ),
        "cache_read_bytes": int(getattr(record, "cache_read_bytes", 0) or 0),
        "dsa_index_cache_read_bytes": int(
            getattr(record, "dsa_index_cache_read_bytes", 0) or 0
        ),
        "mla_cache_read_bytes": int(
            getattr(record, "mla_cache_read_bytes", 0) or 0
        ),
        "estimated_peak_bytes": int(
            getattr(record, "estimated_peak_bytes", 0) or 0
        ),
        "router_stage_peak_bytes": int(
            getattr(record, "router_stage_peak_bytes", 0) or 0
        ),
        "moe_stage_peak_bytes": int(
            getattr(record, "moe_stage_peak_bytes", 0) or 0
        ),
        "elapsed_seconds": float(getattr(record, "elapsed_seconds", 0.0) or 0.0),
        "attention_elapsed_seconds": float(
            getattr(record, "attention_elapsed_seconds", 0.0) or 0.0
        ),
        "mlp_elapsed_seconds": float(
            getattr(record, "mlp_elapsed_seconds", 0.0) or 0.0
        ),
        "input_in_memory": bool(getattr(record, "input_in_memory", False)),
        "output_in_memory": bool(getattr(record, "output_in_memory", False)),
        "attention_stage_elapsed_seconds": {
            str(name): float(value)
            for name, value in (
                getattr(record, "attention_stage_elapsed_seconds", {}) or {}
            ).items()
        },
        "mlp_stage_elapsed_seconds": {
            str(name): float(value)
            for name, value in (
                getattr(record, "mlp_stage_elapsed_seconds", {}) or {}
            ).items()
        },
        "mlp_diagnostics": dict(getattr(record, "mlp_diagnostics", {}) or {}),
        "mla_attention_timing_elapsed_seconds": {
            str(name): float(value)
            for name, value in (
                getattr(record, "mla_attention_timing_elapsed_seconds", {}) or {}
            ).items()
        },
        "mla_attention_diagnostics": dict(
            getattr(record, "mla_attention_diagnostics", {}) or {}
        ),
        "dsa_indexer_mode": getattr(record, "dsa_indexer_mode", "none"),
        "dsa_rope_interleave": bool(getattr(record, "dsa_rope_interleave", False)),
        "command_has_mla_kv_b_cache_dir": mla_kv_b_cache_dir is not None,
        "command_mla_kv_b_cache_dir": mla_kv_b_cache_dir,
    }


def _generated_step_payload(step: object) -> dict[str, Any]:
    return {
        "position": getattr(step, "position", None),
        "input_token_id": getattr(step, "input_token_id", None),
        "selected_token_id": getattr(step, "selected_token_id", None),
        "elapsed_seconds": float(getattr(step, "elapsed_seconds", 0.0) or 0.0),
        "embedding_read_bytes": int(getattr(step, "embedding_read_bytes", 0) or 0),
        "expert_read_bytes": int(getattr(step, "expert_read_bytes", 0) or 0),
        "cache_read_bytes": int(getattr(step, "cache_read_bytes", 0) or 0),
        "logits_read_bytes": int(getattr(step, "logits_read_bytes", 0) or 0),
        "logits_elapsed_seconds": float(
            getattr(step, "logits_elapsed_seconds", 0.0) or 0.0
        ),
        "estimated_read_bytes": int(getattr(step, "estimated_read_bytes", 0) or 0),
        "topk": [
            {
                "token_id": getattr(record, "token_id", None),
                "logit": getattr(record, "logit", None),
            }
            for record in getattr(step, "topk", ()) or ()
        ],
        "decode_layers": [
            _decode_layer_record_payload(record)
            for record in getattr(step, "decode_layers", ()) or ()
        ],
    }


def _attach_applied_launch_profile(
    payload: dict[str, Any],
    applied_launch_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    if applied_launch_profile is not None:
        payload["applied_launch_profile"] = applied_launch_profile
    return payload


def _object_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _nonnegative_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _prompt_prefill_mla_cache_summary(
    prompt_prefill: object,
    *,
    enabled_field: str,
    bytes_field: str,
    total_bytes_field: str,
) -> dict[str, Any] | None:
    layer_count = 0
    enabled_layer_count = 0
    disabled_layer_count = 0
    total_bytes = 0
    observed = False
    for chunk in _object_field(prompt_prefill, "chunks") or ():
        for layer in _object_field(chunk, "layers") or ():
            attention = _object_field(layer, "attention")
            if attention is None:
                continue
            enabled = _object_field(attention, enabled_field)
            cache_bytes = _nonnegative_int_or_none(
                _object_field(attention, bytes_field)
            )
            if isinstance(enabled, bool) or cache_bytes is not None:
                observed = True
                layer_count += 1
                if enabled is True:
                    enabled_layer_count += 1
                elif enabled is False:
                    disabled_layer_count += 1
                if cache_bytes is not None:
                    total_bytes += cache_bytes
    if not observed:
        return None
    return {
        "observed": True,
        "layer_count": layer_count,
        "enabled_layer_count": enabled_layer_count,
        "disabled_layer_count": disabled_layer_count,
        "all_layers_enabled": layer_count > 0 and enabled_layer_count == layer_count,
        total_bytes_field: total_bytes,
    }


def _token_result_payload(
    result: TokenGenerationResult,
    *,
    applied_launch_profile: dict[str, Any] | None = None,
    request_check: dict[str, Any] | None = None,
    launch_audit_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_prefill = result.prompt_prefill
    profile = applied_launch_profile or result.applied_launch_profile
    drift = result.prefill_prompt_chunk_plan_drift
    if drift is None:
        drift = prefill_prompt_chunk_plan_drift_summary(
            applied_launch_profile=profile,
            actual_prompt_chunk_tokens=(
                prompt_prefill.chunk_tokens if prompt_prefill is not None else None
            ),
            actual_auto_plan=getattr(result, "auto_prefill_prompt_chunk_plan", None),
            actual_max_safe_plan=getattr(
                result,
                "max_safe_prefill_prompt_chunk_plan",
                None,
            ),
        )
    payload = {
        "prompt_token_ids": list(result.prompt_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "steps": [_generated_step_payload(step) for step in result.steps],
        "elapsed_seconds": result.elapsed_seconds,
        "estimated_read_bytes": result.estimated_read_bytes,
        "estimated_embedding_read_bytes": result.estimated_embedding_read_bytes,
        "estimated_expert_read_bytes": result.estimated_expert_read_bytes,
        "estimated_cache_read_bytes": result.estimated_cache_read_bytes,
        "estimated_logits_read_bytes": result.estimated_logits_read_bytes,
        "prefill_actual_read_time": getattr(result, "prefill_actual_read_time", None),
        "decode_actual_read_time": getattr(result, "decode_actual_read_time", None),
        "prompt_prefill": (
            {
                "elapsed_seconds": getattr(prompt_prefill, "elapsed_seconds", None),
                "chunk_count": prompt_prefill.chunk_count,
                "chunk_tokens": prompt_prefill.chunk_tokens,
                "linear_backend_counts": prompt_prefill.linear_backend_counts,
                "linear_backend_flops": getattr(
                    prompt_prefill,
                    "linear_backend_flops",
                    {},
                ),
                "linear_backend_elapsed_seconds": getattr(
                    prompt_prefill,
                    "linear_backend_elapsed_seconds",
                    {},
                )
                or {},
                "linear_backend_estimated_tflops": getattr(
                    prompt_prefill,
                    "linear_backend_estimated_tflops",
                    {},
                )
                or {},
                "total_linear_estimated_flops": getattr(
                    prompt_prefill,
                    "total_linear_estimated_flops",
                    0,
                ),
                "accelerated_linear_estimated_flops": getattr(
                    prompt_prefill,
                    "accelerated_linear_estimated_flops",
                    0,
                ),
                "custom_linear_estimated_flops": getattr(
                    prompt_prefill,
                    "custom_linear_estimated_flops",
                    0,
                ),
                "unsupported_linear_estimated_flops": getattr(
                    prompt_prefill,
                    "unsupported_linear_estimated_flops",
                    0,
                ),
                "accelerated_linear_flop_fraction": getattr(
                    prompt_prefill,
                    "accelerated_linear_flop_fraction",
                    0.0,
                ),
                "prefill_acceleration_coverage": getattr(
                    prompt_prefill,
                    "prefill_acceleration_coverage",
                    None,
                ),
                "prefill_acceleration_frontier": getattr(
                    prompt_prefill,
                    "prefill_acceleration_frontier",
                    None,
                ),
                "persistent_moe_plan_server": getattr(
                    prompt_prefill,
                    "persistent_moe_plan_server",
                    False,
                ),
                "persistent_resident_linear_server": getattr(
                    prompt_prefill,
                    "persistent_resident_linear_server",
                    False,
                ),
                "persistent_attention_projection_server": getattr(
                    prompt_prefill,
                    "persistent_attention_projection_server",
                    False,
                ),
                "persistent_attention_output_server": getattr(
                    prompt_prefill,
                    "persistent_attention_output_server",
                    False,
                ),
                "persistent_shared_expert_server": getattr(
                    prompt_prefill,
                    "persistent_shared_expert_server",
                    False,
                ),
                "persistent_rope_split_server": getattr(
                    prompt_prefill,
                    "persistent_rope_split_server",
                    False,
                ),
                "persistent_mla_attention_server": getattr(
                    prompt_prefill,
                    "persistent_mla_attention_server",
                    False,
                ),
                "persistent_rmsnorm_server": getattr(
                    prompt_prefill,
                    "persistent_rmsnorm_server",
                    False,
                ),
                "mla_key_cache": _prompt_prefill_mla_cache_summary(
                    prompt_prefill,
                    enabled_field="mla_key_cache",
                    bytes_field="mla_key_cache_bytes",
                    total_bytes_field="total_mla_key_cache_bytes",
                ),
                "mla_value_cache": _prompt_prefill_mla_cache_summary(
                    prompt_prefill,
                    enabled_field="mla_value_cache",
                    bytes_field="mla_value_cache_bytes",
                    total_bytes_field="total_mla_value_cache_bytes",
                ),
                "total_linear_matrix_scratch_bytes": (
                    prompt_prefill.total_linear_matrix_scratch_bytes
                ),
                "max_linear_matrix_scratch_bytes": (
                    prompt_prefill.max_linear_matrix_scratch_bytes
                ),
                "total_linear_matrix_f32_bytes": (
                    prompt_prefill.total_linear_matrix_f32_bytes
                ),
                "total_linear_matrix_raw_conversion_bytes": (
                    prompt_prefill.total_linear_matrix_raw_conversion_bytes
                ),
                "total_staged_bytes": prompt_prefill.total_staged_bytes,
                "total_compact_stage_bytes": (
                    prompt_prefill.total_compact_stage_bytes
                ),
                "total_compact_stage_materialized_bytes": (
                    prompt_prefill.total_compact_stage_materialized_bytes
                ),
                "max_staged_bytes": prompt_prefill.max_staged_bytes,
                "max_compact_stage_bytes": (
                    prompt_prefill.max_compact_stage_bytes
                ),
                "max_compact_stage_materialized_bytes": (
                    prompt_prefill.max_compact_stage_materialized_bytes
                ),
                "total_stage_plus_compact_bytes": (
                    prompt_prefill.total_stage_plus_compact_bytes
                ),
                "total_stage_plus_compact_materialized_bytes": (
                    prompt_prefill.total_stage_plus_compact_materialized_bytes
                ),
                "max_stage_plus_compact_bytes": (
                    prompt_prefill.max_stage_plus_compact_bytes
                ),
                "max_stage_plus_compact_materialized_bytes": (
                    prompt_prefill.max_stage_plus_compact_materialized_bytes
                ),
                "total_expert_stage_planned_read_bytes": (
                    prompt_prefill.total_expert_stage_planned_read_bytes
                ),
                "total_expert_stage_planned_read_seconds": getattr(
                    prompt_prefill,
                    "total_expert_stage_planned_read_seconds",
                    None,
                ),
                "total_expert_stage_copy_elapsed_seconds": getattr(
                    prompt_prefill,
                    "total_expert_stage_copy_elapsed_seconds",
                    None,
                ),
                "total_expert_stage_copy_throughput_gib_per_second": getattr(
                    prompt_prefill,
                    "total_expert_stage_copy_throughput_gib_per_second",
                    None,
                ),
                "prefill_ssd_read_gib_per_second": getattr(
                    prompt_prefill,
                    "prefill_ssd_read_gib_per_second",
                    0.0,
                ),
                "prefill_max_routed_read_seconds": getattr(
                    prompt_prefill,
                    "prefill_max_routed_read_seconds",
                    0.0,
                ),
                "total_expert_stage_read_seconds_ok": getattr(
                    prompt_prefill,
                    "total_expert_stage_read_seconds_ok",
                    None,
                ),
                "total_expert_stage_copy_seconds_ok": getattr(
                    prompt_prefill,
                    "total_expert_stage_copy_seconds_ok",
                    None,
                ),
                "total_expert_stage_unique_requested_bytes": (
                    prompt_prefill.total_expert_stage_unique_requested_bytes
                ),
                "total_expert_stage_waste_bytes": (
                    prompt_prefill.total_expert_stage_waste_bytes
                ),
                "total_expert_stage_unique_read_amplification": (
                    prompt_prefill.total_expert_stage_unique_read_amplification
                ),
                "total_expert_stage_read_advice_attempted_ranges": getattr(
                    prompt_prefill,
                    "total_expert_stage_read_advice_attempted_ranges",
                    0,
                ),
                "total_expert_stage_read_advice_calls": getattr(
                    prompt_prefill,
                    "total_expert_stage_read_advice_calls",
                    0,
                ),
                "total_expert_stage_read_advice_bytes": getattr(
                    prompt_prefill,
                    "total_expert_stage_read_advice_bytes",
                    0,
                ),
                "total_expert_stage_read_advice_failures": getattr(
                    prompt_prefill,
                    "total_expert_stage_read_advice_failures",
                    0,
                ),
                "total_routed_expert_assignments": (
                    prompt_prefill.total_routed_expert_assignments
                ),
                "max_moe_estimated_peak_bytes": (
                    prompt_prefill.max_moe_estimated_peak_bytes
                ),
                "static_capacity_per_expert": prompt_prefill.static_capacity_per_expert,
            }
            if prompt_prefill is not None
            else None
        ),
        "auto_prefill_prompt_chunk_plan": _auto_prefill_prompt_chunk_plan_summary(
            getattr(result, "auto_prefill_prompt_chunk_plan", None)
        ),
        "max_safe_prefill_prompt_chunk_plan": (
            _auto_prefill_prompt_chunk_plan_summary(
                getattr(result, "max_safe_prefill_prompt_chunk_plan", None)
            )
        ),
        "prefill_prompt_chunk_plan_drift": drift,
    }
    if request_check is not None:
        payload["request_check"] = request_check
    if launch_audit_envelope is not None:
        payload["launch_audit_envelope"] = launch_audit_envelope
    return _attach_applied_launch_profile(payload, profile)


def _metal_runtime_sampling_kwargs(payload: dict[str, Any]) -> None:
    temperature = _float_payload(payload, "temperature", default=0.0, minimum=0.0)
    top_p = _float_payload(payload, "top_p", default=1.0, minimum=0.0, maximum=1.0)
    if top_p <= 0.0:
        raise PreparedServerError("top_p must be > 0.0")
    if temperature != 0.0 or top_p != 1.0 or payload.get("seed") is not None:
        raise PreparedServerError(
            "metal_runtime_generation currently supports greedy generation only"
        )


def _metal_generation_live_cap_mib(config: PreparedServerConfig) -> int:
    value = config.max_live_working_set_mib
    if value is None:
        return 768
    return max(0, int(math.ceil(float(value))))


def _metal_token_result_payload(
    result: MetalTokenGenerationResult,
    *,
    applied_launch_profile: dict[str, Any] | None = None,
    request_check: dict[str, Any] | None = None,
    launch_audit_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    steps = []
    for index, decode_elapsed in enumerate(result.decode_elapsed_seconds):
        logits_elapsed = (
            result.final_logits_elapsed_seconds[index]
            if index < len(result.final_logits_elapsed_seconds)
            else None
        )
        steps.append(
            {
                "step": index,
                "decode_elapsed_seconds": decode_elapsed,
                "final_logits_elapsed_seconds": logits_elapsed,
                "final_logits_bytes_read": (
                    result.final_logits_bytes_read[index]
                    if index < len(result.final_logits_bytes_read)
                    else None
                ),
                "final_logits_lm_head_bytes_read": (
                    result.final_logits_lm_head_bytes_read[index]
                    if index < len(result.final_logits_lm_head_bytes_read)
                    else None
                ),
                "final_logits_read_seconds": (
                    result.final_logits_read_seconds[index]
                    if index < len(result.final_logits_read_seconds)
                    else None
                ),
                "final_logits_kernel_seconds": (
                    result.final_logits_kernel_seconds[index]
                    if index < len(result.final_logits_kernel_seconds)
                    else None
                ),
                "final_logits_resident_mmap_backed": (
                    result.final_logits_resident_mmap_backed[index]
                    if index < len(result.final_logits_resident_mmap_backed)
                    else None
                ),
                "expert_bytes_read": (
                    result.decode_expert_bytes_read[index]
                    if index < len(result.decode_expert_bytes_read)
                    else None
                ),
                "dense_mlp_bytes_read": (
                    result.decode_dense_mlp_bytes_read[index]
                    if index < len(result.decode_dense_mlp_bytes_read)
                    else None
                ),
                "shared_bytes_read": (
                    result.decode_shared_bytes_read[index]
                    if index < len(result.decode_shared_bytes_read)
                    else None
                ),
                "layer_count": (
                    result.decode_layer_count[index]
                    if index < len(result.decode_layer_count)
                    else None
                ),
                "dense_layer_count": (
                    result.decode_dense_layer_count[index]
                    if index < len(result.decode_dense_layer_count)
                    else None
                ),
                "moe_layer_count": (
                    result.decode_moe_layer_count[index]
                    if index < len(result.decode_moe_layer_count)
                    else None
                ),
                "layer_elapsed_seconds": (
                    result.decode_layer_elapsed_seconds[index]
                    if index < len(result.decode_layer_elapsed_seconds)
                    else None
                ),
                "attn_projection_elapsed_seconds": (
                    result.decode_attn_projection_elapsed_seconds[index]
                    if index < len(result.decode_attn_projection_elapsed_seconds)
                    else None
                ),
                "mla_attention_elapsed_seconds": (
                    result.decode_mla_attention_elapsed_seconds[index]
                    if index < len(result.decode_mla_attention_elapsed_seconds)
                    else None
                ),
                "mla_attention_cache_read_seconds": (
                    result.decode_mla_attention_cache_read_seconds[index]
                    if index < len(result.decode_mla_attention_cache_read_seconds)
                    else None
                ),
                "mla_attention_value_read_seconds": (
                    result.decode_mla_attention_value_read_seconds[index]
                    if index < len(result.decode_mla_attention_value_read_seconds)
                    else None
                ),
                "mla_attention_kernel_seconds": (
                    result.decode_mla_attention_kernel_seconds[index]
                    if index < len(result.decode_mla_attention_kernel_seconds)
                    else None
                ),
                "mla_attention_output_write_seconds": (
                    result.decode_mla_attention_output_write_seconds[index]
                    if index < len(result.decode_mla_attention_output_write_seconds)
                    else None
                ),
                "attn_output_elapsed_seconds": (
                    result.decode_attn_output_elapsed_seconds[index]
                    if index < len(result.decode_attn_output_elapsed_seconds)
                    else None
                ),
                "attn_output_bytes_read": (
                    result.decode_attn_output_bytes_read[index]
                    if index < len(result.decode_attn_output_bytes_read)
                    else None
                ),
                "attn_output_read_seconds": (
                    result.decode_attn_output_read_seconds[index]
                    if index < len(result.decode_attn_output_read_seconds)
                    else None
                ),
                "attn_output_projection_kernel_seconds": (
                    result.decode_attn_output_projection_kernel_seconds[index]
                    if index
                    < len(result.decode_attn_output_projection_kernel_seconds)
                    else None
                ),
                "post_attn_norm_weight_bytes_read": (
                    result.decode_post_attn_norm_weight_bytes_read[index]
                    if index < len(result.decode_post_attn_norm_weight_bytes_read)
                    else None
                ),
                "post_attn_norm_weight_read_seconds": (
                    result.decode_post_attn_norm_weight_read_seconds[index]
                    if index < len(result.decode_post_attn_norm_weight_read_seconds)
                    else None
                ),
                "router_bytes_read": (
                    result.decode_router_bytes_read[index]
                    if index < len(result.decode_router_bytes_read)
                    else None
                ),
                "router_correction_bias_bytes_read": (
                    result.decode_router_correction_bias_bytes_read[index]
                    if index < len(result.decode_router_correction_bias_bytes_read)
                    else None
                ),
                "router_read_seconds": (
                    result.decode_router_read_seconds[index]
                    if index < len(result.decode_router_read_seconds)
                    else None
                ),
                "router_kernel_seconds": (
                    result.decode_router_kernel_seconds[index]
                    if index < len(result.decode_router_kernel_seconds)
                    else None
                ),
                "mlp_elapsed_seconds": (
                    result.decode_mlp_elapsed_seconds[index]
                    if index < len(result.decode_mlp_elapsed_seconds)
                    else None
                ),
                "dense_mlp_elapsed_seconds": (
                    result.decode_dense_mlp_elapsed_seconds[index]
                    if index < len(result.decode_dense_mlp_elapsed_seconds)
                    else None
                ),
                "moe_mlp_elapsed_seconds": (
                    result.decode_moe_mlp_elapsed_seconds[index]
                    if index < len(result.decode_moe_mlp_elapsed_seconds)
                    else None
                ),
                "expert_read_seconds": (
                    result.decode_expert_read_seconds[index]
                    if index < len(result.decode_expert_read_seconds)
                    else None
                ),
                "shared_read_seconds": (
                    result.decode_shared_read_seconds[index]
                    if index < len(result.decode_shared_read_seconds)
                    else None
                ),
                "shared_prefetch_seconds": (
                    result.decode_shared_prefetch_seconds[index]
                    if index < len(result.decode_shared_prefetch_seconds)
                    else None
                ),
                "shared_prefetch_used_count": (
                    result.decode_shared_prefetch_used_count[index]
                    if index < len(result.decode_shared_prefetch_used_count)
                    else None
                ),
                "moe_mlp_kernel_seconds": (
                    result.decode_moe_mlp_kernel_seconds[index]
                    if index < len(result.decode_moe_mlp_kernel_seconds)
                    else None
                ),
                "moe_mlp_output_write_seconds": (
                    result.decode_moe_mlp_output_write_seconds[index]
                    if index < len(result.decode_moe_mlp_output_write_seconds)
                    else None
                ),
                "moe_mlp_overhead_seconds": (
                    result.decode_moe_mlp_overhead_seconds[index]
                    if index < len(result.decode_moe_mlp_overhead_seconds)
                    else None
                ),
                "layer_overhead_seconds": (
                    result.decode_layer_overhead_seconds[index]
                    if index < len(result.decode_layer_overhead_seconds)
                    else None
                ),
                "attn_projection_command_buffer_count": (
                    result.decode_attn_projection_command_buffer_count[index]
                    if index < len(
                        result.decode_attn_projection_command_buffer_count
                    )
                    else None
                ),
                "attn_projection_synchronous_wait_count": (
                    result.decode_attn_projection_synchronous_wait_count[index]
                    if index
                    < len(result.decode_attn_projection_synchronous_wait_count)
                    else None
                ),
                "attn_projection_async_submitted_count": (
                    result.decode_attn_projection_async_submitted_count[index]
                    if index < len(result.decode_attn_projection_async_submitted_count)
                    else None
                ),
                "rope_mla_command_buffer_count": (
                    result.decode_rope_mla_command_buffer_count[index]
                    if index < len(result.decode_rope_mla_command_buffer_count)
                    else None
                ),
                "attn_output_command_buffer_count": (
                    result.decode_attn_output_command_buffer_count[index]
                    if index < len(result.decode_attn_output_command_buffer_count)
                    else None
                ),
                "attn_output_context1_o_proj_cache_count": (
                    result.decode_attn_output_context1_o_proj_cache_count[index]
                    if index
                    < len(result.decode_attn_output_context1_o_proj_cache_count)
                    else None
                ),
                "attn_output_resident_mmap_backed_count": (
                    result.decode_attn_output_resident_mmap_backed_count[index]
                    if index
                    < len(result.decode_attn_output_resident_mmap_backed_count)
                    else None
                ),
                "post_attn_norm_command_buffer_count": (
                    result.decode_post_attn_norm_command_buffer_count[index]
                    if index < len(
                        result.decode_post_attn_norm_command_buffer_count
                    )
                    else None
                ),
                "router_command_buffer_count": (
                    result.decode_router_command_buffer_count[index]
                    if index < len(result.decode_router_command_buffer_count)
                    else None
                ),
                "post_attn_norm_router_command_buffer_count": (
                    result.decode_post_attn_norm_router_command_buffer_count[index]
                    if index < len(
                        result.decode_post_attn_norm_router_command_buffer_count
                    )
                    else None
                ),
                "dense_mlp_command_buffer_count": (
                    result.decode_dense_mlp_command_buffer_count[index]
                    if index < len(result.decode_dense_mlp_command_buffer_count)
                    else None
                ),
                "dense_mlp_synchronous_wait_count": (
                    result.decode_dense_mlp_synchronous_wait_count[index]
                    if index < len(result.decode_dense_mlp_synchronous_wait_count)
                    else None
                ),
                "dense_mlp_async_submitted_count": (
                    result.decode_dense_mlp_async_submitted_count[index]
                    if index < len(result.decode_dense_mlp_async_submitted_count)
                    else None
                ),
                "moe_mlp_command_buffer_count": (
                    result.decode_moe_mlp_command_buffer_count[index]
                    if index < len(result.decode_moe_mlp_command_buffer_count)
                    else None
                ),
                "moe_mlp_synchronous_wait_count": (
                    result.decode_moe_mlp_synchronous_wait_count[index]
                    if index < len(result.decode_moe_mlp_synchronous_wait_count)
                    else None
                ),
                "attn_output_norm_router_fused_count": (
                    result.decode_attn_output_norm_router_fused_count[index]
                    if index < len(result.decode_attn_output_norm_router_fused_count)
                    else None
                ),
                "rope_mla_attn_output_norm_router_fused_count": (
                    result.decode_rope_mla_attn_output_norm_router_fused_count[index]
                    if index
                    < len(result.decode_rope_mla_attn_output_norm_router_fused_count)
                    else None
                ),
                "rope_mla_input_buffer_direct_count": (
                    result.decode_rope_mla_input_buffer_direct_count[index]
                    if index < len(result.decode_rope_mla_input_buffer_direct_count)
                    else None
                ),
                "attn_output_buffer_direct_count": (
                    result.decode_attn_output_buffer_direct_count[index]
                    if index < len(result.decode_attn_output_buffer_direct_count)
                    else None
                ),
                "moe_mlp_input_buffer_direct_count": (
                    result.decode_moe_mlp_input_buffer_direct_count[index]
                    if index < len(result.decode_moe_mlp_input_buffer_direct_count)
                    else None
                ),
                "layer_input_buffer_direct_count": (
                    result.decode_layer_input_buffer_direct_count[index]
                    if index < len(result.decode_layer_input_buffer_direct_count)
                    else None
                ),
                "command_buffer_count": (
                    result.decode_command_buffer_count[index]
                    if index < len(result.decode_command_buffer_count)
                    else None
                ),
                "synchronous_wait_count_estimate": (
                    result.decode_synchronous_wait_count_estimate[index]
                    if index < len(result.decode_synchronous_wait_count_estimate)
                    else None
                ),
                "expert_read_dispatch_count": (
                    result.decode_expert_read_dispatch_count[index]
                    if index < len(result.decode_expert_read_dispatch_count)
                    else None
                ),
                "expert_read_task_count": (
                    result.decode_expert_read_task_count[index]
                    if index < len(result.decode_expert_read_task_count)
                    else None
                ),
                "expert_read_max_task_count": (
                    result.decode_expert_read_max_task_count[index]
                    if index < len(result.decode_expert_read_max_task_count)
                    else None
                ),
                "expert_read_max_worker_count": (
                    result.decode_expert_read_max_worker_count[index]
                    if index < len(result.decode_expert_read_max_worker_count)
                    else None
                ),
                "expert_read_pool_dispatch_count": (
                    result.decode_expert_read_pool_dispatch_count[index]
                    if index < len(result.decode_expert_read_pool_dispatch_count)
                    else None
                ),
                "expert_read_serial_dispatch_count": (
                    result.decode_expert_read_serial_dispatch_count[index]
                    if index < len(result.decode_expert_read_serial_dispatch_count)
                    else None
                ),
                "mla_value_cache_hit_count": (
                    result.decode_mla_value_cache_hit_count[index]
                    if index < len(result.decode_mla_value_cache_hit_count)
                    else None
                ),
                "mla_value_cache_store_count": (
                    result.decode_mla_value_cache_store_count[index]
                    if index < len(result.decode_mla_value_cache_store_count)
                    else None
                ),
                "mla_value_cache_bytes": (
                    result.decode_mla_value_cache_bytes[index]
                    if index < len(result.decode_mla_value_cache_bytes)
                    else None
                ),
                "mla_value_cache_total_bytes": (
                    result.decode_mla_value_cache_total_bytes[index]
                    if index < len(result.decode_mla_value_cache_total_bytes)
                    else None
                ),
            }
        )
    prompt_prefill = None
    if (
        result.prompt_prefill is not None
        or result.prompt_prefill_elapsed_seconds is not None
    ):
        prompt_prefill = {
            "elapsed_seconds": result.prompt_prefill_elapsed_seconds,
            "estimated_live_working_set_bytes": (
                result.prompt_prefill_estimated_live_working_set_bytes
            ),
            "max_live_working_set_mib": result.prefill_max_live_working_set_mib,
            "source": (
                "python_prefill_bridge"
                if result.prompt_prefill is not None
                else "runtime_prompt_token_ids"
            ),
        }
    payload: dict[str, Any] = {
        "runtime": "glm_moe_infer",
        "prompt_token_ids": list(result.prompt_token_ids),
        "generated_token_ids": list(result.generated_token_ids),
        "steps": steps,
        "elapsed_seconds": result.elapsed_seconds,
        "metal_elapsed_seconds": result.metal_elapsed_seconds,
        "prompt_prefill": prompt_prefill,
        "prefill_final_logits_elapsed_seconds": (
            result.prefill_final_logits_elapsed_seconds
        ),
        "estimated_live_working_set_bytes": result.estimated_live_working_set_bytes,
        "max_live_working_set_mib": result.max_live_working_set_mib,
        "cache_total_bytes": result.cache_total_bytes,
        "min_free_unified_memory_gib": result.min_free_unified_memory_gib,
        "admission_ok": result.admission_ok,
        "available_unified_memory_ok": result.available_unified_memory_ok,
        "system_available_memory_bytes": result.system_available_memory_bytes,
        "required_available_memory_bytes": result.required_available_memory_bytes,
        "expert_buffer_count_allocated": result.expert_buffer_count_allocated,
        "mla_kv_b_cache_enabled": result.mla_kv_b_cache_enabled,
        "mla_kv_b_cache_current_bytes": result.mla_kv_b_cache_current_bytes,
        "mla_kv_b_cache_live_estimate_bytes": (
            result.mla_kv_b_cache_live_estimate_bytes
        ),
        "prepared_dir": str(result.prepared_dir),
        "binary": str(result.binary),
        "work_dir": str(result.work_dir),
        "kept_work_dir": result.kept_work_dir,
        "cache_layout_path": str(result.cache_layout_path),
        "cache_file_path": str(result.cache_file_path),
        "note": result.note,
    }
    if request_check is not None:
        payload["request_check"] = request_check
    if launch_audit_envelope is not None:
        payload["launch_audit_envelope"] = launch_audit_envelope
    return _attach_applied_launch_profile(payload, applied_launch_profile)


def _validate_metal_runtime_context1_o_proj_cache(
    config: PreparedServerConfig,
    prepared: PreparedManifest,
) -> dict[str, Any] | None:
    layout_path = config.metal_runtime_context1_o_proj_cache_layout
    if layout_path is None:
        return None
    use_override = config.metal_runtime_context1_o_proj_cache_file is not None
    try:
        layout = load_context1_o_proj_cache_layout(
            layout_path,
            prepared_dir=prepared.manifest_path.parent,
            require_cache_file=not use_override,
        )
    except Context1OProjCacheError as exc:
        raise PreparedServerError(
            "metal runtime context1 o_proj cache validation failed: "
            f"{exc}"
        ) from exc
    cache_file = (
        Path(config.metal_runtime_context1_o_proj_cache_file)
        if config.metal_runtime_context1_o_proj_cache_file is not None
        else layout.cache_file_path
    )
    try:
        cache_file_bytes = cache_file.stat().st_size
    except OSError as exc:
        raise PreparedServerError(
            "metal runtime context1 o_proj cache validation failed: "
            f"failed to stat cache file {cache_file}: {exc}"
        ) from exc
    if cache_file_bytes < layout.total_bytes:
        raise PreparedServerError(
            "metal runtime context1 o_proj cache validation failed: "
            f"cache file is smaller than total_bytes: "
            f"{cache_file_bytes} < {layout.total_bytes}"
        )
    try:
        progress_summary = load_context1_o_proj_cache_progress(
            layout,
            require_complete=True,
        ).to_json()
    except Context1OProjCacheError as exc:
        raise PreparedServerError(
            "metal runtime context1 o_proj cache validation failed: "
            f"{exc}"
        ) from exc
    summary = layout.to_json()
    summary["ok"] = True
    summary["validated_prepared_dir"] = str(prepared.manifest_path.parent)
    summary["runtime_cache_file"] = str(cache_file)
    summary["runtime_cache_file_bytes"] = int(cache_file_bytes)
    summary["cache_file_override"] = bool(use_override)
    summary["progress"] = progress_summary
    return summary


def _require_actual_prefill_acceleration(
    result: TokenGenerationResult,
    *,
    min_accelerated_flop_fraction: float = 0.0,
    allow_router_gate_only_acceleration: bool = False,
) -> None:
    reason = prompt_prefill_acceleration_failure_reason(
        result.prompt_prefill,
        min_accelerated_flop_fraction=min_accelerated_flop_fraction,
        allow_router_gate_only_acceleration=allow_router_gate_only_acceleration,
    )
    if reason is not None:
        raise PreparedServerError(
            "prefill acceleration actual coverage failed: "
            f"{reason}"
        )


class PreparedGenerationApp:
    def __init__(self, config: PreparedServerConfig):
        self._metal_generate_session: MetalGenerateServerSession | None = None
        prepared = load_prepared_manifest(config.prepared_path)
        if config.max_live_working_set_mib is None:
            config = replace(
                config,
                max_live_working_set_mib=(
                    prepared.recommended_max_live_working_set_bytes / 1024**2
                    if prepared.recommended_max_live_working_set_bytes is not None
                    else 8192.0
                ),
            )
        if config.min_free_unified_memory_gib is None:
            config = replace(
                config,
                min_free_unified_memory_gib=(
                    prepared.recommended_min_free_unified_memory_bytes / 1024**3
                    if prepared.recommended_min_free_unified_memory_bytes is not None
                    else 0.0
                ),
            )
        model_config_path = config.model_config_path or prepared.model_dir
        try:
            validate_layout_model_config_sha256(
                model_config_path=model_config_path,
                expert_layout_path=prepared.experts_layout,
                resident_layout_path=prepared.resident_layout,
            )
        except PreparedServerError:
            raise
        except Exception as exc:
            raise PreparedServerError(str(exc)) from exc
        try:
            cfg = load_config(model_config_path)
        except ConfigError as exc:
            raise PreparedServerError(str(exc)) from exc
        if type(config.require_glm_4bit) is not bool:
            raise PreparedServerError("require_glm_4bit must be a boolean")
        if type(config.require_prepared_memory_profile) is not bool:
            raise PreparedServerError(
                "require_prepared_memory_profile must be a boolean"
            )
        if type(config.require_public_glm_5_2_shape) is not bool:
            raise PreparedServerError("require_public_glm_5_2_shape must be a boolean")
        if (
            config.require_public_glm_5_2_shape
            and config.allow_missing_dsa_indexer
        ):
            raise PreparedServerError(
                "require_public_glm_5_2_shape cannot be used with "
                "allow_missing_dsa_indexer"
            )
        if (
            config.require_public_glm_5_2_shape
            and not config.require_prepared_memory_profile
        ):
            config = replace(config, require_prepared_memory_profile=True)
        glm_readiness = None
        if config.require_glm_4bit:
            glm_readiness = _glm_4bit_readiness(prepared, cfg)
            reason = _glm_4bit_readiness_failure_reason(
                glm_readiness
            )
            if reason is not None:
                raise PreparedServerError(
                    "prepared GLM 4bit readiness failed: "
                    f"{reason}"
                )
        if config.require_public_glm_5_2_shape:
            if glm_readiness is None:
                glm_readiness = _glm_4bit_readiness(prepared, cfg)
            reason = _public_glm_5_2_shape_failure_reason(glm_readiness)
            if reason is not None:
                raise PreparedServerError(
                    "prepared public GLM-5.2 readiness failed: "
                    f"{reason}"
                )
        try:
            cache_layout = load_decode_cache_layout(prepared.decode_cache_layout)
        except DecodeCacheError as exc:
            raise PreparedServerError(str(exc)) from exc
        context_candidates = [int(cache_layout.max_context_tokens)]
        if prepared.max_context_tokens is not None:
            context_candidates.append(int(prepared.max_context_tokens))
        if cfg.max_position_embeddings is not None:
            context_candidates.append(int(cfg.max_position_embeddings))
        effective_context_tokens = min(context_candidates)
        if config.max_new_tokens_cap < 0:
            raise PreparedServerError("max_new_tokens_cap must be non-negative")
        if config.max_prompt_tokens <= 0:
            raise PreparedServerError("max_prompt_tokens must be positive")
        if config.max_request_bytes <= 0:
            raise PreparedServerError("max_request_bytes must be positive")
        if config.logits_top_k_cap <= 0:
            raise PreparedServerError("logits_top_k_cap must be positive")
        for name in (
            "max_cache_read_mib",
            "max_cache_file_mib",
            "max_runner_scratch_mib",
        ):
            _require_positive_config_value(config, name)
        _require_nonnegative_config_value(
            config,
            "decode_max_routed_read_gib_per_token",
        )
        _require_nonnegative_config_value(
            config,
            "decode_max_routed_read_seconds_per_token",
        )
        if config.max_live_working_set_mib is not None:
            _require_nonnegative_config_value(config, "max_live_working_set_mib")
        if config.min_free_unified_memory_gib is not None:
            _require_nonnegative_config_value(config, "min_free_unified_memory_gib")
        for name in (
            "expert_read_advise_merge_gap_kib",
            "expert_read_advise_align_kib",
        ):
            value = getattr(config, name)
            if type(value) is not int:
                raise PreparedServerError(f"{name} must be an integer")
            if value < 0:
                raise PreparedServerError(f"{name} must be non-negative")
        if config.prefill_prompt_chunk_tokens < 0:
            raise PreparedServerError("prefill_prompt_chunk_tokens must be non-negative")
        for name in (
            "prefill_max_prompt_batch_mib",
            "prefill_max_cache_write_mib",
            "prefill_max_stage_mib",
            "prefill_max_compact_stage_mib",
            "prefill_copy_chunk_mib",
        ):
            _require_positive_config_value(config, name)
        for name in (
            "prefill_max_stage_raw_ranges",
            "prefill_max_stage_coalesced_ranges",
        ):
            value = getattr(config, name)
            if type(value) is not int:
                raise PreparedServerError(f"{name} must be an integer")
            if value < 0:
                raise PreparedServerError(f"{name} must be non-negative")
        if type(config.prefill_expert_stage_tiling) is not bool:
            raise PreparedServerError(
                "prefill_expert_stage_tiling must be a boolean"
            )
        if type(config.prefill_persistent_moe_plan_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_moe_plan_server must be a boolean"
            )
        if type(config.prefill_persistent_resident_linear_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_resident_linear_server must be a boolean"
            )
        if type(config.prefill_persistent_attention_projection_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_attention_projection_server must be a boolean"
            )
        if type(config.prefill_persistent_attention_output_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_attention_output_server must be a boolean"
            )
        if type(config.prefill_persistent_shared_expert_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_shared_expert_server must be a boolean"
            )
        if type(config.prefill_persistent_rope_split_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_rope_split_server must be a boolean"
            )
        if type(config.prefill_persistent_mla_attention_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_mla_attention_server must be a boolean"
            )
        if type(config.prefill_persistent_rmsnorm_server) is not bool:
            raise PreparedServerError(
                "prefill_persistent_rmsnorm_server must be a boolean"
            )
        _require_nonnegative_config_value(config, "prefill_stage_disk_margin_mib")
        _require_nonnegative_config_value(
            config,
            "prefill_max_routed_read_amplification",
        )
        _require_nonnegative_config_value(config, "prefill_max_routed_read_gib")
        _require_nonnegative_config_value(
            config,
            "prefill_ssd_read_gib_per_second",
        )
        _require_nonnegative_config_value(
            config,
            "prefill_max_routed_read_seconds",
        )
        if (
            config.prefill_max_routed_read_seconds > 0
            and config.prefill_ssd_read_gib_per_second <= 0
        ):
            raise PreparedServerError(
                "prefill_ssd_read_gib_per_second must be positive when "
                "prefill_max_routed_read_seconds is set"
            )
        if (
            config.decode_max_routed_read_seconds_per_token > 0
            and config.prefill_ssd_read_gib_per_second <= 0
        ):
            raise PreparedServerError(
                "prefill_ssd_read_gib_per_second must be positive when "
                "decode_max_routed_read_seconds_per_token is set"
            )
        block = config.prefill_moe_token_block
        if isinstance(block, bool):
            raise PreparedServerError("prefill_moe_token_block must be a positive integer or auto")
        if isinstance(block, int):
            if block <= 0:
                raise PreparedServerError(
                    "prefill_moe_token_block must be a positive integer or auto"
                )
        else:
            text_block = str(block).strip().lower()
            if text_block == "auto":
                config = replace(config, prefill_moe_token_block="auto")
            else:
                try:
                    parsed_block = int(text_block)
                except ValueError as exc:
                    raise PreparedServerError(
                        "prefill_moe_token_block must be a positive integer or auto"
                    ) from exc
                if parsed_block <= 0:
                    raise PreparedServerError(
                        "prefill_moe_token_block must be a positive integer or auto"
                    )
                config = replace(config, prefill_moe_token_block=parsed_block)
        accumulator = str(config.prefill_moe_output_accumulator).strip().lower()
        if accumulator not in {"env", "file", "memory"}:
            raise PreparedServerError(
                "prefill_moe_output_accumulator must be env, file, or memory"
            )
        config = replace(config, prefill_moe_output_accumulator=accumulator)
        if type(config.prefill_mla_key_cache) is not bool:
            raise PreparedServerError("prefill_mla_key_cache must be a boolean")
        if type(config.decode_mla_key_cache) is not bool:
            raise PreparedServerError("decode_mla_key_cache must be a boolean")
        if type(config.metal_runtime_cache_mla_kv_b_f32) is not bool:
            raise PreparedServerError(
                "metal_runtime_cache_mla_kv_b_f32 must be a boolean"
            )
        if type(config.metal_runtime_mmap_final_logits) is not bool:
            raise PreparedServerError(
                "metal_runtime_mmap_final_logits must be a boolean"
            )
        _require_nonnegative_config_value(
            config,
            "metal_runtime_max_mla_kv_b_cache_mib",
        )
        _require_nonnegative_config_value(
            config,
            "metal_runtime_max_adaptive_expert_cache_gib",
        )
        if (
            config.metal_runtime_cache_mla_kv_b_f32
            and config.metal_runtime_max_mla_kv_b_cache_mib <= 0
        ):
            raise PreparedServerError(
                "metal_runtime_max_mla_kv_b_cache_mib must be positive when "
                "metal_runtime_cache_mla_kv_b_f32 is enabled"
            )
        if config.metal_runtime_expert_pin_plan is not None:
            try:
                expert_pin_plan = Path(
                    config.metal_runtime_expert_pin_plan
                ).expanduser().resolve()
            except TypeError as exc:
                raise PreparedServerError(
                    "metal_runtime_expert_pin_plan must be a path"
                ) from exc
            if not expert_pin_plan.is_file():
                raise PreparedServerError(
                    f"metal runtime expert pin plan does not exist: {expert_pin_plan}"
                )
            config = replace(
                config,
                metal_runtime_expert_pin_plan=expert_pin_plan,
            )
        if config.metal_runtime_context1_o_proj_cache_layout is not None:
            try:
                config = replace(
                    config,
                    metal_runtime_context1_o_proj_cache_layout=Path(
                        config.metal_runtime_context1_o_proj_cache_layout
                    ),
                )
            except TypeError as exc:
                raise PreparedServerError(
                    "metal_runtime_context1_o_proj_cache_layout must be a path"
                ) from exc
        if config.metal_runtime_context1_o_proj_cache_file is not None:
            try:
                config = replace(
                    config,
                    metal_runtime_context1_o_proj_cache_file=Path(
                        config.metal_runtime_context1_o_proj_cache_file
                    ),
                )
            except TypeError as exc:
                raise PreparedServerError(
                    "metal_runtime_context1_o_proj_cache_file must be a path"
                ) from exc
        if (
            config.metal_runtime_context1_o_proj_cache_file is not None
            and config.metal_runtime_context1_o_proj_cache_layout is None
        ):
            raise PreparedServerError(
                "metal_runtime_context1_o_proj_cache_file requires "
                "metal_runtime_context1_o_proj_cache_layout"
            )
        if config.prefill_mla_kv_b_cache_dir is not None:
            try:
                config = replace(
                    config,
                    prefill_mla_kv_b_cache_dir=Path(
                        config.prefill_mla_kv_b_cache_dir
                    ),
                )
            except TypeError as exc:
                raise PreparedServerError(
                    "prefill_mla_kv_b_cache_dir must be a path"
                ) from exc
        if type(config.metal_final_logits) is not bool:
            raise PreparedServerError("metal_final_logits must be a boolean")
        if type(config.metal_runtime_generation) is not bool:
            raise PreparedServerError("metal_runtime_generation must be a boolean")
        try:
            config = replace(
                config,
                metal_binary_path=Path(config.metal_binary_path),
            )
        except TypeError as exc:
            raise PreparedServerError("metal_binary_path must be a path") from exc
        if config.prefill_linear_backend not in {
            "custom-metal",
            "mpp-f32",
            "mpsgraph-f32",
            "mps-matrix-f32",
            "auto",
        }:
            raise PreparedServerError(
                "prefill_linear_backend must be custom-metal, mpp-f32, "
                "mpsgraph-f32, mps-matrix-f32, or auto"
            )
        if type(config.require_prefill_acceleration) is not bool:
            raise PreparedServerError("require_prefill_acceleration must be a boolean")
        if type(config.allow_router_gate_only_prefill_acceleration) is not bool:
            raise PreparedServerError(
                "allow_router_gate_only_prefill_acceleration must be a boolean"
            )
        if type(config.prefill_compile_mpp_probe) is not bool:
            raise PreparedServerError("prefill_compile_mpp_probe must be a boolean")
        if type(config.prefill_run_mpp_probe) is not bool:
            raise PreparedServerError("prefill_run_mpp_probe must be a boolean")
        if type(config.prefill_run_mpsgraph_probe) is not bool:
            raise PreparedServerError("prefill_run_mpsgraph_probe must be a boolean")
        _require_positive_config_value(
            config,
            "prefill_backend_probe_timeout_seconds",
        )
        _require_nonnegative_config_value(
            config,
            "prefill_min_accelerated_flop_fraction",
        )
        if config.prefill_min_accelerated_flop_fraction > 1.0:
            raise PreparedServerError(
                "prefill_min_accelerated_flop_fraction must be <= 1"
            )
        if config.metal_runtime_generation and (
            config.require_prefill_acceleration
            or config.prefill_min_accelerated_flop_fraction > 0.0
        ):
            raise PreparedServerError(
                "metal_runtime_generation currently cannot satisfy prefill "
                "acceleration requirements"
            )
        _require_nonnegative_config_value(
            config,
            "prefill_router_hybrid_margin_threshold",
        )
        if (
            type(config.prefill_mpsgraph_min_batch_tokens) is not int
            or config.prefill_mpsgraph_min_batch_tokens <= 0
        ):
            raise PreparedServerError(
                "prefill_mpsgraph_min_batch_tokens must be positive"
            )
        if (
            type(config.prefill_mpsgraph_min_matrix_dim) is not int
            or config.prefill_mpsgraph_min_matrix_dim <= 0
        ):
            raise PreparedServerError("prefill_mpsgraph_min_matrix_dim must be positive")
        if config.enforce_prefill_acceleration_probe:
            _require_server_prefill_acceleration_backend(config)
        runtime_prefill_linear_backend = _resolve_server_prefill_linear_backend(
            config.prefill_linear_backend,
            compile_mpp_probe=config.prefill_compile_mpp_probe,
            run_mpp_probe=config.prefill_run_mpp_probe,
            run_mpsgraph_probe=config.prefill_run_mpsgraph_probe,
            probe_timeout_seconds=config.prefill_backend_probe_timeout_seconds,
        )
        metal_runtime_context1_o_proj_cache = (
            _validate_metal_runtime_context1_o_proj_cache(config, prepared)
        )
        if not config.served_model_name:
            raise PreparedServerError("served_model_name must be non-empty")
        self.state = PreparedServerState(
            config=config,
            prepared=prepared,
            model_config=cfg,
            decode_cache_context_tokens=int(cache_layout.max_context_tokens),
            model_context_tokens=(
                int(cfg.max_position_embeddings)
                if cfg.max_position_embeddings is not None
                else None
            ),
            effective_context_tokens=effective_context_tokens,
            runtime_prefill_linear_backend=runtime_prefill_linear_backend,
            metal_runtime_context1_o_proj_cache=(
                metal_runtime_context1_o_proj_cache
            ),
            lock=threading.Lock(),
        )

    def _metal_runtime_session(self) -> MetalGenerateServerSession:
        if self._metal_generate_session is None or self._metal_generate_session.closed:
            runtime_session_kwargs: dict[str, object] = {
                "binary": self.state.config.metal_binary_path,
                "prepared_dir": self.state.prepared.manifest_path.parent,
                "quiet": not self.state.config.echo_runner_output,
            }
            if self.state.config.metal_runtime_expert_pin_plan is not None:
                runtime_session_kwargs["expert_pin_plan"] = (
                    self.state.config.metal_runtime_expert_pin_plan
                )
            if self.state.config.metal_runtime_max_adaptive_expert_cache_gib > 0.0:
                runtime_session_kwargs["max_adaptive_expert_cache_gib"] = (
                    self.state.config.metal_runtime_max_adaptive_expert_cache_gib
                )
            self._metal_generate_session = MetalGenerateServerSession(
                **runtime_session_kwargs,
            )
        return self._metal_generate_session

    def _reset_metal_runtime_session(self) -> None:
        if self._metal_generate_session is not None:
            self._metal_generate_session.close()
            self._metal_generate_session = None

    def close(self) -> None:
        self._reset_metal_runtime_session()

    def health(self) -> dict[str, Any]:
        prepared = self.state.prepared
        cfg = self.state.config
        system_memory = _system_memory_health()
        glm_4bit_readiness = _glm_4bit_readiness(
            prepared,
            self.state.model_config,
        )
        suggested_launch_guard_flags = _suggest_prepared_launch_guard_flags(
            prepared,
            require_memory_profile=cfg.require_prepared_memory_profile,
        )
        suggested_prepared_ssd_read_flags = _suggest_prepared_ssd_read_flags(
            prepared,
            ssd_read_gib_per_second=cfg.prefill_ssd_read_gib_per_second,
        )
        suggested_prefill_backend_probe_flags = _suggest_prefill_backend_probe_flags(
            compile_mpp_probe=cfg.prefill_compile_mpp_probe,
            run_mpp_probe=cfg.prefill_run_mpp_probe,
            run_mpsgraph_probe=cfg.prefill_run_mpsgraph_probe,
            source="prepared_health",
        )
        suggested_prefill_runtime_policy_flags = (
            _suggest_prefill_runtime_policy_flags(
                cfg,
                source="prepared_health",
            )
        )
        suggested_prefill_copy_policy_flags = _suggest_prefill_copy_policy_flags(
            cfg,
            source="prepared_health",
        )
        suggested_prefill_persistent_moe_plan_server_flags = (
            _suggest_prefill_persistent_moe_plan_server_flags(
                enabled=cfg.prefill_persistent_moe_plan_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_resident_linear_server_flags = (
            _suggest_prefill_persistent_resident_linear_server_flags(
                enabled=cfg.prefill_persistent_resident_linear_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_attention_projection_server_flags = (
            _suggest_prefill_persistent_attention_projection_server_flags(
                enabled=cfg.prefill_persistent_attention_projection_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_attention_output_server_flags = (
            _suggest_prefill_persistent_attention_output_server_flags(
                enabled=cfg.prefill_persistent_attention_output_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_shared_expert_server_flags = (
            _suggest_prefill_persistent_shared_expert_server_flags(
                enabled=cfg.prefill_persistent_shared_expert_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_rope_split_server_flags = (
            _suggest_prefill_persistent_rope_split_server_flags(
                enabled=cfg.prefill_persistent_rope_split_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_mla_attention_server_flags = (
            _suggest_prefill_persistent_mla_attention_server_flags(
                enabled=cfg.prefill_persistent_mla_attention_server,
                source="prepared_health",
            )
        )
        suggested_prefill_persistent_rmsnorm_server_flags = (
            _suggest_prefill_persistent_rmsnorm_server_flags(
                enabled=cfg.prefill_persistent_rmsnorm_server,
                source="prepared_health",
            )
        )
        suggested_prefill_moe_output_accumulator_flags = (
            _suggest_prefill_moe_output_accumulator_flags(
                mode=cfg.prefill_moe_output_accumulator,
                source="prepared_health",
            )
        )
        suggested_decode_guard_flags = _suggest_prepared_decode_guard_flags(
            glm_4bit_readiness,
            ssd_read_gib_per_second=cfg.prefill_ssd_read_gib_per_second,
            decode_mla_key_cache=cfg.decode_mla_key_cache,
        )
        suggested_glm_4bit_guard_flags = _suggest_glm_4bit_guard_flags(
            glm_4bit_readiness,
            source="prepared_health",
        )
        suggested_public_glm_5_2_shape_guard_flags = (
            _suggest_public_glm_5_2_shape_guard_flags(
                glm_4bit_readiness,
                source="prepared_health",
            )
        )
        prepared_run_lock = inspect_prepared_run_lock_path(
            prepared_run_lock_path_for_manifest(prepared.manifest_path)
        )
        prefill_backend = _prefill_backend_health(
            cfg.prefill_linear_backend,
            mpsgraph_min_batch_tokens=cfg.prefill_mpsgraph_min_batch_tokens,
            mpsgraph_min_matrix_dim=cfg.prefill_mpsgraph_min_matrix_dim,
            compile_mpp_probe=cfg.prefill_compile_mpp_probe,
            run_mpp_probe=cfg.prefill_run_mpp_probe,
            run_mpsgraph_probe=cfg.prefill_run_mpsgraph_probe,
            probe_timeout_seconds=cfg.prefill_backend_probe_timeout_seconds,
        )
        prefill_backend["effective_backend"] = (
            self.state.runtime_prefill_linear_backend
        )
        prefill_acceleration_requirement = _prefill_acceleration_requirement_health(
            cfg,
            prefill_backend,
        )
        capability = (
            prefill_backend.get("capability")
            if isinstance(prefill_backend, dict)
            else None
        )
        prefill_acceleration_flags = (
            capability.get("suggested_prefill_acceleration_flags")
            if isinstance(capability, dict)
            else None
        )
        prefill_acceleration_launch_profile_flags = (
            _prefill_acceleration_launch_profile_flags(
                cfg,
                prefill_acceleration_flags,
            )
        )
        metal_runtime_mla_kv_b_cache_plan = _estimate_mla_kv_b_f32_cache_plan(
            self.state.model_config
        )
        recommended_mla_kv_b_cache_mib = None
        estimated_mla_kv_b_cache_bytes = None
        if (
            cfg.metal_runtime_generation
            and isinstance(metal_runtime_mla_kv_b_cache_plan, dict)
        ):
            recommended = metal_runtime_mla_kv_b_cache_plan.get(
                "recommended_max_cache_mib"
            )
            estimated = metal_runtime_mla_kv_b_cache_plan.get(
                "estimated_full_cache_bytes"
            )
            if isinstance(recommended, (int, float)):
                recommended_mla_kv_b_cache_mib = float(recommended)
            if isinstance(estimated, int):
                estimated_mla_kv_b_cache_bytes = estimated
        suggested_metal_runtime_mla_kv_b_cache_flags = (
            suggest_metal_runtime_mla_kv_b_cache_flags(
                enabled=cfg.metal_runtime_cache_mla_kv_b_f32,
                max_cache_mib=cfg.metal_runtime_max_mla_kv_b_cache_mib,
                source="prepared_health",
                recommended_cache_mib=recommended_mla_kv_b_cache_mib,
                estimated_cache_bytes=estimated_mla_kv_b_cache_bytes,
            )
        )
        suggested_metal_runtime_mmap_final_logits_flags = (
            suggest_metal_runtime_mmap_final_logits_flags(
                enabled=cfg.metal_runtime_mmap_final_logits,
                metal_runtime_generation=cfg.metal_runtime_generation,
                source="prepared_health",
            )
        )
        suggested_metal_runtime_context1_o_proj_cache_flags = (
            suggest_metal_runtime_context1_o_proj_cache_flags(
                layout_path=cfg.metal_runtime_context1_o_proj_cache_layout,
                cache_file_path=cfg.metal_runtime_context1_o_proj_cache_file,
                metal_runtime_generation=cfg.metal_runtime_generation,
                source="prepared_health",
                validated_cache=self.state.metal_runtime_context1_o_proj_cache,
            )
        )
        prepared_target = prepared_launch_profile_target(prepared)
        suggested_launch_profile = combine_suggested_launch_profile(
            launch_guard_flags=suggested_launch_guard_flags,
            prepared_ssd_read_flags=suggested_prepared_ssd_read_flags,
            prefill_backend_probe_flags=suggested_prefill_backend_probe_flags,
            prefill_runtime_policy_flags=suggested_prefill_runtime_policy_flags,
            prefill_copy_policy_flags=suggested_prefill_copy_policy_flags,
            glm_4bit_guard_flags=suggested_glm_4bit_guard_flags,
            public_glm_5_2_shape_guard_flags=(
                suggested_public_glm_5_2_shape_guard_flags
            ),
            prefill_acceleration_flags=(
                prefill_acceleration_launch_profile_flags
            ),
            prefill_persistent_moe_plan_server_flags=(
                suggested_prefill_persistent_moe_plan_server_flags
            ),
            prefill_persistent_resident_linear_server_flags=(
                suggested_prefill_persistent_resident_linear_server_flags
            ),
            prefill_persistent_attention_projection_server_flags=(
                suggested_prefill_persistent_attention_projection_server_flags
            ),
            prefill_persistent_attention_output_server_flags=(
                suggested_prefill_persistent_attention_output_server_flags
            ),
            prefill_persistent_shared_expert_server_flags=(
                suggested_prefill_persistent_shared_expert_server_flags
            ),
            prefill_persistent_rope_split_server_flags=(
                suggested_prefill_persistent_rope_split_server_flags
            ),
            prefill_persistent_mla_attention_server_flags=(
                suggested_prefill_persistent_mla_attention_server_flags
            ),
            prefill_persistent_rmsnorm_server_flags=(
                suggested_prefill_persistent_rmsnorm_server_flags
            ),
            prefill_moe_output_accumulator_flags=(
                suggested_prefill_moe_output_accumulator_flags
            ),
            decode_guard_flags=suggested_decode_guard_flags,
            metal_runtime_mla_kv_b_cache_flags=(
                suggested_metal_runtime_mla_kv_b_cache_flags
            ),
            metal_runtime_mmap_final_logits_flags=(
                suggested_metal_runtime_mmap_final_logits_flags
            ),
            metal_runtime_context1_o_proj_cache_flags=(
                suggested_metal_runtime_context1_o_proj_cache_flags
            ),
            prepared_target=prepared_target,
            source="prepared_health",
        )
        return {
            "ok": True,
            "prepared_manifest": str(prepared.manifest_path),
            "model_dir": str(prepared.model_dir),
            "prepared_run_lock": prepared_run_lock.to_dict(),
            "prepared_storage": _prepared_storage_health(prepared),
            "prepare_live_memory": _prepare_live_memory_health(
                prepared,
                system_memory,
            ),
            "suggested_launch_guard_flags": suggested_launch_guard_flags,
            "suggested_prepared_ssd_read_flags": suggested_prepared_ssd_read_flags,
            "suggested_prefill_backend_probe_flags": (
                suggested_prefill_backend_probe_flags
            ),
            "suggested_prefill_runtime_policy_flags": (
                suggested_prefill_runtime_policy_flags
            ),
            "suggested_prefill_copy_policy_flags": (
                suggested_prefill_copy_policy_flags
            ),
            "suggested_prefill_persistent_moe_plan_server_flags": (
                suggested_prefill_persistent_moe_plan_server_flags
            ),
            "suggested_prefill_persistent_resident_linear_server_flags": (
                suggested_prefill_persistent_resident_linear_server_flags
            ),
            "suggested_prefill_persistent_attention_projection_server_flags": (
                suggested_prefill_persistent_attention_projection_server_flags
            ),
            "suggested_prefill_persistent_attention_output_server_flags": (
                suggested_prefill_persistent_attention_output_server_flags
            ),
            "suggested_prefill_persistent_shared_expert_server_flags": (
                suggested_prefill_persistent_shared_expert_server_flags
            ),
            "suggested_prefill_persistent_rope_split_server_flags": (
                suggested_prefill_persistent_rope_split_server_flags
            ),
            "suggested_prefill_persistent_mla_attention_server_flags": (
                suggested_prefill_persistent_mla_attention_server_flags
            ),
            "suggested_prefill_persistent_rmsnorm_server_flags": (
                suggested_prefill_persistent_rmsnorm_server_flags
            ),
            "suggested_prefill_moe_output_accumulator_flags": (
                suggested_prefill_moe_output_accumulator_flags
            ),
            "suggested_glm_4bit_guard_flags": suggested_glm_4bit_guard_flags,
            "suggested_public_glm_5_2_shape_guard_flags": (
                suggested_public_glm_5_2_shape_guard_flags
            ),
            "suggested_decode_guard_flags": suggested_decode_guard_flags,
            "suggested_launch_profile": suggested_launch_profile,
            "applied_launch_profile": cfg.applied_launch_profile,
            "launch_audit_envelope": cfg.launch_audit_envelope,
            "require_prepared_memory_profile": cfg.require_prepared_memory_profile,
            "max_new_tokens_cap": cfg.max_new_tokens_cap,
            "max_prompt_tokens": cfg.max_prompt_tokens,
            "prepared_max_context_tokens": prepared.max_context_tokens,
            "decode_cache_context_tokens": self.state.decode_cache_context_tokens,
            "model_context_tokens": self.state.model_context_tokens,
            "effective_context_tokens": self.state.effective_context_tokens,
            "effective_max_prompt_tokens": min(
                cfg.max_prompt_tokens,
                self.state.effective_context_tokens,
            ),
            "batch_prefill_prompt": cfg.batch_prefill_prompt,
            "metal_runtime_generation": cfg.metal_runtime_generation,
            "metal_binary_path": str(cfg.metal_binary_path),
            "metal_runtime_cache_mla_kv_b_f32": (
                cfg.metal_runtime_cache_mla_kv_b_f32
            ),
            "metal_runtime_max_mla_kv_b_cache_mib": (
                cfg.metal_runtime_max_mla_kv_b_cache_mib
            ),
            "metal_runtime_expert_pin_plan": (
                str(cfg.metal_runtime_expert_pin_plan)
                if cfg.metal_runtime_expert_pin_plan is not None
                else None
            ),
            "metal_runtime_max_adaptive_expert_cache_gib": (
                cfg.metal_runtime_max_adaptive_expert_cache_gib
            ),
            "metal_runtime_mmap_final_logits": (
                cfg.metal_runtime_mmap_final_logits
            ),
            "metal_runtime_context1_o_proj_cache_layout": (
                str(cfg.metal_runtime_context1_o_proj_cache_layout)
                if cfg.metal_runtime_context1_o_proj_cache_layout is not None
                else None
            ),
            "metal_runtime_context1_o_proj_cache_file": (
                str(cfg.metal_runtime_context1_o_proj_cache_file)
                if cfg.metal_runtime_context1_o_proj_cache_file is not None
                else None
            ),
            "metal_runtime_context1_o_proj_cache": (
                self.state.metal_runtime_context1_o_proj_cache
            ),
            "metal_runtime_mla_kv_b_cache_plan": (
                metal_runtime_mla_kv_b_cache_plan
            ),
            "suggested_metal_runtime_mla_kv_b_cache_flags": (
                suggested_metal_runtime_mla_kv_b_cache_flags
            ),
            "suggested_metal_runtime_mmap_final_logits_flags": (
                suggested_metal_runtime_mmap_final_logits_flags
            ),
            "suggested_metal_runtime_context1_o_proj_cache_flags": (
                suggested_metal_runtime_context1_o_proj_cache_flags
            ),
            "metal_runtime_session_started": (
                self._metal_generate_session is not None
                and not self._metal_generate_session.closed
            ),
            "metal_runtime_session_request_count": (
                self._metal_generate_session.request_count
                if self._metal_generate_session is not None
                else 0
            ),
            "max_cache_read_mib": cfg.max_cache_read_mib,
            "max_cache_file_mib": cfg.max_cache_file_mib,
            "decode_max_routed_read_gib_per_token": (
                cfg.decode_max_routed_read_gib_per_token
            ),
            "decode_max_routed_read_seconds_per_token": (
                cfg.decode_max_routed_read_seconds_per_token
            ),
            "max_runner_scratch_mib": cfg.max_runner_scratch_mib,
            "max_live_working_set_mib": cfg.max_live_working_set_mib,
            "min_free_unified_memory_gib": cfg.min_free_unified_memory_gib,
            "expert_read_advise_merge_gap_kib": (
                cfg.expert_read_advise_merge_gap_kib
            ),
            "expert_read_advise_align_kib": cfg.expert_read_advise_align_kib,
            "system_memory": system_memory,
            "memory_guard": _memory_guard_health(cfg, system_memory),
            "prepared_runtime_profile": _prepared_runtime_profile_health(
                prepared,
                system_memory,
            ),
            "glm_4bit_readiness": glm_4bit_readiness,
            "prefill_prompt_chunk_tokens": cfg.prefill_prompt_chunk_tokens,
            "prefill_max_prompt_batch_mib": cfg.prefill_max_prompt_batch_mib,
            "prefill_max_cache_write_mib": cfg.prefill_max_cache_write_mib,
            "prefill_max_stage_mib": cfg.prefill_max_stage_mib,
            "prefill_max_compact_stage_mib": cfg.prefill_max_compact_stage_mib,
            "prefill_max_stage_raw_ranges": cfg.prefill_max_stage_raw_ranges,
            "prefill_max_stage_coalesced_ranges": (
                cfg.prefill_max_stage_coalesced_ranges
            ),
            "prefill_expert_stage_tiling": cfg.prefill_expert_stage_tiling,
            "prefill_persistent_moe_plan_server": (
                cfg.prefill_persistent_moe_plan_server
            ),
            "prefill_persistent_resident_linear_server": (
                cfg.prefill_persistent_resident_linear_server
            ),
            "prefill_persistent_attention_projection_server": (
                cfg.prefill_persistent_attention_projection_server
            ),
            "prefill_persistent_attention_output_server": (
                cfg.prefill_persistent_attention_output_server
            ),
            "prefill_persistent_shared_expert_server": (
                cfg.prefill_persistent_shared_expert_server
            ),
            "prefill_persistent_rope_split_server": (
                cfg.prefill_persistent_rope_split_server
            ),
            "prefill_persistent_mla_attention_server": (
                cfg.prefill_persistent_mla_attention_server
            ),
            "prefill_persistent_rmsnorm_server": (
                cfg.prefill_persistent_rmsnorm_server
            ),
            "prefill_copy_chunk_mib": cfg.prefill_copy_chunk_mib,
            "prefill_stage_disk_margin_mib": cfg.prefill_stage_disk_margin_mib,
            "prefill_max_routed_read_amplification": (
                cfg.prefill_max_routed_read_amplification
            ),
            "prefill_max_routed_read_gib": cfg.prefill_max_routed_read_gib,
            "prefill_ssd_read_gib_per_second": cfg.prefill_ssd_read_gib_per_second,
            "prefill_max_routed_read_seconds": cfg.prefill_max_routed_read_seconds,
            "prefill_moe_token_block": cfg.prefill_moe_token_block,
            "prefill_moe_output_accumulator": cfg.prefill_moe_output_accumulator,
            "prefill_static_capacity_per_expert": (
                cfg.prefill_static_capacity_per_expert
            ),
            "prefill_allow_static_capacity_overflow": (
                cfg.prefill_allow_static_capacity_overflow
            ),
            "prefill_mla_kv_b_cache_dir": (
                str(cfg.prefill_mla_kv_b_cache_dir)
                if cfg.prefill_mla_kv_b_cache_dir is not None
                else None
            ),
            "prefill_mla_key_cache": cfg.prefill_mla_key_cache,
            "decode_mla_key_cache": cfg.decode_mla_key_cache,
            "prefill_linear_backend": cfg.prefill_linear_backend,
            "runtime_prefill_linear_backend": self.state.runtime_prefill_linear_backend,
            "require_prefill_acceleration": cfg.require_prefill_acceleration,
            "allow_router_gate_only_prefill_acceleration": (
                cfg.allow_router_gate_only_prefill_acceleration
            ),
            "prefill_min_accelerated_flop_fraction": (
                cfg.prefill_min_accelerated_flop_fraction
            ),
            "prefill_acceleration_requirement": prefill_acceleration_requirement,
            "prefill_mpsgraph_min_batch_tokens": (
                cfg.prefill_mpsgraph_min_batch_tokens
            ),
            "prefill_mpsgraph_min_matrix_dim": cfg.prefill_mpsgraph_min_matrix_dim,
            "prefill_router_hybrid_margin_threshold": (
                cfg.prefill_router_hybrid_margin_threshold
            ),
            "prefill_compile_mpp_probe": cfg.prefill_compile_mpp_probe,
            "prefill_run_mpp_probe": cfg.prefill_run_mpp_probe,
            "prefill_run_mpsgraph_probe": cfg.prefill_run_mpsgraph_probe,
            "metal_final_logits": cfg.metal_final_logits,
            "prefill_backend_probe_timeout_seconds": (
                cfg.prefill_backend_probe_timeout_seconds
            ),
            "prefill_backend": prefill_backend,
            "served_model_name": cfg.served_model_name,
            "endpoints": (
                "/generate-token-ids",
                "/generate-text",
                "/v1/models",
                "/v1/completions",
                "/v1/chat/completions",
            ),
        }

    def inspect_token_request(
        self,
        *,
        prompt_token_count: int,
        payload: dict[str, Any],
        runtime_preflight: bool = False,
        generation_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(prompt_token_count, bool) or prompt_token_count <= 0:
            raise PreparedServerError("prompt_token_count must be positive")
        config = self.state.config
        max_new_tokens = _max_new_tokens(payload, self.state.config.max_new_tokens_cap)
        prompt_cap = _request_prompt_token_cap(
            self.state,
            max_new_tokens=max_new_tokens,
        )
        if prompt_token_count > prompt_cap:
            raise PreparedServerError(
                f"prompt token count {prompt_token_count} exceeds server cap "
                f"{prompt_cap}"
            )
        launch_audit_envelope = _launch_audit_envelope_request_summary(
            config.launch_audit_envelope,
            prompt_token_count=prompt_token_count,
            max_new_tokens=max_new_tokens,
        )
        kwargs = _base_generation_kwargs(
            self.state,
            payload=payload,
            prompt_token_count=prompt_token_count,
            generation_overrides=generation_overrides,
        )
        configured_chunk = self.state.config.prefill_prompt_chunk_tokens
        resolved_chunk: int | None = None
        max_safe_chunk: int | None = None
        auto_chunk_plan = None
        max_safe_chunk_plan = None
        prefill_chunk_error: str | None = None
        if kwargs["batch_prefill_prompt"]:
            chunk_plan_kwargs = {
                "expert_layout_path": self.state.prepared.experts_layout,
                "resident_layout_path": self.state.prepared.resident_layout,
                "cache_layout_path": self.state.prepared.decode_cache_layout,
                "prompt_tokens": prompt_token_count,
                "start_position": 0,
                "layers": kwargs["layers"],
                "dense_layers": kwargs["dense_layers"],
                "work_dir": kwargs["work_dir"],
                "top_k": kwargs["top_k"],
                "max_prompt_batch_mib": kwargs["prefill_max_prompt_batch_mib"],
                "max_cache_read_mib": kwargs["max_cache_read_mib"],
                "max_cache_write_mib": kwargs["prefill_max_cache_write_mib"],
                "max_runner_scratch_mib": kwargs["max_runner_scratch_mib"],
                "prefill_max_stage_mib": kwargs["prefill_max_stage_mib"],
                "prefill_max_compact_stage_mib": kwargs[
                    "prefill_max_compact_stage_mib"
                ],
                "prefill_expert_stage_align_kib": kwargs[
                    "prefill_expert_stage_align_kib"
                ],
                "prefill_stage_disk_margin_mib": kwargs[
                    "prefill_stage_disk_margin_mib"
                ],
                "dsa_indexer_types": kwargs["dsa_indexer_types"],
                "dsa_index_topk": kwargs["dsa_index_topk"],
                "prefill_expert_stage_tiling": kwargs[
                    "prefill_expert_stage_tiling"
                ],
                "prefill_linear_backend": kwargs["prefill_linear_backend"],
                "prefill_mpsgraph_min_batch_tokens": kwargs[
                    "prefill_mpsgraph_min_batch_tokens"
                ],
                "prefill_mpsgraph_min_matrix_dim": kwargs[
                    "prefill_mpsgraph_min_matrix_dim"
                ],
            }
            max_safe_chunk_plan = _auto_prefill_prompt_chunk_plan(
                **chunk_plan_kwargs,
                tile_tokens=1,
            )
            max_safe_chunk = max_safe_chunk_plan.chunk_tokens
            if configured_chunk <= 0:
                auto_chunk_plan = _auto_prefill_prompt_chunk_plan(**chunk_plan_kwargs)
                resolved_chunk = auto_chunk_plan.chunk_tokens
            elif configured_chunk > max_safe_chunk:
                prefill_chunk_error = (
                    f"prefill_prompt_chunk_tokens {configured_chunk} exceeds "
                    f"safety-capped maximum {max_safe_chunk}"
                )
                resolved_chunk = configured_chunk
            else:
                resolved_chunk = configured_chunk
        prefill_prompt_chunk_plan = (
            {
                "source": "prepared_request_check",
                "configured_is_auto": configured_chunk <= 0,
                "auto": _auto_prefill_prompt_chunk_plan_summary(auto_chunk_plan),
                "max_safe": _auto_prefill_prompt_chunk_plan_summary(
                    max_safe_chunk_plan
                ),
            }
            if kwargs["batch_prefill_prompt"]
            else None
        )
        prefill_prompt_chunk_plan_drift = (
            prefill_prompt_chunk_plan_drift_summary(
                applied_launch_profile=config.applied_launch_profile,
                actual_prompt_chunk_tokens=resolved_chunk,
                actual_auto_plan=auto_chunk_plan,
                actual_max_safe_plan=max_safe_chunk_plan,
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        if prefill_chunk_error is not None:
            raise PreparedRequestCheckError(
                prefill_chunk_error,
                payload={
                    "ok": False,
                    "error": prefill_chunk_error,
                    "prompt_token_count": prompt_token_count,
                    "max_new_tokens": max_new_tokens,
                    "required_context_tokens": prompt_token_count + max_new_tokens,
                    "prompt_token_cap": prompt_cap,
                    "launch_audit_envelope": launch_audit_envelope,
                    "effective_context_tokens": self.state.effective_context_tokens,
                    "batch_prefill_prompt": kwargs["batch_prefill_prompt"],
                    "prefill_prompt_chunk_tokens": {
                        "configured": configured_chunk,
                        "resolved": resolved_chunk,
                        "max_safe": max_safe_chunk,
                    },
                    "prefill_prompt_chunk_plan": prefill_prompt_chunk_plan,
                    "prefill_prompt_chunk_plan_drift": (
                        prefill_prompt_chunk_plan_drift
                    ),
                    "logits_top_k": kwargs["logits_top_k"],
                    "metal_final_logits": kwargs["metal_final_logits"],
                },
            )
        _require_prefill_chunk_plan_profile_admission(
            prefill_prompt_chunk_plan_drift
        )
        runtime_summary: dict[str, Any] = {"ran": False}
        decode_routed_read: dict[str, Any] | None = None
        decode_read_guard_requested = (
            config.decode_max_routed_read_gib_per_token > 0.0
            or config.decode_max_routed_read_seconds_per_token > 0.0
        )
        run_runtime_preflight = runtime_preflight or decode_read_guard_requested
        request_has_decode_work = max_new_tokens > 0
        request_has_prefill_work = (
            bool(kwargs["batch_prefill_prompt"]) and prompt_token_count > 0
        )
        if run_runtime_preflight and (
            request_has_decode_work or request_has_prefill_work
        ):
            extra_live_working_set_bytes = 0
            prefill_live_memory: dict[str, Any] | None = None
            if kwargs["batch_prefill_prompt"]:
                live_estimate = estimate_prompt_prefill_live_memory(
                    max_prompt_batch_mib=kwargs["prefill_max_prompt_batch_mib"],
                    max_runner_scratch_mib=kwargs["max_runner_scratch_mib"],
                    max_cache_read_mib=kwargs["max_cache_read_mib"],
                    max_cache_write_mib=kwargs["prefill_max_cache_write_mib"],
                    copy_chunk_mib=kwargs["prefill_copy_chunk_mib"],
                )
                extra_live_working_set_bytes = (
                    live_estimate.estimated_live_working_set_bytes
                )
                prefill_live_memory = asdict(live_estimate)
            try:
                guard = check_generation_runtime(
                    expert_layout_path=self.state.prepared.experts_layout,
                    resident_layout_path=self.state.prepared.resident_layout,
                    cache_layout_path=self.state.prepared.decode_cache_layout,
                    cache_file_path=self.state.prepared.decode_cache_file,
                    requested_context_tokens=prompt_token_count + max_new_tokens,
                    layers=kwargs["layers"],
                    dense_layers=kwargs["dense_layers"],
                    top_k=kwargs["top_k"],
                    max_k=kwargs["max_k"],
                    num_heads=kwargs["num_heads"],
                    qk_nope_dim=kwargs["qk_nope_dim"],
                    rope_dim=kwargs["rope_dim"],
                    v_head_dim=kwargs["v_head_dim"],
                    include_shared_expert=kwargs["include_shared_expert"],
                    logits_top_k=kwargs["logits_top_k"],
                    logits_chunk_rows=kwargs["logits_chunk_rows"],
                    logits_max_chunk_mib=kwargs["logits_max_chunk_mib"],
                    rms_norm_eps=kwargs["rms_norm_eps"],
                    max_slot_mib=kwargs["max_slot_mib"],
                    max_router_mib=kwargs["max_router_mib"],
                    max_resident_matrix_mib=kwargs["max_resident_matrix_mib"],
                    max_cache_file_mib=kwargs["max_cache_file_mib"],
                    max_cache_read_mib=kwargs["max_cache_read_mib"],
                    max_runner_scratch_mib=kwargs["max_runner_scratch_mib"],
                    cache_dtype_bytes=kwargs["cache_dtype_bytes"],
                    metal_final_logits=kwargs["metal_final_logits"],
                    decode_mla_key_cache=kwargs["decode_mla_key_cache"],
                    allow_tied_embeddings=kwargs["allow_tied_embeddings"],
                    expected_vocab_size=kwargs["expected_vocab_size"],
                    expected_hidden_size=kwargs["expected_hidden_size"],
                    max_embedding_row_mib=kwargs["max_embedding_row_mib"],
                    max_live_working_set_mib=kwargs["max_live_working_set_mib"],
                    min_free_unified_memory_mib=(
                        kwargs["min_free_unified_memory_gib"] * 1024
                    ),
                    extra_live_working_set_bytes=extra_live_working_set_bytes,
                    dsa_indexer_runtime=bool(kwargs["dsa_indexer_types"]),
                    dsa_indexer_types=kwargs["dsa_indexer_types"],
                    dsa_index_topk=kwargs["dsa_index_topk"],
                    dsa_index_head_dim=kwargs["dsa_index_head_dim"],
                    allow_missing_dsa_indexer=kwargs["allow_missing_dsa_indexer"],
                )
            except (GenerationGuardError, RuntimeCheckError) as exc:
                runtime_payload: dict[str, Any] = {
                    "ran": True,
                    "error": str(exc),
                    "required_for_decode_routed_read_guard": (
                        decode_read_guard_requested and not runtime_preflight
                    ),
                    "requested_context_tokens": (
                        prompt_token_count + max_new_tokens
                    ),
                    "prefill_live_memory": prefill_live_memory,
                }
                guard_payload = getattr(exc, "payload", None)
                if isinstance(guard_payload, dict):
                    runtime_payload.update(guard_payload)
                raise PreparedRequestCheckError(
                    str(exc),
                    payload={
                        "ok": False,
                        "error": str(exc),
                        "prompt_token_count": prompt_token_count,
                        "max_new_tokens": max_new_tokens,
                        "required_context_tokens": (
                            prompt_token_count + max_new_tokens
                        ),
                        "prompt_token_cap": prompt_cap,
                        "launch_audit_envelope": launch_audit_envelope,
                        "effective_context_tokens": (
                            self.state.effective_context_tokens
                        ),
                        "batch_prefill_prompt": kwargs["batch_prefill_prompt"],
                        "prefill_prompt_chunk_tokens": {
                            "configured": configured_chunk,
                            "resolved": resolved_chunk,
                            "max_safe": max_safe_chunk,
                        },
                        "prefill_prompt_chunk_plan": prefill_prompt_chunk_plan,
                        "prefill_prompt_chunk_plan_drift": (
                            prefill_prompt_chunk_plan_drift
                        ),
                        "runtime_preflight": runtime_payload,
                    },
                ) from exc
            live_budget = guard.live_memory_budget
            required_available_memory_bytes = (
                live_budget.estimated_live_working_set_bytes
                + live_budget.min_available_memory_bytes
            )
            system_available_memory_bytes = getattr(
                live_budget,
                "system_available_bytes",
                None,
            )
            available_memory_ok = (
                system_available_memory_bytes >= required_available_memory_bytes
                if system_available_memory_bytes is not None
                else None
            )
            runtime_summary = {
                "ran": True,
                "required_for_decode_routed_read_guard": (
                    decode_read_guard_requested and not runtime_preflight
                ),
                "requested_context_tokens": guard.requested_context_tokens,
                "layers": guard.layers,
                "dense_layers": guard.dense_layers,
                "max_layer_peak_bytes": guard.max_layer_peak_bytes,
                "max_layer_cache_read_bytes": guard.max_layer_cache_read_bytes,
                "read_bytes_per_token": guard.read_bytes_per_token,
                "final_logits_peak_bytes": (
                    guard.final_logits_budget.estimated_peak_bytes
                ),
                "embedding_row_bytes": guard.embedding_budget.row_bytes,
                "embedding_output_bytes": guard.embedding_budget.output_bytes,
                "live_working_set_bytes": (
                    live_budget.estimated_live_working_set_bytes
                ),
                "resident_backing_bytes": getattr(
                    live_budget,
                    "resident_backing_bytes",
                    None,
                ),
                "nonresident_peak_bytes": getattr(
                    live_budget,
                    "nonresident_peak_bytes",
                    None,
                ),
                "extra_live_working_set_bytes": getattr(
                    live_budget,
                    "extra_live_working_set_bytes",
                    None,
                ),
                "max_live_working_set_bytes": (
                    live_budget.max_live_working_set_bytes
                ),
                "min_available_memory_bytes": (
                    live_budget.min_available_memory_bytes
                ),
                "required_available_memory_bytes": required_available_memory_bytes,
                "system_available_memory_bytes": system_available_memory_bytes,
                "system_total_memory_bytes": getattr(
                    live_budget,
                    "system_total_bytes",
                    None,
                ),
                "system_memory_source": getattr(live_budget, "system_source", None),
                "available_memory_ok": available_memory_ok,
                "prefill_live_memory": prefill_live_memory,
            }
            if request_has_decode_work:
                decode_routed_read = _decode_routed_read_request_summary(
                    read_bytes_per_token=guard.read_bytes_per_token,
                    max_read_gib_per_token=(
                        config.decode_max_routed_read_gib_per_token
                    ),
                    ssd_read_gib_per_second=config.prefill_ssd_read_gib_per_second,
                    max_read_seconds_per_token=(
                        config.decode_max_routed_read_seconds_per_token
                    ),
                )
                if decode_routed_read.get("within_limit") is False:
                    if decode_routed_read.get("within_read_limit") is False:
                        limit_label = (
                            "read "
                            f"{decode_routed_read['read_bytes_per_token']} "
                            "bytes/token exceeds cap "
                            f"{decode_routed_read['max_read_bytes_per_token']} "
                            "bytes/token"
                        )
                    else:
                        limit_label = (
                            "read time "
                            f"{decode_routed_read['planned_read_seconds_per_token']:.3g}"
                            "s/token exceeds cap "
                            f"{decode_routed_read['max_read_seconds_per_token']:.3g}"
                            "s/token"
                        )
                    raise PreparedServerError(
                        "decode routed expert "
                        f"{limit_label}"
                    )
        elif run_runtime_preflight:
            runtime_summary = {
                "ran": False,
                "reason": "no decode or batch prefill work",
                "required_for_decode_routed_read_guard": (
                    decode_read_guard_requested and not runtime_preflight
                ),
            }
        prefill_cache_io = (
            _prefill_cache_io_request_summary(
                cfg=self.state.model_config,
                prompt_token_count=prompt_token_count,
                dtype_bytes=kwargs["cache_dtype_bytes"],
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        prefill_routed_read = (
            _prefill_routed_read_request_summary(
                expert_layout_path=self.state.prepared.experts_layout,
                prompt_token_count=prompt_token_count,
                prompt_chunk_tokens=resolved_chunk,
                top_k=kwargs["top_k"],
                layers=kwargs["layers"],
                max_read_amplification=kwargs[
                    "prefill_max_routed_read_amplification"
                ],
                max_planned_read_bytes=int(
                    kwargs["prefill_max_routed_read_gib"] * 1024**3
                ),
                ssd_read_gib_per_second=kwargs["prefill_ssd_read_gib_per_second"],
                max_read_seconds=kwargs["prefill_max_routed_read_seconds"],
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        prefill_routed_stage_temp = (
            _prefill_routed_stage_temp_request_summary(
                expert_layout_path=self.state.prepared.experts_layout,
                prompt_token_count=prompt_token_count,
                prompt_chunk_tokens=resolved_chunk,
                top_k=kwargs["top_k"],
                layers=kwargs["layers"],
                stage_align_bytes=int(kwargs["prefill_expert_stage_align_kib"] * 1024),
                static_capacity_per_expert=kwargs[
                    "prefill_static_capacity_per_expert"
                ],
                allow_static_capacity_overflow=kwargs[
                    "prefill_allow_static_capacity_overflow"
                ],
                max_stage_bytes=int(kwargs["prefill_max_stage_mib"] * 1024**2),
                max_compact_stage_bytes=int(
                    kwargs["prefill_max_compact_stage_mib"] * 1024**2
                ),
                max_stage_raw_ranges=kwargs["prefill_max_stage_raw_ranges"],
                max_stage_coalesced_ranges=(
                    kwargs["prefill_max_stage_coalesced_ranges"]
                ),
                expert_stage_tiling=kwargs["prefill_expert_stage_tiling"],
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        prefill_stage_temp_disk_free = (
            _prefill_stage_temp_disk_free_request_summary(
                stage_temp=prefill_routed_stage_temp,
                disk_safety_margin_bytes=int(
                    kwargs["prefill_stage_disk_margin_mib"] * 1024**2
                ),
                temp_dir=(
                    Path(kwargs["work_dir"])
                    if kwargs["work_dir"] is not None
                    else Path("/private/tmp")
                ),
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        prefill_routed_chunk_frontier = (
            _prefill_routed_chunk_frontier_request_summary(
                expert_layout_path=self.state.prepared.experts_layout,
                prompt_token_count=prompt_token_count,
                prompt_chunk_tokens=resolved_chunk,
                max_safe_chunk_tokens=max_safe_chunk,
                top_k=kwargs["top_k"],
                layers=kwargs["layers"],
                stage_align_bytes=int(kwargs["prefill_expert_stage_align_kib"] * 1024),
                ssd_read_gib_per_second=kwargs["prefill_ssd_read_gib_per_second"],
                static_capacity_per_expert=kwargs[
                    "prefill_static_capacity_per_expert"
                ],
                allow_static_capacity_overflow=kwargs[
                    "prefill_allow_static_capacity_overflow"
                ],
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        if (
            isinstance(prefill_routed_stage_temp, dict)
            and prefill_routed_stage_temp.get("analyzed")
            and prefill_routed_stage_temp.get("within_limit") is False
        ):
            if prefill_routed_stage_temp.get("within_stage_limit") is False:
                stage_bytes = prefill_routed_stage_temp.get(
                    "effective_max_stage_bytes",
                    prefill_routed_stage_temp.get("max_stage_bytes"),
                )
                limit_label = (
                    "stage "
                    f"{stage_bytes} bytes "
                    "exceeds cap "
                    f"{prefill_routed_stage_temp['max_stage_limit_bytes']} bytes"
                )
            elif prefill_routed_stage_temp.get("within_compact_stage_limit") is False:
                compact_bytes = prefill_routed_stage_temp.get(
                    "effective_max_compact_stage_bytes",
                    prefill_routed_stage_temp.get("max_compact_stage_bytes"),
                )
                limit_label = (
                    "compact stage "
                    f"{compact_bytes} bytes "
                    "exceeds cap "
                    f"{prefill_routed_stage_temp['max_compact_stage_limit_bytes']} bytes"
                )
            else:
                limit_label = "stage or compact stage exceeds configured cap"
            raise PreparedServerError(
                "prompt prefill routed stage temp "
                f"{limit_label}"
            )
        if isinstance(prefill_stage_temp_disk_free, dict):
            if prefill_stage_temp_disk_free.get("within_free_space") is False:
                raise PreparedServerError(
                    "prompt prefill temp disk free space is below routed stage "
                    "requirement: "
                    f"required={prefill_stage_temp_disk_free['required_free_bytes']} "
                    f"bytes free={prefill_stage_temp_disk_free.get('free_bytes')} "
                    f"bytes path={prefill_stage_temp_disk_free['path']}"
                )
            if prefill_stage_temp_disk_free.get("within_free_space") is None:
                raise PreparedServerError(
                    "could not verify prompt prefill temp disk free space for "
                    "routed stage requirement"
                )
        if (
            isinstance(prefill_routed_read, dict)
            and prefill_routed_read.get("analyzed")
            and prefill_routed_read.get("within_limit") is False
        ):
            if prefill_routed_read.get("within_amplification_limit") is False:
                limit_label = (
                    "read amplification "
                    f"{prefill_routed_read['read_amplification']:.3g} exceeds cap "
                    f"{prefill_routed_read['max_read_amplification']:.3g}"
                )
            elif prefill_routed_read.get("within_planned_read_limit") is False:
                limit_label = (
                    "planned read "
                    f"{prefill_routed_read['planned_read_bytes']} bytes exceeds cap "
                    f"{prefill_routed_read['max_planned_read_bytes']} bytes"
                )
            else:
                limit_label = (
                    "planned read time "
                    f"{prefill_routed_read['planned_read_seconds']:.3g}s exceeds cap "
                    f"{prefill_routed_read['max_read_seconds']:.3g}s"
                )
            minimum_chunk = prefill_routed_read.get(
                "minimum_chunk_tokens_for_limits"
            )
            if isinstance(minimum_chunk, int):
                suggestion = (
                    f"; use prefill_prompt_chunk_tokens>={minimum_chunk} "
                    "or relax the cap"
                )
            else:
                suggestion = (
                    "; no prompt chunk size can satisfy these routed-read caps"
                )
            raise PreparedServerError(
                "prefill routed expert "
                f"{limit_label}; "
                f"planned_read_bytes={prefill_routed_read['planned_read_bytes']} "
                f"baseline_read_bytes={prefill_routed_read['baseline_read_bytes']} "
                f"extra_read_bytes={prefill_routed_read['extra_read_bytes']}"
                f"{suggestion}"
            )
        streamed_routed_hidden_dim, streamed_routed_moe_hidden_dims = (
            _streamed_routed_expert_request_dims(
                self.state.model_config,
                prefill_routed_read,
            )
        )
        prefill_linear_backend = _prefill_linear_backend_request_summary(
            resident_layout_path=self.state.prepared.resident_layout,
            configured_backend=self.state.config.prefill_linear_backend,
            effective_backend=kwargs["prefill_linear_backend"],
            prompt_chunk_tokens=resolved_chunk,
            mpsgraph_min_batch_tokens=kwargs["prefill_mpsgraph_min_batch_tokens"],
            mpsgraph_min_matrix_dim=kwargs["prefill_mpsgraph_min_matrix_dim"],
            streamed_routed_expert_hidden_dim=streamed_routed_hidden_dim,
            streamed_routed_expert_top_k=kwargs["top_k"],
            streamed_routed_expert_moe_hidden_dims=streamed_routed_moe_hidden_dims,
        )
        prefill_acceleration_requested = (
            self.state.config.require_prefill_acceleration
            or self.state.config.prefill_min_accelerated_flop_fraction > 0.0
        )
        prefill_acceleration_required_for_request = (
            prefill_acceleration_requested and bool(kwargs["batch_prefill_prompt"])
        )
        prefill_acceleration_coverage = _prefill_acceleration_coverage_summary(
            prefill_linear_backend,
            required=prefill_acceleration_required_for_request,
            min_accelerated_flop_fraction=(
                self.state.config.prefill_min_accelerated_flop_fraction
            ),
            allow_router_gate_only_acceleration=(
                self.state.config.allow_router_gate_only_prefill_acceleration
            ),
        )
        prefill_acceleration_frontier = (
            _prefill_acceleration_frontier_request_summary(
                resident_layout_path=self.state.prepared.resident_layout,
                configured_backend=self.state.config.prefill_linear_backend,
                effective_backend=kwargs["prefill_linear_backend"],
                prompt_token_count=prompt_token_count,
                prompt_chunk_tokens=resolved_chunk,
                max_safe_chunk_tokens=max_safe_chunk,
                mpsgraph_min_batch_tokens=kwargs[
                    "prefill_mpsgraph_min_batch_tokens"
                ],
                mpsgraph_min_matrix_dim=kwargs["prefill_mpsgraph_min_matrix_dim"],
                require_prefill_acceleration=prefill_acceleration_required_for_request,
                min_accelerated_flop_fraction=(
                    self.state.config.prefill_min_accelerated_flop_fraction
                ),
                streamed_routed_expert_hidden_dim=streamed_routed_hidden_dim,
                streamed_routed_expert_top_k=kwargs["top_k"],
                streamed_routed_expert_moe_hidden_dims=(
                    streamed_routed_moe_hidden_dims
                ),
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        if (
            prefill_acceleration_required_for_request
            and prefill_acceleration_coverage.get("ok") is not True
        ):
            suggestion = ""
            if (
                prefill_acceleration_coverage.get(
                    "f32_backend_has_unsupported_matrices"
                )
                is not True
                and isinstance(prefill_acceleration_frontier, dict)
            ):
                suggested = prefill_acceleration_frontier.get("suggested_guard_flags")
                if isinstance(suggested, dict):
                    minimum_chunk = suggested.get("prefill_prompt_chunk_tokens")
                    if isinstance(minimum_chunk, int):
                        suggestion = (
                            f"; use prefill_prompt_chunk_tokens>={minimum_chunk}"
                        )
            raise PreparedRequestCheckError(
                "prefill acceleration coverage failed: "
                f"{prefill_acceleration_coverage.get('reason')}"
                f"{suggestion}",
                payload={
                    "prefill_prompt_chunk_tokens": {
                        "configured": configured_chunk,
                        "resolved": resolved_chunk,
                        "max_safe": max_safe_chunk,
                    },
                    "prefill_prompt_chunk_plan": prefill_prompt_chunk_plan,
                    "prefill_prompt_chunk_plan_drift": (
                        prefill_prompt_chunk_plan_drift
                    ),
                    "prefill_linear_backend": prefill_linear_backend,
                    "prefill_acceleration_coverage": prefill_acceleration_coverage,
                    "prefill_acceleration_frontier": prefill_acceleration_frontier,
                },
            )
        suggested_guard_flags = _prefill_routed_guard_flag_suggestions(
            routed_read=prefill_routed_read,
            prompt_chunk_tokens=resolved_chunk,
        )
        suggested_stage_temp_guard_flags = (
            _prefill_routed_stage_guard_flag_suggestions(
                stage_temp=prefill_routed_stage_temp,
                prompt_chunk_tokens=resolved_chunk,
            )
        )
        suggested_cache_io_guard_flags = (
            _prefill_cache_io_guard_flag_suggestions(
                max_cache_read_mib=kwargs["max_cache_read_mib"],
                prefill_max_cache_write_mib=kwargs["prefill_max_cache_write_mib"],
                source="prepared_request_check",
            )
            if kwargs["batch_prefill_prompt"]
            else None
        )
        suggested_prefill_guard_flags = combine_prefill_guard_flags(
            routed_read_flags=suggested_guard_flags,
            stage_temp_flags=suggested_stage_temp_guard_flags,
            cache_io_flags=suggested_cache_io_guard_flags,
            source="prepared_request_check",
        )

        return {
            "ok": True,
            "prompt_token_count": prompt_token_count,
            "max_new_tokens": max_new_tokens,
            "required_context_tokens": prompt_token_count + max_new_tokens,
            "prompt_token_cap": prompt_cap,
            "launch_audit_envelope": launch_audit_envelope,
            "effective_context_tokens": self.state.effective_context_tokens,
            "batch_prefill_prompt": kwargs["batch_prefill_prompt"],
            "prefill_prompt_chunk_tokens": {
                "configured": configured_chunk,
                "resolved": resolved_chunk,
                "max_safe": max_safe_chunk,
            },
            "prefill_prompt_chunk_plan": prefill_prompt_chunk_plan,
            "prefill_prompt_chunk_plan_drift": prefill_prompt_chunk_plan_drift,
            "prefill_expert_stage_tiling": kwargs["prefill_expert_stage_tiling"],
            "prefill_persistent_moe_plan_server": kwargs[
                "prefill_persistent_moe_plan_server"
            ],
            "prefill_persistent_resident_linear_server": kwargs[
                "prefill_persistent_resident_linear_server"
            ],
            "prefill_persistent_attention_projection_server": kwargs[
                "prefill_persistent_attention_projection_server"
            ],
            "prefill_persistent_attention_output_server": kwargs[
                "prefill_persistent_attention_output_server"
            ],
            "prefill_persistent_shared_expert_server": kwargs[
                "prefill_persistent_shared_expert_server"
            ],
            "prefill_persistent_rope_split_server": kwargs[
                "prefill_persistent_rope_split_server"
            ],
            "prefill_persistent_mla_attention_server": kwargs[
                "prefill_persistent_mla_attention_server"
            ],
            "prefill_persistent_rmsnorm_server": kwargs[
                "prefill_persistent_rmsnorm_server"
            ],
            "prefill_moe_output_accumulator": kwargs[
                "prefill_moe_output_accumulator"
            ],
            "logits_top_k": kwargs["logits_top_k"],
            "metal_final_logits": kwargs["metal_final_logits"],
            "prefill_mla_kv_b_cache_dir": (
                str(kwargs["prefill_mla_kv_b_cache_dir"])
                if kwargs["prefill_mla_kv_b_cache_dir"] is not None
                else None
            ),
            "prefill_linear_backend": prefill_linear_backend,
            "prefill_router_hybrid_margin_threshold": kwargs[
                "prefill_router_hybrid_margin_threshold"
            ],
            "decode_mla_key_cache": kwargs["decode_mla_key_cache"],
            "prefill_acceleration_coverage": prefill_acceleration_coverage,
            "prefill_acceleration_frontier": prefill_acceleration_frontier,
            "prefill_cache_io": prefill_cache_io,
            "prefill_routed_expert_read": prefill_routed_read,
            "prefill_routed_stage_temp_disk": prefill_routed_stage_temp,
            "prefill_stage_temp_disk_free": prefill_stage_temp_disk_free,
            "prefill_routed_chunk_frontier": prefill_routed_chunk_frontier,
            "decode_routed_expert_read": decode_routed_read,
            "suggested_decode_guard_flags": (
                _with_decode_mla_key_cache_guard_flag(
                    suggest_decode_routed_read_guard_flags(
                        read_bytes_per_token=decode_routed_read.get(
                            "read_bytes_per_token"
                        ),
                        ssd_read_gib_per_second=decode_routed_read.get(
                            "ssd_read_gib_per_second"
                        ),
                        source="prepared_request_check",
                    ),
                    source="prepared_request_check",
                    decode_mla_key_cache=kwargs["decode_mla_key_cache"],
                )
                if isinstance(decode_routed_read, dict)
                else None
            ),
            "suggested_guard_flags": suggested_guard_flags,
            "suggested_stage_temp_guard_flags": suggested_stage_temp_guard_flags,
            "suggested_cache_io_guard_flags": suggested_cache_io_guard_flags,
            "suggested_prefill_guard_flags": suggested_prefill_guard_flags,
            "runtime_preflight": runtime_summary,
        }

    def openai_models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.state.config.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "largerlm",
                }
            ],
        }

    def generate_token_ids(self, payload: dict[str, Any]) -> dict[str, Any]:
        _require_prepared_runtime_profile_for_generation(
            self.state.prepared,
            require_memory_profile=self.state.config.require_prepared_memory_profile,
        )
        max_new_tokens = _max_new_tokens(payload, self.state.config.max_new_tokens_cap)
        prompt_cap = _request_prompt_token_cap(
            self.state,
            max_new_tokens=max_new_tokens,
        )
        prompt = _prompt_token_ids(
            payload,
            max_prompt_tokens=prompt_cap,
        )
        kwargs = _base_generation_kwargs(
            self.state,
            payload=payload,
            prompt_token_count=len(prompt),
        )
        generation_overrides = (
            {"batch_prefill_prompt": False}
            if self.state.config.metal_runtime_generation
            else None
        )
        request_check = self.inspect_token_request(
            prompt_token_count=len(prompt),
            payload=payload,
            runtime_preflight=True,
            generation_overrides=generation_overrides,
        )
        require_prepared_request_check_ok(request_check)
        if self.state.config.metal_runtime_generation:
            _metal_runtime_sampling_kwargs(payload)
            if max_new_tokens <= 0:
                raise PreparedServerError(
                    "metal_runtime_generation requires max_new_tokens >= 1"
                )
            request_check = dict(request_check)
            request_check["metal_runtime_generation"] = True
            request_check["batch_prefill_prompt"] = False
            with self.state.lock:
                prepared_lock = _acquire_server_prepared_generation_lock(
                    self.state.prepared
                )
                try:
                    try:
                        metal_result = generate_metal_token_ids(
                            self.state.prepared.manifest_path.parent,
                            prompt_token_ids=prompt,
                            max_new_tokens=max_new_tokens,
                            binary=self.state.config.metal_binary_path,
                            expert_pin_plan=(
                                self.state.config.metal_runtime_expert_pin_plan
                            ),
                            max_adaptive_expert_cache_gib=(
                                self.state.config.metal_runtime_max_adaptive_expert_cache_gib
                            ),
                            top_k=kwargs["top_k"],
                            logits_top_k=int(kwargs["top_k"]),
                            max_live_working_set_mib=(
                                _metal_generation_live_cap_mib(self.state.config)
                            ),
                            max_cache_file_mib=self.state.config.max_cache_file_mib,
                            max_cache_read_mib=self.state.config.max_cache_read_mib,
                            logits_max_chunk_mib=kwargs["logits_max_chunk_mib"],
                            mmap_final_logits=(
                                self.state.config.metal_runtime_mmap_final_logits
                            ),
                            max_embedding_row_mib=kwargs["max_embedding_row_mib"],
                            cache_mla_kv_b_f32=(
                                self.state.config.metal_runtime_cache_mla_kv_b_f32
                            ),
                            max_mla_kv_b_cache_mib=(
                                self.state.config.metal_runtime_max_mla_kv_b_cache_mib
                            ),
                            context1_o_proj_cache_layout=(
                                self.state.config.metal_runtime_context1_o_proj_cache_layout
                            ),
                            context1_o_proj_cache_file=(
                                self.state.config.metal_runtime_context1_o_proj_cache_file
                            ),
                            prefill_prompt=len(prompt) > 1,
                            min_free_unified_memory_gib=(
                                self.state.config.min_free_unified_memory_gib
                            ),
                            use_generate_server_jsonl=True,
                            generate_server_session=self._metal_runtime_session(),
                            quiet=not self.state.config.echo_runner_output,
                        )
                    except MetalGenerateError as exc:
                        self._reset_metal_runtime_session()
                        raise PreparedServerError(str(exc)) from exc
                finally:
                    prepared_lock.close()
            return _metal_token_result_payload(
                metal_result,
                applied_launch_profile=self.state.config.applied_launch_profile,
                request_check=request_check,
                launch_audit_envelope=self.state.config.launch_audit_envelope,
            )
        with self.state.lock:
            prepared_lock = _acquire_server_prepared_generation_lock(
                self.state.prepared
            )
            try:
                result = generate_token_ids(
                    runner_path=self.state.config.runner_path,
                    expert_layout_path=self.state.prepared.experts_layout,
                    resident_layout_path=self.state.prepared.resident_layout,
                    cache_layout_path=self.state.prepared.decode_cache_layout,
                    cache_file_path=self.state.prepared.decode_cache_file,
                    prompt_token_ids=prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=None,
                    **kwargs,
                )
            finally:
                prepared_lock.close()
        if (
            self.state.config.require_prefill_acceleration
            or self.state.config.prefill_min_accelerated_flop_fraction > 0.0
        ) and result.prompt_prefill is not None:
            _require_actual_prefill_acceleration(
                result,
                min_accelerated_flop_fraction=(
                    self.state.config.prefill_min_accelerated_flop_fraction
                ),
                allow_router_gate_only_acceleration=(
                    self.state.config.allow_router_gate_only_prefill_acceleration
                ),
            )
        return _token_result_payload(
            result,
            applied_launch_profile=self.state.config.applied_launch_profile,
            request_check=request_check,
            launch_audit_envelope=self.state.config.launch_audit_envelope,
        )

    def generate_text(self, payload: dict[str, Any]) -> dict[str, Any]:
        _require_prepared_runtime_profile_for_generation(
            self.state.prepared,
            require_memory_profile=self.state.config.require_prepared_memory_profile,
        )
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise PreparedServerError("prompt must be a non-empty string")
        max_new_tokens = _max_new_tokens(payload, self.state.config.max_new_tokens_cap)
        prompt_cap = _request_prompt_token_cap(
            self.state,
            max_new_tokens=max_new_tokens,
        )
        tokenizer_path = self.state.config.tokenizer_path or self.state.prepared.model_dir
        add_special_tokens = _bool_payload(payload, "add_special_tokens", True)
        skip_special_tokens = _bool_payload(payload, "skip_special_tokens", True)
        try:
            tokenizer = load_tokenizer(
                tokenizer_path,
                backend=self.state.config.tokenizer_backend,
                trust_remote_code=self.state.config.trust_remote_code,
            )
            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=add_special_tokens,
            )
        except TokenizerError as exc:
            raise PreparedServerError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - optional tokenizer backends
            raise PreparedServerError(f"tokenizer failed to encode prompt: {exc}") from exc
        if not prompt_ids:
            raise PreparedServerError("prompt encoded to zero tokens")
        kwargs = _base_generation_kwargs(
            self.state,
            payload=payload,
            prompt_token_count=len(prompt_ids),
        )
        auto_batch_prefill = _batch_prefill_requested(self.state.config, payload)
        kwargs["auto_batch_prefill_prompt"] = auto_batch_prefill
        if auto_batch_prefill:
            kwargs["prefill_static_capacity_per_expert"] = (
                self.state.config.prefill_static_capacity_per_expert
            )
        request_check = self.inspect_token_request(
            prompt_token_count=len(prompt_ids),
            payload=payload,
            runtime_preflight=True,
            generation_overrides=(
                {"batch_prefill_prompt": False}
                if self.state.config.metal_runtime_generation
                else None
            ),
        )
        require_prepared_request_check_ok(request_check)
        if self.state.config.metal_runtime_generation:
            _metal_runtime_sampling_kwargs(payload)
            if max_new_tokens <= 0:
                raise PreparedServerError(
                    "metal_runtime_generation requires max_new_tokens >= 1"
                )
            request_check = dict(request_check)
            request_check["metal_runtime_generation"] = True
            request_check["batch_prefill_prompt"] = False
            with self.state.lock:
                prepared_lock = _acquire_server_prepared_generation_lock(
                    self.state.prepared
                )
                try:
                    try:
                        metal_text_result: MetalTextGenerationResult = (
                            generate_metal_text(
                                prepared_dir=self.state.prepared.manifest_path.parent,
                                tokenizer_path=tokenizer_path,
                                tokenizer_backend=self.state.config.tokenizer_backend,
                                trust_remote_code=(
                                    self.state.config.trust_remote_code
                                ),
                                prompt=prompt,
                                max_new_tokens=max_new_tokens,
                                add_special_tokens=add_special_tokens,
                                skip_special_tokens=skip_special_tokens,
                                max_prompt_tokens=prompt_cap,
                                binary=self.state.config.metal_binary_path,
                                expert_pin_plan=(
                                    self.state.config.metal_runtime_expert_pin_plan
                                ),
                                max_adaptive_expert_cache_gib=(
                                    self.state.config.metal_runtime_max_adaptive_expert_cache_gib
                                ),
                                top_k=kwargs["top_k"],
                                logits_top_k=int(kwargs["top_k"]),
                                max_live_working_set_mib=(
                                    _metal_generation_live_cap_mib(
                                        self.state.config
                                    )
                                ),
                                max_cache_file_mib=(
                                    self.state.config.max_cache_file_mib
                                ),
                                max_cache_read_mib=(
                                    self.state.config.max_cache_read_mib
                                ),
                                logits_max_chunk_mib=kwargs[
                                    "logits_max_chunk_mib"
                                ],
                                mmap_final_logits=(
                                    self.state.config.metal_runtime_mmap_final_logits
                                ),
                                max_embedding_row_mib=kwargs[
                                    "max_embedding_row_mib"
                                ],
                                cache_mla_kv_b_f32=(
                                    self.state.config.metal_runtime_cache_mla_kv_b_f32
                                ),
                                max_mla_kv_b_cache_mib=(
                                    self.state.config.metal_runtime_max_mla_kv_b_cache_mib
                                ),
                                context1_o_proj_cache_layout=(
                                    self.state.config.metal_runtime_context1_o_proj_cache_layout
                                ),
                                context1_o_proj_cache_file=(
                                    self.state.config.metal_runtime_context1_o_proj_cache_file
                                ),
                                min_free_unified_memory_gib=(
                                    self.state.config.min_free_unified_memory_gib
                                ),
                                use_generate_server_jsonl=True,
                                generate_server_session=self._metal_runtime_session(),
                                quiet=not self.state.config.echo_runner_output,
                            )
                        )
                    except (MetalGenerateError, MetalTextGenerationError) as exc:
                        if isinstance(exc, MetalGenerateError):
                            self._reset_metal_runtime_session()
                        raise PreparedServerError(str(exc)) from exc
                finally:
                    prepared_lock.close()
            token_payload = _metal_token_result_payload(
                metal_text_result.token_result,
                applied_launch_profile=self.state.config.applied_launch_profile,
                request_check=request_check,
                launch_audit_envelope=self.state.config.launch_audit_envelope,
            )
            payload = {
                "runtime": "glm_moe_infer",
                "prompt": metal_text_result.prompt,
                "generated_text": metal_text_result.generated_text,
                "full_text": metal_text_result.full_text,
                "tokenizer_backend": metal_text_result.tokenizer_backend,
                "prompt_token_ids": list(metal_text_result.prompt_token_ids),
                "generated_token_ids": list(metal_text_result.generated_token_ids),
                "token_result": token_payload,
                "request_check": request_check,
            }
            if self.state.config.launch_audit_envelope is not None:
                payload["launch_audit_envelope"] = (
                    self.state.config.launch_audit_envelope
                )
            return _attach_applied_launch_profile(
                payload,
                self.state.config.applied_launch_profile,
            )
        with self.state.lock:
            prepared_lock = _acquire_server_prepared_generation_lock(
                self.state.prepared
            )
            try:
                result: TextGenerationResult = generate_text(
                    tokenizer_path=tokenizer_path,
                    tokenizer_backend=self.state.config.tokenizer_backend,
                    trust_remote_code=self.state.config.trust_remote_code,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    max_prompt_tokens=prompt_cap,
                    runner_path=self.state.config.runner_path,
                    expert_layout_path=self.state.prepared.experts_layout,
                    resident_layout_path=self.state.prepared.resident_layout,
                    cache_layout_path=self.state.prepared.decode_cache_layout,
                    cache_file_path=self.state.prepared.decode_cache_file,
                    add_special_tokens=add_special_tokens,
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
            finally:
                prepared_lock.close()
        if (
            self.state.config.require_prefill_acceleration
            or self.state.config.prefill_min_accelerated_flop_fraction > 0.0
        ) and result.token_result.prompt_prefill is not None:
            _require_actual_prefill_acceleration(
                result.token_result,
                min_accelerated_flop_fraction=(
                    self.state.config.prefill_min_accelerated_flop_fraction
                ),
                allow_router_gate_only_acceleration=(
                    self.state.config.allow_router_gate_only_prefill_acceleration
                ),
            )
        payload = {
            "prompt": result.prompt,
            "generated_text": result.generated_text,
            "full_text": result.full_text,
            "tokenizer_backend": result.tokenizer_backend,
            "prompt_token_ids": list(result.prompt_token_ids),
            "generated_token_ids": list(result.generated_token_ids),
            "token_result": _token_result_payload(
                result.token_result,
                applied_launch_profile=self.state.config.applied_launch_profile,
                request_check=request_check,
                launch_audit_envelope=self.state.config.launch_audit_envelope,
            ),
        }
        payload["request_check"] = request_check
        if self.state.config.launch_audit_envelope is not None:
            payload["launch_audit_envelope"] = self.state.config.launch_audit_envelope
        return _attach_applied_launch_profile(
            payload,
            self.state.config.applied_launch_profile,
        )

    def openai_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        _reject_openai_completion_unsupported(payload)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise PreparedServerError("prompt must be a non-empty string")
        model = _openai_model_name(
            payload,
            default=self.state.config.served_model_name,
        )
        max_tokens = _openai_max_tokens(payload, self.state.config.max_new_tokens_cap)
        echo = _bool_payload(payload, "echo", False)
        text_result = self.generate_text(
            {
                **payload,
                "prompt": prompt,
                "max_new_tokens": max_tokens,
            }
        )
        generated_ids = text_result["generated_token_ids"]
        prompt_ids = text_result["prompt_token_ids"]
        finish_reason = "stop" if len(generated_ids) < max_tokens else "length"
        choice_text = text_result["full_text"] if echo else text_result["generated_text"]
        return {
            "id": f"cmpl-largerlm-{time.time_ns()}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": choice_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
            "largerlm": {
                "tokenizer_backend": text_result["tokenizer_backend"],
                "token_result": text_result["token_result"],
                "applied_launch_profile": text_result.get("applied_launch_profile"),
                "request_check": text_result.get("request_check"),
                "launch_audit_envelope": text_result.get("launch_audit_envelope"),
            },
        }

    def openai_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        _reject_openai_chat_unsupported(payload)
        messages = _openai_chat_messages(payload)
        model = _openai_model_name(
            payload,
            default=self.state.config.served_model_name,
        )
        max_tokens = _openai_max_tokens(payload, self.state.config.max_new_tokens_cap)
        tokenizer_path = self.state.config.tokenizer_path or self.state.prepared.model_dir
        try:
            rendered = render_chat_prompt(
                tokenizer_path,
                messages,
                backend=self.state.config.tokenizer_backend,
                trust_remote_code=self.state.config.trust_remote_code,
                add_generation_prompt=_bool_payload(payload, "add_generation_prompt", True),
            )
        except TokenizerError as exc:
            raise PreparedServerError(str(exc)) from exc
        text_result = self.generate_text(
            {
                **payload,
                "prompt": rendered.text,
                "max_new_tokens": max_tokens,
                "add_special_tokens": False,
            }
        )
        generated_ids = text_result["generated_token_ids"]
        prompt_ids = text_result["prompt_token_ids"]
        finish_reason = "stop" if len(generated_ids) < max_tokens else "length"
        return {
            "id": f"chatcmpl-largerlm-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_result["generated_text"],
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
            "largerlm": {
                "chat_template_backend": rendered.backend,
                "chat_template_tokenizer_path": str(rendered.tokenizer_path),
                "tokenizer_backend": text_result["tokenizer_backend"],
                "token_result": text_result["token_result"],
                "applied_launch_profile": text_result.get("applied_launch_profile"),
                "request_check": text_result.get("request_check"),
                "launch_audit_envelope": text_result.get("launch_audit_envelope"),
            },
        }


class _PreparedHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], app: PreparedGenerationApp):
        self.app = app
        super().__init__(address, _PreparedRequestHandler)

    def server_close(self) -> None:
        try:
            app = getattr(self, "app", None)
            if app is not None:
                app.close()
        finally:
            super().server_close()


class _PreparedRequestHandler(BaseHTTPRequestHandler):
    server: _PreparedHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if content_type and "application/json" not in content_type:
            raise PreparedServerError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise PreparedServerError("Content-Length must be an integer") from exc
        if length <= 0:
            raise PreparedServerError("request body must be non-empty")
        if length > self.server.app.state.config.max_request_bytes:
            raise PreparedServerError("request body exceeds server max_request_bytes")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreparedServerError(f"invalid JSON request: {exc}") from exc
        if not isinstance(payload, dict):
            raise PreparedServerError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, self.server.app.health())
            return
        if self.path == "/v1/models":
            self._send_json(HTTPStatus.OK, self.server.app.openai_models())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/generate-token-ids":
                result = self.server.app.generate_token_ids(payload)
            elif self.path == "/generate-text":
                result = self.server.app.generate_text(payload)
            elif self.path == "/v1/completions":
                result = self.server.app.openai_completion(payload)
            elif self.path == "/v1/chat/completions":
                result = self.server.app.openai_chat_completion(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
        except PreparedRequestCheckError as exc:
            payload: dict[str, Any] = {"error": str(exc)}
            if exc.payload:
                payload["request_check"] = exc.payload
            self._send_json(HTTPStatus.BAD_REQUEST, payload)
            return
        except (PreparedServerError, TextGenerationError, TokenGeneratorError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, result)


def run_prepared_server(config: PreparedServerConfig) -> None:
    app = PreparedGenerationApp(config)
    httpd = _PreparedHTTPServer((config.host, int(config.port)), app)
    print(
        f"LargerLM prepared server listening on http://{config.host}:{httpd.server_port}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
