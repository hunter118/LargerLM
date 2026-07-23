from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .prepared import PreparedManifest, load_prepared_manifest
from .prepared_lock import (
    PreparedRunLock,
    PreparedRunLockError,
    acquire_prepared_run_lock_path,
    prepared_run_lock_already_held,
    prepared_run_lock_path_for_manifest,
)
from .prompt_prefill import prompt_prefill_acceleration_failure_reason
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
from .prefill_plan import DEFAULT_MPP_MIN_TOKENS, MPP_TENSOR_OPS_MIN_MATRIX_DIM
from .routed_read import (
    RoutedExpertReadError,
    combine_prefill_guard_flags,
    estimate_routed_prefill_chunk_frontier,
    suggest_decode_routed_read_guard_flags,
    suggest_routed_read_guard_flags,
    suggest_routed_stage_temp_guard_flags,
)
from .token_generator import TokenGenerationResult, generate_token_ids


class BenchmarkError(RuntimeError):
    """Raised when a prepared benchmark cannot run."""


def _acquire_benchmark_prepared_generation_lock(
    prepared: PreparedManifest,
) -> PreparedRunLock | None:
    path = prepared_run_lock_path_for_manifest(prepared.manifest_path)
    if prepared_run_lock_already_held(path):
        return None
    try:
        return acquire_prepared_run_lock_path(
            path,
            busy_message=(
                "another prepared generation is already running for this "
                "prepared package"
            ),
        )
    except PreparedRunLockError as exc:
        raise BenchmarkError(str(exc)) from exc


@dataclass(frozen=True)
class GenerationBenchmark:
    prepared_manifest: Path
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    estimated_read_bytes: int
    estimated_read_gib_per_second: float
    estimated_embedding_read_bytes: int
    estimated_expert_read_bytes: int
    estimated_cache_read_bytes: int
    estimated_logits_read_bytes: int
    prompt_prefill_chunk_tokens: int
    prompt_prefill_chunk_count: int
    prompt_prefill_estimated_peak_bytes: int
    prompt_prefill_total_embedding_read_bytes: int
    prompt_prefill_total_embedding_output_bytes: int
    prompt_prefill_total_staged_bytes: int
    prompt_prefill_total_compact_stage_bytes: int
    prompt_prefill_total_compact_stage_materialized_bytes: int
    prompt_prefill_max_staged_bytes: int
    prompt_prefill_max_compact_stage_bytes: int
    prompt_prefill_max_compact_stage_materialized_bytes: int
    prompt_prefill_total_stage_plus_compact_bytes: int
    prompt_prefill_total_stage_plus_compact_materialized_bytes: int
    prompt_prefill_max_stage_plus_compact_bytes: int
    prompt_prefill_max_stage_plus_compact_materialized_bytes: int
    linear_backend_counts: dict[str, int]
    total_routed_expert_assignments: int
    total_routed_unique_expert_slots: int
    max_routed_unique_experts_per_call: int
    max_routed_tokens_per_expert: int
    total_expert_stage_serial_read_bytes: int
    total_expert_stage_unique_requested_bytes: int
    total_expert_stage_planned_read_bytes: int
    total_expert_stage_waste_bytes: int
    total_expert_stage_coalesced_savings_bytes: int
    total_expert_stage_planned_read_seconds: float | None
    prefill_ssd_read_gib_per_second: float
    prefill_max_routed_read_seconds: float
    total_expert_stage_read_seconds_ok: bool | None
    total_expert_stage_copy_seconds_ok: bool | None
    prefill_max_stage_raw_ranges: int
    prefill_max_stage_coalesced_ranges: int
    total_expert_stage_raw_ranges: int
    total_expert_stage_coalesced_ranges: int
    max_expert_stage_raw_ranges: int
    max_expert_stage_coalesced_ranges: int
    total_expert_stage_raw_ranges_ok: bool | None
    total_expert_stage_coalesced_ranges_ok: bool | None
    total_expert_stage_read_advice_attempted_ranges: int
    total_expert_stage_read_advice_calls: int
    total_expert_stage_read_advice_bytes: int
    total_expert_stage_read_advice_failures: int
    total_expert_stage_assignment_read_amplification: float
    total_expert_stage_unique_read_amplification: float
    max_expert_stage_unique_read_amplification: float
    max_expert_stage_stage_budget_utilization: float
    max_effective_moe_token_block: int
    max_moe_max_expert_tokens: int
    max_moe_batch_buffer_bytes: int
    max_moe_estimated_peak_bytes: int
    prefill_static_capacity_per_expert: object | None
    max_static_capacity_per_expert: int
    total_static_capacity_used_slots: int
    total_static_capacity_slots: int
    total_static_capacity_overflow_assignments: int
    total_static_capacity_binary_bytes: int
    total_linear_matrix_scratch_bytes: int
    max_linear_matrix_scratch_bytes: int
    total_linear_matrix_f32_bytes: int
    total_linear_matrix_raw_conversion_bytes: int
    linear_backend_flops: dict[str, int]
    linear_backend_elapsed_seconds: dict[str, float]
    linear_backend_estimated_tflops: dict[str, float]
    total_linear_estimated_flops: int
    accelerated_linear_estimated_flops: int
    custom_linear_estimated_flops: int
    unsupported_linear_estimated_flops: int
    accelerated_linear_flop_fraction: float
    suggested_guard_flags: dict[str, object] | None
    suggested_stage_temp_guard_flags: dict[str, object] | None
    suggested_prefill_guard_flags: dict[str, object] | None
    suggested_decode_guard_flags: dict[str, object] | None
    suggested_launch_profile: dict[str, object] | None
    routed_chunk_frontier: dict[str, object] | None
    prefill_acceleration_coverage: dict[str, object] | None
    prefill_acceleration_frontier: dict[str, object] | None
    token_result: TokenGenerationResult
    applied_launch_profile: dict[str, object] | None = None
    total_expert_stage_copy_elapsed_seconds: float | None = None
    total_expert_stage_copy_throughput_gib_per_second: float | None = None


def _positive_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise BenchmarkError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise BenchmarkError(f"{label} must be finite")
    if parsed <= 0:
        raise BenchmarkError(f"{label} must be positive")
    return parsed


def _benchmark_kw_float(
    generation_kwargs: dict[str, object],
    name: str,
    default: float,
) -> float:
    value = generation_kwargs.get(name, default)
    if value is None:
        return default
    return float(value)


def _benchmark_kw_int(
    generation_kwargs: dict[str, object],
    name: str,
    default: int,
) -> int:
    value = generation_kwargs.get(name, default)
    if value is None:
        return default
    return int(value)


def _benchmark_kw_bool(
    generation_kwargs: dict[str, object],
    name: str,
    default: bool,
) -> bool:
    value = generation_kwargs.get(name, default)
    if value is None:
        return default
    return bool(value)


def _require_no_missing_dsa_indexer_for_public_glm_5_2(
    *,
    require_public_glm_5_2_shape: bool,
    generation_kwargs: dict[str, object],
) -> None:
    if not bool(require_public_glm_5_2_shape):
        return
    if _benchmark_kw_bool(generation_kwargs, "allow_missing_dsa_indexer", False):
        raise BenchmarkError(
            "require_public_glm_5_2_shape cannot be used with "
            "allow_missing_dsa_indexer"
        )


def _benchmark_request_payload(
    *,
    max_new_tokens: int,
    generation_kwargs: dict[str, object],
) -> dict[str, object]:
    return {
        "max_new_tokens": int(max_new_tokens),
        "logits_top_k": _benchmark_kw_int(generation_kwargs, "logits_top_k", 1),
        "temperature": _benchmark_kw_float(
            generation_kwargs,
            "sampling_temperature",
            0.0,
        ),
        "top_p": _benchmark_kw_float(generation_kwargs, "sampling_top_p", 1.0),
        "metal_final_logits": _benchmark_kw_bool(
            generation_kwargs,
            "metal_final_logits",
            False,
        ),
        "batch_prefill_prompt": _benchmark_kw_bool(
            generation_kwargs,
            "batch_prefill_prompt",
            False,
        ),
    }


def _require_benchmark_request_admission(
    *,
    prepared: PreparedManifest,
    runner_path: str | Path,
    prompt_token_count: int,
    max_new_tokens: int,
    require_prefill_acceleration: bool,
    prefill_min_accelerated_flop_fraction: float,
    require_glm_4bit: bool,
    require_public_glm_5_2_shape: bool,
    compile_mpp_probe: bool,
    run_mpp_probe: bool,
    run_mpsgraph_probe: bool,
    generation_kwargs: dict[str, object],
) -> str:
    if isinstance(prompt_token_count, bool) or prompt_token_count <= 0:
        raise BenchmarkError("prompt_token_ids must be non-empty")
    from .server import (
        PreparedGenerationApp,
        PreparedServerConfig,
        PreparedServerError,
        require_prepared_request_check_ok,
    )

    logits_top_k = _benchmark_kw_int(generation_kwargs, "logits_top_k", 1)
    configured_prefill_backend = str(
        generation_kwargs.get("prefill_linear_backend", "auto") or "auto"
    )
    config = PreparedServerConfig(
        prepared_path=prepared.manifest_path,
        runner_path=Path(runner_path),
        require_glm_4bit=bool(require_glm_4bit),
        require_public_glm_5_2_shape=bool(require_public_glm_5_2_shape),
        max_new_tokens_cap=max(0, int(max_new_tokens)),
        max_prompt_tokens=int(prompt_token_count),
        batch_prefill_prompt=_benchmark_kw_bool(
            generation_kwargs,
            "batch_prefill_prompt",
            False,
        ),
        max_cache_read_mib=_benchmark_kw_float(
            generation_kwargs,
            "max_cache_read_mib",
            256.0,
        ),
        max_cache_file_mib=_benchmark_kw_float(
            generation_kwargs,
            "max_cache_file_mib",
            32768.0,
        ),
        decode_max_routed_read_gib_per_token=_benchmark_kw_float(
            generation_kwargs,
            "decode_max_routed_read_gib_per_token",
            0.0,
        ),
        decode_max_routed_read_seconds_per_token=_benchmark_kw_float(
            generation_kwargs,
            "decode_max_routed_read_seconds_per_token",
            0.0,
        ),
        max_runner_scratch_mib=_benchmark_kw_float(
            generation_kwargs,
            "max_runner_scratch_mib",
            4096.0,
        ),
        max_live_working_set_mib=(
            None
            if generation_kwargs.get("max_live_working_set_mib") is None
            else float(generation_kwargs["max_live_working_set_mib"])
        ),
        min_free_unified_memory_gib=(
            None
            if generation_kwargs.get("min_free_unified_memory_gib") is None
            else float(generation_kwargs["min_free_unified_memory_gib"])
        ),
        prefill_prompt_chunk_tokens=_benchmark_kw_int(
            generation_kwargs,
            "prefill_prompt_chunk_tokens",
            0,
        ),
        prefill_max_prompt_batch_mib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_prompt_batch_mib",
            1024.0,
        ),
        prefill_max_cache_write_mib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_cache_write_mib",
            4096.0,
        ),
        prefill_max_stage_mib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_stage_mib",
            4096.0,
        ),
        prefill_max_compact_stage_mib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_compact_stage_mib",
            4096.0,
        ),
        prefill_copy_chunk_mib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_copy_chunk_mib",
            8.0,
        ),
        prefill_stage_disk_margin_mib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_stage_disk_margin_mib",
            0.0,
        ),
        prefill_max_routed_read_amplification=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_routed_read_amplification",
            0.0,
        ),
        prefill_max_routed_read_gib=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_routed_read_gib",
            0.0,
        ),
        prefill_ssd_read_gib_per_second=_benchmark_kw_float(
            generation_kwargs,
            "prefill_ssd_read_gib_per_second",
            0.0,
        ),
        prefill_max_routed_read_seconds=_benchmark_kw_float(
            generation_kwargs,
            "prefill_max_routed_read_seconds",
            0.0,
        ),
        prefill_moe_token_block=generation_kwargs.get(
            "prefill_moe_token_block",
            "auto",
        ),
        prefill_linear_backend=configured_prefill_backend,
        require_prefill_acceleration=bool(require_prefill_acceleration),
        allow_router_gate_only_prefill_acceleration=_benchmark_kw_bool(
            generation_kwargs,
            "allow_router_gate_only_prefill_acceleration",
            False,
        ),
        prefill_min_accelerated_flop_fraction=float(
            prefill_min_accelerated_flop_fraction
        ),
        prefill_mpsgraph_min_batch_tokens=_benchmark_kw_int(
            generation_kwargs,
            "prefill_mpsgraph_min_batch_tokens",
            AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        ),
        prefill_mpsgraph_min_matrix_dim=_benchmark_kw_int(
            generation_kwargs,
            "prefill_mpsgraph_min_matrix_dim",
            AUTO_MPSGRAPH_MIN_DIM,
        ),
        prefill_compile_mpp_probe=bool(compile_mpp_probe),
        prefill_run_mpp_probe=bool(run_mpp_probe),
        prefill_run_mpsgraph_probe=bool(run_mpsgraph_probe),
        logits_top_k_cap=max(64, int(logits_top_k)),
        echo_runner_output=_benchmark_kw_bool(
            generation_kwargs,
            "echo_runner_output",
            True,
        ),
        allow_missing_dsa_indexer=_benchmark_kw_bool(
            generation_kwargs,
            "allow_missing_dsa_indexer",
            False,
        ),
    )
    try:
        app = PreparedGenerationApp(config)
        admission_overrides = dict(generation_kwargs)
        if configured_prefill_backend == "auto":
            admission_overrides.pop("prefill_linear_backend", None)
        request_check = app.inspect_token_request(
            prompt_token_count=int(prompt_token_count),
            payload=_benchmark_request_payload(
                max_new_tokens=int(max_new_tokens),
                generation_kwargs=generation_kwargs,
            ),
            runtime_preflight=_benchmark_kw_bool(
                generation_kwargs,
                "preflight_runtime",
                True,
            ),
            generation_overrides=admission_overrides,
        )
        require_prepared_request_check_ok(request_check)
        return app.state.runtime_prefill_linear_backend
    except PreparedServerError as exc:
        raise BenchmarkError(f"prepared request admission failed: {exc}") from exc


def _sorted_positive_backend_ints(values: dict[str, int] | None) -> dict[str, int]:
    if not values:
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


def _prefill_acceleration_coverage_from_counts(
    counts: dict[str, int],
    *,
    flops_by_backend: dict[str, int] | None = None,
    router_gate_matrix_count: int = 0,
    router_gate_estimated_flops: int = 0,
    router_gate_accelerated_matrix_count: int = 0,
    router_gate_accelerated_estimated_flops: int = 0,
    mpp_tensor_ops_candidate_matrix_count: int = 0,
    mpp_tensor_ops_candidate_estimated_flops: int = 0,
    mpp_tensor_ops_candidate_backend_counts: dict[str, int] | None = None,
    mpp_tensor_ops_candidate_backend_flops: dict[str, int] | None = None,
    required: bool = False,
    min_accelerated_flop_fraction: float = 0.0,
) -> dict[str, object]:
    matrix_count = sum(max(0, int(value)) for value in counts.values())
    mpsgraph_count = max(0, int(counts.get("mpsgraph-f32", 0)))
    custom_count = max(0, int(counts.get("custom-metal", 0)))
    unsupported_count = max(0, int(counts.get("unsupported-mpsgraph", 0)))
    accelerated_backends = tuple(
        backend
        for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        if max(0, int(counts.get(backend, 0))) > 0
    )
    accelerated_count = sum(
        max(0, int(counts.get(backend, 0)))
        for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
    )
    other_count = max(
        0,
        matrix_count - accelerated_count - custom_count - unsupported_count,
    )
    flops = flops_by_backend or {}
    total_estimated_flops = sum(max(0, int(value)) for value in flops.values())
    accelerated_estimated_flops = sum(
        max(0, int(flops.get(backend, 0)))
        for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
    )
    custom_estimated_flops = max(0, int(flops.get("custom-metal", 0)))
    unsupported_estimated_flops = max(0, int(flops.get("unsupported-mpsgraph", 0)))
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
    router_gate_count = min(max(0, int(router_gate_matrix_count)), matrix_count)
    router_gate_flops = min(
        max(0, int(router_gate_estimated_flops)),
        total_estimated_flops,
    )
    router_gate_accelerated_count = min(
        max(0, int(router_gate_accelerated_matrix_count)),
        accelerated_count,
        router_gate_count,
    )
    router_gate_accelerated_flops = min(
        max(0, int(router_gate_accelerated_estimated_flops)),
        accelerated_estimated_flops,
        router_gate_flops,
    )
    non_router_accelerated_count = max(
        0,
        accelerated_count - router_gate_accelerated_count,
    )
    non_router_accelerated_flops = max(
        0,
        accelerated_estimated_flops - router_gate_accelerated_flops,
    )
    accelerated_router_gate_flop_share = (
        router_gate_accelerated_flops / accelerated_estimated_flops
        if accelerated_estimated_flops > 0
        else 0.0
    )
    mpp_candidate_count = max(0, int(mpp_tensor_ops_candidate_matrix_count))
    mpp_candidate_flops = max(0, int(mpp_tensor_ops_candidate_estimated_flops))
    mpp_candidate_backend_counts = _sorted_positive_backend_ints(
        mpp_tensor_ops_candidate_backend_counts
    )
    mpp_candidate_backend_flops = _sorted_positive_backend_ints(
        mpp_tensor_ops_candidate_backend_flops
    )
    mpp_candidate_flop_fraction = (
        mpp_candidate_flops / total_estimated_flops
        if total_estimated_flops > 0
        else 0.0
    )
    any_accelerated = accelerated_count > 0
    all_accelerated = (
        matrix_count > 0
        and accelerated_count == matrix_count
        and custom_count == 0
        and unsupported_count == 0
        and other_count == 0
    )
    if any_accelerated:
        reason = ""
    elif matrix_count <= 0:
        reason = "no resident prefill matrices were reported"
    else:
        reason = "no resident prefill matrices used an accelerated backend"
    ok = True
    if required:
        ok = any_accelerated and (
            accelerated_flop_fraction >= min_accelerated_flop_fraction
        )
        if any_accelerated and not ok:
            reason = (
                f"accelerated prefill FLOP fraction "
                f"{accelerated_flop_fraction:.3g} is below required "
                f"{min_accelerated_flop_fraction:.3g}"
            )
    return {
        "required": required,
        "analyzed": True,
        "ok": ok,
        "min_accelerated_flop_fraction": float(min_accelerated_flop_fraction),
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
        "router_gate_estimated_flops": router_gate_flops,
        "router_gate_accelerated_matrix_count": router_gate_accelerated_count,
        "router_gate_accelerated_estimated_flops": router_gate_accelerated_flops,
        "non_router_accelerated_matrix_count": non_router_accelerated_count,
        "non_router_accelerated_estimated_flops": non_router_accelerated_flops,
        "accelerated_router_gate_flop_share": accelerated_router_gate_flop_share,
        "accelerated_router_gate_only": (
            accelerated_count > 0
            and router_gate_accelerated_count == accelerated_count
            and router_gate_accelerated_flops == accelerated_estimated_flops
        ),
        "mpp_candidate_policy": {
            "candidate_backend": "mpp_tensor_ops_prefill",
            "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
            "mpp_tensor_ops_min_batch_tokens": DEFAULT_MPP_MIN_TOKENS,
            "mpp_tensor_ops_min_matrix_dim": MPP_TENSOR_OPS_MIN_MATRIX_DIM,
            "selectable_prefill_backend": False,
        },
        "mpp_tensor_ops_candidate_matrix_count": mpp_candidate_count,
        "mpp_tensor_ops_candidate_estimated_flops": mpp_candidate_flops,
        "mpp_tensor_ops_candidate_flop_fraction": mpp_candidate_flop_fraction,
        "mpp_tensor_ops_candidate_backend_counts": mpp_candidate_backend_counts,
        "mpp_tensor_ops_candidate_backend_flops": mpp_candidate_backend_flops,
        "accelerated_flop_fraction": accelerated_flop_fraction,
        "dominant_resident_flops_accelerated": (
            total_estimated_flops > 0
            and accelerated_estimated_flops * 2 >= total_estimated_flops
        ),
        "accelerated_backends": accelerated_backends,
        "any_resident_matrix_accelerated": any_accelerated,
        "all_resident_matrices_accelerated": all_accelerated,
        "reason": reason,
    }


def _benchmark_prefill_actual_acceleration_coverage_section(
    coverage: dict[str, object],
    *,
    required: bool,
    min_accelerated_flop_fraction: float,
) -> dict[str, object]:
    section = dict(coverage)
    section["required"] = bool(required)
    section["min_accelerated_flop_fraction"] = float(
        min_accelerated_flop_fraction
    )
    any_accelerated = section.get("any_resident_matrix_accelerated") is True
    fraction = float(section.get("accelerated_flop_fraction") or 0.0)
    if required:
        ok = any_accelerated and fraction >= min_accelerated_flop_fraction
        section["ok"] = ok
        if not ok and any_accelerated:
            section["reason"] = (
                f"accelerated prefill FLOP fraction {fraction:.3g} is below "
                f"required {min_accelerated_flop_fraction:.3g}"
            )
        elif not ok:
            section["reason"] = str(
                section.get("reason")
                or "actual prompt prefill did not use an accelerated backend"
            )
    else:
        section.setdefault("ok", True)
        section.setdefault("reason", "")
    return section


def _benchmark_prefill_linear_summary(
    *,
    resident_layout_path: Path,
    prefill_linear_backend: str,
    prompt_chunk_tokens: int,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
) -> dict[str, object]:
    try:
        payload = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"failed to inspect resident layout for benchmark acceleration frontier: {exc}"
        ) from exc
    tensors = payload.get("tensors")
    if not isinstance(tensors, list):
        raise BenchmarkError(
            "resident layout missing tensors array for benchmark acceleration frontier"
        )

    counts: dict[str, int] = {}
    flops_by_backend: dict[str, int] = {}
    mpp_candidate_backend_counts: dict[str, int] = {}
    mpp_candidate_backend_flops: dict[str, int] = {}
    mpp_candidate_count = 0
    mpp_candidate_flops = 0
    max_scratch = 0
    total_scratch = 0
    total_raw_conversion = 0
    router_gate_matrix_count = 0
    router_gate_estimated_flops = 0
    router_gate_accelerated_matrix_count = 0
    router_gate_accelerated_estimated_flops = 0
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = tensor.get("name")
        shape = tensor.get("shape")
        router_gate_tensor = is_prompt_prefill_router_gate_tensor(name)
        if (
            not is_prompt_prefill_resident_linear_backend_tensor(name)
            and not router_gate_tensor
        ):
            continue
        if not isinstance(shape, list) or len(shape) < 2:
            continue
        if type(shape[0]) is not int or type(shape[1]) is not int:
            raise BenchmarkError(
                f"resident tensor {name} shape must use integer rows and cols"
            )
        rows, cols = int(shape[0]), int(shape[1])
        if rows <= 0 or cols <= 0:
            continue
        raw_size = tensor.get("size")
        if type(raw_size) is not int:
            raise BenchmarkError(f"resident tensor {name} size must be an integer")
        size = int(raw_size)
        dtype = str(tensor.get("dtype") or "")
        estimated_flops = 2 * int(prompt_chunk_tokens) * rows * cols
        if prefill_linear_backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS:
            if dtype not in PREFILL_LINEAR_MPSGRAPH_DTYPES:
                backend = "unsupported-mpsgraph"
            else:
                backend = prefill_linear_backend
        elif prefill_linear_backend in {"auto", PREFILL_LINEAR_AUTO_MPP_BACKEND}:
            backend = (
                (
                    "mpp-f32"
                    if prefill_linear_backend == PREFILL_LINEAR_AUTO_MPP_BACKEND
                    else "mpsgraph-f32"
                )
                if dtype in PREFILL_LINEAR_MPSGRAPH_DTYPES
                and prompt_chunk_tokens >= mpsgraph_min_batch_tokens
                and min(rows, cols) >= mpsgraph_min_matrix_dim
                else "custom-metal"
            )
        else:
            backend = "custom-metal"
        if router_gate_tensor and backend == "custom-metal":
            continue
        if router_gate_tensor:
            router_gate_matrix_count += 1
            router_gate_estimated_flops += estimated_flops
            if backend in PREFILL_LINEAR_ACCELERATED_BACKENDS:
                router_gate_accelerated_matrix_count += 1
                router_gate_accelerated_estimated_flops += estimated_flops
        if (
            prompt_chunk_tokens >= DEFAULT_MPP_MIN_TOKENS
            and rows >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
            and cols >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
        ):
            mpp_candidate_count += 1
            mpp_candidate_flops += estimated_flops
            mpp_candidate_backend_counts[backend] = (
                mpp_candidate_backend_counts.get(backend, 0) + 1
            )
            mpp_candidate_backend_flops[backend] = (
                mpp_candidate_backend_flops.get(backend, 0) + estimated_flops
            )
        counts[backend] = counts.get(backend, 0) + 1
        flops_by_backend[backend] = (
            flops_by_backend.get(backend, 0) + estimated_flops
        )
        scratch_backend = (
            backend if backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS else "custom-metal"
        )
        scratch = _resident_linear_matrix_scratch(
            matrix_bytes=size,
            dtype=dtype,
            in_dim=cols,
            out_dim=rows,
            backend=scratch_backend,
        )
        total_scratch += scratch.matrix_scratch_bytes
        max_scratch = max(max_scratch, scratch.matrix_scratch_bytes)
        total_raw_conversion += scratch.matrix_raw_conversion_bytes
    coverage = _prefill_acceleration_coverage_from_counts(
        counts,
        flops_by_backend=flops_by_backend,
        router_gate_matrix_count=router_gate_matrix_count,
        router_gate_estimated_flops=router_gate_estimated_flops,
        router_gate_accelerated_matrix_count=router_gate_accelerated_matrix_count,
        router_gate_accelerated_estimated_flops=(
            router_gate_accelerated_estimated_flops
        ),
        mpp_tensor_ops_candidate_matrix_count=mpp_candidate_count,
        mpp_tensor_ops_candidate_estimated_flops=mpp_candidate_flops,
        mpp_tensor_ops_candidate_backend_counts=mpp_candidate_backend_counts,
        mpp_tensor_ops_candidate_backend_flops=mpp_candidate_backend_flops,
    )
    return coverage | {
        "linear_backend_counts": dict(sorted(counts.items())),
        "linear_backend_flops": dict(sorted(flops_by_backend.items())),
        "max_matrix_scratch_bytes": max_scratch,
        "total_matrix_scratch_bytes": total_scratch,
        "total_matrix_raw_conversion_bytes": total_raw_conversion,
    }


def _prefill_acceleration_frontier_from_layout(
    *,
    resident_layout_path: Path,
    prompt_token_count: int,
    prompt_chunk_tokens: int,
    actual_coverage: dict[str, object],
    actual_total_matrix_scratch_bytes: int,
    actual_total_matrix_raw_conversion_bytes: int,
    prefill_linear_backend: str,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
) -> dict[str, object]:
    candidate_values = {
        1,
        int(prompt_chunk_tokens),
        int(prompt_token_count),
        int(mpsgraph_min_batch_tokens),
    }
    candidate_chunks = tuple(sorted(value for value in candidate_values if value > 0))
    candidates: list[dict[str, object]] = []
    minimum_accelerated: int | None = None
    for chunk_tokens in candidate_chunks:
        if chunk_tokens == prompt_chunk_tokens:
            coverage = actual_coverage | {
                "total_matrix_scratch_bytes": actual_total_matrix_scratch_bytes,
                "total_matrix_raw_conversion_bytes": (
                    actual_total_matrix_raw_conversion_bytes
                ),
            }
        else:
            coverage = _benchmark_prefill_linear_summary(
                resident_layout_path=resident_layout_path,
                prefill_linear_backend=prefill_linear_backend,
                prompt_chunk_tokens=chunk_tokens,
                mpsgraph_min_batch_tokens=mpsgraph_min_batch_tokens,
                mpsgraph_min_matrix_dim=mpsgraph_min_matrix_dim,
            )
        viable = chunk_tokens <= prompt_token_count
        accelerated = coverage.get("any_resident_matrix_accelerated") is True
        if viable and accelerated and minimum_accelerated is None:
            minimum_accelerated = chunk_tokens
        candidates.append(
            {
                "prompt_chunk_tokens": chunk_tokens,
                "is_resolved": chunk_tokens == prompt_chunk_tokens,
                "is_auto_mpsgraph_threshold": (
                    chunk_tokens == mpsgraph_min_batch_tokens
                ),
                "viable_for_request": viable,
                "exceeds_prompt_tokens": not viable,
                "matrix_count": coverage["matrix_count"],
                "accelerated_matrix_count": coverage["accelerated_matrix_count"],
                "mpsgraph_matrix_count": coverage["mpsgraph_matrix_count"],
                "custom_metal_matrix_count": coverage["custom_metal_matrix_count"],
                "unsupported_mpsgraph_matrix_count": (
                    coverage["unsupported_mpsgraph_matrix_count"]
                ),
                "other_matrix_count": coverage["other_matrix_count"],
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
                "accelerated_router_gate_flop_share": coverage.get(
                    "accelerated_router_gate_flop_share",
                    0.0,
                ),
                "accelerated_router_gate_only": coverage.get(
                    "accelerated_router_gate_only",
                    False,
                ),
                "mpp_tensor_ops_candidate_matrix_count": coverage.get(
                    "mpp_tensor_ops_candidate_matrix_count",
                    0,
                ),
                "mpp_tensor_ops_candidate_estimated_flops": coverage.get(
                    "mpp_tensor_ops_candidate_estimated_flops",
                    0,
                ),
                "mpp_tensor_ops_candidate_flop_fraction": coverage.get(
                    "mpp_tensor_ops_candidate_flop_fraction",
                    0.0,
                ),
                "mpp_tensor_ops_candidate_backend_counts": coverage.get(
                    "mpp_tensor_ops_candidate_backend_counts",
                    {},
                ),
                "mpp_tensor_ops_candidate_backend_flops": coverage.get(
                    "mpp_tensor_ops_candidate_backend_flops",
                    {},
                ),
                "accelerated_flop_fraction": coverage["accelerated_flop_fraction"],
                "dominant_resident_flops_accelerated": (
                    coverage["dominant_resident_flops_accelerated"]
                ),
                "any_resident_matrix_accelerated": accelerated,
                "all_resident_matrices_accelerated": (
                    coverage["all_resident_matrices_accelerated"]
                ),
                "total_matrix_scratch_bytes": coverage.get(
                    "total_matrix_scratch_bytes",
                    0,
                ),
                "total_matrix_raw_conversion_bytes": coverage.get(
                    "total_matrix_raw_conversion_bytes",
                    0,
                ),
            }
        )
    suggested = (
        {
            "prefill_prompt_chunk_tokens": minimum_accelerated,
            "argv": (
                "--prefill-prompt-chunk-tokens",
                str(minimum_accelerated),
            ),
        }
        if minimum_accelerated is not None
        else None
    )
    return {
        "source": "benchmark_actual_prefill",
        "analyzed": True,
        "prompt_token_count": prompt_token_count,
        "resolved_prompt_chunk_tokens": prompt_chunk_tokens,
        "configured_backend": prefill_linear_backend,
        "auto_policy": {
            "mpsgraph_min_batch_tokens": mpsgraph_min_batch_tokens,
            "mpsgraph_min_matrix_dim": mpsgraph_min_matrix_dim,
        },
        "mpp_candidate_policy": {
            "candidate_backend": "mpp_tensor_ops_prefill",
            "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
            "mpp_tensor_ops_min_batch_tokens": DEFAULT_MPP_MIN_TOKENS,
            "mpp_tensor_ops_min_matrix_dim": MPP_TENSOR_OPS_MIN_MATRIX_DIM,
            "selectable_prefill_backend": False,
        },
        "minimum_accelerated_prompt_chunk_tokens": minimum_accelerated,
        "suggested_guard_flags": suggested,
        "candidates": tuple(candidates),
        "reason": str(actual_coverage.get("reason") or ""),
    }


def _benchmark_launch_profile(
    *,
    prepared: PreparedManifest,
    ssd_read_gib_per_second: float,
    require_prepared_memory_profile: bool,
    require_glm_4bit: bool,
    require_public_glm_5_2_shape: bool,
    require_prefill_acceleration: bool,
    allow_router_gate_only_prefill_acceleration: bool,
    prefill_min_accelerated_flop_fraction: float,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int,
    prefill_mpsgraph_min_matrix_dim: int,
    compile_mpp_probe: bool,
    run_mpp_probe: bool,
    run_mpsgraph_probe: bool,
    prefill_guard_flags: dict[str, object] | None,
    decode_guard_flags: dict[str, object] | None,
    prefill_actual_read_time: dict[str, object] | None = None,
    prefill_actual_acceleration_coverage: dict[str, object] | None = None,
    prefill_actual_acceleration_frontier: dict[str, object] | None = None,
    prefill_actual_linear_backend: dict[str, object] | None = None,
    decode_actual_read_time: dict[str, object] | None = None,
) -> dict[str, object] | None:
    from .server import (
        PreparedServerConfig,
        _suggest_glm_4bit_guard_flags,
        _suggest_prepared_launch_guard_flags,
        _suggest_prepared_ssd_read_flags,
        _suggest_prefill_backend_probe_flags,
        _suggest_prefill_runtime_policy_flags,
        _suggest_public_glm_5_2_shape_guard_flags,
        combine_suggested_launch_profile,
        prepared_glm_4bit_readiness,
        prepared_launch_profile_target,
    )

    prefill_linear_backend = (
        "auto" if prefill_linear_backend is None else str(prefill_linear_backend)
    )
    readiness = None
    if require_glm_4bit or require_public_glm_5_2_shape:
        readiness = prepared_glm_4bit_readiness(prepared)
    runtime_policy_config = PreparedServerConfig(
        prepared_path=prepared.manifest_path,
        runner_path=Path("."),
        require_prefill_acceleration=bool(require_prefill_acceleration),
        allow_router_gate_only_prefill_acceleration=bool(
            allow_router_gate_only_prefill_acceleration
        ),
        prefill_min_accelerated_flop_fraction=float(
            prefill_min_accelerated_flop_fraction
        ),
        prefill_mpsgraph_min_batch_tokens=int(prefill_mpsgraph_min_batch_tokens),
        prefill_mpsgraph_min_matrix_dim=int(prefill_mpsgraph_min_matrix_dim),
    )
    prefill_acceleration_flags = None
    if prefill_linear_backend != "auto":
        prefill_acceleration_flags = {
            "source": "benchmark_actual",
            "prefill_linear_backend": prefill_linear_backend,
            "argv": ("--prefill-linear-backend", prefill_linear_backend),
        }
    profile = combine_suggested_launch_profile(
        launch_guard_flags=_suggest_prepared_launch_guard_flags(
            prepared,
            require_memory_profile=bool(require_prepared_memory_profile),
        ),
        prepared_ssd_read_flags=_suggest_prepared_ssd_read_flags(
            prepared,
            ssd_read_gib_per_second=float(ssd_read_gib_per_second),
        ),
        prefill_backend_probe_flags=_suggest_prefill_backend_probe_flags(
            compile_mpp_probe=bool(compile_mpp_probe),
            run_mpp_probe=bool(run_mpp_probe),
            run_mpsgraph_probe=bool(run_mpsgraph_probe),
            source="benchmark_actual",
        ),
        prefill_runtime_policy_flags=_suggest_prefill_runtime_policy_flags(
            runtime_policy_config,
            source="benchmark_actual",
        ),
        glm_4bit_guard_flags=(
            _suggest_glm_4bit_guard_flags(readiness, source="benchmark_actual")
            if isinstance(readiness, dict)
            else None
        ),
        public_glm_5_2_shape_guard_flags=(
            _suggest_public_glm_5_2_shape_guard_flags(
                readiness,
                source="benchmark_actual",
            )
            if require_public_glm_5_2_shape and isinstance(readiness, dict)
            else None
        ),
        prefill_acceleration_flags=prefill_acceleration_flags,
        prefill_guard_flags=(
            prefill_guard_flags if isinstance(prefill_guard_flags, dict) else None
        ),
        decode_guard_flags=(
            decode_guard_flags if isinstance(decode_guard_flags, dict) else None
        ),
        prepared_target=prepared_launch_profile_target(prepared),
        source="benchmark_actual",
    )
    if profile is None:
        return None
    if isinstance(prefill_actual_read_time, dict):
        sections = profile.get("sections")
        if isinstance(sections, dict):
            sections["prefill_actual_read_time"] = prefill_actual_read_time
    if isinstance(prefill_actual_acceleration_coverage, dict):
        sections = profile.get("sections")
        if isinstance(sections, dict):
            sections["prefill_actual_acceleration_coverage"] = (
                prefill_actual_acceleration_coverage
            )
    if isinstance(prefill_actual_acceleration_frontier, dict):
        sections = profile.get("sections")
        if isinstance(sections, dict):
            sections["prefill_actual_acceleration_frontier"] = (
                prefill_actual_acceleration_frontier
            )
    if isinstance(prefill_actual_linear_backend, dict):
        sections = profile.get("sections")
        if isinstance(sections, dict):
            sections["prefill_actual_linear_backend"] = (
                prefill_actual_linear_backend
            )
    if isinstance(decode_actual_read_time, dict):
        sections = profile.get("sections")
        if isinstance(sections, dict):
            sections["decode_actual_read_time"] = decode_actual_read_time
    return profile


def _float_dict(values: dict[str, object]) -> dict[str, float]:
    converted: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            converted[str(key)] = parsed
    return dict(sorted(converted.items()))


def _int_dict(values: dict[str, object]) -> dict[str, int]:
    converted: dict[str, int] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        converted[str(key)] = parsed
    return dict(sorted(converted.items()))


def _benchmark_prefill_actual_linear_backend_section(
    *,
    linear_backend_counts: dict[str, int],
    linear_backend_flops: dict[str, int],
    linear_backend_elapsed_seconds: dict[str, float],
    linear_backend_estimated_tflops: dict[str, float],
    total_linear_estimated_flops: int,
    accelerated_linear_estimated_flops: int,
    custom_linear_estimated_flops: int,
    unsupported_linear_estimated_flops: int,
    accelerated_linear_flop_fraction: float,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int,
    prefill_mpsgraph_min_matrix_dim: int,
) -> dict[str, object] | None:
    counts = _int_dict(linear_backend_counts)
    flops = _int_dict(linear_backend_flops)
    elapsed = _float_dict(linear_backend_elapsed_seconds)
    tflops = _float_dict(linear_backend_estimated_tflops)
    if not counts and not flops and not elapsed:
        return None
    return {
        "source": "benchmark_actual_prefill",
        "configured_backend": str(prefill_linear_backend),
        "auto_policy": {
            "mpsgraph_min_batch_tokens": int(prefill_mpsgraph_min_batch_tokens),
            "mpsgraph_min_matrix_dim": int(prefill_mpsgraph_min_matrix_dim),
        },
        "linear_backend_counts": counts,
        "linear_backend_flops": flops,
        "linear_backend_elapsed_seconds": elapsed,
        "linear_backend_estimated_tflops": tflops,
        "total_linear_estimated_flops": int(total_linear_estimated_flops),
        "accelerated_linear_estimated_flops": int(
            accelerated_linear_estimated_flops
        ),
        "custom_linear_estimated_flops": int(custom_linear_estimated_flops),
        "unsupported_linear_estimated_flops": int(
            unsupported_linear_estimated_flops
        ),
        "accelerated_linear_flop_fraction": float(
            accelerated_linear_flop_fraction
        ),
    }


def _benchmark_prefill_actual_read_time_section(
    *,
    total_expert_stage_planned_read_bytes: int,
    total_expert_stage_planned_read_seconds: float | None,
    prefill_ssd_read_gib_per_second: float,
    prefill_max_routed_read_seconds: float,
    total_expert_stage_read_seconds_ok: bool | None,
    prefill_max_stage_raw_ranges: int,
    prefill_max_stage_coalesced_ranges: int,
    total_expert_stage_raw_ranges: int,
    total_expert_stage_coalesced_ranges: int,
    max_expert_stage_raw_ranges: int,
    max_expert_stage_coalesced_ranges: int,
    total_expert_stage_raw_ranges_ok: bool | None,
    total_expert_stage_coalesced_ranges_ok: bool | None,
    total_expert_stage_copy_seconds_ok: bool | None = None,
    total_expert_stage_copy_elapsed_seconds: float | None = None,
    total_expert_stage_copy_throughput_gib_per_second: float | None = None,
) -> dict[str, object] | None:
    if total_expert_stage_planned_read_seconds is None:
        return None
    read_seconds_ok = (
        bool(total_expert_stage_read_seconds_ok)
        if total_expert_stage_read_seconds_ok is not None
        else None
    )
    section: dict[str, object] = {
        "source": "benchmark_actual_prefill",
        "total_expert_stage_planned_read_bytes": int(
            total_expert_stage_planned_read_bytes
        ),
        "total_expert_stage_planned_read_seconds": float(
            total_expert_stage_planned_read_seconds
        ),
        "prefill_ssd_read_gib_per_second": float(prefill_ssd_read_gib_per_second),
        "prefill_max_routed_read_seconds": float(prefill_max_routed_read_seconds),
        "total_expert_stage_read_seconds_ok": read_seconds_ok,
        "prefill_max_stage_raw_ranges": int(prefill_max_stage_raw_ranges),
        "prefill_max_stage_coalesced_ranges": int(
            prefill_max_stage_coalesced_ranges
        ),
        "total_expert_stage_raw_ranges": int(total_expert_stage_raw_ranges),
        "total_expert_stage_coalesced_ranges": int(
            total_expert_stage_coalesced_ranges
        ),
        "max_expert_stage_raw_ranges": int(max_expert_stage_raw_ranges),
        "max_expert_stage_coalesced_ranges": int(
            max_expert_stage_coalesced_ranges
        ),
    }
    if total_expert_stage_raw_ranges_ok is not None:
        section["total_expert_stage_raw_ranges_ok"] = bool(
            total_expert_stage_raw_ranges_ok
        )
    if total_expert_stage_coalesced_ranges_ok is not None:
        section["total_expert_stage_coalesced_ranges_ok"] = bool(
            total_expert_stage_coalesced_ranges_ok
        )
    if total_expert_stage_copy_seconds_ok is not None:
        section["total_expert_stage_copy_seconds_ok"] = bool(
            total_expert_stage_copy_seconds_ok
        )
    if total_expert_stage_copy_elapsed_seconds is not None:
        section["total_expert_stage_copy_elapsed_seconds"] = float(
            total_expert_stage_copy_elapsed_seconds
        )
    if total_expert_stage_copy_throughput_gib_per_second is not None:
        section["total_expert_stage_copy_throughput_gib_per_second"] = float(
            total_expert_stage_copy_throughput_gib_per_second
        )
    return section


def _benchmark_decode_actual_read_time_section(
    *,
    result: TokenGenerationResult,
    decode_guard_flags: dict[str, object] | None,
    ssd_read_gib_per_second: float,
) -> dict[str, object] | None:
    if result.runtime_guard is None or ssd_read_gib_per_second <= 0:
        return None
    decode_steps = tuple(
        step
        for step in result.steps
        if bool(getattr(step, "decode_layers", ()))
    )
    if not decode_steps:
        return None
    read_bytes_per_token = int(getattr(result.runtime_guard, "read_bytes_per_token", 0))
    if read_bytes_per_token <= 0:
        return None
    actual_read_bytes = sum(
        int(getattr(step, "expert_read_bytes", 0) or 0)
        for step in decode_steps
    )
    planned_read_bytes = read_bytes_per_token * len(decode_steps)
    actual_bytes_ok = actual_read_bytes <= planned_read_bytes
    actual_read_seconds = actual_read_bytes / (ssd_read_gib_per_second * 1024**3)
    planned_read_seconds = planned_read_bytes / (ssd_read_gib_per_second * 1024**3)
    max_seconds_per_token = None
    if isinstance(decode_guard_flags, dict):
        raw = decode_guard_flags.get("decode_max_routed_read_seconds_per_token")
        if raw is not None:
            max_seconds_per_token = float(raw)
    total_max_seconds = (
        max_seconds_per_token * len(decode_steps)
        if max_seconds_per_token is not None
        else None
    )
    seconds_ok = (
        actual_read_seconds <= total_max_seconds
        if total_max_seconds is not None
        else None
    )
    return {
        "source": "benchmark_actual_decode",
        "decode_step_count": len(decode_steps),
        "decode_read_bytes_per_token": read_bytes_per_token,
        "planned_decode_routed_read_bytes": planned_read_bytes,
        "actual_decode_routed_read_bytes": actual_read_bytes,
        "actual_decode_routed_read_bytes_ok": actual_bytes_ok,
        "planned_decode_routed_read_seconds": planned_read_seconds,
        "actual_decode_routed_read_seconds": actual_read_seconds,
        "prefill_ssd_read_gib_per_second": float(ssd_read_gib_per_second),
        "decode_max_routed_read_seconds_per_token": max_seconds_per_token,
        "total_decode_max_routed_read_seconds": total_max_seconds,
        "total_decode_routed_read_seconds_ok": seconds_ok,
    }


def summarize_generation(
    prepared: PreparedManifest,
    result: TokenGenerationResult,
    *,
    ssd_read_gib_per_second: float = 0.0,
    prefill_max_routed_read_seconds: float = 0.0,
    top_k: int = 8,
    stage_align_bytes: int = 4096,
    prefill_linear_backend: str = "auto",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    require_prepared_memory_profile: bool = False,
    require_glm_4bit: bool = False,
    require_public_glm_5_2_shape: bool = False,
    require_prefill_acceleration: bool = False,
    allow_router_gate_only_prefill_acceleration: bool = False,
    prefill_min_accelerated_flop_fraction: float = 0.0,
    compile_mpp_probe: bool = False,
    run_mpp_probe: bool = False,
    run_mpsgraph_probe: bool = False,
) -> GenerationBenchmark:
    generated = len(result.generated_token_ids)
    elapsed = result.elapsed_seconds
    tok_s = generated / elapsed if generated and elapsed > 0 else 0.0
    gib_s = (
        result.estimated_read_bytes / (1024**3) / elapsed
        if result.estimated_read_bytes and elapsed > 0
        else 0.0
    )
    linear_backend_counts: dict[str, int] = {}
    total_routed_assignments = 0
    total_routed_unique_slots = 0
    max_routed_unique_per_call = 0
    max_routed_tokens_per_expert = 0
    total_expert_stage_serial_read = 0
    total_expert_stage_unique_requested = 0
    total_expert_stage_planned_read = 0
    total_expert_stage_waste = 0
    total_expert_stage_coalesced_savings = 0
    total_expert_stage_planned_read_seconds: float | None = None
    total_expert_stage_copy_elapsed_seconds: float | None = None
    total_expert_stage_copy_throughput_gib_per_second: float | None = None
    prefill_max_routed_read_seconds = float(prefill_max_routed_read_seconds or 0.0)
    total_expert_stage_read_seconds_ok: bool | None = None
    total_expert_stage_copy_seconds_ok: bool | None = None
    prefill_max_stage_raw_ranges = 0
    prefill_max_stage_coalesced_ranges = 0
    total_expert_stage_raw_ranges = 0
    total_expert_stage_coalesced_ranges = 0
    max_expert_stage_raw_ranges = 0
    max_expert_stage_coalesced_ranges = 0
    total_expert_stage_raw_ranges_ok: bool | None = None
    total_expert_stage_coalesced_ranges_ok: bool | None = None
    total_expert_stage_read_advice_attempted_ranges = 0
    total_expert_stage_read_advice_calls = 0
    total_expert_stage_read_advice_bytes = 0
    total_expert_stage_read_advice_failures = 0
    total_expert_stage_assignment_read_amplification = 0.0
    total_expert_stage_unique_read_amplification = 0.0
    max_expert_stage_unique_read_amplification = 0.0
    max_expert_stage_stage_budget_utilization = 0.0
    max_effective_moe_token_block = 0
    max_moe_max_expert_tokens = 0
    max_moe_batch_buffer_bytes = 0
    max_moe_estimated_peak_bytes = 0
    prefill_static_capacity_per_expert: object | None = None
    max_static_capacity_per_expert = 0
    total_static_capacity_used_slots = 0
    total_static_capacity_slots = 0
    total_static_capacity_overflow = 0
    total_static_capacity_binary_bytes = 0
    prompt_prefill_chunk_tokens = 0
    prompt_prefill_chunk_count = 0
    prompt_prefill_estimated_peak = 0
    prompt_prefill_total_embedding_read = 0
    prompt_prefill_total_embedding_output = 0
    prompt_prefill_total_staged = 0
    prompt_prefill_total_compact_stage = 0
    prompt_prefill_total_compact_stage_materialized = 0
    prompt_prefill_max_staged = 0
    prompt_prefill_max_compact_stage = 0
    prompt_prefill_max_compact_stage_materialized = 0
    prompt_prefill_total_stage_plus_compact = 0
    prompt_prefill_total_stage_plus_compact_materialized = 0
    prompt_prefill_max_stage_plus_compact = 0
    prompt_prefill_max_stage_plus_compact_materialized = 0
    total_linear_matrix_scratch = 0
    max_linear_matrix_scratch = 0
    total_linear_matrix_f32 = 0
    total_linear_matrix_raw_conversion = 0
    linear_backend_flops: dict[str, int] = {}
    linear_backend_elapsed_seconds: dict[str, float] = {}
    linear_backend_estimated_tflops: dict[str, float] = {}
    total_linear_estimated_flops = 0
    accelerated_linear_estimated_flops = 0
    custom_linear_estimated_flops = 0
    unsupported_linear_estimated_flops = 0
    accelerated_linear_flop_fraction = 0.0
    routed_chunk_frontier: dict[str, object] | None = None
    prefill_acceleration_coverage: dict[str, object] | None = None
    prefill_acceleration_frontier: dict[str, object] | None = None
    ssd_read_gib_per_second = (
        _positive_float(
            ssd_read_gib_per_second,
            label="prefill_ssd_read_gib_per_second",
        )
        if ssd_read_gib_per_second
        else 0.0
    )
    if result.prompt_prefill is not None:
        prefill = result.prompt_prefill
        prompt_prefill_chunk_tokens = int(getattr(prefill, "chunk_tokens", 0))
        prompt_prefill_chunk_count = int(getattr(prefill, "chunk_count", 0))
        prompt_prefill_estimated_peak = int(getattr(prefill, "estimated_peak_bytes", 0))
        prompt_prefill_total_embedding_read = int(
            getattr(prefill, "total_embedding_read_bytes", 0)
        )
        prompt_prefill_total_embedding_output = int(
            getattr(prefill, "total_embedding_output_bytes", 0)
        )
        prompt_prefill_total_staged = int(getattr(prefill, "total_staged_bytes", 0))
        prompt_prefill_total_compact_stage = int(
            getattr(prefill, "total_compact_stage_bytes", 0)
        )
        prompt_prefill_total_compact_stage_materialized = int(
            getattr(prefill, "total_compact_stage_materialized_bytes", 0)
        )
        prompt_prefill_max_staged = int(getattr(prefill, "max_staged_bytes", 0))
        prompt_prefill_max_compact_stage = int(
            getattr(prefill, "max_compact_stage_bytes", 0)
        )
        prompt_prefill_max_compact_stage_materialized = int(
            getattr(prefill, "max_compact_stage_materialized_bytes", 0)
        )
        prompt_prefill_total_stage_plus_compact = int(
            getattr(prefill, "total_stage_plus_compact_bytes", 0)
        )
        prompt_prefill_total_stage_plus_compact_materialized = int(
            getattr(prefill, "total_stage_plus_compact_materialized_bytes", 0)
        )
        prompt_prefill_max_stage_plus_compact = int(
            getattr(prefill, "max_stage_plus_compact_bytes", 0)
        )
        prompt_prefill_max_stage_plus_compact_materialized = int(
            getattr(prefill, "max_stage_plus_compact_materialized_bytes", 0)
        )
        linear_backend_counts = dict(prefill.linear_backend_counts)
        total_linear_matrix_scratch = int(
            getattr(prefill, "total_linear_matrix_scratch_bytes", 0)
        )
        max_linear_matrix_scratch = int(
            getattr(prefill, "max_linear_matrix_scratch_bytes", 0)
        )
        total_linear_matrix_f32 = int(
            getattr(prefill, "total_linear_matrix_f32_bytes", 0)
        )
        total_linear_matrix_raw_conversion = int(
            getattr(prefill, "total_linear_matrix_raw_conversion_bytes", 0)
        )
        linear_backend_flops = dict(getattr(prefill, "linear_backend_flops", {}))
        linear_backend_elapsed_seconds = dict(
            getattr(prefill, "linear_backend_elapsed_seconds", {}) or {}
        )
        linear_backend_estimated_tflops = dict(
            getattr(prefill, "linear_backend_estimated_tflops", {}) or {}
        )
        total_linear_estimated_flops = int(
            getattr(prefill, "total_linear_estimated_flops", 0)
        )
        accelerated_linear_estimated_flops = int(
            getattr(prefill, "accelerated_linear_estimated_flops", 0)
        )
        custom_linear_estimated_flops = int(
            getattr(prefill, "custom_linear_estimated_flops", 0)
        )
        unsupported_linear_estimated_flops = int(
            getattr(prefill, "unsupported_linear_estimated_flops", 0)
        )
        accelerated_linear_flop_fraction = float(
            getattr(prefill, "accelerated_linear_flop_fraction", 0.0)
        )
        total_routed_assignments = int(
            getattr(prefill, "total_routed_expert_assignments", 0)
        )
        total_routed_unique_slots = int(
            getattr(prefill, "total_routed_unique_expert_slots", 0)
        )
        max_routed_unique_per_call = int(
            getattr(prefill, "max_routed_unique_experts_per_call", 0)
        )
        max_routed_tokens_per_expert = int(
            getattr(prefill, "max_routed_tokens_per_expert", 0)
        )
        total_expert_stage_serial_read = int(
            getattr(prefill, "total_expert_stage_serial_read_bytes", 0)
        )
        total_expert_stage_unique_requested = int(
            getattr(prefill, "total_expert_stage_unique_requested_bytes", 0)
        )
        total_expert_stage_planned_read = int(
            getattr(prefill, "total_expert_stage_planned_read_bytes", 0)
        )
        raw_planned_read_seconds = getattr(
            prefill,
            "total_expert_stage_planned_read_seconds",
            None,
        )
        total_expert_stage_planned_read_seconds = (
            float(raw_planned_read_seconds)
            if raw_planned_read_seconds is not None
            else None
        )
        raw_copy_elapsed_seconds = getattr(
            prefill,
            "total_expert_stage_copy_elapsed_seconds",
            None,
        )
        total_expert_stage_copy_elapsed_seconds = (
            float(raw_copy_elapsed_seconds)
            if raw_copy_elapsed_seconds is not None
            else None
        )
        raw_copy_throughput = getattr(
            prefill,
            "total_expert_stage_copy_throughput_gib_per_second",
            None,
        )
        total_expert_stage_copy_throughput_gib_per_second = (
            float(raw_copy_throughput)
            if raw_copy_throughput is not None
            else None
        )
        prefill_max_routed_read_seconds = float(
            getattr(
                prefill,
                "prefill_max_routed_read_seconds",
                prefill_max_routed_read_seconds,
            )
            or 0.0
        )
        total_expert_stage_read_seconds_ok = getattr(
            prefill,
            "total_expert_stage_read_seconds_ok",
            None,
        )
        total_expert_stage_copy_seconds_ok = getattr(
            prefill,
            "total_expert_stage_copy_seconds_ok",
            None,
        )
        prefill_max_stage_raw_ranges = int(
            getattr(prefill, "prefill_max_stage_raw_ranges", 0)
        )
        prefill_max_stage_coalesced_ranges = int(
            getattr(prefill, "prefill_max_stage_coalesced_ranges", 0)
        )
        total_expert_stage_raw_ranges = int(
            getattr(prefill, "total_expert_stage_raw_ranges", 0)
        )
        total_expert_stage_coalesced_ranges = int(
            getattr(prefill, "total_expert_stage_coalesced_ranges", 0)
        )
        max_expert_stage_raw_ranges = int(
            getattr(prefill, "max_expert_stage_raw_ranges", 0)
        )
        max_expert_stage_coalesced_ranges = int(
            getattr(prefill, "max_expert_stage_coalesced_ranges", 0)
        )
        total_expert_stage_raw_ranges_ok = getattr(
            prefill,
            "total_expert_stage_raw_ranges_ok",
            None,
        )
        total_expert_stage_coalesced_ranges_ok = getattr(
            prefill,
            "total_expert_stage_coalesced_ranges_ok",
            None,
        )
        total_expert_stage_waste = int(
            getattr(prefill, "total_expert_stage_waste_bytes", 0)
        )
        total_expert_stage_coalesced_savings = int(
            getattr(prefill, "total_expert_stage_coalesced_savings_bytes", 0)
        )
        total_expert_stage_read_advice_attempted_ranges = int(
            getattr(prefill, "total_expert_stage_read_advice_attempted_ranges", 0)
        )
        total_expert_stage_read_advice_calls = int(
            getattr(prefill, "total_expert_stage_read_advice_calls", 0)
        )
        total_expert_stage_read_advice_bytes = int(
            getattr(prefill, "total_expert_stage_read_advice_bytes", 0)
        )
        total_expert_stage_read_advice_failures = int(
            getattr(prefill, "total_expert_stage_read_advice_failures", 0)
        )
        total_expert_stage_assignment_read_amplification = float(
            getattr(
                prefill,
                "total_expert_stage_assignment_read_amplification",
                0.0,
            )
        )
        total_expert_stage_unique_read_amplification = float(
            getattr(prefill, "total_expert_stage_unique_read_amplification", 0.0)
        )
        max_expert_stage_unique_read_amplification = float(
            getattr(prefill, "max_expert_stage_unique_read_amplification", 0.0)
        )
        max_expert_stage_stage_budget_utilization = float(
            getattr(prefill, "max_expert_stage_stage_budget_utilization", 0.0)
        )
        max_effective_moe_token_block = int(
            getattr(prefill, "max_effective_moe_token_block", 0)
        )
        max_moe_max_expert_tokens = int(
            getattr(prefill, "max_moe_max_expert_tokens", 0)
        )
        max_moe_batch_buffer_bytes = int(
            getattr(prefill, "max_moe_batch_buffer_bytes", 0)
        )
        max_moe_estimated_peak_bytes = int(
            getattr(prefill, "max_moe_estimated_peak_bytes", 0)
        )
        prefill_static_capacity_per_expert = getattr(
            prefill,
            "static_capacity_per_expert",
            None,
        )
        max_static_capacity_per_expert = int(
            getattr(prefill, "max_static_capacity_per_expert", 0)
        )
        total_static_capacity_used_slots = int(
            getattr(prefill, "total_static_capacity_used_slots", 0)
        )
        total_static_capacity_slots = int(
            getattr(prefill, "total_static_capacity_slots", 0)
        )
        total_static_capacity_overflow = int(
            getattr(prefill, "total_static_capacity_overflow_assignments", 0)
        )
        total_static_capacity_binary_bytes = int(
            getattr(prefill, "total_static_capacity_binary_bytes", 0)
        )
        if prompt_prefill_chunk_tokens > 0:
            actual_coverage = getattr(prefill, "prefill_acceleration_coverage", None)
            if isinstance(actual_coverage, dict):
                prefill_acceleration_coverage = dict(actual_coverage)
            else:
                prefill_acceleration_coverage = (
                    _prefill_acceleration_coverage_from_counts(
                        linear_backend_counts,
                        flops_by_backend=linear_backend_flops,
                        required=(
                            require_prefill_acceleration
                            or prefill_min_accelerated_flop_fraction > 0.0
                        ),
                        min_accelerated_flop_fraction=(
                            prefill_min_accelerated_flop_fraction
                        ),
                    )
                )
            prefill_acceleration_coverage = (
                _benchmark_prefill_actual_acceleration_coverage_section(
                    prefill_acceleration_coverage,
                    required=(
                        require_prefill_acceleration
                        or prefill_min_accelerated_flop_fraction > 0.0
                    ),
                    min_accelerated_flop_fraction=(
                        prefill_min_accelerated_flop_fraction
                    ),
                )
            )
            prefill_acceleration_frontier = _prefill_acceleration_frontier_from_layout(
                resident_layout_path=prepared.resident_layout,
                prompt_token_count=len(result.prompt_token_ids),
                prompt_chunk_tokens=prompt_prefill_chunk_tokens,
                actual_coverage=prefill_acceleration_coverage,
                actual_total_matrix_scratch_bytes=total_linear_matrix_scratch,
                actual_total_matrix_raw_conversion_bytes=total_linear_matrix_raw_conversion,
                prefill_linear_backend=prefill_linear_backend,
                mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
                mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            )
            try:
                frontier = estimate_routed_prefill_chunk_frontier(
                    expert_layout_path=prepared.experts_layout,
                    prompt_token_count=len(result.prompt_token_ids),
                    top_k=top_k,
                    stage_align_bytes=stage_align_bytes,
                    layers=getattr(prefill, "layers", None),
                    include_chunk_tokens=(prompt_prefill_chunk_tokens,),
                    ssd_read_gib_per_second=ssd_read_gib_per_second,
                    static_capacity_per_expert=prefill_static_capacity_per_expert,
                )
            except RoutedExpertReadError as exc:
                raise BenchmarkError(str(exc)) from exc
            routed_chunk_frontier = asdict(frontier) | {
                "source": "benchmark_actual_prefill",
                "resolved_prompt_chunk_tokens": prompt_prefill_chunk_tokens,
            }
    suggested_guard_flags = suggest_routed_read_guard_flags(
        prompt_chunk_tokens=prompt_prefill_chunk_tokens,
        planned_read_bytes=total_expert_stage_planned_read,
        read_amplification=total_expert_stage_unique_read_amplification,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
        source="benchmark_actual_prefill",
    )
    suggested_stage_temp_guard_flags = suggest_routed_stage_temp_guard_flags(
        prompt_chunk_tokens=prompt_prefill_chunk_tokens,
        max_stage_bytes=prompt_prefill_max_staged,
        max_compact_stage_bytes=prompt_prefill_max_compact_stage,
        max_stage_raw_ranges=max_expert_stage_raw_ranges,
        max_stage_coalesced_ranges=max_expert_stage_coalesced_ranges,
        max_stage_plus_compact_bytes=prompt_prefill_max_stage_plus_compact,
        total_stage_plus_compact_bytes=prompt_prefill_total_stage_plus_compact,
        total_static_capacity_binary_bytes=total_static_capacity_binary_bytes,
        total_stage_plus_compact_plus_static_bytes=(
            prompt_prefill_total_stage_plus_compact
            + total_static_capacity_binary_bytes
        ),
        static_capacity_per_expert=prefill_static_capacity_per_expert,
        source="benchmark_actual_prefill",
    )
    suggested_prefill_guard_flags = combine_prefill_guard_flags(
        routed_read_flags=suggested_guard_flags,
        stage_temp_flags=suggested_stage_temp_guard_flags,
        source="benchmark_actual_prefill",
    )
    decode_read_bytes_per_token = (
        result.runtime_guard.read_bytes_per_token
        if result.runtime_guard is not None
        else None
    )
    suggested_decode_guard_flags = suggest_decode_routed_read_guard_flags(
        read_bytes_per_token=decode_read_bytes_per_token,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
        source="benchmark_actual_decode",
    )
    prefill_actual_read_time = _benchmark_prefill_actual_read_time_section(
        total_expert_stage_planned_read_bytes=total_expert_stage_planned_read,
        total_expert_stage_planned_read_seconds=(
            total_expert_stage_planned_read_seconds
        ),
        prefill_ssd_read_gib_per_second=ssd_read_gib_per_second,
        prefill_max_routed_read_seconds=prefill_max_routed_read_seconds,
        total_expert_stage_read_seconds_ok=total_expert_stage_read_seconds_ok,
        prefill_max_stage_raw_ranges=prefill_max_stage_raw_ranges,
        prefill_max_stage_coalesced_ranges=prefill_max_stage_coalesced_ranges,
        total_expert_stage_raw_ranges=total_expert_stage_raw_ranges,
        total_expert_stage_coalesced_ranges=total_expert_stage_coalesced_ranges,
        max_expert_stage_raw_ranges=max_expert_stage_raw_ranges,
        max_expert_stage_coalesced_ranges=max_expert_stage_coalesced_ranges,
        total_expert_stage_raw_ranges_ok=total_expert_stage_raw_ranges_ok,
        total_expert_stage_coalesced_ranges_ok=(
            total_expert_stage_coalesced_ranges_ok
        ),
        total_expert_stage_copy_seconds_ok=total_expert_stage_copy_seconds_ok,
        total_expert_stage_copy_elapsed_seconds=(
            total_expert_stage_copy_elapsed_seconds
        ),
        total_expert_stage_copy_throughput_gib_per_second=(
            total_expert_stage_copy_throughput_gib_per_second
        ),
    )
    prefill_actual_linear_backend = _benchmark_prefill_actual_linear_backend_section(
        linear_backend_counts=linear_backend_counts,
        linear_backend_flops=linear_backend_flops,
        linear_backend_elapsed_seconds=linear_backend_elapsed_seconds,
        linear_backend_estimated_tflops=linear_backend_estimated_tflops,
        total_linear_estimated_flops=total_linear_estimated_flops,
        accelerated_linear_estimated_flops=accelerated_linear_estimated_flops,
        custom_linear_estimated_flops=custom_linear_estimated_flops,
        unsupported_linear_estimated_flops=unsupported_linear_estimated_flops,
        accelerated_linear_flop_fraction=accelerated_linear_flop_fraction,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
    )
    decode_actual_read_time = _benchmark_decode_actual_read_time_section(
        result=result,
        decode_guard_flags=suggested_decode_guard_flags,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
    )
    suggested_launch_profile = _benchmark_launch_profile(
        prepared=prepared,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
        require_prepared_memory_profile=require_prepared_memory_profile,
        require_glm_4bit=require_glm_4bit,
        require_public_glm_5_2_shape=require_public_glm_5_2_shape,
        require_prefill_acceleration=require_prefill_acceleration,
        allow_router_gate_only_prefill_acceleration=(
            allow_router_gate_only_prefill_acceleration
        ),
        prefill_min_accelerated_flop_fraction=prefill_min_accelerated_flop_fraction,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        compile_mpp_probe=compile_mpp_probe,
        run_mpp_probe=run_mpp_probe,
        run_mpsgraph_probe=run_mpsgraph_probe,
        prefill_guard_flags=suggested_prefill_guard_flags,
        decode_guard_flags=suggested_decode_guard_flags,
        prefill_actual_read_time=prefill_actual_read_time,
        prefill_actual_acceleration_coverage=prefill_acceleration_coverage,
        prefill_actual_acceleration_frontier=prefill_acceleration_frontier,
        prefill_actual_linear_backend=prefill_actual_linear_backend,
        decode_actual_read_time=decode_actual_read_time,
    )
    return GenerationBenchmark(
        prepared_manifest=prepared.manifest_path,
        generated_tokens=generated,
        elapsed_seconds=elapsed,
        tokens_per_second=tok_s,
        estimated_read_bytes=result.estimated_read_bytes,
        estimated_read_gib_per_second=gib_s,
        estimated_embedding_read_bytes=result.estimated_embedding_read_bytes,
        estimated_expert_read_bytes=result.estimated_expert_read_bytes,
        estimated_cache_read_bytes=result.estimated_cache_read_bytes,
        estimated_logits_read_bytes=result.estimated_logits_read_bytes,
        prompt_prefill_chunk_tokens=prompt_prefill_chunk_tokens,
        prompt_prefill_chunk_count=prompt_prefill_chunk_count,
        prompt_prefill_estimated_peak_bytes=prompt_prefill_estimated_peak,
        prompt_prefill_total_embedding_read_bytes=(
            prompt_prefill_total_embedding_read
        ),
        prompt_prefill_total_embedding_output_bytes=(
            prompt_prefill_total_embedding_output
        ),
        prompt_prefill_total_staged_bytes=prompt_prefill_total_staged,
        prompt_prefill_total_compact_stage_bytes=(
            prompt_prefill_total_compact_stage
        ),
        prompt_prefill_total_compact_stage_materialized_bytes=(
            prompt_prefill_total_compact_stage_materialized
        ),
        prompt_prefill_max_staged_bytes=prompt_prefill_max_staged,
        prompt_prefill_max_compact_stage_bytes=prompt_prefill_max_compact_stage,
        prompt_prefill_max_compact_stage_materialized_bytes=(
            prompt_prefill_max_compact_stage_materialized
        ),
        prompt_prefill_total_stage_plus_compact_bytes=(
            prompt_prefill_total_stage_plus_compact
        ),
        prompt_prefill_total_stage_plus_compact_materialized_bytes=(
            prompt_prefill_total_stage_plus_compact_materialized
        ),
        prompt_prefill_max_stage_plus_compact_bytes=(
            prompt_prefill_max_stage_plus_compact
        ),
        prompt_prefill_max_stage_plus_compact_materialized_bytes=(
            prompt_prefill_max_stage_plus_compact_materialized
        ),
        linear_backend_counts=linear_backend_counts,
        total_routed_expert_assignments=total_routed_assignments,
        total_routed_unique_expert_slots=total_routed_unique_slots,
        max_routed_unique_experts_per_call=max_routed_unique_per_call,
        max_routed_tokens_per_expert=max_routed_tokens_per_expert,
        total_expert_stage_serial_read_bytes=total_expert_stage_serial_read,
        total_expert_stage_unique_requested_bytes=(
            total_expert_stage_unique_requested
        ),
        total_expert_stage_planned_read_bytes=total_expert_stage_planned_read,
        total_expert_stage_waste_bytes=total_expert_stage_waste,
        total_expert_stage_coalesced_savings_bytes=(
            total_expert_stage_coalesced_savings
        ),
        total_expert_stage_planned_read_seconds=(
            total_expert_stage_planned_read_seconds
        ),
        total_expert_stage_copy_elapsed_seconds=(
            total_expert_stage_copy_elapsed_seconds
        ),
        total_expert_stage_copy_throughput_gib_per_second=(
            total_expert_stage_copy_throughput_gib_per_second
        ),
        prefill_ssd_read_gib_per_second=ssd_read_gib_per_second,
        prefill_max_routed_read_seconds=prefill_max_routed_read_seconds,
        total_expert_stage_read_seconds_ok=total_expert_stage_read_seconds_ok,
        total_expert_stage_copy_seconds_ok=total_expert_stage_copy_seconds_ok,
        prefill_max_stage_raw_ranges=prefill_max_stage_raw_ranges,
        prefill_max_stage_coalesced_ranges=prefill_max_stage_coalesced_ranges,
        total_expert_stage_raw_ranges=total_expert_stage_raw_ranges,
        total_expert_stage_coalesced_ranges=total_expert_stage_coalesced_ranges,
        max_expert_stage_raw_ranges=max_expert_stage_raw_ranges,
        max_expert_stage_coalesced_ranges=max_expert_stage_coalesced_ranges,
        total_expert_stage_raw_ranges_ok=total_expert_stage_raw_ranges_ok,
        total_expert_stage_coalesced_ranges_ok=(
            total_expert_stage_coalesced_ranges_ok
        ),
        total_expert_stage_read_advice_attempted_ranges=(
            total_expert_stage_read_advice_attempted_ranges
        ),
        total_expert_stage_read_advice_calls=total_expert_stage_read_advice_calls,
        total_expert_stage_read_advice_bytes=total_expert_stage_read_advice_bytes,
        total_expert_stage_read_advice_failures=(
            total_expert_stage_read_advice_failures
        ),
        total_expert_stage_assignment_read_amplification=(
            total_expert_stage_assignment_read_amplification
        ),
        total_expert_stage_unique_read_amplification=(
            total_expert_stage_unique_read_amplification
        ),
        max_expert_stage_unique_read_amplification=(
            max_expert_stage_unique_read_amplification
        ),
        max_expert_stage_stage_budget_utilization=(
            max_expert_stage_stage_budget_utilization
        ),
        max_effective_moe_token_block=max_effective_moe_token_block,
        max_moe_max_expert_tokens=max_moe_max_expert_tokens,
        max_moe_batch_buffer_bytes=max_moe_batch_buffer_bytes,
        max_moe_estimated_peak_bytes=max_moe_estimated_peak_bytes,
        prefill_static_capacity_per_expert=prefill_static_capacity_per_expert,
        max_static_capacity_per_expert=max_static_capacity_per_expert,
        total_static_capacity_used_slots=total_static_capacity_used_slots,
        total_static_capacity_slots=total_static_capacity_slots,
        total_static_capacity_overflow_assignments=total_static_capacity_overflow,
        total_static_capacity_binary_bytes=total_static_capacity_binary_bytes,
        total_linear_matrix_scratch_bytes=total_linear_matrix_scratch,
        max_linear_matrix_scratch_bytes=max_linear_matrix_scratch,
        total_linear_matrix_f32_bytes=total_linear_matrix_f32,
        total_linear_matrix_raw_conversion_bytes=total_linear_matrix_raw_conversion,
        linear_backend_flops=linear_backend_flops,
        linear_backend_elapsed_seconds=linear_backend_elapsed_seconds,
        linear_backend_estimated_tflops=linear_backend_estimated_tflops,
        total_linear_estimated_flops=total_linear_estimated_flops,
        accelerated_linear_estimated_flops=accelerated_linear_estimated_flops,
        custom_linear_estimated_flops=custom_linear_estimated_flops,
        unsupported_linear_estimated_flops=unsupported_linear_estimated_flops,
        accelerated_linear_flop_fraction=accelerated_linear_flop_fraction,
        suggested_guard_flags=suggested_guard_flags,
        suggested_stage_temp_guard_flags=suggested_stage_temp_guard_flags,
        suggested_prefill_guard_flags=suggested_prefill_guard_flags,
        suggested_decode_guard_flags=suggested_decode_guard_flags,
        suggested_launch_profile=suggested_launch_profile,
        routed_chunk_frontier=routed_chunk_frontier,
        prefill_acceleration_coverage=prefill_acceleration_coverage,
        prefill_acceleration_frontier=prefill_acceleration_frontier,
        token_result=result,
    )


def benchmark_prepared_token_ids(
    prepared_path: str | Path,
    *,
    prompt_token_ids: Iterable[int],
    max_new_tokens: int,
    runner_path: str | Path,
    auto_batch_prefill_prompt: bool = True,
    require_prefill_acceleration: bool = False,
    allow_router_gate_only_prefill_acceleration: bool = False,
    prefill_min_accelerated_flop_fraction: float = 0.0,
    require_prepared_runtime_profile: bool = True,
    require_prepared_memory_profile: bool = False,
    require_glm_4bit: bool = False,
    require_public_glm_5_2_shape: bool = False,
    compile_mpp_probe: bool = False,
    run_mpp_probe: bool = False,
    run_mpsgraph_probe: bool = False,
    **generation_kwargs: object,
) -> GenerationBenchmark:
    if max_new_tokens <= 0:
        raise BenchmarkError("max_new_tokens must be positive for a benchmark")
    if not 0.0 <= float(prefill_min_accelerated_flop_fraction) <= 1.0:
        raise BenchmarkError("prefill_min_accelerated_flop_fraction must be 0..1")
    _require_no_missing_dsa_indexer_for_public_glm_5_2(
        require_public_glm_5_2_shape=bool(require_public_glm_5_2_shape),
        generation_kwargs=generation_kwargs,
    )
    prepared = load_prepared_manifest(prepared_path)
    memory_profile_required = bool(
        require_prepared_memory_profile or require_public_glm_5_2_shape
    )
    if require_glm_4bit or require_public_glm_5_2_shape:
        from .server import (
            _public_glm_5_2_shape_failure_reason,
            prepared_glm_4bit_readiness,
        )

        readiness = prepared_glm_4bit_readiness(prepared)
        if readiness.get("ok") is not True:
            issues = readiness.get("issues")
            if isinstance(issues, list) and issues:
                detail = "; ".join(str(issue) for issue in issues[:5])
            else:
                detail = "readiness check did not pass"
            raise BenchmarkError(
                "prepared GLM 4bit readiness failed: "
                f"{detail}"
            )
        if (
            require_public_glm_5_2_shape
            and readiness.get("matches_public_glm_5_2_shape") is not True
        ):
            reason = (
                _public_glm_5_2_shape_failure_reason(readiness)
                or "prepared config does not match the public GLM-5.2 shape"
            )
            raise BenchmarkError(
                "prepared public GLM-5.2 readiness failed: "
                f"{reason}"
            )
    if require_prepared_runtime_profile:
        from .server import (
            prepared_runtime_profile_failure_reason,
        )

        detail = prepared_runtime_profile_failure_reason(
            prepared,
            require_memory_profile=memory_profile_required,
        )
        if detail is not None:
            raise BenchmarkError(
                "prepared runtime profile check failed: "
                f"{detail}"
            )
    try:
        prompt = tuple(prompt_token_ids)
    except TypeError as exc:
        raise BenchmarkError("prompt_token_ids must be iterable") from exc
    if (
        auto_batch_prefill_prompt
        and len(prompt) > 1
        and "batch_prefill_prompt" not in generation_kwargs
    ):
        generation_kwargs["batch_prefill_prompt"] = True
    if (
        generation_kwargs.get("batch_prefill_prompt") is True
        and "prefill_prompt_chunk_tokens" not in generation_kwargs
    ):
        generation_kwargs["prefill_prompt_chunk_tokens"] = 0
    if (
        generation_kwargs.get("batch_prefill_prompt") is True
        and "prefill_static_capacity_per_expert" not in generation_kwargs
    ):
        generation_kwargs["prefill_static_capacity_per_expert"] = "auto"
    generation_kwargs["prefill_min_accelerated_flop_fraction"] = float(
        prefill_min_accelerated_flop_fraction
    )
    generation_kwargs["allow_router_gate_only_prefill_acceleration"] = bool(
        allow_router_gate_only_prefill_acceleration
    )
    if (
        "max_live_working_set_mib" not in generation_kwargs
        and prepared.recommended_max_live_working_set_bytes is not None
    ):
        generation_kwargs["max_live_working_set_mib"] = (
            prepared.recommended_max_live_working_set_bytes / 1024**2
        )
    if (
        "min_free_unified_memory_gib" not in generation_kwargs
        and prepared.recommended_min_free_unified_memory_bytes is not None
    ):
        generation_kwargs["min_free_unified_memory_gib"] = (
            prepared.recommended_min_free_unified_memory_bytes / 1024**3
        )
    if (
        "allow_tied_embeddings" not in generation_kwargs
        or "expected_vocab_size" not in generation_kwargs
        or "expected_hidden_size" not in generation_kwargs
    ):
        from .config import ConfigError, load_config

        try:
            cfg = load_config(prepared.model_dir)
        except (ConfigError, OSError, json.JSONDecodeError):
            cfg = None
        if cfg is not None:
            if "expected_vocab_size" not in generation_kwargs:
                generation_kwargs["expected_vocab_size"] = cfg.vocab_size
            if "expected_hidden_size" not in generation_kwargs:
                generation_kwargs["expected_hidden_size"] = cfg.hidden_size
            if "allow_tied_embeddings" not in generation_kwargs:
                generation_kwargs["allow_tied_embeddings"] = (
                    cfg.tie_word_embeddings is not False
                )
    runtime_prefill_linear_backend = _require_benchmark_request_admission(
        prepared=prepared,
        runner_path=runner_path,
        prompt_token_count=len(prompt),
        max_new_tokens=max_new_tokens,
        require_prefill_acceleration=bool(require_prefill_acceleration),
        prefill_min_accelerated_flop_fraction=(
            float(prefill_min_accelerated_flop_fraction)
        ),
        require_glm_4bit=bool(require_glm_4bit),
        require_public_glm_5_2_shape=bool(require_public_glm_5_2_shape),
        compile_mpp_probe=bool(compile_mpp_probe),
        run_mpp_probe=bool(run_mpp_probe),
        run_mpsgraph_probe=bool(run_mpsgraph_probe),
        generation_kwargs=generation_kwargs,
    )
    if str(generation_kwargs.get("prefill_linear_backend", "auto") or "auto") == "auto":
        generation_kwargs["prefill_linear_backend"] = runtime_prefill_linear_backend
    prepared_lock = _acquire_benchmark_prepared_generation_lock(prepared)
    try:
        result = generate_token_ids(
            runner_path=runner_path,
            expert_layout_path=prepared.experts_layout,
            resident_layout_path=prepared.resident_layout,
            cache_layout_path=prepared.decode_cache_layout,
            cache_file_path=prepared.decode_cache_file,
            prompt_token_ids=prompt,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
    finally:
        if prepared_lock is not None:
            prepared_lock.close()
    if require_prefill_acceleration or prefill_min_accelerated_flop_fraction > 0.0:
        reason = prompt_prefill_acceleration_failure_reason(
            result.prompt_prefill,
            min_accelerated_flop_fraction=prefill_min_accelerated_flop_fraction,
            allow_router_gate_only_acceleration=bool(
                allow_router_gate_only_prefill_acceleration
            ),
        )
        if reason is not None:
            raise BenchmarkError(
                "prefill acceleration actual coverage failed: "
                f"{reason}"
            )
    ssd_read_gib_per_second = generation_kwargs.get(
        "prefill_ssd_read_gib_per_second",
        0.0,
    )
    prefill_max_routed_read_seconds = generation_kwargs.get(
        "prefill_max_routed_read_seconds",
        0.0,
    )
    raw_top_k = generation_kwargs.get("top_k", 8)
    top_k = 8 if raw_top_k is None else int(raw_top_k)
    raw_stage_align = generation_kwargs.get("prefill_expert_stage_align_kib", 4.0)
    stage_align_kib = 4.0 if raw_stage_align is None else float(raw_stage_align)
    raw_prefill_linear_backend = generation_kwargs.get("prefill_linear_backend", "auto")
    prefill_linear_backend = (
        "auto" if raw_prefill_linear_backend is None else str(raw_prefill_linear_backend)
    )
    raw_mpsgraph_min_batch_tokens = generation_kwargs.get(
        "prefill_mpsgraph_min_batch_tokens",
        AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    )
    mpsgraph_min_batch_tokens = (
        AUTO_MPSGRAPH_MIN_BATCH_TOKENS
        if raw_mpsgraph_min_batch_tokens is None
        else int(raw_mpsgraph_min_batch_tokens)
    )
    raw_mpsgraph_min_matrix_dim = generation_kwargs.get(
        "prefill_mpsgraph_min_matrix_dim",
        AUTO_MPSGRAPH_MIN_DIM,
    )
    mpsgraph_min_matrix_dim = (
        AUTO_MPSGRAPH_MIN_DIM
        if raw_mpsgraph_min_matrix_dim is None
        else int(raw_mpsgraph_min_matrix_dim)
    )
    return summarize_generation(
        prepared,
        result,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
        prefill_max_routed_read_seconds=prefill_max_routed_read_seconds,
        top_k=top_k,
        stage_align_bytes=int(stage_align_kib * 1024),
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=mpsgraph_min_matrix_dim,
        require_prepared_memory_profile=memory_profile_required,
        require_glm_4bit=bool(require_glm_4bit),
        require_public_glm_5_2_shape=bool(require_public_glm_5_2_shape),
        require_prefill_acceleration=bool(require_prefill_acceleration),
        allow_router_gate_only_prefill_acceleration=bool(
            allow_router_gate_only_prefill_acceleration
        ),
        prefill_min_accelerated_flop_fraction=float(
            prefill_min_accelerated_flop_fraction
        ),
        compile_mpp_probe=bool(compile_mpp_probe),
        run_mpp_probe=bool(run_mpp_probe),
        run_mpsgraph_probe=bool(run_mpsgraph_probe),
    )
