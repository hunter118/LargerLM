from __future__ import annotations

import operator
import math
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig, load_config
from .expert_io import (
    ExpertIOPlanError,
    static_expert_capacity_binary_bytes_for_counts,
)
from .planner import ExpertLayout
from .prefill_backend import (
    DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
    PrefillBackendCapability,
    suggested_prefill_acceleration_flags,
)
from .prefill_execute import (
    AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    AUTO_MPSGRAPH_MIN_DIM,
    PREFILL_LINEAR_BACKENDS,
)
from .routed_read import (
    combine_prefill_guard_flags,
    format_routed_read_guard_flag_float,
    suggest_routed_read_guard_flags,
    suggest_routed_stage_temp_guard_flags,
)


class PrefillPlanError(RuntimeError):
    """Raised when prefill planning arguments are invalid."""


DEFAULT_MPP_MIN_TOKENS = 128
MPP_TENSOR_OPS_MIN_MATRIX_DIM = 32


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise PrefillPlanError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise PrefillPlanError(f"{name} must be an integer") from exc


def _positive_integer_value(name: str, value: object) -> int:
    parsed = _integer_value(name, value)
    if parsed <= 0:
        raise PrefillPlanError(f"{name} must be positive")
    return parsed


def _optional_positive_float_value(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillPlanError(f"{name} must be a finite positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise PrefillPlanError(f"{name} must be a finite positive number")
    return parsed


def _fraction_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillPlanError(f"{name} must be a finite number between 0 and 1")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise PrefillPlanError(f"{name} must be a finite number between 0 and 1")
    return parsed


def _normalize_static_capacity_per_expert(value: object | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PrefillPlanError(
            "prefill_static_capacity_per_expert must be a positive integer, "
            "'auto', or 'none'"
        )
    if type(value) is int:
        if value <= 0:
            raise PrefillPlanError(
                "prefill_static_capacity_per_expert must be a positive integer, "
                "'auto', or 'none'"
            )
        return value
    if not isinstance(value, str):
        raise PrefillPlanError(
            "prefill_static_capacity_per_expert must be a positive integer, "
            "'auto', or 'none'"
        )
    text = value.strip().lower()
    if text in {"", "none"}:
        return None
    if text == "auto":
        return "auto"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise PrefillPlanError(
            "prefill_static_capacity_per_expert must be a positive integer, "
            "'auto', or 'none'"
        ) from exc
    if parsed <= 0:
        raise PrefillPlanError(
            "prefill_static_capacity_per_expert must be a positive integer, "
            "'auto', or 'none'"
        )
    return parsed


@dataclass(frozen=True)
class PrefillTilePlan:
    simdgroup_tile_m: int
    simdgroup_tile_n: int
    simdgroups_m: int
    simdgroups_n: int
    threadgroup_tile_m: int
    threadgroup_tile_n: int
    k_tile: int
    grid_m: int
    grid_n: int
    full_threadgroup_tiles: int
    edge_threadgroup_tiles: int
    aligned_m: bool
    aligned_n: bool
    aligned_k: bool
    static_extent_full_tiles: bool
    accumulator_bytes_per_threadgroup: int
    arithmetic_intensity_ops_per_byte: float


@dataclass(frozen=True)
class PrefillChunkPlan:
    max_activation_bytes: int
    max_activation_bytes_per_token: int
    recommended_chunk_tokens: int
    chunks: int
    tile_aligned: bool
    threadgroup_tile_m: int
    estimated_peak_activation_bytes: int
    activation_limited: bool


@dataclass(frozen=True)
class RoutedExpertCapacityPlan:
    capacity_tokens: int
    chunks_per_prompt: int
    assignments_per_capacity_chunk: int
    unique_experts_per_capacity_chunk: int
    activation_dtype_bytes: int
    balanced_capacity_per_expert: int
    balanced_capacity_assignments_per_capacity_chunk: int
    balanced_capacity_overprovision_assignments_per_capacity_chunk: int
    balanced_capacity_utilization: float
    balanced_capacity_activation_bytes_per_moe_layer: int
    spill_free_capacity_per_expert: int
    spill_free_capacity_assignments_per_capacity_chunk: int
    spill_free_capacity_overprovision_assignments_per_capacity_chunk: int
    spill_free_capacity_utilization: float
    spill_free_capacity_activation_bytes_per_moe_layer: int
    requires_overflow_path_for_balanced_capacity: bool
    backend_hint: str


@dataclass(frozen=True)
class StagedMoERunnerScratchPlan:
    capacity_tokens: int
    assignments_per_capacity_chunk: int
    assignment_table_bytes_per_moe_layer: int
    token_seen_bytes_per_moe_layer: int
    max_expert_tokens_per_capacity_chunk: int
    auto_token_block: int
    token_block_buffer_bytes_per_moe_layer: int
    expert_slot_alloc_bytes: int
    estimated_peak_bytes_per_moe_layer: int
    max_runner_scratch_bytes: int | None
    fits_runner_scratch: bool | None
    backend_hint: str


@dataclass(frozen=True)
class RoutedExpertReadCostPlan:
    baseline_read_bytes: int
    planned_read_bytes: int
    extra_read_bytes: int
    read_amplification: float
    chunks_per_prompt: int
    ssd_read_bytes_per_second: float | None
    baseline_read_seconds: float | None
    planned_read_seconds: float | None
    extra_read_seconds: float | None
    backend_hint: str


@dataclass(frozen=True)
class RoutedStageTempPlan:
    prompt_chunk_tokens: int
    chunks_per_prompt: int
    stage_align_bytes: int
    static_capacity_per_expert: int | str | None
    max_static_capacity_per_expert: int
    static_capacity_strict_overflow_safe: bool
    max_stage_raw_ranges: int
    max_stage_coalesced_ranges: int
    max_stage_bytes: int
    max_compact_stage_bytes: int
    max_stage_plus_compact_bytes: int
    max_static_capacity_binary_bytes: int
    max_stage_plus_compact_plus_static_bytes: int
    total_stage_bytes: int
    total_compact_stage_bytes: int
    total_stage_plus_compact_bytes: int
    total_static_capacity_binary_bytes: int
    total_stage_plus_compact_plus_static_bytes: int
    backend_hint: str


@dataclass(frozen=True)
class PrefillOp:
    name: str
    layers: int
    m_tokens: int
    n_out: int
    k_in: int
    dtype_bytes: int
    flops_per_layer: int
    weight_bytes_per_layer: int
    activation_bytes_per_layer: int
    backend_hint: str
    tile_plan: PrefillTilePlan | None

    @property
    def total_flops(self) -> int:
        return self.layers * self.flops_per_layer

    @property
    def total_weight_bytes(self) -> int:
        return self.layers * self.weight_bytes_per_layer


@dataclass(frozen=True)
class PrefillCacheIOPlan:
    dtype_bytes: int
    mla_cache_width: int
    dsa_index_head_dim: int | None
    dsa_index_topk: int | None
    indexed_attention_layers: int
    full_attention_layers: int
    dsa_full_indexer_layers: int
    causal_rows_per_layer: int
    indexed_rows_per_layer: int
    mla_cache_read_bytes: int
    dsa_index_cache_read_bytes: int
    total_cache_read_bytes: int
    mla_cache_write_bytes: int
    dsa_index_cache_write_bytes: int
    total_cache_write_bytes: int


@dataclass(frozen=True)
class PrefillBackendCandidate:
    rank: int
    op_name: str
    layers: int
    m_tokens: int
    k_in: int
    n_out: int
    dtype_bytes: int
    total_flops: int
    total_weight_bytes: int
    arithmetic_intensity_ops_per_byte: float
    static_extent_full_tiles: bool
    edge_threadgroup_tiles: int
    preferred_backend: str
    execution_path: str
    availability: str
    reason: str


@dataclass(frozen=True)
class PrefillLinearCalibrationShape:
    rank: int
    candidate_rank: int
    op_name: str
    layers: int
    batch_tokens: int
    in_dim: int
    out_dim: int
    min_matrix_dim: int
    calibration_matrix_bytes: int
    calibration_input_bytes: int
    calibration_output_bytes: int
    calibration_case_bytes: int
    calibration_estimated_peak_bytes: int
    estimated_flops: int
    total_weight_bytes: int
    matrix_shape_arg: str
    candidate_count: int
    candidate_ranks: tuple[int, ...]
    candidate_op_names: tuple[str, ...]
    candidate_layers: int
    candidate_total_flops: int
    candidate_total_weight_bytes: int


@dataclass(frozen=True)
class PrefillPlan:
    model_path: Path
    model_type: str
    public_glm_5_2_shape: dict[str, object]
    prompt_tokens: int
    dtype_bits: int
    expert_bits: int
    group_size: int
    hidden_size: int
    num_layers: int
    moe_layers: int
    dense_layers: int
    total_flops: int
    total_weight_bytes: int
    resident_gemm_weight_bytes: int
    routed_expert_slot_bytes: int
    routed_expert_assignments_per_moe_layer: int
    routed_expert_unique_per_moe_layer: int
    routed_expert_read_bytes: int
    routed_expert_chunked_read_bytes: int
    routed_expert_read_chunks_per_prompt: int
    routed_expert_flops: int
    routed_expert_backend_hint: str
    peak_activation_bytes: int
    mpp_candidate_ops: int
    ops: tuple[PrefillOp, ...]
    prefill_backend_candidates: tuple[PrefillBackendCandidate, ...] = ()
    prefill_linear_calibration_shapes: tuple[PrefillLinearCalibrationShape, ...] = ()
    prefill_linear_calibration_candidate_coverage: dict[str, object] | None = None
    backend_capability: PrefillBackendCapability | None = None
    effective_metal4_candidate_ops: int | None = None
    effective_mpp_candidate_ops: int | None = None
    cache_io_plan: PrefillCacheIOPlan | None = None
    chunk_plan: PrefillChunkPlan | None = None
    routed_expert_read_cost_plan: RoutedExpertReadCostPlan | None = None
    routed_stage_temp_plan: RoutedStageTempPlan | None = None
    suggested_guard_flags: dict[str, object] | None = None
    suggested_stage_temp_guard_flags: dict[str, object] | None = None
    suggested_prefill_guard_flags: dict[str, object] | None = None
    suggested_prefill_runtime_policy_flags: dict[str, object] | None = None
    suggested_prefill_linear_calibration_flags: dict[str, object] | None = None
    suggested_public_glm_5_2_shape_guard_flags: dict[str, object] | None = None
    suggested_launch_profile: dict[str, object] | None = None
    routed_expert_capacity_plan: RoutedExpertCapacityPlan | None = None
    staged_moe_runner_scratch_plan: StagedMoERunnerScratchPlan | None = None


def _gemm_op(
    name: str,
    *,
    layers: int,
    tokens: int,
    k_in: int | None,
    n_out: int | None,
    dtype_bytes: int,
    mpp_min_tokens: int,
    simdgroup_tile_m: int,
    simdgroup_tile_n: int,
    simdgroups_m: int,
    simdgroups_n: int,
    k_tile: int,
) -> PrefillOp | None:
    if layers <= 0 or tokens <= 0 or k_in is None or n_out is None:
        return None
    if k_in <= 0 or n_out <= 0:
        return None
    flops = 2 * tokens * int(k_in) * int(n_out)
    weight_bytes = int(k_in) * int(n_out) * dtype_bytes
    activation_bytes = tokens * (int(k_in) + int(n_out)) * dtype_bytes
    backend = (
        "mpp_tensor_ops_candidate"
        if (
            tokens >= mpp_min_tokens
            and int(k_in) >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
            and int(n_out) >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
        )
        else "custom_metal_or_small_gemm"
    )
    tile_plan = _tile_plan(
        tokens=tokens,
        k_in=int(k_in),
        n_out=int(n_out),
        dtype_bytes=dtype_bytes,
        simdgroup_tile_m=simdgroup_tile_m,
        simdgroup_tile_n=simdgroup_tile_n,
        simdgroups_m=simdgroups_m,
        simdgroups_n=simdgroups_n,
        k_tile=k_tile,
    )
    return PrefillOp(
        name=name,
        layers=layers,
        m_tokens=tokens,
        n_out=int(n_out),
        k_in=int(k_in),
        dtype_bytes=dtype_bytes,
        flops_per_layer=flops,
        weight_bytes_per_layer=weight_bytes,
        activation_bytes_per_layer=activation_bytes,
        backend_hint=backend,
        tile_plan=tile_plan if backend == "mpp_tensor_ops_candidate" else None,
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _backend_candidate_path(
    backend_capability: PrefillBackendCapability | None,
) -> tuple[str, str, str, str]:
    if backend_capability is None:
        return (
            "mpp_tensor_ops_prefill",
            "mpp_tensor_ops_gpu_neural_accelerator",
            "not_inspected",
            "run prefill-plan --inspect-backend --compile-mpp-probe to confirm local runtime support",
        )
    if backend_capability.mpp_runtime_available:
        return (
            "mpp_tensor_ops_prefill",
            "mpp_tensor_ops_gpu_neural_accelerator",
            "available",
            "Metal 4 ML and MPP tensor-op probes succeeded",
        )
    if backend_capability.mps_graph_runtime_available:
        if (
            backend_capability.mpp_tensor_ops_symbol_declared
            and not backend_capability.mpp_compile_probe_ran
        ):
            reason = (
                "MPP symbols are visible but the compile probe was not run; "
                "MPSGraph is the safe fallback"
            )
        elif backend_capability.reasons:
            reason = backend_capability.reasons[0]
        else:
            reason = (
                "mpp::tensor_ops symbols are unavailable; "
                "MPSGraph matmul is available"
            )
        return (
            "mpsgraph_prefill_fallback",
            "mpsgraph_gpu_matmul",
            "fallback",
            reason,
        )
    reason = (
        backend_capability.reasons[0]
        if backend_capability.reasons
        else "no MPP tensor-op or MPSGraph matmul support was detected"
    )
    return (
        "custom_metal_tile",
        "custom_metal_gpu_fallback",
        "fallback",
        reason,
    )


def _prefill_backend_candidates(
    ops: list[PrefillOp],
    backend_capability: PrefillBackendCapability | None,
) -> tuple[PrefillBackendCandidate, ...]:
    preferred_backend, execution_path, availability, reason = _backend_candidate_path(
        backend_capability,
    )
    candidates = sorted(
        (op for op in ops if op.backend_hint == "mpp_tensor_ops_candidate"),
        key=lambda op: (-op.total_flops, -op.total_weight_bytes, op.name),
    )
    plans: list[PrefillBackendCandidate] = []
    for rank, op in enumerate(candidates, start=1):
        tile = op.tile_plan
        plans.append(
            PrefillBackendCandidate(
                rank=rank,
                op_name=op.name,
                layers=op.layers,
                m_tokens=op.m_tokens,
                k_in=op.k_in,
                n_out=op.n_out,
                dtype_bytes=op.dtype_bytes,
                total_flops=op.total_flops,
                total_weight_bytes=op.total_weight_bytes,
                arithmetic_intensity_ops_per_byte=(
                    tile.arithmetic_intensity_ops_per_byte if tile is not None else 0.0
                ),
                static_extent_full_tiles=(
                    tile.static_extent_full_tiles if tile is not None else False
                ),
                edge_threadgroup_tiles=(
                    tile.edge_threadgroup_tiles if tile is not None else 0
                ),
                preferred_backend=preferred_backend,
                execution_path=execution_path,
                availability=availability,
                reason=reason,
            )
        )
    return tuple(plans)


_PREFILL_LINEAR_CALIBRATION_SHAPE_LIMIT = 8
_CALIBRATION_F32_BYTES = 4
_CALIBRATION_SCRATCH_ALIGNMENT = 2 * 1024 * 1024
_CALIBRATION_HEADROOM = 1.10


def _ceil_mib_with_headroom(value: int) -> int:
    return max(1, math.ceil(value * _CALIBRATION_HEADROOM / (1024 * 1024)))


def _prefill_linear_calibration_shapes(
    candidates: tuple[PrefillBackendCandidate, ...],
    *,
    batch_tokens: int,
    limit: int = _PREFILL_LINEAR_CALIBRATION_SHAPE_LIMIT,
) -> tuple[PrefillLinearCalibrationShape, ...]:
    if limit <= 0:
        return ()
    candidates_by_shape: dict[tuple[int, int], list[PrefillBackendCandidate]] = {}
    for candidate in candidates:
        candidates_by_shape.setdefault((candidate.k_in, candidate.n_out), []).append(
            candidate
        )
    shapes: list[PrefillLinearCalibrationShape] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        key = (candidate.k_in, candidate.n_out)
        if key in seen:
            continue
        seen.add(key)
        shape_candidates = tuple(candidates_by_shape[key])
        matrix_bytes = candidate.k_in * candidate.n_out * _CALIBRATION_F32_BYTES
        input_bytes = batch_tokens * candidate.k_in * _CALIBRATION_F32_BYTES
        output_bytes = batch_tokens * candidate.n_out * _CALIBRATION_F32_BYTES
        case_bytes = matrix_bytes + input_bytes + output_bytes
        estimated_peak_bytes = (
            _align_up(matrix_bytes, _CALIBRATION_SCRATCH_ALIGNMENT)
            + input_bytes
            + output_bytes
        )
        shapes.append(
            PrefillLinearCalibrationShape(
                rank=len(shapes) + 1,
                candidate_rank=candidate.rank,
                op_name=candidate.op_name,
                layers=candidate.layers,
                batch_tokens=batch_tokens,
                in_dim=candidate.k_in,
                out_dim=candidate.n_out,
                min_matrix_dim=min(candidate.k_in, candidate.n_out),
                calibration_matrix_bytes=matrix_bytes,
                calibration_input_bytes=input_bytes,
                calibration_output_bytes=output_bytes,
                calibration_case_bytes=case_bytes,
                calibration_estimated_peak_bytes=estimated_peak_bytes,
                estimated_flops=2 * batch_tokens * candidate.k_in * candidate.n_out,
                total_weight_bytes=candidate.total_weight_bytes,
                matrix_shape_arg=f"{candidate.k_in}x{candidate.n_out}",
                candidate_count=len(shape_candidates),
                candidate_ranks=tuple(item.rank for item in shape_candidates),
                candidate_op_names=tuple(item.op_name for item in shape_candidates),
                candidate_layers=sum(item.layers for item in shape_candidates),
                candidate_total_flops=sum(item.total_flops for item in shape_candidates),
                candidate_total_weight_bytes=sum(
                    item.total_weight_bytes for item in shape_candidates
                ),
            )
        )
        if len(shapes) >= limit:
            break
    return tuple(shapes)


def _prefill_linear_calibration_candidate_coverage(
    candidates: tuple[PrefillBackendCandidate, ...],
    shapes: tuple[PrefillLinearCalibrationShape, ...],
    *,
    source: str,
) -> dict[str, object] | None:
    if not candidates or not shapes:
        return None
    covered_ranks = tuple(
        rank
        for shape in shapes
        for rank in shape.candidate_ranks
    )
    covered_rank_set = set(covered_ranks)
    planned_total_flops = sum(candidate.total_flops for candidate in candidates)
    covered_total_flops = sum(
        candidate.total_flops
        for candidate in candidates
        if candidate.rank in covered_rank_set
    )
    planned_total_weight_bytes = sum(
        candidate.total_weight_bytes for candidate in candidates
    )
    covered_total_weight_bytes = sum(
        candidate.total_weight_bytes
        for candidate in candidates
        if candidate.rank in covered_rank_set
    )
    shape_payloads = tuple(
        {
            "rank": shape.rank,
            "matrix_shape_arg": shape.matrix_shape_arg,
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
            "batch_tokens": shape.batch_tokens,
            "min_matrix_dim": shape.min_matrix_dim,
            "candidate_count": shape.candidate_count,
            "candidate_ranks": shape.candidate_ranks,
            "candidate_op_names": shape.candidate_op_names,
            "candidate_layers": shape.candidate_layers,
            "candidate_total_flops": shape.candidate_total_flops,
            "candidate_total_weight_bytes": shape.candidate_total_weight_bytes,
            "calibration_case_bytes": shape.calibration_case_bytes,
            "calibration_estimated_peak_bytes": (
                shape.calibration_estimated_peak_bytes
            ),
            "calibration_estimated_flops": shape.estimated_flops,
        }
        for shape in shapes
    )
    return {
        "source": source,
        "candidate_count": len(candidates),
        "covered_candidate_count": len(covered_rank_set),
        "covered_candidate_ranks": tuple(sorted(covered_rank_set)),
        "uncovered_candidate_ranks": tuple(
            candidate.rank
            for candidate in candidates
            if candidate.rank not in covered_rank_set
        ),
        "unique_shape_count": len(shapes),
        "shape_limit": _PREFILL_LINEAR_CALIBRATION_SHAPE_LIMIT,
        "coverage_truncated": len(covered_rank_set) < len(candidates),
        "planned_candidate_total_flops": planned_total_flops,
        "covered_candidate_total_flops": covered_total_flops,
        "covered_candidate_flop_fraction": (
            covered_total_flops / planned_total_flops
            if planned_total_flops > 0
            else 0.0
        ),
        "planned_candidate_total_weight_bytes": planned_total_weight_bytes,
        "covered_candidate_total_weight_bytes": covered_total_weight_bytes,
        "covered_candidate_weight_fraction": (
            covered_total_weight_bytes / planned_total_weight_bytes
            if planned_total_weight_bytes > 0
            else 0.0
        ),
        "shapes": shape_payloads,
    }


def _suggest_prefill_linear_calibration_flags(
    shapes: tuple[PrefillLinearCalibrationShape, ...],
    *,
    source: str,
) -> dict[str, object] | None:
    if not shapes:
        return None
    batch_tokens = tuple(sorted({shape.batch_tokens for shape in shapes}))
    matrix_shapes = tuple(shape.matrix_shape_arg for shape in shapes)
    max_case_bytes = max(shape.calibration_case_bytes for shape in shapes)
    max_matrix_bytes = max(shape.calibration_matrix_bytes for shape in shapes)
    max_peak_bytes = max(shape.calibration_estimated_peak_bytes for shape in shapes)
    max_calibration_case_mib = _ceil_mib_with_headroom(max_case_bytes)
    max_resident_matrix_mib = _ceil_mib_with_headroom(max_matrix_bytes)
    max_runner_scratch_mib = _ceil_mib_with_headroom(max_peak_bytes)
    argv = (
        "--batch-tokens",
        ",".join(str(value) for value in batch_tokens),
        "--matrix-shapes",
        ",".join(matrix_shapes),
        "--max-calibration-case-mib",
        str(max_calibration_case_mib),
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    )
    return {
        "source": source,
        "command": "prefill-linear-calibrate",
        "runner_argument_required": True,
        "batch_tokens": batch_tokens,
        "matrix_shapes": matrix_shapes,
        "shape_limit": _PREFILL_LINEAR_CALIBRATION_SHAPE_LIMIT,
        "max_calibration_case_bytes": max_case_bytes,
        "max_resident_matrix_bytes": max_matrix_bytes,
        "max_runner_scratch_bytes": max_peak_bytes,
        "max_calibration_case_mib": max_calibration_case_mib,
        "max_resident_matrix_mib": max_resident_matrix_mib,
        "max_runner_scratch_mib": max_runner_scratch_mib,
        "argv": argv,
    }


_PROFILE_VALUELESS_FLAGS = frozenset(
    {
        "--require-prefill-acceleration",
        "--require-public-glm-5-2-shape",
        "--compile-mpp-probe",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
    }
)


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
        if flag in _PROFILE_VALUELESS_FLAGS:
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


def _suggest_prefill_backend_probe_flags(
    *,
    compile_mpp_probe: bool,
    run_mpp_probe: bool,
    run_mpsgraph_probe: bool,
    probe_timeout_seconds: float = DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
) -> dict[str, object] | None:
    if not compile_mpp_probe and not run_mpp_probe and not run_mpsgraph_probe:
        return None
    if isinstance(probe_timeout_seconds, bool):
        raise PrefillPlanError("probe_timeout_seconds must be positive")
    try:
        timeout = float(probe_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise PrefillPlanError("probe_timeout_seconds must be positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise PrefillPlanError("probe_timeout_seconds must be positive")
    argv: list[str] = []
    payload: dict[str, object] = {"source": "prefill_plan"}
    if compile_mpp_probe:
        payload["compile_mpp_probe"] = True
        argv.append("--compile-mpp-probe")
    if run_mpp_probe:
        payload["run_mpp_probe"] = True
        argv.append("--run-mpp-probe")
    if run_mpsgraph_probe:
        payload["run_mpsgraph_probe"] = True
        argv.append("--run-mpsgraph-probe")
    if timeout != DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS:
        payload["prefill_backend_probe_timeout_seconds"] = timeout
        argv.extend(
            [
                "--prefill-backend-probe-timeout-seconds",
                format_routed_read_guard_flag_float(timeout),
            ]
        )
    payload["argv"] = tuple(argv)
    return {
        **payload,
    }


def _suggest_prefill_runtime_policy_flags(
    *,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int,
    prefill_mpsgraph_min_matrix_dim: int,
    prefill_min_accelerated_flop_fraction: float,
    require_prefill_acceleration: bool,
    source: str,
) -> dict[str, object] | None:
    if (
        prefill_linear_backend == "auto"
        and prefill_mpsgraph_min_batch_tokens == AUTO_MPSGRAPH_MIN_BATCH_TOKENS
        and prefill_mpsgraph_min_matrix_dim == AUTO_MPSGRAPH_MIN_DIM
        and prefill_min_accelerated_flop_fraction == 0.0
        and not require_prefill_acceleration
    ):
        return None

    argv: list[str] = []
    payload: dict[str, object] = {"source": source}
    if prefill_linear_backend != "auto":
        payload["prefill_linear_backend"] = prefill_linear_backend
        argv.extend(["--prefill-linear-backend", prefill_linear_backend])
    payload["prefill_mpsgraph_min_batch_tokens"] = prefill_mpsgraph_min_batch_tokens
    payload["prefill_mpsgraph_min_matrix_dim"] = prefill_mpsgraph_min_matrix_dim
    argv.extend(
        [
            "--prefill-mpsgraph-min-batch-tokens",
            str(prefill_mpsgraph_min_batch_tokens),
            "--prefill-mpsgraph-min-matrix-dim",
            str(prefill_mpsgraph_min_matrix_dim),
        ]
    )
    if prefill_min_accelerated_flop_fraction > 0.0:
        payload["prefill_min_accelerated_flop_fraction"] = (
            prefill_min_accelerated_flop_fraction
        )
        argv.extend(
            [
                "--prefill-min-accelerated-flop-fraction",
                format_routed_read_guard_flag_float(
                    prefill_min_accelerated_flop_fraction
                ),
            ]
        )
    if require_prefill_acceleration:
        payload["require_prefill_acceleration"] = True
        argv.append("--require-prefill-acceleration")
    payload["argv"] = tuple(argv)
    return payload


def _public_glm_5_2_shape_report(config: ModelConfig) -> dict[str, object]:
    from .server import _public_glm_5_2_shape_report as report

    return report(config)


def _public_glm_5_2_shape_failure_detail(report: dict[str, object]) -> str | None:
    fields = report.get("mismatched_fields")
    if not isinstance(fields, (list, tuple)) or not fields:
        return None
    preview = ", ".join(str(field) for field in tuple(fields)[:6])
    if len(fields) > 6:
        preview += f", +{len(fields) - 6} more"
    return preview


def _suggest_public_glm_5_2_shape_guard_flags(
    report: dict[str, object],
    *,
    expert_bits: int,
    source: str,
) -> dict[str, object] | None:
    if report.get("matches") is not True or expert_bits != 4:
        return None
    return {
        "source": source,
        "require_public_glm_5_2_shape": True,
        "argv": ("--require-public-glm-5-2-shape",),
    }


def _combine_prefill_launch_profile(
    *,
    prefill_guard_flags: dict[str, object] | None = None,
    prefill_runtime_policy_flags: dict[str, object] | None = None,
    prefill_acceleration_flags: dict[str, object] | None = None,
    prefill_backend_probe_flags: dict[str, object] | None = None,
    public_glm_5_2_shape_guard_flags: dict[str, object] | None = None,
    source: str,
) -> dict[str, object] | None:
    sections = (
        ("prefill_guard_flags", prefill_guard_flags),
        ("prefill_runtime_policy_flags", prefill_runtime_policy_flags),
        ("prefill_acceleration_flags", prefill_acceleration_flags),
        ("prefill_backend_probe_flags", prefill_backend_probe_flags),
        ("public_glm_5_2_shape_guard_flags", public_glm_5_2_shape_guard_flags),
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
    if conflicts:
        profile["argv_conflicts"] = tuple(conflicts)
    return profile


def _tile_plan(
    *,
    tokens: int,
    k_in: int,
    n_out: int,
    dtype_bytes: int,
    simdgroup_tile_m: int,
    simdgroup_tile_n: int,
    simdgroups_m: int,
    simdgroups_n: int,
    k_tile: int,
) -> PrefillTilePlan:
    threadgroup_tile_m = simdgroup_tile_m * simdgroups_m
    threadgroup_tile_n = simdgroup_tile_n * simdgroups_n
    grid_m = _ceil_div(tokens, threadgroup_tile_m)
    grid_n = _ceil_div(n_out, threadgroup_tile_n)
    full_m = tokens // threadgroup_tile_m
    full_n = n_out // threadgroup_tile_n
    full_tiles = full_m * full_n
    total_tiles = grid_m * grid_n
    input_bytes = ((tokens * k_in) + (k_in * n_out)) * dtype_bytes
    output_bytes = tokens * n_out * dtype_bytes
    total_bytes = input_bytes + output_bytes
    intensity = (2 * tokens * k_in * n_out / total_bytes) if total_bytes else 0.0
    return PrefillTilePlan(
        simdgroup_tile_m=simdgroup_tile_m,
        simdgroup_tile_n=simdgroup_tile_n,
        simdgroups_m=simdgroups_m,
        simdgroups_n=simdgroups_n,
        threadgroup_tile_m=threadgroup_tile_m,
        threadgroup_tile_n=threadgroup_tile_n,
        k_tile=k_tile,
        grid_m=grid_m,
        grid_n=grid_n,
        full_threadgroup_tiles=full_tiles,
        edge_threadgroup_tiles=total_tiles - full_tiles,
        aligned_m=tokens % threadgroup_tile_m == 0,
        aligned_n=n_out % threadgroup_tile_n == 0,
        aligned_k=k_in % k_tile == 0,
        static_extent_full_tiles=(
            tokens % threadgroup_tile_m == 0
            and n_out % threadgroup_tile_n == 0
            and k_in % k_tile == 0
        ),
        accumulator_bytes_per_threadgroup=threadgroup_tile_m * threadgroup_tile_n * 4,
        arithmetic_intensity_ops_per_byte=float(intensity),
    )


def _chunk_plan(
    *,
    prompt_tokens: int,
    max_activation_bytes: int | None,
    ops: list[PrefillOp],
    threadgroup_tile_m: int,
) -> PrefillChunkPlan | None:
    if max_activation_bytes is None:
        return None
    per_token = max(
        (op.activation_bytes_per_layer // op.m_tokens for op in ops if op.m_tokens > 0),
        default=0,
    )
    if per_token <= 0:
        return None
    raw_tokens = max_activation_bytes // per_token
    if raw_tokens <= 0:
        raise PrefillPlanError("max_prefill_activation_bytes is too small for one token")
    if raw_tokens >= threadgroup_tile_m:
        chunk_tokens = (raw_tokens // threadgroup_tile_m) * threadgroup_tile_m
        tile_aligned = True
    else:
        chunk_tokens = raw_tokens
        tile_aligned = raw_tokens == threadgroup_tile_m
    chunk_tokens = max(1, min(prompt_tokens, chunk_tokens))
    return PrefillChunkPlan(
        max_activation_bytes=max_activation_bytes,
        max_activation_bytes_per_token=per_token,
        recommended_chunk_tokens=chunk_tokens,
        chunks=_ceil_div(prompt_tokens, chunk_tokens),
        tile_aligned=tile_aligned,
        threadgroup_tile_m=threadgroup_tile_m,
        estimated_peak_activation_bytes=chunk_tokens * per_token,
        activation_limited=chunk_tokens < prompt_tokens,
    )


def _routed_expert_capacity_plan(
    *,
    prompt_tokens: int,
    chunks_per_prompt: int,
    routed_experts: int,
    routed_assignments: int,
    routed_unique: int,
    hidden_size: int,
    moe_hidden_size: int,
    activation_dtype_bytes: int = 4,
) -> RoutedExpertCapacityPlan | None:
    if (
        prompt_tokens <= 0
        or routed_experts <= 0
        or routed_assignments <= 0
        or routed_unique <= 0
        or hidden_size <= 0
        or moe_hidden_size <= 0
        or activation_dtype_bytes <= 0
    ):
        return None

    balanced_capacity = _ceil_div(routed_assignments, routed_unique)
    balanced_assignments = balanced_capacity * routed_unique
    balanced_overprovision = balanced_assignments - routed_assignments
    balanced_utilization = routed_assignments / balanced_assignments

    # If each token selects distinct experts, one expert can receive at most one
    # assignment per prompt token. Keep this no smaller than the balanced floor
    # so malformed tiny configs still produce enough static assignment slots.
    spill_free_capacity = max(prompt_tokens, balanced_capacity)
    spill_free_assignments = spill_free_capacity * routed_unique
    spill_free_overprovision = spill_free_assignments - routed_assignments
    spill_free_utilization = routed_assignments / spill_free_assignments
    slot_activation_width = (2 * hidden_size) + (2 * moe_hidden_size)
    balanced_activation_bytes = (
        balanced_assignments * slot_activation_width * activation_dtype_bytes
    )
    spill_free_activation_bytes = (
        spill_free_assignments * slot_activation_width * activation_dtype_bytes
    )

    backend = (
        "static_capacity_full_expert_sweep_candidate"
        if routed_unique == routed_experts
        else "static_capacity_grouped_routed_expert_candidate"
    )

    return RoutedExpertCapacityPlan(
        capacity_tokens=prompt_tokens,
        chunks_per_prompt=chunks_per_prompt,
        assignments_per_capacity_chunk=routed_assignments,
        unique_experts_per_capacity_chunk=routed_unique,
        activation_dtype_bytes=activation_dtype_bytes,
        balanced_capacity_per_expert=balanced_capacity,
        balanced_capacity_assignments_per_capacity_chunk=balanced_assignments,
        balanced_capacity_overprovision_assignments_per_capacity_chunk=balanced_overprovision,
        balanced_capacity_utilization=float(balanced_utilization),
        balanced_capacity_activation_bytes_per_moe_layer=balanced_activation_bytes,
        spill_free_capacity_per_expert=spill_free_capacity,
        spill_free_capacity_assignments_per_capacity_chunk=spill_free_assignments,
        spill_free_capacity_overprovision_assignments_per_capacity_chunk=spill_free_overprovision,
        spill_free_capacity_utilization=float(spill_free_utilization),
        spill_free_capacity_activation_bytes_per_moe_layer=spill_free_activation_bytes,
        requires_overflow_path_for_balanced_capacity=balanced_capacity < spill_free_capacity,
        backend_hint=backend,
    )


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _staged_moe_runner_scratch_plan(
    *,
    capacity_tokens: int,
    assignments_per_capacity_chunk: int,
    hidden_size: int,
    moe_hidden_size: int,
    expert_slot_bytes: int,
    max_runner_scratch_bytes: int | None = None,
) -> StagedMoERunnerScratchPlan | None:
    if (
        capacity_tokens <= 0
        or assignments_per_capacity_chunk <= 0
        or hidden_size <= 0
        or moe_hidden_size <= 0
        or expert_slot_bytes <= 0
    ):
        return None
    assignment_table_bytes = assignments_per_capacity_chunk * 16
    token_seen_bytes = capacity_tokens
    max_expert_tokens = capacity_tokens
    per_token_block_bytes = (3 * hidden_size * 4) + (3 * moe_hidden_size * 4) + 4
    expert_slot_alloc = _align_up(expert_slot_bytes, 2 * 1024 * 1024)
    route_live_bytes = assignment_table_bytes + token_seen_bytes

    if max_runner_scratch_bytes is None:
        auto_token_block = max_expert_tokens
        fits_runner_scratch: bool | None = None
    else:
        available = max_runner_scratch_bytes - expert_slot_alloc - route_live_bytes
        if available < per_token_block_bytes:
            auto_token_block = 0
            fits_runner_scratch = False
        else:
            auto_token_block = max(1, min(max_expert_tokens, available // per_token_block_bytes))
            fits_runner_scratch = True

    token_block_buffer_bytes = auto_token_block * per_token_block_bytes
    peak = expert_slot_alloc + route_live_bytes + token_block_buffer_bytes
    backend = (
        "auto_token_block_fits_runner_scratch"
        if fits_runner_scratch is not False
        else "auto_token_block_exceeds_runner_scratch"
    )
    return StagedMoERunnerScratchPlan(
        capacity_tokens=capacity_tokens,
        assignments_per_capacity_chunk=assignments_per_capacity_chunk,
        assignment_table_bytes_per_moe_layer=assignment_table_bytes,
        token_seen_bytes_per_moe_layer=token_seen_bytes,
        max_expert_tokens_per_capacity_chunk=max_expert_tokens,
        auto_token_block=auto_token_block,
        token_block_buffer_bytes_per_moe_layer=token_block_buffer_bytes,
        expert_slot_alloc_bytes=expert_slot_alloc,
        estimated_peak_bytes_per_moe_layer=peak,
        max_runner_scratch_bytes=max_runner_scratch_bytes,
        fits_runner_scratch=fits_runner_scratch,
        backend_hint=backend,
    )


def _routed_expert_chunked_read_bytes(
    *,
    prompt_tokens: int,
    chunk_tokens: int,
    experts_per_token: int,
    routed_experts: int,
    moe_layers: int,
    expert_slot_bytes: int,
) -> int:
    if (
        prompt_tokens <= 0
        or chunk_tokens <= 0
        or experts_per_token <= 0
        or routed_experts <= 0
        or moe_layers <= 0
        or expert_slot_bytes <= 0
    ):
        return 0
    per_layer_reads = 0
    remaining_tokens = prompt_tokens
    while remaining_tokens > 0:
        current_tokens = min(chunk_tokens, remaining_tokens)
        current_assignments = current_tokens * experts_per_token
        current_unique = min(routed_experts, current_assignments)
        per_layer_reads += current_unique * expert_slot_bytes
        remaining_tokens -= current_tokens
    return moe_layers * per_layer_reads


def _static_capacity_for_chunk(
    value: int | str | None,
    *,
    chunk_tokens: int,
) -> int | None:
    if value == "auto":
        return chunk_tokens
    return value


def _routed_stage_temp_plan(
    *,
    prompt_tokens: int,
    chunk_tokens: int,
    experts_per_token: int,
    routed_experts: int,
    moe_layers: int,
    expert_slot_bytes: int,
    stage_align_bytes: int,
    static_capacity_per_expert: int | str | None,
    backend_hint: str,
) -> RoutedStageTempPlan | None:
    if (
        prompt_tokens <= 0
        or chunk_tokens <= 0
        or experts_per_token <= 0
        or routed_experts <= 0
        or moe_layers <= 0
        or expert_slot_bytes <= 0
        or stage_align_bytes <= 0
    ):
        return None

    chunks = _ceil_div(prompt_tokens, chunk_tokens)
    total_stage = 0
    total_compact = 0
    total_static = 0
    max_stage = 0
    max_compact = 0
    max_stage_plus_compact = 0
    max_static = 0
    max_stage_plus_compact_plus_static = 0
    max_stage_raw_ranges = 0
    max_stage_coalesced_ranges = 0
    max_static_capacity_per_expert = 0
    strict_overflow_safe = True
    remaining = prompt_tokens
    while remaining > 0:
        current_tokens = min(chunk_tokens, remaining)
        current_unique = min(routed_experts, current_tokens * experts_per_token)
        max_stage_raw_ranges = max(max_stage_raw_ranges, current_unique)
        max_stage_coalesced_ranges = max(max_stage_coalesced_ranges, current_unique)
        stage = current_unique * (expert_slot_bytes + stage_align_bytes)
        compact = current_unique * expert_slot_bytes
        static_bytes = 0
        capacity = _static_capacity_for_chunk(
            static_capacity_per_expert,
            chunk_tokens=current_tokens,
        )
        if capacity is not None:
            try:
                static_bytes = static_expert_capacity_binary_bytes_for_counts(
                    expert_count=current_unique,
                    capacity_per_expert=capacity,
                )
            except ExpertIOPlanError as exc:
                raise PrefillPlanError(str(exc)) from exc
            max_static_capacity_per_expert = max(
                max_static_capacity_per_expert,
                capacity,
            )
            strict_overflow_safe = strict_overflow_safe and capacity >= current_tokens
        stage_plus_compact = stage + compact
        stage_plus_compact_plus_static = stage_plus_compact + static_bytes
        max_stage = max(max_stage, stage)
        max_compact = max(max_compact, compact)
        max_stage_plus_compact = max(max_stage_plus_compact, stage_plus_compact)
        max_static = max(max_static, static_bytes)
        max_stage_plus_compact_plus_static = max(
            max_stage_plus_compact_plus_static,
            stage_plus_compact_plus_static,
        )
        total_stage += stage * moe_layers
        total_compact += compact * moe_layers
        total_static += static_bytes * moe_layers
        remaining -= current_tokens

    total_stage_plus_compact = total_stage + total_compact
    return RoutedStageTempPlan(
        prompt_chunk_tokens=chunk_tokens,
        chunks_per_prompt=chunks,
        stage_align_bytes=stage_align_bytes,
        static_capacity_per_expert=static_capacity_per_expert,
        max_static_capacity_per_expert=max_static_capacity_per_expert,
        static_capacity_strict_overflow_safe=strict_overflow_safe,
        max_stage_raw_ranges=max_stage_raw_ranges,
        max_stage_coalesced_ranges=max_stage_coalesced_ranges,
        max_stage_bytes=max_stage,
        max_compact_stage_bytes=max_compact,
        max_stage_plus_compact_bytes=max_stage_plus_compact,
        max_static_capacity_binary_bytes=max_static,
        max_stage_plus_compact_plus_static_bytes=(
            max_stage_plus_compact_plus_static
        ),
        total_stage_bytes=total_stage,
        total_compact_stage_bytes=total_compact,
        total_stage_plus_compact_bytes=total_stage_plus_compact,
        total_static_capacity_binary_bytes=total_static,
        total_stage_plus_compact_plus_static_bytes=(
            total_stage_plus_compact + total_static
        ),
        backend_hint=backend_hint,
    )


def _routed_expert_read_cost_plan(
    *,
    baseline_read_bytes: int,
    planned_read_bytes: int,
    chunks_per_prompt: int,
    ssd_read_gib_per_second: float | None,
    backend_hint: str,
) -> RoutedExpertReadCostPlan | None:
    if baseline_read_bytes <= 0:
        return None
    planned = planned_read_bytes if planned_read_bytes > 0 else baseline_read_bytes
    extra = max(0, planned - baseline_read_bytes)
    bytes_per_second = (
        ssd_read_gib_per_second * 1024**3
        if ssd_read_gib_per_second is not None
        else None
    )
    return RoutedExpertReadCostPlan(
        baseline_read_bytes=baseline_read_bytes,
        planned_read_bytes=planned,
        extra_read_bytes=extra,
        read_amplification=float(planned / baseline_read_bytes),
        chunks_per_prompt=chunks_per_prompt if chunks_per_prompt > 0 else 1,
        ssd_read_bytes_per_second=bytes_per_second,
        baseline_read_seconds=(
            baseline_read_bytes / bytes_per_second if bytes_per_second else None
        ),
        planned_read_seconds=planned / bytes_per_second if bytes_per_second else None,
        extra_read_seconds=extra / bytes_per_second if bytes_per_second else None,
        backend_hint=backend_hint,
    )


def _causal_prefix_rows(tokens: int) -> int:
    return tokens * (tokens + 1) // 2


def _capped_causal_prefix_rows(tokens: int, cap: int) -> int:
    capped = min(tokens, cap)
    return capped * (capped + 1) // 2 + max(0, tokens - capped) * cap


def build_prefill_cache_io_plan(
    cfg: ModelConfig,
    *,
    prompt_tokens: int,
    dtype_bytes: int,
) -> PrefillCacheIOPlan | None:
    cache_width = cfg.mla_cache_width
    if cache_width is None:
        return None

    dsa_types = tuple(str(item).lower() for item in cfg.indexer_types or ())
    dsa_index_head_dim = cfg.index_head_dim
    dsa_index_topk = cfg.index_topk
    dsa_full_layers = (
        sum(1 for item in dsa_types[: cfg.num_hidden_layers] if item == "full")
        if dsa_index_head_dim is not None
        else 0
    )
    indexed_layers = (
        sum(
            1
            for item in dsa_types[: cfg.num_hidden_layers]
            if item in {"full", "shared"}
        )
        if dsa_index_topk is not None and dsa_index_topk > 0
        else 0
    )
    full_attention_layers = cfg.num_hidden_layers - indexed_layers
    causal_rows = _causal_prefix_rows(prompt_tokens)
    indexed_rows = (
        _capped_causal_prefix_rows(prompt_tokens, int(dsa_index_topk))
        if indexed_layers
        else 0
    )

    mla_cache_read_rows = (
        indexed_layers * indexed_rows + full_attention_layers * causal_rows
    )
    mla_cache_read = mla_cache_read_rows * int(cache_width) * dtype_bytes
    dsa_index_read = (
        dsa_full_layers * causal_rows * int(dsa_index_head_dim) * dtype_bytes
        if dsa_full_layers and dsa_index_topk is not None and dsa_index_topk > 0
        else 0
    )
    mla_cache_write = (
        cfg.num_hidden_layers * prompt_tokens * int(cache_width) * dtype_bytes
    )
    dsa_index_write = (
        dsa_full_layers * prompt_tokens * int(dsa_index_head_dim) * dtype_bytes
        if dsa_full_layers
        else 0
    )
    return PrefillCacheIOPlan(
        dtype_bytes=dtype_bytes,
        mla_cache_width=int(cache_width),
        dsa_index_head_dim=(
            int(dsa_index_head_dim) if dsa_index_head_dim is not None else None
        ),
        dsa_index_topk=int(dsa_index_topk) if dsa_index_topk is not None else None,
        indexed_attention_layers=indexed_layers,
        full_attention_layers=full_attention_layers,
        dsa_full_indexer_layers=dsa_full_layers,
        causal_rows_per_layer=causal_rows,
        indexed_rows_per_layer=indexed_rows,
        mla_cache_read_bytes=mla_cache_read,
        dsa_index_cache_read_bytes=dsa_index_read,
        total_cache_read_bytes=mla_cache_read + dsa_index_read,
        mla_cache_write_bytes=mla_cache_write,
        dsa_index_cache_write_bytes=dsa_index_write,
        total_cache_write_bytes=mla_cache_write + dsa_index_write,
    )


def build_prefill_plan(
    model: str | Path | ModelConfig,
    *,
    prompt_tokens: int,
    dtype_bits: int = 16,
    expert_bits: int = 4,
    group_size: int = 64,
    mpp_min_tokens: int = DEFAULT_MPP_MIN_TOKENS,
    backend_capability: PrefillBackendCapability | None = None,
    compile_mpp_probe: bool = False,
    run_mpp_probe: bool = False,
    run_mpsgraph_probe: bool = False,
    probe_timeout_seconds: float = DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
    simdgroup_tile_m: int = 32,
    simdgroup_tile_n: int = 32,
    simdgroups_m: int = 2,
    simdgroups_n: int = 2,
    k_tile: int = 128,
    max_prefill_activation_bytes: int | None = None,
    max_runner_scratch_bytes: int | None = None,
    expert_stage_align_bytes: int = 4096,
    prefill_static_capacity_per_expert: object | None = "auto",
    ssd_read_gib_per_second: float | None = None,
    prefill_linear_backend: str = "auto",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    prefill_min_accelerated_flop_fraction: float = 0.0,
    require_prefill_acceleration: bool = False,
    require_public_glm_5_2_shape: bool = False,
) -> PrefillPlan:
    prompt_tokens = _positive_integer_value("prompt_tokens", prompt_tokens)
    dtype_bits = _integer_value("dtype_bits", dtype_bits)
    expert_bits = _integer_value("expert_bits", expert_bits)
    group_size = _positive_integer_value("group_size", group_size)
    mpp_min_tokens = _positive_integer_value("mpp_min_tokens", mpp_min_tokens)
    if type(compile_mpp_probe) is not bool:
        raise PrefillPlanError("compile_mpp_probe must be a boolean")
    if type(run_mpsgraph_probe) is not bool:
        raise PrefillPlanError("run_mpsgraph_probe must be a boolean")
    if isinstance(probe_timeout_seconds, bool):
        raise PrefillPlanError("probe_timeout_seconds must be positive")
    try:
        probe_timeout_seconds = float(probe_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise PrefillPlanError("probe_timeout_seconds must be positive") from exc
    if not math.isfinite(probe_timeout_seconds) or probe_timeout_seconds <= 0:
        raise PrefillPlanError("probe_timeout_seconds must be positive")
    simdgroup_tile_m = _positive_integer_value("simdgroup_tile_m", simdgroup_tile_m)
    simdgroup_tile_n = _positive_integer_value("simdgroup_tile_n", simdgroup_tile_n)
    simdgroups_m = _positive_integer_value("simdgroups_m", simdgroups_m)
    simdgroups_n = _positive_integer_value("simdgroups_n", simdgroups_n)
    k_tile = _positive_integer_value("k_tile", k_tile)
    if max_prefill_activation_bytes is not None:
        max_prefill_activation_bytes = _positive_integer_value(
            "max_prefill_activation_bytes",
            max_prefill_activation_bytes,
        )
    if max_runner_scratch_bytes is not None:
        max_runner_scratch_bytes = _positive_integer_value(
            "max_runner_scratch_bytes",
            max_runner_scratch_bytes,
        )
    expert_stage_align_bytes = _positive_integer_value(
        "expert_stage_align_bytes",
        expert_stage_align_bytes,
    )
    static_capacity_per_expert = _normalize_static_capacity_per_expert(
        prefill_static_capacity_per_expert
    )
    ssd_read_gib_per_second = _optional_positive_float_value(
        "ssd_read_gib_per_second",
        ssd_read_gib_per_second,
    )
    if prefill_linear_backend not in PREFILL_LINEAR_BACKENDS:
        raise PrefillPlanError(
            "prefill_linear_backend must be custom-metal, mpp-f32, "
            "mpsgraph-f32, mps-matrix-f32, or auto"
        )
    prefill_mpsgraph_min_batch_tokens = _positive_integer_value(
        "prefill_mpsgraph_min_batch_tokens",
        prefill_mpsgraph_min_batch_tokens,
    )
    prefill_mpsgraph_min_matrix_dim = _positive_integer_value(
        "prefill_mpsgraph_min_matrix_dim",
        prefill_mpsgraph_min_matrix_dim,
    )
    prefill_min_accelerated_flop_fraction = _fraction_value(
        "prefill_min_accelerated_flop_fraction",
        prefill_min_accelerated_flop_fraction,
    )
    if type(require_prefill_acceleration) is not bool:
        raise PrefillPlanError("require_prefill_acceleration must be a boolean")
    if type(require_public_glm_5_2_shape) is not bool:
        raise PrefillPlanError("require_public_glm_5_2_shape must be a boolean")
    if dtype_bits not in {16, 32}:
        raise PrefillPlanError("dtype_bits must be 16 or 32")
    if expert_bits not in {2, 3, 4, 8}:
        raise PrefillPlanError("expert_bits must be 2, 3, 4, or 8")

    def gemm_kwargs() -> dict[str, int]:
        return {
            "simdgroup_tile_m": simdgroup_tile_m,
            "simdgroup_tile_n": simdgroup_tile_n,
            "simdgroups_m": simdgroups_m,
            "simdgroups_n": simdgroups_n,
            "k_tile": k_tile,
        }

    cfg = model if isinstance(model, ModelConfig) else load_config(model)
    model_path = Path("<config>") if isinstance(model, ModelConfig) else Path(model)
    public_glm_5_2_shape = _public_glm_5_2_shape_report(cfg)
    if (
        require_public_glm_5_2_shape
        and public_glm_5_2_shape.get("matches") is not True
    ):
        detail = _public_glm_5_2_shape_failure_detail(public_glm_5_2_shape)
        suffix = f" ({detail})" if detail else ""
        raise PrefillPlanError(
            f"config does not match the public GLM-5.2 shape{suffix}"
        )
    if require_public_glm_5_2_shape and expert_bits != 4:
        raise PrefillPlanError(
            "public GLM-5.2 prefill planning requires expert_bits=4"
        )
    if (
        cfg.max_position_embeddings is not None
        and prompt_tokens > int(cfg.max_position_embeddings)
    ):
        raise PrefillPlanError(
            "prompt_tokens "
            f"{prompt_tokens} exceeds model max_position_embeddings "
            f"{cfg.max_position_embeddings}"
        )
    dtype_bytes = dtype_bits // 8
    cache_io_plan = build_prefill_cache_io_plan(
        cfg,
        prompt_tokens=prompt_tokens,
        dtype_bytes=dtype_bytes,
    )
    moe_layers = cfg.num_moe_layers
    dense_layers = cfg.num_hidden_layers - moe_layers
    ops: list[PrefillOp] = []

    def add(op: PrefillOp | None) -> None:
        if op is not None:
            ops.append(op)

    q_lora = cfg.q_lora_rank
    q_out = cfg.attention_q_projection_output_dim
    kv_a_out = cfg.attention_kv_a_output_dim
    kv_lora = cfg.kv_lora_rank
    kv_b_out = cfg.attention_kv_b_output_dim
    value_out = cfg.attention_value_output_dim
    add(
        _gemm_op(
            "attention.q_a_proj",
            layers=cfg.num_hidden_layers,
            tokens=prompt_tokens,
            k_in=cfg.hidden_size,
            n_out=q_lora,
            dtype_bytes=dtype_bytes,
            mpp_min_tokens=mpp_min_tokens,
            **gemm_kwargs(),
        )
    )
    add(
        _gemm_op(
            "attention.q_b_proj",
            layers=cfg.num_hidden_layers,
            tokens=prompt_tokens,
            k_in=q_lora,
            n_out=q_out,
            dtype_bytes=dtype_bytes,
            mpp_min_tokens=mpp_min_tokens,
            **gemm_kwargs(),
        )
    )
    add(
        _gemm_op(
            "attention.kv_a_proj",
            layers=cfg.num_hidden_layers,
            tokens=prompt_tokens,
            k_in=cfg.hidden_size,
            n_out=kv_a_out,
            dtype_bytes=dtype_bytes,
            mpp_min_tokens=mpp_min_tokens,
            **gemm_kwargs(),
        )
    )
    add(
        _gemm_op(
            "attention.kv_b_proj",
            layers=cfg.num_hidden_layers,
            tokens=prompt_tokens,
            k_in=kv_lora,
            n_out=kv_b_out,
            dtype_bytes=dtype_bytes,
            mpp_min_tokens=mpp_min_tokens,
            **gemm_kwargs(),
        )
    )
    add(
        _gemm_op(
            "attention.o_proj",
            layers=cfg.num_hidden_layers,
            tokens=prompt_tokens,
            k_in=value_out,
            n_out=cfg.hidden_size,
            dtype_bytes=dtype_bytes,
            mpp_min_tokens=mpp_min_tokens,
            **gemm_kwargs(),
        )
    )

    if cfg.n_routed_experts:
        add(
            _gemm_op(
                "moe.router",
                layers=moe_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=cfg.routed_experts,
                dtype_bytes=4,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
    if cfg.n_shared_experts and cfg.moe_intermediate_size:
        shared_width = int(cfg.n_shared_experts) * cfg.moe_hidden_size
        add(
            _gemm_op(
                "moe.shared_gate_proj",
                layers=moe_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=shared_width,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
        add(
            _gemm_op(
                "moe.shared_up_proj",
                layers=moe_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=shared_width,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
        add(
            _gemm_op(
                "moe.shared_down_proj",
                layers=moe_layers,
                tokens=prompt_tokens,
                k_in=shared_width,
                n_out=cfg.hidden_size,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )

    if dense_layers > 0 and cfg.intermediate_size:
        add(
            _gemm_op(
                "dense_mlp.gate_proj",
                layers=dense_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=cfg.intermediate_size,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
        add(
            _gemm_op(
                "dense_mlp.up_proj",
                layers=dense_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=cfg.intermediate_size,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
        add(
            _gemm_op(
                "dense_mlp.down_proj",
                layers=dense_layers,
                tokens=prompt_tokens,
                k_in=cfg.intermediate_size,
                n_out=cfg.hidden_size,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )

    if cfg.index_head_dim and cfg.indexer_types:
        full_index_layers = sum(1 for item in cfg.indexer_types if item == "full")
        add(
            _gemm_op(
                "dsa.index_wk",
                layers=full_index_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=cfg.index_head_dim,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
        add(
            _gemm_op(
                "dsa.index_wq_b",
                layers=full_index_layers,
                tokens=prompt_tokens,
                k_in=cfg.q_lora_rank,
                n_out=cfg.dsa_full_indexer_q_output_dim,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )
        add(
            _gemm_op(
                "dsa.index_weights_proj",
                layers=full_index_layers,
                tokens=prompt_tokens,
                k_in=cfg.hidden_size,
                n_out=cfg.index_n_heads,
                dtype_bytes=dtype_bytes,
                mpp_min_tokens=mpp_min_tokens,
                **gemm_kwargs(),
            )
        )

    resident_weight_bytes = sum(op.total_weight_bytes for op in ops)
    routed_expert_slot_bytes = 0
    routed_assignments = 0
    routed_unique = 0
    routed_read_bytes = 0
    routed_chunked_read_bytes = 0
    routed_read_chunks = 0
    routed_flops = 0
    routed_backend = "none"
    if cfg.n_routed_experts and cfg.num_experts_per_tok and moe_layers > 0:
        expert_layout = ExpertLayout(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_hidden_size,
            weight_bits=expert_bits,
            group_size=group_size,
        )
        routed_expert_slot_bytes = expert_layout.total_bytes
        routed_assignments = prompt_tokens * cfg.experts_per_token
        routed_unique = min(cfg.routed_experts, routed_assignments)
        routed_read_bytes = moe_layers * routed_unique * routed_expert_slot_bytes
        routed_flops = (
            moe_layers
            * prompt_tokens
            * cfg.experts_per_token
            * 6
            * cfg.hidden_size
            * cfg.moe_hidden_size
        )
        routed_backend = (
            "ssd_full_layer_expert_sweep"
            if routed_unique == cfg.routed_experts
            else "ssd_grouped_routed_expert_streaming"
        )

    total_flops = sum(op.total_flops for op in ops) + routed_flops
    peak_activation = max((op.activation_bytes_per_layer for op in ops), default=0)
    mpp_candidate_ops = sum(1 for op in ops if op.backend_hint == "mpp_tensor_ops_candidate")
    effective_metal4_candidate_ops = (
        mpp_candidate_ops
        if backend_capability and backend_capability.metal4_ml_runtime_available
        else 0
    )
    effective_mpp_candidate_ops = (
        mpp_candidate_ops if backend_capability and backend_capability.mpp_runtime_available else 0
    )
    prefill_backend_candidates = _prefill_backend_candidates(ops, backend_capability)
    chunk_plan = _chunk_plan(
        prompt_tokens=prompt_tokens,
        max_activation_bytes=max_prefill_activation_bytes,
        ops=ops,
        threadgroup_tile_m=simdgroup_tile_m * simdgroups_m,
    )
    calibration_batch_tokens = (
        chunk_plan.recommended_chunk_tokens if chunk_plan is not None else prompt_tokens
    )
    prefill_linear_calibration_shapes = _prefill_linear_calibration_shapes(
        prefill_backend_candidates,
        batch_tokens=calibration_batch_tokens,
    )
    prefill_linear_calibration_candidate_coverage = (
        _prefill_linear_calibration_candidate_coverage(
            prefill_backend_candidates,
            prefill_linear_calibration_shapes,
            source="prefill_plan",
        )
    )
    suggested_prefill_linear_calibration = _suggest_prefill_linear_calibration_flags(
        prefill_linear_calibration_shapes,
        source="prefill_plan",
    )
    routed_capacity_plan = None
    staged_moe_runner_scratch_plan = None
    routed_read_cost_plan = None
    routed_stage_temp_plan = None
    suggested_guard_flags = None
    suggested_stage_temp_guard_flags = None
    suggested_prefill_guard_flags = None
    suggested_prefill_acceleration = (
        suggested_prefill_acceleration_flags(
            backend_capability,
            source="prefill_plan",
        )
        if backend_capability is not None
        else None
    )
    suggested_prefill_runtime_policy = _suggest_prefill_runtime_policy_flags(
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        prefill_min_accelerated_flop_fraction=(
            prefill_min_accelerated_flop_fraction
        ),
        require_prefill_acceleration=require_prefill_acceleration,
        source="prefill_plan",
    )
    suggested_prefill_backend_probe = _suggest_prefill_backend_probe_flags(
        compile_mpp_probe=compile_mpp_probe,
        run_mpp_probe=run_mpp_probe,
        run_mpsgraph_probe=run_mpsgraph_probe,
        probe_timeout_seconds=probe_timeout_seconds,
    )
    suggested_public_glm_5_2_shape_guard = _suggest_public_glm_5_2_shape_guard_flags(
        public_glm_5_2_shape,
        expert_bits=expert_bits,
        source="prefill_plan",
    )
    if routed_assignments and routed_unique:
        capacity_tokens = (
            chunk_plan.recommended_chunk_tokens if chunk_plan is not None else prompt_tokens
        )
        routed_read_chunks = chunk_plan.chunks if chunk_plan is not None else 1
        capacity_assignments = capacity_tokens * cfg.experts_per_token
        capacity_unique = min(cfg.routed_experts, capacity_assignments)
        routed_chunked_read_bytes = _routed_expert_chunked_read_bytes(
            prompt_tokens=prompt_tokens,
            chunk_tokens=capacity_tokens,
            experts_per_token=cfg.experts_per_token,
            routed_experts=cfg.routed_experts,
            moe_layers=moe_layers,
            expert_slot_bytes=routed_expert_slot_bytes,
        )
        routed_capacity_plan = _routed_expert_capacity_plan(
            prompt_tokens=capacity_tokens,
            chunks_per_prompt=routed_read_chunks,
            routed_experts=cfg.routed_experts,
            routed_assignments=capacity_assignments,
            routed_unique=capacity_unique,
            hidden_size=cfg.hidden_size,
            moe_hidden_size=cfg.moe_hidden_size,
        )
        staged_moe_runner_scratch_plan = _staged_moe_runner_scratch_plan(
            capacity_tokens=capacity_tokens,
            assignments_per_capacity_chunk=capacity_assignments,
            hidden_size=cfg.hidden_size,
            moe_hidden_size=cfg.moe_hidden_size,
            expert_slot_bytes=routed_expert_slot_bytes,
            max_runner_scratch_bytes=max_runner_scratch_bytes,
        )
        routed_stage_temp_plan = _routed_stage_temp_plan(
            prompt_tokens=prompt_tokens,
            chunk_tokens=capacity_tokens,
            experts_per_token=cfg.experts_per_token,
            routed_experts=cfg.routed_experts,
            moe_layers=moe_layers,
            expert_slot_bytes=routed_expert_slot_bytes,
            stage_align_bytes=expert_stage_align_bytes,
            static_capacity_per_expert=static_capacity_per_expert,
            backend_hint=routed_backend,
        )
        routed_read_cost_plan = _routed_expert_read_cost_plan(
            baseline_read_bytes=routed_read_bytes,
            planned_read_bytes=routed_chunked_read_bytes,
            chunks_per_prompt=routed_read_chunks,
            ssd_read_gib_per_second=ssd_read_gib_per_second,
            backend_hint=routed_backend,
        )
        if routed_read_cost_plan is not None:
            suggested_guard_flags = suggest_routed_read_guard_flags(
                prompt_chunk_tokens=capacity_tokens,
                planned_read_bytes=routed_read_cost_plan.planned_read_bytes,
                read_amplification=routed_read_cost_plan.read_amplification,
                ssd_read_gib_per_second=ssd_read_gib_per_second,
                planned_read_seconds=routed_read_cost_plan.planned_read_seconds,
                source="prefill_plan",
            )
        if routed_stage_temp_plan is not None:
            suggested_stage_temp_guard_flags = suggest_routed_stage_temp_guard_flags(
                prompt_chunk_tokens=capacity_tokens,
                max_stage_bytes=routed_stage_temp_plan.max_stage_bytes,
                max_compact_stage_bytes=(
                    routed_stage_temp_plan.max_compact_stage_bytes
                ),
                max_stage_raw_ranges=(
                    routed_stage_temp_plan.max_stage_raw_ranges
                ),
                max_stage_coalesced_ranges=(
                    routed_stage_temp_plan.max_stage_coalesced_ranges
                ),
                max_stage_plus_compact_bytes=(
                    routed_stage_temp_plan.max_stage_plus_compact_bytes
                ),
                total_stage_plus_compact_bytes=(
                    routed_stage_temp_plan.total_stage_plus_compact_bytes
                ),
                max_static_capacity_binary_bytes=(
                    routed_stage_temp_plan.max_static_capacity_binary_bytes
                ),
                total_static_capacity_binary_bytes=(
                    routed_stage_temp_plan.total_static_capacity_binary_bytes
                ),
                max_stage_plus_compact_plus_static_bytes=(
                    routed_stage_temp_plan.max_stage_plus_compact_plus_static_bytes
                ),
                total_stage_plus_compact_plus_static_bytes=(
                    routed_stage_temp_plan.total_stage_plus_compact_plus_static_bytes
                ),
                static_capacity_per_expert=(
                    routed_stage_temp_plan.static_capacity_per_expert
                ),
                source="prefill_plan",
            )
        suggested_prefill_guard_flags = combine_prefill_guard_flags(
            routed_read_flags=suggested_guard_flags,
            stage_temp_flags=suggested_stage_temp_guard_flags,
            source="prefill_plan",
        )
    suggested_launch_profile = _combine_prefill_launch_profile(
        prefill_guard_flags=suggested_prefill_guard_flags,
        prefill_runtime_policy_flags=suggested_prefill_runtime_policy,
        prefill_acceleration_flags=suggested_prefill_acceleration,
        prefill_backend_probe_flags=suggested_prefill_backend_probe,
        public_glm_5_2_shape_guard_flags=suggested_public_glm_5_2_shape_guard,
        source="prefill_plan",
    )
    total_weight_bytes = resident_weight_bytes + (
        routed_chunked_read_bytes if routed_chunked_read_bytes else routed_read_bytes
    )
    return PrefillPlan(
        model_path=model_path,
        model_type=cfg.model_type,
        public_glm_5_2_shape=public_glm_5_2_shape,
        prompt_tokens=prompt_tokens,
        dtype_bits=dtype_bits,
        expert_bits=expert_bits,
        group_size=group_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_hidden_layers,
        moe_layers=moe_layers,
        dense_layers=dense_layers,
        total_flops=total_flops,
        total_weight_bytes=total_weight_bytes,
        resident_gemm_weight_bytes=resident_weight_bytes,
        routed_expert_slot_bytes=routed_expert_slot_bytes,
        routed_expert_assignments_per_moe_layer=routed_assignments,
        routed_expert_unique_per_moe_layer=routed_unique,
        routed_expert_read_bytes=routed_read_bytes,
        routed_expert_chunked_read_bytes=routed_chunked_read_bytes,
        routed_expert_read_chunks_per_prompt=routed_read_chunks,
        routed_expert_flops=routed_flops,
        routed_expert_backend_hint=routed_backend,
        peak_activation_bytes=peak_activation,
        mpp_candidate_ops=mpp_candidate_ops,
        ops=tuple(ops),
        prefill_backend_candidates=prefill_backend_candidates,
        prefill_linear_calibration_shapes=prefill_linear_calibration_shapes,
        prefill_linear_calibration_candidate_coverage=(
            prefill_linear_calibration_candidate_coverage
        ),
        backend_capability=backend_capability,
        effective_metal4_candidate_ops=(
            effective_metal4_candidate_ops if backend_capability is not None else None
        ),
        effective_mpp_candidate_ops=(
            effective_mpp_candidate_ops if backend_capability is not None else None
        ),
        cache_io_plan=cache_io_plan,
        chunk_plan=chunk_plan,
        routed_expert_read_cost_plan=routed_read_cost_plan,
        routed_stage_temp_plan=routed_stage_temp_plan,
        suggested_guard_flags=suggested_guard_flags,
        suggested_stage_temp_guard_flags=suggested_stage_temp_guard_flags,
        suggested_prefill_guard_flags=suggested_prefill_guard_flags,
        suggested_prefill_runtime_policy_flags=suggested_prefill_runtime_policy,
        suggested_prefill_linear_calibration_flags=(
            suggested_prefill_linear_calibration
        ),
        suggested_public_glm_5_2_shape_guard_flags=(
            suggested_public_glm_5_2_shape_guard
        ),
        suggested_launch_profile=suggested_launch_profile,
        routed_expert_capacity_plan=routed_capacity_plan,
        staged_moe_runner_scratch_plan=staged_moe_runner_scratch_plan,
    )
