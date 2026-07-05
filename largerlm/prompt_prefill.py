from __future__ import annotations

import json
import math
import operator
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

from .config import first_shared_indexer_without_previous_full
from .decode_cache import DecodeCacheError, load_decode_cache_layout
from .decode_driver import DecodeDriverError, layers_from_expert_layout
from .embedding import EmbeddingError, _load_embedding_metadata, embed_tokens_batch
from .generation_guard import (
    GenerationGuardError,
    LiveMemoryBudget,
    _nonnegative_limit,
    _positive_limit,
    check_live_memory_budget,
    estimate_prompt_prefill_live_memory,
)
from .prefill_execute import (
    AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    AUTO_MPSGRAPH_MIN_DIM,
    PREFILL_LINEAR_ACCELERATED_BACKENDS,
    PREFILL_LINEAR_F32_CONVERSION_BACKENDS,
    PREFILL_LINEAR_MPSGRAPH_DTYPES,
    AttentionProjectionsServerSession,
    AttentionOutputBatchServerSession,
    MLAAttentionBatchServerSession,
    PrefillAttentionBlockBatchResult,
    PrefillDenseMLPBlockBatchResult,
    PrefillExecuteError,
    PrefillStagedRoutedMLPBlockBatchResult,
    ResidentBatchLinearServerSession,
    ResidentBatchRMSNormServerSession,
    ResidentSharedExpertBatchServerSession,
    RopeSplitBatchServerSession,
    _resident_linear_matrix_scratch,
    is_prompt_prefill_resident_linear_backend_tensor,
    is_prompt_prefill_router_gate_tensor,
    run_prefill_attention_block_batch,
    run_prefill_dense_mlp_block_batch,
    run_prefill_staged_routed_mlp_block_batch,
)
from .prefill_plan import DEFAULT_MPP_MIN_TOKENS, MPP_TENSOR_OPS_MIN_MATRIX_DIM
from .resident_affine import (
    ResidentAffineLayoutError,
    is_affine_int4_weight_dtype,
    is_mxfp4_scale_dtype,
    resident_affine_int4_layout_info,
    resident_layout_tensors_by_name,
    resident_mxfp4_layout_info,
)
from .safety import disk_budget
from .staged_moe import (
    MoEOutputAccumulator,
    MoETokenBlock,
    StagedMoEError,
    StagedRoutedMoEBatchPlanServerSession,
    _normalize_moe_output_accumulator,
    _normalize_moe_token_block,
)

StaticCapacityPerExpert = Union[int, str, None]
_EXPERT_STAGE_IO_HOTSPOT_LIMIT = 5


class PromptPrefillError(RuntimeError):
    """Raised when prompt prefill cannot be executed within configured limits."""


class _PromptPrefillLiveMemoryError(PromptPrefillError):
    """Raised when live memory headroom falls during prompt prefill."""


@dataclass(frozen=True)
class PromptPrefillLayerRecord:
    layer: int
    kind: str
    input_path: Path
    attention_output_path: Path
    output_path: Path
    output_dir: Path
    dsa_indexer_mode: str
    dsa_rope_interleave: bool
    dsa_indices_u32_path: Path | None
    estimated_peak_bytes: int
    attention: PrefillAttentionBlockBatchResult
    dense_mlp: PrefillDenseMLPBlockBatchResult | None
    staged_mlp: PrefillStagedRoutedMLPBlockBatchResult | None


@dataclass(frozen=True)
class PromptPrefillChunkRecord:
    chunk_index: int
    start_position: int
    batch_tokens: int
    token_ids: tuple[int, ...]
    embedding_output_path: Path
    output_path: Path
    last_hidden_path: Path | None
    embedding_read_bytes: int
    embedding_output_bytes: int
    estimated_peak_bytes: int
    layers: tuple[PromptPrefillLayerRecord, ...]


@dataclass(frozen=True)
class PromptPrefillStageIOHotspotRecord:
    chunk_index: int
    layer: int
    tile_index: int
    batch_tokens: int
    selected_experts: tuple[int, ...]
    selected_expert_count: int
    total_assignments: int
    raw_range_count: int
    coalesced_range_count: int
    planned_read_bytes: int
    staged_bytes: int
    unique_requested_bytes: int
    waste_bytes: int
    copy_chunk_bytes: int
    copy_elapsed_seconds: float | None
    copy_throughput_gib_per_second: float | None
    copy_read_calls: int
    copy_write_calls: int
    copy_average_read_bytes: float | None
    stage_budget_utilization: float
    unique_read_amplification: float


@dataclass(frozen=True)
class PromptPrefillResult:
    runner_path: Path
    expert_layout_path: Path
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    output_last_hidden_path: Path
    output_final_chunk_path: Path | None
    work_dir: Path
    kept_work_dir: bool
    elapsed_seconds: float
    prompt_token_ids: tuple[int, ...]
    start_position: int
    chunk_tokens: int
    chunk_count: int
    layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    hidden_dim: int
    total_embedding_read_bytes: int
    total_embedding_output_bytes: int
    total_staged_bytes: int
    total_compact_stage_bytes: int
    total_compact_stage_materialized_bytes: int
    max_staged_bytes: int
    max_compact_stage_bytes: int
    max_compact_stage_materialized_bytes: int
    total_stage_plus_compact_bytes: int
    total_stage_plus_compact_materialized_bytes: int
    max_stage_plus_compact_bytes: int
    max_stage_plus_compact_materialized_bytes: int
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
    total_routed_expert_assignments: int
    total_routed_unique_expert_slots: int
    max_routed_unique_experts_per_call: int
    max_routed_tokens_per_expert: int
    moe_token_block: MoETokenBlock
    moe_token_block_mode_counts: dict[str, int]
    max_effective_moe_token_block: int
    max_moe_max_expert_tokens: int
    max_moe_batch_buffer_bytes: int
    max_moe_estimated_peak_bytes: int
    persistent_moe_plan_server: bool
    persistent_resident_linear_server: bool
    persistent_attention_projection_server: bool
    persistent_attention_output_server: bool
    persistent_shared_expert_server: bool
    persistent_rope_split_server: bool
    persistent_mla_attention_server: bool
    persistent_rmsnorm_server: bool
    moe_output_accumulator: str
    moe_plan_server_plan_count: int
    moe_plan_server_job_count: int
    routed_moe_runner_command_count: int
    static_capacity_per_expert: StaticCapacityPerExpert
    max_static_capacity_per_expert: int
    total_static_capacity_used_slots: int
    total_static_capacity_slots: int
    total_static_capacity_overflow_assignments: int
    total_static_capacity_binary_bytes: int
    estimated_peak_bytes: int
    live_memory_budget: LiveMemoryBudget
    linear_backend_counts: dict[str, int]
    linear_backend_flops: dict[str, int]
    total_linear_matrix_scratch_bytes: int
    max_linear_matrix_scratch_bytes: int
    total_linear_matrix_f32_bytes: int
    total_linear_matrix_raw_conversion_bytes: int
    total_linear_estimated_flops: int
    accelerated_linear_estimated_flops: int
    custom_linear_estimated_flops: int
    unsupported_linear_estimated_flops: int
    accelerated_linear_flop_fraction: float
    prefill_acceleration_coverage: dict[str, object] | None
    prefill_acceleration_frontier: dict[str, object] | None
    chunks: tuple[PromptPrefillChunkRecord, ...]
    total_expert_stage_copy_elapsed_seconds: float | None = None
    total_expert_stage_copy_throughput_gib_per_second: float | None = None
    total_expert_stage_copy_read_calls: int = 0
    total_expert_stage_copy_write_calls: int = 0
    total_expert_stage_copy_average_read_bytes: float | None = None
    total_expert_stage_copy_average_write_bytes: float | None = None
    total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib: (
        dict[str, int] | None
    ) = None
    expert_stage_io_stage_count: int = 0
    expert_stage_copy_hotspots: tuple[PromptPrefillStageIOHotspotRecord, ...] = ()
    expert_stage_range_hotspots: tuple[PromptPrefillStageIOHotspotRecord, ...] = ()
    linear_backend_elapsed_seconds: dict[str, float] | None = None
    linear_backend_estimated_tflops: dict[str, float] | None = None
    linear_backend_component_stats: dict[str, dict[str, object]] | None = None


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


def _integer_value(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise PromptPrefillError(f"{label} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise PromptPrefillError(f"{label} must be an integer") from exc


def _nonnegative_integer_value(value: object, *, label: str) -> int:
    parsed = _integer_value(value, label=label)
    if parsed < 0:
        raise PromptPrefillError(f"{label} must be non-negative")
    return parsed


def _positive_integer_value(value: object, *, label: str) -> int:
    parsed = _integer_value(value, label=label)
    if parsed <= 0:
        raise PromptPrefillError(f"{label} must be positive")
    return parsed


def _optional_positive_integer_value(value: object | None, *, label: str) -> int | None:
    if value is None:
        return None
    return _positive_integer_value(value, label=label)


def _normalize_layer_set(layers: Iterable[int] | None) -> set[int]:
    return {_nonnegative_integer_value(layer, label="layers") for layer in layers or ()}


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


def _router_gate_totals_from_component_stats(
    component_stats: dict[str, dict[str, object]] | None,
) -> dict[str, int]:
    if not component_stats:
        return {
            "router_gate_matrix_count": 0,
            "router_gate_estimated_flops": 0,
            "router_gate_accelerated_matrix_count": 0,
            "router_gate_accelerated_estimated_flops": 0,
        }
    router = component_stats.get("moe.router_gate_proj")
    if not isinstance(router, dict):
        return {
            "router_gate_matrix_count": 0,
            "router_gate_estimated_flops": 0,
            "router_gate_accelerated_matrix_count": 0,
            "router_gate_accelerated_estimated_flops": 0,
        }
    raw_counts = router.get("linear_backend_counts")
    raw_flops = router.get("linear_backend_flops")
    counts = _sorted_positive_backend_ints(
        raw_counts if isinstance(raw_counts, dict) else None
    )
    flops = _sorted_positive_backend_ints(
        raw_flops if isinstance(raw_flops, dict) else None
    )
    return {
        "router_gate_matrix_count": sum(counts.values()),
        "router_gate_estimated_flops": sum(flops.values()),
        "router_gate_accelerated_matrix_count": sum(
            counts.get(backend, 0) for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        ),
        "router_gate_accelerated_estimated_flops": sum(
            flops.get(backend, 0) for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        ),
    }


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
    streamed_routed_expert_layer_count: int = 0,
    streamed_routed_expert_matrix_count: int = 0,
    streamed_routed_expert_assignments: int = 0,
    streamed_routed_expert_estimated_flops: int = 0,
    streamed_routed_expert_mpp_candidate_matrix_count: int = 0,
    streamed_routed_expert_mpp_candidate_estimated_flops: int = 0,
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
    non_router_matrix_count = max(0, matrix_count - router_gate_count)
    non_router_estimated_flops = max(0, total_estimated_flops - router_gate_flops)
    non_router_unaccelerated_count = max(
        0,
        non_router_matrix_count - non_router_accelerated_count,
    )
    non_router_unaccelerated_flops = max(
        0,
        non_router_estimated_flops - non_router_accelerated_flops,
    )
    non_router_unaccelerated_fraction = (
        non_router_unaccelerated_flops / total_estimated_flops
        if total_estimated_flops > 0
        else 0.0
    )
    streamed_layer_count = max(0, int(streamed_routed_expert_layer_count))
    streamed_matrix_count = max(0, int(streamed_routed_expert_matrix_count))
    streamed_assignments = max(0, int(streamed_routed_expert_assignments))
    streamed_flops = max(0, int(streamed_routed_expert_estimated_flops))
    streamed_mpp_count = max(
        0,
        int(streamed_routed_expert_mpp_candidate_matrix_count),
    )
    streamed_mpp_flops = max(
        0,
        int(streamed_routed_expert_mpp_candidate_estimated_flops),
    )
    streamed_unaccelerated_count = min(
        streamed_matrix_count,
        non_router_unaccelerated_count,
    )
    streamed_unaccelerated_flops = min(streamed_flops, non_router_unaccelerated_flops)
    non_streamed_unaccelerated_count = max(
        0,
        non_router_unaccelerated_count - streamed_unaccelerated_count,
    )
    non_streamed_unaccelerated_flops = max(
        0,
        non_router_unaccelerated_flops - streamed_unaccelerated_flops,
    )
    unaccelerated_backend_counts = _sorted_positive_backend_ints(
        {
            "custom-metal": custom_count,
            "unsupported-mpsgraph": unsupported_count,
            "other": other_count,
        }
    )
    unaccelerated_backend_flops = _sorted_positive_backend_ints(
        {
            "custom-metal": custom_estimated_flops,
            "unsupported-mpsgraph": unsupported_estimated_flops,
            "other": other_estimated_flops,
        }
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
        "router_gate_estimated_flops": router_gate_flops,
        "router_gate_accelerated_matrix_count": router_gate_accelerated_count,
        "router_gate_accelerated_estimated_flops": router_gate_accelerated_flops,
        "non_router_matrix_count": non_router_matrix_count,
        "non_router_estimated_flops": non_router_estimated_flops,
        "non_router_accelerated_matrix_count": non_router_accelerated_count,
        "non_router_accelerated_estimated_flops": non_router_accelerated_flops,
        "non_router_unaccelerated_matrix_count": non_router_unaccelerated_count,
        "non_router_unaccelerated_estimated_flops": non_router_unaccelerated_flops,
        "non_router_unaccelerated_flop_fraction": (
            non_router_unaccelerated_fraction
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
        "unaccelerated_backend_matrix_counts": unaccelerated_backend_counts,
        "unaccelerated_backend_estimated_flops": unaccelerated_backend_flops,
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
        "streamed_routed_expert_layer_count": streamed_layer_count,
        "streamed_routed_expert_matrix_count": streamed_matrix_count,
        "streamed_routed_expert_assignments": streamed_assignments,
        "streamed_routed_expert_estimated_flops": streamed_flops,
        "streamed_routed_expert_mpp_candidate_matrix_count": streamed_mpp_count,
        "streamed_routed_expert_mpp_candidate_estimated_flops": (
            streamed_mpp_flops
        ),
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


def _prefill_linear_summary_from_layout(
    *,
    resident_layout_path: Path,
    prefill_linear_backend: str,
    prompt_chunk_tokens: int,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
    streamed_routed_expert_hidden_dim: int = 0,
    streamed_routed_expert_top_k: int = 0,
    streamed_routed_expert_moe_hidden_dims: Iterable[int] = (),
) -> dict[str, object]:
    try:
        payload = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "analyzed": False,
            "reason": (
                "failed to inspect resident layout for prompt prefill "
                f"acceleration frontier: {exc}"
            ),
        }
    tensors = payload.get("tensors")
    if not isinstance(tensors, list):
        return {
            "analyzed": False,
            "reason": (
                "resident layout missing tensors array for prompt prefill "
                "acceleration frontier"
            ),
        }
    try:
        tensors_by_name = resident_layout_tensors_by_name(payload)
    except ResidentAffineLayoutError as exc:
        return {
            "analyzed": False,
            "reason": str(exc),
        }

    counts: dict[str, int] = {}
    flops_by_backend: dict[str, int] = {}
    mpp_candidate_backend_counts: dict[str, int] = {}
    mpp_candidate_backend_flops: dict[str, int] = {}
    mpp_candidate_count = 0
    mpp_candidate_flops = 0
    max_scratch = 0
    total_scratch = 0
    total_raw_conversion = 0
    streamed_totals: dict[str, int] = {}
    router_gate_matrix_count = 0
    router_gate_estimated_flops = 0
    router_gate_accelerated_matrix_count = 0
    router_gate_accelerated_estimated_flops = 0
    affine_companion_names: set[str] = set()
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
                    payload,
                    companion_weight,
                    tensors_by_name=tensors_by_name,
                )
            except ResidentAffineLayoutError as exc:
                return {
                    "analyzed": False,
                    "reason": str(exc),
                }
            if affine is not None:
                affine_companion_names.add(name)
                continue

        try:
            mxfp4 = _resident_mxfp4_2d_layout_info(
                payload,
                tensor,
                tensors_by_name=tensors_by_name,
            )
        except ResidentAffineLayoutError as exc:
            return {
                "analyzed": False,
                "reason": str(exc),
            }
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
                    payload,
                    tensor,
                    tensors_by_name=tensors_by_name,
                )
            except ResidentAffineLayoutError as exc:
                return {
                    "analyzed": False,
                    "reason": str(exc),
                }
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
                    return {
                        "analyzed": False,
                        "reason": (
                            f"resident tensor {name} shape must use integer rows and cols"
                        ),
                    }
                rows, cols = int(shape[0]), int(shape[1])
                if rows <= 0 or cols <= 0:
                    continue
                raw_size = tensor.get("size")
                if type(raw_size) is not int:
                    return {
                        "analyzed": False,
                        "reason": f"resident tensor {name} size must be an integer",
                    }
                size = int(raw_size)
                dtype = str(tensor.get("dtype") or "")
        if rows <= 0 or cols <= 0:
            continue
        estimated_flops = 2 * int(prompt_chunk_tokens) * rows * cols
        if prefill_linear_backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS:
            if dtype not in PREFILL_LINEAR_MPSGRAPH_DTYPES:
                backend = "unsupported-mpsgraph"
            else:
                backend = prefill_linear_backend
        elif prefill_linear_backend == "auto":
            backend = (
                "mpsgraph-f32"
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
    _accumulate_streamed_routed_expert_accounting(
        counts=counts,
        flops_by_backend=flops_by_backend,
        mpp_candidate_totals={
            "count": mpp_candidate_count,
            "estimated_flops": mpp_candidate_flops,
        },
        mpp_candidate_backend_counts=mpp_candidate_backend_counts,
        mpp_candidate_backend_flops=mpp_candidate_backend_flops,
        streamed_totals=streamed_totals,
        accounting=_streamed_routed_expert_linear_accounting_for_layers(
            batch_tokens=prompt_chunk_tokens,
            top_k=streamed_routed_expert_top_k,
            hidden_dim=streamed_routed_expert_hidden_dim,
            moe_hidden_dims=streamed_routed_expert_moe_hidden_dims,
        ),
    )
    mpp_candidate_count = int(
        streamed_totals.get("mpp_candidate_matrix_count", 0)
    ) + mpp_candidate_count
    mpp_candidate_flops = int(
        streamed_totals.get("mpp_candidate_estimated_flops", 0)
    ) + mpp_candidate_flops
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
        streamed_routed_expert_layer_count=streamed_totals.get("layer_count", 0),
        streamed_routed_expert_matrix_count=streamed_totals.get("matrix_count", 0),
        streamed_routed_expert_assignments=streamed_totals.get("assignments", 0),
        streamed_routed_expert_estimated_flops=streamed_totals.get(
            "estimated_flops",
            0,
        ),
        streamed_routed_expert_mpp_candidate_matrix_count=(
            streamed_totals.get("mpp_candidate_matrix_count", 0)
        ),
        streamed_routed_expert_mpp_candidate_estimated_flops=(
            streamed_totals.get("mpp_candidate_estimated_flops", 0)
        ),
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
    streamed_routed_expert_hidden_dim: int = 0,
    streamed_routed_expert_top_k: int = 0,
    streamed_routed_expert_moe_hidden_dims: Iterable[int] = (),
    min_accelerated_flop_fraction: float = 0.0,
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
    failed_reason = ""
    for chunk_tokens in candidate_chunks:
        if chunk_tokens == prompt_chunk_tokens:
            coverage = actual_coverage | {
                "total_matrix_scratch_bytes": actual_total_matrix_scratch_bytes,
                "total_matrix_raw_conversion_bytes": (
                    actual_total_matrix_raw_conversion_bytes
                ),
            }
        else:
            coverage = _prefill_linear_summary_from_layout(
                resident_layout_path=resident_layout_path,
                prefill_linear_backend=prefill_linear_backend,
                prompt_chunk_tokens=chunk_tokens,
                mpsgraph_min_batch_tokens=mpsgraph_min_batch_tokens,
                mpsgraph_min_matrix_dim=mpsgraph_min_matrix_dim,
                streamed_routed_expert_hidden_dim=streamed_routed_expert_hidden_dim,
                streamed_routed_expert_top_k=streamed_routed_expert_top_k,
                streamed_routed_expert_moe_hidden_dims=(
                    streamed_routed_expert_moe_hidden_dims
                ),
            )
        if coverage.get("analyzed") is False:
            failed_reason = str(coverage.get("reason") or "not analyzed")
            candidates.append(
                {
                    "prompt_chunk_tokens": chunk_tokens,
                    "is_resolved": chunk_tokens == prompt_chunk_tokens,
                    "is_auto_mpsgraph_threshold": (
                        chunk_tokens == mpsgraph_min_batch_tokens
                    ),
                    "viable_for_request": False,
                    "exceeds_prompt_tokens": chunk_tokens > prompt_token_count,
                    "analyzed": False,
                    "reason": failed_reason,
                }
            )
            continue
        viable = chunk_tokens <= prompt_token_count
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
                "is_auto_mpsgraph_threshold": (
                    chunk_tokens == mpsgraph_min_batch_tokens
                ),
                "viable_for_request": viable,
                "exceeds_prompt_tokens": not viable,
                "analyzed": True,
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
                "streamed_routed_expert_layer_count": coverage.get(
                    "streamed_routed_expert_layer_count",
                    0,
                ),
                "streamed_routed_expert_matrix_count": coverage.get(
                    "streamed_routed_expert_matrix_count",
                    0,
                ),
                "streamed_routed_expert_assignments": coverage.get(
                    "streamed_routed_expert_assignments",
                    0,
                ),
                "streamed_routed_expert_estimated_flops": coverage.get(
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
    if minimum_accelerated is not None:
        reason = ""
        suggested = {
            "prefill_prompt_chunk_tokens": minimum_accelerated,
            "argv": (
                "--prefill-prompt-chunk-tokens",
                str(minimum_accelerated),
            ),
        }
    elif failed_reason:
        reason = failed_reason
        suggested = None
    elif prefill_linear_backend == "custom-metal":
        reason = "effective prefill backend is custom-metal"
        suggested = None
    elif (
        prefill_linear_backend == "auto"
        and mpsgraph_min_batch_tokens > prompt_token_count
    ):
        reason = "prompt token count is below the MPSGraph auto threshold"
        suggested = None
    else:
        reason = str(
            actual_coverage.get("reason")
            or "no viable prompt chunk resolves resident matrices to MPSGraph"
        )
        suggested = None
    return {
        "source": "prompt_prefill_actual",
        "analyzed": not failed_reason,
        "prompt_token_count": prompt_token_count,
        "resolved_prompt_chunk_tokens": prompt_chunk_tokens,
        "configured_backend": prefill_linear_backend,
        "min_accelerated_flop_fraction": min_accelerated_flop_fraction,
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
        "reason": reason,
    }


def prompt_prefill_acceleration_failure_reason(
    result: PromptPrefillResult | None,
    *,
    min_accelerated_flop_fraction: float = 0.0,
    allow_router_gate_only_acceleration: bool = False,
) -> str | None:
    if result is None:
        return "batch prompt prefill did not run, so no accelerated matrices were observed"
    coverage = result.prefill_acceleration_coverage
    reason = ""
    if isinstance(coverage, dict):
        any_accelerated = coverage.get("any_resident_matrix_accelerated") is True
        fraction = float(coverage.get("accelerated_flop_fraction") or 0.0)
        if (
            any_accelerated
            and coverage.get("accelerated_router_gate_only") is True
            and not allow_router_gate_only_acceleration
        ):
            return (
                "accelerated prefill coverage comes only from MoE router gates; "
                "pass --allow-router-gate-only-prefill-acceleration only for "
                "explicit routing-drift experiments"
            )
        if any_accelerated and fraction >= min_accelerated_flop_fraction:
            return None
        if any_accelerated:
            reason = (
                f"accelerated prefill FLOP fraction {fraction:.3g} is below "
                f"required {min_accelerated_flop_fraction:.3g}"
            )
        else:
            reason = str(
                coverage.get("reason")
                or "actual prompt prefill did not use an accelerated backend"
            )
    if not reason:
        reason = "actual prompt prefill acceleration coverage was not reported"
    frontier = result.prefill_acceleration_frontier
    if isinstance(frontier, dict):
        suggested = frontier.get("suggested_guard_flags")
        if isinstance(suggested, dict):
            minimum_chunk = suggested.get("prefill_prompt_chunk_tokens")
            if isinstance(minimum_chunk, int):
                reason += f"; use prefill_prompt_chunk_tokens>={minimum_chunk}"
    return reason


def _normalize_static_capacity_per_expert(
    value: StaticCapacityPerExpert,
) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PromptPrefillError(
            "static_capacity_per_expert must be an integer, 'auto', or None"
        )
    if type(value) is int:
        if value <= 0:
            raise PromptPrefillError(
                "static_capacity_per_expert must be positive, 'auto', or None"
            )
        return value
    if not isinstance(value, str):
        raise PromptPrefillError(
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
        raise PromptPrefillError(
            "static_capacity_per_expert must be positive, 'auto', or None"
        ) from exc
    if parsed <= 0:
        raise PromptPrefillError(
            "static_capacity_per_expert must be positive, 'auto', or None"
        )
    return parsed


def _static_capacity_for_batch(
    value: int | str | None,
    *,
    batch_tokens: int,
) -> int | None:
    if value == "auto":
        return batch_tokens
    return value


def _reject_unsafe_strict_static_capacity(
    *,
    static_capacity_request: int | str | None,
    max_batch_tokens: int,
    allow_static_capacity_overflow: bool,
    label: str = "static_capacity_per_expert",
) -> None:
    if (
        isinstance(static_capacity_request, int)
        and not allow_static_capacity_overflow
        and static_capacity_request < max_batch_tokens
    ):
        raise PromptPrefillError(
            f"{label} {static_capacity_request} cannot guarantee overflow-free "
            f"strict routing for a prompt chunk of {max_batch_tokens} tokens; "
            "use 'auto', raise the capacity to the chunk size, or enable overflow "
            "explicitly for analysis"
        )


def _resolve_layers(
    *,
    expert_layout_path: str | Path,
    layers: Iterable[int] | None,
    dense_layers: Iterable[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    try:
        expert_layers = set(layers_from_expert_layout(expert_layout_path))
    except DecodeDriverError as exc:
        raise PromptPrefillError(str(exc)) from exc
    dense = _normalize_layer_set(dense_layers)
    if expert_layers & dense:
        overlap = ",".join(str(layer) for layer in sorted(expert_layers & dense))
        raise PromptPrefillError(f"dense layers overlap expert layout layers: {overlap}")
    selected = (
        _normalize_layer_set(layers)
        if layers is not None
        else set(expert_layers) | set(dense)
    )
    unknown = selected - expert_layers - dense
    if unknown:
        joined = ",".join(str(layer) for layer in sorted(unknown))
        raise PromptPrefillError(f"layers not found in dense or expert layouts: {joined}")
    if not selected:
        raise PromptPrefillError("no layers selected")
    return tuple(sorted(selected)), tuple(layer for layer in sorted(selected) if layer in dense)


def _copy_last_f32_row(
    *,
    input_path: Path,
    output_path: Path,
    batch_tokens: int,
    hidden_dim: int,
) -> int:
    if batch_tokens <= 0 or hidden_dim <= 0:
        raise PromptPrefillError("last-row copy dimensions must be positive")
    row_bytes = hidden_dim * 4
    expected_bytes = batch_tokens * row_bytes
    try:
        actual_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PromptPrefillError(f"failed to stat final chunk {input_path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise PromptPrefillError(
            f"final chunk bytes {actual_bytes} do not match expected {expected_bytes}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open("rb") as source, output_path.open("wb") as dest:
            source.seek((batch_tokens - 1) * row_bytes)
            row = source.read(row_bytes)
            if len(row) != row_bytes:
                raise PromptPrefillError(f"failed to read final f32 row from {input_path}")
            dest.write(row)
    except OSError as exc:
        raise PromptPrefillError(f"failed to write last hidden row {output_path}: {exc}") from exc
    return row_bytes


def _count_linear_backend(counts: dict[str, int], result: object | None) -> None:
    if result is None:
        return
    backend = getattr(result, "backend", None)
    if isinstance(backend, str) and backend:
        counts[backend] = counts.get(backend, 0) + 1


def _linear_estimated_flops(result: object | None) -> int:
    if result is None:
        return 0
    if getattr(result, "name", None) == "shared_expert_batch":
        try:
            batch_tokens = int(getattr(result, "batch_tokens"))
            hidden_dim = int(getattr(result, "hidden_dim"))
            intermediate_dim = int(getattr(result, "intermediate_dim"))
        except (TypeError, ValueError):
            return 0
        if batch_tokens <= 0 or hidden_dim <= 0 or intermediate_dim <= 0:
            return 0
        return 6 * batch_tokens * hidden_dim * intermediate_dim
    try:
        batch_tokens = int(getattr(result, "batch_tokens"))
        in_dim = int(getattr(result, "in_dim"))
        out_dim = int(getattr(result, "out_dim"))
    except (TypeError, ValueError):
        return 0
    if batch_tokens <= 0 or in_dim <= 0 or out_dim <= 0:
        return 0
    return 2 * batch_tokens * in_dim * out_dim


def _expert_layout_routed_moe_hidden_dims_by_layer(
    expert_layout_path: str | Path,
    layers: Iterable[int],
) -> dict[int, int]:
    selected = set(layers)
    if not selected:
        return {}
    try:
        payload = json.loads(Path(expert_layout_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list):
        return {}
    out: dict[int, int] = {}
    for item in raw_layers:
        if not isinstance(item, dict) or type(item.get("layer")) is not int:
            continue
        layer_id = int(item["layer"])
        if layer_id not in selected:
            continue
        components = item.get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            if component.get("name") not in {"gate_proj.weight", "up_proj.weight"}:
                continue
            shape = component.get("shape")
            if (
                isinstance(shape, list)
                and len(shape) >= 2
                and type(shape[0]) is int
                and int(shape[0]) > 0
            ):
                out[layer_id] = int(shape[0])
                break
    return out


def _streamed_routed_expert_linear_accounting_for_layers(
    *,
    batch_tokens: int,
    top_k: int,
    hidden_dim: int,
    moe_hidden_dims: Iterable[int],
) -> dict[str, int]:
    if batch_tokens <= 0 or top_k <= 0 or hidden_dim <= 0:
        return {}
    layer_dims = tuple(int(dim) for dim in moe_hidden_dims if int(dim) > 0)
    if not layer_dims:
        return {}
    assignments_per_layer = int(batch_tokens) * int(top_k)
    estimated_flops = sum(
        6 * assignments_per_layer * int(hidden_dim) * moe_hidden
        for moe_hidden in layer_dims
    )
    matrix_count = 3 * len(layer_dims)
    mpp_candidate_dims = tuple(
        moe_hidden
        for moe_hidden in layer_dims
        if (
            batch_tokens >= DEFAULT_MPP_MIN_TOKENS
            and hidden_dim >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
            and moe_hidden >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
        )
    )
    mpp_candidate_flops = sum(
        6 * assignments_per_layer * int(hidden_dim) * moe_hidden
        for moe_hidden in mpp_candidate_dims
    )
    return {
        "layer_count": len(layer_dims),
        "matrix_count": matrix_count,
        "assignments": assignments_per_layer * len(layer_dims),
        "estimated_flops": estimated_flops,
        "mpp_candidate_matrix_count": 3 * len(mpp_candidate_dims),
        "mpp_candidate_estimated_flops": mpp_candidate_flops,
    }


def _stage_results_for_staged_mlp(
    result: PrefillStagedRoutedMLPBlockBatchResult,
):
    if result.tiled_staged_moe is not None:
        return result.tiled_staged_moe.tile_stage_results
    return (result.stage_result,)


def _prefill_stage_io_hotspot_record(
    *,
    chunk_index: int,
    tile_index: int,
    stage_result: object,
) -> PromptPrefillStageIOHotspotRecord:
    summary = stage_result.io_summary
    return PromptPrefillStageIOHotspotRecord(
        chunk_index=chunk_index,
        layer=int(stage_result.layer),
        tile_index=tile_index,
        batch_tokens=int(summary.batch_tokens),
        selected_experts=tuple(int(item) for item in stage_result.selected_experts),
        selected_expert_count=int(summary.selected_expert_count),
        total_assignments=int(summary.total_assignments),
        raw_range_count=int(summary.raw_range_count),
        coalesced_range_count=int(summary.coalesced_range_count),
        planned_read_bytes=int(summary.planned_read_bytes),
        staged_bytes=int(summary.staged_bytes),
        unique_requested_bytes=int(summary.unique_requested_bytes),
        waste_bytes=int(summary.waste_bytes),
        copy_chunk_bytes=int(stage_result.copy_chunk_bytes),
        copy_elapsed_seconds=summary.copy_elapsed_seconds,
        copy_throughput_gib_per_second=summary.copy_throughput_gib_per_second,
        copy_read_calls=int(summary.copy_read_calls),
        copy_write_calls=int(summary.copy_write_calls),
        copy_average_read_bytes=summary.copy_average_read_bytes,
        stage_budget_utilization=float(summary.stage_budget_utilization),
        unique_read_amplification=float(summary.unique_read_amplification),
    )


def _top_prefill_stage_io_hotspots(
    records: list[PromptPrefillStageIOHotspotRecord],
    *,
    metric: str,
) -> tuple[PromptPrefillStageIOHotspotRecord, ...]:
    if metric == "copy":
        key = lambda item: (
            item.copy_elapsed_seconds if item.copy_elapsed_seconds is not None else -1.0,
            item.planned_read_bytes,
            item.coalesced_range_count,
            item.raw_range_count,
            -item.layer,
            -item.tile_index,
        )
    elif metric == "ranges":
        key = lambda item: (
            item.coalesced_range_count,
            item.raw_range_count,
            item.planned_read_bytes,
            item.copy_elapsed_seconds if item.copy_elapsed_seconds is not None else -1.0,
            -item.layer,
            -item.tile_index,
        )
    else:
        raise ValueError(f"unknown stage io hotspot metric {metric!r}")
    return tuple(sorted(records, key=key, reverse=True)[:_EXPERT_STAGE_IO_HOTSPOT_LIMIT])


def _moe_results_for_staged_mlp(
    result: PrefillStagedRoutedMLPBlockBatchResult,
):
    if result.tiled_staged_moe is not None:
        return result.tiled_staged_moe.tile_results
    return (result.staged_moe,)


def _streamed_routed_expert_linear_accounting_for_result(
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
    *,
    moe_hidden_dim: int,
) -> dict[str, int]:
    if result is None or moe_hidden_dim <= 0:
        return {}
    try:
        assignments = sum(
            int(item.batch_plan.total_assignments)
            for item in _stage_results_for_staged_mlp(result)
        )
        batch_tokens = int(result.batch_tokens)
        hidden_dim = int(result.hidden_dim)
    except (AttributeError, TypeError, ValueError):
        return {}
    if assignments <= 0 or batch_tokens <= 0 or hidden_dim <= 0:
        return {}
    estimated_flops = 6 * assignments * hidden_dim * int(moe_hidden_dim)
    mpp_candidate = (
        batch_tokens >= DEFAULT_MPP_MIN_TOKENS
        and hidden_dim >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
        and int(moe_hidden_dim) >= MPP_TENSOR_OPS_MIN_MATRIX_DIM
    )
    return {
        "layer_count": 1,
        "matrix_count": 3,
        "assignments": assignments,
        "estimated_flops": estimated_flops,
        "mpp_candidate_matrix_count": 3 if mpp_candidate else 0,
        "mpp_candidate_estimated_flops": estimated_flops if mpp_candidate else 0,
    }


def _accumulate_streamed_routed_expert_accounting(
    *,
    counts: dict[str, int],
    flops_by_backend: dict[str, int],
    mpp_candidate_totals: dict[str, int],
    mpp_candidate_backend_counts: dict[str, int] | None = None,
    mpp_candidate_backend_flops: dict[str, int] | None = None,
    streamed_totals: dict[str, int],
    accounting: dict[str, int],
) -> None:
    estimated_flops = int(accounting.get("estimated_flops") or 0)
    if estimated_flops <= 0:
        return
    matrix_count = int(accounting.get("matrix_count") or 0)
    if matrix_count > 0:
        counts["custom-metal"] = counts.get("custom-metal", 0) + matrix_count
        streamed_totals["matrix_count"] = (
            streamed_totals.get("matrix_count", 0) + matrix_count
        )
    flops_by_backend["custom-metal"] = (
        flops_by_backend.get("custom-metal", 0) + estimated_flops
    )
    streamed_totals["estimated_flops"] = (
        streamed_totals.get("estimated_flops", 0) + estimated_flops
    )
    for key in ("layer_count", "assignments"):
        value = int(accounting.get(key) or 0)
        if value > 0:
            streamed_totals[key] = streamed_totals.get(key, 0) + value
    mpp_count = int(accounting.get("mpp_candidate_matrix_count") or 0)
    mpp_flops = int(accounting.get("mpp_candidate_estimated_flops") or 0)
    if mpp_count > 0:
        mpp_candidate_totals["count"] = (
            mpp_candidate_totals.get("count", 0) + mpp_count
        )
        if mpp_candidate_backend_counts is not None:
            mpp_candidate_backend_counts["custom-metal"] = (
                mpp_candidate_backend_counts.get("custom-metal", 0) + mpp_count
            )
        streamed_totals["mpp_candidate_matrix_count"] = (
            streamed_totals.get("mpp_candidate_matrix_count", 0) + mpp_count
        )
    if mpp_flops > 0:
        mpp_candidate_totals["estimated_flops"] = (
            mpp_candidate_totals.get("estimated_flops", 0) + mpp_flops
        )
        if mpp_candidate_backend_flops is not None:
            mpp_candidate_backend_flops["custom-metal"] = (
                mpp_candidate_backend_flops.get("custom-metal", 0) + mpp_flops
            )
        streamed_totals["mpp_candidate_estimated_flops"] = (
            streamed_totals.get("mpp_candidate_estimated_flops", 0) + mpp_flops
        )


def _accumulate_linear_flops(totals: dict[str, int], result: object | None) -> None:
    if result is None:
        return
    backend = getattr(result, "backend", None)
    if not isinstance(backend, str) or not backend:
        return
    flops = _linear_estimated_flops(result)
    if flops > 0:
        totals[backend] = totals.get(backend, 0) + flops


def _accumulate_linear_mpp_candidates(
    totals: dict[str, int],
    result: object | None,
    *,
    backend_counts: dict[str, int] | None = None,
    backend_flops: dict[str, int] | None = None,
) -> None:
    if result is None:
        return
    flops = _linear_estimated_flops(result)
    if flops <= 0:
        return
    try:
        batch_tokens = int(getattr(result, "batch_tokens"))
        in_dim = int(getattr(result, "in_dim"))
        out_dim = int(getattr(result, "out_dim"))
    except (AttributeError, TypeError, ValueError):
        return
    if (
        batch_tokens < DEFAULT_MPP_MIN_TOKENS
        or in_dim < MPP_TENSOR_OPS_MIN_MATRIX_DIM
        or out_dim < MPP_TENSOR_OPS_MIN_MATRIX_DIM
    ):
        return
    totals["count"] = totals.get("count", 0) + 1
    totals["estimated_flops"] = totals.get("estimated_flops", 0) + flops
    backend = getattr(result, "backend", None)
    if isinstance(backend, str) and backend:
        if backend_counts is not None:
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if backend_flops is not None:
            backend_flops[backend] = backend_flops.get(backend, 0) + flops


def _accumulate_linear_elapsed(
    totals: dict[str, float],
    result: object | None,
) -> None:
    if result is None:
        return
    backend = getattr(result, "backend", None)
    if not isinstance(backend, str) or not backend:
        return
    elapsed = getattr(result, "elapsed_seconds", None)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        return
    elapsed_seconds = float(elapsed)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        return
    totals[backend] = totals.get(backend, 0.0) + elapsed_seconds


def _component_stats_bucket(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
    component: str,
) -> dict[str, dict[str, int] | dict[str, float]]:
    bucket = totals.get(component)
    if bucket is None:
        bucket = {
            "linear_backend_counts": {},
            "linear_backend_flops": {},
            "linear_backend_elapsed_seconds": {},
        }
        totals[component] = bucket
    return bucket


def _accumulate_linear_component_stats(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
    component: str,
    result: object | None,
) -> None:
    if result is None:
        return
    backend = getattr(result, "backend", None)
    if not isinstance(backend, str) or not backend:
        return
    bucket = _component_stats_bucket(totals, component)
    counts = bucket["linear_backend_counts"]
    flops_by_backend = bucket["linear_backend_flops"]
    elapsed_by_backend = bucket["linear_backend_elapsed_seconds"]
    counts[backend] = int(counts.get(backend, 0)) + 1
    flops = _linear_estimated_flops(result)
    if flops > 0:
        flops_by_backend[backend] = int(flops_by_backend.get(backend, 0)) + flops
    elapsed = getattr(result, "elapsed_seconds", None)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        return
    elapsed_seconds = float(elapsed)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        return
    elapsed_by_backend[backend] = (
        float(elapsed_by_backend.get(backend, 0.0)) + elapsed_seconds
    )


def _accumulate_streamed_component_stats(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
    *,
    component: str,
    accounting: dict[str, int],
    elapsed_seconds: float | None,
) -> None:
    matrix_count = int(accounting.get("matrix_count") or 0)
    estimated_flops = int(accounting.get("estimated_flops") or 0)
    if matrix_count <= 0 and estimated_flops <= 0:
        return
    bucket = _component_stats_bucket(totals, component)
    counts = bucket["linear_backend_counts"]
    flops_by_backend = bucket["linear_backend_flops"]
    elapsed_by_backend = bucket["linear_backend_elapsed_seconds"]
    if matrix_count > 0:
        counts["custom-metal"] = int(counts.get("custom-metal", 0)) + matrix_count
    if estimated_flops > 0:
        flops_by_backend["custom-metal"] = (
            int(flops_by_backend.get("custom-metal", 0)) + estimated_flops
        )
    if elapsed_seconds is not None and elapsed_seconds >= 0:
        elapsed_by_backend["custom-metal"] = (
            float(elapsed_by_backend.get("custom-metal", 0.0)) + elapsed_seconds
        )


def _sorted_linear_component_stats(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for component, bucket in sorted(totals.items()):
        counts = {
            backend: int(count)
            for backend, count in sorted(
                bucket.get("linear_backend_counts", {}).items()
            )
            if int(count) > 0
        }
        flops_by_backend = {
            backend: int(flops)
            for backend, flops in sorted(
                bucket.get("linear_backend_flops", {}).items()
            )
            if int(flops) > 0
        }
        elapsed_by_backend = {
            backend: float(elapsed)
            for backend, elapsed in sorted(
                bucket.get("linear_backend_elapsed_seconds", {}).items()
            )
            if float(elapsed) >= 0.0
        }
        payload: dict[str, object] = {
            "linear_backend_counts": counts,
            "linear_backend_flops": flops_by_backend,
            "linear_backend_elapsed_seconds": elapsed_by_backend,
        }
        tflops = _linear_backend_estimated_tflops(
            flops_by_backend=flops_by_backend,
            elapsed_by_backend=elapsed_by_backend,
        )
        if tflops:
            payload["linear_backend_estimated_tflops"] = tflops
        result[component] = payload
    return result


def _linear_backend_estimated_tflops(
    *,
    flops_by_backend: dict[str, int],
    elapsed_by_backend: dict[str, float],
) -> dict[str, float]:
    rates: dict[str, float] = {}
    for backend, flops in flops_by_backend.items():
        elapsed = elapsed_by_backend.get(backend)
        if elapsed is None or elapsed <= 0:
            continue
        rates[backend] = flops / elapsed / 1e12
    return dict(sorted(rates.items()))


def _accumulate_linear_scratch(totals: dict[str, int], result: object | None) -> None:
    if result is None:
        return
    scratch = getattr(result, "matrix_scratch_bytes", 0)
    f32 = getattr(result, "matrix_f32_bytes", 0)
    raw = getattr(result, "matrix_raw_conversion_bytes", 0)
    if isinstance(scratch, int) and scratch > 0:
        totals["total_scratch"] = totals.get("total_scratch", 0) + scratch
        totals["max_scratch"] = max(totals.get("max_scratch", 0), scratch)
    if isinstance(f32, int) and f32 > 0:
        totals["total_f32"] = totals.get("total_f32", 0) + f32
    if isinstance(raw, int) and raw > 0:
        totals["total_raw_conversion"] = totals.get("total_raw_conversion", 0) + raw


def _count_attention_backends(
    counts: dict[str, int],
    result: PrefillAttentionBlockBatchResult,
) -> None:
    prefix = result.projections.prefix
    _count_linear_backend(counts, prefix.q_a_proj)
    _count_linear_backend(counts, prefix.kv_a_proj_with_mqa)
    _count_linear_backend(counts, result.projections.q_b_proj)
    _count_linear_backend(counts, result.projections.kv_b_proj)
    _count_linear_backend(counts, result.attention_output.o_proj)


def _accumulate_attention_flops(
    totals: dict[str, int],
    result: PrefillAttentionBlockBatchResult,
) -> None:
    prefix = result.projections.prefix
    _accumulate_linear_flops(totals, prefix.q_a_proj)
    _accumulate_linear_flops(totals, prefix.kv_a_proj_with_mqa)
    _accumulate_linear_flops(totals, result.projections.q_b_proj)
    _accumulate_linear_flops(totals, result.projections.kv_b_proj)
    _accumulate_linear_flops(totals, result.attention_output.o_proj)


def _accumulate_attention_mpp_candidates(
    totals: dict[str, int],
    result: PrefillAttentionBlockBatchResult,
    *,
    backend_counts: dict[str, int] | None = None,
    backend_flops: dict[str, int] | None = None,
) -> None:
    prefix = result.projections.prefix
    _accumulate_linear_mpp_candidates(
        totals,
        prefix.q_a_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        prefix.kv_a_proj_with_mqa,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.projections.q_b_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.projections.kv_b_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.attention_output.o_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )


def _accumulate_attention_elapsed(
    totals: dict[str, float],
    result: PrefillAttentionBlockBatchResult,
) -> None:
    prefix = result.projections.prefix
    _accumulate_linear_elapsed(totals, prefix.q_a_proj)
    _accumulate_linear_elapsed(totals, prefix.kv_a_proj_with_mqa)
    _accumulate_linear_elapsed(totals, result.projections.q_b_proj)
    _accumulate_linear_elapsed(totals, result.projections.kv_b_proj)
    _accumulate_linear_elapsed(totals, result.attention_output.o_proj)


def _accumulate_attention_component_stats(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
    result: PrefillAttentionBlockBatchResult,
) -> None:
    prefix = result.projections.prefix
    _accumulate_linear_component_stats(totals, "attention.q_a_proj", prefix.q_a_proj)
    _accumulate_linear_component_stats(
        totals,
        "attention.kv_a_proj_with_mqa",
        prefix.kv_a_proj_with_mqa,
    )
    _accumulate_linear_component_stats(
        totals,
        "attention.q_b_proj",
        result.projections.q_b_proj,
    )
    _accumulate_linear_component_stats(
        totals,
        "attention.kv_b_proj",
        result.projections.kv_b_proj,
    )
    _accumulate_linear_component_stats(
        totals,
        "attention.o_proj",
        result.attention_output.o_proj,
    )


def _count_dense_backends(
    counts: dict[str, int],
    result: PrefillDenseMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _count_linear_backend(counts, result.gate_proj)
    _count_linear_backend(counts, result.up_proj)
    _count_linear_backend(counts, result.down_proj)


def _accumulate_dense_flops(
    totals: dict[str, int],
    result: PrefillDenseMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_flops(totals, result.gate_proj)
    _accumulate_linear_flops(totals, result.up_proj)
    _accumulate_linear_flops(totals, result.down_proj)


def _accumulate_dense_mpp_candidates(
    totals: dict[str, int],
    result: PrefillDenseMLPBlockBatchResult | None,
    *,
    backend_counts: dict[str, int] | None = None,
    backend_flops: dict[str, int] | None = None,
) -> None:
    if result is None:
        return
    _accumulate_linear_mpp_candidates(
        totals,
        result.gate_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.up_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.down_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )


def _accumulate_dense_elapsed(
    totals: dict[str, float],
    result: PrefillDenseMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_elapsed(totals, result.gate_proj)
    _accumulate_linear_elapsed(totals, result.up_proj)
    _accumulate_linear_elapsed(totals, result.down_proj)


def _accumulate_dense_component_stats(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
    result: PrefillDenseMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_component_stats(totals, "dense_mlp.gate_proj", result.gate_proj)
    _accumulate_linear_component_stats(totals, "dense_mlp.up_proj", result.up_proj)
    _accumulate_linear_component_stats(totals, "dense_mlp.down_proj", result.down_proj)


def _count_staged_backends(
    counts: dict[str, int],
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _count_linear_backend(counts, result.router_gate_proj)
    _count_linear_backend(counts, result.shared_expert_batch)
    _count_linear_backend(counts, result.shared_gate_proj)
    _count_linear_backend(counts, result.shared_up_proj)
    _count_linear_backend(counts, result.shared_down_proj)


def _accumulate_staged_flops(
    totals: dict[str, int],
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_flops(totals, result.router_gate_proj)
    _accumulate_linear_flops(totals, result.shared_expert_batch)
    _accumulate_linear_flops(totals, result.shared_gate_proj)
    _accumulate_linear_flops(totals, result.shared_up_proj)
    _accumulate_linear_flops(totals, result.shared_down_proj)


def _accumulate_staged_mpp_candidates(
    totals: dict[str, int],
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
    *,
    backend_counts: dict[str, int] | None = None,
    backend_flops: dict[str, int] | None = None,
) -> None:
    if result is None:
        return
    _accumulate_linear_mpp_candidates(
        totals,
        result.router_gate_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.shared_expert_batch,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.shared_gate_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.shared_up_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )
    _accumulate_linear_mpp_candidates(
        totals,
        result.shared_down_proj,
        backend_counts=backend_counts,
        backend_flops=backend_flops,
    )


def _accumulate_staged_elapsed(
    totals: dict[str, float],
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_elapsed(totals, result.router_gate_proj)
    _accumulate_linear_elapsed(totals, result.shared_expert_batch)
    _accumulate_linear_elapsed(totals, result.shared_gate_proj)
    _accumulate_linear_elapsed(totals, result.shared_up_proj)
    _accumulate_linear_elapsed(totals, result.shared_down_proj)


def _accumulate_staged_component_stats(
    totals: dict[str, dict[str, dict[str, int] | dict[str, float]]],
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_component_stats(
        totals,
        "moe.router_gate_proj",
        result.router_gate_proj,
    )
    _accumulate_linear_component_stats(
        totals,
        "moe.shared_expert_fused",
        result.shared_expert_batch,
    )
    _accumulate_linear_component_stats(
        totals,
        "moe.shared_gate_proj",
        result.shared_gate_proj,
    )
    _accumulate_linear_component_stats(
        totals,
        "moe.shared_up_proj",
        result.shared_up_proj,
    )
    _accumulate_linear_component_stats(
        totals,
        "moe.shared_down_proj",
        result.shared_down_proj,
    )


def _accumulate_attention_scratch(
    totals: dict[str, int],
    result: PrefillAttentionBlockBatchResult,
) -> None:
    prefix = result.projections.prefix
    _accumulate_linear_scratch(totals, prefix.q_a_proj)
    _accumulate_linear_scratch(totals, prefix.kv_a_proj_with_mqa)
    _accumulate_linear_scratch(totals, result.projections.q_b_proj)
    _accumulate_linear_scratch(totals, result.projections.kv_b_proj)
    _accumulate_linear_scratch(totals, result.attention_output.o_proj)


def _accumulate_dense_scratch(
    totals: dict[str, int],
    result: PrefillDenseMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_scratch(totals, result.gate_proj)
    _accumulate_linear_scratch(totals, result.up_proj)
    _accumulate_linear_scratch(totals, result.down_proj)


def _accumulate_staged_scratch(
    totals: dict[str, int],
    result: PrefillStagedRoutedMLPBlockBatchResult | None,
) -> None:
    if result is None:
        return
    _accumulate_linear_scratch(totals, result.router_gate_proj)
    _accumulate_linear_scratch(totals, result.shared_expert_batch)
    _accumulate_linear_scratch(totals, result.shared_gate_proj)
    _accumulate_linear_scratch(totals, result.shared_up_proj)
    _accumulate_linear_scratch(totals, result.shared_down_proj)


def _require_disk_budget(
    *,
    output_dir: Path,
    required_bytes: int,
    safety_margin_bytes: int,
    label: str,
) -> None:
    budget = disk_budget(
        output_dir,
        required_bytes,
        safety_margin_bytes=safety_margin_bytes,
    )
    if not budget.ok:
        raise PromptPrefillError(
            f"not enough free disk for {label}: "
            f"need {required_bytes + safety_margin_bytes} bytes including margin, "
            f"have {budget.available_bytes} bytes"
        )


def _planned_chunk_work_bytes(
    *,
    batch_tokens: int,
    hidden_dim: int,
    layer_count: int,
) -> int:
    hidden_batch_bytes = batch_tokens * hidden_dim * 4
    # Main prompt-prebuild files kept live for a chunk: embedding, attention
    # output and layer output per selected layer. Substeps keep their own caps.
    return hidden_batch_bytes * (1 + 2 * layer_count)


def _contains_path(parent: Path, child: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _should_cleanup_chunk_dir(
    *,
    chunk_dir: Path,
    output_last: Path,
    output_final_chunk: Path | None,
) -> bool:
    protected = [output_last]
    if output_final_chunk is not None:
        protected.append(output_final_chunk)
    return not any(_contains_path(chunk_dir, path) for path in protected)


def _cleanup_prompt_prefill_chunk_dirs(
    *,
    root: Path,
    output_last: Path,
    output_final_chunk: Path | None,
) -> None:
    try:
        children = tuple(root.iterdir())
    except OSError:
        return
    for child in children:
        if not child.is_dir() or not child.name.startswith("chunk_"):
            continue
        if _should_cleanup_chunk_dir(
            chunk_dir=child,
            output_last=output_last,
            output_final_chunk=output_final_chunk,
        ):
            shutil.rmtree(child, ignore_errors=True)


def _recheck_live_memory_budget(budget: LiveMemoryBudget) -> None:
    if budget.min_available_memory_bytes <= 0:
        return
    try:
        check_live_memory_budget(
            estimated_live_working_set_bytes=budget.estimated_live_working_set_bytes,
            max_live_working_set_bytes=budget.max_live_working_set_bytes,
            min_available_memory_bytes=budget.min_available_memory_bytes,
        )
    except GenerationGuardError as exc:
        raise _PromptPrefillLiveMemoryError(
            f"live memory guard failed during prompt prefill: {exc}"
        ) from exc


def run_prompt_prefill(
    *,
    runner_path: str | Path,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    prompt_token_ids: Iterable[int],
    output_last_hidden_f32_path: str | Path,
    output_final_chunk_f32_path: str | Path | None = None,
    layers: Iterable[int] | None = None,
    dense_layers: Iterable[int] | None = None,
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
    start_position: int = 0,
    prompt_chunk_tokens: int = 64,
    max_prompt_batch_mib: float = 1024.0,
    num_heads: int = 0,
    qk_nope_dim: int = 0,
    rope_dim: int = 0,
    v_head_dim: int = 0,
    mla_kv_b_cache_dir: str | Path | None = None,
    mla_key_cache: bool = False,
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
    max_embedding_row_mib: float = 64.0,
    max_slot_mib: float = 256.0,
    max_router_mib: float = 64.0,
    max_resident_matrix_mib: float = 512.0,
    max_cache_file_mib: float = 32768.0,
    max_cache_write_mib: float = 4096.0,
    max_cache_read_mib: float = 256.0,
    max_runner_scratch_mib: float = 4096.0,
    max_live_working_set_mib: float | None = 8192.0,
    min_free_unified_memory_gib: float = 0.0,
    moe_token_block: MoETokenBlock = "auto",
    moe_output_accumulator: MoEOutputAccumulator = "env",
    static_capacity_per_expert: StaticCapacityPerExpert = None,
    write_static_capacity_json: bool = False,
    allow_static_capacity_overflow: bool = False,
    expert_stage_merge_gap_kib: float = 0.0,
    expert_stage_align_kib: float = 4.0,
    max_stage_mib: float = 4096.0,
    max_compact_stage_mib: float = 4096.0,
    copy_chunk_mib: float = 8.0,
    stage_disk_safety_margin_bytes: int = 0,
    prefill_ssd_read_gib_per_second: float = 0.0,
    prefill_max_routed_read_seconds: float = 0.0,
    expert_stage_max_raw_ranges: int = 0,
    expert_stage_max_coalesced_ranges: int = 0,
    expert_stage_tiling: bool = False,
    prefill_linear_backend: str = "auto",
    prefill_min_accelerated_flop_fraction: float = 0.0,
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    router_hybrid_margin_threshold: float = 0.0,
    persistent_moe_plan_server: bool = False,
    persistent_resident_linear_server: bool = False,
    persistent_attention_projection_server: bool = False,
    persistent_attention_output_server: bool = False,
    persistent_shared_expert_server: bool = False,
    persistent_rope_split_server: bool = False,
    persistent_mla_attention_server: bool = False,
    persistent_rmsnorm_server: bool = False,
    dsa_indexer_types: Iterable[str] | None = None,
    dsa_index_topk: int | None = None,
    dsa_index_n_heads: int | None = None,
    dsa_qk_rope_dim: int | None = None,
    dsa_rope_interleave: bool = False,
    dsa_layer_norm_eps: float = 1e-6,
    write_dsa_future_cache: bool = True,
    expected_vocab_size: int | None = None,
    expected_hidden_size: int | None = None,
    echo_runner_output: bool = True,
) -> PromptPrefillResult:
    prefill_started = time.perf_counter()
    if type(write_dsa_future_cache) is not bool:
        raise PromptPrefillError("write_dsa_future_cache must be a boolean")
    if type(mla_key_cache) is not bool:
        raise PromptPrefillError("mla_key_cache must be a boolean")
    if type(expert_stage_tiling) is not bool:
        raise PromptPrefillError("expert_stage_tiling must be a boolean")
    if type(persistent_moe_plan_server) is not bool:
        raise PromptPrefillError("persistent_moe_plan_server must be a boolean")
    if type(persistent_resident_linear_server) is not bool:
        raise PromptPrefillError(
            "persistent_resident_linear_server must be a boolean"
        )
    if type(persistent_attention_projection_server) is not bool:
        raise PromptPrefillError(
            "persistent_attention_projection_server must be a boolean"
        )
    if type(persistent_attention_output_server) is not bool:
        raise PromptPrefillError(
            "persistent_attention_output_server must be a boolean"
        )
    if type(persistent_shared_expert_server) is not bool:
        raise PromptPrefillError(
            "persistent_shared_expert_server must be a boolean"
        )
    if type(persistent_rope_split_server) is not bool:
        raise PromptPrefillError("persistent_rope_split_server must be a boolean")
    if type(persistent_mla_attention_server) is not bool:
        raise PromptPrefillError("persistent_mla_attention_server must be a boolean")
    if type(persistent_rmsnorm_server) is not bool:
        raise PromptPrefillError("persistent_rmsnorm_server must be a boolean")
    if (
        isinstance(router_hybrid_margin_threshold, bool)
        or not isinstance(router_hybrid_margin_threshold, (int, float))
        or not math.isfinite(float(router_hybrid_margin_threshold))
        or float(router_hybrid_margin_threshold) < 0.0
    ):
        raise PromptPrefillError(
            "router_hybrid_margin_threshold must be a non-negative finite number"
        )
    router_hybrid_margin_threshold = float(router_hybrid_margin_threshold)
    if (
        isinstance(prefill_min_accelerated_flop_fraction, bool)
        or not isinstance(prefill_min_accelerated_flop_fraction, (int, float))
        or not 0.0 <= float(prefill_min_accelerated_flop_fraction) <= 1.0
    ):
        raise PromptPrefillError(
            "prefill_min_accelerated_flop_fraction must be 0..1"
        )
    prefill_min_accelerated_flop_fraction = float(
        prefill_min_accelerated_flop_fraction
    )
    prompt = tuple(
        _nonnegative_integer_value(token, label="prompt_token_ids")
        for token in prompt_token_ids
    )
    if not prompt:
        raise PromptPrefillError("prompt_token_ids must be non-empty")
    start_position = _nonnegative_integer_value(
        start_position,
        label="start_position",
    )
    prompt_chunk_tokens = _positive_integer_value(
        prompt_chunk_tokens,
        label="prompt_chunk_tokens",
    )
    num_heads = _positive_integer_value(num_heads, label="num_heads")
    qk_nope_dim = _positive_integer_value(qk_nope_dim, label="qk_nope_dim")
    rope_dim = _positive_integer_value(rope_dim, label="rope_dim")
    v_head_dim = _positive_integer_value(v_head_dim, label="v_head_dim")
    kv_lora_dim = _optional_positive_integer_value(
        kv_lora_dim,
        label="kv_lora_dim",
    )
    cache_position_offset = _nonnegative_integer_value(
        cache_position_offset,
        label="cache_position_offset",
    )
    top_k = _positive_integer_value(top_k, label="top_k")
    max_k = _positive_integer_value(max_k, label="max_k")
    router_n_group = _optional_positive_integer_value(
        router_n_group,
        label="router_n_group",
    )
    router_topk_group = _optional_positive_integer_value(
        router_topk_group,
        label="router_topk_group",
    )
    dsa_index_topk = _optional_positive_integer_value(
        dsa_index_topk,
        label="dsa_index_topk",
    )
    dsa_index_n_heads = _optional_positive_integer_value(
        dsa_index_n_heads,
        label="dsa_index_n_heads",
    )
    dsa_qk_rope_dim = _optional_positive_integer_value(
        dsa_qk_rope_dim,
        label="dsa_qk_rope_dim",
    )
    expected_vocab_size = _optional_positive_integer_value(
        expected_vocab_size,
        label="expected_vocab_size",
    )
    expected_hidden_size = _optional_positive_integer_value(
        expected_hidden_size,
        label="expected_hidden_size",
    )
    try:
        max_prompt_batch_mib = _positive_limit(
            "max_prompt_batch_mib",
            max_prompt_batch_mib,
        )
        max_embedding_row_mib = _positive_limit(
            "max_embedding_row_mib",
            max_embedding_row_mib,
        )
        max_slot_mib = _positive_limit("max_slot_mib", max_slot_mib)
        max_router_mib = _positive_limit("max_router_mib", max_router_mib)
        max_resident_matrix_mib = _positive_limit(
            "max_resident_matrix_mib",
            max_resident_matrix_mib,
        )
        max_cache_file_mib = _positive_limit(
            "max_cache_file_mib",
            max_cache_file_mib,
        )
        max_cache_write_mib = _positive_limit(
            "max_cache_write_mib",
            max_cache_write_mib,
        )
        max_cache_read_mib = _positive_limit(
            "max_cache_read_mib",
            max_cache_read_mib,
        )
        max_runner_scratch_mib = _positive_limit(
            "max_runner_scratch_mib",
            max_runner_scratch_mib,
        )
        expert_stage_merge_gap_kib = _nonnegative_limit(
            "expert_stage_merge_gap_kib",
            expert_stage_merge_gap_kib,
        )
        expert_stage_align_kib = _positive_limit(
            "expert_stage_align_kib",
            expert_stage_align_kib,
        )
        max_stage_mib = _positive_limit("max_stage_mib", max_stage_mib)
        max_compact_stage_mib = _positive_limit(
            "max_compact_stage_mib",
            max_compact_stage_mib,
        )
        copy_chunk_mib = _positive_limit("copy_chunk_mib", copy_chunk_mib)
        stage_disk_safety_margin_bytes = _nonnegative_integer_value(
            stage_disk_safety_margin_bytes,
            label="stage_disk_safety_margin_bytes",
        )
        prefill_ssd_read_gib_per_second = _nonnegative_limit(
            "prefill_ssd_read_gib_per_second",
            prefill_ssd_read_gib_per_second,
        )
        prefill_max_routed_read_seconds = _nonnegative_limit(
            "prefill_max_routed_read_seconds",
            prefill_max_routed_read_seconds,
        )
        expert_stage_max_raw_ranges = _nonnegative_integer_value(
            expert_stage_max_raw_ranges,
            label="expert_stage_max_raw_ranges",
        )
        expert_stage_max_coalesced_ranges = _nonnegative_integer_value(
            expert_stage_max_coalesced_ranges,
            label="expert_stage_max_coalesced_ranges",
        )
        if (
            prefill_max_routed_read_seconds > 0
            and prefill_ssd_read_gib_per_second <= 0
        ):
            raise PromptPrefillError(
                "prefill_ssd_read_gib_per_second must be positive when "
                "prefill_max_routed_read_seconds is set"
            )
        min_free_unified_memory_gib = _nonnegative_limit(
            "min_free_unified_memory_gib",
            min_free_unified_memory_gib,
        )
        if max_live_working_set_mib is not None:
            max_live_working_set_mib = _nonnegative_limit(
                "max_live_working_set_mib",
                max_live_working_set_mib,
            )
    except GenerationGuardError as exc:
        raise PromptPrefillError(str(exc)) from exc
    try:
        decode_cache_layout = load_decode_cache_layout(cache_layout_path)
    except DecodeCacheError as exc:
        raise PromptPrefillError(f"failed to load decode cache layout: {exc}") from exc
    end_position = start_position + len(prompt)
    if end_position > decode_cache_layout.max_context_tokens:
        raise PromptPrefillError(
            f"prompt positions [{start_position}, {end_position}) exceed "
            f"decode cache context {decode_cache_layout.max_context_tokens}"
        )
    try:
        _moe_token_block_arg, moe_token_block_value = _normalize_moe_token_block(
            moe_token_block
        )
        moe_output_accumulator = _normalize_moe_output_accumulator(
            moe_output_accumulator
        )
    except StagedMoEError as exc:
        raise PromptPrefillError(str(exc)) from exc
    static_capacity_request = _normalize_static_capacity_per_expert(
        static_capacity_per_expert
    )
    _reject_unsafe_strict_static_capacity(
        static_capacity_request=static_capacity_request,
        max_batch_tokens=min(prompt_chunk_tokens, len(prompt)),
        allow_static_capacity_overflow=allow_static_capacity_overflow,
    )
    if num_heads <= 0 or qk_nope_dim <= 0 or rope_dim <= 0 or v_head_dim <= 0:
        raise PromptPrefillError("attention dimensions must be positive")
    selected_layers, selected_dense = _resolve_layers(
        expert_layout_path=expert_layout_path,
        layers=layers,
        dense_layers=dense_layers,
    )
    dsa_types = tuple(str(item).lower() for item in dsa_indexer_types or ())
    invalid_dsa_types = sorted(set(dsa_types) - {"none", "full", "shared"})
    if invalid_dsa_types:
        joined = ",".join(invalid_dsa_types)
        raise PromptPrefillError(f"invalid DSA indexer_type values: {joined}")
    bad_shared_layer = first_shared_indexer_without_previous_full(
        dsa_types,
        selected_layers=selected_layers,
    )
    if bad_shared_layer is not None:
        raise PromptPrefillError(
            f"selected DSA layer {bad_shared_layer} is shared but no previous "
            "selected full-indexer layer is available"
        )
    if dsa_types:
        if len(dsa_types) <= max(selected_layers):
            raise PromptPrefillError("dsa_indexer_types does not cover selected layers")
        if dsa_index_topk is None or dsa_index_topk <= 0:
            raise PromptPrefillError("DSA index_topk must be positive when DSA is enabled")
        if dsa_index_n_heads is None or dsa_index_n_heads <= 0:
            raise PromptPrefillError("DSA index_n_heads must be positive when DSA is enabled")
        if dsa_layer_norm_eps <= 0:
            raise PromptPrefillError("dsa_layer_norm_eps must be positive")
    dense_set = set(selected_dense)
    routed_moe_hidden_dims_by_layer = _expert_layout_routed_moe_hidden_dims_by_layer(
        expert_layout_path,
        (layer for layer in selected_layers if layer not in dense_set),
    )
    output_last = Path(output_last_hidden_f32_path)
    output_final_chunk = (
        Path(output_final_chunk_f32_path)
        if output_final_chunk_f32_path is not None
        else None
    )
    max_prompt_batch_bytes = int(max_prompt_batch_mib * 1024 * 1024)
    max_embedding_row_bytes = int(max_embedding_row_mib * 1024 * 1024)
    try:
        embedding_meta = _load_embedding_metadata(resident_layout_path)
        planned_hidden_dim = embedding_meta.hidden_dim
    except EmbeddingError as exc:
        raise PromptPrefillError(str(exc)) from exc
    try:
        live_memory_estimate = estimate_prompt_prefill_live_memory(
            max_prompt_batch_mib=max_prompt_batch_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            max_cache_read_mib=max_cache_read_mib,
            max_cache_write_mib=max_cache_write_mib,
            copy_chunk_mib=copy_chunk_mib,
        )
        live_memory_budget = check_live_memory_budget(
            estimated_live_working_set_bytes=(
                live_memory_estimate.estimated_live_working_set_bytes
            ),
            max_live_working_set_bytes=(
                int(max_live_working_set_mib * 1024**2)
                if max_live_working_set_mib is not None
                else None
            ),
            min_available_memory_bytes=int(min_free_unified_memory_gib * 1024**3),
        )
    except GenerationGuardError as exc:
        raise PromptPrefillError(str(exc)) from exc

    created_work_dir = False
    if work_dir is None:
        root = Path(tempfile.mkdtemp(prefix="largerlm-prefill-prompt-", dir="/private/tmp"))
        created_work_dir = True
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)

    chunk_records: list[PromptPrefillChunkRecord] = []
    hidden_dim = 0
    total_embedding_read = 0
    total_embedding_output = 0
    total_staged = 0
    total_compact_stage = 0
    total_compact_stage_materialized = 0
    max_staged = 0
    max_compact_stage = 0
    max_compact_stage_materialized = 0
    total_stage_plus_compact = 0
    total_stage_plus_compact_materialized = 0
    max_stage_plus_compact = 0
    max_stage_plus_compact_materialized = 0
    total_expert_stage_serial_read = 0
    total_expert_stage_unique_requested = 0
    total_expert_stage_planned_read = 0
    total_expert_stage_waste = 0
    total_expert_stage_coalesced_savings = 0
    total_expert_stage_planned_read_seconds = (
        0.0 if prefill_ssd_read_gib_per_second > 0 else None
    )
    total_expert_stage_copy_elapsed_seconds: float | None = None
    total_expert_stage_copy_bytes = 0
    total_expert_stage_copy_read_calls = 0
    total_expert_stage_copy_write_calls = 0
    total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib: dict[
        str,
        int,
    ] = {}
    expert_stage_io_hotspot_records: list[PromptPrefillStageIOHotspotRecord] = []
    total_expert_stage_read_advice_attempted_ranges = 0
    total_expert_stage_read_advice_calls = 0
    total_expert_stage_read_advice_bytes = 0
    total_expert_stage_read_advice_failures = 0
    total_expert_stage_raw_ranges = 0
    total_expert_stage_coalesced_ranges = 0
    max_expert_stage_raw_ranges = 0
    max_expert_stage_coalesced_ranges = 0
    total_expert_stage_raw_ranges_ok = (
        None if expert_stage_max_raw_ranges <= 0 else True
    )
    total_expert_stage_coalesced_ranges_ok = (
        None if expert_stage_max_coalesced_ranges <= 0 else True
    )
    max_expert_stage_unique_read_amplification = 0.0
    max_expert_stage_stage_budget_utilization = 0.0
    total_routed_assignments = 0
    total_routed_unique_slots = 0
    max_routed_unique_per_call = 0
    max_routed_tokens_per_expert = 0
    moe_token_block_mode_counts: dict[str, int] = {}
    max_effective_moe_token_block = 0
    max_moe_max_expert_tokens = 0
    max_moe_batch_buffer_bytes = 0
    max_moe_estimated_peak_bytes = 0
    moe_plan_server_plan_count = 0
    moe_plan_server_job_count = 0
    routed_moe_runner_command_count = 0
    max_static_capacity_per_expert = 0
    total_static_capacity_used_slots = 0
    total_static_capacity_slots = 0
    total_static_capacity_overflow = 0
    total_static_capacity_binary_bytes = 0
    estimated_peak = 0
    linear_backend_counts: dict[str, int] = {}
    linear_backend_flops: dict[str, int] = {}
    linear_backend_elapsed_seconds: dict[str, float] = {}
    linear_component_stats: dict[
        str,
        dict[str, dict[str, int] | dict[str, float]],
    ] = {}
    linear_scratch_totals: dict[str, int] = {}
    mpp_candidate_totals: dict[str, int] = {}
    mpp_candidate_backend_counts: dict[str, int] = {}
    mpp_candidate_backend_flops: dict[str, int] = {}
    streamed_routed_expert_totals: dict[str, int] = {}
    moe_plan_server_session: StagedRoutedMoEBatchPlanServerSession | None = None
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None
    attention_projection_server_session: AttentionProjectionsServerSession | None = None
    attention_output_server_session: AttentionOutputBatchServerSession | None = None
    shared_expert_server_session: ResidentSharedExpertBatchServerSession | None = None
    rope_split_server_session: RopeSplitBatchServerSession | None = None
    mla_attention_server_session: MLAAttentionBatchServerSession | None = None
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None
    try:
        if persistent_moe_plan_server:
            moe_plan_server_session = StagedRoutedMoEBatchPlanServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
                moe_output_accumulator=moe_output_accumulator,
            )
            moe_plan_server_session.start()
        if persistent_resident_linear_server:
            resident_linear_server_session = ResidentBatchLinearServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
            )
            resident_linear_server_session.start()
        if persistent_attention_projection_server:
            attention_projection_server_session = AttentionProjectionsServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
            )
            attention_projection_server_session.start()
        if persistent_attention_output_server:
            attention_output_server_session = AttentionOutputBatchServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
            )
            attention_output_server_session.start()
        if persistent_shared_expert_server:
            shared_expert_server_session = ResidentSharedExpertBatchServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
            )
            shared_expert_server_session.start()
        if persistent_rope_split_server:
            rope_split_server_session = RopeSplitBatchServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
            )
            rope_split_server_session.start()
        if persistent_mla_attention_server:
            mla_attention_server_session = MLAAttentionBatchServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
                mla_key_cache=mla_key_cache,
            )
            mla_attention_server_session.start()
        if persistent_rmsnorm_server:
            rmsnorm_server_session = ResidentBatchRMSNormServerSession(
                runner_path=runner_path,
                echo_runner_output=echo_runner_output,
            )
            rmsnorm_server_session.start()
        for chunk_index, offset in enumerate(range(0, len(prompt), prompt_chunk_tokens)):
            _recheck_live_memory_budget(live_memory_budget)
            chunk_tokens = prompt[offset : offset + prompt_chunk_tokens]
            batch_tokens = len(chunk_tokens)
            chunk_start = start_position + offset
            is_last_chunk = offset + batch_tokens == len(prompt)
            chunk_dir = root / f"chunk_{chunk_index:04d}"
            planned_work_bytes = _planned_chunk_work_bytes(
                batch_tokens=batch_tokens,
                hidden_dim=planned_hidden_dim,
                layer_count=len(selected_layers),
            )
            _require_disk_budget(
                output_dir=chunk_dir,
                required_bytes=planned_work_bytes,
                safety_margin_bytes=stage_disk_safety_margin_bytes,
                label="prompt prefill work files",
            )
            if is_last_chunk:
                _require_disk_budget(
                    output_dir=output_last.parent,
                    required_bytes=planned_hidden_dim * 4,
                    safety_margin_bytes=stage_disk_safety_margin_bytes,
                    label="prompt prefill last hidden file",
                )
                if output_final_chunk is not None:
                    _require_disk_budget(
                        output_dir=output_final_chunk.parent,
                        required_bytes=batch_tokens * planned_hidden_dim * 4,
                        safety_margin_bytes=stage_disk_safety_margin_bytes,
                        label="prompt prefill final chunk file",
                    )
            chunk_dir.mkdir(parents=True, exist_ok=True)
            embedding_path = chunk_dir / "embedding.f32"
            embedding = embed_tokens_batch(
                resident_layout_path,
                token_ids=chunk_tokens,
                output_f32_path=embedding_path,
                max_row_bytes=max_embedding_row_bytes,
                max_output_bytes=max_prompt_batch_bytes,
                expected_vocab_size=expected_vocab_size,
                expected_hidden_size=expected_hidden_size,
            )
            if hidden_dim == 0:
                hidden_dim = embedding.hidden_dim
            elif hidden_dim != embedding.hidden_dim:
                raise PromptPrefillError(
                    f"embedding hidden dim changed from {hidden_dim} to "
                    f"{embedding.hidden_dim}"
                )
            current_input = embedding.output_path
            layer_records: list[PromptPrefillLayerRecord] = []
            chunk_peak = embedding.output_bytes
            previous_dsa_indices: Path | None = None
            for layer_index, layer in enumerate(selected_layers):
                _recheck_live_memory_budget(live_memory_budget)
                layer_dir = chunk_dir / f"layer_{layer:04d}"
                attention_output = layer_dir / "attention_hidden.f32"
                chunk_context_length = chunk_start + batch_tokens
                dsa_visible_context_covered = (
                    dsa_index_topk is not None
                    and chunk_context_length <= int(dsa_index_topk)
                )
                dsa_mode = "none"
                if dsa_types:
                    scheduled_mode = dsa_types[layer]
                    if scheduled_mode in {"full", "shared"}:
                        dsa_mode = scheduled_mode
                    if dsa_mode == "shared" and previous_dsa_indices is None:
                        if not dsa_visible_context_covered:
                            raise PromptPrefillError(
                                f"layer {layer} uses shared DSA indexer without a previous full layer"
                            )
                attention = run_prefill_attention_block_batch(
                    runner_path=runner_path,
                    resident_layout_path=resident_layout_path,
                    cache_layout_path=cache_layout_path,
                    cache_file_path=cache_file_path,
                    layer=layer,
                    input_f32_path=current_input,
                    output_dir=layer_dir / "attention",
                    output_f32_path=attention_output,
                    start_position=chunk_start,
                    batch_tokens=batch_tokens,
                    context_length=chunk_context_length,
                    num_heads=num_heads,
                    qk_nope_dim=qk_nope_dim,
                    rope_dim=rope_dim,
                    v_head_dim=v_head_dim,
                    mla_kv_b_cache_dir=mla_kv_b_cache_dir,
                    mla_key_cache=mla_key_cache,
                    kv_lora_dim=kv_lora_dim,
                    cache_position_offset=cache_position_offset,
                    attention_scale=attention_scale,
                    rope_theta=rope_theta,
                    rope_interleave=rope_interleave,
                    dsa_indexer_mode=dsa_mode,
                    dsa_prev_indices_u32_path=previous_dsa_indices,
                    dsa_index_topk=dsa_index_topk,
                    dsa_index_n_heads=dsa_index_n_heads,
                    dsa_qk_rope_dim=dsa_qk_rope_dim or rope_dim,
                    dsa_rope_interleave=dsa_rope_interleave,
                    dsa_layer_norm_eps=dsa_layer_norm_eps,
                    write_dsa_future_cache=write_dsa_future_cache,
                    rms_norm_eps=rms_norm_eps,
                    max_cache_file_mib=max_cache_file_mib,
                    max_cache_write_mib=max_cache_write_mib,
                    max_cache_read_mib=max_cache_read_mib,
                    max_resident_matrix_mib=max_resident_matrix_mib,
                    max_runner_scratch_mib=max_runner_scratch_mib,
                    prefill_linear_backend=prefill_linear_backend,
                    prefill_mpsgraph_min_batch_tokens=(
                        prefill_mpsgraph_min_batch_tokens
                    ),
                    prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
                    echo_runner_output=echo_runner_output,
                    resident_linear_server_session=resident_linear_server_session,
                    attention_projection_server_session=(
                        attention_projection_server_session
                    ),
                    attention_output_server_session=attention_output_server_session,
                    rope_split_server_session=rope_split_server_session,
                    mla_attention_server_session=mla_attention_server_session,
                    rmsnorm_server_session=rmsnorm_server_session,
                )
                if attention.dsa_indices_u32_path is not None:
                    previous_dsa_indices = attention.dsa_indices_u32_path
                is_last_layer = layer_index + 1 == len(selected_layers)
                layer_output = (
                    output_final_chunk
                    if output_final_chunk is not None and is_last_chunk and is_last_layer
                    else layer_dir / "layer_output.f32"
                )
                if layer in dense_set:
                    dense_result = run_prefill_dense_mlp_block_batch(
                        runner_path=runner_path,
                        resident_layout_path=resident_layout_path,
                        layer=layer,
                        input_f32_path=attention.output_path,
                        output_dir=layer_dir / "dense_mlp",
                        output_f32_path=layer_output,
                        batch_tokens=batch_tokens,
                        rms_norm_eps=rms_norm_eps,
                        max_resident_matrix_mib=max_resident_matrix_mib,
                        max_runner_scratch_mib=max_runner_scratch_mib,
                        prefill_linear_backend=prefill_linear_backend,
                        prefill_mpsgraph_min_batch_tokens=(
                            prefill_mpsgraph_min_batch_tokens
                        ),
                        prefill_mpsgraph_min_matrix_dim=(
                            prefill_mpsgraph_min_matrix_dim
                        ),
                        echo_runner_output=echo_runner_output,
                        resident_linear_server_session=(
                            resident_linear_server_session
                        ),
                        rmsnorm_server_session=rmsnorm_server_session,
                    )
                    staged_result = None
                    layer_peak = max(
                        attention.estimated_peak_bytes,
                        dense_result.estimated_peak_bytes,
                    )
                    kind = "dense"
                    current_output = dense_result.output_path
                else:
                    layer_static_capacity = _static_capacity_for_batch(
                        static_capacity_request,
                        batch_tokens=batch_tokens,
                    )
                    stage_read_seconds_budget = 0.0
                    if prefill_max_routed_read_seconds > 0:
                        elapsed_read_seconds = (
                            total_expert_stage_planned_read_seconds or 0.0
                        )
                        stage_read_seconds_budget = (
                            prefill_max_routed_read_seconds - elapsed_read_seconds
                        )
                        if stage_read_seconds_budget <= 0:
                            raise PromptPrefillError(
                                "prefill routed expert cumulative planned read "
                                f"time {elapsed_read_seconds:.6g}s exceeds cap "
                                f"{prefill_max_routed_read_seconds:.6g}s before "
                                f"layer {layer} chunk {chunk_index}; "
                                "no read-time budget remains for the next stage"
                            )
                    staged_result = run_prefill_staged_routed_mlp_block_batch(
                        runner_path=runner_path,
                        expert_layout_path=expert_layout_path,
                        resident_layout_path=resident_layout_path,
                        layer=layer,
                        input_f32_path=attention.output_path,
                        output_dir=layer_dir / "staged_mlp",
                        output_f32_path=layer_output,
                        batch_tokens=batch_tokens,
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
                        max_resident_matrix_mib=max_resident_matrix_mib,
                        max_slot_mib=max_slot_mib,
                        max_router_mib=max_router_mib,
                        max_runner_scratch_mib=max_runner_scratch_mib,
                        expert_stage_merge_gap_kib=expert_stage_merge_gap_kib,
                        expert_stage_align_kib=expert_stage_align_kib,
                        max_stage_mib=max_stage_mib,
                        max_compact_stage_mib=max_compact_stage_mib,
                        copy_chunk_mib=copy_chunk_mib,
                        stage_disk_safety_margin_bytes=stage_disk_safety_margin_bytes,
                        prefill_ssd_read_gib_per_second=(
                            prefill_ssd_read_gib_per_second
                        ),
                        prefill_max_routed_read_seconds=stage_read_seconds_budget,
                        expert_stage_max_raw_ranges=expert_stage_max_raw_ranges,
                        expert_stage_max_coalesced_ranges=(
                            expert_stage_max_coalesced_ranges
                        ),
                        expert_stage_tiling=expert_stage_tiling,
                        shared_expert_server_session=shared_expert_server_session,
                        moe_token_block=moe_token_block_value,
                        static_capacity_per_expert=layer_static_capacity,
                        write_static_capacity_json=write_static_capacity_json,
                        allow_static_capacity_overflow=allow_static_capacity_overflow,
                        prefill_linear_backend=prefill_linear_backend,
                        prefill_mpsgraph_min_batch_tokens=(
                            prefill_mpsgraph_min_batch_tokens
                        ),
                        prefill_mpsgraph_min_matrix_dim=(
                            prefill_mpsgraph_min_matrix_dim
                        ),
                        router_hybrid_margin_threshold=(
                            router_hybrid_margin_threshold
                        ),
                        keep_token_files=keep_work_dir,
                        echo_runner_output=echo_runner_output,
                        moe_plan_server_session=moe_plan_server_session,
                        resident_linear_server_session=(
                            resident_linear_server_session
                        ),
                        rmsnorm_server_session=rmsnorm_server_session,
                        moe_output_accumulator=moe_output_accumulator,
                    )
                    dense_result = None
                    routed_moe_runner_command_count += staged_result.routed_command_count
                    total_staged += staged_result.staged_bytes
                    total_compact_stage += staged_result.compact_stage_bytes
                    total_compact_stage_materialized += (
                        staged_result.compact_stage_materialized_bytes
                    )
                    max_staged = max(max_staged, staged_result.staged_bytes)
                    max_compact_stage = max(
                        max_compact_stage,
                        staged_result.compact_stage_bytes,
                    )
                    max_compact_stage_materialized = max(
                        max_compact_stage_materialized,
                        staged_result.compact_stage_materialized_bytes,
                    )
                    total_stage_plus_compact += (
                        staged_result.stage_plus_compact_bytes
                    )
                    total_stage_plus_compact_materialized += (
                        staged_result.stage_plus_compact_materialized_bytes
                    )
                    max_stage_plus_compact = max(
                        max_stage_plus_compact,
                        staged_result.stage_plus_compact_bytes,
                    )
                    max_stage_plus_compact_materialized = max(
                        max_stage_plus_compact_materialized,
                        staged_result.stage_plus_compact_materialized_bytes,
                    )
                    for tile_index, tile_stage in enumerate(
                        _stage_results_for_staged_mlp(staged_result)
                    ):
                        stage_summary = tile_stage.io_summary
                        expert_stage_io_hotspot_records.append(
                            _prefill_stage_io_hotspot_record(
                                chunk_index=chunk_index,
                                tile_index=tile_index,
                                stage_result=tile_stage,
                            )
                        )
                        total_expert_stage_serial_read += (
                            stage_summary.serial_read_bytes
                        )
                        total_expert_stage_unique_requested += (
                            stage_summary.unique_requested_bytes
                        )
                        total_expert_stage_planned_read += (
                            stage_summary.planned_read_bytes
                        )
                        if (
                            total_expert_stage_planned_read_seconds is not None
                            and stage_summary.planned_read_seconds is not None
                        ):
                            total_expert_stage_planned_read_seconds += (
                                stage_summary.planned_read_seconds
                            )
                        if stage_summary.copy_elapsed_seconds is not None:
                            total_expert_stage_copy_elapsed_seconds = (
                                total_expert_stage_copy_elapsed_seconds or 0.0
                            ) + stage_summary.copy_elapsed_seconds
                            total_expert_stage_copy_bytes += tile_stage.staged_bytes
                            total_expert_stage_copy_read_calls += (
                                stage_summary.copy_read_calls
                            )
                            total_expert_stage_copy_write_calls += (
                                stage_summary.copy_write_calls
                            )
                            for chunk_mib, calls in (
                                stage_summary.copy_read_call_counterfactuals_by_chunk_mib
                                or {}
                            ).items():
                                total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib[
                                    str(chunk_mib)
                                ] = (
                                    total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib.get(
                                        str(chunk_mib),
                                        0,
                                    )
                                    + int(calls)
                                )
                        total_expert_stage_waste += stage_summary.waste_bytes
                        total_expert_stage_coalesced_savings += (
                            stage_summary.coalesced_savings_bytes
                        )
                        read_advice = tile_stage.read_advice
                        total_expert_stage_raw_ranges += stage_summary.raw_range_count
                        total_expert_stage_coalesced_ranges += (
                            stage_summary.coalesced_range_count
                        )
                        max_expert_stage_raw_ranges = max(
                            max_expert_stage_raw_ranges,
                            stage_summary.raw_range_count,
                        )
                        max_expert_stage_coalesced_ranges = max(
                            max_expert_stage_coalesced_ranges,
                            stage_summary.coalesced_range_count,
                        )
                        if total_expert_stage_raw_ranges_ok is not None:
                            total_expert_stage_raw_ranges_ok = (
                                total_expert_stage_raw_ranges_ok
                                and stage_summary.raw_range_count_ok is True
                            )
                        if total_expert_stage_coalesced_ranges_ok is not None:
                            total_expert_stage_coalesced_ranges_ok = (
                                total_expert_stage_coalesced_ranges_ok
                                and stage_summary.coalesced_range_count_ok is True
                            )
                        total_expert_stage_read_advice_attempted_ranges += (
                            read_advice.attempted_ranges
                        )
                        total_expert_stage_read_advice_calls += read_advice.calls
                        total_expert_stage_read_advice_bytes += (
                            read_advice.advised_bytes
                        )
                        if read_advice.error is not None:
                            total_expert_stage_read_advice_failures += 1
                        max_expert_stage_unique_read_amplification = max(
                            max_expert_stage_unique_read_amplification,
                            stage_summary.unique_read_amplification,
                        )
                        max_expert_stage_stage_budget_utilization = max(
                            max_expert_stage_stage_budget_utilization,
                            stage_summary.stage_budget_utilization,
                        )
                        total_routed_assignments += (
                            tile_stage.batch_plan.total_assignments
                        )
                        unique_experts = len(tile_stage.selected_experts)
                        total_routed_unique_slots += unique_experts
                        max_routed_unique_per_call = max(
                            max_routed_unique_per_call,
                            unique_experts,
                        )
                        for assignment in tile_stage.batch_plan.expert_tokens:
                            max_routed_tokens_per_expert = max(
                                max_routed_tokens_per_expert,
                                len(assignment.tokens),
                            )
                    if staged_result.static_capacity_per_expert is not None:
                        max_static_capacity_per_expert = max(
                            max_static_capacity_per_expert,
                            staged_result.static_capacity_per_expert,
                        )
                    total_static_capacity_used_slots += (
                        staged_result.static_capacity_used_slots
                    )
                    total_static_capacity_slots += staged_result.static_capacity_total_slots
                    total_static_capacity_overflow += (
                        staged_result.static_capacity_overflow_assignments
                    )
                    total_static_capacity_binary_bytes += (
                        staged_result.static_capacity_binary_bytes
                    )
                    for staged_moe in _moe_results_for_staged_mlp(staged_result):
                        if staged_moe.command_count == 0:
                            moe_plan_server_plan_count += 1
                            moe_plan_server_job_count += 1
                        if staged_moe.moe_token_block_mode:
                            moe_token_block_mode_counts[
                                staged_moe.moe_token_block_mode
                            ] = (
                                moe_token_block_mode_counts.get(
                                    staged_moe.moe_token_block_mode,
                                    0,
                                )
                                + 1
                            )
                        if staged_moe.effective_moe_token_block is not None:
                            max_effective_moe_token_block = max(
                                max_effective_moe_token_block,
                                staged_moe.effective_moe_token_block,
                            )
                        if staged_moe.moe_max_expert_tokens is not None:
                            max_moe_max_expert_tokens = max(
                                max_moe_max_expert_tokens,
                                staged_moe.moe_max_expert_tokens,
                            )
                        if staged_moe.moe_batch_buffer_bytes is not None:
                            max_moe_batch_buffer_bytes = max(
                                max_moe_batch_buffer_bytes,
                                staged_moe.moe_batch_buffer_bytes,
                            )
                        if staged_moe.moe_estimated_peak_bytes is not None:
                            max_moe_estimated_peak_bytes = max(
                                max_moe_estimated_peak_bytes,
                                staged_moe.moe_estimated_peak_bytes,
                            )
                    layer_peak = max(
                        attention.estimated_peak_bytes,
                        staged_result.estimated_peak_bytes,
                    )
                    kind = "moe"
                    current_output = staged_result.output_path
                _count_attention_backends(linear_backend_counts, attention)
                _count_dense_backends(linear_backend_counts, dense_result)
                _count_staged_backends(linear_backend_counts, staged_result)
                _accumulate_attention_flops(linear_backend_flops, attention)
                _accumulate_dense_flops(linear_backend_flops, dense_result)
                _accumulate_staged_flops(linear_backend_flops, staged_result)
                _accumulate_attention_mpp_candidates(
                    mpp_candidate_totals,
                    attention,
                    backend_counts=mpp_candidate_backend_counts,
                    backend_flops=mpp_candidate_backend_flops,
                )
                _accumulate_dense_mpp_candidates(
                    mpp_candidate_totals,
                    dense_result,
                    backend_counts=mpp_candidate_backend_counts,
                    backend_flops=mpp_candidate_backend_flops,
                )
                _accumulate_staged_mpp_candidates(
                    mpp_candidate_totals,
                    staged_result,
                    backend_counts=mpp_candidate_backend_counts,
                    backend_flops=mpp_candidate_backend_flops,
                )
                streamed_accounting: dict[str, int] | None = None
                if staged_result is not None:
                    streamed_accounting = (
                        _streamed_routed_expert_linear_accounting_for_result(
                            staged_result,
                            moe_hidden_dim=routed_moe_hidden_dims_by_layer.get(
                                staged_result.layer,
                                0,
                            ),
                        )
                    )
                    _accumulate_streamed_routed_expert_accounting(
                        counts=linear_backend_counts,
                        flops_by_backend=linear_backend_flops,
                        mpp_candidate_totals=mpp_candidate_totals,
                        mpp_candidate_backend_counts=mpp_candidate_backend_counts,
                        mpp_candidate_backend_flops=mpp_candidate_backend_flops,
                        streamed_totals=streamed_routed_expert_totals,
                        accounting=streamed_accounting,
                    )
                _accumulate_attention_elapsed(linear_backend_elapsed_seconds, attention)
                _accumulate_dense_elapsed(linear_backend_elapsed_seconds, dense_result)
                _accumulate_staged_elapsed(linear_backend_elapsed_seconds, staged_result)
                _accumulate_attention_component_stats(linear_component_stats, attention)
                _accumulate_dense_component_stats(linear_component_stats, dense_result)
                _accumulate_staged_component_stats(linear_component_stats, staged_result)
                if staged_result is not None and streamed_accounting is not None:
                    routed_elapsed = staged_result.routed_moe_elapsed_seconds
                    if routed_elapsed is not None and routed_elapsed > 0:
                        linear_backend_elapsed_seconds["custom-metal"] = (
                            linear_backend_elapsed_seconds.get("custom-metal", 0.0)
                            + routed_elapsed
                        )
                    else:
                        routed_elapsed = None
                    _accumulate_streamed_component_stats(
                        linear_component_stats,
                        component="moe.routed_experts_streamed",
                        accounting=streamed_accounting,
                        elapsed_seconds=routed_elapsed,
                    )
                _accumulate_attention_scratch(linear_scratch_totals, attention)
                _accumulate_dense_scratch(linear_scratch_totals, dense_result)
                _accumulate_staged_scratch(linear_scratch_totals, staged_result)
                chunk_peak = max(chunk_peak, layer_peak)
                layer_records.append(
                    PromptPrefillLayerRecord(
                        layer=layer,
                        kind=kind,
                        input_path=current_input,
                        attention_output_path=attention.output_path,
                        output_path=current_output,
                        output_dir=layer_dir,
                        dsa_indexer_mode=dsa_mode,
                        dsa_rope_interleave=attention.dsa_rope_interleave,
                        dsa_indices_u32_path=attention.dsa_indices_u32_path,
                        estimated_peak_bytes=layer_peak,
                        attention=attention,
                        dense_mlp=dense_result,
                        staged_mlp=staged_result,
                    )
                )
                current_input = current_output

            last_hidden_path: Path | None = None
            if is_last_chunk:
                _copy_last_f32_row(
                    input_path=current_input,
                    output_path=output_last,
                    batch_tokens=batch_tokens,
                    hidden_dim=hidden_dim,
                )
                last_hidden_path = output_last
            total_embedding_read += embedding.read_bytes
            total_embedding_output += embedding.output_bytes
            estimated_peak = max(estimated_peak, chunk_peak)
            chunk_records.append(
                PromptPrefillChunkRecord(
                    chunk_index=chunk_index,
                    start_position=chunk_start,
                    batch_tokens=batch_tokens,
                    token_ids=chunk_tokens,
                    embedding_output_path=embedding.output_path,
                    output_path=current_input,
                    last_hidden_path=last_hidden_path,
                    embedding_read_bytes=embedding.read_bytes,
                    embedding_output_bytes=embedding.output_bytes,
                    estimated_peak_bytes=chunk_peak,
                    layers=tuple(layer_records),
                )
            )
            if not keep_work_dir and _should_cleanup_chunk_dir(
                chunk_dir=chunk_dir,
                output_last=output_last,
                output_final_chunk=output_final_chunk,
            ):
                shutil.rmtree(chunk_dir, ignore_errors=True)
    except _PromptPrefillLiveMemoryError:
        if not keep_work_dir:
            _cleanup_prompt_prefill_chunk_dirs(
                root=root,
                output_last=output_last,
                output_final_chunk=output_final_chunk,
            )
        raise
    except (EmbeddingError, PrefillExecuteError, OSError) as exc:
        if keep_work_dir or not created_work_dir:
            keep_work_dir = True
        raise PromptPrefillError(str(exc)) from exc
    except Exception:
        if keep_work_dir or not created_work_dir:
            keep_work_dir = True
        raise
    finally:
        if moe_plan_server_session is not None:
            moe_plan_server_session.close()
        if resident_linear_server_session is not None:
            resident_linear_server_session.close()
        if attention_projection_server_session is not None:
            attention_projection_server_session.close()
        if attention_output_server_session is not None:
            attention_output_server_session.close()
        if shared_expert_server_session is not None:
            shared_expert_server_session.close()
        if rope_split_server_session is not None:
            rope_split_server_session.close()
        if mla_attention_server_session is not None:
            mla_attention_server_session.close()
        if rmsnorm_server_session is not None:
            rmsnorm_server_session.close()
        if created_work_dir and not keep_work_dir:
            shutil.rmtree(root, ignore_errors=True)

    sorted_linear_backend_counts = dict(sorted(linear_backend_counts.items()))
    sorted_linear_backend_flops = dict(sorted(linear_backend_flops.items()))
    sorted_linear_backend_elapsed_seconds = dict(
        sorted(linear_backend_elapsed_seconds.items())
    )
    sorted_linear_component_stats = _sorted_linear_component_stats(
        linear_component_stats
    )
    linear_backend_estimated_tflops = _linear_backend_estimated_tflops(
        flops_by_backend=sorted_linear_backend_flops,
        elapsed_by_backend=sorted_linear_backend_elapsed_seconds,
    )
    router_gate_totals = _router_gate_totals_from_component_stats(
        sorted_linear_component_stats
    )
    prefill_acceleration_coverage = _prefill_acceleration_coverage_from_counts(
        sorted_linear_backend_counts,
        flops_by_backend=sorted_linear_backend_flops,
        router_gate_matrix_count=router_gate_totals["router_gate_matrix_count"],
        router_gate_estimated_flops=router_gate_totals["router_gate_estimated_flops"],
        router_gate_accelerated_matrix_count=router_gate_totals[
            "router_gate_accelerated_matrix_count"
        ],
        router_gate_accelerated_estimated_flops=router_gate_totals[
            "router_gate_accelerated_estimated_flops"
        ],
        mpp_tensor_ops_candidate_matrix_count=mpp_candidate_totals.get("count", 0),
        mpp_tensor_ops_candidate_estimated_flops=mpp_candidate_totals.get(
            "estimated_flops",
            0,
        ),
        mpp_tensor_ops_candidate_backend_counts=mpp_candidate_backend_counts,
        mpp_tensor_ops_candidate_backend_flops=mpp_candidate_backend_flops,
        streamed_routed_expert_layer_count=streamed_routed_expert_totals.get(
            "layer_count",
            0,
        ),
        streamed_routed_expert_matrix_count=streamed_routed_expert_totals.get(
            "matrix_count",
            0,
        ),
        streamed_routed_expert_assignments=streamed_routed_expert_totals.get(
            "assignments",
            0,
        ),
        streamed_routed_expert_estimated_flops=streamed_routed_expert_totals.get(
            "estimated_flops",
            0,
        ),
        streamed_routed_expert_mpp_candidate_matrix_count=(
            streamed_routed_expert_totals.get("mpp_candidate_matrix_count", 0)
        ),
        streamed_routed_expert_mpp_candidate_estimated_flops=(
            streamed_routed_expert_totals.get("mpp_candidate_estimated_flops", 0)
        ),
        required=prefill_min_accelerated_flop_fraction > 0.0,
        min_accelerated_flop_fraction=prefill_min_accelerated_flop_fraction,
    )
    prefill_acceleration_frontier = _prefill_acceleration_frontier_from_layout(
        resident_layout_path=Path(resident_layout_path),
        prompt_token_count=len(prompt),
        prompt_chunk_tokens=prompt_chunk_tokens,
        actual_coverage=prefill_acceleration_coverage,
        actual_total_matrix_scratch_bytes=linear_scratch_totals.get(
            "total_scratch",
            0,
        ),
        actual_total_matrix_raw_conversion_bytes=linear_scratch_totals.get(
            "total_raw_conversion",
            0,
        ),
        prefill_linear_backend=prefill_linear_backend,
        mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        streamed_routed_expert_hidden_dim=hidden_dim,
        streamed_routed_expert_top_k=top_k,
        streamed_routed_expert_moe_hidden_dims=tuple(
            routed_moe_hidden_dims_by_layer.get(layer, 0)
            for layer in selected_layers
            if layer not in dense_set
        ),
        min_accelerated_flop_fraction=prefill_min_accelerated_flop_fraction,
    )

    return PromptPrefillResult(
        runner_path=Path(runner_path),
        expert_layout_path=Path(expert_layout_path),
        resident_layout_path=Path(resident_layout_path),
        cache_layout_path=Path(cache_layout_path),
        cache_file_path=Path(cache_file_path),
        output_last_hidden_path=output_last,
        output_final_chunk_path=output_final_chunk,
        work_dir=root,
        kept_work_dir=keep_work_dir or not created_work_dir,
        elapsed_seconds=time.perf_counter() - prefill_started,
        prompt_token_ids=prompt,
        start_position=start_position,
        chunk_tokens=prompt_chunk_tokens,
        chunk_count=len(chunk_records),
        layers=selected_layers,
        dense_layers=selected_dense,
        hidden_dim=hidden_dim,
        total_embedding_read_bytes=total_embedding_read,
        total_embedding_output_bytes=total_embedding_output,
        total_staged_bytes=total_staged,
        total_compact_stage_bytes=total_compact_stage,
        total_compact_stage_materialized_bytes=total_compact_stage_materialized,
        max_staged_bytes=max_staged,
        max_compact_stage_bytes=max_compact_stage,
        max_compact_stage_materialized_bytes=max_compact_stage_materialized,
        total_stage_plus_compact_bytes=total_stage_plus_compact,
        total_stage_plus_compact_materialized_bytes=(
            total_stage_plus_compact_materialized
        ),
        max_stage_plus_compact_bytes=max_stage_plus_compact,
        max_stage_plus_compact_materialized_bytes=(
            max_stage_plus_compact_materialized
        ),
        total_expert_stage_serial_read_bytes=total_expert_stage_serial_read,
        total_expert_stage_unique_requested_bytes=total_expert_stage_unique_requested,
        total_expert_stage_planned_read_bytes=total_expert_stage_planned_read,
        total_expert_stage_waste_bytes=total_expert_stage_waste,
        total_expert_stage_coalesced_savings_bytes=(
            total_expert_stage_coalesced_savings
        ),
        total_expert_stage_planned_read_seconds=(
            total_expert_stage_planned_read_seconds
        ),
        prefill_ssd_read_gib_per_second=prefill_ssd_read_gib_per_second,
        prefill_max_routed_read_seconds=prefill_max_routed_read_seconds,
        total_expert_stage_read_seconds_ok=(
            None
            if (
                prefill_max_routed_read_seconds <= 0
                or total_expert_stage_planned_read_seconds is None
            )
            else total_expert_stage_planned_read_seconds
            <= prefill_max_routed_read_seconds
        ),
        total_expert_stage_copy_seconds_ok=(
            None
            if (
                prefill_max_routed_read_seconds <= 0
                or total_expert_stage_copy_elapsed_seconds is None
            )
            else total_expert_stage_copy_elapsed_seconds
            <= prefill_max_routed_read_seconds
        ),
        prefill_max_stage_raw_ranges=expert_stage_max_raw_ranges,
        prefill_max_stage_coalesced_ranges=expert_stage_max_coalesced_ranges,
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
            total_expert_stage_planned_read / total_expert_stage_serial_read
            if total_expert_stage_serial_read
            else 0.0
        ),
        total_expert_stage_unique_read_amplification=(
            total_expert_stage_planned_read / total_expert_stage_unique_requested
            if total_expert_stage_unique_requested
            else 0.0
        ),
        max_expert_stage_unique_read_amplification=(
            max_expert_stage_unique_read_amplification
        ),
        max_expert_stage_stage_budget_utilization=(
            max_expert_stage_stage_budget_utilization
        ),
        total_routed_expert_assignments=total_routed_assignments,
        total_routed_unique_expert_slots=total_routed_unique_slots,
        max_routed_unique_experts_per_call=max_routed_unique_per_call,
        max_routed_tokens_per_expert=max_routed_tokens_per_expert,
        moe_token_block=moe_token_block_value,
        moe_token_block_mode_counts=dict(sorted(moe_token_block_mode_counts.items())),
        max_effective_moe_token_block=max_effective_moe_token_block,
        max_moe_max_expert_tokens=max_moe_max_expert_tokens,
        max_moe_batch_buffer_bytes=max_moe_batch_buffer_bytes,
        max_moe_estimated_peak_bytes=max_moe_estimated_peak_bytes,
        persistent_moe_plan_server=persistent_moe_plan_server,
        persistent_resident_linear_server=persistent_resident_linear_server,
        persistent_attention_projection_server=persistent_attention_projection_server,
        persistent_attention_output_server=persistent_attention_output_server,
        persistent_shared_expert_server=persistent_shared_expert_server,
        persistent_rope_split_server=persistent_rope_split_server,
        persistent_mla_attention_server=persistent_mla_attention_server,
        persistent_rmsnorm_server=persistent_rmsnorm_server,
        moe_output_accumulator=moe_output_accumulator,
        moe_plan_server_plan_count=moe_plan_server_plan_count,
        moe_plan_server_job_count=moe_plan_server_job_count,
        routed_moe_runner_command_count=routed_moe_runner_command_count,
        static_capacity_per_expert=static_capacity_request,
        max_static_capacity_per_expert=max_static_capacity_per_expert,
        total_static_capacity_used_slots=total_static_capacity_used_slots,
        total_static_capacity_slots=total_static_capacity_slots,
        total_static_capacity_overflow_assignments=total_static_capacity_overflow,
        total_static_capacity_binary_bytes=total_static_capacity_binary_bytes,
        estimated_peak_bytes=estimated_peak,
        live_memory_budget=live_memory_budget,
        linear_backend_counts=sorted_linear_backend_counts,
        linear_backend_flops=sorted_linear_backend_flops,
        total_linear_matrix_scratch_bytes=linear_scratch_totals.get("total_scratch", 0),
        max_linear_matrix_scratch_bytes=linear_scratch_totals.get("max_scratch", 0),
        total_linear_matrix_f32_bytes=linear_scratch_totals.get("total_f32", 0),
        total_linear_matrix_raw_conversion_bytes=linear_scratch_totals.get(
            "total_raw_conversion",
            0,
        ),
        total_linear_estimated_flops=prefill_acceleration_coverage[
            "total_estimated_flops"
        ],
        accelerated_linear_estimated_flops=prefill_acceleration_coverage[
            "accelerated_estimated_flops"
        ],
        custom_linear_estimated_flops=prefill_acceleration_coverage[
            "custom_metal_estimated_flops"
        ],
        unsupported_linear_estimated_flops=prefill_acceleration_coverage[
            "unsupported_mpsgraph_estimated_flops"
        ],
        accelerated_linear_flop_fraction=prefill_acceleration_coverage[
            "accelerated_flop_fraction"
        ],
        prefill_acceleration_coverage=prefill_acceleration_coverage,
        prefill_acceleration_frontier=prefill_acceleration_frontier,
        chunks=tuple(chunk_records),
        total_expert_stage_copy_elapsed_seconds=(
            total_expert_stage_copy_elapsed_seconds
        ),
        total_expert_stage_copy_throughput_gib_per_second=(
            (total_expert_stage_copy_bytes / 1024**3)
            / total_expert_stage_copy_elapsed_seconds
            if (
                total_expert_stage_copy_elapsed_seconds is not None
                and total_expert_stage_copy_elapsed_seconds > 0
            )
            else None
        ),
        total_expert_stage_copy_read_calls=total_expert_stage_copy_read_calls,
        total_expert_stage_copy_write_calls=total_expert_stage_copy_write_calls,
        total_expert_stage_copy_average_read_bytes=(
            total_expert_stage_copy_bytes / total_expert_stage_copy_read_calls
            if total_expert_stage_copy_read_calls
            else None
        ),
        total_expert_stage_copy_average_write_bytes=(
            total_expert_stage_copy_bytes / total_expert_stage_copy_write_calls
            if total_expert_stage_copy_write_calls
            else None
        ),
        total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib=(
            dict(
                sorted(
                    total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib.items(),
                    key=lambda item: int(item[0]),
                )
            )
            if total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib
            else None
        ),
        expert_stage_io_stage_count=len(expert_stage_io_hotspot_records),
        expert_stage_copy_hotspots=_top_prefill_stage_io_hotspots(
            expert_stage_io_hotspot_records,
            metric="copy",
        ),
        expert_stage_range_hotspots=_top_prefill_stage_io_hotspots(
            expert_stage_io_hotspot_records,
            metric="ranges",
        ),
        linear_backend_elapsed_seconds=sorted_linear_backend_elapsed_seconds,
        linear_backend_estimated_tflops=linear_backend_estimated_tflops,
        linear_backend_component_stats=sorted_linear_component_stats,
    )
