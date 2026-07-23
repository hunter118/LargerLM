from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import shlex
import sys
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from .artifact_status import (
    ArtifactStatusError,
    CheckpointArtifactStatus,
    inspect_checkpoint_artifact,
    missing_shard_download_entries,
    missing_shard_download_manifest,
)
from .baseline import BaselineError, BaselineValidation, validate_baseline
from .benchmark import BenchmarkError, GenerationBenchmark, benchmark_prepared_token_ids
from .config import ConfigError, load_config
from .context1_o_proj_cache import (
    Context1OProjCacheBuildResult,
    Context1OProjCacheError,
    Context1OProjCacheLayout,
    build_context1_o_proj_cache,
    load_context1_o_proj_cache_layout,
    load_context1_o_proj_cache_progress,
)
from .decode_cache import (
    DecodeCacheError,
    DecodeCacheInitResult,
    DecodeCacheLayout,
    build_decode_cache_layout,
    init_decode_cache_file,
)
from .decode_driver import (
    DecodeDriverError,
    DecodeLayersResult,
    run_decode_layers,
)
from .disk_benchmark import (
    DiskBenchmarkError,
    MAX_SEQUENTIAL_READ_CHUNK_BYTES,
    SequentialReadBenchmark,
    benchmark_sequential_read,
)
from .dsa_indexer import (
    DSAIndexerBatchResult,
    DSAIndexerError,
    run_dsa_indexer_batch,
)
from .embedding import (
    EmbeddingBatchResult,
    EmbeddingError,
    EmbeddingResult,
    embed_token,
    embed_tokens_batch,
)
from .expert_io import (
    BatchExpertIOPlan,
    BatchExpertIOTilingPlan,
    BatchExpertStageResult,
    ExpertIOPlan,
    ExpertIOPlanError,
    StaticExpertCapacityBinaryValidation,
    StaticExpertCapacityPlan,
    plan_batch_expert_io,
    plan_batch_expert_io_tiles,
    plan_expert_io,
    plan_static_expert_capacity,
    stage_batch_experts,
    validate_static_expert_capacity_binary,
    write_static_expert_capacity_binary,
    write_static_expert_capacity_plan,
)
from .formatting import format_bytes
from .generation_guard import system_memory_snapshot
from .final_logits import (
    FinalLogitsError,
    FinalLogitsResult,
    compute_final_logits,
    compute_final_logits_metal,
)
from .hardware import detect_hardware
from .mlx_baseline import (
    MlxBaselineError,
    MlxBaselinePreflight,
    MlxBaselineResult,
    export_mlx_baseline,
    preflight_mlx_baseline,
)
from .metal_generate import (
    MetalGenerateError,
    MetalTokenGenerationResult,
    generate_metal_token_ids,
)
from .metal_text_generator import (
    MetalTextGenerationBatchResult,
    MetalTextGenerationError,
    MetalTextGenerationResult,
    generate_metal_text_batch,
    generate_metal_text,
)
from .packer import PackReport, PackerError, pack_experts
from .planner import ModelPlan, PlannerError, build_plan
from .prefill_plan import (
    PrefillLinearCalibrationShape,
    PrefillPlan,
    PrefillPlanError,
    build_prefill_plan,
)
from .prefill_backend import (
    PrefillBackendCapability,
    PrefillBackendError,
    evaluate_prefill_acceleration_requirement,
    inspect_prefill_backend,
    prefill_acceleration_runtimes,
    selectable_accelerated_prefill_backends,
    suggested_prefill_acceleration_flags,
)
from .prefill_execute import (
    AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    AUTO_MPSGRAPH_MIN_DIM,
    CALIBRATION_MATRIX_DTYPE_BYTES,
    PREFILL_LINEAR_ACCELERATED_BACKENDS,
    PREFILL_LINEAR_BACKENDS,
    PREFILL_LINEAR_CALIBRATION_BACKENDS,
    PrefillCacheWriteResult,
    PrefillAttentionBlockBatchResult,
    PrefillAttentionProjectionBatchResult,
    PrefillDenseMLPBlockBatchResult,
    PrefillExecuteError,
    PrefillAttentionPrefixResult,
    PrefillAttentionOutputBatchResult,
    PrefillMLAAttentionBatchResult,
    PrefillRoutedMLPBlockBatchResult,
    PrefillRopeBatchResult,
    PrefillStagedRoutedMLPBlockBatchResult,
    ResidentBatchLinearResult,
    ResidentLinearCalibrationResult,
    ResidentBatchRMSNormResult,
    run_prefill_attention_projection_batch,
    run_prefill_attention_block_batch,
    run_prefill_attention_prefix_batch,
    run_prefill_attention_output_batch,
    run_prefill_dense_mlp_block_batch,
    run_prefill_mla_attention_batch,
    run_prefill_routed_mlp_block_batch,
    run_prefill_rope_batch,
    run_prefill_staged_routed_mlp_block_batch,
    run_resident_batch_linear,
    run_resident_linear_calibration,
    run_resident_batch_rmsnorm,
    write_prefill_kv_cache_batch,
)
from .prompt_prefill import (
    PromptPrefillError,
    PromptPrefillResult,
    prompt_prefill_acceleration_failure_reason,
    run_prompt_prefill,
)
from .preflight import GlmPreflightReport, PreflightError, preflight_glm_checkpoint
from .prepare import (
    PrepareError,
    PrepareFlagsProvenance,
    PrepareGlmReport,
    prepare_glm_checkpoint,
)
from .prepared import (
    PreparedManifest,
    PreparedManifestError,
    load_prepared_manifest,
    validate_layout_model_config_sha256,
)
from .prepared_lock import (
    PREPARED_RUN_LOCK_ENV,
    PREPARED_RUN_LOCK_FILE,
    PreparedRunLock,
    PreparedRunLockError,
    acquire_prepared_run_lock_path,
    prepared_run_lock_already_held,
    prepared_run_lock_path_for_manifest,
)
from .result_summary import (
    ResultSummaryError,
    compare_result_files,
    format_result_bakeoff_text,
    format_result_comparison_text,
    format_result_summary_text,
    result_bakeoff_files,
    summarize_result_file,
)
from .resident import ResidentPackReport, ResidentPackerError, pack_resident_weights
from .runtime_check import (
    LayerRuntimeBudget,
    RuntimeCheckError,
    check_layer_runtime,
)
from .routed_read import format_routed_read_guard_flag_float
from .safety import SafetyError
from .safetensors import (
    HEADER_MANIFEST_NAME,
    SafetensorsError,
    categorize_tensor_for_moe_layers,
    scan_checkpoint,
)
from .server import (
    DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
    PreparedGenerationApp,
    PreparedServerConfig,
    PreparedServerError,
    PreparedRequestCheckError,
    combine_suggested_launch_profile,
    prepared_launch_profile_target,
    prepared_glm_4bit_readiness,
    prepared_memory_profile_failure_reason,
    require_prepared_request_check_ok,
    prepared_runtime_profile_failure_reason,
    run_prepared_server,
    suggest_final_logits_flags,
)
from .staged_moe import (
    StagedMoEError,
    StagedRoutedMoEBatchPlanServerResult,
    StagedRoutedMoEBatchPlanResult,
    StagedRoutedMoEBatchResult,
    TiledStagedRoutedMoEBatchResult,
    run_staged_routed_moe_batch_plan,
    run_staged_routed_moe_batch_plan_server,
    run_staged_routed_moe_batch,
    run_tiled_staged_routed_moe_batch,
)
from .text_generator import (
    TextGenerationError,
    TextGenerationResult,
    generate_text,
    read_prompt_arg,
)
from .token_generator import (
    TokenGenerationResult,
    TokenGeneratorError,
    _auto_prefill_prompt_chunk_plan,
    _auto_prefill_prompt_chunk_tokens,
    generate_token_ids,
    prefill_prompt_chunk_plan_drift_summary,
)
from .tokenizer import TokenizerError, encode_prompt, render_chat_prompt


class CliArgumentError(RuntimeError):
    """Raised when CLI arguments fail validation before dispatch."""


_LAUNCH_PROFILE_COMMANDS = frozenset(
    {
        "generate-prepared-token-ids",
        "generate-prepared-text",
        "bench-prepared-token-ids",
        "inspect-prepared",
        "serve-prepared",
    }
)
_REQUIRED_LAUNCH_AUDIT_CHECK_CODES = (
    "locked_launch_profile",
    "prepared_identity_strong",
    "prepared_storage_validated",
    "prepare_expert_pack_heap_envelope_ok",
    "prepare_resident_alias_rewrite_ok",
    "prepared_memory_profile_required",
    "prepared_memory_profile_ok",
    "prepared_context_budget_profile_ok",
    "prepared_runtime_profile_ok",
    "memory_guard_available_ok",
    "memory_guard_free_reserve",
    "glm_4bit_required",
    "glm_4bit_ready",
    "public_glm_5_2_shape_required",
    "public_glm_5_2_shape_ok",
    "prefill_acceleration_required",
    "prefill_acceleration_gate_ok",
    "prefill_acceleration_probe_ok",
    "prefill_acceleration_profile_replays_probe",
    "request_check_requested",
    "request_check_ok",
    "request_runtime_preflight_ran",
    "request_runtime_memory_ok",
    "request_prefill_backend_effective_ok",
    "applied_prefill_actual_read_time_ok",
    "applied_prefill_actual_acceleration_coverage_valid",
    "applied_decode_actual_read_time_ok",
    "request_prefill_routed_read_budget_ok",
    "request_decode_routed_read_budget_ok",
    "request_prefill_prompt_chunk_plan_ok",
    "request_prefill_stage_temp_limit_ok",
    "request_prefill_stage_temp_disk_ok",
    "request_prefill_acceleration_coverage_ok",
    "request_launch_profile_safe",
)
_GLM_4BIT_AUDIT_ENVELOPE_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "moe_layer_count",
    "routed_experts",
    "experts_per_token",
    "model_config_sha256",
    "expert_layout_config_sha256",
    "expert_layout_quantization",
    "expert_layout_group_size",
    "expert_layout_model_layer_count",
    "expert_layout_moe_layer_count",
    "expected_expert_slot_bytes",
    "expected_expert_layer_bytes",
    "expected_total_expert_bytes",
    "prepared_expert_layout_bytes",
    "prepared_expert_layer_file_bytes",
    "expert_layer_file_count",
    "unique_expert_layer_file_count",
    "expert_layer_files_exact_size",
    "expected_decode_token_routed_expert_read_bytes",
    "expected_full_prompt_routed_expert_sweep_bytes",
    "prepared_resident_layout_bytes",
    "resident_layout_total_bytes",
    "resident_weight_file_bytes",
    "resident_weight_file_exact_size",
    "prepared_decode_cache_file_bytes",
    "decode_cache_layout_ok",
    "decode_cache_layout_total_bytes",
    "decode_cache_context_tokens",
    "decode_cache_segment_count",
    "decode_cache_mla_kv_segments_checked",
    "decode_cache_mla_kv_segments_ok",
    "decode_cache_dsa_index_segments_checked",
    "decode_cache_dsa_index_segments_ok",
)
_PREPARE_EXPERT_PACK_HEAP_AUDIT_FIELDS = (
    "prepare_expert_pack_chunk_size_bytes",
    "prepare_expert_pack_estimated_peak_heap_bytes",
    "prepare_expert_pack_max_heap_bytes",
    "prepare_raw_quantization_extra_heap_bytes",
    "prepare_raw_quantization_max_source_block_bytes",
    "prepare_raw_quantization_max_output_block_bytes",
    "prepare_raw_quantization_max_rows_per_block",
)
_PREPARE_RESIDENT_ALIAS_REWRITE_AUDIT_FIELDS = (
    "prepare_resident_component_alias_source_tensor_count",
    "prepare_resident_component_alias_renamed_tensor_count",
    "prepare_resident_component_alias_bytes",
    "prepare_resident_fused_gate_up_source_tensor_count",
    "prepare_resident_fused_gate_up_expanded_tensor_count",
    "prepare_resident_fused_gate_up_expanded_bytes",
)
_PREPARED_RUNTIME_PROFILE_MANIFEST_AUDIT_FIELDS = (
    "prepare_effective_unified_memory_bytes",
    "prepare_effective_unified_memory_source",
    "prepare_system_reserve_bytes",
    "prepared_recommended_max_live_working_set_bytes",
    "prepared_recommended_min_free_unified_memory_bytes",
    "prepared_recommended_required_available_memory_bytes",
)
_PREPARED_RUNTIME_PROFILE_AUDIT_FIELDS = (
    *_PREPARED_RUNTIME_PROFILE_MANIFEST_AUDIT_FIELDS,
    "system_total_memory_bytes",
    "system_available_memory_bytes",
    "system_memory_source",
    "system_total_meets_prepare_effective_unified_memory",
    "system_available_meets_prepare_system_reserve",
    "system_available_meets_prepared_recommended_required_available",
    "profile_ok",
)
_PREPARED_SSD_READ_MANIFEST_AUDIT_FIELDS = (
    "prepare_cold_read_gib_per_second",
    "prepare_cold_read_source",
    "prepare_cold_read_benchmark_path",
    "prepare_cold_read_benchmark_requested_bytes",
    "prepare_cold_read_benchmark_measured_bytes",
    "prepare_cold_read_benchmark_elapsed_seconds",
)
_PREPARED_SSD_READ_AUDIT_FIELDS = (
    "source",
    "prefill_ssd_read_gib_per_second",
    "matches_prepare_cold_read",
    *_PREPARED_SSD_READ_MANIFEST_AUDIT_FIELDS,
)
_PREFILL_ACCELERATION_AUDIT_FIELDS = (
    "reason_code",
    "prefill_backend_configured_backend",
    "prefill_backend_effective_backend",
    "recommended_backend",
    "host_probe_requested",
    "host_probe_path",
    "host_probe_ran",
    "host_probe_ok",
    "prefill_backend_probe_timeout_seconds",
    "mps_graph_runtime_available",
    "mps_graph_probe_requested",
    "mps_graph_probe_ran",
    "mps_graph_probe_ok",
    "metal4_ml_runtime_available",
    "mpp_runtime_available",
    "prefill_acceleration_runtimes",
    "selectable_accelerated_prefill_backends",
    "validated_accelerated_prefill_backends",
    "prefill_acceleration_runtime_gaps",
    "prefill_neural_accelerator_status",
    "selectable_prefill_acceleration_available",
    "validated_prefill_acceleration_available",
)
_PREFILL_NEURAL_ACCELERATOR_AUDIT_FIELDS = (
    "runtime",
    "execution_path",
    "status",
    "ready_for_generation",
    "runtime_visible",
    "selectable",
    "reason",
)
_PREFILL_NEURAL_ACCELERATOR_RUN_PROBE_AUDIT_FIELDS = (
    "mpp_run_probe_kernel_variant",
    "mpp_run_probe_shape",
    "mpp_run_probe_dtype",
    "mpp_run_probe_execution_path",
)
_REQUEST_ROUTED_READ_LAUNCH_CHECK_FIELDS = (
    "analyzed",
    "within_limit",
    "baseline_read_bytes",
    "planned_read_bytes",
    "extra_read_bytes",
    "read_amplification",
    "max_read_amplification",
    "within_amplification_limit",
    "max_planned_read_bytes",
    "within_planned_read_limit",
    "ssd_read_gib_per_second",
    "planned_read_seconds",
    "max_read_seconds",
    "within_seconds_limit",
    "minimum_chunk_tokens_for_limits",
)
_REQUEST_ROUTED_READ_AUDIT_FIELDS = (
    "prompt_chunk_tokens",
    "top_k",
    *_REQUEST_ROUTED_READ_LAUNCH_CHECK_FIELDS,
)
_REQUEST_DECODE_ROUTED_READ_AUDIT_FIELDS = (
    "analyzed",
    "read_bytes_per_token",
    "max_read_bytes_per_token",
    "within_read_limit",
    "ssd_read_gib_per_second",
    "planned_read_seconds_per_token",
    "max_read_seconds_per_token",
    "within_seconds_limit",
    "within_limit",
)
_PREFILL_ROUTED_READ_PROFILE_FLAGS = (
    "--prefill-prompt-chunk-tokens",
    "--prefill-max-routed-read-amplification",
    "--prefill-max-routed-read-gib",
    "--prefill-ssd-read-gib-s",
    "--prefill-max-routed-read-seconds",
)
_DECODE_ROUTED_READ_PROFILE_FLAGS = (
    "--decode-max-routed-read-gib-per-token",
    "--prefill-ssd-read-gib-s",
    "--decode-max-routed-read-seconds-per-token",
)
_REQUEST_PREFILL_STAGE_TEMP_AUDIT_FIELDS = (
    "analyzed",
    "prompt_chunk_tokens",
    "top_k",
    "chunks_per_prompt",
    "layers",
    "stage_align_bytes",
    "static_capacity_per_expert",
    "allow_static_capacity_overflow",
    "max_static_capacity_per_expert",
    "static_capacity_strict_overflow_safe",
    "max_stage_bytes",
    "max_stage_limit_bytes",
    "within_stage_limit",
    "max_compact_stage_bytes",
    "max_compact_stage_limit_bytes",
    "within_compact_stage_limit",
    "max_stage_raw_ranges",
    "max_stage_raw_range_limit",
    "within_stage_raw_range_limit",
    "max_stage_coalesced_ranges",
    "max_stage_coalesced_range_limit",
    "within_stage_coalesced_range_limit",
    "max_stage_plus_compact_bytes",
    "total_stage_plus_compact_bytes",
    "max_static_capacity_binary_bytes",
    "total_static_capacity_binary_bytes",
    "max_stage_plus_compact_plus_static_bytes",
    "total_stage_plus_compact_plus_static_bytes",
    "within_limit",
)
_REQUEST_PREFILL_STAGE_TEMP_DISK_AUDIT_FIELDS = (
    "analyzed",
    "path",
    "required_stage_temp_bytes",
    "disk_safety_margin_bytes",
    "required_free_bytes",
    "within_free_space",
)
_REQUEST_RUNTIME_PREFLIGHT_AUDIT_FIELDS = (
    "ran",
    "available_memory_ok",
    "requested_context_tokens",
    "max_layer_peak_bytes",
    "max_layer_cache_read_bytes",
    "read_bytes_per_token",
    "final_logits_peak_bytes",
    "embedding_row_bytes",
    "embedding_output_bytes",
    "live_working_set_bytes",
    "resident_backing_bytes",
    "nonresident_peak_bytes",
    "extra_live_working_set_bytes",
    "max_live_working_set_bytes",
    "min_available_memory_bytes",
    "required_available_memory_bytes",
    "system_available_memory_bytes",
    "system_total_memory_bytes",
    "system_memory_source",
)
_REQUEST_PREFILL_LIVE_MEMORY_AUDIT_FIELDS = (
    "prompt_batch_bytes",
    "runner_scratch_bytes",
    "cache_read_bytes",
    "cache_write_bytes",
    "stage_copy_bytes",
    "estimated_live_working_set_bytes",
)
_REQUEST_PREFILL_PROMPT_CHUNK_TOKEN_FIELDS = (
    "configured",
    "resolved",
    "max_safe",
)
_REQUEST_PREFILL_PROMPT_CHUNK_PLAN_FIELDS = (
    "source",
    "configured_is_auto",
    "max_safe",
)
_REQUEST_PREFILL_PROMPT_CHUNK_PLAN_SUMMARY_FIELDS = (
    "prompt_tokens",
    "start_position",
    "raw_tokens",
    "chunk_tokens",
    "tile_tokens",
    "limiting_cap_tokens",
    "limiting_caps",
    "caps",
    "hidden_dim",
    "per_token_activation_bytes",
    "max_matrix_scratch_bytes",
    "next_token_matrix_scratch_bytes",
    "usable_disk_bytes",
    "per_token_disk_bytes",
)
_REQUEST_PREFILL_CACHE_IO_AUDIT_FIELDS = (
    "dtype_bytes",
    "mla_cache_width",
    "dsa_index_head_dim",
    "dsa_index_topk",
    "indexed_attention_layers",
    "full_attention_layers",
    "dsa_full_indexer_layers",
    "causal_rows_per_layer",
    "indexed_rows_per_layer",
    "mla_cache_read_bytes",
    "dsa_index_cache_read_bytes",
    "total_cache_read_bytes",
    "mla_cache_write_bytes",
    "dsa_index_cache_write_bytes",
    "total_cache_write_bytes",
)
_REQUEST_PREFILL_ROUTED_CHUNK_FRONTIER_AUDIT_FIELDS = (
    "analyzed",
    "prompt_token_count",
    "resolved_prompt_chunk_tokens",
    "max_safe_prompt_chunk_tokens",
    "top_k",
    "layers",
    "stage_align_bytes",
    "static_capacity_per_expert",
    "allow_static_capacity_overflow",
    "baseline_read_bytes",
    "saturation_chunk_tokens",
    "candidates",
)
_REQUEST_PREFILL_ROUTED_CHUNK_CANDIDATE_AUDIT_FIELDS = (
    "prompt_chunk_tokens",
    "chunks_per_prompt",
    "saturates_all_experts_per_layer",
    "planned_read_bytes",
    "extra_read_bytes",
    "read_amplification",
    "max_layer_planned_read_bytes",
    "max_stage_plus_compact_bytes",
    "max_chunk_stage_plus_compact_bytes",
    "total_stage_plus_compact_bytes",
    "max_static_capacity_binary_bytes",
    "max_chunk_static_capacity_binary_bytes",
    "total_static_capacity_binary_bytes",
    "max_stage_plus_compact_plus_static_bytes",
    "max_chunk_stage_plus_compact_plus_static_bytes",
    "total_stage_plus_compact_plus_static_bytes",
    "planned_read_seconds",
)
_PREFILL_STAGE_TEMP_PROFILE_FLAGS = (
    "--prefill-max-stage-mib",
    "--prefill-max-compact-stage-mib",
    "--prefill-max-stage-raw-ranges",
    "--prefill-max-stage-coalesced-ranges",
)
_PREFILL_ACTUAL_READ_TIME_AUDIT_FIELDS = (
    "source",
    "total_expert_stage_planned_read_bytes",
    "total_expert_stage_planned_read_seconds",
    "prefill_ssd_read_gib_per_second",
    "prefill_max_routed_read_seconds",
    "total_expert_stage_read_seconds_ok",
    "total_expert_stage_copy_seconds_ok",
    "prefill_max_stage_raw_ranges",
    "prefill_max_stage_coalesced_ranges",
    "total_expert_stage_raw_ranges",
    "total_expert_stage_coalesced_ranges",
    "max_expert_stage_raw_ranges",
    "max_expert_stage_coalesced_ranges",
)
_PREFILL_ACTUAL_READ_TIME_OPTIONAL_AUDIT_FIELDS = (
    "total_expert_stage_raw_ranges_ok",
    "total_expert_stage_coalesced_ranges_ok",
    "total_expert_stage_copy_elapsed_seconds",
    "total_expert_stage_copy_throughput_gib_per_second",
)
_PREFILL_ACTUAL_LINEAR_BACKEND_AUDIT_FIELDS = (
    "source",
    "configured_backend",
    "auto_policy",
    "linear_backend_counts",
    "linear_backend_flops",
    "linear_backend_elapsed_seconds",
    "linear_backend_estimated_tflops",
    "total_linear_estimated_flops",
    "accelerated_linear_estimated_flops",
    "custom_linear_estimated_flops",
    "unsupported_linear_estimated_flops",
    "accelerated_linear_flop_fraction",
)
_REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS = (
    "required",
    "analyzed",
    "ok",
    "min_accelerated_flop_fraction",
    "matrix_count",
    "accelerated_matrix_count",
    "mpsgraph_matrix_count",
    "custom_metal_matrix_count",
    "unsupported_mpsgraph_matrix_count",
    "total_estimated_flops",
    "accelerated_estimated_flops",
    "custom_metal_estimated_flops",
    "unsupported_mpsgraph_estimated_flops",
    "other_estimated_flops",
    "mpp_candidate_policy",
    "mpp_tensor_ops_candidate_matrix_count",
    "mpp_tensor_ops_candidate_estimated_flops",
    "mpp_tensor_ops_candidate_flop_fraction",
    "streamed_routed_expert_layer_count",
    "streamed_routed_expert_matrix_count",
    "streamed_routed_expert_assignments",
    "streamed_routed_expert_estimated_flops",
    "streamed_routed_expert_mpp_candidate_matrix_count",
    "streamed_routed_expert_mpp_candidate_estimated_flops",
    "accelerated_flop_fraction",
    "dominant_resident_flops_accelerated",
    "accelerated_backends",
    "any_resident_matrix_accelerated",
    "all_resident_matrices_accelerated",
    "reason",
)
_DECODE_ACTUAL_READ_TIME_AUDIT_FIELDS = (
    "source",
    "decode_step_count",
    "decode_read_bytes_per_token",
    "planned_decode_routed_read_bytes",
    "actual_decode_routed_read_bytes",
    "actual_decode_routed_read_bytes_ok",
    "planned_decode_routed_read_seconds",
    "actual_decode_routed_read_seconds",
    "prefill_ssd_read_gib_per_second",
    "decode_max_routed_read_seconds_per_token",
    "total_decode_max_routed_read_seconds",
    "total_decode_routed_read_seconds_ok",
)
_LAUNCH_PROFILE_VALUELESS_FLAGS = frozenset(
    {
        "--require-prefill-acceleration",
        "--allow-router-gate-only-prefill-acceleration",
        "--allow-non-accelerated-prefill-launch-audit",
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--compile-mpp-probe",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
        "--no-runtime-preflight",
        "--prefill-mla-key-cache",
        "--decode-mla-key-cache",
        "--metal-runtime-cache-mla-kv-b-f32",
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
    }
)
_LAUNCH_PROFILE_NEGATED_VALUELESS_FLAGS = frozenset({"--no-runtime-preflight"})
_LAUNCH_PROFILE_ALLOWED_FLAGS = frozenset(
    {
        "--max-live-working-set-mib",
        "--min-free-unified-memory-gib",
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--prefill-linear-backend",
        "--prefill-mla-key-cache",
        "--require-prefill-acceleration",
        "--allow-router-gate-only-prefill-acceleration",
        "--allow-non-accelerated-prefill-launch-audit",
        "--prefill-prompt-chunk-tokens",
        "--prefill-max-routed-read-amplification",
        "--prefill-max-routed-read-gib",
        "--prefill-ssd-read-gib-s",
        "--prefill-max-routed-read-seconds",
        "--max-cache-read-mib",
        "--prefill-max-cache-write-mib",
        "--prefill-max-stage-mib",
        "--prefill-max-compact-stage-mib",
        "--prefill-copy-chunk-mib",
        "--prefill-max-stage-raw-ranges",
        "--prefill-max-stage-coalesced-ranges",
        "--prefill-expert-stage-tiling",
        "--prefill-persistent-moe-plan-server",
        "--prefill-persistent-resident-linear-server",
        "--prefill-persistent-attention-projection-server",
        "--prefill-persistent-attention-output-server",
        "--prefill-persistent-shared-expert-server",
        "--prefill-persistent-rope-split-server",
        "--prefill-persistent-mla-attention-server",
        "--prefill-persistent-rmsnorm-server",
        "--prefill-moe-output-accumulator",
        "--prefill-static-capacity-per-expert",
        "--prefill-backend-probe-timeout-seconds",
        "--prefill-min-accelerated-flop-fraction",
        "--prefill-mpsgraph-min-batch-tokens",
        "--prefill-mpsgraph-min-matrix-dim",
        "--prefill-router-hybrid-margin-threshold",
        "--compile-mpp-probe",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
        "--no-runtime-preflight",
        "--decode-mla-key-cache",
        "--metal-runtime-cache-mla-kv-b-f32",
        "--metal-runtime-max-mla-kv-b-cache-mib",
        "--decode-max-routed-read-gib-per-token",
        "--decode-max-routed-read-seconds-per-token",
        "--metal-final-logits",
    }
)
_PREPARE_FLAGS_COMMANDS = frozenset({"prepare-glm"})
_PREPARE_FLAGS_VALUELESS_FLAGS = frozenset({"--auto-context-from-budget"})
_PREPARE_FLAGS_ALLOWED_FLAGS = frozenset(
    {
        "--auto-context-from-budget",
        "--group-size",
        "--max-cache-gib",
        "--system-reserve-gib",
        "--runtime-buffer-gib",
        "--page-cache-fraction",
        "--unified-memory-gib",
        "--cold-read-gib-s",
    }
)
_LAUNCH_PROFILE_FLAG_ATTRS = {
    "--max-live-working-set-mib": "max_live_working_set_mib",
    "--min-free-unified-memory-gib": "min_free_unified_memory_gib",
    "--require-prepared-memory-profile": "require_prepared_memory_profile",
    "--require-glm-4bit": "require_glm_4bit",
    "--require-public-glm-5-2-shape": "require_public_glm_5_2_shape",
    "--prefill-linear-backend": "prefill_linear_backend",
    "--prefill-mla-key-cache": "prefill_mla_key_cache",
    "--require-prefill-acceleration": "require_prefill_acceleration",
    "--allow-router-gate-only-prefill-acceleration": (
        "allow_router_gate_only_prefill_acceleration"
    ),
    "--allow-non-accelerated-prefill-launch-audit": (
        "allow_non_accelerated_prefill_launch_audit"
    ),
    "--prefill-prompt-chunk-tokens": "prefill_prompt_chunk_tokens",
    "--prefill-max-routed-read-amplification": (
        "prefill_max_routed_read_amplification"
    ),
    "--prefill-max-routed-read-gib": "prefill_max_routed_read_gib",
    "--prefill-ssd-read-gib-s": "prefill_ssd_read_gib_per_second",
    "--prefill-max-routed-read-seconds": "prefill_max_routed_read_seconds",
    "--max-cache-read-mib": "max_cache_read_mib",
    "--prefill-max-cache-write-mib": "prefill_max_cache_write_mib",
    "--prefill-max-stage-mib": "prefill_max_stage_mib",
    "--prefill-max-compact-stage-mib": "prefill_max_compact_stage_mib",
    "--prefill-copy-chunk-mib": "prefill_copy_chunk_mib",
    "--prefill-max-stage-raw-ranges": "prefill_max_stage_raw_ranges",
    "--prefill-max-stage-coalesced-ranges": (
        "prefill_max_stage_coalesced_ranges"
    ),
    "--prefill-expert-stage-tiling": "prefill_expert_stage_tiling",
    "--prefill-persistent-moe-plan-server": (
        "prefill_persistent_moe_plan_server"
    ),
    "--prefill-persistent-resident-linear-server": (
        "prefill_persistent_resident_linear_server"
    ),
    "--prefill-persistent-attention-projection-server": (
        "prefill_persistent_attention_projection_server"
    ),
    "--prefill-persistent-attention-output-server": (
        "prefill_persistent_attention_output_server"
    ),
    "--prefill-persistent-shared-expert-server": (
        "prefill_persistent_shared_expert_server"
    ),
    "--prefill-persistent-rope-split-server": (
        "prefill_persistent_rope_split_server"
    ),
    "--prefill-persistent-mla-attention-server": (
        "prefill_persistent_mla_attention_server"
    ),
    "--prefill-persistent-rmsnorm-server": (
        "prefill_persistent_rmsnorm_server"
    ),
    "--prefill-moe-output-accumulator": "prefill_moe_output_accumulator",
    "--prefill-static-capacity-per-expert": "prefill_static_capacity_per_expert",
    "--prefill-backend-probe-timeout-seconds": (
        "prefill_backend_probe_timeout_seconds"
    ),
    "--prefill-min-accelerated-flop-fraction": (
        "prefill_min_accelerated_flop_fraction"
    ),
    "--prefill-mpsgraph-min-batch-tokens": (
        "prefill_mpsgraph_min_batch_tokens"
    ),
    "--prefill-mpsgraph-min-matrix-dim": "prefill_mpsgraph_min_matrix_dim",
    "--prefill-router-hybrid-margin-threshold": (
        "prefill_router_hybrid_margin_threshold"
    ),
    "--compile-mpp-probe": "compile_mpp_probe",
    "--run-mpp-probe": "run_mpp_probe",
    "--run-mpsgraph-probe": "run_mpsgraph_probe",
    "--no-runtime-preflight": "preflight_runtime",
    "--metal-final-logits": "metal_final_logits",
    "--decode-mla-key-cache": "decode_mla_key_cache",
    "--metal-runtime-cache-mla-kv-b-f32": (
        "metal_runtime_cache_mla_kv_b_f32"
    ),
    "--metal-runtime-max-mla-kv-b-cache-mib": (
        "metal_runtime_max_mla_kv_b_cache_mib"
    ),
    "--decode-max-routed-read-gib-per-token": (
        "decode_max_routed_read_gib_per_token"
    ),
    "--decode-max-routed-read-seconds-per-token": (
        "decode_max_routed_read_seconds_per_token"
    ),
}
_LAUNCH_PROFILE_FLOAT_FLAGS = frozenset(
    {
        "--max-live-working-set-mib",
        "--min-free-unified-memory-gib",
        "--prefill-max-routed-read-amplification",
        "--prefill-max-routed-read-gib",
        "--prefill-ssd-read-gib-s",
        "--prefill-max-routed-read-seconds",
        "--max-cache-read-mib",
        "--prefill-max-cache-write-mib",
        "--prefill-max-stage-mib",
        "--prefill-max-compact-stage-mib",
        "--prefill-copy-chunk-mib",
        "--prefill-backend-probe-timeout-seconds",
        "--prefill-min-accelerated-flop-fraction",
        "--prefill-router-hybrid-margin-threshold",
        "--metal-runtime-max-mla-kv-b-cache-mib",
        "--decode-max-routed-read-gib-per-token",
        "--decode-max-routed-read-seconds-per-token",
    }
)
_LAUNCH_PROFILE_INT_FLAGS = frozenset(
    {
        "--prefill-prompt-chunk-tokens",
        "--prefill-max-stage-raw-ranges",
        "--prefill-max-stage-coalesced-ranges",
        "--prefill-mpsgraph-min-batch-tokens",
        "--prefill-mpsgraph-min-matrix-dim",
    }
)


def _add_launch_profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--apply-launch-profile",
        default=None,
        help=(
            "read a suggested_launch_profile/request_launch_profile JSON file "
            "and apply its safe argv before command-line overrides"
        ),
    )
    parser.add_argument(
        "--lock-launch-profile",
        action="store_true",
        help=(
            "fail if any argv flag from --apply-launch-profile is changed by "
            "later explicit command-line arguments"
        ),
    )
    parser.add_argument(
        "--require-locked-launch-profile",
        action="store_true",
        help=(
            "fail unless --apply-launch-profile is used together with "
            "--lock-launch-profile"
        ),
    )


def _launch_profile_from_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise CliArgumentError("launch profile JSON must be an object")
    if isinstance(payload.get("argv"), (list, tuple)):
        return payload
    for key in (
        "request_launch_profile",
        "suggested_launch_profile",
        "combined_launch_profile",
        "profile",
    ):
        value = payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("argv"), (list, tuple)):
            return value
    raise CliArgumentError(
        "launch profile JSON must contain argv, request_launch_profile, "
        "or suggested_launch_profile"
    )


def _load_launch_profile(profile_path: str | Path) -> dict[str, object]:
    path = Path(profile_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliArgumentError(f"failed to read launch profile {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliArgumentError(f"failed to parse launch profile {path}: {exc}") from exc
    return _launch_profile_from_payload(payload)


def _launch_profile_file_sha256(profile_path: str | Path) -> str:
    path = Path(profile_path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CliArgumentError(f"failed to read launch profile {path}: {exc}") from exc


def _safe_launch_profile_argv(profile_path: str | Path) -> list[str]:
    profile = _load_launch_profile(profile_path)
    if profile.get("argv_safe_to_replay") is False:
        raise CliArgumentError(
            "launch profile argv is not safe to replay because it has conflicts"
        )
    raw_argv = profile.get("argv")
    if not isinstance(raw_argv, (list, tuple)) or not raw_argv:
        raise CliArgumentError("launch profile argv must be a non-empty array")
    items = [str(item) for item in raw_argv]
    expanded: list[str] = []
    index = 0
    while index < len(items):
        flag = items[index]
        if flag not in _LAUNCH_PROFILE_ALLOWED_FLAGS:
            raise CliArgumentError(f"launch profile flag {flag!r} is not supported")
        expanded.append(flag)
        if flag in _LAUNCH_PROFILE_VALUELESS_FLAGS:
            index += 1
            continue
        if index + 1 >= len(items) or items[index + 1].startswith("--"):
            raise CliArgumentError(f"launch profile flag {flag} is missing a value")
        expanded.append(items[index + 1])
        index += 2
    return expanded


def _expand_launch_profile_args(argv: list[str]) -> list[str]:
    if "--apply-launch-profile" not in argv:
        return argv
    if not argv:
        return argv
    command = argv[0]
    if command not in _LAUNCH_PROFILE_COMMANDS:
        raise CliArgumentError(
            "--apply-launch-profile is only supported by prepared commands"
        )
    result: list[str] = [command]
    profile_argv: list[str] = []
    profile_path: str | None = None
    index = 1
    while index < len(argv):
        item = argv[index]
        if item != "--apply-launch-profile":
            result.append(item)
            index += 1
            continue
        if index + 1 >= len(argv):
            raise CliArgumentError("--apply-launch-profile requires a path")
        if profile_path is not None:
            raise CliArgumentError("--apply-launch-profile can only be used once")
        profile_path = argv[index + 1]
        profile_argv.extend(_safe_launch_profile_argv(profile_path))
        index += 2
    profile_args = (
        ["--apply-launch-profile", profile_path] if profile_path is not None else []
    )
    return [command, *profile_args, *profile_argv, *result[1:]]


def _safe_prepare_flags_argv(path_arg: str | Path) -> list[str]:
    path = Path(path_arg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliArgumentError(f"failed to read prepare flags {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliArgumentError(f"failed to parse prepare flags {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliArgumentError("prepare flags JSON must be an object")
    if payload.get("source") != "plan":
        raise CliArgumentError("prepare flags must have source='plan'")
    if payload.get("argv_safe_to_replay") is not True:
        raise CliArgumentError("prepare flags argv is not marked safe to replay")
    raw_argv = payload.get("argv")
    if not isinstance(raw_argv, (list, tuple)) or not raw_argv:
        raise CliArgumentError("prepare flags argv must be a non-empty array")
    items = [str(item) for item in raw_argv]
    expanded: list[str] = []
    index = 0
    while index < len(items):
        flag = items[index]
        if flag not in _PREPARE_FLAGS_ALLOWED_FLAGS:
            raise CliArgumentError(f"prepare flag {flag!r} is not supported")
        expanded.append(flag)
        if flag in _PREPARE_FLAGS_VALUELESS_FLAGS:
            index += 1
            continue
        if index + 1 >= len(items) or items[index + 1].startswith("--"):
            raise CliArgumentError(f"prepare flag {flag} is missing a value")
        expanded.append(items[index + 1])
        index += 2
    return expanded


def _prepare_flags_provenance(
    path_arg: str | Path | None,
) -> PrepareFlagsProvenance | None:
    if path_arg is None:
        return None
    path = Path(path_arg)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CliArgumentError(f"failed to read prepare flags {path}: {exc}") from exc
    return PrepareFlagsProvenance(
        source="plan",
        path=str(path),
        sha256=digest,
    )


def _expand_prepare_flags_args(argv: list[str]) -> list[str]:
    if "--apply-prepare-flags" not in argv:
        return argv
    if not argv:
        return argv
    command = argv[0]
    if command not in _PREPARE_FLAGS_COMMANDS:
        raise CliArgumentError(
            "--apply-prepare-flags is only supported by prepare-glm"
        )
    result: list[str] = [command]
    flags_argv: list[str] = []
    flags_path: str | None = None
    index = 1
    while index < len(argv):
        item = argv[index]
        if item != "--apply-prepare-flags":
            result.append(item)
            index += 1
            continue
        if index + 1 >= len(argv):
            raise CliArgumentError("--apply-prepare-flags requires a path")
        if flags_path is not None:
            raise CliArgumentError("--apply-prepare-flags can only be used once")
        flags_path = argv[index + 1]
        flags_argv.extend(_safe_prepare_flags_argv(flags_path))
        index += 2
    applied_args = (
        ["--apply-prepare-flags", flags_path] if flags_path is not None else []
    )
    return [command, *applied_args, *flags_argv, *result[1:]]


_LAUNCH_PROFILE_TARGET_FIELDS = (
    "model_config_sha256",
    "expert_layout_bytes",
    "resident_layout_bytes",
    "decode_cache_layout_bytes",
    "decode_cache_file_bytes",
    "max_context_tokens",
    "expert_quantization",
    "expert_group_size",
)
_LAUNCH_PROFILE_CONDITIONAL_TARGET_FIELDS = (
    "prepare_expert_pack_chunk_size_bytes",
    "prepare_expert_pack_estimated_peak_heap_bytes",
    "prepare_expert_pack_max_heap_bytes",
    "prepare_raw_quantization_extra_heap_bytes",
    "prepare_raw_quantization_max_source_block_bytes",
    "prepare_raw_quantization_max_output_block_bytes",
    "prepare_raw_quantization_max_rows_per_block",
    "prepare_resident_component_alias_source_tensor_count",
    "prepare_resident_component_alias_renamed_tensor_count",
    "prepare_resident_component_alias_bytes",
    "prepare_resident_fused_gate_up_source_tensor_count",
    "prepare_resident_fused_gate_up_expanded_tensor_count",
    "prepare_resident_fused_gate_up_expanded_bytes",
)
_LAUNCH_PROFILE_OPTIONAL_TARGET_FIELDS = (
    "prepare_hardware_chip_name",
    "prepare_hardware_unified_memory_bytes",
    "prepare_hardware_gpu_cores",
    "prepare_hardware_apple_silicon_generation",
    "prepare_hardware_apple_silicon_tier",
    "prepare_flags_applied",
    "prepare_flags_source",
    "prepare_flags_sha256",
    "prepare_public_glm_5_2_shape_required",
    "prepare_public_glm_5_2_shape_matches",
    "prepare_public_glm_5_2_shape_mismatched_fields",
)


def _launch_profile_conditional_target_field_applies(
    target: dict[str, object],
    current: dict[str, object],
    field: str,
) -> bool:
    if field in target:
        return True
    value = current.get(field)
    return value not in (None, 0)


def _launch_profile_target_values_equal(left: object, right: object) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    return left == right


def _launch_profile_conditional_target_fields_match(
    target: dict[str, object],
    current: dict[str, object],
) -> bool:
    return all(
        _launch_profile_target_values_equal(target.get(field), current.get(field))
        for field in _LAUNCH_PROFILE_CONDITIONAL_TARGET_FIELDS
        if _launch_profile_conditional_target_field_applies(target, current, field)
    )


_PREFILL_PLAN_PROFILE_SECTIONS = frozenset(
    {
        "prefill_guard_flags",
        "prefill_runtime_policy_flags",
        "prefill_acceleration_flags",
        "prefill_backend_probe_flags",
        "public_glm_5_2_shape_guard_flags",
    }
)
_PREFILL_ONLY_PROFILE_SOURCES = frozenset(
    {
        "prefill_linear_calibration",
        "prefill_plan",
        "prefill_plan_calibration",
    }
)
_PREFILL_ONLY_PROFILE_SECTIONS_BY_SOURCE = {
    "prefill_linear_calibration": frozenset({"prefill_runtime_policy_flags"}),
    "prefill_plan": _PREFILL_PLAN_PROFILE_SECTIONS,
    "prefill_plan_calibration": _PREFILL_PLAN_PROFILE_SECTIONS,
}
_PLAN_PROFILE_SECTIONS = frozenset({"launch_guard_flags", "decode_guard_flags"})
_PLAN_ONLY_PROFILE_SOURCES = frozenset({"plan"})
_PREFILL_PLAN_PROFILE_ALLOWED_FLAGS = frozenset(
    {
        "--prefill-linear-backend",
        "--require-prefill-acceleration",
        "--allow-router-gate-only-prefill-acceleration",
        "--require-public-glm-5-2-shape",
        "--prefill-prompt-chunk-tokens",
        "--prefill-max-routed-read-amplification",
        "--prefill-max-routed-read-gib",
        "--prefill-ssd-read-gib-s",
        "--prefill-max-routed-read-seconds",
        "--prefill-max-stage-mib",
        "--prefill-max-compact-stage-mib",
        "--prefill-static-capacity-per-expert",
        "--prefill-backend-probe-timeout-seconds",
        "--prefill-min-accelerated-flop-fraction",
        "--prefill-mpsgraph-min-batch-tokens",
        "--prefill-mpsgraph-min-matrix-dim",
        "--prefill-router-hybrid-margin-threshold",
        "--compile-mpp-probe",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
    }
)
_PLAN_PROFILE_ALLOWED_FLAGS = frozenset(
    {
        "--require-prepared-memory-profile",
        "--max-live-working-set-mib",
        "--min-free-unified-memory-gib",
        "--prefill-ssd-read-gib-s",
        "--decode-mla-key-cache",
        "--decode-max-routed-read-gib-per-token",
        "--decode-max-routed-read-seconds-per-token",
    }
)


def _launch_profile_flags(profile: dict[str, object]) -> tuple[str, ...]:
    raw_argv = profile.get("argv")
    if not isinstance(raw_argv, (list, tuple)):
        return ()
    items = [str(item) for item in raw_argv]
    flags: list[str] = []
    index = 0
    while index < len(items):
        flag = items[index]
        if not flag.startswith("--"):
            return ()
        flags.append(flag)
        if flag in _LAUNCH_PROFILE_VALUELESS_FLAGS:
            index += 1
        else:
            index += 2
    return tuple(flags)


def _launch_profile_argv_pairs(
    profile: dict[str, object],
) -> tuple[tuple[str, str | None], ...]:
    raw_argv = profile.get("argv")
    if not isinstance(raw_argv, (list, tuple)):
        return ()
    items = [str(item) for item in raw_argv]
    pairs: list[tuple[str, str | None]] = []
    index = 0
    while index < len(items):
        flag = items[index]
        if flag not in _LAUNCH_PROFILE_ALLOWED_FLAGS:
            return ()
        if flag in _LAUNCH_PROFILE_VALUELESS_FLAGS:
            pairs.append((flag, None))
            index += 1
            continue
        if index + 1 >= len(items) or items[index + 1].startswith("--"):
            return ()
        pairs.append((flag, items[index + 1]))
        index += 2
    return tuple(pairs)


def _launch_profile_value_matches(
    *,
    flag: str,
    profile_value: str | None,
    actual: object,
) -> bool:
    if flag in _LAUNCH_PROFILE_NEGATED_VALUELESS_FLAGS:
        return actual is False
    if flag in _LAUNCH_PROFILE_VALUELESS_FLAGS:
        return actual is True
    if profile_value is None:
        return False
    if flag in _LAUNCH_PROFILE_FLOAT_FLAGS:
        if actual is None or isinstance(actual, bool):
            return False
        try:
            return math.isclose(
                float(actual),
                float(profile_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            return False
    if flag in _LAUNCH_PROFILE_INT_FLAGS:
        try:
            expected = 0 if profile_value == "auto" else int(profile_value)
            if actual is None or isinstance(actual, bool):
                return False
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual) == profile_value


_LAUNCH_AUDIT_REQUEST_UPPER_BOUND_FLOAT_FLAGS = frozenset(
    {
        "--prefill-max-routed-read-amplification",
        "--prefill-max-routed-read-gib",
        "--prefill-max-routed-read-seconds",
        "--prefill-max-stage-mib",
        "--prefill-max-compact-stage-mib",
        "--decode-max-routed-read-gib-per-token",
        "--decode-max-routed-read-seconds-per-token",
    }
)
_LAUNCH_AUDIT_REQUEST_UPPER_BOUND_INT_FLAGS = frozenset(
    {
        "--prefill-max-stage-raw-ranges",
        "--prefill-max-stage-coalesced-ranges",
    }
)


def _launch_audit_request_value_is_no_wider(
    *,
    flag: str,
    profile_value: str | None,
    actual: object,
) -> bool:
    if profile_value is None or actual is None or isinstance(actual, bool):
        return False
    if flag in _LAUNCH_AUDIT_REQUEST_UPPER_BOUND_FLOAT_FLAGS:
        try:
            audited = float(profile_value)
            current = float(actual)
        except (TypeError, ValueError):
            return False
        tolerance = max(abs(audited) * 1e-6, 1e-12)
        return math.isfinite(current) and current > 0 and current <= audited + tolerance
    if flag in _LAUNCH_AUDIT_REQUEST_UPPER_BOUND_INT_FLAGS:
        try:
            audited = int(profile_value)
            current = int(actual)
        except (TypeError, ValueError):
            return False
        return current > 0 and current <= audited
    return False


def _require_locked_launch_profile_args(
    args: argparse.Namespace,
    profile: dict[str, object],
) -> None:
    if not bool(getattr(args, "lock_launch_profile", False)):
        return
    for flag, profile_value in _launch_profile_argv_pairs(profile):
        attr = _LAUNCH_PROFILE_FLAG_ATTRS.get(flag)
        if attr is None:
            continue
        actual = getattr(args, attr, None)
        if _launch_profile_value_matches(
            flag=flag,
            profile_value=profile_value,
            actual=actual,
        ):
            continue
        expected = (
            "disabled"
            if flag in _LAUNCH_PROFILE_NEGATED_VALUELESS_FLAGS
            else "enabled"
            if profile_value is None
            else profile_value
        )
        raise CliArgumentError(
            "launch profile is locked but command-line arguments changed "
            f"{flag}: profile={expected!r} current={actual!r}"
        )


def _launch_profile_allows_missing_prepared_identity(
    profile: dict[str, object],
) -> bool:
    source = profile.get("source")
    if (
        source not in _PREFILL_ONLY_PROFILE_SOURCES
        and source not in _PLAN_ONLY_PROFILE_SOURCES
    ):
        return False
    if profile.get("argv_safe_to_replay") is False:
        return False
    sections = profile.get("sections")
    if not isinstance(sections, dict) or not sections:
        return False
    flags = set(_launch_profile_flags(profile))
    if source in _PREFILL_ONLY_PROFILE_SOURCES:
        allowed_sections = _PREFILL_ONLY_PROFILE_SECTIONS_BY_SOURCE.get(source)
        if allowed_sections is None:
            return False
        if set(str(name) for name in sections) > allowed_sections:
            return False
        return bool(flags) and flags <= _PREFILL_PLAN_PROFILE_ALLOWED_FLAGS
    if source in _PLAN_ONLY_PROFILE_SOURCES:
        if set(str(name) for name in sections) > _PLAN_PROFILE_SECTIONS:
            return False
        return bool(flags) and flags <= _PLAN_PROFILE_ALLOWED_FLAGS
    return False


def _require_launch_profile_matches_prepared(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> None:
    profile_path = getattr(args, "apply_launch_profile", None)
    if bool(getattr(args, "require_locked_launch_profile", False)):
        if profile_path is None:
            raise CliArgumentError(
                "--require-locked-launch-profile requires --apply-launch-profile"
            )
        if not bool(getattr(args, "lock_launch_profile", False)):
            raise CliArgumentError(
                "--require-locked-launch-profile requires --lock-launch-profile"
            )
    if profile_path is None:
        return
    profile = _load_launch_profile(profile_path)
    target = profile.get("prepared")
    if not isinstance(target, dict):
        if _launch_profile_allows_missing_prepared_identity(profile):
            _require_locked_launch_profile_args(args, profile)
            return
        raise CliArgumentError(
            "launch profile is missing prepared identity metadata; "
            "regenerate it with inspect-prepared --write-launch-profile"
        )
    current = prepared_launch_profile_target(prepared)
    for field in _LAUNCH_PROFILE_TARGET_FIELDS:
        if field not in target:
            raise CliArgumentError(
                f"launch profile prepared identity is missing {field}"
            )
        if not _launch_profile_target_values_equal(
            target.get(field),
            current.get(field),
        ):
            raise CliArgumentError(
                "launch profile does not match this prepared package: "
                f"{field} profile={target.get(field)!r} "
                f"current={current.get(field)!r}"
            )
    for field in _LAUNCH_PROFILE_CONDITIONAL_TARGET_FIELDS:
        if not _launch_profile_conditional_target_field_applies(
            target,
            current,
            field,
        ):
            continue
        if not _launch_profile_target_values_equal(
            target.get(field),
            current.get(field),
        ):
            raise CliArgumentError(
                "launch profile does not match this prepared package: "
                f"{field} profile={target.get(field)!r} "
                f"current={current.get(field)!r}"
            )
    for field in _LAUNCH_PROFILE_OPTIONAL_TARGET_FIELDS:
        if field in target and not _launch_profile_target_values_equal(
            target.get(field),
            current.get(field),
        ):
            raise CliArgumentError(
                "launch profile does not match this prepared package: "
                f"{field} profile={target.get(field)!r} "
                f"current={current.get(field)!r}"
            )
    _require_locked_launch_profile_args(args, profile)


def _launch_profile_targets_match(
    left: object,
    right: object,
) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if not all(
        _launch_profile_target_values_equal(left.get(field), right.get(field))
        for field in _LAUNCH_PROFILE_TARGET_FIELDS
    ):
        return False
    if not _launch_profile_conditional_target_fields_match(left, right):
        return False
    if not _launch_profile_conditional_target_fields_match(right, left):
        return False
    return all(
        _launch_profile_target_values_equal(left.get(field), right.get(field))
        for field in _LAUNCH_PROFILE_OPTIONAL_TARGET_FIELDS
        if field in left or field in right
    )


def _launch_profiles_have_same_argv(
    left: object,
    right: object,
) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_argv = left.get("argv")
    right_argv = right.get("argv")
    if not isinstance(left_argv, (list, tuple)) or not isinstance(
        right_argv,
        (list, tuple),
    ):
        return False
    return tuple(str(item) for item in left_argv) == tuple(
        str(item) for item in right_argv
    )


def _applied_launch_profile_summary(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> dict[str, object] | None:
    profile_path = getattr(args, "apply_launch_profile", None)
    if profile_path is None:
        return None
    profile = _load_launch_profile(profile_path)
    sections = profile.get("sections")
    section_names = (
        tuple(sorted(str(name) for name in sections))
        if isinstance(sections, dict)
        else ()
    )
    raw_argv = profile.get("argv")
    argv = tuple(str(item) for item in raw_argv) if isinstance(raw_argv, list) else ()
    argv_pairs = _launch_profile_argv_pairs(profile)
    locked = bool(getattr(args, "lock_launch_profile", False))
    path = Path(profile_path)
    target = profile.get("prepared")
    prompt_chunk_plan = (
        sections.get("prefill_prompt_chunk_plan")
        if isinstance(sections, dict)
        else None
    )
    prefill_actual_read_time = (
        sections.get("prefill_actual_read_time")
        if isinstance(sections, dict)
        else None
    )
    prefill_actual_acceleration_coverage = (
        sections.get("prefill_actual_acceleration_coverage")
        if isinstance(sections, dict)
        else None
    )
    prefill_actual_acceleration_frontier = (
        sections.get("prefill_actual_acceleration_frontier")
        if isinstance(sections, dict)
        else None
    )
    prefill_actual_linear_backend = (
        sections.get("prefill_actual_linear_backend")
        if isinstance(sections, dict)
        else None
    )
    decode_actual_read_time = (
        sections.get("decode_actual_read_time")
        if isinstance(sections, dict)
        else None
    )
    summary = {
        "path": str(path),
        "sha256": _launch_profile_file_sha256(path),
        "source": profile.get("source"),
        "argv": argv,
        "argv_safe_to_replay": profile.get("argv_safe_to_replay", True),
        "locked": locked,
        "lock_required": bool(
            getattr(args, "require_locked_launch_profile", False)
        ),
        "profile_flag_count": len(argv_pairs),
        "lock_checked_flags": (
            tuple(flag for flag, _value in argv_pairs) if locked else ()
        ),
        "prepared": target,
        "section_names": section_names,
        "matches_prepared": True if isinstance(target, dict) else None,
        "current_prepared_manifest": str(prepared.manifest_path),
    }
    if isinstance(prompt_chunk_plan, dict):
        summary["prefill_prompt_chunk_plan"] = prompt_chunk_plan
    if isinstance(prefill_actual_read_time, dict):
        summary["prefill_actual_read_time"] = prefill_actual_read_time
    if isinstance(prefill_actual_acceleration_coverage, dict):
        summary["prefill_actual_acceleration_coverage"] = (
            prefill_actual_acceleration_coverage
        )
    if isinstance(prefill_actual_acceleration_frontier, dict):
        summary["prefill_actual_acceleration_frontier"] = (
            prefill_actual_acceleration_frontier
        )
    if isinstance(prefill_actual_linear_backend, dict):
        summary["prefill_actual_linear_backend"] = prefill_actual_linear_backend
    if isinstance(decode_actual_read_time, dict):
        summary["decode_actual_read_time"] = decode_actual_read_time
    return summary


def _attach_applied_profile_to_token_result(
    result: TokenGenerationResult,
    applied_launch_profile: dict[str, object] | None,
) -> TokenGenerationResult:
    if applied_launch_profile is None:
        return result
    prompt_prefill = result.prompt_prefill
    drift = prefill_prompt_chunk_plan_drift_summary(
        applied_launch_profile=applied_launch_profile,
        actual_prompt_chunk_tokens=(
            prompt_prefill.chunk_tokens if prompt_prefill is not None else None
        ),
        actual_auto_plan=result.auto_prefill_prompt_chunk_plan,
        actual_max_safe_plan=result.max_safe_prefill_prompt_chunk_plan,
    )
    return replace(
        result,
        applied_launch_profile=applied_launch_profile,
        prefill_prompt_chunk_plan_drift=drift,
    )


def _scaled_bytes_arg(
    value: object,
    *,
    name: str,
    scale: int,
    minimum: float = 0.0,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CliArgumentError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CliArgumentError(f"{name} must be finite")
    if parsed < minimum:
        raise CliArgumentError(f"{name} must be >= {minimum:g}")
    return int(parsed * scale)


def _scaled_count_arg(
    value: object,
    *,
    name: str,
    scale: float,
    minimum: float = 0.0,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CliArgumentError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CliArgumentError(f"{name} must be finite")
    if parsed < minimum:
        raise CliArgumentError(f"{name} must be >= {minimum:g}")
    return int(parsed * scale)


def _positive_float_arg(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise CliArgumentError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CliArgumentError(f"{name} must be finite")
    if parsed <= 0:
        raise CliArgumentError(f"{name} must be positive")
    return parsed


def _json_default(obj: Any) -> Any:
    if isinstance(obj, PrefillBackendCapability):
        payload = asdict(obj)
        payload["metal4_ml_runtime_available"] = obj.metal4_ml_runtime_available
        payload["mps_graph_runtime_available"] = obj.mps_graph_runtime_available
        payload["mpp_runtime_available"] = obj.mpp_runtime_available
        payload["prefill_acceleration_runtimes"] = (
            obj.prefill_acceleration_runtimes
        )
        payload["selectable_accelerated_prefill_backends"] = (
            obj.selectable_accelerated_prefill_backends
        )
        payload["selectable_prefill_acceleration_available"] = (
            obj.selectable_prefill_acceleration_available
        )
        payload["validated_accelerated_prefill_backends"] = (
            obj.validated_accelerated_prefill_backends
        )
        payload["validated_prefill_acceleration_available"] = (
            obj.validated_prefill_acceleration_available
        )
        payload["prefill_acceleration_runtime_gaps"] = (
            obj.prefill_acceleration_runtime_gaps
        )
        payload["prefill_neural_accelerator_status"] = (
            obj.prefill_neural_accelerator_status
        )
        payload["suggested_prefill_acceleration_flags"] = (
            obj.suggested_prefill_acceleration_flags
        )
        return payload
    if is_dataclass(obj):
        return {field.name: getattr(obj, field.name) for field in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json_file_atomic(path_arg: str | Path, payload: object) -> None:
    path = Path(path_arg)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, default=_json_default, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise CliArgumentError(f"failed to write JSON file {path}: {exc}") from exc


def _read_json_object_file(path_arg: str | Path, label: str) -> dict[str, object]:
    path = Path(path_arg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliArgumentError(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliArgumentError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliArgumentError(f"{label} JSON must be an object")
    return payload


def _selected_replay_required_environment(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    env: dict[str, str] = {}
    for key, raw_value in value.items():
        if isinstance(key, str) and key and isinstance(raw_value, str):
            env[key] = raw_value
    return env


def _bakeoff_selected_replay_payload(bakeoff: dict[str, object]) -> dict[str, object]:
    selected = bakeoff.get("selected")
    if not isinstance(selected, dict):
        raise CliArgumentError("result bakeoff did not produce a selected result")
    launch_binding = selected.get("launch_binding")
    if not isinstance(launch_binding, dict):
        raise CliArgumentError("selected result is missing launch binding")
    if launch_binding.get("replay_ready") is not True:
        raise CliArgumentError(
            "selected result is not replay-ready"
            f" (role={selected.get('role')}; path={selected.get('path')})"
        )
    if launch_binding.get("replay_files_ready") is not True:
        raise CliArgumentError(
            "selected replay files are not ready"
            f" (role={selected.get('role')}; path={selected.get('path')})"
        )
    argv = launch_binding.get("replay_generate_token_ids_argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(part, str) or not part for part in argv)
    ):
        raise CliArgumentError("selected replay argv is missing or invalid")
    selected_result_path = selected.get("path")
    selected_prefill_plan_signature: dict[str, object] | None = None
    selected_total_elapsed_seconds: float | None = None
    if isinstance(selected_result_path, str) and selected_result_path:
        selected_summary = summarize_result_file(selected_result_path)
        signature = selected_summary.get("prefill_plan_signature")
        if isinstance(signature, dict):
            selected_prefill_plan_signature = dict(signature)
        elapsed = selected_summary.get("total_elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            parsed_elapsed = float(elapsed)
            if math.isfinite(parsed_elapsed):
                selected_total_elapsed_seconds = parsed_elapsed
    pre_run_checks = _selected_replay_pre_run_check_policy(launch_binding)
    required_environment = _selected_replay_required_environment(
        selected.get("required_environment")
    )
    return {
        "schema": "largerlm.selected_replay.v1",
        "source": "result_bakeoff",
        "selected_role": selected.get("role"),
        "selected_result": selected.get("path"),
        "selected_decision": selected.get("decision"),
        "selected_reasons": selected.get("reasons"),
        "selected_total_elapsed_seconds": selected_total_elapsed_seconds,
        "selected_prefill_plan_signature": selected_prefill_plan_signature,
        "baseline": bakeoff.get("baseline"),
        "baseline_retained": bakeoff.get("baseline_retained"),
        "winner": bakeoff.get("winner"),
        "launch_binding": launch_binding,
        "required_environment": required_environment,
        "pre_run_checks": pre_run_checks,
        "argv": argv,
        "command": shlex.join(argv),
    }


def _selected_replay_pre_run_check_policy(
    launch_binding: dict[str, object],
) -> dict[str, object]:
    manifest_path_value = launch_binding.get("prepared_manifest")
    requested_bytes: int | None = None
    auto_enabled = True
    if isinstance(manifest_path_value, str) and manifest_path_value:
        try:
            manifest = _read_json_object_file(
                Path(manifest_path_value),
                "prepared manifest",
            )
        except CliArgumentError:
            manifest = {}
        value = manifest.get("prepare_cold_read_benchmark_requested_bytes")
        if (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
        ):
            requested_bytes = int(value)
            auto_enabled = requested_bytes >= 64 * 1024**2
    bytes_mib = (
        requested_bytes / float(1024**2)
        if requested_bytes is not None
        else 1024.0
    )
    return {
        "ssd_read_speed": {
            "enabled": auto_enabled,
            "source": "prepared_manifest_cold_read_benchmark",
            "auto_enabled_min_bytes": 64 * 1024**2,
            "min_ratio": 0.75,
            "bytes_mib": bytes_mib,
            "chunk_mib": 8.0,
            "max_chunk_mib": 512.0,
        }
    }


def _write_selected_replay_script(
    path_arg: str | Path,
    payload: dict[str, object],
    *,
    selected_replay_json_path: str | Path | None = None,
) -> None:
    argv = payload.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(part, str) or not part for part in argv)
    ):
        raise CliArgumentError("selected replay argv is missing or invalid")
    path = Path(path_arg)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    script_parent_abs = (
        path.parent if path.is_absolute() else Path.cwd() / path.parent
    ).resolve()
    repo_dir_abs = Path.cwd().resolve()
    repo_dir_relpath = os.path.relpath(repo_dir_abs, start=script_parent_abs)

    def script_dir_ref(relpath: str) -> str:
        if relpath == ".":
            return '"$SCRIPT_DIR"'
        return '"$SCRIPT_DIR"/' + shlex.quote(relpath)

    repo_dir_script_ref = script_dir_ref(repo_dir_relpath)
    replay_json_text = (
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True)
        if selected_replay_json_path is None
        else None
    )
    selected_replay_json_script_ref: str | None = None
    if selected_replay_json_path is not None:
        replay_json_path = Path(selected_replay_json_path)
        if replay_json_path.is_absolute():
            selected_replay_json_script_ref = shlex.quote(str(replay_json_path))
        else:
            replay_json_abs = (Path.cwd() / replay_json_path).resolve()
            relpath = os.path.relpath(replay_json_abs, start=script_parent_abs)
            selected_replay_json_script_ref = script_dir_ref(relpath)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# LargerLM selected replay command generated by result-bakeoff.",
        "# Includes bounded SSD read-speed validation before model loading.",
        "# Runs the checked replay quietly unless extra args override runner behavior.",
        f"# selected_role: {payload.get('selected_role')}",
        f"# selected_result: {payload.get('selected_result')}",
        'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f"REPO_DIR={repo_dir_script_ref}",
        'cd "$REPO_DIR"',
    ]
    required_environment = _selected_replay_required_environment(
        payload.get("required_environment")
    )
    for key in sorted(required_environment):
        lines.append(f"export {key}={shlex.quote(required_environment[key])}")
    if selected_replay_json_path is not None:
        lines.extend(
            [
                "exec python -m largerlm selected-replay-run "
                + (
                    selected_replay_json_script_ref
                    or shlex.quote(str(selected_replay_json_path))
                )
                + " --quiet-runner --check-ssd-read-speed"
                + ' "$@"',
            ]
        )
    else:
        lines.extend(
            [
                'selected_replay_json="$(mktemp "${TMPDIR:-/tmp}/largerlm-selected-replay.XXXXXX.json")"',
                'cleanup() { rm -f "$selected_replay_json"; }',
                "trap cleanup EXIT",
                "cat > \"$selected_replay_json\" <<'LARGERLM_SELECTED_REPLAY_JSON'",
                replay_json_text or "{}",
                "LARGERLM_SELECTED_REPLAY_JSON",
                (
                    'python -m largerlm selected-replay-run "$selected_replay_json" '
                    '--quiet-runner --check-ssd-read-speed "$@"'
                ),
            ]
        )
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.chmod(0o755)
        tmp.replace(path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise CliArgumentError(
            f"failed to write selected replay script {path}: {exc}"
        ) from exc


def _selected_replay_argv(value: object) -> list[str] | None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(part, str) or not part for part in value)
    ):
        return None
    return list(value)


def _selected_replay_check_record(
    code: str,
    ok: bool,
    message: str,
    **details: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "code": code,
        "ok": bool(ok),
        "message": message,
    }
    record.update(details)
    return record


def _strict_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _selected_replay_launch_audit_request_check(
    current_binding: dict[str, object],
) -> tuple[dict[str, object], str | None]:
    audit_payload, audit_error = _selected_replay_launch_audit_payload(
        current_binding
    )
    return (
        audit_payload.get("request_check")
        if isinstance(audit_payload, dict)
        and isinstance(audit_payload.get("request_check"), dict)
        else {}
    ), audit_error


def _selected_replay_launch_audit_payload(
    current_binding: dict[str, object],
) -> tuple[dict[str, object], str | None]:
    launch_audit_path = current_binding.get("launch_audit_path")
    audit_payload: dict[str, object] | None = None
    audit_error: str | None = None
    if isinstance(launch_audit_path, str) and launch_audit_path:
        try:
            audit_payload = _read_json_object_file(launch_audit_path, "launch audit")
        except CliArgumentError as exc:
            audit_error = str(exc)
    return audit_payload or {}, audit_error


def _selected_replay_current_memory_checks(
    current_binding: dict[str, object],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    launch_audit_path = current_binding.get("launch_audit_path")
    request_check, audit_error = _selected_replay_launch_audit_request_check(
        current_binding
    )
    runtime = (
        request_check.get("runtime_preflight")
        if isinstance(request_check, dict)
        and isinstance(request_check.get("runtime_preflight"), dict)
        else {}
    )
    runtime_present = bool(runtime)
    checks.append(
        _selected_replay_check_record(
            "launch_audit_runtime_preflight_present",
            runtime_present,
            "launch audit must include request runtime_preflight evidence",
            audit_path=launch_audit_path,
            error=audit_error,
        )
    )
    runtime_ran = runtime.get("ran") is True if runtime_present else False
    runtime_ok = (
        runtime.get("available_memory_ok") is True if runtime_present else False
    )
    required_available = (
        _strict_nonnegative_int(runtime.get("required_available_memory_bytes"))
        if runtime_present
        else None
    )
    audited_available = (
        _strict_nonnegative_int(runtime.get("system_available_memory_bytes"))
        if runtime_present
        else None
    )
    checks.extend(
        [
            _selected_replay_check_record(
                "launch_audit_runtime_preflight_ran",
                runtime_ran,
                "launch audit runtime_preflight must have run",
                actual=runtime.get("ran") if runtime_present else None,
            ),
            _selected_replay_check_record(
                "launch_audit_runtime_preflight_ok",
                runtime_ok,
                "launch audit runtime_preflight must have passed its memory check",
                actual=(
                    runtime.get("available_memory_ok") if runtime_present else None
                ),
            ),
            _selected_replay_check_record(
                "launch_audit_required_available_memory_present",
                required_available is not None,
                "launch audit runtime_preflight must record required available memory",
                required_available_memory_bytes=required_available,
                audited_system_available_memory_bytes=audited_available,
            ),
        ]
    )

    snapshot = system_memory_snapshot()
    current_available = (
        _strict_nonnegative_int(getattr(snapshot, "available_bytes", None))
        if snapshot is not None
        else None
    )
    current_total = (
        _strict_nonnegative_int(getattr(snapshot, "total_bytes", None))
        if snapshot is not None
        else None
    )
    source = getattr(snapshot, "source", None) if snapshot is not None else None
    checks.append(
        _selected_replay_check_record(
            "current_system_memory_available",
            current_available is not None,
            "current available unified/system memory must be inspectable before replay",
            current_system_available_memory_bytes=current_available,
            current_system_total_memory_bytes=current_total,
            current_system_memory_source=source,
        )
    )
    checks.append(
        _selected_replay_check_record(
            "current_available_memory_meets_audit_requirement",
            required_available is not None
            and current_available is not None
            and current_available >= required_available,
            "current available memory must meet the audited runtime requirement before replay",
            required_available_memory_bytes=required_available,
            current_system_available_memory_bytes=current_available,
            current_system_total_memory_bytes=current_total,
            current_system_memory_source=source,
        )
    )
    return checks


def _selected_replay_disk_usage(path_arg: object) -> dict[str, object] | None:
    if not isinstance(path_arg, str) or not path_arg:
        return None
    path = Path(path_arg)
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return None
    return {
        "path": str(path),
        "probe_path": str(probe),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _selected_replay_current_disk_checks(
    current_binding: dict[str, object],
) -> list[dict[str, object]]:
    request_check, audit_error = _selected_replay_launch_audit_request_check(
        current_binding
    )
    disk = (
        request_check.get("prefill_stage_temp_disk_free")
        if isinstance(request_check, dict)
        and isinstance(request_check.get("prefill_stage_temp_disk_free"), dict)
        else {}
    )
    disk_present = bool(disk)
    path = disk.get("path") if disk_present else None
    required_free = (
        _strict_nonnegative_int(disk.get("required_free_bytes"))
        if disk_present
        else None
    )
    audited_free = (
        _strict_nonnegative_int(disk.get("free_bytes")) if disk_present else None
    )
    usage = _selected_replay_disk_usage(path)
    current_free = (
        _strict_nonnegative_int(usage.get("free_bytes"))
        if isinstance(usage, dict)
        else None
    )
    return [
        _selected_replay_check_record(
            "launch_audit_stage_temp_disk_present",
            disk_present,
            "launch audit must include request stage temp disk evidence",
            audit_path=current_binding.get("launch_audit_path"),
            error=audit_error,
        ),
        _selected_replay_check_record(
            "launch_audit_stage_temp_disk_ok",
            disk.get("within_free_space") is True if disk_present else False,
            "launch audit stage temp disk check must have passed",
            actual=disk.get("within_free_space") if disk_present else None,
            audited_free_bytes=audited_free,
            required_free_bytes=required_free,
            path=path,
        ),
        _selected_replay_check_record(
            "launch_audit_stage_temp_required_free_present",
            required_free is not None,
            "launch audit stage temp disk check must record required free bytes",
            required_free_bytes=required_free,
            audited_free_bytes=audited_free,
            path=path,
        ),
        _selected_replay_check_record(
            "current_stage_temp_disk_free_available",
            current_free is not None,
            "current stage temp disk free bytes must be inspectable before replay",
            current_free_bytes=current_free,
            current_total_bytes=(usage or {}).get("total_bytes")
            if isinstance(usage, dict)
            else None,
            current_used_bytes=(usage or {}).get("used_bytes")
            if isinstance(usage, dict)
            else None,
            path=path,
            probe_path=(usage or {}).get("probe_path")
            if isinstance(usage, dict)
            else None,
        ),
        _selected_replay_check_record(
            "current_stage_temp_disk_meets_audit_requirement",
            required_free is not None
            and current_free is not None
            and current_free >= required_free,
            "current stage temp disk free bytes must meet the audited request requirement",
            required_free_bytes=required_free,
            current_free_bytes=current_free,
            current_total_bytes=(usage or {}).get("total_bytes")
            if isinstance(usage, dict)
            else None,
            current_used_bytes=(usage or {}).get("used_bytes")
            if isinstance(usage, dict)
            else None,
            path=path,
            probe_path=(usage or {}).get("probe_path")
            if isinstance(usage, dict)
            else None,
        ),
    ]


def _selected_replay_benchmark_path(
    manifest_path: Path,
    value: object,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw
    return manifest_path.parent / raw


def _selected_replay_current_ssd_read_speed_checks(
    current_binding: dict[str, object],
    *,
    enabled: bool,
    min_ratio: float,
    bytes_to_read: int,
    chunk_bytes: int,
    max_chunk_bytes: int | None,
) -> list[dict[str, object]]:
    if not enabled:
        return []

    manifest_path_value = current_binding.get("prepared_manifest")
    benchmark_path: Path | None = None
    baseline_gib_s: float | None = None
    manifest_error: str | None = None
    benchmark_error: str | None = None
    benchmark = None
    if not isinstance(manifest_path_value, str) or not manifest_path_value:
        manifest_error = "current launch binding is missing prepared_manifest"
    else:
        manifest_path = Path(manifest_path_value)
        try:
            manifest = _read_json_object_file(manifest_path, "prepared manifest")
        except CliArgumentError as exc:
            manifest_error = str(exc)
        else:
            baseline = manifest.get("prepare_cold_read_gib_per_second")
            if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
                baseline_gib_s = None
            else:
                parsed_baseline = float(baseline)
                if math.isfinite(parsed_baseline) and parsed_baseline > 0:
                    baseline_gib_s = parsed_baseline
            benchmark_path = _selected_replay_benchmark_path(
                manifest_path,
                manifest.get("prepare_cold_read_benchmark_path"),
            )

    if manifest_error is None and baseline_gib_s is None:
        manifest_error = "prepared manifest does not record prepare_cold_read_gib_per_second"
    if manifest_error is None and benchmark_path is None:
        manifest_error = "prepared manifest does not record prepare_cold_read_benchmark_path"

    if manifest_error is None and benchmark_path is not None:
        try:
            benchmark = benchmark_sequential_read(
                benchmark_path,
                bytes_to_read=bytes_to_read,
                chunk_bytes=chunk_bytes,
                max_chunk_bytes=max_chunk_bytes,
            )
        except DiskBenchmarkError as exc:
            benchmark_error = str(exc)

    actual_gib_s = (
        float(benchmark.gib_per_second)
        if benchmark is not None
        and isinstance(benchmark.gib_per_second, (int, float))
        and math.isfinite(float(benchmark.gib_per_second))
        else None
    )
    required_gib_s = (
        float(baseline_gib_s) * min_ratio if baseline_gib_s is not None else None
    )
    ratio = (
        actual_gib_s / float(baseline_gib_s)
        if actual_gib_s is not None and baseline_gib_s is not None
        else None
    )
    measured_bytes = int(benchmark.measured_bytes) if benchmark is not None else None
    short_read = bool(benchmark.short_read) if benchmark is not None else None
    ok = (
        manifest_error is None
        and benchmark_error is None
        and actual_gib_s is not None
        and required_gib_s is not None
        and measured_bytes is not None
        and measured_bytes > 0
        and short_read is False
        and actual_gib_s >= required_gib_s
    )
    return [
        _selected_replay_check_record(
            "current_ssd_read_speed_ok",
            ok,
            (
                "current bounded SSD read speed must meet the prepared "
                "cold-read baseline ratio before replay"
            ),
            prepared_manifest=manifest_path_value,
            benchmark_path=str(benchmark_path) if benchmark_path is not None else None,
            baseline_gib_per_second=baseline_gib_s,
            min_ratio=min_ratio,
            required_gib_per_second=required_gib_s,
            actual_gib_per_second=actual_gib_s,
            actual_ratio=ratio,
            requested_bytes=bytes_to_read,
            measured_bytes=measured_bytes,
            chunk_bytes=chunk_bytes,
            elapsed_seconds=(
                float(benchmark.elapsed_seconds) if benchmark is not None else None
            ),
            short_read=short_read,
            manifest_error=manifest_error,
            benchmark_error=benchmark_error,
        )
    ]


def _serve_prepared_ssd_read_speed_check(args: argparse.Namespace, prepared) -> None:
    if not bool(getattr(args, "check_ssd_read_speed", False)):
        return
    min_ratio = _positive_float_arg(
        (
            getattr(args, "ssd_read_speed_min_ratio", None)
            if getattr(args, "ssd_read_speed_min_ratio", None) is not None
            else 0.75
        ),
        name="--ssd-read-speed-min-ratio",
    )
    bytes_to_read = _scaled_bytes_arg(
        (
            getattr(args, "ssd_read_speed_bytes_mib", None)
            if getattr(args, "ssd_read_speed_bytes_mib", None) is not None
            else 1024.0
        ),
        name="--ssd-read-speed-bytes-mib",
        scale=1024**2,
        minimum=1 / 1024**2,
    )
    chunk_bytes = _scaled_bytes_arg(
        (
            getattr(args, "ssd_read_speed_chunk_mib", None)
            if getattr(args, "ssd_read_speed_chunk_mib", None) is not None
            else 8.0
        ),
        name="--ssd-read-speed-chunk-mib",
        scale=1024**2,
        minimum=1 / 1024**2,
    )
    max_chunk_bytes = _scaled_bytes_arg(
        (
            getattr(args, "ssd_read_speed_max_chunk_mib", None)
            if getattr(args, "ssd_read_speed_max_chunk_mib", None) is not None
            else 512.0
        ),
        name="--ssd-read-speed-max-chunk-mib",
        scale=1024**2,
        minimum=1 / 1024**2,
    )
    if bytes_to_read is None or chunk_bytes is None or max_chunk_bytes is None:
        raise AssertionError("serve-prepared SSD read check defaults are invalid")
    checks = _selected_replay_current_ssd_read_speed_checks(
        {"prepared_manifest": str(prepared.manifest_path)},
        enabled=True,
        min_ratio=min_ratio,
        bytes_to_read=bytes_to_read,
        chunk_bytes=chunk_bytes,
        max_chunk_bytes=max_chunk_bytes,
    )
    check = checks[0] if checks else None
    if not isinstance(check, dict) or check.get("ok") is not True:
        detail_parts = []
        if isinstance(check, dict):
            for field in (
                "actual_gib_per_second",
                "required_gib_per_second",
                "baseline_gib_per_second",
                "actual_ratio",
                "manifest_error",
                "benchmark_error",
            ):
                value = check.get(field)
                if value is not None:
                    detail_parts.append(f"{field}={value!r}")
        detail = ": " + ", ".join(detail_parts) if detail_parts else ""
        raise CliArgumentError(
            "current SSD read speed check failed before starting serve-prepared"
            + detail
        )


def _selected_replay_launch_profile_argv(
    current_binding: dict[str, object],
) -> tuple[list[str] | None, str | None]:
    profile_path = current_binding.get("launch_profile_path")
    if not isinstance(profile_path, str) or not profile_path:
        return None, "current launch profile path is missing"
    try:
        payload = _read_json_object_file(profile_path, "launch profile")
    except CliArgumentError as exc:
        return None, str(exc)
    argv = _selected_replay_argv(payload.get("argv"))
    if argv is None:
        return None, "launch profile argv must be a non-empty string array"
    return argv, None


def _selected_replay_audit_check_by_code(
    audit_payload: dict[str, object],
    code: str,
) -> dict[str, object]:
    audit = audit_payload.get("launch_audit")
    checks = audit.get("checks") if isinstance(audit, dict) else None
    if not isinstance(checks, list):
        return {}
    for item in checks:
        if isinstance(item, dict) and item.get("code") == code:
            return item
    return {}


def _argv_has_flag(argv: list[str] | None, flag: str) -> bool:
    return isinstance(argv, list) and flag in argv


def _selected_replay_prefill_acceleration_checks(
    current_binding: dict[str, object],
) -> list[dict[str, object]]:
    audit_payload, audit_error = _selected_replay_launch_audit_payload(
        current_binding
    )
    request_check = (
        audit_payload.get("request_check")
        if isinstance(audit_payload.get("request_check"), dict)
        else {}
    )
    coverage = (
        request_check.get("prefill_acceleration_coverage")
        if isinstance(request_check, dict)
        and isinstance(request_check.get("prefill_acceleration_coverage"), dict)
        else {}
    )
    required_check = _selected_replay_audit_check_by_code(
        audit_payload,
        "prefill_acceleration_required",
    )
    gate_check = _selected_replay_audit_check_by_code(
        audit_payload,
        "prefill_acceleration_gate_ok",
    )
    probe_check = _selected_replay_audit_check_by_code(
        audit_payload,
        "prefill_acceleration_probe_ok",
    )
    profile_probe_check = _selected_replay_audit_check_by_code(
        audit_payload,
        "prefill_acceleration_profile_replays_probe",
    )
    request_coverage_check = _selected_replay_audit_check_by_code(
        audit_payload,
        "request_prefill_acceleration_coverage_ok",
    )
    acceleration_required = (
        required_check.get("required") is True
        or coverage.get("required") is True
        or request_coverage_check.get("required") is True
    )
    profile_argv, profile_error = _selected_replay_launch_profile_argv(
        current_binding
    )
    mpsgraph_selected = acceleration_required and (
        "mpsgraph-f32" in (gate_check.get("prefill_acceleration_runtimes") or ())
        or "mpsgraph-f32"
        in (gate_check.get("selectable_accelerated_prefill_backends") or ())
        or "mpsgraph-f32" in (coverage.get("accelerated_backends") or ())
        or _strict_nonnegative_int(coverage.get("mpsgraph_matrix_count")) not in (
            None,
            0,
        )
    )
    runtime_probe_required = acceleration_required and (
        profile_probe_check.get("required") is True
        or profile_probe_check.get("runtime_probe_required") is True
        or probe_check.get("mps_graph_probe_requested") is True
    )
    coverage_errors = (
        _request_prefill_acceleration_coverage_errors(coverage)
        if isinstance(coverage, dict)
        else ("missing prefill_acceleration_coverage",)
    )
    coverage_check_errors = (
        _request_prefill_acceleration_coverage_errors(request_coverage_check)
        if isinstance(request_coverage_check, dict)
        else ("missing request_prefill_acceleration_coverage_ok check",)
    )
    coverage_audit_fields_match = (
        isinstance(coverage, dict)
        and isinstance(request_coverage_check, dict)
        and not coverage_errors
        and not coverage_check_errors
        and all(
            request_coverage_check.get(field) == coverage.get(field)
            for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS
        )
    )
    return [
        _selected_replay_check_record(
            "launch_audit_prefill_acceleration_required",
            True,
            "launch audit prefill acceleration requirement must be explicit",
            audit_error=audit_error,
            required=acceleration_required,
            required_check_required=required_check.get("required"),
            required_check_ok=required_check.get("ok"),
            request_coverage_required=coverage.get("required"),
        ),
        _selected_replay_check_record(
            "launch_audit_prefill_acceleration_gate_ok",
            not acceleration_required or gate_check.get("ok") is True,
            "launch audit prefill acceleration gate must have passed",
            actual=gate_check.get("ok"),
            runtimes=gate_check.get("prefill_acceleration_runtimes"),
            validated=gate_check.get("validated_prefill_acceleration_available"),
        ),
        _selected_replay_check_record(
            "launch_audit_prefill_acceleration_probe_ok",
            not acceleration_required or probe_check.get("ok") is True,
            "launch audit prefill acceleration runtime probe must have passed",
            actual=probe_check.get("ok"),
            mps_graph_probe_requested=probe_check.get("mps_graph_probe_requested"),
            mps_graph_probe_ran=probe_check.get("mps_graph_probe_ran"),
            mps_graph_probe_ok=probe_check.get("mps_graph_probe_ok"),
        ),
        _selected_replay_check_record(
            "launch_audit_request_prefill_acceleration_coverage_ok",
            not acceleration_required
            or (
                coverage.get("ok") is True
                and request_coverage_check.get("ok") is True
                and _strict_nonnegative_int(coverage.get("accelerated_matrix_count"))
                not in (None, 0)
            ),
            "launch audit checked request must preserve accelerated prefill coverage",
            coverage_ok=coverage.get("ok"),
            coverage_check_ok=request_coverage_check.get("ok"),
            accelerated_matrix_count=coverage.get("accelerated_matrix_count"),
            accelerated_backends=coverage.get("accelerated_backends"),
        ),
        _selected_replay_check_record(
            "launch_audit_request_prefill_acceleration_coverage_schema_valid",
            not acceleration_required or not coverage_errors,
            "launch audit request prefill acceleration coverage must carry full audit schema evidence",
            errors=coverage_errors,
        ),
        _selected_replay_check_record(
            "launch_audit_request_prefill_acceleration_check_schema_valid",
            not acceleration_required or not coverage_check_errors,
            "launch audit prefill acceleration coverage check must carry full audit schema evidence",
            errors=coverage_check_errors,
            evidence_present=request_coverage_check.get("evidence_present"),
        ),
        _selected_replay_check_record(
            "launch_audit_request_prefill_acceleration_coverage_matches_check",
            not acceleration_required or coverage_audit_fields_match,
            "launch audit prefill acceleration coverage check must match request evidence",
        ),
        _selected_replay_check_record(
            "launch_profile_argv_loadable",
            profile_argv is not None,
            "current launch profile argv must be readable before replay",
            error=profile_error,
        ),
        _selected_replay_check_record(
            "launch_profile_requires_prefill_acceleration",
            not acceleration_required
            or _argv_has_flag(profile_argv, "--require-prefill-acceleration"),
            "current launch profile must replay the prefill acceleration requirement",
        ),
        _selected_replay_check_record(
            "launch_profile_replays_mpsgraph_probe",
            not runtime_probe_required
            or _argv_has_flag(profile_argv, "--run-mpsgraph-probe"),
            "current launch profile must replay the audited MPSGraph runtime probe",
            runtime_probe_required=runtime_probe_required,
        ),
        _selected_replay_check_record(
            "launch_profile_mpsgraph_thresholds_present",
            not mpsgraph_selected
            or (
                _argv_has_flag(profile_argv, "--prefill-mpsgraph-min-batch-tokens")
                and _argv_has_flag(profile_argv, "--prefill-mpsgraph-min-matrix-dim")
            ),
            "current launch profile must preserve MPSGraph auto-threshold flags",
            mpsgraph_selected=mpsgraph_selected,
        ),
    ]


def _selected_replay_ssd_read_speed_policy(
    selected_replay: dict[str, object],
) -> dict[str, object]:
    pre_run_checks = selected_replay.get("pre_run_checks")
    if not isinstance(pre_run_checks, dict):
        return {}
    policy = pre_run_checks.get("ssd_read_speed")
    if not isinstance(policy, dict):
        return {}
    enabled = policy.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise CliArgumentError(
            "selected replay pre_run_checks.ssd_read_speed.enabled must be boolean"
        )
    resolved: dict[str, object] = {}
    if enabled is not None:
        resolved["enabled"] = enabled
    for key in ("min_ratio", "bytes_mib", "chunk_mib", "max_chunk_mib"):
        value = policy.get(key)
        if value is not None:
            resolved[key] = value
    return resolved


def _selected_replay_check_payload(
    path_arg: str | Path,
    *,
    check_ssd_read_speed: bool | None = None,
    ssd_read_speed_min_ratio: float | None = None,
    ssd_read_speed_bytes_mib: float | None = None,
    ssd_read_speed_chunk_mib: float | None = None,
    ssd_read_speed_max_chunk_mib: float | None = None,
) -> dict[str, object]:
    path = Path(path_arg)
    payload = _read_json_object_file(path, "selected replay")
    ssd_policy = _selected_replay_ssd_read_speed_policy(payload)
    ssd_read_speed_enabled = (
        bool(check_ssd_read_speed)
        if check_ssd_read_speed is not None
        else bool(ssd_policy.get("enabled"))
    )
    ssd_read_speed_min_ratio_value = (
        ssd_read_speed_min_ratio
        if ssd_read_speed_min_ratio is not None
        else ssd_policy.get("min_ratio", 0.75)
    )
    ssd_read_speed_bytes_mib_value = (
        ssd_read_speed_bytes_mib
        if ssd_read_speed_bytes_mib is not None
        else ssd_policy.get("bytes_mib", 1024.0)
    )
    ssd_read_speed_chunk_mib_value = (
        ssd_read_speed_chunk_mib
        if ssd_read_speed_chunk_mib is not None
        else ssd_policy.get("chunk_mib", 8.0)
    )
    ssd_read_speed_max_chunk_mib_value = (
        ssd_read_speed_max_chunk_mib
        if ssd_read_speed_max_chunk_mib is not None
        else ssd_policy.get("max_chunk_mib", 512.0)
    )
    ssd_read_speed_min_ratio = _positive_float_arg(
        ssd_read_speed_min_ratio_value,
        name="--ssd-read-speed-min-ratio",
    )
    ssd_read_speed_bytes = _scaled_bytes_arg(
        ssd_read_speed_bytes_mib_value,
        name="--ssd-read-speed-bytes-mib",
        scale=1024**2,
        minimum=1 / 1024**2,
    )
    ssd_read_speed_chunk = _scaled_bytes_arg(
        ssd_read_speed_chunk_mib_value,
        name="--ssd-read-speed-chunk-mib",
        scale=1024**2,
        minimum=1 / 1024**2,
    )
    ssd_read_speed_max_chunk = _scaled_bytes_arg(
        ssd_read_speed_max_chunk_mib_value,
        name="--ssd-read-speed-max-chunk-mib",
        scale=1024**2,
        minimum=1 / 1024**2,
    )
    if (
        ssd_read_speed_bytes is None
        or ssd_read_speed_chunk is None
        or ssd_read_speed_max_chunk is None
    ):
        raise AssertionError("selected replay SSD read check defaults are invalid")
    checks: list[dict[str, object]] = []
    schema = payload.get("schema")
    schema_ok = schema == "largerlm.selected_replay.v1"
    checks.append(
        _selected_replay_check_record(
            "schema",
            schema_ok,
            "selected replay schema must be largerlm.selected_replay.v1",
            actual=schema,
        )
    )

    selected_result = payload.get("selected_result")
    selected_result_ok = isinstance(selected_result, str) and bool(selected_result)
    checks.append(
        _selected_replay_check_record(
            "selected_result_path",
            selected_result_ok,
            "selected replay must name a selected result JSON",
            actual=selected_result,
        )
    )

    summary: dict[str, object] | None = None
    result_error: str | None = None
    if selected_result_ok:
        try:
            summary = summarize_result_file(str(selected_result))
        except ResultSummaryError as exc:
            result_error = str(exc)
    checks.append(
        _selected_replay_check_record(
            "selected_result_loadable",
            summary is not None,
            "selected result JSON must still be readable and summarizable",
            error=result_error,
        )
    )

    stored_binding = (
        payload.get("launch_binding")
        if isinstance(payload.get("launch_binding"), dict)
        else {}
    )
    current_binding = (
        summary.get("launch_binding")
        if isinstance(summary, dict)
        and isinstance(summary.get("launch_binding"), dict)
        else {}
    )
    stored_prefill_plan_signature = (
        payload.get("selected_prefill_plan_signature")
        if isinstance(payload.get("selected_prefill_plan_signature"), dict)
        else None
    )
    current_prefill_plan_signature = (
        summary.get("prefill_plan_signature")
        if isinstance(summary, dict)
        and isinstance(summary.get("prefill_plan_signature"), dict)
        else None
    )
    if stored_prefill_plan_signature is not None:
        checks.append(
            _selected_replay_check_record(
                "selected_prefill_plan_signature_stable",
                stored_prefill_plan_signature == current_prefill_plan_signature,
                "selected replay prefill plan signature must match the selected result",
                stored=stored_prefill_plan_signature,
                current=current_prefill_plan_signature,
            )
        )
    stored_argv = _selected_replay_argv(payload.get("argv"))
    stored_binding_argv = _selected_replay_argv(
        stored_binding.get("replay_generate_token_ids_argv")
    )
    current_argv = _selected_replay_argv(
        current_binding.get("replay_generate_token_ids_argv")
    )
    checks.extend(
        [
            _selected_replay_check_record(
                "stored_replay_ready",
                stored_binding.get("replay_ready") is True,
                "stored launch binding must be replay-ready",
                actual=stored_binding.get("replay_ready"),
            ),
            _selected_replay_check_record(
                "stored_replay_files_ready",
                stored_binding.get("replay_files_ready") is True,
                "stored launch binding must have been file-ready when written",
                actual=stored_binding.get("replay_files_ready"),
            ),
            _selected_replay_check_record(
                "current_replay_ready",
                current_binding.get("replay_ready") is True,
                "selected result must still be replay-ready",
                actual=current_binding.get("replay_ready"),
            ),
            _selected_replay_check_record(
                "current_replay_files_ready",
                current_binding.get("replay_files_ready") is True,
                "selected result replay files and audit binding must still be current",
                actual=current_binding.get("replay_files_ready"),
            ),
            _selected_replay_check_record(
                "current_launch_profile_sha256_matches_file",
                current_binding.get("launch_profile_sha256_matches_file") is True,
                "current launch profile file hash must match the measured result",
                actual=current_binding.get("launch_profile_sha256_matches_file"),
            ),
            _selected_replay_check_record(
                "current_launch_audit_binding_matches",
                current_binding.get("launch_audit_binding_matches") is True,
                "current launch audit must be passing and bound to the same profile",
                actual=current_binding.get("launch_audit_binding_matches"),
            ),
            _selected_replay_check_record(
                "current_launch_audit_profile_sha256_matches_file",
                current_binding.get("launch_audit_profile_sha256")
                == current_binding.get("launch_profile_file_sha256"),
                "current launch audit profile hash must match the current profile file",
                audit=current_binding.get("launch_audit_profile_sha256"),
                file=current_binding.get("launch_profile_file_sha256"),
            ),
            _selected_replay_check_record(
                "argv_valid",
                stored_argv is not None,
                "selected replay argv must be a non-empty string array",
            ),
            _selected_replay_check_record(
                "argv_matches_stored_binding",
                stored_argv is not None and stored_argv == stored_binding_argv,
                "selected replay argv must match the stored launch binding argv",
            ),
            _selected_replay_check_record(
                "argv_matches_current_binding",
                stored_argv is not None and stored_argv == current_argv,
                "selected replay argv must match the current launch binding argv",
            ),
            _selected_replay_check_record(
                "launch_profile_path_stable",
                stored_binding.get("launch_profile_path")
                == current_binding.get("launch_profile_path"),
                "stored and current launch profile paths must match",
                stored=stored_binding.get("launch_profile_path"),
                current=current_binding.get("launch_profile_path"),
            ),
            _selected_replay_check_record(
                "launch_audit_path_stable",
                stored_binding.get("launch_audit_path")
                == current_binding.get("launch_audit_path"),
                "stored and current launch audit paths must match",
                stored=stored_binding.get("launch_audit_path"),
                current=current_binding.get("launch_audit_path"),
            ),
            _selected_replay_check_record(
                "applied_launch_profile_sha256_stable",
                stored_binding.get("applied_launch_profile_sha256")
                == current_binding.get("applied_launch_profile_sha256"),
                "stored and current applied profile hashes must match",
                stored=stored_binding.get("applied_launch_profile_sha256"),
                current=current_binding.get("applied_launch_profile_sha256"),
            ),
            _selected_replay_check_record(
                "launch_profile_file_sha256_stable",
                stored_binding.get("launch_profile_file_sha256")
                == current_binding.get("launch_profile_file_sha256"),
                "current launch profile file hash must match the selected replay snapshot",
                stored=stored_binding.get("launch_profile_file_sha256"),
                current=current_binding.get("launch_profile_file_sha256"),
            ),
        ]
    )
    checks.extend(_selected_replay_current_memory_checks(current_binding))
    checks.extend(_selected_replay_current_disk_checks(current_binding))
    checks.extend(
        _selected_replay_current_ssd_read_speed_checks(
            current_binding,
            enabled=ssd_read_speed_enabled,
            min_ratio=ssd_read_speed_min_ratio,
            bytes_to_read=ssd_read_speed_bytes,
            chunk_bytes=ssd_read_speed_chunk,
            max_chunk_bytes=ssd_read_speed_max_chunk,
        )
    )
    checks.extend(_selected_replay_prefill_acceleration_checks(current_binding))
    required_environment = _selected_replay_required_environment(
        payload.get("required_environment")
    )
    ok = all(check.get("ok") is True for check in checks)
    return {
        "schema": "largerlm.selected_replay_check.v1",
        "selected_replay": str(path),
        "selected_role": payload.get("selected_role"),
        "selected_result": selected_result if selected_result_ok else None,
        "ok": ok,
        "checks": checks,
        "stored_launch_binding": stored_binding or None,
        "current_launch_binding": current_binding or None,
        "argv": stored_argv,
        "current_argv": current_argv,
        "command": shlex.join(stored_argv) if stored_argv is not None else None,
        "required_environment": required_environment,
        "selected_total_elapsed_seconds": payload.get(
            "selected_total_elapsed_seconds"
        ),
        "selected_prefill_plan_signature": stored_prefill_plan_signature,
        "current_selected_prefill_plan_signature": current_prefill_plan_signature,
        "pre_run_checks": payload.get("pre_run_checks")
        if isinstance(payload.get("pre_run_checks"), dict)
        else None,
    }


def _format_selected_replay_check_text(payload: dict[str, object]) -> str:
    lines = [
        f"selected replay: {payload.get('selected_replay')}",
        f"selected replay ok: {bool(payload.get('ok'))}",
        f"selected role: {payload.get('selected_role')}",
        f"selected result: {payload.get('selected_result')}",
    ]
    expected_signature = payload.get("selected_prefill_plan_signature")
    if isinstance(expected_signature, dict):
        backend_counts = expected_signature.get("linear_backend_counts")
        backend_text = ""
        if isinstance(backend_counts, dict) and backend_counts:
            backend_text = " backends=" + ",".join(
                f"{key}={backend_counts[key]}" for key in sorted(backend_counts)
            )
        lines.append(
            "selected prefill plan: "
            f"chunks={expected_signature.get('chunk_count')}x"
            f"{expected_signature.get('chunk_tokens')}"
            f"{backend_text}"
        )
    selected_elapsed = payload.get("selected_total_elapsed_seconds")
    if isinstance(selected_elapsed, (int, float)) and not isinstance(
        selected_elapsed, bool
    ):
        parsed_elapsed = float(selected_elapsed)
        if math.isfinite(parsed_elapsed):
            lines.append(f"selected elapsed: {parsed_elapsed:.3f}s")
    required_environment = _selected_replay_required_environment(
        payload.get("required_environment")
    )
    if required_environment:
        lines.append(
            "required env: "
            + " ".join(
                f"{key}={required_environment[key]}"
                for key in sorted(required_environment)
            )
        )
    binding = (
        payload.get("current_launch_binding")
        if isinstance(payload.get("current_launch_binding"), dict)
        else {}
    )
    if binding:
        lines.append(
            "current launch: "
            f"profile={binding.get('launch_profile_path')} "
            f"audit={binding.get('launch_audit_path')} "
            f"replay_ready={bool(binding.get('replay_ready'))} "
            f"files_ready={bool(binding.get('replay_files_ready'))} "
            f"audit_ok={bool(binding.get('launch_audit_ok'))} "
            f"audit_bound={bool(binding.get('launch_audit_binding_matches'))}"
        )
    checks = payload.get("checks")
    if isinstance(checks, list):
        failed = [
            check
            for check in checks
            if isinstance(check, dict) and check.get("ok") is not True
        ]
        lines.append(f"failed checks: {len(failed)}")
        for check in failed:
            lines.append(
                "  "
                f"{check.get('code')}: {check.get('message')}"
            )
            detail = _format_selected_replay_failed_check_detail(check)
            if detail:
                lines.append(f"    {detail}")
    return "\n".join(lines)


def _format_check_float(value: object, *, digits: int = 3) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return f"{parsed:.{digits}f}"


def _format_check_bytes(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return f"{format_bytes(value)} ({value} bytes)"


def _format_selected_replay_failed_check_detail(
    check: dict[str, object],
) -> str | None:
    code = check.get("code")
    parts: list[str] = []
    if code == "current_ssd_read_speed_ok":
        actual = _format_check_float(check.get("actual_gib_per_second"))
        required = _format_check_float(check.get("required_gib_per_second"))
        baseline = _format_check_float(check.get("baseline_gib_per_second"))
        ratio = _format_check_float(check.get("actual_ratio"))
        min_ratio = _format_check_float(check.get("min_ratio"))
        if actual is not None:
            parts.append(f"actual={actual}GiB/s")
        if required is not None:
            parts.append(f"required={required}GiB/s")
        if baseline is not None:
            parts.append(f"baseline={baseline}GiB/s")
        if ratio is not None:
            parts.append(f"ratio={ratio}x")
        if min_ratio is not None:
            parts.append(f"min_ratio={min_ratio}x")
        for key, label in (
            ("requested_bytes", "requested"),
            ("measured_bytes", "measured"),
            ("chunk_bytes", "chunk"),
        ):
            rendered = _format_check_bytes(check.get(key))
            if rendered is not None:
                parts.append(f"{label}={rendered}")
        path = check.get("benchmark_path")
        if isinstance(path, str) and path:
            parts.append(f"path={path}")
        if check.get("short_read") is True:
            parts.append("short_read=True")
        for key in ("manifest_error", "benchmark_error"):
            value = check.get(key)
            if isinstance(value, str) and value:
                parts.append(f"{key}={value}")
    elif code == "current_available_memory_meets_audit_requirement":
        for key, label in (
            ("current_system_available_memory_bytes", "current_available"),
            ("required_available_memory_bytes", "required_available"),
        ):
            rendered = _format_check_bytes(check.get(key))
            if rendered is not None:
                parts.append(f"{label}={rendered}")
        source = check.get("current_system_memory_probe_source")
        if isinstance(source, str) and source:
            parts.append(f"source={source}")
    elif code == "current_stage_temp_disk_meets_audit_requirement":
        for key, label in (
            ("current_free_bytes", "current_free"),
            ("required_free_bytes", "required_free"),
            ("current_total_bytes", "total"),
        ):
            rendered = _format_check_bytes(check.get(key))
            if rendered is not None:
                parts.append(f"{label}={rendered}")
        path = check.get("path")
        if isinstance(path, str) and path:
            parts.append(f"path={path}")
        probe_path = check.get("probe_path")
        if isinstance(probe_path, str) and probe_path and probe_path != path:
            parts.append(f"probe_path={probe_path}")
    return "; ".join(parts) if parts else None


def _selected_replay_check_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "check_ssd_read_speed": getattr(args, "check_ssd_read_speed", None),
        "ssd_read_speed_min_ratio": getattr(
            args,
            "ssd_read_speed_min_ratio",
            None,
        ),
        "ssd_read_speed_bytes_mib": getattr(
            args,
            "ssd_read_speed_bytes_mib",
            None,
        ),
        "ssd_read_speed_chunk_mib": getattr(
            args,
            "ssd_read_speed_chunk_mib",
            None,
        ),
        "ssd_read_speed_max_chunk_mib": getattr(
            args,
            "ssd_read_speed_max_chunk_mib",
            None,
        ),
    }


def _selected_replay_check(args: argparse.Namespace) -> int:
    payload = _selected_replay_check_payload(
        args.selected_replay,
        **_selected_replay_check_kwargs(args),
    )
    if args.json:
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        print(_format_selected_replay_check_text(payload))
    return 0 if payload.get("ok") is True else 1


def _selected_replay_request_profile_argv(
    payload: dict[str, object],
    argv: list[str],
) -> tuple[list[str], list[str]]:
    binding = (
        payload.get("current_launch_binding")
        if isinstance(payload.get("current_launch_binding"), dict)
        else payload.get("stored_launch_binding")
        if isinstance(payload.get("stored_launch_binding"), dict)
        else {}
    )
    audit_path = binding.get("launch_audit_path") if isinstance(binding, dict) else None
    if not isinstance(audit_path, str) or not audit_path:
        return argv, []
    audit = _read_json_object_file(Path(audit_path), "launch audit")
    request_profile = audit.get("request_launch_profile")
    if not isinstance(request_profile, dict):
        return argv, []
    request_check = audit.get("request_check")
    request_linear_backend = (
        request_check.get("prefill_linear_backend")
        if isinstance(request_check, dict)
        else None
    )
    request_uses_auto_backend_policy = _request_prefill_backend_is_auto(
        request_linear_backend
    )
    appended: list[str] = []
    resolved = list(argv)
    for flag, profile_value in _launch_profile_argv_pairs(request_profile):
        if flag == "--prefill-linear-backend" and request_uses_auto_backend_policy:
            continue
        if flag in resolved:
            continue
        if flag not in _LAUNCH_PROFILE_ALLOWED_FLAGS:
            raise CliArgumentError(
                f"launch audit request profile flag {flag!r} is not supported"
            )
        if profile_value is None:
            resolved.append(flag)
            appended.append(flag)
        else:
            rendered = str(profile_value)
            resolved.extend([flag, rendered])
            appended.extend([flag, rendered])
    return resolved, appended


def _selected_replay_lock_path(
    selected_replay_path: str | Path,
    payload: dict[str, object],
) -> Path:
    binding = (
        payload.get("launch_binding")
        if isinstance(payload.get("launch_binding"), dict)
        else {}
    )
    prepared_dir = binding.get("prepared_dir") if isinstance(binding, dict) else None
    if isinstance(prepared_dir, str) and prepared_dir:
        return Path(prepared_dir) / PREPARED_RUN_LOCK_FILE
    manifest = binding.get("prepared_manifest") if isinstance(binding, dict) else None
    if isinstance(manifest, str) and manifest:
        return Path(manifest).parent / PREPARED_RUN_LOCK_FILE
    return Path(selected_replay_path).parent / PREPARED_RUN_LOCK_FILE


def _acquire_selected_replay_run_lock(
    selected_replay_path: str | Path,
    payload: dict[str, object],
) -> PreparedRunLock:
    return acquire_prepared_run_lock_path(
        _selected_replay_lock_path(selected_replay_path, payload),
        busy_message="another selected replay is already running for this prepared package",
    )


def _acquire_direct_prepared_generation_lock(
    prepared: PreparedManifest,
) -> PreparedRunLock | None:
    path = prepared_run_lock_path_for_manifest(prepared.manifest_path)
    if prepared_run_lock_already_held(path):
        return None
    return acquire_prepared_run_lock_path(
        path,
        busy_message="another prepared generation is already running for this prepared package",
    )


def _selected_replay_run(args: argparse.Namespace) -> int:
    replay_lock: PreparedRunLock | None = None
    if not args.dry_run:
        selected_replay_payload = _read_json_object_file(
            Path(args.selected_replay),
            "selected replay",
        )
        replay_lock = _acquire_selected_replay_run_lock(
            args.selected_replay,
            selected_replay_payload,
        )
    try:
        payload = _selected_replay_check_payload(
            args.selected_replay,
            **_selected_replay_check_kwargs(args),
        )
        argv = _selected_replay_argv(payload.get("argv"))
        request_profile_argv: list[str] = []
        if argv is not None and payload.get("ok") is True:
            argv, request_profile_argv = _selected_replay_request_profile_argv(
                payload,
                argv,
            )
        if argv is not None and args.quiet_runner:
            if "--quiet-runner" not in argv:
                if "generate-prepared-token-ids" not in argv:
                    raise CliArgumentError(
                        "--quiet-runner is only supported for generate-prepared-token-ids replays"
                    )
                argv = [*argv, "--quiet-runner"]
        if argv is not None and args.write_result is not None:
            if "--write-result" in argv:
                raise CliArgumentError(
                    "selected replay argv already contains --write-result"
                )
            if "generate-prepared-token-ids" not in argv:
                raise CliArgumentError(
                    "--write-result is only supported for generate-prepared-token-ids replays"
                )
            argv = [*argv, "--write-result", str(args.write_result)]
        required_environment = _selected_replay_required_environment(
            payload.get("required_environment")
        )
        if args.json:
            json_payload = dict(payload)
            json_payload["dry_run"] = bool(args.dry_run)
            if args.quiet_runner:
                json_payload["quiet_runner"] = True
            if args.write_result is not None:
                json_payload["write_result_path"] = str(args.write_result)
            if request_profile_argv:
                json_payload["launch_audit_request_profile_argv"] = request_profile_argv
            if argv is not None and payload.get("ok") is True:
                json_payload["run_argv"] = list(argv)
                json_payload["run_command"] = shlex.join(argv)
                if required_environment:
                    json_payload["run_environment"] = dict(required_environment)
            print(
                json.dumps(json_payload, default=_json_default, indent=2, sort_keys=True)
            )
        elif args.dry_run or payload.get("ok") is not True:
            print(_format_selected_replay_check_text(payload))
        if payload.get("ok") is not True:
            return 1
        if argv is None:
            raise CliArgumentError("selected replay argv is missing or invalid")
        if args.dry_run:
            if not args.json:
                if required_environment:
                    print(
                        "env: "
                        + " ".join(
                            f"{key}={required_environment[key]}"
                            for key in sorted(required_environment)
                        )
                    )
                print("command: " + shlex.join(argv))
            return 0
        if replay_lock is not None:
            os.environ[PREPARED_RUN_LOCK_ENV] = str(replay_lock.path.resolve())
        for key, value in required_environment.items():
            os.environ[key] = value
        os.execvp(argv[0], argv)
        return 1
    finally:
        if replay_lock is not None:
            replay_lock.close()


def _prepared_token_generation_result_payload(
    *,
    prepared: PreparedManifest,
    max_new_tokens: int,
    launch_audit_path: str | Path | None,
    prefill_mla_kv_b_cache_dir: str | Path | None,
    result: TokenGenerationResult,
) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt_token_ids": tuple(result.prompt_token_ids),
        "max_new_tokens": int(max_new_tokens),
    }
    if launch_audit_path is not None:
        request["launch_audit_path"] = str(launch_audit_path)
    _add_prefill_mla_kv_b_cache_request_fields(
        request,
        prefill_mla_kv_b_cache_dir,
    )
    return {
        "schema": "largerlm.prepared_token_generation_result.v1",
        "source": "generate_prepared_token_ids",
        "prepared_manifest": str(prepared.manifest_path),
        "request": request,
        "token_result": result,
    }


def _prepared_text_generation_result_payload(
    *,
    prepared: PreparedManifest,
    max_new_tokens: int,
    launch_audit_path: str | Path | None,
    prefill_mla_kv_b_cache_dir: str | Path | None,
    result: TextGenerationResult,
) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt_token_ids": tuple(result.prompt_token_ids),
        "max_new_tokens": int(max_new_tokens),
    }
    if launch_audit_path is not None:
        request["launch_audit_path"] = str(launch_audit_path)
    _add_prefill_mla_kv_b_cache_request_fields(
        request,
        prefill_mla_kv_b_cache_dir,
    )
    return {
        "schema": "largerlm.prepared_text_generation_result.v1",
        "source": "generate_prepared_text",
        "prepared_manifest": str(prepared.manifest_path),
        "request": request,
        "text_result": result,
    }


def _prefill_mla_kv_b_cache_stats(path: str | Path) -> dict[str, int] | None:
    try:
        root = Path(path)
        if not root.is_dir():
            return None
        file_count = 0
        total_bytes = 0
        for item in root.rglob("*"):
            try:
                if not item.is_file():
                    continue
                file_count += 1
                total_bytes += item.stat().st_size
            except OSError:
                continue
        return {"file_count": file_count, "total_bytes": total_bytes}
    except OSError:
        return None


def _add_prefill_mla_kv_b_cache_request_fields(
    request: dict[str, object],
    path: str | Path | None,
) -> None:
    if path is None:
        return
    request["prefill_mla_kv_b_cache_dir"] = str(path)
    stats = _prefill_mla_kv_b_cache_stats(path)
    if stats is None:
        return
    request["prefill_mla_kv_b_cache_file_count"] = stats["file_count"]
    request["prefill_mla_kv_b_cache_total_bytes"] = stats["total_bytes"]


def _print_plan(plan: ModelPlan) -> None:
    cfg = plan.config
    print("LargerLM model plan")
    print(f"  model_type:             {cfg.model_type}")
    print(f"  layers:                 {cfg.num_hidden_layers}")
    print(f"  MoE layers:             {cfg.num_moe_layers}")
    print(f"  hidden size:            {cfg.hidden_size}")
    print(f"  MoE intermediate:       {cfg.moe_hidden_size}")
    print(f"  routed experts/layer:   {cfg.routed_experts}")
    print(f"  active experts/token:   {cfg.experts_per_token}")
    print(f"  quantization:           {plan.quant_bits}-bit affine, group={plan.group_size}")
    print(f"  one routed expert:      {format_bytes(plan.expert_layout.total_bytes)}")
    print(
        "  routed expert disk:     "
        f"{format_bytes(plan.routed_expert_disk_bytes_estimate)} (config estimate)"
    )
    print(
        "  decode read/token:      "
        f"{format_bytes(plan.routed_expert_read_bytes_per_decode_token)}"
    )
    if plan.cold_read_seconds_per_token is not None:
        tok_s = 1.0 / plan.cold_read_seconds_per_token
        print(
            "  cold I/O floor:         "
            f"{plan.cold_read_seconds_per_token * 1000:.1f} ms/token ({tok_s:.2f} tok/s)"
        )
    if plan.suggested_decode_guard_flags:
        argv = plan.suggested_decode_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested decode args:  {' '.join(str(item) for item in argv)}")
    if plan.suggested_launch_guard_flags:
        argv = plan.suggested_launch_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested launch guard: {' '.join(str(item) for item in argv)}")
    if plan.suggested_prepare_flags:
        argv = plan.suggested_prepare_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested prepare args: {' '.join(str(item) for item in argv)}")

    cache = plan.cache_estimate
    if cache.bytes_per_token is not None:
        print(f"  MLA/DSA KV per token:   {format_bytes(cache.bytes_per_token)}")
        if cache.mla_cache_width is not None:
            print(f"  MLA cache width/layer:  {cache.mla_cache_width}")
        if cache.indexer_full_layers is not None:
            print(f"  DSA full-index layers:  {cache.indexer_full_layers}")
        if cache.indexer_bytes_per_token:
            print(f"  DSA index per token:    {format_bytes(cache.indexer_bytes_per_token)}")

    if cfg.attention_q_projection_output_dim is not None:
        print(f"  attention Q output dim: {cfg.attention_q_projection_output_dim}")
    if cfg.attention_kv_a_output_dim is not None:
        print(f"  attention KV-A dim:     {cfg.attention_kv_a_output_dim}")
    if cfg.attention_kv_b_output_dim is not None:
        print(f"  attention KV-B dim:     {cfg.attention_kv_b_output_dim}")
    if cfg.attention_value_output_dim is not None:
        print(f"  attention value dim:    {cfg.attention_value_output_dim}")
    if cfg.dsa_full_indexer_q_output_dim is not None:
        print(f"  DSA indexer Q dim:      {cfg.dsa_full_indexer_q_output_dim}")

    if plan.checkpoint_stats:
        stats = plan.checkpoint_stats
        print("")
        print("Checkpoint scan")
        print(f"  tensors:                {stats.tensor_count}")
        print(f"  total bytes:            {format_bytes(stats.total_bytes)}")
        print(f"  routed expert bytes:    {format_bytes(stats.routed_expert_bytes)}")
        print(f"  resident bytes:         {format_bytes(stats.resident_bytes)}")
        for category, nbytes in stats.by_category.items():
            print(f"  {category:22s} {format_bytes(nbytes)}")

    if plan.page_cache_budget_bytes is not None:
        print("")
        print("Memory budget")
        if plan.unified_memory_bytes is not None:
            print(f"  unified memory:        {format_bytes(plan.unified_memory_bytes)}")
        print(f"  system reserve:        {format_bytes(plan.system_reserve_bytes)}")
        print(f"  runtime buffer:        {format_bytes(plan.runtime_buffer_bytes)}")
        if plan.resident_memory_budget_bytes is not None:
            print(
                "  resident budget:       "
                f"{format_bytes(plan.resident_memory_budget_bytes)}"
            )
        if plan.resident_memory_headroom_bytes is not None:
            print(
                "  resident headroom:     "
                f"{format_bytes(plan.resident_memory_headroom_bytes)}"
            )
        if plan.resident_memory_fits_budget is not None:
            print(
                "  resident budget fits:  "
                f"{'yes' if plan.resident_memory_fits_budget else 'no'}"
            )
        print(f"  target OS page cache:   {format_bytes(plan.page_cache_budget_bytes)}")
        if plan.resident_bytes_estimate is None:
            print("  resident weights:       unknown without safetensors scan")
        else:
            source = (
                f" ({plan.resident_bytes_estimate_source})"
                if plan.resident_bytes_estimate_source
                else ""
            )
            print(
                "  resident weights:       "
                f"{format_bytes(plan.resident_bytes_estimate)}{source}"
            )
        if plan.decode_cache_budget_bytes is not None:
            print(f"  decode cache budget:    {format_bytes(plan.decode_cache_budget_bytes)}")
        if plan.decode_cache_safe_context_tokens is not None:
            print(f"  safe cache context:     {plan.decode_cache_safe_context_tokens}")
    if plan.decode_cache_bytes_estimate is not None:
        print(
            f"  decode cache @ctx {plan.max_context_tokens}: "
            f"{format_bytes(plan.decode_cache_bytes_estimate)}"
        )
        if plan.decode_cache_fits_budget is not None:
            print(
                "  decode cache fits:      "
                f"{'yes' if plan.decode_cache_fits_budget else 'no'}"
            )


def _parse_layers(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return layers


def _dense_layers_from_config(cfg: Any | None) -> set[int]:
    if cfg is None:
        return set()
    return set(range(cfg.num_hidden_layers)) - set(cfg.moe_layers)


def _effective_dense_layers(args: argparse.Namespace, cfg: Any | None) -> set[int]:
    layers = _dense_layers_from_config(cfg)
    manual = _parse_layers(getattr(args, "dense_layers", None))
    if manual is not None:
        layers.update(manual)
    return layers


def _parse_token_ids(spec: str) -> tuple[int, ...]:
    tokens: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        token = int(part)
        if token < 0:
            raise ConfigError("token ids must be non-negative")
        tokens.append(token)
    if not tokens:
        raise ConfigError("at least one token id is required")
    return tuple(tokens)


def _parse_token_ids_text(text: str) -> tuple[int, ...]:
    stripped = text.strip()
    if not stripped:
        raise ConfigError("at least one token id is required")
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"failed to parse token id JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise ConfigError("token id JSON must be an array")
        tokens = payload
    else:
        normalized = stripped.replace(",", " ")
        tokens = normalized.split()
    parsed: list[int] = []
    for item in tokens:
        try:
            token = int(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid token id {item!r}") from exc
        if token < 0:
            raise ConfigError("token ids must be non-negative")
        parsed.append(token)
    if not parsed:
        raise ConfigError("at least one token id is required")
    return tuple(parsed)


def _read_batch_token_ids(args: argparse.Namespace) -> tuple[int, ...]:
    if args.token_ids and args.token_ids_file:
        raise ConfigError("--token-ids and --token-ids-file are mutually exclusive")
    if args.token_ids:
        return _parse_token_ids_text(args.token_ids)
    if args.token_ids_file:
        path = Path(args.token_ids_file)
        try:
            return _parse_token_ids_text(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"failed to read token ids file {path}: {exc}") from exc
    raise ConfigError("--token-ids or --token-ids-file is required")


def _parse_expert_ids(spec: str) -> tuple[int, ...]:
    experts: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        expert = int(part)
        if expert < 0:
            raise ExpertIOPlanError("expert ids must be non-negative")
        experts.append(expert)
    if not experts:
        raise ExpertIOPlanError("at least one expert id is required")
    return tuple(experts)


def _plan(args: argparse.Namespace) -> int:
    config = load_config(args.model)
    stats = None
    if args.scan_safetensors:
        moe_layers = set(config.moe_layers)
        stats = scan_checkpoint(
            args.model,
            category_fn=lambda name: categorize_tensor_for_moe_layers(
                name,
                moe_layers,
                num_hidden_layers=config.num_hidden_layers,
            ),
        )

    hw = detect_hardware()
    memory = args.unified_memory_gib
    memory_bytes = (
        _scaled_bytes_arg(memory, name="--unified-memory-gib", scale=1024**3)
        if memory is not None
        else hw.unified_memory_bytes
    )

    plan = build_plan(
        config,
        quant_bits=args.quant_bits,
        group_size=args.group_size,
        checkpoint_stats=stats,
        unified_memory_bytes=memory_bytes,
        system_reserve_bytes=(
            _scaled_bytes_arg(
                args.system_reserve_gib,
                name="--system-reserve-gib",
                scale=1024**3,
            )
            if args.system_reserve_gib is not None
            else None
        ),
        runtime_buffer_bytes=_scaled_bytes_arg(
            args.runtime_buffer_gib,
            name="--runtime-buffer-gib",
            scale=1024**3,
        ),
        target_page_cache_fraction=args.page_cache_fraction,
        cold_read_gib_per_second=args.cold_read_gib_s,
        max_context_tokens=args.max_context_tokens,
        max_cache_bytes=(
            _scaled_bytes_arg(args.max_cache_gib, name="--max-cache-gib", scale=1024**3)
            if args.max_cache_gib is not None
            else None
        ),
    )
    if args.write_launch_profile is not None:
        _write_launch_profile_file(
            args.write_launch_profile,
            plan.suggested_launch_profile,
        )
    if args.write_prepare_flags is not None:
        _write_plan_prepare_flags_file(
            args.write_prepare_flags,
            plan.suggested_prepare_flags,
        )

    if args.json:
        print(json.dumps(plan, default=_json_default, indent=2, sort_keys=True))
    else:
        if hw.chip_name:
            mem = (
                f"{hw.unified_memory_gib:.1f} GiB"
                if hw.unified_memory_gib is not None
                else "unknown memory"
            )
            cores = f", {hw.gpu_cores} GPU cores" if hw.gpu_cores else ""
            print(f"Hardware: {hw.chip_name}, {mem}{cores}")
            print("")
        _print_plan(plan)
    return 0


def _print_pack_report(report: PackReport) -> None:
    mode = "dry-run" if report.dry_run else "execute"
    print("LargerLM expert pack report")
    print(f"  mode:                  {mode}")
    print(f"  output dir:            {report.output_dir}")
    print(f"  chunk size:            {format_bytes(report.chunk_size)}")
    print(
        "  estimated heap peak:   "
        f"{format_bytes(report.estimated_peak_heap_bytes)} "
        f"/ limit {format_bytes(report.max_heap_bytes)}"
    )
    if report.raw_quantization_max_source_block_bytes:
        print(
            "  raw quant source blk:  "
            f"{format_bytes(report.raw_quantization_max_source_block_bytes)}"
        )
        print(
            "  raw quant output blk:  "
            f"{format_bytes(report.raw_quantization_max_output_block_bytes)}"
        )
        print(
            "  raw quant extra heap:  "
            f"{format_bytes(report.raw_quantization_extra_heap_bytes)}"
        )
        print(f"  raw quant rows/block:  {report.raw_quantization_max_rows_per_block}")
    print(f"  total output bytes:    {format_bytes(report.layout.total_bytes)}")
    print(
        "  disk available:        "
        f"{format_bytes(report.disk_budget.available_bytes)} "
        f"(margin {format_bytes(report.disk_budget.safety_margin_bytes)})"
    )
    print(f"  layers:                {len(report.layout.layers)}")
    print(f"  routed experts/layer:  {report.layout.num_experts}")
    print(f"  component order:       {', '.join(report.layout.component_order)}")
    print("")
    for layer in report.layout.layers:
        print(
            f"  {layer.layer_file}: layer={layer.layer}, "
            f"slot={format_bytes(layer.expert_slot_bytes)}, "
            f"file={format_bytes(layer.layer_file_bytes)}"
        )
    if report.dry_run:
        print("")
        print("No files were written. Re-run with --execute to pack experts.")


def _print_resident_report(report: ResidentPackReport) -> None:
    mode = "dry-run" if report.dry_run else "execute"
    print("LargerLM resident pack report")
    print(f"  mode:                  {mode}")
    print(f"  output dir:            {report.output_dir}")
    print(f"  weight file:           {report.layout.weight_file}")
    print(f"  alignment:             {report.layout.alignment}")
    print(f"  chunk size:            {format_bytes(report.chunk_size)}")
    print(
        "  estimated heap peak:   "
        f"{format_bytes(report.estimated_peak_heap_bytes)} "
        f"/ limit {format_bytes(report.max_heap_bytes)}"
    )
    print(f"  total output bytes:    {format_bytes(report.layout.total_bytes)}")
    print(
        "  disk available:        "
        f"{format_bytes(report.disk_budget.available_bytes)} "
        f"(margin {format_bytes(report.disk_budget.safety_margin_bytes)})"
    )
    print(f"  resident tensors:      {len(report.layout.tensors)}")
    by_category: dict[str, int] = {}
    for tensor in report.layout.tensors:
        by_category[tensor.category] = by_category.get(tensor.category, 0) + tensor.size
    for category, nbytes in sorted(by_category.items()):
        print(f"  {category:22s} {format_bytes(nbytes)}")
    if report.component_alias_source_tensor_count:
        print(
            "  resident w1/w3/w2:     "
            f"{report.component_alias_source_tensor_count} source -> "
            f"{report.component_alias_renamed_tensor_count} renamed "
            f"({format_bytes(report.component_alias_bytes)})"
        )
    if report.dry_run:
        print("")
        print("No files were written. Re-run with --execute to pack resident weights.")


def _pack_experts(args: argparse.Namespace) -> int:
    output = args.output
    if output is None:
        output = str(Path(args.model) / "experts")
    report = pack_experts(
        args.model,
        output,
        dry_run=not args.execute,
        force=args.force,
        chunk_size=_scaled_bytes_arg(args.chunk_mib, name="--chunk-mib", scale=1024**2),
        max_chunk_size=_scaled_bytes_arg(
            args.max_chunk_mib,
            name="--max-chunk-mib",
            scale=1024**2,
        ),
        max_heap_bytes=_scaled_bytes_arg(
            args.max_pack_heap_mib,
            name="--max-pack-heap-mib",
            scale=1024**2,
        ),
        disk_safety_margin_bytes=_scaled_bytes_arg(
            args.disk_margin_gib,
            name="--disk-margin-gib",
            scale=1024**3,
        ),
        layers=_parse_layers(args.layers),
        quantize_raw_to_int4=args.quantize_bf16_affine_int4,
        group_size=args.group_size,
    )

    if args.json:
        print(json.dumps(report, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_pack_report(report)
    return 0


def _pack_resident(args: argparse.Namespace) -> int:
    output = args.output
    if output is None:
        output = str(Path(args.model) / "resident")
    report = pack_resident_weights(
        args.model,
        output,
        dry_run=not args.execute,
        force=args.force,
        chunk_size=_scaled_bytes_arg(args.chunk_mib, name="--chunk-mib", scale=1024**2),
        max_chunk_size=_scaled_bytes_arg(
            args.max_chunk_mib,
            name="--max-chunk-mib",
            scale=1024**2,
        ),
        max_heap_bytes=_scaled_bytes_arg(
            args.max_pack_heap_mib,
            name="--max-pack-heap-mib",
            scale=1024**2,
        ),
        disk_safety_margin_bytes=_scaled_bytes_arg(
            args.disk_margin_gib,
            name="--disk-margin-gib",
            scale=1024**3,
        ),
        alignment=args.alignment,
    )

    if args.json:
        print(json.dumps(report, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_resident_report(report)
    return 0


def _print_mlx_preflight(report: MlxBaselinePreflight) -> None:
    print("LargerLM MLX baseline preflight")
    print(f"  model dir:             {report.model_dir}")
    print(f"  output dir:            {report.output_dir}")
    print(f"  model_type:            {report.model_type}")
    if report.checkpoint_total_bytes is None:
        print("  checkpoint bytes:      unknown")
    else:
        print(f"  checkpoint bytes:      {format_bytes(report.checkpoint_total_bytes)}")
    print(f"  load limit:            {format_bytes(report.max_model_load_bytes)}")
    print(f"  safe to load:          {report.safe_to_load}")
    print(f"  reason:                {report.reason}")
    print("")
    print("No model was loaded. Re-run with --execute after the preflight is safe.")


def _print_mlx_result(result: MlxBaselineResult) -> None:
    print("LargerLM MLX baseline export")
    print(f"  output dir:            {result.output_dir}")
    print(f"  generated tokens:      {list(result.generated_tokens)}")
    print(f"  tensors:               {result.tensor_count}")


def _print_baseline_validation(result: BaselineValidation) -> None:
    print("LargerLM baseline validation")
    print(f"  root:                  {result.root}")
    print(f"  tensors:               {result.tensor_count}")
    print(f"  total bytes:           {format_bytes(result.total_bytes)}")
    print(f"  ok:                    {result.ok}")


def _format_optional_bytes(value: int | None) -> str:
    return "unknown" if value is None else format_bytes(value)


def _format_artifact_probe_status(*, present: bool, valid: bool) -> str:
    if valid:
        return "valid"
    if present:
        return "present but invalid"
    return "missing"


def _print_checkpoint_artifact_status(status: CheckpointArtifactStatus) -> None:
    print("LargerLM checkpoint artifact status")
    print(f"  model dir:             {status.model_dir}")
    print(f"  config.json:           {'yes' if status.config_present else 'no'}")
    print(f"  safetensors index:     {'yes' if status.index_present else 'no'}")
    print(
        "  header manifest:      "
        f"{'valid' if status.header_manifest_valid else 'present but invalid' if status.header_manifest_present else 'missing'}"
    )
    if status.header_manifest_error:
        print(f"  header error:          {status.header_manifest_error}")
    if status.source_repo:
        print(f"  source repo:           {status.source_repo}")
        print(f"  source revision:       {status.source_revision}")
    print(f"  expected shards:       {status.expected_shard_count}")
    print(f"  present shards:        {status.present_shard_count}")
    print(f"  complete shards:       {status.complete_shard_count}")
    print(f"  missing shards:        {status.missing_shard_count}")
    print(f"  partial shards:        {status.partial_shard_count}")
    print(f"  extra shards:          {status.extra_shard_count}")
    print(
        "  expected tensor bytes: "
        f"{_format_optional_bytes(status.expected_tensor_bytes)}"
    )
    print(
        "  expected file bytes:   "
        f"{_format_optional_bytes(status.expected_safetensors_file_bytes)}"
    )
    print(
        "  remaining download:    "
        f"{_format_optional_bytes(status.remaining_safetensors_file_bytes)}"
    )
    print(
        "  present file bytes:    "
        f"{format_bytes(status.present_safetensors_file_bytes)}"
    )
    if status.download_disk_budget is not None:
        budget = status.download_disk_budget
        print(
            "  download disk need:    "
            f"{format_bytes(budget.required_bytes)} + margin "
            f"{format_bytes(budget.safety_margin_bytes)}"
        )
        print(f"  download disk free:    {format_bytes(budget.available_bytes)}")
        print(f"  download disk ok:      {status.download_disk_ok}")
    if status.local_header_check_requested:
        print(
            "  local headers:        "
            f"{status.local_header_ok_shard_count}/"
            f"{status.local_header_checked_shard_count} ok"
        )
        print(f"  local header errors:   {status.local_header_error_count}")
    print(f"  download complete:     {status.download_complete}")
    print(f"  complete proven:       {status.download_complete_proven}")
    print(f"  artifact clean:        {status.artifact_clean}")
    print(f"  metadata preflight:    {status.can_run_metadata_preflight}")
    print(f"  prepare attempt:       {status.can_attempt_prepare}")
    if status.prefill_backend_command or status.prefill_backend_report_present:
        print(
            "  prefill backend:      "
            f"{_format_artifact_probe_status(present=status.prefill_backend_report_present, valid=status.prefill_backend_report_valid)}"
        )
        if status.prefill_backend_report_error:
            print(f"  prefill backend error: {status.prefill_backend_report_error}")
    if status.prepare_execute_command or status.prepared_manifest_present:
        print(
            "  prepared manifest:    "
            f"{_format_artifact_probe_status(present=status.prepared_manifest_present, valid=status.prepared_manifest_valid)}"
        )
        if status.prepared_manifest_error:
            print(f"  prepared error:        {status.prepared_manifest_error}")
    if status.preflight_command or status.preflight_report_present:
        print(
            "  preflight report:     "
            f"{_format_artifact_probe_status(present=status.preflight_report_present, valid=status.preflight_report_valid)}"
        )
        if status.preflight_report_error:
            print(f"  preflight error:       {status.preflight_report_error}")
    if status.prepare_dry_run_command or status.prepare_dry_run_report_present:
        print(
            "  dry-run report:       "
            f"{_format_artifact_probe_status(present=status.prepare_dry_run_report_present, valid=status.prepare_dry_run_report_valid)}"
        )
        if status.prepare_dry_run_report_error:
            print(f"  dry-run error:         {status.prepare_dry_run_report_error}")
    if status.inspect_prepared_command or status.launch_profile_present:
        print(
            "  launch profile:       "
            f"{_format_artifact_probe_status(present=status.launch_profile_present, valid=status.launch_profile_valid)}"
        )
        if status.launch_profile_error:
            print(f"  launch profile error:  {status.launch_profile_error}")
    if status.launch_audit_command or status.launch_audit_present:
        print(
            "  launch audit:         "
            f"{_format_artifact_probe_status(present=status.launch_audit_present, valid=status.launch_audit_valid)}"
        )
        if status.launch_audit_error:
            print(f"  launch audit error:    {status.launch_audit_error}")
    if status.minimal_smoke_command or status.minimal_smoke_result_present:
        print(
            "  minimal smoke result: "
            f"{_format_artifact_probe_status(present=status.minimal_smoke_result_present, valid=status.minimal_smoke_result_valid)}"
        )
        if status.minimal_smoke_result_error:
            print(f"  smoke result error:    {status.minimal_smoke_result_error}")
    if status.issues:
        print("  issues:")
        for issue in status.issues[:8]:
            print(f"    - {issue}")
        if len(status.issues) > 8:
            print(f"    - +{len(status.issues) - 8} more")
    missing_or_partial = [
        shard for shard in status.shards if shard.issue in {"missing", "truncated", "size_mismatch"}
    ]
    if missing_or_partial:
        print("  missing/partial shards:")
        for shard in missing_or_partial[:8]:
            actual = _format_optional_bytes(shard.actual_file_bytes)
            expected = _format_optional_bytes(shard.expected_file_bytes)
            print(f"    - {shard.name}: {shard.issue} ({actual} / {expected})")
        if len(missing_or_partial) > 8:
            print(f"    - +{len(missing_or_partial) - 8} more")
    header_errors = [
        shard
        for shard in status.shards
        if shard.header_checked and shard.header_ok is False
    ]
    if header_errors:
        print("  header-check failures:")
        for shard in header_errors[:8]:
            print(f"    - {shard.name}: {shard.header_error}")
        if len(header_errors) > 8:
            print(f"    - +{len(header_errors) - 8} more")
    if status.hf_download_command:
        print(f"  download command:      {' '.join(status.hf_download_command)}")
    if status.header_fetch_command and not status.header_manifest_valid:
        print(f"  header command:        {' '.join(status.header_fetch_command)}")
    if status.download_precheck_command:
        print(f"  download precheck:     {' '.join(status.download_precheck_command)}")
    if status.post_copy_check_command:
        print(f"  post-copy check:       {' '.join(status.post_copy_check_command)}")
    if status.prefill_backend_command:
        print(f"  prefill backend:       {' '.join(status.prefill_backend_command)}")
    if status.preflight_command and status.can_run_metadata_preflight:
        print(f"  preflight command:     {' '.join(status.preflight_command)}")
    if status.prepare_dry_run_command and status.can_run_metadata_preflight:
        print(f"  prepare dry-run:       {' '.join(status.prepare_dry_run_command)}")
    if status.prepare_execute_command:
        print(f"  prepare execute:       {' '.join(status.prepare_execute_command)}")
    if status.inspect_prepared_command:
        print(f"  inspect prepared:      {' '.join(status.inspect_prepared_command)}")
    if status.launch_audit_command:
        print(f"  launch audit:          {' '.join(status.launch_audit_command)}")
    if status.minimal_smoke_command:
        print(f"  minimal smoke:         {' '.join(status.minimal_smoke_command)}")
    if status.bringup_plan:
        print("  bring-up plan:")
        for step in status.bringup_plan:
            state = (
                "complete"
                if step.step_status == "complete"
                else "ready"
                if step.command_available
                else f"blocked: {step.blocked_reason}"
            )
            tags = []
            if step.reads_weight_payloads:
                tags.append("reads-weights")
            if step.writes_artifacts:
                tags.append("writes")
            if step.runs_model:
                tags.append("runs-model")
            suffix = f" ({', '.join(tags)})" if tags else ""
            print(f"    - {step.step_id}: {state}{suffix}")
    if status.next_bringup_step is not None:
        print(f"  next bring-up:         {status.next_bringup_step.step_id}")


def _write_checkpoint_missing_shard_outputs(
    status: CheckpointArtifactStatus,
    *,
    json_path: str | None,
    urls_path: str | None,
) -> None:
    if json_path:
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                missing_shard_download_manifest(status),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if urls_path:
        entries = missing_shard_download_entries(status)
        missing_urls = [entry.name for entry in entries if entry.url is None]
        if missing_urls:
            preview = ", ".join(missing_urls[:4])
            raise CliArgumentError(
                "cannot write missing shard URL list without source repo URLs: "
                f"{preview}"
            )
        path = Path(urls_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(f"{entry.url}\n" for entry in entries if entry.url),
            encoding="utf-8",
        )


def _external_download_header_metadata(
    status: CheckpointArtifactStatus,
) -> dict[str, dict[str, object]]:
    manifest_path = status.model_dir / HEADER_MANIFEST_NAME
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    shards = loaded.get("shards")
    if not isinstance(shards, dict):
        return {}
    metadata: dict[str, dict[str, object]] = {}
    for shard_name, info in shards.items():
        if not isinstance(shard_name, str) or not isinstance(info, dict):
            continue
        data_start = info.get("data_start")
        header = info.get("header")
        if isinstance(data_start, bool) or not isinstance(data_start, int):
            continue
        if not isinstance(header, dict):
            continue
        metadata[shard_name] = {
            "expected_data_start": data_start,
            "expected_header": header,
        }
    return metadata


def _external_download_payload(status: CheckpointArtifactStatus) -> dict[str, object]:
    base = missing_shard_download_manifest(status)
    header_metadata = _external_download_header_metadata(status)
    entries = []
    for entry in missing_shard_download_entries(status):
        target = status.model_dir / entry.name
        payload = {
            "name": entry.name,
            "url": entry.url,
            "target_path": str(target),
            "issue": entry.issue,
            "expected_file_bytes": entry.expected_file_bytes,
            "actual_file_bytes": entry.actual_file_bytes,
            "remaining_file_bytes": entry.remaining_file_bytes,
        }
        payload.update(header_metadata.get(entry.name, {}))
        entries.append(payload)
    return {
        "schema": "largerlm.external_safetensors_download.v1",
        "version": 1,
        "model_dir": str(status.model_dir),
        "source": base["source"],
        "download_complete": status.download_complete,
        "download_complete_proven": status.download_complete_proven,
        "expected_safetensors_file_bytes": status.expected_safetensors_file_bytes,
        "present_safetensors_file_bytes": status.present_safetensors_file_bytes,
        "remaining_safetensors_file_bytes": status.remaining_safetensors_file_bytes,
        "download_disk_ok": status.download_disk_ok,
        "download_disk_budget": base["download_disk_budget"],
        "entry_count": len(entries),
        "entries": entries,
        "post_copy_check_command": (
            None
            if status.post_copy_check_command is None
            else list(status.post_copy_check_command)
        ),
    }


def _next_bringup_range_download_command(
    status: CheckpointArtifactStatus,
    *,
    external_download_json_path: str | None,
) -> tuple[str, ...] | None:
    step = status.next_bringup_step
    if (
        step is None
        or step.step_id != "download_weights"
        or external_download_json_path is None
    ):
        return None
    if not missing_shard_download_entries(status):
        return None
    return (
        "python",
        "scripts/download_safetensors_ranges.py",
        str(external_download_json_path),
        "--model-dir",
        str(status.model_dir),
        "--start-index",
        "1",
        "--end-index",
        "1",
        "--max-bytes",
        str(6 * 1024**3),
        "--chunk-mib",
        "32",
        "--http-retries",
        "5",
        "--retry-delay-seconds",
        "2",
        "--timeout-seconds",
        "60",
        "--json",
    )


def _write_checkpoint_external_download_outputs(
    status: CheckpointArtifactStatus,
    *,
    json_path: str | None,
    shell_path: str | None,
) -> None:
    if json_path:
        _write_json_file_atomic(json_path, _external_download_payload(status))
    if not shell_path:
        return
    entries = missing_shard_download_entries(status)
    missing_urls = [entry.name for entry in entries if entry.url is None]
    if missing_urls:
        preview = ", ".join(missing_urls[:4])
        raise CliArgumentError(
            "cannot write external download script without source repo URLs: "
            f"{preview}"
        )
    missing_sizes = [
        entry.name for entry in entries if entry.expected_file_bytes is None
    ]
    if missing_sizes:
        preview = ", ".join(missing_sizes[:4])
        raise CliArgumentError(
            "cannot write external download script without expected file sizes: "
            f"{preview}"
        )
    path = Path(shell_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# LargerLM external safetensors download script for "
        + shlex.quote(str(status.model_dir)),
        "# Generated by checkpoint-status; run on a machine with model download access.",
        "# Preferred on flaky links: use the bounded HTTP Range downloader from this repo:",
        "#   python scripts/download_safetensors_ranges.py "
        + shlex.quote(str(json_path or "<external-download.json>"))
        + " --model-dir \"$MODEL_DIR\" --chunk-mib 16 --http-retries 5",
        "MODEL_DIR=${LARGERLM_MODEL_DIR:-" + shlex.quote(str(status.model_dir)) + "}",
        "RETRIES=${LARGERLM_DOWNLOAD_RETRIES:-5}",
        "SLEEP_SECONDS=${LARGERLM_DOWNLOAD_RETRY_SLEEP_SECONDS:-5}",
        "START_INDEX=${LARGERLM_DOWNLOAD_START_INDEX:-1}",
        "END_INDEX=${LARGERLM_DOWNLOAD_END_INDEX:-999999}",
        "MAX_BYTES=${LARGERLM_DOWNLOAD_MAX_BYTES:-0}",
        "CONNECT_TIMEOUT=${LARGERLM_DOWNLOAD_CONNECT_TIMEOUT_SECONDS:-30}",
        "LOW_SPEED_LIMIT=${LARGERLM_DOWNLOAD_LOW_SPEED_LIMIT_BYTES_PER_SECOND:-1}",
        "LOW_SPEED_TIME=${LARGERLM_DOWNLOAD_LOW_SPEED_TIME_SECONDS:-60}",
        "PLANNED_BYTES=0",
        "validate_uint() {",
        "  case \"$2\" in ''|*[!0-9]*) echo \"$1 must be an unsigned integer\" >&2; exit 2 ;; esac",
        "}",
        "validate_uint LARGERLM_DOWNLOAD_START_INDEX \"$START_INDEX\"",
        "validate_uint LARGERLM_DOWNLOAD_END_INDEX \"$END_INDEX\"",
        "validate_uint LARGERLM_DOWNLOAD_MAX_BYTES \"$MAX_BYTES\"",
        "validate_uint LARGERLM_DOWNLOAD_CONNECT_TIMEOUT_SECONDS \"$CONNECT_TIMEOUT\"",
        "validate_uint LARGERLM_DOWNLOAD_LOW_SPEED_LIMIT_BYTES_PER_SECOND \"$LOW_SPEED_LIMIT\"",
        "validate_uint LARGERLM_DOWNLOAD_LOW_SPEED_TIME_SECONDS \"$LOW_SPEED_TIME\"",
        "if [ \"$START_INDEX\" -lt 1 ]; then echo 'LARGERLM_DOWNLOAD_START_INDEX must be >= 1' >&2; exit 2; fi",
        "if [ \"$END_INDEX\" -lt \"$START_INDEX\" ]; then echo 'LARGERLM_DOWNLOAD_END_INDEX must be >= start index' >&2; exit 2; fi",
        "if [ \"$CONNECT_TIMEOUT\" -lt 1 ]; then echo 'LARGERLM_DOWNLOAD_CONNECT_TIMEOUT_SECONDS must be >= 1' >&2; exit 2; fi",
        "if [ \"$LOW_SPEED_TIME\" -lt 1 ]; then echo 'LARGERLM_DOWNLOAD_LOW_SPEED_TIME_SECONDS must be >= 1' >&2; exit 2; fi",
        "actual_size() { wc -c < \"$1\" | tr -d '[:space:]'; }",
        "download_one() {",
        "  local url=\"$1\"",
        "  local out=\"$2\"",
        "  local expected=\"$3\"",
        "  mkdir -p \"$(dirname \"$out\")\"",
        "  local attempt actual",
        "  for attempt in $(seq 1 \"$RETRIES\"); do",
        "    if [ -f \"$out\" ]; then",
        "      actual=$(actual_size \"$out\")",
        "      if [ \"$actual\" = \"$expected\" ]; then",
        "        echo \"ok: $out ($actual bytes)\"",
        "        return 0",
        "      fi",
        "      if [ \"$actual\" -gt \"$expected\" ]; then",
        (
            "        echo \"refusing to resume oversized shard: "
            "$out ($actual > $expected)\" >&2"
        ),
        "        return 1",
        "      fi",
        "      if head -c 4096 \"$out\" | grep -a -q 'Unsupported content type'; then",
        "        echo \"retrying transient Unsupported content type body for $out\" >&2",
        "        rm -f \"$out\"",
        "      fi",
        "    fi",
        "    echo \"download: $out (attempt $attempt/$RETRIES)\"",
        "    if ! "
        + (
            "curl -L --fail --retry 5 --retry-delay 2 "
            "--connect-timeout \"$CONNECT_TIMEOUT\" "
            "--speed-limit \"$LOW_SPEED_LIMIT\" --speed-time \"$LOW_SPEED_TIME\" "
            "--continue-at - --output \"$out\" \"$url\""
        )
        + "; then",
        "      if [ \"$attempt\" = \"$RETRIES\" ]; then",
        "        echo \"curl failed for $out after $attempt attempts\" >&2",
        "        return 1",
        "      fi",
        "      echo \"curl failed for $out; retrying after $SLEEP_SECONDS seconds\" >&2",
        "      sleep \"$SLEEP_SECONDS\"",
        "      continue",
        "    fi",
        "    if [ ! -f \"$out\" ]; then",
        "      echo \"curl reported success but did not create $out\" >&2",
        "      return 1",
        "    fi",
        "    actual=$(actual_size \"$out\")",
        "    if [ \"$actual\" = \"$expected\" ]; then",
        "      echo \"ok: $out ($actual bytes)\"",
        "      return 0",
        "    fi",
        "    if head -c 4096 \"$out\" | grep -a -q 'Unsupported content type'; then",
        "      echo \"retrying transient Unsupported content type body for $out\" >&2",
        "      rm -f \"$out\"",
        "    elif [ \"$attempt\" = \"$RETRIES\" ]; then",
        (
            "      echo \"wrong shard size for $out: got $actual bytes, "
            "expected $expected\" >&2"
        ),
        "      return 1",
        "    else",
        "      echo \"partial shard remains for resume: $out ($actual / $expected bytes)\" >&2",
        "    fi",
        "    sleep \"$SLEEP_SECONDS\"",
        "  done",
        "}",
        "maybe_download_one() {",
        "  local index=\"$1\"",
        "  local url=\"$2\"",
        "  local out=\"$3\"",
        "  local expected=\"$4\"",
        "  local actual remaining",
        "  if [ \"$index\" -lt \"$START_INDEX\" ] || [ \"$index\" -gt \"$END_INDEX\" ]; then",
        "    echo \"skip by shard range: $out (#$index)\"",
        "    return 0",
        "  fi",
        "  remaining=\"$expected\"",
        "  if [ -f \"$out\" ]; then",
        "    actual=$(actual_size \"$out\")",
        "    if [ \"$actual\" = \"$expected\" ]; then",
        "      echo \"ok: $out ($actual bytes)\"",
        "      return 0",
        "    fi",
        "    if [ \"$actual\" -gt \"$expected\" ]; then",
        "      echo \"refusing to resume oversized shard: $out ($actual > $expected)\" >&2",
        "      return 1",
        "    fi",
        "    remaining=$((expected - actual))",
        "  fi",
        "  if [ \"$MAX_BYTES\" -gt 0 ] && [ $((PLANNED_BYTES + remaining)) -gt \"$MAX_BYTES\" ]; then",
        "    echo \"skip by byte cap: $out ($remaining bytes remaining, cap $MAX_BYTES)\"",
        "    return 0",
        "  fi",
        "  PLANNED_BYTES=$((PLANNED_BYTES + remaining))",
        "  download_one \"$url\" \"$out\" \"$expected\"",
        "}",
    ]
    if not entries:
        lines.extend(
            [
                "echo 'No missing or partial safetensors shards are currently reported.'",
                "exit 0",
            ]
        )
    else:
        for index, entry in enumerate(entries, start=1):
            assert entry.url is not None
            assert entry.expected_file_bytes is not None
            lines.append(
                "maybe_download_one "
                + str(index)
                + " "
                + shlex.quote(entry.url)
                + " "
                + '"$MODEL_DIR"/'
                + shlex.quote(entry.name)
                + " "
                + shlex.quote(str(entry.expected_file_bytes))
            )
        if status.post_copy_check_command is not None:
            lines.extend(
                [
                    "echo \"Download batch scheduled bytes: $PLANNED_BYTES\"",
                    "echo 'Downloads finished. On the target Mac, verify with:'",
                    "echo "
                    + shlex.quote(
                        " ".join(
                            shlex.quote(str(part))
                            for part in status.post_copy_check_command
                        )
                    ),
                ]
            )
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.chmod(0o755)
        tmp.replace(path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise CliArgumentError(
            f"failed to write external download script {path}: {exc}"
        ) from exc


def _bringup_next_payload(
    status: CheckpointArtifactStatus,
    *,
    command_override: tuple[str, ...] | None = None,
) -> dict[str, object]:
    step = status.next_bringup_step
    command = None if step is None else command_override or step.command
    payload_step = (
        None
        if step is None
        else replace(step, command=command)
        if command_override is not None
        else step
    )
    return {
        "version": 1,
        "model_dir": str(status.model_dir),
        "ready": step is not None,
        "step_id": None if step is None else step.step_id,
        "command": None if command is None else list(command),
        "next_bringup_step": payload_step,
        "reads_weight_payloads": False if step is None else step.reads_weight_payloads,
        "writes_artifacts": False if step is None else step.writes_artifacts,
        "runs_model": False if step is None else step.runs_model,
    }


def _write_checkpoint_next_bringup_outputs(
    status: CheckpointArtifactStatus,
    *,
    json_path: str | None,
    shell_path: str | None,
    command_override: tuple[str, ...] | None = None,
) -> None:
    if json_path:
        _write_json_file_atomic(
            json_path,
            _bringup_next_payload(status, command_override=command_override),
        )
    if not shell_path:
        return
    path = Path(shell_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    step = status.next_bringup_step
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# LargerLM next bring-up command for {shlex.quote(str(status.model_dir))}",
    ]
    if step is None or step.command is None:
        lines.extend(
            [
                "# No ready bring-up step is currently available.",
                "echo 'No ready LargerLM bring-up step is currently available.'",
                "exit 0",
            ]
        )
    else:
        command = command_override or step.command
        lines.extend(
            [
                f"# step_id: {step.step_id}",
                f"# reads_weight_payloads: {str(step.reads_weight_payloads).lower()}",
                f"# writes_artifacts: {str(step.writes_artifacts).lower()}",
                f"# runs_model: {str(step.runs_model).lower()}",
                "exec " + " ".join(shlex.quote(str(part)) for part in command),
            ]
        )
        if command_override is not None:
            lines.insert(-1, "# download transport: bounded_http_range")
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.chmod(0o755)
        tmp.replace(path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise CliArgumentError(
            f"failed to write next bring-up script {path}: {exc}"
        ) from exc


def _print_preflight_report(report: GlmPreflightReport) -> None:
    print("LargerLM GLM checkpoint preflight")
    print(f"  model dir:             {report.model_dir}")
    print(f"  model_type:            {report.model_type}")
    print(f"  hidden size:           {report.hidden_size}")
    print(f"  layers:                {report.num_hidden_layers}")
    print(f"  MoE layers:            {report.moe_layers}")
    print(f"  routed experts:        {report.routed_experts}")
    print(f"  active experts/token:  {report.experts_per_token}")
    if report.dsa_index_head_dim is not None:
        print(f"  DSA index head dim:    {report.dsa_index_head_dim}")
    if report.dsa_index_n_heads is not None:
        print(f"  DSA index heads:       {report.dsa_index_n_heads}")
    if report.dsa_index_topk is not None:
        print(f"  DSA index top-k:       {report.dsa_index_topk}")
    if report.dsa_full_indexer_q_output_dim is not None:
        print(f"  DSA indexer Q dim:     {report.dsa_full_indexer_q_output_dim}")
    print(
        "  public GLM-5.2 shape: "
        f"{'yes' if report.public_glm_5_2_shape.get('matches') is True else 'no'}"
    )
    quant = report.quantization
    if quant.detected:
        descriptor = quant.source or "unknown"
        if quant.method is not None:
            descriptor += f" ({quant.method})"
        print(f"  config quantization:   {descriptor}")
        if quant.bits is not None:
            print(
                "  config quant bits:     "
                f"{quant.bits} (target {quant.target_bits})"
            )
        if quant.group_size is not None:
            print(
                "  config quant group:    "
                f"{quant.group_size} (target {quant.target_group_size})"
            )
    else:
        print("  config quantization:   not declared")
    print(f"  checkpoint tensors:    {report.checkpoint_tensor_count}")
    print(f"  checkpoint total:      {format_bytes(report.checkpoint_total_bytes)}")
    print(f"  checkpoint routed:     {format_bytes(report.checkpoint_routed_bytes)}")
    print(f"  checkpoint resident:   {format_bytes(report.checkpoint_resident_bytes)}")
    if report.checkpoint_ignored_bytes:
        print(f"  checkpoint ignored:    {format_bytes(report.checkpoint_ignored_bytes)}")
    print(f"  hardware chip:         {report.hardware_chip_name}")
    if report.hardware_apple_silicon_generation is not None:
        tier = (
            f" {report.hardware_apple_silicon_tier}"
            if report.hardware_apple_silicon_tier is not None
            else ""
        )
        print(
            "  Apple Silicon:        "
            f"M{report.hardware_apple_silicon_generation}{tier}"
        )
    if report.hardware_gpu_cores is not None:
        print(f"  hardware GPU cores:    {report.hardware_gpu_cores}")
    print(
        "  detected memory:       "
        f"{format_bytes(report.hardware_unified_memory_bytes) if report.hardware_unified_memory_bytes is not None else 'unknown'}"
    )
    print(
        "  planning memory:       "
        f"{format_bytes(report.effective_unified_memory_bytes) if report.effective_unified_memory_bytes is not None else 'unknown'} "
        f"({report.effective_unified_memory_source})"
    )
    print(
        "  system reserve:        "
        f"{format_bytes(report.effective_system_reserve_bytes)}"
    )
    print("")
    cov = report.tensor_coverage
    print("Tensor coverage")
    print(f"  embedding:             {'yes' if cov.embedding else 'no'}")
    print(f"  final norm:            {'yes' if cov.final_norm else 'no'}")
    print(f"  lm_head:               {'yes' if cov.lm_head else 'no'}")
    print(f"  tied lm_head fallback: {'yes' if cov.tied_lm_head else 'no'}")
    print(
        "  attention layers:      "
        f"{cov.attention_layers_ok}/{cov.attention_layers_checked}"
    )
    print(f"  router layers:         {cov.router_layers_ok}/{cov.router_layers_checked}")
    if cov.router_bias_layers_checked:
        print(
            "  router corr bias:      "
            f"{cov.router_bias_layers_found}/{cov.router_bias_layers_checked}"
        )
    if cov.dense_layers_checked:
        print(
            "  dense layers:          "
            f"{cov.dense_layers_ok}/{cov.dense_layers_checked}"
        )
    if cov.shared_layers_checked:
        print(
            "  shared layers:         "
            f"{cov.shared_layers_ok}/{cov.shared_layers_checked}"
        )
    if cov.indexer_layers_checked:
        print(
            "  DSA indexer layers:    "
            f"{cov.indexer_layers_ok}/{cov.indexer_layers_checked}"
        )
    print("")
    print("Packed estimates")
    print(
        "  experts:               "
        f"{format_bytes(report.expert_coverage.packed_bytes_estimate)} "
        f"({report.expert_coverage.quantization_mode})"
    )
    print(
        "  max expert slot:       "
        f"{format_bytes(report.expert_coverage.max_expert_slot_bytes)}"
    )
    print(
        "  resident:              "
        f"{format_bytes(report.resident_coverage.packed_bytes_estimate)}"
    )
    print(
        "  max resident tensor:   "
        f"{format_bytes(report.resident_coverage.max_tensor_bytes)}"
    )
    if report.plan and report.plan.decode_cache_bytes_estimate is not None:
        print(
            "  decode cache:          "
            f"{format_bytes(report.plan.decode_cache_bytes_estimate)}"
        )
        if report.plan.decode_cache_safe_context_tokens is not None:
            print(
                "  safe cache context:    "
                f"{report.plan.decode_cache_safe_context_tokens}"
            )
    if report.disk_budget:
        print(
            "  disk required:         "
            f"{format_bytes(report.disk_budget.required_bytes)} "
            f"+ margin {format_bytes(report.disk_budget.safety_margin_bytes)}"
        )
        print(f"  disk available:        {format_bytes(report.disk_budget.available_bytes)}")
    print(
        "  generation live cap:   "
        f"{format_bytes(report.recommended_max_live_working_set_bytes)}"
    )
    print(
        "  min free memory guard: "
        f"{format_bytes(report.recommended_min_free_unified_memory_bytes)}"
    )
    if report.plan and report.plan.resident_memory_headroom_bytes is not None:
        print(
            "  resident headroom:     "
            f"{format_bytes(report.plan.resident_memory_headroom_bytes)}"
        )
    print("")
    print("Tokenizer")
    print(f"  found:                 {'yes' if report.tokenizer.found else 'no'}")
    print(f"  loaded:                {'yes' if report.tokenizer.loaded else 'no'}")
    if report.tokenizer.backend:
        print(f"  backend:               {report.tokenizer.backend}")
    if report.tokenizer.eos_token_id is not None:
        print(f"  eos token id:          {report.tokenizer.eos_token_id}")
    print("")
    if report.issues:
        print("Issues")
        for issue in report.issues:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    else:
        print("Issues")
        print("  none")
    print(f"  result:                {'ok' if report.ok else 'not ready'}")


def _print_prepare_report(report: PrepareGlmReport) -> None:
    print("LargerLM GLM prepare")
    print(f"  mode:                  {'execute' if report.executed else 'dry-run'}")
    print(f"  ok:                    {report.ok}")
    print(f"  output dir:            {report.paths.output_dir}")
    print(f"  experts dir:           {report.paths.experts_dir}")
    print(f"  resident dir:          {report.paths.resident_dir}")
    print(f"  cache layout:          {report.paths.cache_layout_path}")
    print(f"  cache file:            {report.paths.cache_file_path}")
    print(f"  manifest:              {report.paths.manifest_path}")
    if report.expert_pack:
        print(f"  expert bytes:          {format_bytes(report.expert_pack.layout.total_bytes)}")
    if report.resident_pack:
        print(f"  resident bytes:        {format_bytes(report.resident_pack.layout.total_bytes)}")
    if report.cache_layout:
        print(f"  cache context:         {report.cache_layout.max_context_tokens}")
        print(f"  decode cache bytes:    {format_bytes(report.cache_layout.total_bytes)}")
    if report.context_budget is not None:
        budget = report.context_budget
        print(
            "  auto cache context:    "
            f"{'yes' if budget.auto_context_from_budget else 'no'}"
        )
        if budget.decode_cache_budget_bytes is not None:
            print(
                "  decode cache budget:   "
                f"{format_bytes(budget.decode_cache_budget_bytes)}"
            )
        if budget.decode_cache_safe_context_tokens is not None:
            print(
                "  safe cache context:    "
                f"{budget.decode_cache_safe_context_tokens}"
            )
    if report.prepare_flags is not None:
        suffix = (
            f" sha256={report.prepare_flags.sha256[:12]}"
            if report.prepare_flags.sha256
            else ""
        )
        print(f"  prepare flags:         {report.prepare_flags.source}{suffix}")
    if report.public_glm_5_2_shape is not None:
        print(
            "  public GLM-5.2 shape: "
            f"{'yes' if report.public_glm_5_2_shape.get('matches') is True else 'no'}"
        )
    if report.prepare_disk_budget is not None:
        budget = report.prepare_disk_budget
        print(
            "  prepare disk need:   "
            f"{format_bytes(budget.required_bytes + budget.safety_margin_bytes)}"
        )
        print(f"  prepare disk avail:  {format_bytes(budget.available_bytes)}")
    if report.prepare_live_memory is not None:
        live = report.prepare_live_memory
        required = (
            live.estimated_live_working_set_bytes
            + live.min_available_memory_bytes
        )
        print(
            "  prepare live memory: "
            f"{format_bytes(live.estimated_live_working_set_bytes)}"
        )
        print(f"  prepare mem reserve: {format_bytes(live.min_available_memory_bytes)}")
        print(f"  prepare mem required:{format_bytes(required)}")
        if live.system_available_bytes is not None:
            print(
                "  prepare mem avail:   "
                f"{format_bytes(live.system_available_bytes)}"
            )
    if report.cold_read_benchmark is not None:
        bench = report.cold_read_benchmark
        print(f"  cold read file:        {bench.path}")
        print(f"  cold read measured:    {bench.gib_per_second:.3f} GiB/s")
        print(f"  cold read bytes:       {format_bytes(bench.measured_bytes)}")
    if report.glm_4bit_readiness is not None:
        print(
            "  GLM 4bit readiness:   "
            f"{'ok' if report.glm_4bit_readiness.get('ok') is True else 'not ready'}"
        )
    if report.preflight.issues:
        print("")
        print("Preflight issues")
        for issue in report.preflight.issues:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}")
    if not report.executed:
        print("")
        print("No files were written. Re-run with --execute after the dry-run looks sane.")
    print(f"  result:                {'ok' if report.ok else 'not ready'}")


def _print_applied_launch_profile(profile: object) -> None:
    if not isinstance(profile, dict):
        return
    path = profile.get("path")
    if path:
        print(f"  applied profile:       {path}")
    sha256 = profile.get("sha256")
    if sha256:
        print(f"  applied profile sha:   {sha256}")
    source = profile.get("source")
    if source:
        print(f"  applied profile src:   {source}")
    if profile.get("locked") is not None:
        print(f"  applied profile lock:  {profile.get('locked')}")
    if profile.get("lock_required") is True:
        print("  applied lock required: True")
    flag_count = profile.get("profile_flag_count")
    if isinstance(flag_count, int):
        print(f"  applied profile flags: {flag_count}")
    if profile.get("matches_prepared") is not None:
        print(f"  applied profile match: {profile.get('matches_prepared')}")


def _print_benchmark_result(result: GenerationBenchmark) -> None:
    print("LargerLM prepared generation benchmark")
    print(f"  manifest:              {result.prepared_manifest}")
    _print_applied_launch_profile(result.applied_launch_profile)
    print(f"  generated tokens:      {result.generated_tokens}")
    print(f"  elapsed:               {result.elapsed_seconds:.3f} s")
    print(f"  throughput:            {result.tokens_per_second:.3f} tok/s")
    print(f"  estimated read:        {format_bytes(result.estimated_read_bytes)}")
    print(
        "  estimated read BW:     "
        f"{result.estimated_read_gib_per_second:.3f} GiB/s"
    )
    print(f"  embedding read:        {format_bytes(result.estimated_embedding_read_bytes)}")
    print(f"  expert read:           {format_bytes(result.estimated_expert_read_bytes)}")
    print(f"  cache read:            {format_bytes(result.estimated_cache_read_bytes)}")
    print(f"  logits read:           {format_bytes(result.estimated_logits_read_bytes)}")
    if result.prompt_prefill_chunk_count:
        print(f"  prefill chunks:        {result.prompt_prefill_chunk_count}")
        print(f"  prefill chunk tokens:  {result.prompt_prefill_chunk_tokens}")
        print(
            "  prefill peak:          "
            f"{format_bytes(result.prompt_prefill_estimated_peak_bytes)}"
        )
        print(
            "  prefill stage bytes:   "
            f"{format_bytes(result.prompt_prefill_total_staged_bytes)}"
        )
        print(
            "  prefill compact stage: "
            f"{format_bytes(result.prompt_prefill_total_compact_stage_bytes)}"
        )
        print(
            "  compact materialized:  "
            f"{format_bytes(result.prompt_prefill_total_compact_stage_materialized_bytes)}"
        )
        if result.prompt_prefill_max_staged_bytes:
            print(
                "  max prefill stage:    "
                f"{format_bytes(result.prompt_prefill_max_staged_bytes)}"
            )
        if result.prompt_prefill_max_compact_stage_bytes:
            print(
                "  max compact stage:    "
                f"{format_bytes(result.prompt_prefill_max_compact_stage_bytes)}"
            )
        if result.prompt_prefill_total_stage_plus_compact_bytes:
            print(
                "  prefill stage+compact:"
                f" {format_bytes(result.prompt_prefill_total_stage_plus_compact_bytes)}"
            )
        if result.prompt_prefill_total_stage_plus_compact_materialized_bytes:
            print(
                "  stage+compact real:   "
                f"{format_bytes(result.prompt_prefill_total_stage_plus_compact_materialized_bytes)}"
            )
        if result.prompt_prefill_max_stage_plus_compact_bytes:
            print(
                "  max stage+compact:    "
                f"{format_bytes(result.prompt_prefill_max_stage_plus_compact_bytes)}"
            )
    if result.total_routed_expert_assignments:
        print(f"  routed assignments:    {result.total_routed_expert_assignments}")
        print(f"  routed unique slots:   {result.total_routed_unique_expert_slots}")
        print(f"  max unique/call:       {result.max_routed_unique_experts_per_call}")
        print(f"  max tokens/expert:     {result.max_routed_tokens_per_expert}")
    if result.total_expert_stage_planned_read_bytes:
        print(
            "  stage planned read:    "
            f"{format_bytes(result.total_expert_stage_planned_read_bytes)}"
        )
        if result.total_expert_stage_planned_read_seconds is not None:
            print(
                "  stage read time:       "
                f"{result.total_expert_stage_planned_read_seconds:.6g}s "
                f"@ {result.prefill_ssd_read_gib_per_second:.6g} GiB/s"
            )
            if result.prefill_max_routed_read_seconds > 0:
                print(
                    "  stage read cap:        "
                    f"{result.prefill_max_routed_read_seconds:.6g}s "
                    f"ok={result.total_expert_stage_read_seconds_ok}"
                )
        print(
            "  stage unique amp:      "
            f"{result.total_expert_stage_unique_read_amplification:.3f}x"
        )
    if result.total_expert_stage_read_advice_attempted_ranges:
        print(
            "  stage read advice:     "
            f"ranges={result.total_expert_stage_read_advice_attempted_ranges} "
            f"calls={result.total_expert_stage_read_advice_calls} "
            f"bytes={format_bytes(result.total_expert_stage_read_advice_bytes)} "
            f"failures={result.total_expert_stage_read_advice_failures}"
        )
    if result.max_effective_moe_token_block:
        print(f"  max moe block:         {result.max_effective_moe_token_block}")
    if result.max_moe_max_expert_tokens:
        print(f"  max moe expert tokens: {result.max_moe_max_expert_tokens}")
    if result.max_moe_batch_buffer_bytes:
        print(f"  max moe batch buffer:  {format_bytes(result.max_moe_batch_buffer_bytes)}")
    if result.max_moe_estimated_peak_bytes:
        print(f"  max moe runner peak:   {format_bytes(result.max_moe_estimated_peak_bytes)}")
    if result.prefill_static_capacity_per_expert is not None:
        print(f"  static cap request:    {result.prefill_static_capacity_per_expert}")
        print(f"  max static cap:        {result.max_static_capacity_per_expert}")
        print(f"  static used slots:     {result.total_static_capacity_used_slots}")
        print(f"  static total slots:    {result.total_static_capacity_slots}")
        print(f"  static overflow:       {result.total_static_capacity_overflow_assignments}")
        print(f"  static route bytes:    {format_bytes(result.total_static_capacity_binary_bytes)}")
    if result.token_result.runtime_guard is not None:
        live = result.token_result.runtime_guard.live_memory_budget
        print(
            "  preflight live peak:   "
            f"{format_bytes(live.estimated_live_working_set_bytes)}"
        )
        if live.max_live_working_set_bytes is not None:
            print(
                "  preflight live cap:    "
                f"{format_bytes(live.max_live_working_set_bytes)}"
            )
    if result.linear_backend_counts:
        backends = ", ".join(
            f"{backend}={count}"
            for backend, count in result.linear_backend_counts.items()
        )
        print(f"  linear backends:       {backends}")
    if result.prefill_acceleration_coverage:
        coverage = result.prefill_acceleration_coverage
        print(f"  prefill accel ok:      {coverage.get('ok')}")
        print(
            "  prefill accel mats:    "
            f"{coverage.get('accelerated_matrix_count')}/"
            f"{coverage.get('matrix_count')}"
        )
        total_flops = int(coverage.get("total_estimated_flops") or 0)
        if total_flops:
            accelerated_flops = int(coverage.get("accelerated_estimated_flops") or 0)
            fraction = float(coverage.get("accelerated_flop_fraction") or 0.0)
            print(
                "  prefill accel FLOPs:   "
                f"{accelerated_flops:,}/{total_flops:,} ({fraction:.1%})"
            )
    if result.prefill_acceleration_frontier:
        suggested = result.prefill_acceleration_frontier.get("suggested_guard_flags")
        if isinstance(suggested, dict):
            argv = suggested.get("argv")
            if isinstance(argv, (list, tuple)) and argv:
                print(
                    "  suggested accel args: "
                    f"{' '.join(str(item) for item in argv)}"
                )
    if result.total_linear_matrix_scratch_bytes:
        print(
            "  linear matrix scratch: "
            f"{format_bytes(result.total_linear_matrix_scratch_bytes)}"
        )
    if result.max_linear_matrix_scratch_bytes:
        print(
            "  max linear scratch:    "
            f"{format_bytes(result.max_linear_matrix_scratch_bytes)}"
        )
    if result.total_linear_matrix_f32_bytes:
        print(
            "  linear f32 matrix:     "
            f"{format_bytes(result.total_linear_matrix_f32_bytes)}"
        )
    if result.total_linear_matrix_raw_conversion_bytes:
        print(
            "  linear raw conversion: "
            f"{format_bytes(result.total_linear_matrix_raw_conversion_bytes)}"
        )
    if result.suggested_prefill_guard_flags:
        argv = result.suggested_prefill_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested prefill args: {' '.join(str(item) for item in argv)}")
    if result.suggested_guard_flags:
        argv = result.suggested_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested guard args: {' '.join(str(item) for item in argv)}")
    if result.suggested_stage_temp_guard_flags:
        argv = result.suggested_stage_temp_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested stage args: {' '.join(str(item) for item in argv)}")
    if result.suggested_decode_guard_flags:
        argv = result.suggested_decode_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested decode args: {' '.join(str(item) for item in argv)}")
    if result.suggested_launch_profile:
        argv = result.suggested_launch_profile.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested launch args: {' '.join(str(item) for item in argv)}")
    print("  result:                ok")


def _print_decode_cache_layout(layout: DecodeCacheLayout) -> None:
    print("LargerLM decode cache layout")
    print(f"  model_type:            {layout.model_type}")
    print(f"  max context tokens:    {layout.max_context_tokens}")
    print(f"  dtype:                 {layout.dtype}")
    print(f"  segments:              {len(layout.segments)}")
    print(f"  total bytes:           {format_bytes(layout.total_bytes)}")
    main = [segment for segment in layout.segments if segment.kind == "mla_kv"]
    index = [segment for segment in layout.segments if segment.kind == "dsa_index"]
    if main:
        print(f"  MLA layers:            {len(main)}")
        print(f"  MLA width/layer:       {main[0].width}")
    if index:
        print(f"  DSA index layers:      {len(index)}")
        print(f"  DSA index width:       {index[0].width}")
    print("  result:                ok")


def _print_decode_cache_init(result: DecodeCacheInitResult) -> None:
    print("LargerLM decode cache file")
    print(f"  layout json:           {result.layout_path}")
    print(f"  cache file:            {result.cache_file_path}")
    print(f"  sparse:                {result.sparse}")
    print(f"  existed before:        {result.existed}")
    print(f"  logical size:          {format_bytes(result.total_bytes)}")
    print(
        "  disk available:        "
        f"{format_bytes(result.disk_budget.available_bytes)} "
        f"(margin {format_bytes(result.disk_budget.safety_margin_bytes)})"
    )
    print("  result:                ok")


def _print_context1_o_proj_cache(result: Context1OProjCacheBuildResult) -> None:
    plan = result.plan
    print("LargerLM context=1 o_proj*B_v cache")
    print(f"  prepared:              {plan.prepared_dir}")
    print(f"  output dir:            {plan.output_dir}")
    print(f"  dtype:                 {plan.dtype}")
    print(f"  backend:               {result.backend}")
    print(f"  executed:              {result.executed}")
    print(f"  layers:                {len(plan.layers)}")
    selected = result.selected_build_report()
    if result.requested_build_layers is not None:
        if result.requested_build_next_layers is not None:
            print(f"  build next layers:     {result.requested_build_next_layers}")
        print(f"  build layers:          {len(result.requested_build_layers)}")
        print(f"  selected cache bytes:  {format_bytes(selected['cache_bytes'])}")
        print(f"  selected source bytes: {format_bytes(selected['source_bytes'])}")
        print(f"  selected build FMA:    {selected['fma_total']}")
        print(f"  max layer FMA:         {selected['max_layer_fma']}")
        print(f"  min --max-build-fma:   {selected['min_max_build_fma']}")
        print(
            "  min --max-build-gfma:  "
            f"{selected['min_max_build_fma'] / 1.0e9:.9g}"
        )
    print(
        "  dims:                  "
        f"hidden={plan.hidden_dim} value={plan.attention_value_dim} "
        f"kv_lora={plan.kv_lora_dim}"
    )
    print(f"  cache bytes:           {format_bytes(plan.total_bytes)}")
    if result.disk_budget is not None:
        print(
            "  disk available:        "
            f"{format_bytes(result.disk_budget.available_bytes)} "
            f"(margin {format_bytes(result.disk_budget.safety_margin_bytes)})"
        )
        print(f"  disk budget ok:        {result.disk_budget.ok}")
    if result.max_metal_builder_live_bytes is not None:
        print(
            "  builder live cap:      "
            f"{format_bytes(result.max_metal_builder_live_bytes)}"
        )
    print(
        "  current o_proj/token:  "
        f"{format_bytes(plan.current_o_proj_storage_per_token)}"
    )
    print(f"  build FMA:             {plan.fma_total}")
    print(
        "  max builder live:      "
        f"{format_bytes(selected['max_estimated_metal_builder_live_bytes'])}"
    )
    if result.executed:
        print(f"  completed layers:      {len(result.completed_layers)}")
        if result.skipped_layers:
            print(f"  skipped layers:        {len(result.skipped_layers)}")
        summary = result.layer_result_summary()
        if summary["measured_layer_count"]:
            print(f"  measured layers:       {summary['measured_layer_count']}")
            gfma_per_second = summary["measured_gfma_per_second"]
            if gfma_per_second is not None:
                print(f"  measured GFMA/s:       {gfma_per_second:.6f}")
            full_seconds = summary["estimated_full_build_seconds_from_measured"]
            if full_seconds is not None:
                print(f"  est full build sec:    {full_seconds:.3f}")
            remaining_seconds = summary[
                "estimated_remaining_build_seconds_from_measured"
            ]
            if remaining_seconds is not None:
                print(f"  est remaining sec:     {remaining_seconds:.3f}")
        print(f"  layout json:           {plan.cache_layout_path}")
        print(f"  cache file:            {plan.cache_file_path}")
        print(f"  elapsed seconds:       {result.elapsed_seconds:.6f}")
    else:
        print("  result:                dry-run ok")
        return
    print("  result:                ok")


def _print_context1_o_proj_cache_layout(layout: Context1OProjCacheLayout) -> None:
    print("LargerLM context=1 o_proj*B_v cache layout")
    print(f"  layout json:           {layout.layout_path}")
    print(f"  cache file:            {layout.cache_file_path}")
    print(f"  dtype:                 {layout.dtype}")
    print(f"  layers:                {len(layout.layers)}")
    print(
        "  dims:                  "
        f"hidden={layout.hidden_dim} value={layout.attention_value_dim} "
        f"kv_lora={layout.kv_lora_dim}"
    )
    print(f"  total bytes:           {format_bytes(layout.total_bytes)}")
    if layout.config_sha256 is not None:
        print(f"  config sha256:         {layout.config_sha256}")
    print("  result:                ok")


def _print_context1_o_proj_cache_progress(progress: Any) -> None:
    print(f"  progress file:         {progress.progress_path}")
    print(f"  progress exists:       {'yes' if progress.exists else 'no'}")
    if progress.exists:
        print(f"  progress complete:     {'yes' if progress.complete else 'no'}")
        print(f"  completed layers:      {len(progress.completed_layers)}")
        if progress.total_layers is not None:
            print(f"  total layers:          {progress.total_layers}")
        if progress.missing_layers:
            print(f"  missing layers:        {len(progress.missing_layers)}")
        if progress.next_missing_layer is not None:
            print(f"  next missing layer:    {progress.next_missing_layer}")
        if progress.backend:
            print(f"  backend:               {progress.backend}")
        suggestion = progress.suggested_resume_build or {}
        if isinstance(suggestion, dict) and suggestion.get("available"):
            dry_run_argv = suggestion.get("dry_run_argv")
            execute_argv = suggestion.get("execute_argv")
            if isinstance(dry_run_argv, (list, tuple)) and dry_run_argv:
                command = " ".join(shlex.quote(str(item)) for item in dry_run_argv)
                print(f"  resume dry-run:        python -m largerlm {command}")
            if isinstance(execute_argv, (list, tuple)) and execute_argv:
                command = " ".join(shlex.quote(str(item)) for item in execute_argv)
                print(f"  resume execute:        python -m largerlm {command}")
        elif isinstance(suggestion, dict) and suggestion.get("reason"):
            print(f"  resume suggestion:     unavailable ({suggestion['reason']})")
        summary = progress.layer_result_summary or {}
        if summary.get("measured_layer_count"):
            print(f"  measured layers:       {summary['measured_layer_count']}")
            gfma_per_second = summary.get("measured_gfma_per_second")
            if gfma_per_second is not None:
                print(f"  measured GFMA/s:       {gfma_per_second:.6f}")
            remaining_seconds = summary.get(
                "estimated_remaining_build_seconds_from_measured"
            )
            if remaining_seconds is not None:
                print(f"  est remaining sec:     {remaining_seconds:.3f}")


def _print_decode_layers_result(result: DecodeLayersResult) -> None:
    print("LargerLM decode layers")
    print(f"  expert layout:         {result.expert_layout_path}")
    print(f"  resident layout:       {result.resident_layout_path}")
    print(f"  cache layout:          {result.cache_layout_path}")
    print(f"  cache file:            {result.cache_file_path}")
    print(f"  input f32:             {result.input_path}")
    print(f"  output f32:            {result.output_path}")
    print(f"  position:              {result.position}")
    print(f"  context length:        {result.context_length}")
    print(f"  layers:                {','.join(str(layer) for layer in result.layers)}")
    if result.dense_layers:
        print(f"  dense layers:          {','.join(str(layer) for layer in result.dense_layers)}")
    print(f"  work dir:              {result.work_dir}")
    print(f"  kept work dir:         {result.kept_work_dir}")
    memory_inputs = sum(1 for record in result.records if record.input_in_memory)
    memory_outputs = sum(1 for record in result.records if record.output_in_memory)
    if memory_inputs or memory_outputs:
        print(
            "  in-memory links:       "
            f"inputs {memory_inputs}/{len(result.records)}, "
            f"outputs {memory_outputs}/{len(result.records)}"
        )
    if result.budgets:
        peak = max(b.estimated_peak_bytes for b in result.budgets)
        cache_read = max(b.decoder_cache_read_bytes for b in result.budgets)
        read_token = max(b.read_bytes_per_token for b in result.budgets)
        print(f"  max runner peak:       {format_bytes(peak)}")
        print(f"  max cache read/layer:  {format_bytes(cache_read)}")
        print(f"  max expert read/layer: {format_bytes(read_token)}")
    print("  result:                ok")


def _format_attention_stages(stages: object) -> str:
    if not isinstance(stages, dict) or not stages:
        return ""
    labels = (
        ("projections", "proj"),
        ("cache_write", "cachew"),
        ("rope", "rope"),
        ("dsa_indexer", "dsa"),
        ("mla_attention", "mla"),
        ("attention_output", "out"),
    )
    parts: list[str] = []
    for key, label in labels:
        value = stages.get(key)
        if isinstance(value, (int, float)) and value > 0:
            parts.append(f"{label}={float(value):.3f}s")
    return " ".join(parts)


def _format_mla_attention_timings(timings: object) -> str:
    if not isinstance(timings, dict) or not timings:
        return ""
    labels = (
        ("kernel", "mla_kernel"),
        ("kernel_weights", "mla_weights_kernel"),
        ("kernel_values", "mla_values_kernel"),
        ("value_read", "mla_value"),
        ("cache_read", "mla_cache"),
        ("metal_setup", "mla_setup"),
        ("input", "mla_input"),
        ("write", "mla_write"),
    )
    parts: list[str] = []
    for key, label in labels:
        value = timings.get(key)
        if isinstance(value, (int, float)) and value > 0:
            parts.append(f"{label}={float(value):.3f}s")
    return " ".join(parts)


def _format_mlp_stages(stages: object) -> str:
    if not isinstance(stages, dict) or not stages:
        return ""
    labels = (
        ("expert_kernel", "mlp_kernel"),
        ("expert_read", "mlp_read"),
        ("shared", "mlp_shared"),
        ("router", "mlp_router"),
        ("rmsnorm", "mlp_norm"),
        ("residual", "mlp_residual"),
        ("output", "mlp_output"),
    )
    parts: list[str] = []
    for key, label in labels:
        value = stages.get(key)
        if isinstance(value, (int, float)) and value > 0:
            parts.append(f"{label}={float(value):.3f}s")
    return " ".join(parts)


def _print_final_logits_result(result: FinalLogitsResult) -> None:
    print("LargerLM final logits")
    print(f"  resident layout:       {result.resident_layout_path}")
    print(f"  input f32:             {result.input_path}")
    print(f"  norm tensor:           {result.norm_tensor or 'skipped'}")
    print(f"  head tensor:           {result.head_tensor}")
    print(f"  hidden dim:            {result.hidden_dim}")
    print(f"  vocab size:            {result.vocab_size}")
    print(f"  dtype:                 {result.dtype}")
    print(f"  chunk rows:            {result.chunk_rows}")
    print(f"  chunks:                {result.chunks}")
    print(f"  read bytes:            {format_bytes(result.read_bytes)}")
    if result.output_logits_path:
        print(f"  logits f32:            {result.output_logits_path}")
    if result.output_topk_path:
        print(f"  top-k json:            {result.output_topk_path}")
    print("  top-k:")
    for record in result.topk:
        print(f"    {record.token_id}: {record.logit:.9g}")
    print("  result:                ok")


def _print_embedding_result(result: EmbeddingResult) -> None:
    print("LargerLM token embedding")
    print(f"  resident layout:       {result.resident_layout_path}")
    print(f"  tensor:                {result.tensor}")
    print(f"  token id:              {result.token_id}")
    print(f"  vocab size:            {result.vocab_size}")
    print(f"  hidden dim:            {result.hidden_dim}")
    print(f"  dtype:                 {result.dtype}")
    print(f"  read bytes:            {format_bytes(result.read_bytes)}")
    print(f"  output bytes:          {format_bytes(result.output_bytes)}")
    print(f"  output f32:            {result.output_path}")
    print("  result:                ok")


def _print_embedding_batch_result(result: EmbeddingBatchResult) -> None:
    print("LargerLM token embedding batch")
    print(f"  resident layout:       {result.resident_layout_path}")
    print(f"  tensor:                {result.tensor}")
    print(f"  token count:           {result.token_count}")
    print(f"  first token id:        {result.first_token_id}")
    print(f"  last token id:         {result.last_token_id}")
    print(f"  vocab size:            {result.vocab_size}")
    print(f"  hidden dim:            {result.hidden_dim}")
    print(f"  dtype:                 {result.dtype}")
    print(f"  read bytes:            {format_bytes(result.read_bytes)}")
    print(f"  output bytes:          {format_bytes(result.output_bytes)}")
    print(f"  output f32:            {result.output_path}")
    print("  result:                ok")


def _print_token_generation_result(result: TokenGenerationResult) -> None:
    print("LargerLM token-id generation")
    print(f"  prompt token ids:      {','.join(str(t) for t in result.prompt_token_ids)}")
    print(f"  generated token ids:   {','.join(str(t) for t in result.generated_token_ids)}")
    print(f"  max context tokens:    {result.max_context_tokens}")
    _print_applied_launch_profile(result.applied_launch_profile)
    print(f"  elapsed:               {result.elapsed_seconds:.3f} s")
    if result.generated_token_ids and result.elapsed_seconds > 0:
        print(
            "  generated throughput:  "
            f"{len(result.generated_token_ids) / result.elapsed_seconds:.3f} tok/s"
        )
    print(f"  estimated read:        {format_bytes(result.estimated_read_bytes)}")
    print(f"  embedding read:        {format_bytes(result.estimated_embedding_read_bytes)}")
    print(f"  expert read:           {format_bytes(result.estimated_expert_read_bytes)}")
    print(f"  cache read:            {format_bytes(result.estimated_cache_read_bytes)}")
    print(f"  logits read:           {format_bytes(result.estimated_logits_read_bytes)}")
    prefill_actual = getattr(result, "prefill_actual_read_time", None)
    if isinstance(prefill_actual, dict):
        print(
            "  prefill actual read:  "
            f"{format_bytes(int(prefill_actual.get('total_expert_stage_planned_read_bytes') or 0))}"
        )
        actual_seconds = prefill_actual.get("total_expert_stage_planned_read_seconds")
        if isinstance(actual_seconds, (int, float)):
            print(
                "  prefill read time:    "
                f"{float(actual_seconds):.6g}s "
                f"@ {float(prefill_actual.get('prefill_ssd_read_gib_per_second') or 0.0):.6g} GiB/s"
            )
            max_seconds = prefill_actual.get("prefill_max_routed_read_seconds")
            if isinstance(max_seconds, (int, float)) and float(max_seconds) > 0:
                print(
                    "  prefill read cap:     "
                    f"{float(max_seconds):.6g}s "
                    f"ok={prefill_actual.get('total_expert_stage_read_seconds_ok')}"
                )
    decode_actual = getattr(result, "decode_actual_read_time", None)
    if isinstance(decode_actual, dict):
        print(
            "  decode actual read:   "
            f"{format_bytes(int(decode_actual.get('actual_decode_routed_read_bytes') or 0))} "
            f"steps={decode_actual.get('decode_step_count')}"
        )
        actual_seconds = decode_actual.get("actual_decode_routed_read_seconds")
        if isinstance(actual_seconds, (int, float)):
            print(
                "  decode read time:     "
                f"{float(actual_seconds):.6g}s "
                f"@ {float(decode_actual.get('prefill_ssd_read_gib_per_second') or 0.0):.6g} GiB/s"
            )
            max_seconds = decode_actual.get("total_decode_max_routed_read_seconds")
            if isinstance(max_seconds, (int, float)):
                print(
                    "  decode read cap:      "
                    f"{float(max_seconds):.6g}s "
                    f"ok={decode_actual.get('total_decode_routed_read_seconds_ok')}"
                )
    if result.runtime_guard:
        print(
            "  preflight layer peak:  "
            f"{format_bytes(result.runtime_guard.max_layer_peak_bytes)}"
        )
        print(
            "  preflight cache read:  "
            f"{format_bytes(result.runtime_guard.max_layer_cache_read_bytes)}"
        )
        print(
            "  preflight logits peak: "
            f"{format_bytes(result.runtime_guard.final_logits_budget.estimated_peak_bytes)}"
        )
        print(
            "  preflight embed row:   "
            f"{format_bytes(result.runtime_guard.embedding_budget.row_bytes)}"
        )
        print(
            "  preflight embed out:   "
            f"{format_bytes(result.runtime_guard.embedding_budget.output_bytes)}"
        )
        live = result.runtime_guard.live_memory_budget
        print(
            "  preflight live peak:   "
            f"{format_bytes(live.estimated_live_working_set_bytes)}"
        )
        if live.max_live_working_set_bytes is not None:
            print(
                "  preflight live cap:    "
                f"{format_bytes(live.max_live_working_set_bytes)}"
            )
        if live.system_available_bytes is not None:
            print(
                "  system available mem:  "
                f"{format_bytes(live.system_available_bytes)}"
            )
    print(f"  work dir:              {result.work_dir}")
    print(f"  kept work dir:         {result.kept_work_dir}")
    for step in result.steps:
        top = ", ".join(
            f"{record.token_id}:{record.logit:.6g}" for record in step.topk
        )
        print(
            f"  step pos={step.position} input={step.input_token_id} "
            f"next={step.selected_token_id} "
            f"time={step.elapsed_seconds:.3f}s "
            f"logits={step.logits_elapsed_seconds:.3f}s "
            f"read={format_bytes(step.estimated_read_bytes)} topk=[{top}]"
        )
        if step.decode_layers:
            layer_summary = ", ".join(
                (
                    f"{record.layer}:{record.kind} "
                    f"time={record.elapsed_seconds:.3f}s "
                    f"attn={record.attention_elapsed_seconds:.3f}s "
                    f"mlp={record.mlp_elapsed_seconds:.3f}s "
                    f"mem={'i' if record.input_in_memory else '-'}"
                    f"{'o' if record.output_in_memory else '-'} "
                    f"{_format_attention_stages(getattr(record, 'attention_stage_elapsed_seconds', {}))} "
                    f"{_format_mla_attention_timings(getattr(record, 'mla_attention_timing_elapsed_seconds', {}))} "
                    f"{_format_mlp_stages(getattr(record, 'mlp_stage_elapsed_seconds', {}))} "
                    f"expert={format_bytes(record.expert_read_bytes)} "
                    f"cache={format_bytes(record.cache_read_bytes)}"
                ).replace("  ", " ")
                for record in step.decode_layers
            )
            print(f"    decode layers:       {layer_summary}")
    print("  result:                ok")


def _print_metal_decode_telemetry(result: MetalTokenGenerationResult) -> None:
    if any(result.decode_expert_bytes_read):
        print(
            "  decode expert bytes:   "
            f"{format_bytes(sum(result.decode_expert_bytes_read))}"
        )
    if any(result.decode_dense_mlp_bytes_read):
        print(
            "  decode dense bytes:    "
            f"{format_bytes(sum(result.decode_dense_mlp_bytes_read))}"
        )
    if any(result.decode_attn_projection_elapsed_seconds):
        print(
            "  decode attn proj:      "
            f"{sum(result.decode_attn_projection_elapsed_seconds):.3f} s"
        )
    if any(result.decode_mla_attention_elapsed_seconds):
        print(
            "  decode MLA attention:  "
            f"{sum(result.decode_mla_attention_elapsed_seconds):.3f} s"
        )
    if any(result.decode_attn_output_elapsed_seconds):
        print(
            "  decode attn output:    "
            f"{sum(result.decode_attn_output_elapsed_seconds):.3f} s"
        )
    if any(result.decode_attn_output_bytes_read):
        print(
            "  decode attn out read:  "
            f"{format_bytes(sum(result.decode_attn_output_bytes_read))} in "
            f"{sum(result.decode_attn_output_read_seconds):.3f} s"
        )
    if any(result.decode_attn_output_resident_mmap_backed_count):
        print(
            "  decode attn mmap:      "
            f"{sum(result.decode_attn_output_resident_mmap_backed_count)} layer hits"
        )
    if any(result.decode_router_bytes_read):
        print(
            "  decode router read:    "
            f"{format_bytes(sum(result.decode_router_bytes_read))} in "
            f"{sum(result.decode_router_read_seconds):.3f} s"
        )
    if any(result.decode_mlp_elapsed_seconds):
        print(
            "  decode MLP:            "
            f"{sum(result.decode_mlp_elapsed_seconds):.3f} s"
        )
    if any(result.decode_expert_read_seconds):
        print(
            "  decode expert read:    "
            f"{sum(result.decode_expert_read_seconds):.3f} s"
        )
    if any(result.decode_shared_bytes_read) or any(result.decode_shared_read_seconds):
        shared_prefetch = sum(result.decode_shared_prefetch_used_count)
        print(
            "  decode shared read:    "
            f"{format_bytes(sum(result.decode_shared_bytes_read))} in "
            f"{sum(result.decode_shared_read_seconds):.3f} s"
            + (
                f", {shared_prefetch} prefetched"
                if shared_prefetch
                else ""
            )
        )
    if any(result.decode_moe_mlp_kernel_seconds):
        print(
            "  decode MoE kernel:     "
            f"{sum(result.decode_moe_mlp_kernel_seconds):.3f} s"
        )
    if any(result.decode_moe_mlp_output_write_seconds):
        print(
            "  decode MoE write:      "
            f"{sum(result.decode_moe_mlp_output_write_seconds):.3f} s"
        )
    if any(result.decode_moe_mlp_overhead_seconds):
        print(
            "  decode MoE overhead:   "
            f"{sum(result.decode_moe_mlp_overhead_seconds):.3f} s"
        )
    if any(result.decode_layer_overhead_seconds):
        print(
            "  decode layer overhead: "
            f"{sum(result.decode_layer_overhead_seconds):.3f} s"
        )
    if any(result.decode_synchronous_wait_count_estimate):
        print(
            "  decode sync waits:     "
            f"{sum(result.decode_synchronous_wait_count_estimate)} est"
        )
    if (
        any(result.decode_attn_projection_synchronous_wait_count)
        or any(result.decode_attn_projection_async_submitted_count)
    ):
        print(
            "  decode attn proj waits:"
            f" {sum(result.decode_attn_projection_synchronous_wait_count)} waits, "
            f"{sum(result.decode_attn_projection_async_submitted_count)} async"
        )
    if (
        any(result.decode_dense_mlp_synchronous_wait_count)
        or any(result.decode_dense_mlp_async_submitted_count)
    ):
        print(
            "  decode dense waits:    "
            f"{sum(result.decode_dense_mlp_synchronous_wait_count)} waits, "
            f"{sum(result.decode_dense_mlp_async_submitted_count)} async"
        )
    if any(result.decode_moe_mlp_synchronous_wait_count):
        print(
            "  decode MoE waits:      "
            f"{sum(result.decode_moe_mlp_synchronous_wait_count)} / "
            f"{sum(result.decode_moe_mlp_command_buffer_count)}"
        )
    if any(result.decode_expert_read_task_count):
        print(
            "  decode read dispatch:  "
            f"{sum(result.decode_expert_read_task_count)} tasks, "
            f"{sum(result.decode_expert_read_pool_dispatch_count)} pooled, "
            f"{sum(result.decode_expert_read_serial_dispatch_count)} serial"
        )
    if result.mla_kv_b_cache_enabled is not None:
        state = "enabled" if result.mla_kv_b_cache_enabled else "disabled"
        print(f"  MLA KV-B cache:        {state}")
    if result.mla_kv_b_cache_current_bytes is not None:
        print(
            "  MLA KV-B cache bytes:  "
            f"{format_bytes(result.mla_kv_b_cache_current_bytes)}"
        )
    if any(result.decode_mla_value_cache_hit_count) or any(
        result.decode_mla_value_cache_store_count
    ):
        print(
            "  MLA KV-B cache hits:   "
            f"{sum(result.decode_mla_value_cache_hit_count)} hit, "
            f"{sum(result.decode_mla_value_cache_store_count)} store"
        )


def _print_metal_token_generation_result(result: MetalTokenGenerationResult) -> None:
    print("LargerLM Metal token-id generation")
    print(f"  prompt token ids:      {','.join(str(t) for t in result.prompt_token_ids)}")
    print(f"  generated token ids:   {','.join(str(t) for t in result.generated_token_ids)}")
    print(f"  prepared:              {result.prepared_dir}")
    print(f"  work dir:              {result.work_dir}")
    print(f"  kept work dir:         {result.kept_work_dir}")
    if result.prompt_prefill is not None:
        print(
            "  prompt prefill:        "
            f"{result.prompt_prefill_elapsed_seconds:.3f} s"
        )
    if result.prompt_prefill_estimated_live_working_set_bytes is not None:
        print(
            "  prefill live envelope: "
            f"{format_bytes(result.prompt_prefill_estimated_live_working_set_bytes)}"
        )
    if result.prefill_max_live_working_set_mib is not None:
        print(
            "  prefill live cap:      "
            f"{result.prefill_max_live_working_set_mib:.3f} MiB"
        )
    if result.prefill_final_logits_elapsed_seconds is not None:
        print(
            "  prefill logits:        "
            f"{result.prefill_final_logits_elapsed_seconds:.3f} s"
        )
    if result.metal_elapsed_seconds is not None:
        print(f"  metal elapsed:         {result.metal_elapsed_seconds:.3f} s")
    print(f"  elapsed:               {result.elapsed_seconds:.3f} s")
    if result.generated_token_ids and result.elapsed_seconds > 0:
        print(
            "  generated throughput:  "
            f"{len(result.generated_token_ids) / result.elapsed_seconds:.3f} tok/s"
        )
    _print_metal_decode_telemetry(result)
    print(f"  live envelope:         {format_bytes(result.estimated_live_working_set_bytes)}")
    print(f"  live cap:              {result.max_live_working_set_mib} MiB")
    print(f"  cache bytes:           {format_bytes(result.cache_total_bytes)}")
    print(f"  note:                  {result.note}")
    print("  result:                ok")


def _print_metal_text_generation_result(result: MetalTextGenerationResult) -> None:
    token_result = result.token_result
    print("LargerLM Metal text generation")
    print(f"  tokenizer:             {result.tokenizer_path}")
    print(f"  tokenizer backend:     {result.tokenizer_backend}")
    print(f"  prompt token ids:      {','.join(str(t) for t in result.prompt_token_ids)}")
    print(f"  generated token ids:   {','.join(str(t) for t in result.generated_token_ids)}")
    print(f"  generated text:        {result.generated_text!r}")
    print(f"  full text:             {result.full_text!r}")
    print(f"  work dir:              {token_result.work_dir}")
    print(f"  kept work dir:         {token_result.kept_work_dir}")
    if token_result.prompt_prefill is not None:
        print(
            "  prompt prefill:        "
            f"{token_result.prompt_prefill_elapsed_seconds:.3f} s"
        )
    if token_result.prompt_prefill_estimated_live_working_set_bytes is not None:
        print(
            "  prefill live envelope: "
            f"{format_bytes(token_result.prompt_prefill_estimated_live_working_set_bytes)}"
        )
    if token_result.prefill_max_live_working_set_mib is not None:
        print(
            "  prefill live cap:      "
            f"{token_result.prefill_max_live_working_set_mib:.3f} MiB"
        )
    if token_result.prefill_final_logits_elapsed_seconds is not None:
        print(
            "  prefill logits:        "
            f"{token_result.prefill_final_logits_elapsed_seconds:.3f} s"
        )
    if token_result.metal_elapsed_seconds is not None:
        print(f"  metal elapsed:         {token_result.metal_elapsed_seconds:.3f} s")
    print(f"  elapsed:               {token_result.elapsed_seconds:.3f} s")
    if result.generated_token_ids and token_result.elapsed_seconds > 0:
        print(
            "  generated throughput:  "
            f"{len(result.generated_token_ids) / token_result.elapsed_seconds:.3f} tok/s"
        )
    _print_metal_decode_telemetry(token_result)
    print(
        "  live envelope:         "
        f"{format_bytes(token_result.estimated_live_working_set_bytes)}"
    )
    print(f"  live cap:              {token_result.max_live_working_set_mib} MiB")
    print(f"  cache bytes:           {format_bytes(token_result.cache_total_bytes)}")
    print(f"  note:                  {token_result.note}")
    print("  result:                ok")


def _print_text_generation_result(result: TextGenerationResult) -> None:
    print("LargerLM text generation")
    print(f"  tokenizer:             {result.tokenizer_path}")
    print(f"  tokenizer backend:     {result.tokenizer_backend}")
    _print_applied_launch_profile(result.applied_launch_profile)
    print(f"  prompt token ids:      {','.join(str(t) for t in result.prompt_token_ids)}")
    print(f"  generated token ids:   {','.join(str(t) for t in result.generated_token_ids)}")
    print(f"  eos token id:          {result.eos_token_id}")
    print(f"  max context tokens:    {result.token_result.max_context_tokens}")
    print(f"  elapsed:               {result.token_result.elapsed_seconds:.3f} s")
    print(f"  estimated read:        {format_bytes(result.token_result.estimated_read_bytes)}")
    print(f"  work dir:              {result.token_result.work_dir}")
    print(f"  kept work dir:         {result.token_result.kept_work_dir}")
    print("")
    print(result.generated_text)
    print("")
    print("  result:                ok")


def _print_runtime_budget(budget: LayerRuntimeBudget) -> None:
    print("LargerLM runtime check")
    print(f"  expert layout:         {budget.expert_layout_path}")
    print(f"  resident layout:       {budget.resident_layout_path}")
    print(f"  layer:                 {budget.layer}")
    print(f"  layer kind:            {budget.layer_kind}")
    print(f"  hidden dim:            {budget.hidden_dim}")
    print(f"  intermediate dim:      {budget.intermediate_dim}")
    if budget.layer_kind == "moe":
        print(f"  routed experts:        {budget.num_experts}")
        print(f"  top-k:                 {budget.top_k} / max {budget.max_k}")
        print(f"  expert slot:           {format_bytes(budget.expert_slot_bytes)}")
        print(f"  aligned slot buffer:   {format_bytes(budget.aligned_slot_bytes)}")
        print(f"  router tensor:         {format_bytes(budget.router_bytes)}")
        print(f"  top-k read/token:      {format_bytes(budget.read_bytes_per_token)}")
        print(f"  include shared:        {budget.include_shared_expert}")
    if budget.include_shared_expert:
        print(f"  shared intermediate:   {budget.shared_intermediate_dim}")
        print(
            "  shared matrix peak:    "
            f"{format_bytes(budget.shared_max_aligned_matrix_bytes)}"
        )
    print(f"  include attention:     {budget.include_attention_projections}")
    if budget.include_attention_projections:
        print(f"  q lora dim:            {budget.attention_q_lora_dim}")
        print(f"  q output dim:          {budget.attention_q_output_dim}")
        print(f"  kv lora dim:           {budget.attention_kv_lora_dim}")
        print(f"  kv rope dim:           {budget.attention_kv_rope_dim}")
        print(f"  kv output dim:         {budget.attention_kv_output_dim}")
        print(
            "  attention read/token:  "
            f"{format_bytes(budget.attention_read_bytes_per_token)}"
        )
        print(
            "  attention matrix peak: "
            f"{format_bytes(budget.attention_max_aligned_matrix_bytes)}"
        )
    print(f"  include decoder layer: {budget.include_decoder_layer}")
    if budget.include_decoder_layer:
        print(f"  decoder context:       {budget.decoder_context_length}")
        print(f"  decoder heads:         {budget.decoder_num_heads}")
        print(f"  decoder qk nope dim:   {budget.decoder_qk_nope_dim}")
        print(f"  decoder rope dim:      {budget.decoder_rope_dim}")
        print(f"  decoder v head dim:    {budget.decoder_v_head_dim}")
        print(
            "  decoder cache read:    "
            f"{format_bytes(budget.decoder_cache_read_bytes)} "
            f"/ limit {format_bytes(budget.max_cache_read_bytes)}"
        )
    if budget.layer_kind == "moe":
        print(f"  router stage peak:     {format_bytes(budget.router_stage_peak_bytes)}")
        print(f"  MoE stage peak:        {format_bytes(budget.moe_stage_peak_bytes)}")
    else:
        print(f"  dense MLP peak:        {format_bytes(budget.moe_stage_peak_bytes)}")
    if budget.include_attention_projections:
        print(
            "  attention peak:        "
            f"{format_bytes(budget.attention_projection_peak_bytes)}"
        )
    if budget.include_decoder_layer:
        print(
            "  MLA attention peak:    "
            f"{format_bytes(budget.decoder_mla_attention_peak_bytes)}"
        )
        print(
            "  attn output peak:      "
            f"{format_bytes(budget.decoder_attention_output_peak_bytes)}"
        )
    print(
        "  estimated peak:        "
        f"{format_bytes(budget.estimated_peak_bytes)} "
        f"/ limit {format_bytes(budget.max_runner_scratch_bytes)}"
    )
    print("  result:                ok")


def _print_expert_io_plan(plan: ExpertIOPlan) -> None:
    print("LargerLM expert I/O plan")
    print(f"  expert layout:         {plan.expert_layout_path}")
    print(f"  layer:                 {plan.layer}")
    print(f"  layer file:            {plan.layer_file_path}")
    print(f"  routed experts:        {plan.num_experts}")
    print(f"  selected experts:      {','.join(str(e) for e in plan.selected_experts)}")
    print(f"  expert slot:           {format_bytes(plan.expert_slot_bytes)}")
    print(f"  merge gap:             {format_bytes(plan.merge_gap_bytes)}")
    print(f"  alignment:             {format_bytes(plan.align_bytes)}")
    print(f"  requested bytes:       {format_bytes(plan.requested_bytes)}")
    print(f"  raw span bytes:        {format_bytes(plan.raw_span_bytes)}")
    print(f"  planned read bytes:    {format_bytes(plan.read_bytes)}")
    print(f"  waste bytes:           {format_bytes(plan.waste_bytes)}")
    print(f"  read amplification:    {plan.read_amplification:.3f}x")
    for item in plan.ranges:
        print(
            "  range:                 "
            f"experts={','.join(str(e) for e in item.experts)} "
            f"offset={item.aligned_offset} length={item.aligned_length} "
            f"raw_offset={item.offset} raw_length={item.length}"
        )
    print("  result:                ok")


def _print_batch_expert_io_plan(plan: BatchExpertIOPlan) -> None:
    print("LargerLM batch expert I/O plan")
    print(f"  expert layout:         {plan.expert_layout_path}")
    print(f"  router json dir:       {plan.router_json_dir}")
    print(f"  router json glob:      {plan.router_json_glob}")
    print(f"  layer:                 {plan.layer}")
    print(f"  batch tokens:          {plan.batch_tokens}")
    print(f"  total assignments:     {plan.total_assignments}")
    print(f"  selected experts:      {','.join(str(e) for e in plan.selected_experts)}")
    print(f"  serial read bytes:     {format_bytes(plan.serial_read_bytes)}")
    print(f"  unique slot bytes:     {format_bytes(plan.unique_requested_bytes)}")
    print(f"  planned read bytes:    {format_bytes(plan.planned_read_bytes)}")
    if plan.planned_read_seconds is not None:
        print(
            "  planned read time:     "
            f"{plan.planned_read_seconds:.6g}s "
            f"@ {plan.ssd_read_gib_per_second:.6g} GiB/s"
        )
    print(f"  coalesced savings:     {format_bytes(plan.coalesced_savings_bytes)}")
    print(
        "  read ranges:           "
        f"{plan.raw_range_count} raw -> {plan.coalesced_range_count} coalesced"
    )
    print(f"  assignment amp:        {plan.assignment_read_amplification:.3f}x")
    print(f"  unique amp:            {plan.unique_read_amplification:.3f}x")
    for item in plan.expert_tokens:
        print(
            "  expert tokens:         "
            f"expert={item.expert} tokens={','.join(str(t) for t in item.tokens)}"
        )
    print("  result:                ok")


def _print_batch_expert_io_tiling_plan(plan: BatchExpertIOTilingPlan) -> None:
    print("LargerLM batch expert I/O tiling plan")
    print(f"  tile count:            {plan.tile_count}")
    print(f"  max stage bytes:       {format_bytes(plan.max_stage_bytes)}")
    print(f"  max compact bytes:     {format_bytes(plan.max_compact_stage_bytes)}")
    print(f"  tile assignments:      {plan.total_tile_assignments}")
    print(f"  tile read bytes:       {format_bytes(plan.total_tile_planned_read_bytes)}")
    print(f"  max tile read bytes:   {format_bytes(plan.max_tile_planned_read_bytes)}")
    print(f"  max tile compact:      {format_bytes(plan.max_tile_compact_stage_bytes)}")
    for tile in plan.tiles:
        print(
            "  tile:                  "
            f"index={tile.tile_index} "
            f"experts={','.join(str(e) for e in tile.selected_experts)} "
            f"tokens={tile.active_token_count} assignments={tile.total_assignments} "
            f"read={format_bytes(tile.planned_read_bytes)}"
        )
    print("  result:                ok")


def _print_static_expert_capacity_plan(plan: StaticExpertCapacityPlan) -> None:
    print("LargerLM static expert capacity plan")
    print(f"  batch tokens:          {plan.batch_tokens}")
    print(f"  selected experts:      {','.join(str(e) for e in plan.selected_experts)}")
    print(f"  capacity/expert:       {plan.capacity_per_expert}")
    print(f"  total assignments:     {plan.total_assignments}")
    print(f"  capacity slots:        {plan.total_capacity_slots}")
    print(f"  used slots:            {plan.used_slots}")
    print(f"  utilization:           {plan.utilization:.3f}")
    print(f"  overflow assignments:  {plan.overflow_assignments}")
    print(f"  max tokens/expert:     {plan.max_tokens_per_expert}")
    print(f"  overflow path needed:  {plan.requires_overflow_path}")
    for item in plan.usages:
        print(
            "  expert capacity:       "
            f"expert={item.expert} assigned={item.assigned_tokens} "
            f"used={item.used_slots} overflow={item.overflow_assignments}"
        )
    print("  result:                ok")


def _print_static_capacity_binary_validation(
    result: StaticExpertCapacityBinaryValidation,
) -> None:
    print("LargerLM static capacity binary")
    print(f"  path:                  {result.path}")
    print(f"  bytes:                 {format_bytes(result.bytes_read)}")
    print(f"  version:               {result.version}")
    print(f"  batch tokens:          {result.batch_tokens}")
    print(f"  expert count:          {result.expert_count}")
    print(f"  capacity/expert:       {result.capacity_per_expert}")
    print(f"  total assignments:     {result.total_assignments}")
    print(f"  used slots:            {result.used_slots}")
    print(f"  overflow records:      {result.overflow_records}")
    print(f"  active slots:          {result.active_slot_records}")
    print(f"  inactive slots:        {result.inactive_slot_records}")
    print("  result:                ok")


def _print_batch_expert_stage(result: BatchExpertStageResult) -> None:
    print("LargerLM batch expert stage")
    print(f"  expert layout:         {result.expert_layout_path}")
    print(f"  router json dir:       {result.router_json_dir}")
    print(f"  layer:                 {result.layer}")
    print(f"  batch tokens:          {result.batch_tokens}")
    print(f"  selected experts:      {','.join(str(e) for e in result.selected_experts)}")
    print(f"  expert slot:           {format_bytes(result.expert_slot_bytes)}")
    print(f"  planned read bytes:    {format_bytes(result.planned_read_bytes)}")
    if result.io_summary.planned_read_seconds is not None:
        print(
            "  planned read time:     "
            f"{result.io_summary.planned_read_seconds:.6g}s "
            f"@ {result.io_summary.ssd_read_gib_per_second:.6g} GiB/s"
        )
        if result.io_summary.max_read_seconds > 0:
            print(
                "  read time cap:         "
                f"{result.io_summary.max_read_seconds:.6g}s "
                f"ok={result.io_summary.read_seconds_ok}"
            )
    print(f"  staged bytes:          {format_bytes(result.staged_bytes)}")
    print(f"  max stage bytes:       {format_bytes(result.max_stage_bytes)}")
    print(f"  copy chunk:            {format_bytes(result.copy_chunk_bytes)}")
    print(
        "  read advice:           "
        f"{'yes' if result.read_advice.supported else 'no'} "
        f"calls={result.read_advice.calls} "
        f"bytes={format_bytes(result.read_advice.advised_bytes)}"
    )
    print(f"  ranges:                {len(result.ranges)}")
    print(f"  staged slots:          {len(result.slots)}")
    summary = result.io_summary
    print(
        "  read ranges:           "
        f"{summary.raw_range_count} raw -> {summary.coalesced_range_count} coalesced"
    )
    if summary.max_raw_ranges > 0:
        print(
            "  raw range cap:         "
            f"{summary.raw_range_count}/{summary.max_raw_ranges} "
            f"ok={summary.raw_range_count_ok}"
        )
    if summary.max_coalesced_ranges > 0:
        print(
            "  coalesced range cap:   "
            f"{summary.coalesced_range_count}/{summary.max_coalesced_ranges} "
            f"ok={summary.coalesced_range_count_ok}"
        )
    print(f"  coalesced savings:     {format_bytes(summary.coalesced_savings_bytes)}")
    print(f"  unique amp:            {summary.unique_read_amplification:.3f}x")
    print(f"  stage utilization:     {summary.stage_budget_utilization:.3f}")
    print(f"  stage file:            {result.stage_file_path}")
    if result.manifest_path is not None:
        print(f"  manifest:              {result.manifest_path}")
    print("  result:                ok")


def _print_staged_routed_moe_batch(result: StagedRoutedMoEBatchResult) -> None:
    print("LargerLM staged routed MoE batch")
    print(f"  runner:                {result.runner_path}")
    print(f"  stage manifest:        {result.stage_manifest_path}")
    print(f"  stage file:            {result.stage_file_path}")
    print(f"  compact layout:        {result.compact_layout_path}")
    print(f"  compact layer:         {result.compact_layer_path}")
    print(f"  compact routes:        {result.compact_routes_path}")
    print(f"  layer:                 {result.layer}")
    print(f"  batch tokens:          {result.batch_tokens}")
    print(f"  hidden dim:            {result.hidden_dim}")
    print(f"  selected experts:      {','.join(str(e) for e in result.selected_experts)}")
    print(f"  input bytes:           {format_bytes(result.input_bytes)}")
    print(f"  output bytes:          {format_bytes(result.output_bytes)}")
    print(f"  token bytes:           {format_bytes(result.token_bytes)}")
    if result.stage_io_summary_available:
        print(f"  stage serial read:     {format_bytes(result.stage_serial_read_bytes)}")
        print(
            "  stage unique requested:"
            f" {format_bytes(result.stage_unique_requested_bytes)}"
        )
        print(f"  stage planned read:    {format_bytes(result.stage_planned_read_bytes)}")
        print(f"  stage file bytes:      {format_bytes(result.stage_staged_bytes)}")
        print(f"  stage waste bytes:     {format_bytes(result.stage_waste_bytes)}")
        print(
            "  stage read ranges:     "
            f"{result.stage_raw_range_count} raw -> "
            f"{result.stage_coalesced_range_count} coalesced"
        )
        print(
            "  stage assignment amp:  "
            f"{result.stage_assignment_read_amplification:.3f}x"
        )
        print(f"  stage unique amp:      {result.stage_unique_read_amplification:.3f}x")
        print(f"  stage utilization:     {result.stage_budget_utilization:.3f}")
    if result.stage_read_advice_available:
        print(
            "  stage read advice:     "
            f"ranges={result.stage_read_advice_attempted_ranges} "
            f"calls={result.stage_read_advice_calls} "
            f"bytes={format_bytes(result.stage_read_advice_bytes)} "
            f"failures={result.stage_read_advice_failures}"
        )
    print(f"  compact stage bytes:   {format_bytes(result.compact_stage_bytes)}")
    print(
        "  compact materialized:  "
        f"{format_bytes(result.compact_stage_materialized_bytes)}"
    )
    print(f"  compact stage storage: {result.compact_stage_storage}")
    print(f"  compact stage limit:   {format_bytes(result.max_compact_stage_bytes)}")
    if result.static_capacity_binary_path is not None:
        print(
            "  static capacity:       "
            f"{result.static_capacity_path if result.static_capacity_path is not None else 'disabled'}"
        )
        print(f"  static capacity bin:   {result.static_capacity_binary_path}")
        print(f"  static cap/expert:     {result.static_capacity_per_expert}")
        print(
            "  static slots:          "
            f"{result.static_capacity_used_slots}/{result.static_capacity_total_slots} "
            f"overflow={result.static_capacity_overflow_assignments}"
        )
        print(f"  static bin bytes:      {format_bytes(result.static_capacity_binary_bytes)}")
    print(f"  moe token block:       {result.moe_token_block}")
    if result.effective_moe_token_block is not None:
        print(f"  effective token block: {result.effective_moe_token_block}")
    if result.moe_max_expert_tokens is not None:
        print(f"  max expert tokens:     {result.moe_max_expert_tokens}")
    if result.moe_batch_buffer_bytes is not None:
        print(f"  moe batch buffers:     {format_bytes(result.moe_batch_buffer_bytes)}")
    if result.moe_estimated_peak_bytes is not None:
        print(f"  runner peak estimate:  {format_bytes(result.moe_estimated_peak_bytes)}")
    print(f"  copy chunk:            {format_bytes(result.copy_chunk_bytes)}")
    print(f"  runner calls:          {result.command_count}")
    print(f"  output dir:            {result.output_dir}")
    print(f"  output f32:            {result.output_path}")
    print("  result:                ok")


def _print_staged_routed_moe_batch_plan(
    result: StagedRoutedMoEBatchPlanResult,
) -> None:
    print("LargerLM staged routed MoE batch plan")
    print(f"  runner:                {result.runner_path}")
    print(f"  plan:                  {result.plan_path}")
    print(f"  jobs:                  {result.job_count}")
    print(f"  runner calls:          {result.command_count}")
    print(f"  wall runner elapsed:   {result.wall_runner_elapsed_seconds:.6f}s")
    if result.runner_reported_elapsed_seconds is not None:
        print(
            "  runner reported:       "
            f"{result.runner_reported_elapsed_seconds:.6f}s"
        )
    for index, output_path in enumerate(result.output_paths):
        print(f"  output[{index}]:             {output_path}")
    print("  result:                ok")


def _print_staged_routed_moe_batch_plan_server(
    result: StagedRoutedMoEBatchPlanServerResult,
) -> None:
    print("LargerLM staged routed MoE batch plan server")
    print(f"  runner:                {result.runner_path}")
    print(f"  plans:                 {result.plan_count}")
    print(f"  jobs:                  {result.job_count}")
    print(f"  runner calls:          {result.command_count}")
    print(f"  wall runner elapsed:   {result.wall_runner_elapsed_seconds:.6f}s")
    for index, plan_path in enumerate(result.plan_paths):
        print(f"  plan[{index}]:               {plan_path}")
    for index, output_path in enumerate(result.output_paths):
        print(f"  output[{index}]:             {output_path}")
    print("  result:                ok")


def _print_tiled_staged_routed_moe_batch(
    result: TiledStagedRoutedMoEBatchResult,
) -> None:
    print("LargerLM tiled staged routed MoE batch")
    print(f"  runner:                {result.runner_path}")
    print(f"  expert layout:         {result.expert_layout_path}")
    print(f"  layer:                 {result.layer}")
    print(f"  batch tokens:          {result.batch_tokens}")
    print(f"  hidden dim:            {result.hidden_dim}")
    print(f"  tile count:            {result.tile_count}")
    print(
        "  stage read bytes:      "
        f"{format_bytes(result.total_stage_planned_read_bytes)}"
    )
    print(
        "  max tile stage read:   "
        f"{format_bytes(result.max_tile_stage_planned_read_bytes)}"
    )
    print(
        "  compact stage bytes:   "
        f"{format_bytes(result.total_compact_stage_bytes)}"
    )
    print(
        "  max tile compact:      "
        f"{format_bytes(result.max_tile_compact_stage_bytes)}"
    )
    for index, token_indices in enumerate(result.tile_original_token_indices):
        tile = result.tile_results[index]
        print(
            "  tile:                  "
            f"index={index} tokens={','.join(str(item) for item in token_indices)} "
            f"experts={','.join(str(item) for item in tile.selected_experts)}"
        )
    print(f"  output dir:            {result.output_dir}")
    print(f"  output f32:            {result.output_path}")
    print("  result:                ok")


def _print_prefill_plan(plan: PrefillPlan) -> None:
    print("LargerLM prefill plan")
    print(f"  model:                {plan.model_path}")
    print(f"  model_type:           {plan.model_type}")
    print(f"  prompt tokens:        {plan.prompt_tokens}")
    print(f"  dtype bits:           {plan.dtype_bits}")
    print(f"  expert bits/group:    {plan.expert_bits}/{plan.group_size}")
    print(f"  layers:               {plan.num_layers}")
    print(f"  MoE layers:           {plan.moe_layers}")
    print(f"  dense layers:         {plan.dense_layers}")
    print(f"  total FLOPs:          {plan.total_flops / 1e12:.3f} TFLOP")
    print(f"  total weight/read:    {format_bytes(plan.total_weight_bytes)}")
    print(f"  resident GEMM weights: {format_bytes(plan.resident_gemm_weight_bytes)}")
    print(f"  routed expert read:   {format_bytes(plan.routed_expert_read_bytes)}")
    if (
        plan.routed_expert_chunked_read_bytes
        and plan.routed_expert_chunked_read_bytes != plan.routed_expert_read_bytes
    ):
        print(
            "  routed chunked read: "
            f"{format_bytes(plan.routed_expert_chunked_read_bytes)} "
            f"({plan.routed_expert_read_chunks_per_prompt} chunks)"
        )
    if plan.routed_expert_slot_bytes:
        print(f"  routed expert slot:   {format_bytes(plan.routed_expert_slot_bytes)}")
        print(
            "  routed unique/layer: "
            f"{plan.routed_expert_unique_per_moe_layer} "
            f"/ assignments {plan.routed_expert_assignments_per_moe_layer}"
        )
        print(f"  routed backend:       {plan.routed_expert_backend_hint}")
    if plan.cache_io_plan is not None:
        cache_io = plan.cache_io_plan
        print(
            "  cache read/write:     "
            f"{format_bytes(cache_io.total_cache_read_bytes)} / "
            f"{format_bytes(cache_io.total_cache_write_bytes)}"
        )
        print(
            "  cache read detail:    "
            f"mla={format_bytes(cache_io.mla_cache_read_bytes)} "
            f"dsa_index={format_bytes(cache_io.dsa_index_cache_read_bytes)} "
            f"indexed_layers={cache_io.indexed_attention_layers}"
        )
    if plan.routed_expert_capacity_plan is not None:
        capacity = plan.routed_expert_capacity_plan
        print(
            "  routed static cap:   "
            f"tokens={capacity.capacity_tokens} "
            f"balanced={capacity.balanced_capacity_per_expert}/expert "
            f"util={capacity.balanced_capacity_utilization:.1%} "
            f"act={format_bytes(capacity.balanced_capacity_activation_bytes_per_moe_layer)} "
            f"over={capacity.balanced_capacity_overprovision_assignments_per_capacity_chunk}; "
            f"spill-free={capacity.spill_free_capacity_per_expert}/expert "
            f"util={capacity.spill_free_capacity_utilization:.1%} "
            f"act={format_bytes(capacity.spill_free_capacity_activation_bytes_per_moe_layer)}"
        )
        print(
            "  static cap backend:  "
            f"{capacity.backend_hint} "
            f"overflow_path={capacity.requires_overflow_path_for_balanced_capacity}"
        )
    if plan.staged_moe_runner_scratch_plan is not None:
        staged = plan.staged_moe_runner_scratch_plan
        fit = (
            "unknown"
            if staged.fits_runner_scratch is None
            else ("yes" if staged.fits_runner_scratch else "no")
        )
        print(
            "  staged MoE runner:   "
            f"auto_block={staged.auto_token_block} "
            f"max_expert_tokens={staged.max_expert_tokens_per_capacity_chunk} "
            f"buffers={format_bytes(staged.token_block_buffer_bytes_per_moe_layer)} "
            f"peak={format_bytes(staged.estimated_peak_bytes_per_moe_layer)} "
            f"fits_runner_scratch={fit}"
        )
    if plan.routed_expert_read_cost_plan is not None:
        read_cost = plan.routed_expert_read_cost_plan
        seconds = (
            "unknown"
            if read_cost.planned_read_seconds is None
            else f"{read_cost.planned_read_seconds:.2f}s"
        )
        print(
            "  routed read cost:    "
            f"{format_bytes(read_cost.planned_read_bytes)} "
            f"x{read_cost.read_amplification:.2f} "
            f"extra={format_bytes(read_cost.extra_read_bytes)} "
            f"ssd_time={seconds}"
        )
    if plan.suggested_prefill_guard_flags:
        argv = plan.suggested_prefill_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested prefill args: {' '.join(str(item) for item in argv)}")
    print(
        "  public GLM-5.2 shape: "
        f"{'yes' if plan.public_glm_5_2_shape.get('matches') is True else 'no'}"
    )
    if plan.suggested_prefill_linear_calibration_flags:
        argv = plan.suggested_prefill_linear_calibration_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested calib args: {' '.join(str(item) for item in argv)}")
    if plan.suggested_launch_profile:
        argv = plan.suggested_launch_profile.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested launch args: {' '.join(str(item) for item in argv)}")
    if plan.suggested_guard_flags:
        argv = plan.suggested_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested guard args: {' '.join(str(item) for item in argv)}")
    if plan.suggested_stage_temp_guard_flags:
        argv = plan.suggested_stage_temp_guard_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested stage args: {' '.join(str(item) for item in argv)}")
    print(f"  peak activation:      {format_bytes(plan.peak_activation_bytes)}")
    if plan.chunk_plan is not None:
        print(
            "  prefill chunk:        "
            f"{plan.chunk_plan.recommended_chunk_tokens} tokens "
            f"x {plan.chunk_plan.chunks} chunks "
            f"peak={format_bytes(plan.chunk_plan.estimated_peak_activation_bytes)} "
            f"/ limit {format_bytes(plan.chunk_plan.max_activation_bytes)}"
        )
    print(f"  MPP candidate ops:    {plan.mpp_candidate_ops} / {len(plan.ops)}")
    for candidate in plan.prefill_backend_candidates[:3]:
        print(
            "  backend priority:    "
            f"#{candidate.rank} {candidate.op_name} "
            f"{candidate.total_flops / 1e12:.3f} TFLOP "
            f"path={candidate.execution_path} "
            f"status={candidate.availability}"
        )
    for shape in plan.prefill_linear_calibration_shapes[:3]:
        print(
            "  calibration shape:   "
            f"#{shape.rank} {shape.op_name} "
            f"B={shape.batch_tokens} "
            f"{shape.matrix_shape_arg} "
            f"peak={format_bytes(shape.calibration_estimated_peak_bytes)}"
        )
    if plan.backend_capability is not None:
        print(f"  backend:              {plan.backend_capability.recommended_backend}")
        print(
            "  effective Metal4 ops: "
            f"{plan.effective_metal4_candidate_ops} / {plan.mpp_candidate_ops}"
        )
        print(
            "  effective MPP ops:    "
            f"{plan.effective_mpp_candidate_ops} / {plan.mpp_candidate_ops}"
        )
        if plan.backend_capability.reasons:
            print(f"  backend note:         {plan.backend_capability.reasons[0]}")
    for op in plan.ops:
        print(
            "  op:                   "
            f"{op.name} layers={op.layers} "
            f"M={op.m_tokens} K={op.k_in} N={op.n_out} "
            f"flops={op.total_flops / 1e9:.3f} GFLOP "
            f"backend={op.backend_hint}"
        )
        if op.tile_plan is not None:
            tile = op.tile_plan
            print(
                "  tile:                 "
                f"{op.name} tg={tile.threadgroup_tile_m}x{tile.threadgroup_tile_n} "
                f"sg={tile.simdgroup_tile_m}x{tile.simdgroup_tile_n} "
                f"grid={tile.grid_m}x{tile.grid_n} "
                f"edge={tile.edge_threadgroup_tiles} "
                f"k_tile={tile.k_tile} static={tile.static_extent_full_tiles}"
            )
    print("  result:               ok")


def _prefill_plan(args: argparse.Namespace) -> int:
    backend_capability = None
    if args.inspect_backend:
        backend_capability = inspect_prefill_backend(
            sdk_path=args.sdk_path,
            probe_binary=args.probe_binary,
            run_host_probe=not args.no_host_probe,
            compile_mpp_probe=args.compile_mpp_probe,
            run_mpp_probe=args.run_mpp_probe,
            run_mpsgraph_probe=args.run_mpsgraph_probe,
            probe_timeout_seconds=args.probe_timeout_seconds,
        )
    plan = build_prefill_plan(
        args.model,
        prompt_tokens=args.prompt_tokens,
        dtype_bits=args.dtype_bits,
        expert_bits=args.expert_bits,
        group_size=args.group_size,
        mpp_min_tokens=args.mpp_min_tokens,
        backend_capability=backend_capability,
        compile_mpp_probe=bool(args.inspect_backend and args.compile_mpp_probe),
        run_mpp_probe=bool(args.inspect_backend and args.run_mpp_probe),
        run_mpsgraph_probe=bool(args.inspect_backend and args.run_mpsgraph_probe),
        probe_timeout_seconds=args.probe_timeout_seconds,
        simdgroup_tile_m=args.simdgroup_tile_m,
        simdgroup_tile_n=args.simdgroup_tile_n,
        simdgroups_m=args.simdgroups_m,
        simdgroups_n=args.simdgroups_n,
        k_tile=args.k_tile,
        max_prefill_activation_bytes=(
            int(args.max_prefill_activation_mib * 1024 * 1024)
            if args.max_prefill_activation_mib is not None
            else None
        ),
        max_runner_scratch_bytes=(
            int(args.max_runner_scratch_mib * 1024 * 1024)
            if args.max_runner_scratch_mib is not None
            else None
        ),
        expert_stage_align_bytes=int(args.expert_stage_align_kib * 1024),
        prefill_static_capacity_per_expert=args.prefill_static_capacity_per_expert,
        ssd_read_gib_per_second=args.ssd_read_gib_s,
        prefill_linear_backend=args.prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=args.prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=args.prefill_mpsgraph_min_matrix_dim,
        prefill_min_accelerated_flop_fraction=(
            args.prefill_min_accelerated_flop_fraction
        ),
        require_prefill_acceleration=args.require_prefill_acceleration,
        require_public_glm_5_2_shape=args.require_public_glm_5_2_shape,
    )
    if args.write_launch_profile is not None:
        _write_launch_profile_file(
            args.write_launch_profile,
            plan.suggested_launch_profile,
        )
    if args.write_calibration_flags is not None:
        _write_prefill_calibration_flags_file(
            args.write_calibration_flags,
            plan.suggested_prefill_linear_calibration_flags,
        )
    if args.json:
        print(json.dumps(plan, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_plan(plan)
    return 0


def _positive_int_cli_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise CliArgumentError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise CliArgumentError(f"{name} must be a positive integer")
    return parsed


def _optional_positive_int_cli_value(value: object | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int_cli_value(value, name)


def _planned_calibration_mib(
    flags: dict[str, object],
    key: str,
    *,
    label: str,
) -> int:
    if key not in flags:
        raise CliArgumentError(f"prefill plan did not produce {label}")
    return _positive_int_cli_value(flags[key], label)


def _resolve_planned_calibration_cap_mib(
    *,
    planned_mib: int,
    requested_mib: object | None,
    auto_limit_mib: object,
    requested_name: str,
    auto_limit_name: str,
) -> int:
    auto_limit = _positive_int_cli_value(auto_limit_mib, auto_limit_name)
    if planned_mib > auto_limit:
        raise CliArgumentError(
            f"planned {requested_name} {planned_mib} MiB exceeds "
            f"{auto_limit_name} {auto_limit} MiB; reduce the prompt chunk or raise "
            f"{auto_limit_name} explicitly"
        )
    requested = _optional_positive_int_cli_value(requested_mib, requested_name)
    if requested is None:
        return planned_mib
    if requested < planned_mib:
        raise CliArgumentError(
            f"{requested_name} {requested} MiB is below the planner requirement "
            f"{planned_mib} MiB for the selected calibration shapes"
        )
    return requested


def _align_up_bytes(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _calibration_cap_mib_with_headroom(value: int) -> int:
    return max(1, math.ceil(value * 1.10 / (1024 * 1024)))


def _dtype_aware_planned_runner_scratch_mib(
    shapes: tuple[PrefillLinearCalibrationShape, ...],
    *,
    matrix_dtype: str,
    planned_scratch_mib: int,
) -> int:
    if matrix_dtype not in CALIBRATION_MATRIX_DTYPE_BYTES:
        raise CliArgumentError("--matrix-dtype must be F32 or BF16")
    dtype_bytes = CALIBRATION_MATRIX_DTYPE_BYTES[matrix_dtype]
    raw_extra_multiplier = 0 if matrix_dtype == "F32" else dtype_bytes
    max_peak_bytes = 0
    for shape in shapes:
        matrix_values = shape.in_dim * shape.out_dim
        input_bytes = shape.batch_tokens * shape.in_dim * 4
        output_bytes = shape.batch_tokens * shape.out_dim * 4
        custom_scratch = _align_up_bytes(
            matrix_values * dtype_bytes,
            2 * 1024 * 1024,
        )
        conversion_scratch = _align_up_bytes(
            matrix_values * 4,
            2 * 1024 * 1024,
        ) + matrix_values * raw_extra_multiplier
        max_peak_bytes = max(
            max_peak_bytes,
            custom_scratch + input_bytes + output_bytes,
            conversion_scratch + input_bytes + output_bytes,
        )
    dtype_scratch_mib = _calibration_cap_mib_with_headroom(max_peak_bytes)
    return max(planned_scratch_mib, dtype_scratch_mib)


def _prefill_calibration_flags_for_run(
    plan: PrefillPlan,
    *,
    max_calibration_case_mib: int,
    max_resident_matrix_mib: int,
    max_runner_scratch_mib: int,
    matrix_dtype: str,
) -> dict[str, object]:
    flags = plan.suggested_prefill_linear_calibration_flags
    if not isinstance(flags, dict):
        raise CliArgumentError("prefill plan did not produce calibration flags")
    shapes = plan.prefill_linear_calibration_shapes
    if not shapes:
        raise CliArgumentError("prefill plan did not produce calibration shapes")
    batch_tokens = tuple(sorted({shape.batch_tokens for shape in shapes}))
    matrix_shapes = tuple(shape.matrix_shape_arg for shape in shapes)
    applied = dict(flags)
    applied["batch_tokens"] = batch_tokens
    applied["matrix_shapes"] = matrix_shapes
    applied["max_calibration_case_mib"] = max_calibration_case_mib
    applied["max_resident_matrix_mib"] = max_resident_matrix_mib
    applied["max_runner_scratch_mib"] = max_runner_scratch_mib
    applied["max_calibration_case_bytes"] = max_calibration_case_mib * 1024 * 1024
    applied["max_resident_matrix_bytes"] = max_resident_matrix_mib * 1024 * 1024
    applied["max_runner_scratch_bytes"] = max_runner_scratch_mib * 1024 * 1024
    applied["matrix_dtype"] = matrix_dtype
    applied["argv"] = (
        "--batch-tokens",
        ",".join(str(value) for value in batch_tokens),
        "--matrix-shapes",
        ",".join(matrix_shapes),
        "--matrix-dtype",
        matrix_dtype,
        "--max-calibration-case-mib",
        str(max_calibration_case_mib),
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    )
    return applied


def _prefill_plan_calibration_work_dir_budget(
    shapes: tuple[PrefillLinearCalibrationShape, ...],
    *,
    repeats: int,
    max_calibration_work_dir_mib: object,
    matrix_dtype: str,
) -> dict[str, object]:
    if not shapes:
        raise CliArgumentError("prefill plan did not produce calibration shapes")
    repeat_count = _positive_int_cli_value(repeats, "--repeats")
    max_work_dir_mib = _positive_int_cli_value(
        max_calibration_work_dir_mib,
        "--max-calibration-work-dir-mib",
    )
    if matrix_dtype not in CALIBRATION_MATRIX_DTYPE_BYTES:
        raise CliArgumentError("--matrix-dtype must be F32 or BF16")
    matrix_dtype_bytes = CALIBRATION_MATRIX_DTYPE_BYTES[matrix_dtype]
    matrix_bytes = sum(
        shape.in_dim * shape.out_dim * matrix_dtype_bytes for shape in shapes
    )
    input_bytes = sum(shape.calibration_input_bytes for shape in shapes)
    single_backend_output_bytes = sum(
        shape.calibration_output_bytes for shape in shapes
    )
    backend_count = len(PREFILL_LINEAR_CALIBRATION_BACKENDS)
    total_backend_output_bytes = (
        backend_count * repeat_count * single_backend_output_bytes
    )
    estimated_work_dir_bytes = matrix_bytes + input_bytes + total_backend_output_bytes
    max_work_dir_bytes = max_work_dir_mib * 1024 * 1024
    if estimated_work_dir_bytes > max_work_dir_bytes:
        raise CliArgumentError(
            "planned calibration work dir "
            f"{format_bytes(estimated_work_dir_bytes)} exceeds "
            f"--max-calibration-work-dir-mib {max_work_dir_mib} MiB; reduce the "
            "prompt chunk or raise --max-calibration-work-dir-mib explicitly"
        )
    return {
        "source": "prefill_plan_calibration",
        "shape_count": len(shapes),
        "repeats": repeat_count,
        "matrix_dtype": matrix_dtype,
        "backend_count": backend_count,
        "calibrated_backends": PREFILL_LINEAR_CALIBRATION_BACKENDS,
        "backend_output_file_count": backend_count * repeat_count * len(shapes),
        "matrix_bytes": matrix_bytes,
        "input_bytes": input_bytes,
        "single_backend_output_bytes": single_backend_output_bytes,
        "total_backend_output_bytes": total_backend_output_bytes,
        "single_case_bytes": (
            matrix_bytes + input_bytes + single_backend_output_bytes
        ),
        "estimated_work_dir_bytes": estimated_work_dir_bytes,
        "max_calibration_work_dir_bytes": max_work_dir_bytes,
        "max_calibration_work_dir_mib": max_work_dir_mib,
    }


def _print_prefill_plan_calibration(
    plan: PrefillPlan,
    applied_flags: dict[str, object],
    work_dir_budget: dict[str, object],
    result: ResidentLinearCalibrationResult,
    combined_launch_profile: dict[str, object] | None,
) -> None:
    print("LargerLM prefill plan-driven calibration")
    print(f"  model:                {plan.model_path}")
    print(f"  prompt tokens:        {plan.prompt_tokens}")
    print(f"  calibration shapes:   {len(plan.prefill_linear_calibration_shapes)}")
    print(
        "  work dir estimate:   "
        f"{format_bytes(work_dir_budget['estimated_work_dir_bytes'])} / limit "
        f"{format_bytes(work_dir_budget['max_calibration_work_dir_bytes'])}"
    )
    argv = applied_flags.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        print(f"  applied calib args:   {' '.join(str(item) for item in argv)}")
    if combined_launch_profile:
        profile_argv = combined_launch_profile.get("argv")
        if isinstance(profile_argv, (list, tuple)) and profile_argv:
            print(
                "  combined launch args: "
                f"{' '.join(str(item) for item in profile_argv)}"
            )
    _print_resident_linear_calibration(result, show_work_dir_budget=False)


def _dict_section(value: object, name: str) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    section = value.get(name)
    return section if isinstance(section, dict) else None


def _positive_int_policy_value(section: dict[str, object], key: str) -> int | None:
    value = section.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float_policy_value(section: dict[str, object], key: str) -> float | None:
    value = section.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _merged_prefill_runtime_policy_flags(
    *,
    plan_policy: dict[str, object] | None,
    calibration_policy: dict[str, object] | None,
) -> dict[str, object] | None:
    if plan_policy is None and calibration_policy is None:
        return None
    plan_section = plan_policy or {}
    calibration_section = calibration_policy or {}
    payload: dict[str, object] = {"source": "prefill_plan_calibration"}
    if isinstance(plan_policy, dict):
        payload["plan_source"] = plan_policy.get("source")
    if isinstance(calibration_policy, dict):
        payload["calibration_source"] = calibration_policy.get("source")

    argv: list[str] = []
    backend = plan_section.get("prefill_linear_backend")
    calibration_backend = calibration_section.get("prefill_linear_backend")
    if not (isinstance(backend, str) and backend != "auto"):
        backend = calibration_backend
    if isinstance(backend, str) and backend != "auto":
        payload["prefill_linear_backend"] = backend
        argv.extend(["--prefill-linear-backend", backend])

    batch_tokens = _positive_int_policy_value(
        calibration_section,
        "prefill_mpsgraph_min_batch_tokens",
    ) or _positive_int_policy_value(
        plan_section,
        "prefill_mpsgraph_min_batch_tokens",
    )
    matrix_dim = _positive_int_policy_value(
        calibration_section,
        "prefill_mpsgraph_min_matrix_dim",
    ) or _positive_int_policy_value(
        plan_section,
        "prefill_mpsgraph_min_matrix_dim",
    )
    if batch_tokens is not None and matrix_dim is not None:
        payload["prefill_mpsgraph_min_batch_tokens"] = batch_tokens
        payload["prefill_mpsgraph_min_matrix_dim"] = matrix_dim
        argv.extend(
            [
                "--prefill-mpsgraph-min-batch-tokens",
                str(batch_tokens),
                "--prefill-mpsgraph-min-matrix-dim",
                str(matrix_dim),
            ]
        )

    min_fraction = _positive_float_policy_value(
        plan_section,
        "prefill_min_accelerated_flop_fraction",
    )
    if min_fraction is not None:
        payload["prefill_min_accelerated_flop_fraction"] = min_fraction
        argv.extend(
            [
                "--prefill-min-accelerated-flop-fraction",
                format_routed_read_guard_flag_float(min_fraction),
            ]
        )
    if plan_section.get("require_prefill_acceleration") is True:
        payload["require_prefill_acceleration"] = True
        argv.append("--require-prefill-acceleration")
    if not argv:
        return None
    payload["argv"] = tuple(argv)
    return payload


def _combined_prefill_plan_calibration_launch_profile(
    plan: PrefillPlan,
    result: ResidentLinearCalibrationResult,
) -> dict[str, object] | None:
    launch_sections = (
        plan.suggested_launch_profile.get("sections")
        if isinstance(plan.suggested_launch_profile, dict)
        else None
    )
    runtime_policy = _merged_prefill_runtime_policy_flags(
        plan_policy=_dict_section(launch_sections, "prefill_runtime_policy_flags"),
        calibration_policy=(
            result.suggested_prefill_runtime_policy_flags
            if isinstance(result.suggested_prefill_runtime_policy_flags, dict)
            else None
        ),
    )
    return combine_suggested_launch_profile(
        prefill_backend_probe_flags=_dict_section(
            launch_sections,
            "prefill_backend_probe_flags",
        ),
        prefill_runtime_policy_flags=runtime_policy,
        prefill_acceleration_flags=_dict_section(
            launch_sections,
            "prefill_acceleration_flags",
        ),
        prefill_guard_flags=_dict_section(launch_sections, "prefill_guard_flags"),
        public_glm_5_2_shape_guard_flags=_dict_section(
            launch_sections,
            "public_glm_5_2_shape_guard_flags",
        ),
        source="prefill_plan_calibration",
    )


def _prefill_plan_calibrate(args: argparse.Namespace) -> int:
    plan = build_prefill_plan(
        args.model,
        prompt_tokens=args.prompt_tokens,
        dtype_bits=args.dtype_bits,
        expert_bits=args.expert_bits,
        group_size=args.group_size,
        mpp_min_tokens=args.mpp_min_tokens,
        prefill_linear_backend=args.prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=args.prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=args.prefill_mpsgraph_min_matrix_dim,
        prefill_min_accelerated_flop_fraction=(
            args.prefill_min_accelerated_flop_fraction
        ),
        require_prefill_acceleration=args.require_prefill_acceleration,
        require_public_glm_5_2_shape=args.require_public_glm_5_2_shape,
        expert_stage_align_bytes=int(args.expert_stage_align_kib * 1024),
        prefill_static_capacity_per_expert=args.prefill_static_capacity_per_expert,
        ssd_read_gib_per_second=args.ssd_read_gib_s,
        max_prefill_activation_bytes=(
            int(args.max_prefill_activation_mib * 1024 * 1024)
            if args.max_prefill_activation_mib is not None
            else None
        ),
        max_runner_scratch_bytes=(
            int(args.plan_max_runner_scratch_mib * 1024 * 1024)
            if args.plan_max_runner_scratch_mib is not None
            else None
        ),
    )
    flags = plan.suggested_prefill_linear_calibration_flags
    if not isinstance(flags, dict):
        raise CliArgumentError("prefill plan did not produce calibration flags")
    planned_case_mib = _planned_calibration_mib(
        flags,
        "max_calibration_case_mib",
        label="planned calibration case cap",
    )
    planned_matrix_mib = _planned_calibration_mib(
        flags,
        "max_resident_matrix_mib",
        label="planned resident matrix cap",
    )
    planned_scratch_mib = _planned_calibration_mib(
        flags,
        "max_runner_scratch_mib",
        label="planned runner scratch cap",
    )
    planned_scratch_mib = _dtype_aware_planned_runner_scratch_mib(
        plan.prefill_linear_calibration_shapes,
        matrix_dtype=args.matrix_dtype,
        planned_scratch_mib=planned_scratch_mib,
    )
    max_calibration_case_mib = _resolve_planned_calibration_cap_mib(
        planned_mib=planned_case_mib,
        requested_mib=args.max_calibration_case_mib,
        auto_limit_mib=args.max_auto_calibration_case_mib,
        requested_name="--max-calibration-case-mib",
        auto_limit_name="--max-auto-calibration-case-mib",
    )
    max_resident_matrix_mib = _resolve_planned_calibration_cap_mib(
        planned_mib=planned_matrix_mib,
        requested_mib=args.max_resident_matrix_mib,
        auto_limit_mib=args.max_auto_resident_matrix_mib,
        requested_name="--max-resident-matrix-mib",
        auto_limit_name="--max-auto-resident-matrix-mib",
    )
    max_runner_scratch_mib = _resolve_planned_calibration_cap_mib(
        planned_mib=planned_scratch_mib,
        requested_mib=args.max_runner_scratch_mib,
        auto_limit_mib=args.max_auto_runner_scratch_mib,
        requested_name="--max-runner-scratch-mib",
        auto_limit_name="--max-auto-runner-scratch-mib",
    )
    applied_flags = _prefill_calibration_flags_for_run(
        plan,
        max_calibration_case_mib=max_calibration_case_mib,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        matrix_dtype=args.matrix_dtype,
    )
    work_dir_budget = _prefill_plan_calibration_work_dir_budget(
        plan.prefill_linear_calibration_shapes,
        repeats=args.repeats,
        max_calibration_work_dir_mib=args.max_calibration_work_dir_mib,
        matrix_dtype=args.matrix_dtype,
    )
    matrix_shapes = tuple(
        (shape.in_dim, shape.out_dim)
        for shape in plan.prefill_linear_calibration_shapes
    )
    batch_token_values = tuple(
        sorted({shape.batch_tokens for shape in plan.prefill_linear_calibration_shapes})
    )
    matrix_dim_values = tuple(
        sorted({shape.min_matrix_dim for shape in plan.prefill_linear_calibration_shapes})
    )
    result = run_resident_linear_calibration(
        runner_path=args.runner,
        batch_token_values=batch_token_values,
        matrix_dim_values=matrix_dim_values,
        matrix_shapes=matrix_shapes,
        repeats=args.repeats,
        min_mpsgraph_speedup=args.min_mpsgraph_speedup,
        max_calibration_case_mib=max_calibration_case_mib,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        max_calibration_work_dir_mib=args.max_calibration_work_dir_mib,
        calibration_work_dir_free_margin_mib=(
            args.calibration_work_dir_free_margin_mib
        ),
        matrix_dtype=args.matrix_dtype,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        echo_runner_output=args.echo_runner_output,
    )
    combined_launch_profile = _combined_prefill_plan_calibration_launch_profile(
        plan,
        result,
    )
    if args.write_calibration_flags is not None:
        _write_prefill_calibration_flags_file(
            args.write_calibration_flags,
            applied_flags,
        )
    if args.write_launch_profile is not None:
        _write_launch_profile_file(
            args.write_launch_profile,
            combined_launch_profile,
        )
    if args.json:
        payload = {
            "prefill_plan": plan,
            "calibration_candidate_coverage": (
                plan.prefill_linear_calibration_candidate_coverage
            ),
            "applied_calibration_flags": applied_flags,
            "auto_calibration_limits": {
                "max_auto_calibration_case_mib": args.max_auto_calibration_case_mib,
                "max_auto_resident_matrix_mib": args.max_auto_resident_matrix_mib,
                "max_auto_runner_scratch_mib": args.max_auto_runner_scratch_mib,
            },
            "calibration_work_dir_budget": work_dir_budget,
            "calibration": result,
            "combined_launch_profile": combined_launch_profile,
        }
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_plan_calibration(
            plan,
            applied_flags,
            work_dir_budget,
            result,
            combined_launch_profile,
        )
    return 0


def _print_disk_read_benchmark(result: SequentialReadBenchmark) -> None:
    print("LargerLM disk read benchmark")
    print(f"  path:                 {result.path}")
    print(f"  file size:            {format_bytes(result.file_size_bytes)}")
    print(f"  offset:               {format_bytes(result.offset_bytes)}")
    print(f"  requested:            {format_bytes(result.requested_bytes)}")
    print(f"  measured:             {format_bytes(result.measured_bytes)}")
    print(f"  chunk:                {format_bytes(result.chunk_bytes)}")
    print(f"  elapsed:              {result.elapsed_seconds:.6f} s")
    print(f"  throughput:           {result.gib_per_second:.3f} GiB/s")
    print(f"  short read:           {result.short_read}")
    print("  method:               sequential pread")
    print("  result:               ok")


def _disk_read_benchmark(args: argparse.Namespace) -> int:
    bytes_to_read = _scaled_bytes_arg(
        args.bytes_mib,
        name="--bytes-mib",
        scale=1024 * 1024,
    )
    chunk_bytes = _scaled_bytes_arg(
        args.chunk_mib,
        name="--chunk-mib",
        scale=1024 * 1024,
    )
    offset_bytes = _scaled_bytes_arg(
        args.offset_mib,
        name="--offset-mib",
        scale=1024 * 1024,
    )
    max_chunk_bytes = _scaled_bytes_arg(
        args.max_chunk_mib,
        name="--max-chunk-mib",
        scale=1024 * 1024,
    )
    assert chunk_bytes is not None
    assert offset_bytes is not None
    result = benchmark_sequential_read(
        args.path,
        bytes_to_read=bytes_to_read,
        chunk_bytes=chunk_bytes,
        offset_bytes=offset_bytes,
        max_chunk_bytes=max_chunk_bytes,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_disk_read_benchmark(result)
    return 0


def _print_prefill_backend(capability: PrefillBackendCapability) -> None:
    print("LargerLM prefill backend")
    print(f"  SDK:                  {capability.sdk_path}")
    print(f"  Metal headers:        {capability.metal_headers_available}")
    print(f"  Metal 4 headers:      {capability.metal4_headers_available}")
    print(f"  MTLTensor:            {capability.metal_tensor_headers_available}")
    print(f"  MTLTensor int4:       {capability.metal_tensor_int4_declared}")
    print(f"  Metal 4 ML API:       {capability.metal4_machine_learning_declared}")
    print(f"  MPSGraph matmul:      {capability.mps_graph_matmul_declared}")
    print(f"  MPSGraph runtime:     {capability.mps_graph_runtime_available}")
    print(f"  MPSGraph probe req:   {capability.mps_graph_probe_requested}")
    print(f"  MPSGraph probe ran:   {capability.mps_graph_probe_ran}")
    print(f"  MPSGraph probe ok:    {capability.mps_graph_probe_ok}")
    print(f"  MPSGraph probe error: {capability.mps_graph_probe_error}")
    print(f"  mpp::tensor_ops:      {capability.mpp_tensor_ops_symbol_declared}")
    print(f"  host probe requested: {capability.host_probe_requested}")
    print(f"  host probe path:      {capability.host_probe_path}")
    print(f"  host probe ran:       {capability.host_probe_ran}")
    print(f"  host probe error:     {capability.host_probe_error}")
    print(f"  device:               {capability.device_name}")
    print(f"  Metal 4 family:       {capability.supports_metal4_family}")
    print(f"  MTL4 queue selector:  {capability.responds_new_mtl4_command_queue}")
    print(f"  MTLTensor selector:   {capability.responds_new_tensor}")
    print(f"  tensor size/align:    {capability.responds_tensor_size_align}")
    print(f"  MTL4 compiler:        {capability.responds_new_compiler}")
    print(f"  tiny ML tensor:       {capability.can_allocate_tiny_ml_tensor}")
    print(f"  tensor error:         {capability.tensor_error}")
    print(f"  Metal 4 ML runtime:   {capability.metal4_ml_runtime_available}")
    print(f"  MPP compile req:      {capability.mpp_compile_probe_requested}")
    print(f"  MPP compile probe:    {capability.mpp_compile_probe_ok}")
    print(f"  MPP compile variant:  {capability.mpp_compile_variant}")
    print(f"  MPP run req:          {capability.mpp_run_probe_requested}")
    print(f"  MPP run probe:        {capability.mpp_run_probe_ok}")
    print(f"  MPP run max error:    {capability.mpp_run_probe_max_abs_error}")
    print(f"  MPP run variant:      {capability.mpp_run_probe_kernel_variant}")
    print(f"  MPP run shape:        {capability.mpp_run_probe_shape}")
    print(f"  MPP runtime:          {capability.mpp_runtime_available}")
    print(
        "  MPP bring-up status:  "
        f"{capability.prefill_neural_accelerator_status.get('status')}"
    )
    print(
        "  accel runtimes:       "
        f"{', '.join(capability.prefill_acceleration_runtimes) or 'none'}"
    )
    print(
        "  selectable accel:     "
        f"{', '.join(capability.selectable_accelerated_prefill_backends) or 'none'}"
    )
    print(
        "  validated accel:      "
        f"{', '.join(capability.validated_accelerated_prefill_backends) or 'none'}"
    )
    for gap in capability.prefill_acceleration_runtime_gaps:
        print(f"  accel gap:            {gap['runtime']}: {gap['reason']}")
    suggested = capability.suggested_prefill_acceleration_flags
    if suggested is not None:
        argv = suggested.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested accel args: {' '.join(str(item) for item in argv)}")
    print(f"  recommended backend:  {capability.recommended_backend}")
    for reason in capability.reasons:
        print(f"  note:                 {reason}")
    print("  result:               ok")


def _print_resident_batch_linear(result: ResidentBatchLinearResult) -> None:
    print("LargerLM prefill resident batch linear")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  runner:               {result.runner_path}")
    print(f"  layer:                {result.layer}")
    print(f"  tensor:               {result.tensor}")
    print(f"  dtype:                {result.dtype}")
    print(f"  backend:              {result.backend}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  shape:                [{result.out_dim},{result.in_dim}]")
    print(f"  matrix bytes:         {format_bytes(result.matrix_bytes)}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  elapsed:              {result.elapsed_seconds:.6f} s")
    if result.runner_backend_elapsed_seconds is not None:
        print(
            "  runner backend:       "
            f"{result.runner_backend_elapsed_seconds:.6f} s"
        )
    if result.runner_matrix_f32_elapsed_seconds is not None:
        print(
            "  runner matrix f32:    "
            f"{result.runner_matrix_f32_elapsed_seconds:.6f} s"
        )
    if result.runner_accelerator_elapsed_seconds is not None:
        print(
            "  runner accelerator:   "
            f"{result.runner_accelerator_elapsed_seconds:.6f} s"
        )
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_resident_linear_calibration(
    result: ResidentLinearCalibrationResult,
    *,
    show_work_dir_budget: bool = True,
) -> None:
    print("LargerLM prefill resident linear calibration")
    print(f"  runner:               {result.runner_path}")
    print(f"  work dir:             {result.work_dir}")
    print(f"  kept work dir:        {'yes' if result.kept_work_dir else 'no'}")
    print(f"  batch tokens:         {', '.join(map(str, result.batch_token_values))}")
    print(f"  matrix dims:          {', '.join(map(str, result.matrix_dim_values))}")
    print(f"  repeats:              {result.repeats}")
    print(f"  min MPSGraph speedup: {result.min_mpsgraph_speedup:.4g}x")
    if show_work_dir_budget and result.work_dir_budget is not None:
        print(
            "  work dir estimate:   "
            f"{format_bytes(result.work_dir_budget.estimated_work_dir_bytes)} / limit "
            f"{format_bytes(result.work_dir_budget.max_calibration_work_dir_bytes)}"
        )
    if (
        result.recommended_prefill_mpsgraph_min_batch_tokens is not None
        and result.recommended_prefill_mpsgraph_min_matrix_dim is not None
    ):
        print(
            "  recommended auto:     "
            f"tokens>={result.recommended_prefill_mpsgraph_min_batch_tokens} "
            f"dim>={result.recommended_prefill_mpsgraph_min_matrix_dim}"
        )
    else:
        print("  recommended auto:     none")
    suggested = result.suggested_prefill_runtime_policy_flags
    if suggested is not None:
        argv = suggested.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested args:       {' '.join(str(item) for item in argv)}")
    comparison = result.backend_comparison
    if isinstance(comparison, dict):
        recommended_backend = comparison.get("recommended_explicit_backend")
        print(
            "  backend recommendation:"
            f" {recommended_backend if recommended_backend is not None else 'none'}"
        )
        speedups = comparison.get("backend_speedup_vs_custom")
        if isinstance(speedups, dict) and speedups:
            parts = []
            for backend, value in speedups.items():
                if value is None:
                    parts.append(f"{backend}=n/a")
                else:
                    parts.append(f"{backend}={float(value):.3f}x")
            print(f"  backend speedups:     {', '.join(parts)}")
    for case in result.cases:
        print(
            "  case:                 "
            f"tokens={case.batch_tokens} shape={case.in_dim}x{case.out_dim} "
            f"custom={case.custom_elapsed_seconds:.6f}s "
            f"mpsgraph={case.mpsgraph_elapsed_seconds:.6f}s "
            f"mpsmatrix={case.mps_matrix_elapsed_seconds:.6f}s "
            f"speedup={case.mpsgraph_speedup:.3f}x "
            f"mpsmatrix_speedup={case.mps_matrix_speedup:.3f}x "
            f"winner={case.winner}"
        )
    print("  result:               ok")


def _print_resident_batch_rmsnorm(result: ResidentBatchRMSNormResult) -> None:
    print("LargerLM prefill resident batch RMSNorm")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  runner:               {result.runner_path}")
    print(f"  layer:                {result.layer}")
    print(f"  tensor:               {result.tensor}")
    print(f"  dtype:                {result.dtype}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  vector bytes:         {format_bytes(result.vector_bytes)}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  RMSNorm eps:          {result.rms_norm_eps:.9g}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prefill_attention_prefix(result: PrefillAttentionPrefixResult) -> None:
    print("LargerLM prefill attention prefix")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  runner:               {result.runner_path}")
    print(f"  layer:                {result.layer}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  q_a dim:              {result.q_a_dim}")
    print(f"  kv_a dim:             {result.kv_a_dim}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  norm output bytes:    {format_bytes(result.norm_output_bytes)}")
    print(f"  q_a output bytes:     {format_bytes(result.q_a_output_bytes)}")
    print(f"  kv_a output bytes:    {format_bytes(result.kv_a_output_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output dir:           {result.output_dir}")
    print("  result:               ok")


def _print_prefill_attention_projection_batch(
    result: PrefillAttentionProjectionBatchResult,
) -> None:
    print("LargerLM prefill attention projections")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  runner:               {result.runner_path}")
    print(f"  layer:                {result.layer}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  q_a dim:              {result.q_a_dim}")
    print(f"  q_b dim:              {result.q_b_dim}")
    print(f"  kv_a dim:             {result.kv_a_dim}")
    print(f"  kv lora dim:          {result.kv_lora_dim}")
    print(f"  kv rope dim:          {result.kv_rope_dim}")
    print(f"  kv_b dim:             {result.kv_b_dim}")
    print(f"  value source:         {result.attention_value_source}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  q_b output bytes:     {format_bytes(result.q_b_output_bytes)}")
    print(f"  kv_a lora bytes:      {format_bytes(result.kv_a_lora_bytes)}")
    print(f"  kv_a rope bytes:      {format_bytes(result.kv_a_rope_bytes)}")
    print(f"  kv_b output bytes:    {format_bytes(result.kv_b_output_bytes)}")
    print(f"  split peak:           {format_bytes(result.split_peak_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output dir:           {result.output_dir}")
    print("  result:               ok")


def _print_prefill_cache_write(result: PrefillCacheWriteResult) -> None:
    print("LargerLM prefill cache write")
    print(f"  cache layout:         {result.cache_layout_path}")
    print(f"  cache file:           {result.cache_file_path}")
    print(f"  input f32:            {result.input_path}")
    print(f"  layer:                {result.layer}")
    print(f"  start position:       {result.start_position}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  width:                {result.width}")
    print(f"  dtype:                {result.dtype}")
    print(f"  token stride:         {format_bytes(result.token_stride_bytes)}")
    print(f"  segment offset:       {result.segment_offset}")
    print(f"  first write offset:   {result.first_write_offset}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  encoded bytes:        {format_bytes(result.encoded_bytes)}")
    print(f"  encoder:              {result.encoder}")
    print(f"  write chunks:         {result.write_chunks}")
    print(f"  chunk tokens:         {result.write_chunk_tokens}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print("  result:               ok")


def _print_dsa_indexer_batch(result: DSAIndexerBatchResult) -> None:
    print("LargerLM DSA indexer batch")
    print(f"  resident layout:      {result.cache_write.resident_layout_path}")
    print(f"  cache layout:         {result.cache_write.cache_layout_path}")
    print(f"  cache file:           {result.cache_write.cache_file_path}")
    print(f"  layer:                {result.cache_write.layer}")
    print(f"  start position:       {result.cache_write.start_position}")
    print(f"  batch tokens:         {result.cache_write.batch_tokens}")
    print(f"  context length:       {result.topk.context_length}")
    print(f"  index top-k:          {result.topk.index_topk}")
    print(f"  index heads:          {result.topk.index_n_heads}")
    print(f"  index head dim:       {result.topk.index_head_dim}")
    print(f"  q lora dim:           {result.topk.q_lora_dim}")
    print(f"  qk rope dim:          {result.topk.qk_rope_dim}")
    print(f"  rope interleave:      {'yes' if result.topk.rope_interleave else 'no'}")
    print(f"  cache write bytes:    {format_bytes(result.cache_write.cache_write_bytes)}")
    print(f"  cache read bytes:     {format_bytes(result.topk.cache_read_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.topk.estimated_peak_bytes)}")
    if result.topk.output_indices_path is not None:
        print(f"  top-k json:           {result.topk.output_indices_path}")
    if result.topk.output_indices_u32_path is not None:
        print(f"  top-k u32:            {result.topk.output_indices_u32_path}")
    print("  result:               ok")


def _print_prefill_rope_batch(result: PrefillRopeBatchResult) -> None:
    print("LargerLM prefill RoPE batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  q_b input:            {result.q_b_input_path}")
    print(f"  k_rope input:         {result.k_rope_input_path}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  num heads:            {result.num_heads}")
    print(f"  qk nope dim:          {result.qk_nope_dim}")
    print(f"  rope dim:             {result.rope_dim}")
    print(f"  start position:       {result.start_position}")
    print(f"  theta:                {result.rope_theta:.9g}")
    print(f"  interleave:           {'yes' if result.rope_interleave else 'no'}")
    print(f"  q_b input bytes:      {format_bytes(result.q_b_input_bytes)}")
    print(f"  k_rope input bytes:   {format_bytes(result.k_rope_input_bytes)}")
    print(f"  q_nope bytes:         {format_bytes(result.q_nope_bytes)}")
    print(f"  q_rope bytes:         {format_bytes(result.q_rope_bytes)}")
    print(f"  k_rope bytes:         {format_bytes(result.k_rope_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output dir:           {result.output_dir}")
    print("  result:               ok")


def _print_prefill_mla_attention_batch(result: PrefillMLAAttentionBatchResult) -> None:
    print("LargerLM prefill MLA attention batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  cache layout:         {result.cache_layout_path}")
    print(f"  cache file:           {result.cache_file_path}")
    print(f"  layer:                {result.layer}")
    print(f"  context length:       {result.context_length}")
    print(f"  start position:       {result.start_position}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  num heads:            {result.num_heads}")
    print(f"  kv lora dim:          {result.kv_lora_dim}")
    print(f"  qk nope dim:          {result.qk_nope_dim}")
    print(f"  rope dim:             {result.rope_dim}")
    print(f"  v head dim:           {result.v_head_dim}")
    print(f"  attention scale:      {result.attention_scale:.9g}")
    print(f"  rope theta:           {result.rope_theta:.9g}")
    print(f"  interleave:           {'yes' if result.rope_interleave else 'no'}")
    print(f"  value source:         {result.attention_value_source}")
    if result.mla_kv_b_cache_dir is not None:
        print(f"  kv_b cache dir:       {result.mla_kv_b_cache_dir}")
    print(f"  MLA key cache:        {'yes' if result.mla_key_cache else 'no'}")
    if result.mla_key_cache_bytes:
        print(f"  key cache bytes:      {format_bytes(result.mla_key_cache_bytes)}")
    print(f"  MLA value cache:      {'yes' if result.mla_value_cache else 'no'}")
    if result.mla_value_cache_bytes:
        print(f"  value cache bytes:    {format_bytes(result.mla_value_cache_bytes)}")
    print(f"  q_nope bytes:         {format_bytes(result.q_nope_bytes)}")
    print(f"  q_rope bytes:         {format_bytes(result.q_rope_bytes)}")
    if result.indexed:
        print(f"  index top-k:          {result.index_topk}")
        print(f"  indices u32 bytes:    {format_bytes(result.indices_u32_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  cache read bytes:     {format_bytes(result.cache_read_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prefill_attention_output_batch(
    result: PrefillAttentionOutputBatchResult,
) -> None:
    print("LargerLM prefill attention output batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  layer:                {result.layer}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  attn value dim:       {result.attn_value_dim}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  attn value bytes:     {format_bytes(result.attn_value_bytes)}")
    print(f"  residual bytes:       {format_bytes(result.residual_bytes)}")
    print(f"  projection bytes:     {format_bytes(result.projection_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  residual add peak:    {format_bytes(result.residual_add_peak_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  projection f32:       {result.projection_path}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prefill_attention_block_batch(
    result: PrefillAttentionBlockBatchResult,
) -> None:
    print("LargerLM prefill attention block batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  cache layout:         {result.cache_layout_path}")
    print(f"  cache file:           {result.cache_file_path}")
    print(f"  layer:                {result.layer}")
    print(f"  context length:       {result.context_length}")
    print(f"  start position:       {result.start_position}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  num heads:            {result.num_heads}")
    print(f"  kv lora dim:          {result.kv_lora_dim}")
    print(f"  qk nope dim:          {result.qk_nope_dim}")
    print(f"  rope dim:             {result.rope_dim}")
    print(f"  v head dim:           {result.v_head_dim}")
    print(f"  value source:         {result.projections.attention_value_source}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  MLA key cache:        {'yes' if result.mla_key_cache else 'no'}")
    if result.mla_key_cache_bytes:
        print(f"  key cache bytes:      {format_bytes(result.mla_key_cache_bytes)}")
    print(f"  MLA value cache:      {'yes' if result.mla_value_cache else 'no'}")
    if result.mla_value_cache_bytes:
        print(f"  value cache bytes:    {format_bytes(result.mla_value_cache_bytes)}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  cache write peak:     {format_bytes(result.cache_write_peak_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output dir:           {result.output_dir}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prefill_dense_mlp_block_batch(
    result: PrefillDenseMLPBlockBatchResult,
) -> None:
    print("LargerLM prefill dense MLP block batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  layer:                {result.layer}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  intermediate dim:     {result.intermediate_dim}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  norm output bytes:    {format_bytes(result.norm_output_bytes)}")
    print(f"  gate output bytes:    {format_bytes(result.gate_output_bytes)}")
    print(f"  up output bytes:      {format_bytes(result.up_output_bytes)}")
    print(f"  SwiGLU output bytes:  {format_bytes(result.swiglu_output_bytes)}")
    print(f"  down output bytes:    {format_bytes(result.down_output_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  SwiGLU peak:          {format_bytes(result.swiglu_peak_bytes)}")
    print(f"  residual add peak:    {format_bytes(result.residual_add_peak_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  output dir:           {result.output_dir}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prefill_routed_mlp_block_batch(
    result: PrefillRoutedMLPBlockBatchResult,
) -> None:
    print("LargerLM prefill routed MLP block batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  expert layout:        {result.expert_layout_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  layer:                {result.layer}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  top-k:                {result.top_k}")
    print(f"  router score:         {result.router_score}")
    print(f"  include shared:       {'yes' if result.include_shared_expert else 'no'}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  expert read bytes:    {format_bytes(result.read_bytes)}")
    print(f"  runner calls:         {result.command_count}")
    if result.router_json_dir is not None:
        print(f"  router json dir:      {result.router_json_dir}")
    print(f"  output dir:           {result.output_dir}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prefill_staged_routed_mlp_block_batch(
    result: PrefillStagedRoutedMLPBlockBatchResult,
) -> None:
    print("LargerLM prefill staged routed MLP block batch")
    print(f"  runner:               {result.runner_path}")
    print(f"  expert layout:        {result.expert_layout_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  layer:                {result.layer}")
    print(f"  batch tokens:         {result.batch_tokens}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  top-k:                {result.top_k}")
    print(f"  router score:         {result.router_score}")
    print(f"  include shared:       {'yes' if result.include_shared_expert else 'no'}")
    print(f"  input bytes:          {format_bytes(result.input_bytes)}")
    print(f"  norm output bytes:    {format_bytes(result.norm_output_bytes)}")
    print(f"  staged bytes:         {format_bytes(result.staged_bytes)}")
    print(f"  compact stage bytes:  {format_bytes(result.compact_stage_bytes)}")
    print(
        "  compact materialized: "
        f"{format_bytes(result.compact_stage_materialized_bytes)}"
    )
    print(f"  compact storage:      {result.compact_stage_storage}")
    if result.static_capacity_binary_path is not None:
        print(
            "  static capacity:      "
            f"{result.static_capacity_path if result.static_capacity_path is not None else 'disabled'}"
        )
        print(f"  static cap bin:       {result.static_capacity_binary_path}")
        print(f"  static cap/expert:    {result.static_capacity_per_expert}")
        print(
            "  static slots:         "
            f"{result.static_capacity_used_slots}/{result.static_capacity_total_slots} "
            f"overflow={result.static_capacity_overflow_assignments}"
        )
        print(f"  static bin bytes:     {format_bytes(result.static_capacity_binary_bytes)}")
    if result.include_shared_expert:
        print(f"  shared output bytes:  {format_bytes(result.shared_output_bytes)}")
        print(f"  shared SwiGLU peak:   {format_bytes(result.shared_swiglu_peak_bytes)}")
        print(f"  shared add peak:      {format_bytes(result.shared_add_peak_bytes)}")
    print(f"  routed output bytes:  {format_bytes(result.routed_output_bytes)}")
    print(f"  output bytes:         {format_bytes(result.output_bytes)}")
    print(f"  residual add peak:    {format_bytes(result.residual_add_peak_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    print(f"  router calls:         {result.router_command_count}")
    print(f"  routed calls:         {result.routed_command_count}")
    print(f"  router json dir:      {result.router_json_dir}")
    print(f"  stage manifest:       {result.stage_manifest_path}")
    print(f"  output dir:           {result.output_dir}")
    print(f"  output f32:           {result.output_path}")
    print("  result:               ok")


def _print_prompt_prefill_result(result: PromptPrefillResult) -> None:
    print("LargerLM prompt prefill")
    print(f"  runner:               {result.runner_path}")
    print(f"  expert layout:        {result.expert_layout_path}")
    print(f"  resident layout:      {result.resident_layout_path}")
    print(f"  cache layout:         {result.cache_layout_path}")
    print(f"  cache file:           {result.cache_file_path}")
    print(f"  prompt tokens:        {len(result.prompt_token_ids)}")
    print(f"  start position:       {result.start_position}")
    print(f"  chunk tokens:         {result.chunk_tokens}")
    print(f"  chunks:               {result.chunk_count}")
    print(f"  layers:               {','.join(str(layer) for layer in result.layers)}")
    if result.dense_layers:
        print(f"  dense layers:         {','.join(str(layer) for layer in result.dense_layers)}")
    print(f"  hidden dim:           {result.hidden_dim}")
    print(f"  embedding read:       {format_bytes(result.total_embedding_read_bytes)}")
    print(f"  embedding output:     {format_bytes(result.total_embedding_output_bytes)}")
    print(f"  staged expert bytes:  {format_bytes(result.total_staged_bytes)}")
    print(f"  compact stage bytes:  {format_bytes(result.total_compact_stage_bytes)}")
    print(
        "  compact materialized: "
        f"{format_bytes(result.total_compact_stage_materialized_bytes)}"
    )
    if result.total_stage_plus_compact_bytes:
        print(
            "  stage+compact bytes:  "
            f"{format_bytes(result.total_stage_plus_compact_bytes)}"
        )
    if result.total_stage_plus_compact_materialized_bytes:
        print(
            "  stage+compact real:   "
            f"{format_bytes(result.total_stage_plus_compact_materialized_bytes)}"
        )
    if result.max_stage_plus_compact_bytes:
        print(
            "  max stage+compact:    "
            f"{format_bytes(result.max_stage_plus_compact_bytes)}"
        )
    if result.total_expert_stage_planned_read_bytes:
        print(
            "  stage serial read:    "
            f"{format_bytes(result.total_expert_stage_serial_read_bytes)}"
        )
        print(
            "  stage unique read:    "
            f"{format_bytes(result.total_expert_stage_unique_requested_bytes)}"
        )
        print(
            "  stage planned read:   "
            f"{format_bytes(result.total_expert_stage_planned_read_bytes)}"
        )
        if result.total_expert_stage_planned_read_seconds is not None:
            print(
                "  stage read time:      "
                f"{result.total_expert_stage_planned_read_seconds:.6g}s "
                f"@ {result.prefill_ssd_read_gib_per_second:.6g} GiB/s"
            )
            if result.prefill_max_routed_read_seconds > 0:
                print(
                    "  stage read cap:       "
                    f"{result.prefill_max_routed_read_seconds:.6g}s "
                    f"ok={result.total_expert_stage_read_seconds_ok}"
                )
        print(
            "  stage waste bytes:    "
            f"{format_bytes(result.total_expert_stage_waste_bytes)}"
        )
        print(
            "  stage savings:        "
            f"{format_bytes(result.total_expert_stage_coalesced_savings_bytes)}"
        )
        print(
            "  stage assignment amp: "
            f"{result.total_expert_stage_assignment_read_amplification:.3f}x"
        )
        print(
            "  stage unique amp:     "
            f"{result.total_expert_stage_unique_read_amplification:.3f}x"
        )
    if result.total_expert_stage_read_advice_attempted_ranges:
        print(
            "  stage read advice:    "
            f"ranges={result.total_expert_stage_read_advice_attempted_ranges} "
            f"calls={result.total_expert_stage_read_advice_calls} "
            f"bytes={format_bytes(result.total_expert_stage_read_advice_bytes)} "
            f"failures={result.total_expert_stage_read_advice_failures}"
        )
    print(f"  routed assignments:   {result.total_routed_expert_assignments}")
    print(f"  routed unique slots:  {result.total_routed_unique_expert_slots}")
    print(f"  max unique/call:      {result.max_routed_unique_experts_per_call}")
    print(f"  max tokens/expert:    {result.max_routed_tokens_per_expert}")
    print(f"  moe token block:      {result.moe_token_block}")
    if result.moe_token_block_mode_counts:
        modes = ", ".join(
            f"{mode}={count}"
            for mode, count in result.moe_token_block_mode_counts.items()
        )
        print(f"  moe block modes:      {modes}")
    if result.max_effective_moe_token_block:
        print(f"  max effective block:  {result.max_effective_moe_token_block}")
    if result.max_moe_max_expert_tokens:
        print(f"  max expert tokens:    {result.max_moe_max_expert_tokens}")
    if result.max_moe_batch_buffer_bytes:
        print(f"  max moe batch buffer: {format_bytes(result.max_moe_batch_buffer_bytes)}")
    if result.max_moe_estimated_peak_bytes:
        print(f"  max moe runner peak:  {format_bytes(result.max_moe_estimated_peak_bytes)}")
    print(
        "  persistent moe srv:   "
        f"{'yes' if result.persistent_moe_plan_server else 'no'}"
    )
    print(
        "  persistent linear srv:"
        f" {'yes' if result.persistent_resident_linear_server else 'no'}"
    )
    print(
        "  persistent proj srv:  "
        f"{'yes' if result.persistent_attention_projection_server else 'no'}"
    )
    print(
        "  persistent out srv:   "
        f"{'yes' if result.persistent_attention_output_server else 'no'}"
    )
    print(
        "  persistent shared srv:"
        f" {'yes' if result.persistent_shared_expert_server else 'no'}"
    )
    print(
        "  persistent rope srv:  "
        f"{'yes' if result.persistent_rope_split_server else 'no'}"
    )
    print(
        "  persistent mla srv:   "
        f"{'yes' if result.persistent_mla_attention_server else 'no'}"
    )
    print(
        "  persistent rms srv:   "
        f"{'yes' if result.persistent_rmsnorm_server else 'no'}"
    )
    print(f"  moe accumulator:      {result.moe_output_accumulator}")
    if result.moe_plan_server_plan_count or result.routed_moe_runner_command_count:
        print(f"  moe server plans:     {result.moe_plan_server_plan_count}")
        print(f"  moe runner launches:  {result.routed_moe_runner_command_count}")
    if result.static_capacity_per_expert is not None:
        print(f"  static cap request:   {result.static_capacity_per_expert}")
        print(f"  max static cap:       {result.max_static_capacity_per_expert}")
        print(f"  static used slots:    {result.total_static_capacity_used_slots}")
        print(f"  static total slots:   {result.total_static_capacity_slots}")
        print(f"  static overflow:      {result.total_static_capacity_overflow_assignments}")
        print(f"  static route bytes:   {format_bytes(result.total_static_capacity_binary_bytes)}")
    print(f"  estimated peak:       {format_bytes(result.estimated_peak_bytes)}")
    live = result.live_memory_budget
    print(
        "  live working set:     "
        f"{format_bytes(live.estimated_live_working_set_bytes)}"
    )
    if live.max_live_working_set_bytes is not None:
        print(f"  live cap:             {format_bytes(live.max_live_working_set_bytes)}")
    if live.system_available_bytes is not None:
        print(f"  system available mem: {format_bytes(live.system_available_bytes)}")
    if result.linear_backend_counts:
        backends = ", ".join(
            f"{backend}={count}"
            for backend, count in result.linear_backend_counts.items()
        )
        print(f"  linear backends:      {backends}")
    if result.linear_backend_elapsed_seconds:
        rates = result.linear_backend_estimated_tflops or {}
        timing = ", ".join(
            (
                f"{backend}={elapsed:.6g}s"
                + (
                    f" {float(rates[backend]):.3g} TFLOP/s"
                    if backend in rates
                    else ""
                )
            )
            for backend, elapsed in result.linear_backend_elapsed_seconds.items()
        )
        print(f"  linear backend time:  {timing}")
    if result.prefill_acceleration_coverage:
        coverage = result.prefill_acceleration_coverage
        print(f"  prefill accel ok:     {coverage.get('ok')}")
        print(
            "  prefill accel mats:   "
            f"{coverage.get('accelerated_matrix_count')}/"
            f"{coverage.get('matrix_count')}"
        )
        total_flops = int(coverage.get("total_estimated_flops") or 0)
        if total_flops:
            accelerated_flops = int(coverage.get("accelerated_estimated_flops") or 0)
            fraction = float(coverage.get("accelerated_flop_fraction") or 0.0)
            print(
                "  prefill accel FLOPs:  "
                f"{accelerated_flops:,}/{total_flops:,} ({fraction:.1%})"
            )
    if result.prefill_acceleration_frontier:
        suggested = result.prefill_acceleration_frontier.get("suggested_guard_flags")
        if isinstance(suggested, dict):
            argv = suggested.get("argv")
            if isinstance(argv, (list, tuple)) and argv:
                print(
                    "  suggested accel args:"
                    f" {' '.join(str(item) for item in argv)}"
                )
    if result.total_linear_matrix_scratch_bytes:
        print(
            "  linear matrix scratch:"
            f" {format_bytes(result.total_linear_matrix_scratch_bytes)}"
        )
    if result.max_linear_matrix_scratch_bytes:
        print(
            "  max linear scratch:   "
            f"{format_bytes(result.max_linear_matrix_scratch_bytes)}"
        )
    if result.total_linear_matrix_f32_bytes:
        print(
            "  linear f32 matrix:    "
            f"{format_bytes(result.total_linear_matrix_f32_bytes)}"
        )
    if result.total_linear_matrix_raw_conversion_bytes:
        print(
            "  linear raw conversion:"
            f" {format_bytes(result.total_linear_matrix_raw_conversion_bytes)}"
        )
    print(f"  last hidden f32:      {result.output_last_hidden_path}")
    if result.output_final_chunk_path is not None:
        print(f"  final chunk f32:      {result.output_final_chunk_path}")
    print(f"  work dir:             {result.work_dir}")
    print(f"  kept work dir:        {'yes' if result.kept_work_dir else 'no'}")
    print("  result:               ok")


def _prefill_linear_batch(args: argparse.Namespace) -> int:
    result = run_resident_batch_linear(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        tensor_suffix=args.tensor_suffix,
        input_f32_path=args.input_f32,
        output_f32_path=args.output_f32,
        batch_tokens=args.batch_tokens,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        prefill_linear_backend=args.prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=args.prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=args.prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_resident_batch_linear(result)
    return 0


def _prefill_linear_calibrate(args: argparse.Namespace) -> int:
    result = run_resident_linear_calibration(
        runner_path=args.runner,
        batch_token_values=args.batch_tokens,
        matrix_dim_values=args.matrix_dims,
        matrix_shapes=args.matrix_shapes,
        repeats=args.repeats,
        min_mpsgraph_speedup=args.min_mpsgraph_speedup,
        max_calibration_case_mib=args.max_calibration_case_mib,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        max_calibration_work_dir_mib=args.max_calibration_work_dir_mib,
        calibration_work_dir_free_margin_mib=args.calibration_work_dir_free_margin_mib,
        matrix_dtype=args.matrix_dtype,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        echo_runner_output=args.echo_runner_output,
    )
    if args.write_launch_profile is not None:
        _write_launch_profile_file(
            args.write_launch_profile,
            result.suggested_launch_profile,
        )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_resident_linear_calibration(result)
    return 0


def _prefill_rmsnorm_batch(args: argparse.Namespace) -> int:
    result = run_resident_batch_rmsnorm(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        norm_suffix=args.norm_suffix,
        input_f32_path=args.input_f32,
        output_f32_path=args.output_f32,
        batch_tokens=args.batch_tokens,
        rms_norm_eps=args.rms_norm_eps,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_resident_batch_rmsnorm(result)
    return 0


def _prefill_attention_prefix(args: argparse.Namespace) -> int:
    result = run_prefill_attention_prefix_batch(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        input_f32_path=args.input_f32,
        output_dir=args.output_dir,
        batch_tokens=args.batch_tokens,
        norm_suffix=args.norm_suffix,
        q_a_suffix=args.q_a_suffix,
        kv_a_suffix=args.kv_a_suffix,
        rms_norm_eps=args.rms_norm_eps,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        prefill_linear_backend=args.prefill_linear_backend,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_attention_prefix(result)
    return 0


def _prefill_attention_projections(args: argparse.Namespace) -> int:
    result = run_prefill_attention_projection_batch(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        input_f32_path=args.input_f32,
        output_dir=args.output_dir,
        batch_tokens=args.batch_tokens,
        norm_suffix=args.norm_suffix,
        q_a_suffix=args.q_a_suffix,
        q_a_norm_suffix=args.q_a_norm_suffix,
        q_b_suffix=args.q_b_suffix,
        kv_a_suffix=args.kv_a_suffix,
        kv_a_norm_suffix=args.kv_a_norm_suffix,
        kv_b_suffix=args.kv_b_suffix,
        rms_norm_eps=args.rms_norm_eps,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        prefill_linear_backend=args.prefill_linear_backend,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_attention_projection_batch(result)
    return 0


def _prefill_cache_write(args: argparse.Namespace) -> int:
    result = write_prefill_kv_cache_batch(
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        layer=args.layer,
        input_f32_path=args.input_f32,
        start_position=args.start_position,
        batch_tokens=args.batch_tokens,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_write_mib=args.max_cache_write_mib,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_cache_write(result)
    return 0


def _dsa_indexer_batch(args: argparse.Namespace) -> int:
    result = run_dsa_indexer_batch(
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        layer=args.layer,
        hidden_f32_path=args.hidden_f32,
        q_resid_f32_path=args.q_resid_f32,
        output_indices_path=args.output_indices_json,
        output_indices_u32_path=args.output_indices_u32,
        start_position=args.start_position,
        batch_tokens=args.batch_tokens,
        context_length=args.context_length,
        index_topk=args.index_topk,
        index_n_heads=args.index_n_heads,
        qk_rope_dim=args.qk_rope_dim,
        rope_theta=args.rope_theta,
        rope_interleave=args.rope_interleave,
        layer_norm_eps=args.layer_norm_eps,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_write_mib=args.max_cache_write_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        collect_topk_indices=(
            args.include_topk_in_json or args.output_indices_json is not None
        ),
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_dsa_indexer_batch(result)
    return 0


def _prefill_rope_batch(args: argparse.Namespace) -> int:
    result = run_prefill_rope_batch(
        runner_path=args.runner,
        q_b_f32_path=args.q_b_f32,
        k_rope_f32_path=args.k_rope_f32,
        output_dir=args.output_dir,
        batch_tokens=args.batch_tokens,
        num_heads=args.num_heads,
        qk_nope_dim=args.qk_nope_dim,
        rope_dim=args.rope_dim,
        start_position=args.start_position,
        rope_theta=args.rope_theta,
        rope_interleave=args.rope_interleave,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_rope_batch(result)
    return 0


def _prefill_mla_attention_batch(args: argparse.Namespace) -> int:
    result = run_prefill_mla_attention_batch(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        layer=args.layer,
        q_nope_f32_path=args.q_nope_f32,
        q_rope_f32_path=args.q_rope_f32,
        indices_u32_path=args.indices_u32,
        output_f32_path=args.output_f32,
        context_length=args.context_length,
        start_position=args.start_position,
        batch_tokens=args.batch_tokens,
        index_topk=args.index_topk,
        num_heads=args.num_heads,
        qk_nope_dim=args.qk_nope_dim,
        rope_dim=args.rope_dim,
        v_head_dim=args.v_head_dim,
        mla_kv_b_cache_dir=args.mla_kv_b_cache_dir,
        mla_key_cache=args.mla_key_cache,
        mla_value_cache=args.mla_value_cache,
        kv_lora_dim=args.kv_lora_dim,
        cache_position_offset=args.cache_position_offset,
        attention_scale=args.attention_scale,
        rope_theta=args.rope_theta,
        rope_interleave=args.rope_interleave,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_mla_attention_batch(result)
    return 0


def _prefill_attention_output_batch(args: argparse.Namespace) -> int:
    result = run_prefill_attention_output_batch(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        attn_value_f32_path=args.attn_value_f32,
        residual_f32_path=args.residual_f32,
        output_f32_path=args.output_f32,
        batch_tokens=args.batch_tokens,
        o_proj_suffix=args.o_proj_suffix,
        projection_f32_path=args.projection_f32,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        prefill_linear_backend=args.prefill_linear_backend,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_attention_output_batch(result)
    return 0


def _prefill_attention_block_batch(args: argparse.Namespace) -> int:
    result = run_prefill_attention_block_batch(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        layer=args.layer,
        input_f32_path=args.input_f32,
        output_dir=args.output_dir,
        output_f32_path=args.output_f32,
        context_length=args.context_length,
        start_position=args.start_position,
        batch_tokens=args.batch_tokens,
        num_heads=args.num_heads,
        qk_nope_dim=args.qk_nope_dim,
        rope_dim=args.rope_dim,
        v_head_dim=args.v_head_dim,
        mla_kv_b_cache_dir=args.mla_kv_b_cache_dir,
        mla_key_cache=args.mla_key_cache,
        mla_value_cache=args.mla_value_cache,
        kv_lora_dim=args.kv_lora_dim,
        cache_position_offset=args.cache_position_offset,
        attention_scale=args.attention_scale,
        rope_theta=args.rope_theta,
        rope_interleave=args.rope_interleave,
        rms_norm_eps=args.rms_norm_eps,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_write_mib=args.max_cache_write_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        prefill_linear_backend=args.prefill_linear_backend,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_attention_block_batch(result)
    return 0


def _prefill_dense_mlp_block_batch(args: argparse.Namespace) -> int:
    result = run_prefill_dense_mlp_block_batch(
        runner_path=args.runner,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        input_f32_path=args.input_f32,
        output_dir=args.output_dir,
        output_f32_path=args.output_f32,
        batch_tokens=args.batch_tokens,
        norm_suffix=args.norm_suffix,
        gate_suffix=args.gate_suffix,
        up_suffix=args.up_suffix,
        down_suffix=args.down_suffix,
        rms_norm_eps=args.rms_norm_eps,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        prefill_linear_backend=args.prefill_linear_backend,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_dense_mlp_block_batch(result)
    return 0


def _prefill_routed_mlp_block_batch(args: argparse.Namespace) -> int:
    result = run_prefill_routed_mlp_block_batch(
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        input_f32_path=args.input_f32,
        output_dir=args.output_dir,
        output_f32_path=args.output_f32,
        batch_tokens=args.batch_tokens,
        top_k=args.top_k,
        max_k=args.max_k,
        router_score=args.router_score,
        routed_scaling_factor=args.routed_scaling_factor,
        norm_topk_prob=args.norm_topk_prob,
        no_norm_topk_prob=args.no_norm_topk_prob,
        router_n_group=args.router_n_group,
        router_topk_group=args.router_topk_group,
        ignore_router_bias=args.ignore_router_bias,
        include_shared_expert=args.include_shared_expert,
        rms_norm_eps=args.rms_norm_eps,
        max_slot_mib=args.max_slot_mib,
        max_router_mib=args.max_router_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        expert_read_advise_merge_gap_kib=args.expert_read_advise_merge_gap_kib,
        expert_read_advise_align_kib=args.expert_read_advise_align_kib,
        router_json_dir=args.router_json_dir,
        keep_token_files=args.keep_token_files,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_routed_mlp_block_batch(result)
    return 0


def _prefill_staged_routed_mlp_block_batch(args: argparse.Namespace) -> int:
    result = run_prefill_staged_routed_mlp_block_batch(
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        resident_layout_path=args.resident_layout,
        layer=args.layer,
        input_f32_path=args.input_f32,
        output_dir=args.output_dir,
        output_f32_path=args.output_f32,
        batch_tokens=args.batch_tokens,
        top_k=args.top_k,
        max_k=args.max_k,
        router_score=args.router_score,
        routed_scaling_factor=args.routed_scaling_factor,
        norm_topk_prob=args.norm_topk_prob,
        no_norm_topk_prob=args.no_norm_topk_prob,
        router_n_group=args.router_n_group,
        router_topk_group=args.router_topk_group,
        ignore_router_bias=args.ignore_router_bias,
        include_shared_expert=args.include_shared_expert,
        rms_norm_eps=args.rms_norm_eps,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_slot_mib=args.max_slot_mib,
        max_router_mib=args.max_router_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        expert_stage_merge_gap_kib=args.expert_stage_merge_gap_kib,
        expert_stage_align_kib=args.expert_stage_align_kib,
        max_stage_mib=args.max_stage_mib,
        max_compact_stage_mib=args.max_compact_stage_mib,
        copy_chunk_mib=args.copy_chunk_mib,
        stage_disk_safety_margin_bytes=int(args.stage_disk_margin_mib * 1024**2),
        prefill_ssd_read_gib_per_second=args.prefill_ssd_read_gib_per_second,
        prefill_max_routed_read_seconds=args.prefill_max_routed_read_seconds,
        expert_stage_max_raw_ranges=args.expert_stage_max_raw_ranges,
        expert_stage_max_coalesced_ranges=args.expert_stage_max_coalesced_ranges,
        expert_stage_tiling=args.expert_stage_tiling,
        moe_token_block=args.moe_token_block,
        moe_output_accumulator=args.moe_output_accumulator,
        static_capacity_per_expert=args.static_capacity_per_expert,
        static_capacity_output_json_path=args.static_capacity_output_json,
        static_capacity_output_bin_path=args.static_capacity_output_bin,
        write_static_capacity_json=not args.no_static_capacity_json,
        allow_static_capacity_overflow=args.allow_static_capacity_overflow,
        prefill_linear_backend=args.prefill_linear_backend,
        router_hybrid_margin_threshold=args.prefill_router_hybrid_margin_threshold,
        keep_token_files=args.keep_token_files,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_staged_routed_mlp_block_batch(result)
    return 0


def _prefill_backend(args: argparse.Namespace) -> int:
    capability = inspect_prefill_backend(
        sdk_path=args.sdk_path,
        probe_binary=args.probe_binary,
        run_host_probe=not args.no_host_probe,
        compile_mpp_probe=args.compile_mpp_probe,
        run_mpp_probe=args.run_mpp_probe,
        run_mpsgraph_probe=args.run_mpsgraph_probe,
        probe_timeout_seconds=args.probe_timeout_seconds,
    )
    if args.write_report is not None:
        _write_json_file_atomic(args.write_report, capability)
    if args.json:
        print(json.dumps(capability, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prefill_backend(capability)
    return 0


def _plan_expert_io(args: argparse.Namespace) -> int:
    plan = plan_expert_io(
        args.expert_layout,
        layer=args.layer,
        expert_ids=_parse_expert_ids(args.experts),
        merge_gap_bytes=int(args.merge_gap_kib * 1024),
        align_bytes=int(args.align_kib * 1024),
    )
    if args.json:
        print(json.dumps(plan, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_expert_io_plan(plan)
    return 0


def _plan_batch_expert_io(args: argparse.Namespace) -> int:
    plan = plan_batch_expert_io(
        args.expert_layout,
        layer=args.layer,
        router_json_dir=args.router_json_dir,
        router_json_glob=args.router_json_glob,
        merge_gap_bytes=int(args.merge_gap_kib * 1024),
        align_bytes=int(args.align_kib * 1024),
        ssd_read_gib_per_second=args.ssd_read_gib_s,
    )
    tiling_plan = None
    if (
        args.tile_max_stage_mib is not None
        or args.tile_max_compact_stage_mib is not None
    ):
        if args.tile_max_stage_mib is None or args.tile_max_compact_stage_mib is None:
            raise ExpertIOPlanError(
                "--tile-max-stage-mib and --tile-max-compact-stage-mib "
                "must be supplied together"
            )
        tiling_plan = plan_batch_expert_io_tiles(
            args.expert_layout,
            layer=args.layer,
            router_json_dir=args.router_json_dir,
            router_json_glob=args.router_json_glob,
            merge_gap_bytes=int(args.merge_gap_kib * 1024),
            align_bytes=int(args.align_kib * 1024),
            max_stage_mib=args.tile_max_stage_mib,
            max_compact_stage_mib=args.tile_max_compact_stage_mib,
            ssd_read_gib_per_second=args.ssd_read_gib_s,
        )
    capacity_plan = None
    if args.static_capacity_per_expert is not None:
        capacity_plan = plan_static_expert_capacity(
            plan,
            capacity_per_expert=args.static_capacity_per_expert,
        )
    if args.static_capacity_output_json is not None:
        if capacity_plan is None:
            raise ExpertIOPlanError(
                "--static-capacity-output-json requires --static-capacity-per-expert"
            )
        write_static_expert_capacity_plan(
            capacity_plan,
            args.static_capacity_output_json,
            allow_overflow=args.allow_static_capacity_overflow,
        )
    if args.static_capacity_output_bin is not None:
        if capacity_plan is None:
            raise ExpertIOPlanError(
                "--static-capacity-output-bin requires --static-capacity-per-expert"
            )
        write_static_expert_capacity_binary(
            capacity_plan,
            args.static_capacity_output_bin,
            allow_overflow=args.allow_static_capacity_overflow,
        )
    if args.json:
        payload: object = plan
        if capacity_plan is not None or tiling_plan is not None:
            payload = {"batch_expert_io_plan": plan}
            if capacity_plan is not None:
                payload["static_capacity_plan"] = capacity_plan
            if tiling_plan is not None:
                payload["batch_expert_io_tiling_plan"] = tiling_plan
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_batch_expert_io_plan(plan)
        if tiling_plan is not None:
            _print_batch_expert_io_tiling_plan(tiling_plan)
        if capacity_plan is not None:
            _print_static_expert_capacity_plan(capacity_plan)
    return 0


def _validate_static_capacity_bin(args: argparse.Namespace) -> int:
    result = validate_static_expert_capacity_binary(args.path)
    expected = {
        "batch_tokens": args.expect_batch_tokens,
        "expert_count": args.expect_expert_count,
        "capacity_per_expert": args.expect_capacity_per_expert,
        "total_assignments": args.expect_total_assignments,
        "used_slots": args.expect_used_slots,
        "overflow_records": args.expect_overflow_records,
    }
    for field, value in expected.items():
        if value is not None and getattr(result, field) != value:
            raise ExpertIOPlanError(
                f"static capacity binary {field}={getattr(result, field)} "
                f"does not match expected {value}"
            )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_static_capacity_binary_validation(result)
    return 0


def _stage_batch_experts(args: argparse.Namespace) -> int:
    result = stage_batch_experts(
        args.expert_layout,
        layer=args.layer,
        router_json_dir=args.router_json_dir,
        router_json_glob=args.router_json_glob,
        stage_file_path=args.stage_file,
        manifest_path=args.manifest,
        merge_gap_bytes=int(args.merge_gap_kib * 1024),
        align_bytes=int(args.align_kib * 1024),
        max_stage_mib=args.max_stage_mib,
        copy_chunk_mib=args.copy_chunk_mib,
        disk_safety_margin_bytes=int(args.stage_disk_margin_mib * 1024**2),
        ssd_read_gib_per_second=args.ssd_read_gib_s,
        max_read_seconds=args.max_read_seconds,
        max_raw_ranges=args.max_raw_ranges,
        max_coalesced_ranges=args.max_coalesced_ranges,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_batch_expert_stage(result)
    return 0


def _run_staged_routed_moe_batch(args: argparse.Namespace) -> int:
    result = run_staged_routed_moe_batch(
        runner_path=args.runner,
        stage_manifest_path=args.stage_manifest,
        input_f32_path=args.input_f32,
        output_f32_path=args.output_f32,
        output_dir=args.output_dir,
        max_compact_stage_mib=args.max_compact_stage_mib,
        copy_chunk_mib=args.copy_chunk_mib,
        disk_safety_margin_bytes=int(args.compact_stage_disk_margin_mib * 1024**2),
        max_slot_mib=args.max_slot_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        moe_token_block=args.moe_token_block,
        static_capacity_per_expert=args.static_capacity_per_expert,
        static_capacity_output_json_path=args.static_capacity_output_json,
        static_capacity_output_bin_path=args.static_capacity_output_bin,
        write_static_capacity_json=not args.no_static_capacity_json,
        allow_static_capacity_overflow=args.allow_static_capacity_overflow,
        keep_token_files=args.keep_token_files,
        echo_runner_output=not args.quiet_runner,
        moe_output_accumulator=args.moe_output_accumulator,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_staged_routed_moe_batch(result)
    return 0


def _run_staged_routed_moe_batch_plan(args: argparse.Namespace) -> int:
    result = run_staged_routed_moe_batch_plan(
        runner_path=args.runner,
        plan_path=args.plan_json,
        echo_runner_output=not args.quiet_runner,
        moe_output_accumulator=args.moe_output_accumulator,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_staged_routed_moe_batch_plan(result)
    return 0


def _run_staged_routed_moe_batch_plan_server(args: argparse.Namespace) -> int:
    result = run_staged_routed_moe_batch_plan_server(
        runner_path=args.runner,
        plan_paths=args.plan_json,
        echo_runner_output=not args.quiet_runner,
        moe_output_accumulator=args.moe_output_accumulator,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_staged_routed_moe_batch_plan_server(result)
    return 0


def _run_tiled_staged_routed_moe_batch(args: argparse.Namespace) -> int:
    result = run_tiled_staged_routed_moe_batch(
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        layer=args.layer,
        router_json_dir=args.router_json_dir,
        router_json_glob=args.router_json_glob,
        input_f32_path=args.input_f32,
        output_f32_path=args.output_f32,
        output_dir=args.output_dir,
        merge_gap_bytes=int(args.merge_gap_kib * 1024),
        align_bytes=int(args.align_kib * 1024),
        max_stage_mib=args.max_stage_mib,
        max_compact_stage_mib=args.max_compact_stage_mib,
        copy_chunk_mib=args.copy_chunk_mib,
        disk_safety_margin_bytes=int(args.stage_disk_margin_mib * 1024**2),
        ssd_read_gib_per_second=args.ssd_read_gib_s,
        max_read_seconds=args.max_read_seconds,
        max_raw_ranges=args.max_raw_ranges,
        max_coalesced_ranges=args.max_coalesced_ranges,
        max_slot_mib=args.max_slot_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        moe_token_block=args.moe_token_block,
        static_capacity_per_expert=args.static_capacity_per_expert,
        write_static_capacity_json=not args.no_static_capacity_json,
        allow_static_capacity_overflow=args.allow_static_capacity_overflow,
        keep_token_files=args.keep_token_files,
        echo_runner_output=not args.quiet_runner,
        moe_output_accumulator=args.moe_output_accumulator,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_tiled_staged_routed_moe_batch(result)
    return 0


def _validate_baseline(args: argparse.Namespace) -> int:
    result = validate_baseline(args.baseline)
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_baseline_validation(result)
    return 0


def _preflight_glm(args: argparse.Namespace) -> int:
    report = preflight_glm_checkpoint(
        args.model,
        tokenizer_path=args.tokenizer,
        tokenizer_backend=args.tokenizer_backend,
        trust_remote_code=args.trust_remote_code,
        load_tokenizer_backend=args.load_tokenizer,
        quant_bits=args.quant_bits,
        group_size=args.group_size,
        quantize_raw_to_int4=args.quantize_bf16_affine_int4,
        require_public_glm_5_2_shape=bool(args.require_public_glm_5_2_shape),
        max_context_tokens=args.max_context_tokens,
        max_cache_bytes=(
            _scaled_bytes_arg(args.max_cache_gib, name="--max-cache-gib", scale=1024**3)
            if args.max_cache_gib is not None
            else None
        ),
        output_dir=args.output_dir,
        disk_safety_margin_bytes=_scaled_bytes_arg(
            args.disk_margin_gib,
            name="--disk-margin-gib",
            scale=1024**3,
        ),
        unified_memory_bytes=(
            _scaled_bytes_arg(
                args.unified_memory_gib,
                name="--unified-memory-gib",
                scale=1024**3,
            )
            if args.unified_memory_gib is not None
            else None
        ),
        system_reserve_bytes=(
            _scaled_bytes_arg(
                args.system_reserve_gib,
                name="--system-reserve-gib",
                scale=1024**3,
            )
            if args.system_reserve_gib is not None
            else None
        ),
        runtime_buffer_bytes=_scaled_bytes_arg(
            args.runtime_buffer_gib,
            name="--runtime-buffer-gib",
            scale=1024**3,
        ),
        page_cache_fraction=args.page_cache_fraction,
        cold_read_gib_per_second=args.cold_read_gib_s,
        prefer_header_manifest=args.metadata_only,
    )
    if args.write_report is not None:
        _write_json_file_atomic(args.write_report, report)
    if args.json:
        print(json.dumps(report, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_preflight_report(report)
    return 0 if report.ok else 1


def _checkpoint_status(args: argparse.Namespace) -> int:
    status = inspect_checkpoint_artifact(
        args.model,
        repo=args.repo,
        revision=args.revision,
        endpoint=args.endpoint,
        download_disk_safety_margin_bytes=_scaled_bytes_arg(
            args.download_disk_margin_gib,
            name="--download-disk-margin-gib",
            scale=1024**3,
        ),
        verify_local_headers=args.verify_local_headers,
    )
    next_bringup_command_override = _next_bringup_range_download_command(
        status,
        external_download_json_path=args.write_external_download_json,
    )
    _write_checkpoint_missing_shard_outputs(
        status,
        json_path=args.write_missing_shards_json,
        urls_path=args.write_missing_shards_urls,
    )
    _write_checkpoint_external_download_outputs(
        status,
        json_path=args.write_external_download_json,
        shell_path=args.write_external_download_sh,
    )
    _write_checkpoint_next_bringup_outputs(
        status,
        json_path=args.write_next_bringup_json,
        shell_path=args.write_next_bringup_sh,
        command_override=next_bringup_command_override,
    )
    if args.write_status_json:
        _write_json_file_atomic(args.write_status_json, status)
    if args.json:
        print(json.dumps(status, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_checkpoint_artifact_status(status)
    if args.require_complete and (
        not status.download_complete_proven
        or (
            status.local_header_check_requested
            and status.local_header_error_count > 0
        )
    ):
        return 1
    if args.require_download_disk_ok and status.download_disk_ok is not True:
        return 1
    if args.require_clean and not status.artifact_clean:
        return 1
    if args.require_preflight_ready and not status.can_run_metadata_preflight:
        return 1
    return 0


def _result_summary(args: argparse.Namespace) -> int:
    if args.top < 1:
        print("--top must be >= 1", file=sys.stderr)
        return 2
    try:
        summary = summarize_result_file(args.result, top_limit=args.top)
    except ResultSummaryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, default=_json_default, indent=2, sort_keys=True))
    else:
        print(format_result_summary_text(summary))
    return 0


def _result_compare(args: argparse.Namespace) -> int:
    if args.top < 1:
        print("--top must be >= 1", file=sys.stderr)
        return 2
    if not math.isfinite(args.system_slowdown_ratio) or args.system_slowdown_ratio <= 1.0:
        print("--system-slowdown-ratio must be > 1", file=sys.stderr)
        return 2
    try:
        comparison = compare_result_files(
            args.baseline,
            args.candidate,
            top_limit=args.top,
            system_slowdown_ratio=args.system_slowdown_ratio,
            allow_prefill_policy_change=bool(args.allow_prefill_policy_change),
        )
    except ResultSummaryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(comparison, default=_json_default, indent=2, sort_keys=True))
    else:
        print(format_result_comparison_text(comparison))
    if args.require_candidate_promotable:
        recommendation = comparison.get("profile_recommendation")
        candidate_promotable = (
            isinstance(recommendation, dict)
            and recommendation.get("candidate_promotable") is True
        )
        if not candidate_promotable:
            decision = (
                recommendation.get("decision")
                if isinstance(recommendation, dict)
                else None
            )
            reasons = (
                recommendation.get("reasons")
                if isinstance(recommendation, dict)
                else None
            )
            reason_text = (
                ", ".join(str(reason) for reason in reasons)
                if isinstance(reasons, list)
                else "profile recommendation did not mark candidate promotable"
            )
            print(
                "candidate result is not promotable"
                f" (decision={decision}; reasons={reason_text})",
                file=sys.stderr,
            )
            return 1
    return 0


def _result_bakeoff(args: argparse.Namespace) -> int:
    if args.top < 1:
        print("--top must be >= 1", file=sys.stderr)
        return 2
    if not math.isfinite(args.system_slowdown_ratio) or args.system_slowdown_ratio <= 1.0:
        print("--system-slowdown-ratio must be > 1", file=sys.stderr)
        return 2
    promote_only_replay_files_ready = (
        bool(args.promote_only_replay_files_ready)
        or bool(args.require_selected_replay_ready)
        or bool(args.write_selected_replay_json)
        or bool(args.write_selected_replay_script)
    )
    try:
        bakeoff = result_bakeoff_files(
            args.baseline,
            args.candidates,
            top_limit=args.top,
            system_slowdown_ratio=args.system_slowdown_ratio,
            promote_only_replay_files_ready=promote_only_replay_files_ready,
            allow_prefill_policy_change=bool(args.allow_prefill_policy_change),
        )
    except ResultSummaryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(bakeoff, default=_json_default, indent=2, sort_keys=True))
    else:
        print(format_result_bakeoff_text(bakeoff))
    if args.write_bakeoff_json:
        _write_json_file_atomic(args.write_bakeoff_json, bakeoff)
    selected_replay_payload: dict[str, object] | None = None
    if (
        args.require_selected_replay_ready
        or args.write_selected_replay_json
        or args.write_selected_replay_script
    ):
        selected_replay_payload = _bakeoff_selected_replay_payload(bakeoff)
    if args.write_selected_replay_json:
        _write_json_file_atomic(
            args.write_selected_replay_json,
            selected_replay_payload,
        )
    if args.write_selected_replay_script:
        _write_selected_replay_script(
            args.write_selected_replay_script,
            selected_replay_payload or _bakeoff_selected_replay_payload(bakeoff),
            selected_replay_json_path=args.write_selected_replay_json,
        )
    if args.require_winner and not isinstance(bakeoff.get("winner"), dict):
        print("no promotable candidate result found", file=sys.stderr)
        return 1
    return 0


def _prepare_glm(args: argparse.Namespace) -> int:
    report = prepare_glm_checkpoint(
        args.model,
        output_dir=args.output_dir,
        max_context_tokens=args.max_context_tokens,
        auto_context_from_budget=args.auto_context_from_budget,
        execute=args.execute,
        force=args.force,
        tokenizer_path=args.tokenizer,
        tokenizer_backend=args.tokenizer_backend,
        trust_remote_code=args.trust_remote_code,
        load_tokenizer_backend=args.load_tokenizer,
        quant_bits=args.quant_bits,
        group_size=args.group_size,
        quantize_raw_to_int4=args.quantize_bf16_affine_int4,
        require_public_glm_5_2_shape=bool(args.require_public_glm_5_2_shape),
        cache_dtype=args.cache_dtype,
        cache_alignment=args.cache_alignment,
        max_cache_bytes=(
            _scaled_bytes_arg(args.max_cache_gib, name="--max-cache-gib", scale=1024**3)
            if args.max_cache_gib is not None
            else None
        ),
        disk_safety_margin_bytes=_scaled_bytes_arg(
            args.disk_margin_gib,
            name="--disk-margin-gib",
            scale=1024**3,
        ),
        chunk_size=_scaled_bytes_arg(args.chunk_mib, name="--chunk-mib", scale=1024**2),
        max_chunk_size=_scaled_bytes_arg(
            args.max_chunk_mib,
            name="--max-chunk-mib",
            scale=1024**2,
        ),
        max_pack_heap_bytes=_scaled_bytes_arg(
            args.max_pack_heap_mib,
            name="--max-pack-heap-mib",
            scale=1024**2,
        ),
        unified_memory_bytes=(
            _scaled_bytes_arg(
                args.unified_memory_gib,
                name="--unified-memory-gib",
                scale=1024**3,
            )
            if args.unified_memory_gib is not None
            else None
        ),
        system_reserve_bytes=(
            _scaled_bytes_arg(
                args.system_reserve_gib,
                name="--system-reserve-gib",
                scale=1024**3,
            )
            if args.system_reserve_gib is not None
            else None
        ),
        runtime_buffer_bytes=_scaled_bytes_arg(
            args.runtime_buffer_gib,
            name="--runtime-buffer-gib",
            scale=1024**3,
        ),
        page_cache_fraction=args.page_cache_fraction,
        cold_read_gib_per_second=args.cold_read_gib_s,
        auto_cold_read_benchmark=bool(args.auto_cold_read_benchmark),
        cold_read_benchmark_bytes=_scaled_bytes_arg(
            args.cold_read_benchmark_mib,
            name="--cold-read-benchmark-mib",
            scale=1024**2,
        ),
        cold_read_benchmark_chunk_bytes=_scaled_bytes_arg(
            args.cold_read_benchmark_chunk_mib,
            name="--cold-read-benchmark-chunk-mib",
            scale=1024**2,
        ),
        prepare_flags=_prepare_flags_provenance(args.apply_prepare_flags),
        prefer_header_manifest=args.metadata_only,
    )
    if args.write_report is not None:
        _write_json_file_atomic(args.write_report, report)
    if args.json:
        print(json.dumps(report, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prepare_report(report)
    return 0 if report.ok else 1


def _plan_decode_cache(args: argparse.Namespace) -> int:
    cfg = load_config(args.model)
    max_cache_bytes = (
        _scaled_bytes_arg(args.max_cache_gib, name="--max-cache-gib", scale=1024**3)
        if args.max_cache_gib is not None
        else None
    )
    layout = build_decode_cache_layout(
        cfg,
        max_context_tokens=args.max_context_tokens,
        dtype=args.dtype,
        alignment=args.alignment,
        max_cache_bytes=max_cache_bytes,
    )
    payload = layout.to_json()
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_decode_cache_layout(layout)
        if args.output:
            print(f"  layout json:           {args.output}")
    return 0


def _init_decode_cache(args: argparse.Namespace) -> int:
    max_cache_bytes = (
        _scaled_bytes_arg(args.max_cache_gib, name="--max-cache-gib", scale=1024**3)
        if args.max_cache_gib is not None
        else None
    )
    result = init_decode_cache_file(
        args.layout,
        args.output,
        force=args.force,
        max_cache_bytes=max_cache_bytes,
        disk_safety_margin_bytes=_scaled_bytes_arg(
            args.disk_margin_gib,
            name="--disk-margin-gib",
            scale=1024**3,
        ),
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_decode_cache_init(result)
    return 0


def _context1_o_proj_cache(args: argparse.Namespace) -> int:
    max_cache_bytes = (
        _scaled_bytes_arg(args.max_cache_gib, name="--max-cache-gib", scale=1024**3)
        if args.max_cache_gib is not None
        else None
    )
    max_build_fma = (
        _scaled_count_arg(
            args.max_build_gfma,
            name="--max-build-gfma",
            scale=1.0e9,
        )
        if args.max_build_gfma is not None
        else args.max_build_fma
    )
    result = build_context1_o_proj_cache(
        args.prepared_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        execute=args.execute,
        force=args.force,
        backend=args.backend,
        max_cache_bytes=max_cache_bytes,
        max_build_fma=max_build_fma,
        disk_safety_margin_bytes=_scaled_bytes_arg(
            args.disk_margin_gib,
            name="--disk-margin-gib",
            scale=1024**3,
        ),
        max_metal_builder_live_bytes=_scaled_bytes_arg(
            args.max_metal_builder_live_mib,
            name="--max-metal-builder-live-mib",
            scale=1024**2,
            minimum=1.0,
        ),
        row_tile=args.row_tile,
        layers=tuple(args.layer) if args.layer else None,
        build_layers=(
            tuple(sorted(_parse_layers(args.build_layers) or ()))
            if args.build_layers
            else None
        ),
        build_next_layers=args.build_next_layers,
        metal_binary=args.metal_binary,
    )
    payload = result.to_json()
    if args.write_report is not None:
        _write_json_file_atomic(args.write_report, payload)
    if args.json:
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_context1_o_proj_cache(result)
        if args.write_report is not None:
            print(f"  report json:           {args.write_report}")
    return 0


def _validate_context1_o_proj_cache(args: argparse.Namespace) -> int:
    layout = load_context1_o_proj_cache_layout(
        args.layout,
        prepared_dir=args.prepared_dir,
        require_cache_file=not args.no_cache_file,
    )
    progress = load_context1_o_proj_cache_progress(
        layout,
        require_complete=not args.allow_incomplete_progress,
    )
    payload = layout.to_json()
    payload["progress"] = progress.to_json()
    if args.json:
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_context1_o_proj_cache_layout(layout)
        _print_context1_o_proj_cache_progress(progress)
    return 0


def _check_runtime(args: argparse.Namespace) -> int:
    budget = check_layer_runtime(
        args.expert_layout,
        args.resident_layout,
        layer=args.layer,
        dense_mlp=args.dense_mlp,
        top_k=args.top_k,
        max_k=args.max_k,
        max_slot_bytes=int(args.max_slot_mib * 1024**2),
        max_router_bytes=int(args.max_router_mib * 1024**2),
        max_resident_matrix_bytes=int(args.max_resident_matrix_mib * 1024**2),
        max_cache_read_bytes=int(args.max_cache_read_mib * 1024**2),
        max_runner_scratch_bytes=int(args.max_runner_scratch_mib * 1024**2),
        include_shared_expert=args.include_shared_expert,
        include_attention_projections=args.include_attention_projections,
        include_decoder_layer=args.include_decoder_layer,
        context_length=args.context_length,
        num_heads=args.num_heads,
        qk_nope_dim=args.qk_nope_dim,
        rope_dim=args.rope_dim,
        v_head_dim=args.v_head_dim,
        cache_dtype_bytes=args.cache_dtype_bytes,
    )
    if args.json:
        print(json.dumps(budget, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_runtime_budget(budget)
    return 0


def _required_decode_int(
    args: argparse.Namespace,
    name: str,
    config_value: int | None,
) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    if config_value is not None:
        return int(config_value)
    flag = "--" + name.replace("_", "-")
    raise ConfigError(f"{flag} is required unless --model-config provides it")


def _decode_router_score(args: argparse.Namespace, cfg: Any | None) -> str:
    if args.router_score is not None:
        return args.router_score
    score = getattr(cfg, "scoring_func", None)
    if score in {"sigmoid", "softmax", "raw"}:
        return str(score)
    return "sigmoid"


def _router_kwargs_from_args_config(
    args: argparse.Namespace,
    cfg: Any | None,
) -> dict[str, Any]:
    routed_scaling_factor = args.routed_scaling_factor
    if routed_scaling_factor is None:
        routed_scaling_factor = getattr(cfg, "routed_scaling_factor", None)

    norm_topk_prob = bool(args.norm_topk_prob)
    no_norm_topk_prob = bool(args.no_norm_topk_prob)
    if norm_topk_prob and no_norm_topk_prob:
        raise ConfigError("--norm-topk-prob and --no-norm-topk-prob conflict")
    if (
        not norm_topk_prob
        and not no_norm_topk_prob
        and getattr(cfg, "norm_topk_prob", None) is not None
    ):
        norm_topk_prob = bool(getattr(cfg, "norm_topk_prob"))

    router_n_group = args.router_n_group
    if router_n_group is None:
        router_n_group = getattr(cfg, "n_group", None)
    router_topk_group = args.router_topk_group
    if router_topk_group is None:
        router_topk_group = getattr(cfg, "topk_group", None)

    return {
        "router_score": _decode_router_score(args, cfg),
        "routed_scaling_factor": routed_scaling_factor,
        "norm_topk_prob": norm_topk_prob,
        "no_norm_topk_prob": no_norm_topk_prob,
        "router_n_group": router_n_group,
        "router_topk_group": router_topk_group,
    }


def _config_eos_token_ids(cfg: Any | None) -> tuple[int, ...]:
    return tuple(getattr(cfg, "eos_token_ids", ()) or ())


def _effective_rope_interleave(args: argparse.Namespace, cfg: Any | None) -> bool:
    if getattr(args, "rope_interleave", False):
        return True
    return bool(getattr(cfg, "rope_interleave", False))


def _prefill_dsa_kwargs(args: argparse.Namespace, cfg: Any | None) -> dict[str, Any]:
    dsa_indexer_types = None
    dsa_index_topk = getattr(args, "dsa_index_topk", None)
    dsa_index_n_heads = getattr(args, "dsa_index_n_heads", None)
    dsa_index_head_dim = getattr(args, "dsa_index_head_dim", None)
    dsa_qk_rope_dim = getattr(args, "dsa_qk_rope_dim", None)
    dsa_rope_interleave = getattr(args, "dsa_rope_interleave", False)
    if not getattr(args, "disable_dsa_indexer", False) and cfg is not None and cfg.indexer_types:
        dsa_indexer_types = cfg.indexer_types
        if dsa_index_topk is None:
            dsa_index_topk = cfg.index_topk
        if dsa_index_n_heads is None:
            dsa_index_n_heads = cfg.index_n_heads
        if dsa_index_head_dim is None and cfg.index_head_dim is not None:
            dsa_index_head_dim = int(cfg.index_head_dim)
        if dsa_qk_rope_dim is None:
            dsa_qk_rope_dim = cfg.qk_rope_head_dim
        dsa_rope_interleave = dsa_rope_interleave or bool(
            getattr(cfg, "indexer_rope_interleave", False)
        )
    return {
        "dsa_indexer_types": dsa_indexer_types,
        "dsa_index_topk": dsa_index_topk,
        "dsa_index_n_heads": dsa_index_n_heads,
        "dsa_index_head_dim": dsa_index_head_dim,
        "dsa_qk_rope_dim": dsa_qk_rope_dim,
        "dsa_rope_interleave": dsa_rope_interleave,
        "dsa_layer_norm_eps": getattr(args, "dsa_layer_norm_eps", 1e-6),
    }


def _validate_generation_layout_config(args: argparse.Namespace) -> None:
    if not getattr(args, "model_config", None):
        return
    expert_layout = getattr(args, "expert_layout", None)
    resident_layout = getattr(args, "resident_layout", None)
    if expert_layout is None or resident_layout is None:
        return
    validate_layout_model_config_sha256(
        model_config_path=args.model_config,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
    )


def _decode_layers(args: argparse.Namespace) -> int:
    cfg = load_config(args.model_config) if args.model_config else None
    num_heads = _required_decode_int(
        args,
        "num_heads",
        cfg.num_attention_heads if cfg is not None else None,
    )
    qk_nope_dim = _required_decode_int(
        args,
        "qk_nope_dim",
        cfg.qk_nope_head_dim if cfg is not None else None,
    )
    rope_dim = _required_decode_int(
        args,
        "rope_dim",
        cfg.qk_rope_head_dim if cfg is not None else None,
    )
    v_head_dim = _required_decode_int(
        args,
        "v_head_dim",
        cfg.v_head_dim if cfg is not None else None,
    )
    kv_lora_dim = args.kv_lora_dim
    if kv_lora_dim is None and cfg is not None and cfg.kv_lora_rank is not None:
        kv_lora_dim = int(cfg.kv_lora_rank)
    top_k = args.top_k
    if top_k is None:
        top_k = cfg.experts_per_token if cfg is not None else 8
    max_k = args.max_k if args.max_k is not None else max(8, int(top_k))
    rope_theta = args.rope_theta
    if rope_theta is None:
        rope_theta = cfg.rope_theta if cfg is not None and cfg.rope_theta else 10000.0
    rms_norm_eps = args.rms_norm_eps
    if rms_norm_eps is None:
        rms_norm_eps = (
            cfg.rms_norm_eps if cfg is not None and cfg.rms_norm_eps is not None else 1e-5
        )
    if args.include_shared_expert and args.no_shared_expert:
        raise ConfigError("--include-shared-expert and --no-shared-expert conflict")
    include_shared = args.include_shared_expert
    if not args.no_shared_expert and cfg is not None and (cfg.n_shared_experts or 0) > 0:
        include_shared = True
    router_kwargs = _router_kwargs_from_args_config(args, cfg)
    dsa_kwargs = _prefill_dsa_kwargs(args, cfg)

    result = run_decode_layers(
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        input_path=args.input_f32,
        output_path=args.output_f32,
        layers=_parse_layers(args.layers),
        dense_layers=_effective_dense_layers(args, cfg),
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        position=args.position,
        context_length=args.context_length,
        num_heads=num_heads,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        kv_lora_dim=kv_lora_dim,
        mla_kv_b_cache_dir=args.mla_kv_b_cache_dir,
        cache_position_offset=args.cache_position_offset,
        attention_scale=args.attention_scale,
        rope_theta=rope_theta,
        rope_interleave=_effective_rope_interleave(args, cfg),
        top_k=top_k,
        max_k=max_k,
        router_score=router_kwargs["router_score"],
        routed_scaling_factor=router_kwargs["routed_scaling_factor"],
        norm_topk_prob=router_kwargs["norm_topk_prob"],
        no_norm_topk_prob=router_kwargs["no_norm_topk_prob"],
        router_n_group=router_kwargs["router_n_group"],
        router_topk_group=router_kwargs["router_topk_group"],
        ignore_router_bias=args.ignore_router_bias,
        include_shared_expert=include_shared,
        rms_norm_eps=rms_norm_eps,
        max_slot_mib=args.max_slot_mib,
        max_router_mib=args.max_router_mib,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_write_mib=args.max_cache_write_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        expert_read_advise_merge_gap_kib=args.expert_read_advise_merge_gap_kib,
        expert_read_advise_align_kib=args.expert_read_advise_align_kib,
        cache_dtype_bytes=args.cache_dtype_bytes,
        dsa_indexer_types=dsa_kwargs["dsa_indexer_types"],
        dsa_index_topk=dsa_kwargs["dsa_index_topk"],
        dsa_index_n_heads=dsa_kwargs["dsa_index_n_heads"],
        dsa_index_head_dim=dsa_kwargs["dsa_index_head_dim"],
        dsa_qk_rope_dim=dsa_kwargs["dsa_qk_rope_dim"],
        dsa_rope_interleave=dsa_kwargs["dsa_rope_interleave"],
        dsa_layer_norm_eps=dsa_kwargs["dsa_layer_norm_eps"],
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_decode_layers_result(result)
    return 0


def _final_logits(args: argparse.Namespace) -> int:
    if args.runner:
        if args.output_logits_f32:
            raise FinalLogitsError("--runner Metal path supports top-k only, not full logits output")
        result = compute_final_logits_metal(
            runner_path=args.runner,
            resident_layout_path=args.resident_layout,
            input_f32_path=args.input_f32,
            output_topk_json_path=args.output_topk_json,
            top_k=args.top_k,
            rms_norm_eps=args.rms_norm_eps,
            chunk_rows=args.chunk_rows,
            max_chunk_bytes=int(args.max_chunk_mib * 1024**2),
            max_runner_scratch_bytes=int(args.max_runner_scratch_mib * 1024**2),
            allow_tied_embeddings=not args.no_tied_embeddings,
            skip_final_norm=args.skip_final_norm,
            echo_runner_output=not args.quiet_runner,
        )
    else:
        result = compute_final_logits(
            args.resident_layout,
            args.input_f32,
            output_logits_f32_path=args.output_logits_f32,
            output_topk_json_path=args.output_topk_json,
            top_k=args.top_k,
            rms_norm_eps=args.rms_norm_eps,
            chunk_rows=args.chunk_rows,
            max_chunk_bytes=int(args.max_chunk_mib * 1024**2),
            max_output_logits_bytes=int(args.max_output_logits_mib * 1024**2),
            allow_tied_embeddings=not args.no_tied_embeddings,
            skip_final_norm=args.skip_final_norm,
        )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_final_logits_result(result)
    return 0


def _embed_token(args: argparse.Namespace) -> int:
    result = embed_token(
        args.resident_layout,
        token_id=args.token_id,
        output_f32_path=args.output_f32,
        max_row_bytes=int(args.max_row_mib * 1024**2),
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_embedding_result(result)
    return 0


def _embed_tokens_batch(args: argparse.Namespace) -> int:
    result = embed_tokens_batch(
        args.resident_layout,
        token_ids=_read_batch_token_ids(args),
        output_f32_path=args.output_f32,
        max_row_bytes=int(args.max_row_mib * 1024**2),
        max_output_bytes=int(args.max_output_mib * 1024**2),
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_embedding_batch_result(result)
    return 0


def _prompt_prefill(args: argparse.Namespace) -> int:
    cfg = load_config(args.model_config) if args.model_config else None
    num_heads = _required_decode_int(
        args,
        "num_heads",
        cfg.num_attention_heads if cfg is not None else None,
    )
    qk_nope_dim = _required_decode_int(
        args,
        "qk_nope_dim",
        cfg.qk_nope_head_dim if cfg is not None else None,
    )
    rope_dim = _required_decode_int(
        args,
        "rope_dim",
        cfg.qk_rope_head_dim if cfg is not None else None,
    )
    v_head_dim = _required_decode_int(
        args,
        "v_head_dim",
        cfg.v_head_dim if cfg is not None else None,
    )
    kv_lora_dim = args.kv_lora_dim
    if kv_lora_dim is None and cfg is not None and cfg.kv_lora_rank is not None:
        kv_lora_dim = int(cfg.kv_lora_rank)
    top_k = args.top_k
    if top_k is None:
        top_k = cfg.experts_per_token if cfg is not None else 8
    max_k = args.max_k if args.max_k is not None else max(8, int(top_k))
    rope_theta = args.rope_theta
    if rope_theta is None:
        rope_theta = cfg.rope_theta if cfg is not None and cfg.rope_theta else 10000.0
    rms_norm_eps = args.rms_norm_eps
    if rms_norm_eps is None:
        rms_norm_eps = (
            cfg.rms_norm_eps if cfg is not None and cfg.rms_norm_eps is not None else 1e-5
        )
    if args.include_shared_expert and args.no_shared_expert:
        raise ConfigError("--include-shared-expert and --no-shared-expert conflict")
    include_shared = args.include_shared_expert
    if not args.no_shared_expert and cfg is not None and (cfg.n_shared_experts or 0) > 0:
        include_shared = True
    router_kwargs = _router_kwargs_from_args_config(args, cfg)
    dsa_kwargs = _prefill_dsa_kwargs(args, cfg)
    prompt_token_ids = _parse_token_ids(args.prompt_token_ids)
    prompt_chunk_tokens = int(args.prompt_chunk_tokens)
    auto_prompt_chunk_plan = None
    if prompt_chunk_tokens <= 0:
        auto_prompt_chunk_plan = _auto_prefill_prompt_chunk_plan(
            expert_layout_path=args.expert_layout,
            resident_layout_path=args.resident_layout,
            cache_layout_path=args.cache_layout,
            prompt_tokens=len(prompt_token_ids),
            start_position=args.start_position,
            layers=_parse_layers(args.layers),
            dense_layers=_effective_dense_layers(args, cfg),
            work_dir=None,
            prefill_work_dir=args.work_dir,
            top_k=top_k,
            max_prompt_batch_mib=args.max_prompt_batch_mib,
            max_cache_read_mib=args.max_cache_read_mib,
            max_cache_write_mib=args.max_cache_write_mib,
            max_runner_scratch_mib=args.max_runner_scratch_mib,
            prefill_max_stage_mib=args.max_stage_mib,
            prefill_max_compact_stage_mib=args.max_compact_stage_mib,
            prefill_expert_stage_align_kib=args.expert_stage_align_kib,
            prefill_stage_disk_margin_mib=args.stage_disk_margin_mib,
            dsa_indexer_types=dsa_kwargs["dsa_indexer_types"],
            dsa_index_topk=dsa_kwargs["dsa_index_topk"],
            prefill_expert_stage_tiling=args.expert_stage_tiling,
            prefill_linear_backend=args.prefill_linear_backend,
        )
        prompt_chunk_tokens = auto_prompt_chunk_plan.chunk_tokens

    result = run_prompt_prefill(
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        prompt_token_ids=prompt_token_ids,
        output_last_hidden_f32_path=args.output_last_hidden_f32,
        output_final_chunk_f32_path=args.output_final_chunk_f32,
        layers=_parse_layers(args.layers),
        dense_layers=_effective_dense_layers(args, cfg),
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        start_position=args.start_position,
        prompt_chunk_tokens=prompt_chunk_tokens,
        max_prompt_batch_mib=args.max_prompt_batch_mib,
        num_heads=num_heads,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        mla_kv_b_cache_dir=args.prefill_mla_kv_b_cache_dir,
        mla_key_cache=args.prefill_mla_key_cache,
        kv_lora_dim=kv_lora_dim,
        cache_position_offset=args.cache_position_offset,
        attention_scale=args.attention_scale,
        rope_theta=rope_theta,
        rope_interleave=_effective_rope_interleave(args, cfg),
        top_k=top_k,
        max_k=max_k,
        router_score=router_kwargs["router_score"],
        routed_scaling_factor=router_kwargs["routed_scaling_factor"],
        norm_topk_prob=router_kwargs["norm_topk_prob"],
        no_norm_topk_prob=router_kwargs["no_norm_topk_prob"],
        router_n_group=router_kwargs["router_n_group"],
        router_topk_group=router_kwargs["router_topk_group"],
        ignore_router_bias=args.ignore_router_bias,
        include_shared_expert=include_shared,
        rms_norm_eps=rms_norm_eps,
        max_embedding_row_mib=args.max_embedding_row_mib,
        max_slot_mib=args.max_slot_mib,
        max_router_mib=args.max_router_mib,
        max_resident_matrix_mib=args.max_resident_matrix_mib,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_write_mib=args.max_cache_write_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        max_live_working_set_mib=(
            8192.0
            if args.max_live_working_set_mib is None
            else args.max_live_working_set_mib
        ),
        min_free_unified_memory_gib=(
            0.0
            if args.min_free_unified_memory_gib is None
            else args.min_free_unified_memory_gib
        ),
        moe_token_block=args.moe_token_block,
        static_capacity_per_expert=args.static_capacity_per_expert,
        allow_static_capacity_overflow=args.allow_static_capacity_overflow,
        expert_stage_merge_gap_kib=args.expert_stage_merge_gap_kib,
        expert_stage_align_kib=args.expert_stage_align_kib,
        max_stage_mib=args.max_stage_mib,
        max_compact_stage_mib=args.max_compact_stage_mib,
        copy_chunk_mib=args.copy_chunk_mib,
        stage_disk_safety_margin_bytes=int(args.stage_disk_margin_mib * 1024**2),
        prefill_ssd_read_gib_per_second=args.prefill_ssd_read_gib_per_second,
        prefill_max_routed_read_seconds=args.prefill_max_routed_read_seconds,
        expert_stage_max_raw_ranges=args.expert_stage_max_raw_ranges,
        expert_stage_max_coalesced_ranges=args.expert_stage_max_coalesced_ranges,
        expert_stage_tiling=args.expert_stage_tiling,
        persistent_moe_plan_server=args.persistent_moe_plan_server,
        persistent_resident_linear_server=args.persistent_resident_linear_server,
        persistent_attention_projection_server=(
            args.persistent_attention_projection_server
        ),
        persistent_attention_output_server=args.persistent_attention_output_server,
        persistent_shared_expert_server=args.persistent_shared_expert_server,
        persistent_rope_split_server=args.persistent_rope_split_server,
        persistent_mla_attention_server=args.persistent_mla_attention_server,
        persistent_rmsnorm_server=args.persistent_rmsnorm_server,
        prefill_linear_backend=args.prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=args.prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=args.prefill_mpsgraph_min_matrix_dim,
        dsa_indexer_types=dsa_kwargs["dsa_indexer_types"],
        dsa_index_topk=dsa_kwargs["dsa_index_topk"],
        dsa_index_n_heads=dsa_kwargs["dsa_index_n_heads"],
        dsa_qk_rope_dim=dsa_kwargs["dsa_qk_rope_dim"],
        dsa_rope_interleave=dsa_kwargs["dsa_rope_interleave"],
        dsa_layer_norm_eps=dsa_kwargs["dsa_layer_norm_eps"],
        expected_vocab_size=None if cfg is None else cfg.vocab_size,
        expected_hidden_size=None if cfg is None else cfg.hidden_size,
        echo_runner_output=not args.quiet_runner,
    )
    if args.json:
        payload = _json_default(result)
        if auto_prompt_chunk_plan is not None:
            payload["auto_prompt_chunk_plan"] = auto_prompt_chunk_plan
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prompt_prefill_result(result)
    return 0


def _generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.model_config) if args.model_config else None
    num_heads = _required_decode_int(
        args,
        "num_heads",
        cfg.num_attention_heads if cfg is not None else None,
    )
    qk_nope_dim = _required_decode_int(
        args,
        "qk_nope_dim",
        cfg.qk_nope_head_dim if cfg is not None else None,
    )
    rope_dim = _required_decode_int(
        args,
        "rope_dim",
        cfg.qk_rope_head_dim if cfg is not None else None,
    )
    v_head_dim = _required_decode_int(
        args,
        "v_head_dim",
        cfg.v_head_dim if cfg is not None else None,
    )
    kv_lora_dim = args.kv_lora_dim
    if kv_lora_dim is None and cfg is not None and cfg.kv_lora_rank is not None:
        kv_lora_dim = int(cfg.kv_lora_rank)
    top_k = args.top_k
    if top_k is None:
        top_k = cfg.experts_per_token if cfg is not None else 8
    max_k = args.max_k if args.max_k is not None else max(8, int(top_k))
    rope_theta = args.rope_theta
    if rope_theta is None:
        rope_theta = cfg.rope_theta if cfg is not None and cfg.rope_theta else 10000.0
    rms_norm_eps = args.rms_norm_eps
    if rms_norm_eps is None:
        rms_norm_eps = (
            cfg.rms_norm_eps if cfg is not None and cfg.rms_norm_eps is not None else 1e-5
        )
    if args.include_shared_expert and args.no_shared_expert:
        raise ConfigError("--include-shared-expert and --no-shared-expert conflict")
    include_shared = args.include_shared_expert
    if not args.no_shared_expert and cfg is not None and (cfg.n_shared_experts or 0) > 0:
        include_shared = True
    router_kwargs = _router_kwargs_from_args_config(args, cfg)
    dsa_kwargs = _prefill_dsa_kwargs(args, cfg)

    return {
        "layers": _parse_layers(args.layers),
        "dense_layers": _effective_dense_layers(args, cfg),
        "work_dir": args.work_dir,
        "keep_work_dir": args.keep_work_dir,
        "num_heads": num_heads,
        "qk_nope_dim": qk_nope_dim,
        "rope_dim": rope_dim,
        "v_head_dim": v_head_dim,
        "prefill_mla_kv_b_cache_dir": getattr(
            args,
            "prefill_mla_kv_b_cache_dir",
            None,
        ),
        "prefill_mla_key_cache": bool(
            getattr(args, "prefill_mla_key_cache", False)
        ),
        "decode_mla_key_cache": bool(getattr(args, "decode_mla_key_cache", False)),
        "kv_lora_dim": kv_lora_dim,
        "cache_position_offset": args.cache_position_offset,
        "attention_scale": args.attention_scale,
        "rope_theta": rope_theta,
        "rope_interleave": _effective_rope_interleave(args, cfg),
        "top_k": top_k,
        "max_k": max_k,
        "router_score": router_kwargs["router_score"],
        "routed_scaling_factor": router_kwargs["routed_scaling_factor"],
        "norm_topk_prob": router_kwargs["norm_topk_prob"],
        "no_norm_topk_prob": router_kwargs["no_norm_topk_prob"],
        "router_n_group": router_kwargs["router_n_group"],
        "router_topk_group": router_kwargs["router_topk_group"],
        "ignore_router_bias": args.ignore_router_bias,
        "include_shared_expert": include_shared,
        "rms_norm_eps": rms_norm_eps,
        "logits_top_k": args.logits_top_k,
        "logits_chunk_rows": args.logits_chunk_rows,
        "logits_max_chunk_mib": args.logits_max_chunk_mib,
        "max_slot_mib": args.max_slot_mib,
        "max_router_mib": args.max_router_mib,
        "max_resident_matrix_mib": args.max_resident_matrix_mib,
        "max_cache_file_mib": args.max_cache_file_mib,
        "max_cache_read_mib": args.max_cache_read_mib,
        "decode_max_routed_read_gib_per_token": getattr(
            args,
            "decode_max_routed_read_gib_per_token",
            0.0,
        ),
        "decode_max_routed_read_seconds_per_token": getattr(
            args,
            "decode_max_routed_read_seconds_per_token",
            0.0,
        ),
        "max_runner_scratch_mib": args.max_runner_scratch_mib,
        "max_live_working_set_mib": (
            8192.0
            if args.max_live_working_set_mib is None
            else args.max_live_working_set_mib
        ),
        "min_free_unified_memory_gib": (
            0.0
            if args.min_free_unified_memory_gib is None
            else args.min_free_unified_memory_gib
        ),
        "expert_read_advise_merge_gap_kib": args.expert_read_advise_merge_gap_kib,
        "expert_read_advise_align_kib": args.expert_read_advise_align_kib,
        "cache_dtype_bytes": args.cache_dtype_bytes,
        "batch_prefill_prompt": getattr(args, "batch_prefill_prompt", False),
        "prefill_prompt_chunk_tokens": getattr(args, "prefill_prompt_chunk_tokens", 0),
        "prefill_max_prompt_batch_mib": getattr(
            args,
            "prefill_max_prompt_batch_mib",
            1024.0,
        ),
        "prefill_max_cache_write_mib": getattr(
            args,
            "prefill_max_cache_write_mib",
            4096.0,
        ),
        "prefill_expert_stage_merge_gap_kib": getattr(
            args,
            "prefill_expert_stage_merge_gap_kib",
            0.0,
        ),
        "prefill_expert_stage_align_kib": getattr(
            args,
            "prefill_expert_stage_align_kib",
            4.0,
        ),
        "prefill_max_stage_mib": getattr(args, "prefill_max_stage_mib", 4096.0),
        "prefill_max_compact_stage_mib": getattr(
            args,
            "prefill_max_compact_stage_mib",
            4096.0,
        ),
        "prefill_max_stage_raw_ranges": getattr(
            args,
            "prefill_max_stage_raw_ranges",
            0,
        ),
        "prefill_max_stage_coalesced_ranges": getattr(
            args,
            "prefill_max_stage_coalesced_ranges",
            0,
        ),
        "prefill_expert_stage_tiling": getattr(
            args,
            "prefill_expert_stage_tiling",
            False,
        ),
        "prefill_persistent_moe_plan_server": getattr(
            args,
            "prefill_persistent_moe_plan_server",
            False,
        ),
        "prefill_persistent_resident_linear_server": getattr(
            args,
            "prefill_persistent_resident_linear_server",
            False,
        ),
        "prefill_persistent_attention_projection_server": getattr(
            args,
            "prefill_persistent_attention_projection_server",
            False,
        ),
        "prefill_persistent_attention_output_server": getattr(
            args,
            "prefill_persistent_attention_output_server",
            False,
        ),
        "prefill_persistent_shared_expert_server": getattr(
            args,
            "prefill_persistent_shared_expert_server",
            False,
        ),
        "prefill_persistent_rope_split_server": getattr(
            args,
            "prefill_persistent_rope_split_server",
            False,
        ),
        "prefill_persistent_mla_attention_server": getattr(
            args,
            "prefill_persistent_mla_attention_server",
            False,
        ),
        "prefill_persistent_rmsnorm_server": getattr(
            args,
            "prefill_persistent_rmsnorm_server",
            False,
        ),
        "prefill_copy_chunk_mib": getattr(args, "prefill_copy_chunk_mib", 8.0),
        "prefill_stage_disk_margin_mib": getattr(
            args,
            "prefill_stage_disk_margin_mib",
            0.0,
        ),
        "prefill_max_routed_read_amplification": getattr(
            args,
            "prefill_max_routed_read_amplification",
            0.0,
        ),
        "prefill_max_routed_read_gib": getattr(
            args,
            "prefill_max_routed_read_gib",
            0.0,
        ),
        "prefill_ssd_read_gib_per_second": getattr(
            args,
            "prefill_ssd_read_gib_per_second",
            0.0,
        ),
        "prefill_max_routed_read_seconds": getattr(
            args,
            "prefill_max_routed_read_seconds",
            0.0,
        ),
        "prefill_moe_token_block": getattr(args, "prefill_moe_token_block", "auto"),
        "prefill_moe_output_accumulator": getattr(
            args,
            "prefill_moe_output_accumulator",
            "env",
        ),
        "prefill_static_capacity_per_expert": getattr(
            args,
            "prefill_static_capacity_per_expert",
            None,
        ),
        "prefill_allow_static_capacity_overflow": getattr(
            args,
            "prefill_allow_static_capacity_overflow",
            False,
        ),
        "prefill_linear_backend": getattr(args, "prefill_linear_backend", "auto"),
        "require_prefill_acceleration": bool(
            getattr(args, "require_prefill_acceleration", False)
        ),
        "allow_router_gate_only_prefill_acceleration": bool(
            getattr(args, "allow_router_gate_only_prefill_acceleration", False)
        ),
        "prefill_min_accelerated_flop_fraction": (
            _prefill_min_accelerated_flop_fraction(args)
        ),
        "prefill_mpsgraph_min_batch_tokens": getattr(
            args,
            "prefill_mpsgraph_min_batch_tokens",
            AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        ),
        "prefill_mpsgraph_min_matrix_dim": getattr(
            args,
            "prefill_mpsgraph_min_matrix_dim",
            AUTO_MPSGRAPH_MIN_DIM,
        ),
        "prefill_router_hybrid_margin_threshold": getattr(
            args,
            "prefill_router_hybrid_margin_threshold",
            0.0,
        ),
        "dsa_indexer_types": dsa_kwargs["dsa_indexer_types"],
        "dsa_index_topk": dsa_kwargs["dsa_index_topk"],
        "dsa_index_n_heads": dsa_kwargs["dsa_index_n_heads"],
        "dsa_index_head_dim": dsa_kwargs["dsa_index_head_dim"],
        "dsa_qk_rope_dim": dsa_kwargs["dsa_qk_rope_dim"],
        "dsa_rope_interleave": dsa_kwargs["dsa_rope_interleave"],
        "dsa_layer_norm_eps": dsa_kwargs["dsa_layer_norm_eps"],
        "max_embedding_row_mib": args.max_embedding_row_mib,
        "echo_runner_output": not args.quiet_runner,
        "sampling_temperature": args.temperature,
        "sampling_top_p": args.top_p,
        "sampling_seed": args.seed,
        "metal_final_logits": args.metal_final_logits,
        "allow_tied_embeddings": (
            True if cfg is None else cfg.tie_word_embeddings is not False
        ),
        "expected_vocab_size": None if cfg is None else cfg.vocab_size,
        "expected_hidden_size": None if cfg is None else cfg.hidden_size,
        "preflight_runtime": args.preflight_runtime,
        "eos_token_ids": (
            ()
            if getattr(args, "eos_token_id", None) is not None
            else _config_eos_token_ids(cfg)
        ),
        "allow_missing_dsa_indexer": getattr(args, "allow_missing_dsa_indexer", False),
    }


def _generate_token_ids(args: argparse.Namespace) -> int:
    _validate_generation_layout_config(args)
    kwargs = _generation_kwargs(args)
    result = generate_token_ids(
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        prompt_token_ids=_parse_token_ids(args.prompt_token_ids),
        max_new_tokens=args.max_new_tokens,
        eos_token_id=args.eos_token_id,
        **kwargs,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_token_generation_result(result)
    return 0


def _generate_metal_token_ids(args: argparse.Namespace) -> int:
    result = generate_metal_token_ids(
        args.prepared_dir,
        prompt_token_ids=_parse_token_ids(args.prompt_token_ids),
        max_new_tokens=args.max_new_tokens,
        binary=args.binary,
        expert_pin_plan=args.expert_pin_plan,
        max_adaptive_expert_cache_gib=args.max_adaptive_expert_cache_gib,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        top_k=args.top_k,
        logits_top_k=args.logits_top_k,
        max_live_working_set_mib=args.max_live_working_set_mib,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        logits_max_chunk_mib=args.logits_max_chunk_mib,
        mmap_final_logits=args.mmap_final_logits,
        max_embedding_row_mib=args.max_embedding_row_mib,
        cache_mla_kv_b_f32=args.cache_mla_kv_b_f32,
        max_mla_kv_b_cache_mib=args.max_mla_kv_b_cache_mib,
        context1_o_proj_cache_layout=args.context1_o_proj_cache_layout,
        context1_o_proj_cache_file=args.context1_o_proj_cache_file,
        allow_decode_only_multi_token_prompt=(
            args.allow_decode_only_multi_token_prompt
        ),
        prefill_prompt=args.prefill_prompt,
        prefill_runner=args.prefill_runner,
        prefill_prompt_chunk_tokens=args.prefill_prompt_chunk_tokens,
        prefill_max_prompt_batch_mib=args.prefill_max_prompt_batch_mib,
        prefill_max_cache_write_mib=args.prefill_max_cache_write_mib,
        prefill_max_runner_scratch_mib=args.prefill_max_runner_scratch_mib,
        prefill_max_live_working_set_mib=args.prefill_max_live_working_set_mib,
        min_free_unified_memory_gib=args.min_free_unified_memory_gib,
        use_generate_server_jsonl=not args.no_generate_server_jsonl,
        use_python_prefill_bridge=args.python_prefill_bridge,
        quiet=args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_metal_token_generation_result(result)
    return 0


def _generate_metal_text(args: argparse.Namespace) -> int:
    prompt_sources = [
        args.prompt is not None,
        args.prompt_file is not None,
        args.chat_messages is not None,
        args.chat_messages_file is not None,
    ]
    if sum(1 for item in prompt_sources if item) > 1:
        raise TextGenerationError(
            "--prompt, --prompt-file, --chat-messages, and --chat-messages-file "
            "are mutually exclusive"
        )
    tokenizer_path = args.tokenizer
    tokenizer_backend = args.tokenizer_backend
    add_special_tokens = not args.no_add_special_tokens
    if args.chat_messages is not None or args.chat_messages_file is not None:
        if tokenizer_path is None:
            manifest_path = Path(args.prepared_dir) / "manifest.json"
            tokenizer_path = str(
                load_prepared_manifest(manifest_path).model_dir
            )
        messages = _read_generation_chat_messages_arg(
            messages_json=args.chat_messages,
            messages_file=args.chat_messages_file,
            max_bytes=args.max_prompt_bytes,
        )
        rendered = render_chat_prompt(
            tokenizer_path,
            messages,
            backend=args.tokenizer_backend,
            trust_remote_code=args.trust_remote_code,
            add_generation_prompt=not args.no_add_generation_prompt,
        )
        prompt = rendered.text
        tokenizer_path = str(rendered.tokenizer_path)
        tokenizer_backend = rendered.tokenizer_backend or rendered.backend
        add_special_tokens = False
    else:
        prompt = _read_generation_prompt_arg(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            max_bytes=args.max_prompt_bytes,
        )
    result = generate_metal_text(
        prepared_dir=args.prepared_dir,
        tokenizer_path=tokenizer_path,
        tokenizer_backend=tokenizer_backend,
        trust_remote_code=args.trust_remote_code,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        add_special_tokens=add_special_tokens,
        skip_special_tokens=not args.no_skip_special_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        binary=args.binary,
        expert_pin_plan=args.expert_pin_plan,
        max_adaptive_expert_cache_gib=args.max_adaptive_expert_cache_gib,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
        top_k=args.top_k,
        logits_top_k=args.logits_top_k,
        max_live_working_set_mib=args.max_live_working_set_mib,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        logits_max_chunk_mib=args.logits_max_chunk_mib,
        mmap_final_logits=args.mmap_final_logits,
        max_embedding_row_mib=args.max_embedding_row_mib,
        cache_mla_kv_b_f32=args.cache_mla_kv_b_f32,
        max_mla_kv_b_cache_mib=args.max_mla_kv_b_cache_mib,
        context1_o_proj_cache_layout=args.context1_o_proj_cache_layout,
        context1_o_proj_cache_file=args.context1_o_proj_cache_file,
        allow_decode_only_multi_token_prompt=(
            args.allow_decode_only_multi_token_prompt
        ),
        prefill_prompt=args.prefill_prompt,
        prefill_runner=args.prefill_runner,
        prefill_prompt_chunk_tokens=args.prefill_prompt_chunk_tokens,
        prefill_max_prompt_batch_mib=args.prefill_max_prompt_batch_mib,
        prefill_max_cache_write_mib=args.prefill_max_cache_write_mib,
        prefill_max_runner_scratch_mib=args.prefill_max_runner_scratch_mib,
        prefill_max_live_working_set_mib=args.prefill_max_live_working_set_mib,
        min_free_unified_memory_gib=args.min_free_unified_memory_gib,
        use_generate_server_jsonl=not args.no_generate_server_jsonl,
        use_python_prefill_bridge=args.python_prefill_bridge,
        quiet=args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_metal_text_generation_result(result)
    return 0


def _read_metal_text_batch_prompts_jsonl(
    path_arg: str | Path,
    *,
    max_bytes: int,
) -> tuple[str, ...]:
    raw_text = _read_check_text_file_bounded(
        path_arg,
        max_bytes=max_bytes,
        label="Metal text batch prompts",
        option_name="--max-prompts-jsonl-bytes",
    )
    prompts: list[str] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TextGenerationError(
                f"failed to parse prompts JSONL line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TextGenerationError(
                f"prompts JSONL line {line_number} must be an object"
            )
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise TextGenerationError(
                f"prompts JSONL line {line_number}.prompt must be a non-empty string"
            )
        prompts.append(prompt)
    if not prompts:
        raise TextGenerationError("prompts JSONL must contain at least one prompt")
    return tuple(prompts)


def _generate_metal_text_batch(args: argparse.Namespace) -> int:
    prompts = _read_metal_text_batch_prompts_jsonl(
        args.prompts_jsonl,
        max_bytes=args.max_prompts_jsonl_bytes,
    )
    result = generate_metal_text_batch(
        prepared_dir=args.prepared_dir,
        prompts=prompts,
        tokenizer_path=args.tokenizer,
        tokenizer_backend=args.tokenizer_backend,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        binary=args.binary,
        expert_pin_plan=args.expert_pin_plan,
        max_adaptive_expert_cache_gib=args.max_adaptive_expert_cache_gib,
        add_special_tokens=not args.no_add_special_tokens,
        skip_special_tokens=not args.no_skip_special_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        top_k=args.top_k,
        logits_top_k=args.logits_top_k,
        max_live_working_set_mib=args.max_live_working_set_mib,
        max_cache_file_mib=args.max_cache_file_mib,
        max_cache_read_mib=args.max_cache_read_mib,
        logits_max_chunk_mib=args.logits_max_chunk_mib,
        mmap_final_logits=args.mmap_final_logits,
        max_embedding_row_mib=args.max_embedding_row_mib,
        context1_o_proj_cache_layout=args.context1_o_proj_cache_layout,
        context1_o_proj_cache_file=args.context1_o_proj_cache_file,
        allow_decode_only_multi_token_prompt=(
            args.allow_decode_only_multi_token_prompt
        ),
        prefill_prompt=args.prefill_prompt,
        prefill_runner=args.prefill_runner,
        prefill_prompt_chunk_tokens=args.prefill_prompt_chunk_tokens,
        prefill_max_prompt_batch_mib=args.prefill_max_prompt_batch_mib,
        prefill_max_cache_write_mib=args.prefill_max_cache_write_mib,
        prefill_max_runner_scratch_mib=args.prefill_max_runner_scratch_mib,
        prefill_max_live_working_set_mib=args.prefill_max_live_working_set_mib,
        min_free_unified_memory_gib=args.min_free_unified_memory_gib,
        use_generate_server_jsonl=not args.no_generate_server_jsonl,
        use_python_prefill_bridge=args.python_prefill_bridge,
        keep_work_dir=args.keep_work_dir,
        quiet=args.quiet_runner,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        print("LargerLM Metal text batch generation")
        print(f"  prompts:               {len(result.prompts)}")
        print(f"  tokenizer:             {result.tokenizer_path}")
        print(f"  tokenizer backend:     {result.tokenizer_backend}")
        print(f"  runtime requests:      {result.runtime_request_count}")
        print(f"  prompt tokens:         {result.total_prompt_tokens}")
        print(f"  generated tokens:      {result.total_generated_tokens}")
        if result.runtime_startup_elapsed_seconds is not None:
            print(
                "  runtime startup:       "
                f"{result.runtime_startup_elapsed_seconds:.3f} s"
            )
        print(f"  elapsed:               {result.elapsed_seconds:.3f} s")
        if result.generated_tokens_per_second is not None:
            print(
                "  generated throughput:  "
                f"{result.generated_tokens_per_second:.3f} tok/s"
            )
        print(
            "  max live envelope:     "
            f"{format_bytes(result.max_estimated_live_working_set_bytes)}"
        )
        if result.max_prompt_prefill_estimated_live_working_set_bytes is not None:
            print(
                "  max prefill envelope:  "
                f"{format_bytes(result.max_prompt_prefill_estimated_live_working_set_bytes)}"
            )
        if result.min_system_available_memory_bytes is not None:
            print(
                "  min available memory:  "
                f"{format_bytes(result.min_system_available_memory_bytes)}"
            )
        if result.max_required_available_memory_bytes is not None:
            print(
                "  max required memory:   "
                f"{format_bytes(result.max_required_available_memory_bytes)}"
            )
        if result.max_expert_buffer_count_allocated is not None:
            print(
                "  max expert buffers:    "
                f"{result.max_expert_buffer_count_allocated}"
            )
        if result.all_admission_ok is not None:
            print(f"  all admission ok:      {result.all_admission_ok}")
        if result.all_available_unified_memory_ok is not None:
            print(
                "  all memory ok:         "
                f"{result.all_available_unified_memory_ok}"
            )
        for index, item in enumerate(result.results):
            print(f"  [{index}] generated ids: {','.join(str(t) for t in item.generated_token_ids)}")
            print(f"  [{index}] generated text: {item.generated_text!r}")
    return 0


def _generate_text(args: argparse.Namespace) -> int:
    prompt = _read_generation_prompt_arg(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        max_bytes=args.max_prompt_bytes,
    )
    _validate_generation_layout_config(args)
    kwargs = _generation_kwargs(args)
    result = generate_text(
        tokenizer_path=args.tokenizer,
        tokenizer_backend=args.tokenizer_backend,
        trust_remote_code=args.trust_remote_code,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        add_special_tokens=not args.no_add_special_tokens,
        skip_special_tokens=not args.no_skip_special_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        eos_token_id=args.eos_token_id,
        use_tokenizer_eos=not args.ignore_tokenizer_eos,
        runner_path=args.runner,
        expert_layout_path=args.expert_layout,
        resident_layout_path=args.resident_layout,
        cache_layout_path=args.cache_layout,
        cache_file_path=args.cache_file,
        **kwargs,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_text_generation_result(result)
    return 0


def _default_prepared_static_capacity(args: argparse.Namespace) -> None:
    if getattr(args, "prefill_static_capacity_per_expert", None) is None:
        args.prefill_static_capacity_per_expert = "auto"


def _apply_prepared_memory_guard_defaults(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> None:
    if args.max_live_working_set_mib is None:
        args.max_live_working_set_mib = (
            prepared.recommended_max_live_working_set_bytes / 1024**2
            if prepared.recommended_max_live_working_set_bytes is not None
            else 8192.0
        )
    if args.min_free_unified_memory_gib is None:
        args.min_free_unified_memory_gib = (
            prepared.recommended_min_free_unified_memory_bytes / 1024**3
            if prepared.recommended_min_free_unified_memory_bytes is not None
            else 0.0
        )
    if (
        getattr(args, "prefill_ssd_read_gib_per_second", None) in (None, 0.0)
        and not bool(getattr(args, "no_prepared_ssd_read_default", False))
        and prepared.prepare_cold_read_gib_per_second is not None
    ):
        args.prefill_ssd_read_gib_per_second = (
            prepared.prepare_cold_read_gib_per_second
        )


def _require_prepared_glm_4bit_if_requested(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> None:
    if not bool(getattr(args, "require_glm_4bit", False)):
        return
    model_config = getattr(args, "model_config", None) or prepared.model_dir
    readiness = prepared_glm_4bit_readiness(prepared, model_config)
    if readiness.get("ok") is True:
        return
    issues = readiness.get("issues")
    if isinstance(issues, list) and issues:
        detail = "; ".join(str(issue) for issue in issues[:5])
    else:
        detail = "readiness check did not pass"
    raise CliArgumentError(f"prepared GLM 4bit readiness failed: {detail}")


def _public_glm_5_2_mismatch_summary(readiness: dict[str, object]) -> str | None:
    shape = readiness.get("public_glm_5_2_shape")
    mismatched = shape.get("mismatched_fields") if isinstance(shape, dict) else None
    if not isinstance(mismatched, (list, tuple)) or not mismatched:
        return None
    preview = ", ".join(str(item) for item in tuple(mismatched)[:6])
    if len(mismatched) > 6:
        preview += f", +{len(mismatched) - 6} more"
    return preview


def _require_prepared_public_glm_5_2_shape_if_requested(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> None:
    if not bool(getattr(args, "require_public_glm_5_2_shape", False)):
        return
    if bool(getattr(args, "allow_missing_dsa_indexer", False)):
        raise CliArgumentError(
            "--require-public-glm-5-2-shape cannot be used with "
            "--allow-missing-dsa-indexer"
        )
    model_config = getattr(args, "model_config", None) or prepared.model_dir
    readiness = prepared_glm_4bit_readiness(prepared, model_config)
    issues = readiness.get("issues")
    if readiness.get("ok") is not True:
        if isinstance(issues, list) and issues:
            detail = "; ".join(str(issue) for issue in issues[:5])
        else:
            detail = "readiness check did not pass"
        raise CliArgumentError(
            "prepared public GLM-5.2 readiness failed: "
            f"{detail}"
        )
    if readiness.get("matches_public_glm_5_2_shape") is not True:
        detail = _public_glm_5_2_mismatch_summary(readiness)
        suffix = f" ({detail})" if detail else ""
        raise CliArgumentError(
            "prepared public GLM-5.2 readiness failed: "
            "prepared config does not match the public GLM-5.2 shape"
            f"{suffix}"
        )


def _require_prepared_runtime_profile(
    prepared: PreparedManifest,
    *,
    require_memory_profile: bool = False,
) -> None:
    reason = prepared_runtime_profile_failure_reason(
        prepared,
        require_memory_profile=require_memory_profile,
    )
    if reason is None:
        return
    raise CliArgumentError(f"prepared runtime profile check failed: {reason}")


def _prepared_memory_profile_required_by_args(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "require_prepared_memory_profile", False)
        or getattr(args, "require_public_glm_5_2_shape", False)
    )


def _prefill_acceleration_gate(
    *,
    configured_backend: str,
    mps_graph_runtime_available: bool | None,
    mpp_runtime_available: bool | None,
    mps_graph_probe_requested: bool | None = None,
    mps_graph_probe_ran: bool | None = None,
    mps_graph_probe_ok: bool | None = None,
    mpp_run_probe_requested: bool | None = None,
    mpp_run_probe_ran: bool | None = None,
    mpp_run_probe_ok: bool | None = None,
    selectable_backends: tuple[str, ...] = (),
    acceleration_runtimes: tuple[str, ...] = (),
) -> dict[str, object]:
    return evaluate_prefill_acceleration_requirement(
        configured_backend=configured_backend,
        mps_graph_runtime_available=mps_graph_runtime_available,
        mpp_runtime_available=mpp_runtime_available,
        mps_graph_probe_requested=mps_graph_probe_requested,
        mps_graph_probe_ran=mps_graph_probe_ran,
        mps_graph_probe_ok=mps_graph_probe_ok,
        mpp_run_probe_requested=mpp_run_probe_requested,
        mpp_run_probe_ran=mpp_run_probe_ran,
        mpp_run_probe_ok=mpp_run_probe_ok,
        selectable_backends=selectable_backends,
        acceleration_runtimes=acceleration_runtimes,
    ).to_json()


def _prefill_min_accelerated_flop_fraction(args: argparse.Namespace) -> float:
    value = float(getattr(args, "prefill_min_accelerated_flop_fraction", 0.0))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise CliArgumentError("--prefill-min-accelerated-flop-fraction must be 0..1")
    return value


def _prefill_acceleration_required(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "require_prefill_acceleration", False)) or (
        _prefill_min_accelerated_flop_fraction(args) > 0.0
    )


def _require_prefill_acceleration_if_requested(args: argparse.Namespace) -> None:
    if not _prefill_acceleration_required(args):
        return
    configured = getattr(args, "prefill_linear_backend", "auto")
    if configured == "custom-metal":
        gate = _prefill_acceleration_gate(
            configured_backend=configured,
            mps_graph_runtime_available=None,
            mpp_runtime_available=None,
            selectable_backends=(),
            acceleration_runtimes=(),
        )
        raise CliArgumentError(
            "prefill acceleration requirement failed: "
            f"{gate['reason']}"
        )
    try:
        capability = inspect_prefill_backend(
            run_host_probe=True,
            compile_mpp_probe=bool(getattr(args, "compile_mpp_probe", False)),
            run_mpp_probe=bool(getattr(args, "run_mpp_probe", False)),
            run_mpsgraph_probe=bool(getattr(args, "run_mpsgraph_probe", False)),
            probe_timeout_seconds=float(
                getattr(
                    args,
                    "prefill_backend_probe_timeout_seconds",
                    DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
                )
            ),
        )
    except PrefillBackendError:
        raise
    except Exception as exc:
        raise CliArgumentError(f"prefill backend inspection failed: {exc}") from exc
    gate = _prefill_acceleration_gate(
        configured_backend=configured,
        mps_graph_runtime_available=capability.mps_graph_runtime_available,
        mpp_runtime_available=capability.mpp_runtime_available,
        mps_graph_probe_requested=getattr(
            capability,
            "mps_graph_probe_requested",
            None,
        ),
        mps_graph_probe_ran=getattr(capability, "mps_graph_probe_ran", None),
        mps_graph_probe_ok=getattr(capability, "mps_graph_probe_ok", None),
        mpp_run_probe_requested=getattr(
            capability,
            "mpp_run_probe_requested",
            None,
        ),
        mpp_run_probe_ran=getattr(capability, "mpp_run_probe_ran", None),
        mpp_run_probe_ok=getattr(capability, "mpp_run_probe_ok", None),
        selectable_backends=selectable_accelerated_prefill_backends(capability),
        acceleration_runtimes=prefill_acceleration_runtimes(capability),
    )
    if gate["ok"] is not True:
        raise CliArgumentError(
            "prefill acceleration requirement failed: "
            f"{gate['reason']}"
        )


def _require_prepared_prefill_acceleration_coverage_if_requested(
    args: argparse.Namespace,
    *,
    prompt_token_count: int,
    generation_overrides: dict[str, Any] | None = None,
) -> None:
    if not _prefill_acceleration_required(args):
        return
    if isinstance(prompt_token_count, bool) or prompt_token_count <= 0:
        raise CliArgumentError("prompt token count must be positive")
    if not _current_request_has_batch_prefill_work(
        args,
        prompt_token_count=prompt_token_count,
    ):
        return
    config = _prepared_server_config_from_args(args)
    max_new_tokens = int(getattr(args, "max_new_tokens", 0))
    max_new_tokens_cap = max(config.max_new_tokens_cap, max_new_tokens)
    max_prompt_tokens = max(config.max_prompt_tokens, int(prompt_token_count))
    if (
        max_new_tokens_cap != config.max_new_tokens_cap
        or max_prompt_tokens != config.max_prompt_tokens
    ):
        config = replace(
            config,
            max_new_tokens_cap=max_new_tokens_cap,
            max_prompt_tokens=max_prompt_tokens,
        )
    app = PreparedGenerationApp(config)
    payload = {
        "max_new_tokens": max_new_tokens,
        "logits_top_k": int(getattr(args, "logits_top_k", 1)),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "top_p": float(getattr(args, "top_p", 1.0)),
        "metal_final_logits": bool(getattr(args, "metal_final_logits", False)),
    }
    try:
        check = app.inspect_token_request(
            prompt_token_count=int(prompt_token_count),
            payload=payload,
            generation_overrides=_prepared_admission_generation_overrides(
                args,
                generation_overrides,
            ),
        )
        require_prepared_request_check_ok(check)
    except PreparedServerError as exc:
        raise CliArgumentError(
            "prefill acceleration request coverage failed: "
            f"{exc}"
        ) from exc
    coverage = check.get("prefill_acceleration_coverage")
    if not isinstance(coverage, dict) or coverage.get("ok") is not True:
        reason = (
            coverage.get("reason")
            if isinstance(coverage, dict) and coverage.get("reason")
            else "request did not resolve to an accelerated prefill backend"
        )
        raise CliArgumentError(
            "prefill acceleration request coverage failed: "
            f"{reason}"
        )


def _prepared_request_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": int(getattr(args, "max_new_tokens", 0)),
        "logits_top_k": int(getattr(args, "logits_top_k", 1)),
        "temperature": float(getattr(args, "temperature", 0.0)),
        "top_p": float(getattr(args, "top_p", 1.0)),
        "metal_final_logits": bool(getattr(args, "metal_final_logits", False)),
    }


def _prepared_admission_config_for_prompt(
    args: argparse.Namespace,
    *,
    prompt_token_count: int,
) -> PreparedServerConfig:
    config = _prepared_server_config_from_args(args)
    if not _current_request_has_batch_prefill_work(
        args,
        prompt_token_count=prompt_token_count,
    ):
        config = replace(
            config,
            require_prefill_acceleration=False,
            prefill_min_accelerated_flop_fraction=0.0,
        )
    max_new_tokens = int(getattr(args, "max_new_tokens", 0))
    max_new_tokens_cap = max(config.max_new_tokens_cap, max_new_tokens)
    max_prompt_tokens = max(config.max_prompt_tokens, int(prompt_token_count))
    if (
        max_new_tokens_cap != config.max_new_tokens_cap
        or max_prompt_tokens != config.max_prompt_tokens
    ):
        config = replace(
            config,
            max_new_tokens_cap=max_new_tokens_cap,
            max_prompt_tokens=max_prompt_tokens,
        )
    return config


_PREFILL_CHUNK_PLAN_ADMISSION_FAILURES = frozenset(
    {
        "missing_profile_max_safe_plan",
        "missing_actual_max_safe_plan",
        "missing_actual_prompt_chunk",
        "selected_chunk_exceeds_current_max_safe",
        "current_max_safe_below_profile",
    }
)


def _require_prefill_chunk_plan_profile_admission(
    *,
    applied_launch_profile: dict[str, object] | None,
    request_check: dict[str, Any],
) -> None:
    chunk_tokens = request_check.get("prefill_prompt_chunk_tokens")
    actual_prompt_chunk_tokens = (
        chunk_tokens.get("resolved") if isinstance(chunk_tokens, dict) else None
    )
    chunk_plan = request_check.get("prefill_prompt_chunk_plan")
    actual_auto_plan = None
    actual_max_safe_plan = None
    if isinstance(chunk_plan, dict):
        auto_plan = chunk_plan.get("auto")
        if isinstance(auto_plan, dict):
            actual_auto_plan = auto_plan
        max_safe_plan = chunk_plan.get("max_safe")
        if isinstance(max_safe_plan, dict):
            actual_max_safe_plan = max_safe_plan
    drift = prefill_prompt_chunk_plan_drift_summary(
        applied_launch_profile=applied_launch_profile,
        actual_prompt_chunk_tokens=(
            actual_prompt_chunk_tokens
            if type(actual_prompt_chunk_tokens) is int
            else None
        ),
        actual_auto_plan=actual_auto_plan,  # type: ignore[arg-type]
        actual_max_safe_plan=actual_max_safe_plan,  # type: ignore[arg-type]
    )
    if not isinstance(drift, dict):
        return
    status = drift.get("status")
    if status not in _PREFILL_CHUNK_PLAN_ADMISSION_FAILURES:
        return
    raise CliArgumentError(
        "prepared request admission failed: prefill chunk-plan drift "
        f"{status}; profile_max_safe={drift.get('profile_max_safe_chunk_tokens')} "
        f"current_max_safe={drift.get('actual_max_safe_chunk_tokens')} "
        f"selected={drift.get('actual_prompt_chunk_tokens')}"
    )


def _prepared_admission_generation_overrides(
    args: argparse.Namespace,
    generation_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if generation_overrides is None:
        return None
    if str(getattr(args, "prefill_linear_backend", "auto") or "auto") != "auto":
        return generation_overrides
    overrides = dict(generation_overrides)
    overrides.pop("prefill_linear_backend", None)
    return overrides


def _apply_prepared_runtime_prefill_backend(
    args: argparse.Namespace,
    generation_kwargs: dict[str, Any],
    runtime_prefill_linear_backend: str | None,
) -> None:
    if not isinstance(runtime_prefill_linear_backend, str):
        return
    if str(getattr(args, "prefill_linear_backend", "auto") or "auto") != "auto":
        return
    generation_kwargs["prefill_linear_backend"] = runtime_prefill_linear_backend


def _require_prepared_request_admission(
    args: argparse.Namespace,
    *,
    prompt_token_count: int,
    generation_overrides: dict[str, Any] | None = None,
    runtime_preflight: bool = False,
) -> str:
    if isinstance(prompt_token_count, bool) or prompt_token_count <= 0:
        raise CliArgumentError("prompt token count must be positive")
    config = _prepared_admission_config_for_prompt(
        args,
        prompt_token_count=int(prompt_token_count),
    )
    app = PreparedGenerationApp(config)
    try:
        check = app.inspect_token_request(
            prompt_token_count=int(prompt_token_count),
            payload=_prepared_request_payload_from_args(args),
            runtime_preflight=runtime_preflight,
            generation_overrides=_prepared_admission_generation_overrides(
                args,
                generation_overrides,
            ),
        )
        require_prepared_request_check_ok(check)
    except PreparedRequestCheckError as exc:
        raise CliArgumentError(f"prepared request admission failed: {exc}") from exc
    except PreparedServerError as exc:
        raise CliArgumentError(f"prepared request admission failed: {exc}") from exc
    _require_prefill_chunk_plan_profile_admission(
        applied_launch_profile=config.applied_launch_profile,
        request_check=check,
    )
    return app.state.runtime_prefill_linear_backend


def _require_actual_prepared_prefill_acceleration_if_requested(
    args: argparse.Namespace,
    result: TokenGenerationResult,
) -> None:
    if not _prefill_acceleration_required(args):
        return
    if result.prompt_prefill is None:
        return
    reason = prompt_prefill_acceleration_failure_reason(
        result.prompt_prefill,
        min_accelerated_flop_fraction=_prefill_min_accelerated_flop_fraction(args),
        allow_router_gate_only_acceleration=bool(
            getattr(args, "allow_router_gate_only_prefill_acceleration", False)
        ),
    )
    if reason is not None:
        raise CliArgumentError(
            "prefill acceleration actual coverage failed: "
            f"{reason}"
        )


def _current_request_has_batch_prefill_work(
    args: argparse.Namespace,
    *,
    prompt_token_count: int,
) -> bool:
    if isinstance(prompt_token_count, bool) or prompt_token_count <= 1:
        return False
    if bool(getattr(args, "no_batch_prefill_prompt", False)):
        return False
    return bool(getattr(args, "batch_prefill_prompt", False))


def _prepared_server_config_from_args(args: argparse.Namespace) -> PreparedServerConfig:
    prepared = load_prepared_manifest(args.prepared)
    _require_launch_profile_matches_prepared(args, prepared)
    _apply_prepared_memory_guard_defaults(args, prepared)
    applied_launch_profile = _applied_launch_profile_summary(args, prepared)
    return PreparedServerConfig(
        prepared_path=Path(args.prepared),
        runner_path=Path(getattr(args, "runner", "metal/largerlm-runner")),
        model_config_path=(
            Path(args.model_config)
            if getattr(args, "model_config", None)
            else None
        ),
        tokenizer_path=Path(args.tokenizer) if getattr(args, "tokenizer", None) else None,
        tokenizer_backend=getattr(args, "tokenizer_backend", "auto"),
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        require_prepared_memory_profile=_prepared_memory_profile_required_by_args(args),
        require_glm_4bit=bool(getattr(args, "require_glm_4bit", False)),
        require_public_glm_5_2_shape=bool(
            getattr(args, "require_public_glm_5_2_shape", False)
        ),
        served_model_name=getattr(args, "served_model_name", "largerlm-prepared"),
        host=getattr(args, "host", "127.0.0.1"),
        port=int(getattr(args, "port", 8000)),
        max_new_tokens_cap=int(getattr(args, "max_new_tokens_cap", 256)),
        max_prompt_tokens=int(getattr(args, "max_prompt_tokens", 4096)),
        max_request_bytes=int(getattr(args, "max_request_bytes", 1024 * 1024)),
        batch_prefill_prompt=not bool(getattr(args, "no_batch_prefill_prompt", False)),
        max_cache_read_mib=float(getattr(args, "max_cache_read_mib", 256.0)),
        max_cache_file_mib=float(getattr(args, "max_cache_file_mib", 32768.0)),
        decode_max_routed_read_gib_per_token=float(
            getattr(args, "decode_max_routed_read_gib_per_token", 0.0)
        ),
        decode_max_routed_read_seconds_per_token=float(
            getattr(args, "decode_max_routed_read_seconds_per_token", 0.0)
        ),
        max_runner_scratch_mib=float(getattr(args, "max_runner_scratch_mib", 4096.0)),
        expert_read_advise_merge_gap_kib=int(
            getattr(args, "expert_read_advise_merge_gap_kib", 0)
        ),
        expert_read_advise_align_kib=int(
            getattr(args, "expert_read_advise_align_kib", 0)
        ),
        prefill_prompt_chunk_tokens=int(
            getattr(args, "prefill_prompt_chunk_tokens", 0)
        ),
        prefill_max_prompt_batch_mib=float(
            getattr(args, "prefill_max_prompt_batch_mib", 1024.0)
        ),
        prefill_max_cache_write_mib=float(
            getattr(args, "prefill_max_cache_write_mib", 4096.0)
        ),
        prefill_max_stage_mib=float(getattr(args, "prefill_max_stage_mib", 4096.0)),
        prefill_max_compact_stage_mib=float(
            getattr(args, "prefill_max_compact_stage_mib", 4096.0)
        ),
        prefill_max_stage_raw_ranges=int(
            getattr(args, "prefill_max_stage_raw_ranges", 0)
        ),
        prefill_max_stage_coalesced_ranges=int(
            getattr(args, "prefill_max_stage_coalesced_ranges", 0)
        ),
        prefill_expert_stage_tiling=bool(
            getattr(args, "prefill_expert_stage_tiling", False)
        ),
        prefill_persistent_moe_plan_server=bool(
            getattr(args, "prefill_persistent_moe_plan_server", False)
        ),
        prefill_persistent_resident_linear_server=bool(
            getattr(args, "prefill_persistent_resident_linear_server", False)
        ),
        prefill_persistent_attention_projection_server=bool(
            getattr(args, "prefill_persistent_attention_projection_server", False)
        ),
        prefill_persistent_attention_output_server=bool(
            getattr(args, "prefill_persistent_attention_output_server", False)
        ),
        prefill_persistent_shared_expert_server=bool(
            getattr(args, "prefill_persistent_shared_expert_server", False)
        ),
        prefill_persistent_rope_split_server=bool(
            getattr(args, "prefill_persistent_rope_split_server", False)
        ),
        prefill_persistent_mla_attention_server=bool(
            getattr(args, "prefill_persistent_mla_attention_server", False)
        ),
        prefill_persistent_rmsnorm_server=bool(
            getattr(args, "prefill_persistent_rmsnorm_server", False)
        ),
        prefill_copy_chunk_mib=float(getattr(args, "prefill_copy_chunk_mib", 8.0)),
        prefill_stage_disk_margin_mib=float(
            getattr(args, "prefill_stage_disk_margin_mib", 0.0)
        ),
        prefill_max_routed_read_amplification=float(
            getattr(args, "prefill_max_routed_read_amplification", 0.0)
        ),
        prefill_max_routed_read_gib=float(
            getattr(args, "prefill_max_routed_read_gib", 0.0)
        ),
        prefill_ssd_read_gib_per_second=float(
            getattr(args, "prefill_ssd_read_gib_per_second", 0.0)
        ),
        prefill_max_routed_read_seconds=float(
            getattr(args, "prefill_max_routed_read_seconds", 0.0)
        ),
        prefill_moe_token_block=getattr(args, "prefill_moe_token_block", "auto"),
        prefill_moe_output_accumulator=getattr(
            args,
            "prefill_moe_output_accumulator",
            "env",
        ),
        prefill_static_capacity_per_expert=getattr(
            args,
            "prefill_static_capacity_per_expert",
            "auto",
        ),
        prefill_allow_static_capacity_overflow=bool(
            getattr(args, "prefill_allow_static_capacity_overflow", False)
        ),
        prefill_mla_kv_b_cache_dir=(
            Path(args.prefill_mla_kv_b_cache_dir)
            if getattr(args, "prefill_mla_kv_b_cache_dir", None)
            else None
        ),
        prefill_mla_key_cache=bool(
            getattr(args, "prefill_mla_key_cache", False)
        ),
        decode_mla_key_cache=bool(getattr(args, "decode_mla_key_cache", False)),
        metal_runtime_cache_mla_kv_b_f32=bool(
            getattr(args, "metal_runtime_cache_mla_kv_b_f32", False)
        ),
        metal_runtime_max_mla_kv_b_cache_mib=float(
            getattr(args, "metal_runtime_max_mla_kv_b_cache_mib", 0.0)
        ),
        metal_runtime_expert_pin_plan=(
            Path(args.metal_runtime_expert_pin_plan)
            if getattr(args, "metal_runtime_expert_pin_plan", None)
            else None
        ),
        metal_runtime_max_adaptive_expert_cache_gib=float(
            getattr(args, "metal_runtime_max_adaptive_expert_cache_gib", 0.0)
        ),
        metal_runtime_mmap_final_logits=bool(
            getattr(args, "metal_runtime_mmap_final_logits", False)
        ),
        metal_runtime_context1_o_proj_cache_layout=(
            Path(args.metal_runtime_context1_o_proj_cache_layout)
            if getattr(args, "metal_runtime_context1_o_proj_cache_layout", None)
            else None
        ),
        metal_runtime_context1_o_proj_cache_file=(
            Path(args.metal_runtime_context1_o_proj_cache_file)
            if getattr(args, "metal_runtime_context1_o_proj_cache_file", None)
            else None
        ),
        prefill_linear_backend=getattr(args, "prefill_linear_backend", "auto"),
        require_prefill_acceleration=bool(
            getattr(args, "require_prefill_acceleration", False)
        ),
        allow_router_gate_only_prefill_acceleration=bool(
            getattr(args, "allow_router_gate_only_prefill_acceleration", False)
        ),
        prefill_min_accelerated_flop_fraction=(
            _prefill_min_accelerated_flop_fraction(args)
        ),
        prefill_mpsgraph_min_batch_tokens=int(
            getattr(
                args,
                "prefill_mpsgraph_min_batch_tokens",
                AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
            )
        ),
        prefill_mpsgraph_min_matrix_dim=int(
            getattr(args, "prefill_mpsgraph_min_matrix_dim", AUTO_MPSGRAPH_MIN_DIM)
        ),
        prefill_router_hybrid_margin_threshold=float(
            getattr(args, "prefill_router_hybrid_margin_threshold", 0.0)
        ),
        prefill_compile_mpp_probe=bool(getattr(args, "compile_mpp_probe", False)),
        prefill_run_mpp_probe=bool(getattr(args, "run_mpp_probe", False)),
        prefill_run_mpsgraph_probe=bool(
            getattr(args, "run_mpsgraph_probe", False)
        ),
        prefill_backend_probe_timeout_seconds=float(
            getattr(
                args,
                "prefill_backend_probe_timeout_seconds",
                DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
            )
        ),
        metal_final_logits=bool(getattr(args, "metal_final_logits", False)),
        metal_runtime_generation=bool(
            getattr(args, "metal_runtime_generation", False)
        ),
        metal_binary_path=Path(
            getattr(args, "metal_binary", "metal/glm_moe_infer")
        ),
        max_live_working_set_mib=args.max_live_working_set_mib,
        min_free_unified_memory_gib=args.min_free_unified_memory_gib,
        logits_top_k_cap=int(getattr(args, "logits_top_k_cap", 64)),
        echo_runner_output=not bool(getattr(args, "quiet_runner", True)),
        allow_missing_dsa_indexer=bool(getattr(args, "allow_missing_dsa_indexer", False)),
        applied_launch_profile=applied_launch_profile,
    )


def _format_optional_bytes(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unknown"
    return format_bytes(value)


def _print_prepared_health(health: dict[str, Any]) -> None:
    print("LargerLM prepared inspection")
    print(f"  manifest:              {health['prepared_manifest']}")
    print(f"  model dir:             {health['model_dir']}")
    print(f"  served model:          {health['served_model_name']}")
    print(f"  effective context:     {health['effective_context_tokens']}")
    print(f"  effective prompt cap:  {health['effective_max_prompt_tokens']}")
    print(f"  batch prefill:         {health['batch_prefill_prompt']}")
    print(
        "  persistent moe srv:   "
        f" {health.get('prefill_persistent_moe_plan_server')}"
    )
    print(
        "  persistent linear srv:"
        f" {health.get('prefill_persistent_resident_linear_server')}"
    )
    print(
        "  persistent proj srv:  "
        f" {health.get('prefill_persistent_attention_projection_server')}"
    )
    print(
        "  persistent out srv:   "
        f" {health.get('prefill_persistent_attention_output_server')}"
    )
    print(
        "  persistent shared srv:"
        f" {health.get('prefill_persistent_shared_expert_server')}"
    )
    print(
        "  persistent rope srv:  "
        f" {health.get('prefill_persistent_rope_split_server')}"
    )
    print(
        "  persistent mla srv:   "
        f" {health.get('prefill_persistent_mla_attention_server')}"
    )
    print(
        "  persistent rms srv:   "
        f" {health.get('prefill_persistent_rmsnorm_server')}"
    )
    print(
        "  moe accumulator:      "
        f" {health.get('prefill_moe_output_accumulator')}"
    )
    print(f"  MLA kv_b cache dir:    {health.get('prefill_mla_kv_b_cache_dir')}")
    print(
        "  routed amp cap:       "
        f" {health.get('prefill_max_routed_read_amplification')}"
    )
    print(
        "  routed read GiB cap:  "
        f" {health.get('prefill_max_routed_read_gib')}"
    )
    print(
        "  decode read GiB cap:  "
        f" {health.get('decode_max_routed_read_gib_per_token')}"
    )
    print(
        "  decode read sec cap:  "
        f" {health.get('decode_max_routed_read_seconds_per_token')}"
    )
    print(
        "  routed read sec cap:  "
        f" {health.get('prefill_max_routed_read_seconds')}"
    )
    print(f"  live working cap MiB:  {health['max_live_working_set_mib']}")
    print(f"  min free memory GiB:   {health['min_free_unified_memory_gib']}")
    _print_applied_launch_profile(health.get("applied_launch_profile"))
    system_memory = health.get("system_memory")
    if isinstance(system_memory, dict):
        available = system_memory.get("available_bytes")
        total = system_memory.get("total_bytes")
        print(
            "  system memory:         "
            f"{format_bytes(available) if isinstance(available, int) else 'unknown'}"
            " available"
        )
        if isinstance(total, int):
            print(f"  system memory total:   {format_bytes(total)}")
    memory_guard = health.get("memory_guard")
    if isinstance(memory_guard, dict):
        print(
            "  guard required mem:    "
            f"{_format_optional_bytes(memory_guard.get('configured_required_available_memory_bytes'))}"
        )
        if memory_guard.get("available_ok") is not None:
            print(f"  guard memory ok:       {memory_guard.get('available_ok')}")
    launch_flags = health.get("suggested_launch_guard_flags")
    if isinstance(launch_flags, dict):
        argv = launch_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested launch args: {' '.join(str(item) for item in argv)}")
    glm_flags = health.get("suggested_glm_4bit_guard_flags")
    if isinstance(glm_flags, dict):
        argv = glm_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested GLM args:    {' '.join(str(item) for item in argv)}")
    decode_flags = health.get("suggested_decode_guard_flags")
    if isinstance(decode_flags, dict):
        argv = decode_flags.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested decode args: {' '.join(str(item) for item in argv)}")
    launch_profile = health.get("suggested_launch_profile")
    if isinstance(launch_profile, dict):
        argv = launch_profile.get("argv")
        if isinstance(argv, (list, tuple)) and argv:
            print(f"  suggested profile args: {' '.join(str(item) for item in argv)}")
        prepared_identity = launch_profile.get("prepared")
        if isinstance(prepared_identity, dict):
            strength = prepared_identity.get("identity_strength")
            if strength:
                print(f"  profile identity:      {strength}")
            for warning in prepared_identity.get("identity_warnings", ()):
                print(f"  warning:               {warning}")
    runtime_profile = health.get("prepared_runtime_profile")
    if isinstance(runtime_profile, dict):
        if runtime_profile.get("profile_ok") is not None:
            print(f"  prepared profile ok:   {runtime_profile.get('profile_ok')}")
        for warning in runtime_profile.get("warnings", ()):
            print(f"  warning:               {warning}")
    glm_readiness = health.get("glm_4bit_readiness")
    if isinstance(glm_readiness, dict):
        if glm_readiness.get("analyzed"):
            print(f"  GLM 4bit ready:        {glm_readiness.get('ok')}")
            print(
                "  GLM 4bit quant:        "
                f"{glm_readiness.get('expert_layout_quantization')} "
                f"group={glm_readiness.get('expert_layout_group_size')}"
            )
            if glm_readiness.get("matches_public_glm_5_2_shape") is not None:
                print(
                    "  GLM-5.2 public shape:  "
                    f"{glm_readiness.get('matches_public_glm_5_2_shape')}"
                )
                mismatch = _public_glm_5_2_mismatch_summary(glm_readiness)
                if mismatch:
                    print(f"  GLM-5.2 mismatch:      {mismatch}")
            for issue in glm_readiness.get("issues", ()):
                print(f"  warning:               {issue}")
    storage = health.get("prepared_storage")
    if isinstance(storage, dict):
        print(
            "  prepared layout bytes: "
            f"{_format_optional_bytes(storage.get('prepared_total_layout_bytes'))}"
        )
        print(
            "  prepared file bytes:   "
            f"{_format_optional_bytes(storage.get('prepared_total_file_bytes'))}"
        )
        print(
            "  resident+cache bytes:  "
            f"{_format_optional_bytes(storage.get('resident_and_cache_layout_bytes'))}"
        )
        if storage.get("expert_quantization"):
            print(
                "  expert quantization:   "
                f"{storage.get('expert_quantization')} "
                f"group={storage.get('expert_group_size')}"
            )
        if storage.get("prepare_effective_unified_memory_bytes") is not None:
            print(
                "  prepare memory:        "
                f"{_format_optional_bytes(storage.get('prepare_effective_unified_memory_bytes'))} "
                f"({storage.get('prepare_effective_unified_memory_source')})"
            )
        if storage.get("prepare_resolved_max_context_tokens") is not None:
            print(
                "  prepare context:       "
                f"{storage.get('prepare_resolved_max_context_tokens')} "
                f"auto={storage.get('prepare_auto_context_from_budget')}"
            )
        if storage.get("prepare_decode_cache_budget_bytes") is not None:
            print(
                "  prepare cache budget:  "
                f"{_format_optional_bytes(storage.get('prepare_decode_cache_budget_bytes'))} "
                f"safe_ctx={storage.get('prepare_decode_cache_safe_context_tokens')}"
            )
        if storage.get("prepare_flags_applied") is True:
            digest = storage.get("prepare_flags_sha256")
            suffix = f" sha256={str(digest)[:12]}" if digest else ""
            print(
                "  prepare flags:         "
                f"{storage.get('prepare_flags_source')}{suffix}"
            )
        if storage.get("prepare_cold_read_gib_per_second") is not None:
            suffix = ""
            if storage.get("prepare_cold_read_source"):
                suffix = f" ({storage.get('prepare_cold_read_source')})"
            print(
                "  prepare SSD read:      "
                f"{storage.get('prepare_cold_read_gib_per_second')} GiB/s"
                f"{suffix}"
            )
        print(
            "  recommended required:  "
            f"{_format_optional_bytes(storage.get('recommended_required_available_memory_bytes'))}"
        )
    backend = health.get("prefill_backend")
    if isinstance(backend, dict):
        print(f"  prefill backend:       {backend.get('configured_backend')}")
        if backend.get("effective_backend") != backend.get("configured_backend"):
            print(f"  prefill runtime:       {backend.get('effective_backend')}")
        capability = backend.get("capability")
        if isinstance(capability, dict):
            print(
                "  backend recommended:   "
                f"{capability.get('recommended_backend')}"
            )
            runtimes = capability.get("prefill_acceleration_runtimes")
            if isinstance(runtimes, list):
                runtimes = tuple(runtimes)
            if isinstance(runtimes, tuple):
                print(
                    "  backend accel runtime: "
                    f"{', '.join(str(item) for item in runtimes) or 'none'}"
                )
            selectable = capability.get("selectable_accelerated_prefill_backends")
            if isinstance(selectable, list):
                selectable = tuple(selectable)
            if isinstance(selectable, tuple):
                print(
                    "  backend accel select:  "
                    f"{', '.join(str(item) for item in selectable) or 'none'}"
                )
            suggested = capability.get("suggested_prefill_acceleration_flags")
            if isinstance(suggested, dict):
                argv = suggested.get("argv")
                if isinstance(argv, (list, tuple)) and argv:
                    print(
                        "  backend accel args:   "
                        f"{' '.join(str(item) for item in argv)}"
                    )
        auto_policy = backend.get("auto_policy")
        if isinstance(auto_policy, dict):
            print(
                "  backend auto policy:   "
                f"tokens>={auto_policy.get('mpsgraph_min_batch_tokens')} "
                f"dim>={auto_policy.get('mpsgraph_min_matrix_dim')}"
            )
        for warning in backend.get("warnings", ()):
            print(f"  warning:               {warning}")
    accel_gate = health.get("prefill_acceleration_requirement")
    if isinstance(accel_gate, dict):
        print(f"  prefill accel ok:      {accel_gate.get('ok')}")
        if accel_gate.get("reason_code"):
            print(f"  prefill accel code:    {accel_gate.get('reason_code')}")
        if accel_gate.get("reason"):
            print(f"  warning:               {accel_gate.get('reason')}")
    request_check = health.get("request_check")
    result_label = "ok"
    launch_audit = health.get("launch_audit")
    if isinstance(launch_audit, dict):
        print(f"  launch audit:         {launch_audit.get('ok')}")
        failures = launch_audit.get("failures")
        if isinstance(failures, (list, tuple)) and failures:
            print(f"  launch audit failures: {', '.join(str(item) for item in failures)}")
    if isinstance(request_check, dict):
        print(f"  request ok:            {request_check.get('ok')}")
        if request_check.get("ok"):
            if request_check.get("prompt_source"):
                print(f"  request prompt source: {request_check.get('prompt_source')}")
            print(f"  request prompt tokens: {request_check.get('prompt_token_count')}")
            print(f"  request max new:       {request_check.get('max_new_tokens')}")
            print(f"  request context:       {request_check.get('required_context_tokens')}")
            tokenizer = request_check.get("tokenizer")
            if isinstance(tokenizer, dict):
                print(f"  request tokenizer:     {tokenizer.get('backend')}")
            if request_check.get("chat_template_backend"):
                print(
                    "  request chat template: "
                    f"{request_check.get('chat_template_backend')}"
                )
            chunk = request_check.get("prefill_prompt_chunk_tokens")
            if isinstance(chunk, dict):
                print(f"  request prefill chunk: {chunk.get('resolved')}")
                print(f"  request max chunk:     {chunk.get('max_safe')}")
            linear = request_check.get("prefill_linear_backend")
            if isinstance(linear, dict):
                print(f"  request linear backend:{linear.get('configured')}")
                if linear.get("effective") != linear.get("configured"):
                    print(
                        "  request linear runtime:"
                        f"{linear.get('effective')}"
                    )
                if linear.get("analyzed"):
                    print(
                        "  request linear mix:    "
                        f"mpsgraph={linear.get('mpsgraph_matrix_count')} "
                        f"custom={linear.get('custom_metal_matrix_count')}"
                    )
                    print(
                        "  request linear scratch:"
                        f" {format_bytes(linear.get('max_matrix_scratch_bytes'))}"
                    )
                    print(
                        "  request scratch total:"
                        f" {format_bytes(linear.get('total_matrix_scratch_bytes'))}"
                    )
                    top_matrices = linear.get("top_matrices")
                    if isinstance(top_matrices, (list, tuple)) and top_matrices:
                        top = top_matrices[0]
                        if isinstance(top, dict):
                            print(
                                "  request top matrix:   "
                                f"#{top.get('rank')} {top.get('name')} "
                                f"backend={top.get('resolved_backend')} "
                                f"flops={float(top.get('estimated_flops', 0)) / 1e9:.3f}G"
                            )
            coverage = request_check.get("prefill_acceleration_coverage")
            if isinstance(coverage, dict):
                print(f"  request accel ok:    {coverage.get('ok')}")
                print(
                    "  request accel mats:  "
                    f"{coverage.get('accelerated_matrix_count')}/"
                    f"{coverage.get('matrix_count')}"
                )
                if coverage.get("reason"):
                    print(f"  warning:               {coverage.get('reason')}")
            accel_frontier = request_check.get("prefill_acceleration_frontier")
            if isinstance(accel_frontier, dict) and accel_frontier.get("analyzed"):
                print(
                    "  request accel min:   "
                    f"{accel_frontier.get('minimum_accelerated_prompt_chunk_tokens')}"
                )
                candidates = accel_frontier.get("candidates")
                if isinstance(candidates, (list, tuple)) and candidates:
                    first = candidates[0]
                    if isinstance(first, dict):
                        print(
                            "  request accel frontier:"
                            f" chunk={first.get('prompt_chunk_tokens')} "
                            f"mpsgraph={first.get('mpsgraph_matrix_count')} "
                            f"custom={first.get('custom_metal_matrix_count')}"
                        )
            routed_read = request_check.get("prefill_routed_expert_read")
            if isinstance(routed_read, dict) and routed_read.get("analyzed"):
                print(
                    "  request routed read: "
                    f"{format_bytes(routed_read.get('planned_read_bytes'))} "
                    f"x{float(routed_read.get('read_amplification', 1.0)):.2f} "
                    f"extra={format_bytes(routed_read.get('extra_read_bytes'))}"
                )
                planned_seconds = routed_read.get("planned_read_seconds")
                if isinstance(planned_seconds, (int, float)):
                    print(
                        "  request routed time: "
                        f"{float(planned_seconds):.3g}s"
                    )
                if routed_read.get("max_read_amplification"):
                    print(
                        "  request routed ok:   "
                        f" {routed_read.get('within_limit')}"
                    )
                elif routed_read.get("max_planned_read_bytes"):
                    print(
                        "  request routed ok:   "
                        f" {routed_read.get('within_limit')}"
                    )
                elif routed_read.get("max_read_seconds"):
                    print(
                        "  request routed ok:   "
                        f" {routed_read.get('within_limit')}"
                    )
                minimum_chunk = routed_read.get("minimum_chunk_tokens_for_limits")
                if isinstance(minimum_chunk, int):
                    print(f"  request routed min:  {minimum_chunk}")
                elif routed_read.get("within_limit") is False:
                    print("  request routed min:  none")
            stage_temp = request_check.get("prefill_routed_stage_temp_disk")
            if isinstance(stage_temp, dict) and stage_temp.get("analyzed"):
                max_temp = stage_temp.get(
                    "effective_max_stage_plus_compact_plus_static_bytes",
                    stage_temp.get(
                        "max_stage_plus_compact_plus_static_bytes",
                        stage_temp.get(
                            "effective_max_stage_plus_compact_bytes",
                            stage_temp.get("max_stage_plus_compact_bytes"),
                        ),
                    ),
                )
                total_temp = stage_temp.get(
                    "total_stage_plus_compact_plus_static_bytes",
                    stage_temp.get("total_stage_plus_compact_bytes"),
                )
                print(
                    "  request stage temp:  "
                    f"{format_bytes(max_temp)}"
                )
                print(
                    "  request stage total: "
                    f"{format_bytes(total_temp)}"
                )
                static_bytes = stage_temp.get("total_static_capacity_binary_bytes")
                if isinstance(static_bytes, int) and static_bytes > 0:
                    print(
                        "  request route bin:   "
                        f"{format_bytes(static_bytes)}"
                    )
            frontier = request_check.get("prefill_routed_chunk_frontier")
            if isinstance(frontier, dict) and frontier.get("analyzed"):
                candidates = frontier.get("candidates")
                candidate_count = (
                    len(candidates)
                    if isinstance(candidates, (list, tuple))
                    else 0
                )
                print(
                    "  routed chunk frontier:"
                    f" {candidate_count} candidates"
                )
                if frontier.get("saturation_chunk_tokens") is not None:
                    print(
                        "  routed saturates at: "
                        f"{frontier.get('saturation_chunk_tokens')} tokens"
                    )
                resolved = frontier.get("resolved_prompt_chunk_tokens")
                if isinstance(candidates, (list, tuple)):
                    for candidate in candidates:
                        if (
                            isinstance(candidate, dict)
                            and candidate.get("prompt_chunk_tokens") == resolved
                        ):
                            candidate_stage = candidate.get(
                                "max_stage_plus_compact_plus_static_bytes",
                                candidate.get("max_stage_plus_compact_bytes"),
                            )
                            if (
                                isinstance(stage_temp, dict)
                                and stage_temp.get("expert_stage_tiling") is True
                            ):
                                candidate_stage = stage_temp.get(
                                    "effective_max_stage_plus_compact_plus_static_bytes",
                                    stage_temp.get(
                                        "effective_max_stage_plus_compact_bytes",
                                        candidate_stage,
                                    ),
                                )
                            print(
                                "  routed resolved read:"
                                f" {format_bytes(candidate.get('planned_read_bytes'))} "
                                f"x{float(candidate.get('read_amplification', 1.0)):.2f} "
                                f"stage={format_bytes(candidate_stage)}"
                            )
                            break
            suggested_prefill_flags = request_check.get(
                "suggested_prefill_guard_flags"
            )
            if isinstance(suggested_prefill_flags, dict):
                argv = suggested_prefill_flags.get("argv")
                if isinstance(argv, (list, tuple)) and argv:
                    print(
                        "  suggested prefill args: "
                        f"{' '.join(str(item) for item in argv)}"
                    )
            suggested_flags = request_check.get("suggested_guard_flags")
            if isinstance(suggested_flags, dict):
                argv = suggested_flags.get("argv")
                if isinstance(argv, (list, tuple)) and argv:
                    print(f"  suggested guard args: {' '.join(str(item) for item in argv)}")
            suggested_stage_flags = request_check.get(
                "suggested_stage_temp_guard_flags"
            )
            if isinstance(suggested_stage_flags, dict):
                argv = suggested_stage_flags.get("argv")
                if isinstance(argv, (list, tuple)) and argv:
                    print(f"  suggested stage args: {' '.join(str(item) for item in argv)}")
            request_profile = health.get("request_launch_profile")
            if isinstance(request_profile, dict):
                argv = request_profile.get("argv")
                if isinstance(argv, (list, tuple)) and argv:
                    print(
                        "  request profile args: "
                        f"{' '.join(str(item) for item in argv)}"
                    )
            runtime = request_check.get("runtime_preflight")
            if isinstance(runtime, dict):
                print(f"  runtime preflight:     {runtime.get('ran')}")
                if runtime.get("ran"):
                    print(
                        "  runtime live set:      "
                        f"{format_bytes(runtime.get('live_working_set_bytes'))}"
                    )
                    print(
                        "  runtime required mem:  "
                        f"{format_bytes(runtime.get('required_available_memory_bytes'))}"
                    )
                    print(
                        "  runtime available mem: "
                        f"{format_bytes(runtime.get('system_available_memory_bytes'))}"
                    )
                    if runtime.get("available_memory_ok") is not None:
                        print(
                            "  runtime memory ok:    "
                            f"{runtime.get('available_memory_ok')}"
                        )
                    prefill_live = runtime.get("prefill_live_memory")
                    if isinstance(prefill_live, dict):
                        prefill_live_bytes = prefill_live.get(
                            "estimated_live_working_set_bytes"
                        )
                        print(
                            "  prefill live estimate:"
                            f" {format_bytes(prefill_live_bytes)}"
                        )
                    print(
                        "  runtime layer peak:    "
                        f"{format_bytes(runtime.get('max_layer_peak_bytes'))}"
                    )
                elif runtime.get("reason"):
                    print(f"  runtime reason:        {runtime.get('reason')}")
        else:
            print(f"  request error:         {request_check.get('error')}")
            chunk = request_check.get("prefill_prompt_chunk_tokens")
            if isinstance(chunk, dict):
                print(f"  request prefill chunk: {chunk.get('resolved')}")
                print(f"  request max chunk:     {chunk.get('max_safe')}")
            chunk_plan = request_check.get("prefill_prompt_chunk_plan")
            max_safe_plan = (
                chunk_plan.get("max_safe") if isinstance(chunk_plan, dict) else None
            )
            if isinstance(max_safe_plan, dict):
                reachable = max_safe_plan.get(
                    "mpp_tensor_ops_candidate_reachable_under_caps"
                )
                dimension_candidates = max_safe_plan.get(
                    "mpp_tensor_ops_dimension_candidate_matrix_count"
                )
                blocking_caps = max_safe_plan.get(
                    "mpp_tensor_ops_candidate_blocking_cap_names"
                )
                blocking_summary = max_safe_plan.get(
                    "mpp_tensor_ops_candidate_blocking_cap_summary"
                )
                print(
                    "  request MPP reachable:"
                    f" {reachable} candidates={dimension_candidates}"
                )
                if isinstance(blocking_summary, dict) and blocking_summary:
                    print(
                        "  request MPP block sum:"
                        f" {', '.join(f'{key}={value}' for key, value in blocking_summary.items())}"
                    )
                stage_tiling_reachable = max_safe_plan.get(
                    "mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable"
                )
                stage_tiling_blockers = max_safe_plan.get(
                    "mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers"
                )
                stage_tiling_plan = max_safe_plan.get(
                    "mpp_tensor_ops_candidate_stage_tiling_plan"
                )
                if stage_tiling_reachable is not None:
                    print(
                        "  request MPP stage tile:"
                        f" {stage_tiling_reachable}"
                    )
                if isinstance(stage_tiling_plan, dict):
                    print(
                        "  request stage tiles:"
                        f" max/layer={stage_tiling_plan.get('max_stage_tile_count_per_layer')}"
                        f" total={stage_tiling_plan.get('total_stage_tile_count')}"
                        f" experts/tile={stage_tiling_plan.get('max_experts_per_stage_tile')}"
                    )
                if (
                    isinstance(stage_tiling_blockers, (list, tuple))
                    and stage_tiling_blockers
                ):
                    print(
                        "  request stage tile blockers: "
                        f"{', '.join(str(item) for item in stage_tiling_blockers)}"
                    )
                if isinstance(blocking_caps, (list, tuple)) and blocking_caps:
                    shown_caps = tuple(str(item) for item in blocking_caps[:8])
                    suffix = (
                        f" (+{len(blocking_caps) - len(shown_caps)} more)"
                        if len(blocking_caps) > len(shown_caps)
                        else ""
                    )
                    print(
                        "  request MPP blockers: "
                        f"{', '.join(shown_caps)}{suffix}"
                    )
            result_label = "rejected"
    print(f"  result:                {result_label}")


def _request_prefill_backend_policy_flags(
    request_linear_backend: object,
) -> dict[str, object] | None:
    backend: object
    if isinstance(request_linear_backend, dict):
        if request_linear_backend.get("configured") == "auto":
            return None
        backend = request_linear_backend.get("effective")
        if not isinstance(backend, str):
            backend = request_linear_backend.get("configured")
    else:
        backend = request_linear_backend
    if (
        not isinstance(backend, str)
        or backend not in PREFILL_LINEAR_BACKENDS
        or backend == "auto"
    ):
        return None
    return {
        "source": "prepared_request_check",
        "prefill_linear_backend": backend,
        "argv": ("--prefill-linear-backend", backend),
    }


def _request_prefill_backend_is_auto(request_linear_backend: object) -> bool:
    if isinstance(request_linear_backend, dict):
        configured = request_linear_backend.get("configured")
        effective = request_linear_backend.get("effective")
        return configured == "auto" or effective == "auto"
    return request_linear_backend == "auto"


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


def _request_prefill_acceleration_profile_flags(
    acceleration_flags: object,
    *,
    request_backend_policy_flags: dict[str, object] | None,
    request_linear_backend: object,
) -> dict[str, object] | None:
    if not isinstance(acceleration_flags, dict):
        return None
    request_backend_name = (
        request_backend_policy_flags.get("prefill_linear_backend")
        if isinstance(request_backend_policy_flags, dict)
        else None
    )
    if request_backend_name not in {None, "auto", "mpp-f32", "mpsgraph-f32"}:
        return None
    if request_backend_name is not None:
        return acceleration_flags
    if not _request_prefill_backend_is_auto(request_linear_backend):
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


def _request_prefill_guard_profile_flags(
    prefill_flags: object,
    *,
    prompt_chunk_plan: object,
) -> dict[str, object] | None:
    if not isinstance(prefill_flags, dict):
        return None
    if (
        not isinstance(prompt_chunk_plan, dict)
        or prompt_chunk_plan.get("configured_is_auto") is not True
    ):
        return prefill_flags
    raw_argv = prefill_flags.get("argv")
    if not isinstance(raw_argv, (list, tuple)):
        return prefill_flags
    argv = [str(item) for item in raw_argv]
    try:
        index = argv.index("--prefill-prompt-chunk-tokens")
    except ValueError:
        return prefill_flags
    if index + 1 >= len(argv):
        return prefill_flags
    filtered = dict(prefill_flags)
    resolved = filtered.get("prefill_prompt_chunk_tokens")
    if type(resolved) is int and resolved > 0:
        filtered["resolved_prefill_prompt_chunk_tokens"] = resolved
    filtered["prefill_prompt_chunk_tokens"] = "auto"
    filtered["prefill_prompt_chunk_tokens_policy"] = "auto"
    argv[index + 1] = "auto"
    filtered["argv"] = tuple(argv)
    return filtered


def _request_launch_profile_from_health(
    health: dict[str, Any],
    request_check: dict[str, Any],
) -> dict[str, object] | None:
    launch_flags = health.get("suggested_launch_guard_flags")
    prepared_ssd_read_flags = health.get("suggested_prepared_ssd_read_flags")
    prefill_backend_probe_flags = health.get("suggested_prefill_backend_probe_flags")
    prefill_runtime_policy_flags = health.get(
        "suggested_prefill_runtime_policy_flags"
    )
    prefill_copy_policy_flags = health.get("suggested_prefill_copy_policy_flags")
    prefill_persistent_moe_plan_server_flags = health.get(
        "suggested_prefill_persistent_moe_plan_server_flags"
    )
    prefill_persistent_resident_linear_server_flags = health.get(
        "suggested_prefill_persistent_resident_linear_server_flags"
    )
    prefill_persistent_attention_projection_server_flags = health.get(
        "suggested_prefill_persistent_attention_projection_server_flags"
    )
    prefill_persistent_attention_output_server_flags = health.get(
        "suggested_prefill_persistent_attention_output_server_flags"
    )
    prefill_persistent_shared_expert_server_flags = health.get(
        "suggested_prefill_persistent_shared_expert_server_flags"
    )
    prefill_persistent_rope_split_server_flags = health.get(
        "suggested_prefill_persistent_rope_split_server_flags"
    )
    prefill_persistent_mla_attention_server_flags = health.get(
        "suggested_prefill_persistent_mla_attention_server_flags"
    )
    prefill_persistent_rmsnorm_server_flags = health.get(
        "suggested_prefill_persistent_rmsnorm_server_flags"
    )
    prefill_moe_output_accumulator_flags = health.get(
        "suggested_prefill_moe_output_accumulator_flags"
    )
    glm_4bit_flags = health.get("suggested_glm_4bit_guard_flags")
    public_glm_5_2_flags = health.get(
        "suggested_public_glm_5_2_shape_guard_flags"
    )
    backend = health.get("prefill_backend")
    capability = backend.get("capability") if isinstance(backend, dict) else None
    acceleration_flags = (
        capability.get("suggested_prefill_acceleration_flags")
        if isinstance(capability, dict)
        else None
    )
    request_linear_backend = request_check.get("prefill_linear_backend")
    request_backend_policy_flags = _request_prefill_backend_policy_flags(
        request_linear_backend,
    )
    acceleration_flags = _request_prefill_acceleration_profile_flags(
        acceleration_flags,
        request_backend_policy_flags=request_backend_policy_flags,
        request_linear_backend=request_linear_backend,
    )
    decode_flags = request_check.get("suggested_decode_guard_flags")
    if not isinstance(decode_flags, dict):
        decode_flags = health.get("suggested_decode_guard_flags")
    prefill_flags = request_check.get("suggested_prefill_guard_flags")
    prompt_chunk_plan = request_check.get("prefill_prompt_chunk_plan")
    prefill_flags = _request_prefill_guard_profile_flags(
        prefill_flags,
        prompt_chunk_plan=prompt_chunk_plan,
    )
    final_logits_flags = suggest_final_logits_flags(
        metal_final_logits=request_check.get("metal_final_logits") is True,
        source="prepared_request_check",
    )
    launch_profile = health.get("suggested_launch_profile")
    prepared_target = (
        launch_profile.get("prepared")
        if isinstance(launch_profile, dict)
        else None
    )
    profile = combine_suggested_launch_profile(
        launch_guard_flags=launch_flags if isinstance(launch_flags, dict) else None,
        prepared_ssd_read_flags=(
            prepared_ssd_read_flags
            if isinstance(prepared_ssd_read_flags, dict)
            else None
        ),
        prefill_backend_probe_flags=(
            prefill_backend_probe_flags
            if isinstance(prefill_backend_probe_flags, dict)
            else None
        ),
        prefill_backend_policy_flags=(
            request_backend_policy_flags
            if isinstance(request_backend_policy_flags, dict)
            else None
        ),
        prefill_runtime_policy_flags=(
            prefill_runtime_policy_flags
            if isinstance(prefill_runtime_policy_flags, dict)
            else None
        ),
        prefill_copy_policy_flags=(
            prefill_copy_policy_flags
            if isinstance(prefill_copy_policy_flags, dict)
            else None
        ),
        glm_4bit_guard_flags=(
            glm_4bit_flags if isinstance(glm_4bit_flags, dict) else None
        ),
        public_glm_5_2_shape_guard_flags=(
            public_glm_5_2_flags if isinstance(public_glm_5_2_flags, dict) else None
        ),
        prefill_acceleration_flags=(
            acceleration_flags if isinstance(acceleration_flags, dict) else None
        ),
        prefill_guard_flags=prefill_flags if isinstance(prefill_flags, dict) else None,
        prefill_persistent_moe_plan_server_flags=(
            prefill_persistent_moe_plan_server_flags
            if isinstance(prefill_persistent_moe_plan_server_flags, dict)
            else None
        ),
        prefill_persistent_resident_linear_server_flags=(
            prefill_persistent_resident_linear_server_flags
            if isinstance(
                prefill_persistent_resident_linear_server_flags,
                dict,
            )
            else None
        ),
        prefill_persistent_attention_projection_server_flags=(
            prefill_persistent_attention_projection_server_flags
            if isinstance(
                prefill_persistent_attention_projection_server_flags,
                dict,
            )
            else None
        ),
        prefill_persistent_attention_output_server_flags=(
            prefill_persistent_attention_output_server_flags
            if isinstance(
                prefill_persistent_attention_output_server_flags,
                dict,
            )
            else None
        ),
        prefill_persistent_shared_expert_server_flags=(
            prefill_persistent_shared_expert_server_flags
            if isinstance(prefill_persistent_shared_expert_server_flags, dict)
            else None
        ),
        prefill_persistent_rope_split_server_flags=(
            prefill_persistent_rope_split_server_flags
            if isinstance(prefill_persistent_rope_split_server_flags, dict)
            else None
        ),
        prefill_persistent_mla_attention_server_flags=(
            prefill_persistent_mla_attention_server_flags
            if isinstance(prefill_persistent_mla_attention_server_flags, dict)
            else None
        ),
        prefill_persistent_rmsnorm_server_flags=(
            prefill_persistent_rmsnorm_server_flags
            if isinstance(prefill_persistent_rmsnorm_server_flags, dict)
            else None
        ),
        prefill_moe_output_accumulator_flags=(
            prefill_moe_output_accumulator_flags
            if isinstance(prefill_moe_output_accumulator_flags, dict)
            else None
        ),
        decode_guard_flags=decode_flags if isinstance(decode_flags, dict) else None,
        final_logits_flags=final_logits_flags,
        prepared_target=prepared_target if isinstance(prepared_target, dict) else None,
        source="prepared_request_check",
    )
    if isinstance(profile, dict) and isinstance(prompt_chunk_plan, dict):
        sections = profile.setdefault("sections", {})
        if isinstance(sections, dict):
            sections["prefill_prompt_chunk_plan"] = prompt_chunk_plan
    return profile


def _inspect_request_check_requested(args: argparse.Namespace) -> bool:
    return any(
        (
            args.check_prompt_tokens is not None,
            args.check_prompt is not None,
            args.check_prompt_file is not None,
            args.check_chat_messages is not None,
            args.check_chat_messages_file is not None,
        )
    )


_REQUIRED_PREPARED_CONTEXT_BUDGET_FIELDS = (
    "prepare_auto_context_from_budget",
    "prepare_resolved_max_context_tokens",
    "prepare_decode_cache_budget_bytes",
    "prepare_decode_cache_safe_context_tokens",
    "prepare_cache_dtype",
    "prepare_cache_alignment",
)


def _prepared_context_budget_requirement(
    storage: object,
) -> dict[str, object]:
    missing: list[str] = []
    if not isinstance(storage, dict):
        missing = list(_REQUIRED_PREPARED_CONTEXT_BUDGET_FIELDS)
        values: dict[str, object] = {}
    else:
        missing = [
            field
            for field in _REQUIRED_PREPARED_CONTEXT_BUDGET_FIELDS
            if storage.get(field) is None
        ]
        values = {
            field: storage.get(field)
            for field in (
                "prepare_auto_context_from_budget",
                "prepare_requested_max_context_tokens",
                "prepare_resolved_max_context_tokens",
                "prepare_decode_cache_budget_bytes",
                "prepare_decode_cache_safe_context_tokens",
                "prepare_effective_max_cache_bytes",
                "prepare_model_max_position_embeddings",
                "prepare_cache_dtype",
                "prepare_cache_alignment",
            )
        }
    ok = not missing
    return {
        "ok": ok,
        "missing_fields": tuple(missing),
        "error": (
            None
            if ok
            else "prepared manifest is missing required context budget fields: "
            + ", ".join(missing)
        ),
    } | values


def _prepared_flags_provenance_requirement(
    storage: object,
) -> dict[str, object] | None:
    if not isinstance(storage, dict):
        return None
    applied = storage.get("prepare_flags_applied")
    source = storage.get("prepare_flags_source")
    path = storage.get("prepare_flags_path")
    sha256 = storage.get("prepare_flags_sha256")
    if applied is not True and source is None and path is None and sha256 is None:
        return None
    ok = (
        applied is True
        and source == "plan"
        and isinstance(path, str)
        and bool(path)
        and isinstance(sha256, str)
        and len(sha256) == 64
    )
    return {
        "ok": ok,
        "prepare_flags_applied": applied,
        "prepare_flags_source": source,
        "prepare_flags_path": path,
        "prepare_flags_sha256": sha256,
        "error": (
            None
            if ok
            else "prepared manifest has incomplete plan prepare-flags provenance"
        ),
    }


def _glm_4bit_readiness_audit_details(readiness: object) -> dict[str, object]:
    if not isinstance(readiness, dict):
        return {}
    fields = (
        "issues",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "moe_layer_count",
        "routed_experts",
        "experts_per_token",
        "model_config_sha256",
        "expert_layout_config_sha256",
        "expert_layout_quantization",
        "expert_layout_group_size",
        "expert_layout_model_layer_count",
        "expert_layout_moe_layer_count",
        "expected_expert_slot_bytes",
        "expected_expert_layer_bytes",
        "expected_total_expert_bytes",
        "prepared_expert_layout_bytes",
        "prepared_expert_layer_file_bytes",
        "expert_layer_file_count",
        "unique_expert_layer_file_count",
        "expert_layer_files_exact_size",
        "expected_decode_token_routed_expert_read_bytes",
        "expected_full_prompt_routed_expert_sweep_bytes",
        "prepared_resident_layout_bytes",
        "resident_layout_total_bytes",
        "resident_weight_file_bytes",
        "resident_weight_file_exact_size",
        "prepared_decode_cache_file_bytes",
        "decode_cache_layout_ok",
        "decode_cache_layout_total_bytes",
        "decode_cache_context_tokens",
        "decode_cache_segment_count",
        "decode_cache_mla_kv_segments_checked",
        "decode_cache_mla_kv_segments_ok",
        "decode_cache_dsa_index_segments_checked",
        "decode_cache_dsa_index_segments_ok",
    )
    return {
        f"glm_4bit_readiness_{field}" if field == "issues" else field: value
        for field in fields
        if (value := readiness.get(field)) is not None
    }


def _public_glm_5_2_shape_audit_details(
    readiness: object,
) -> dict[str, object]:
    if not isinstance(readiness, dict):
        return {}
    report = readiness.get("public_glm_5_2_shape")
    if not isinstance(report, dict):
        return {}
    details: dict[str, object] = {
        "public_glm_5_2_shape_mismatched_fields": report.get("mismatched_fields"),
        "dsa_full_indexer_layer_count": report.get("dsa_full_indexer_layer_count"),
        "expected_dsa_full_indexer_layer_count": report.get(
            "expected_dsa_full_indexer_layer_count"
        ),
        "dsa_full_indexer_layers": report.get("dsa_full_indexer_layers"),
        "dsa_schedule": report.get("dsa_schedule"),
    }
    checks = report.get("checks")
    if isinstance(checks, dict):
        for field in (
            "index_topk_freq",
            "index_skip_topk_offset",
            "num_nextn_predict_layers",
            "full_indexer_layers",
        ):
            check = checks.get(field)
            if isinstance(check, dict):
                details[f"public_glm_5_2_shape_check_{field}"] = check
    return {key: value for key, value in details.items() if value is not None}


def _prefill_acceleration_audit_details(
    prefill_backend: object,
) -> dict[str, object]:
    if not isinstance(prefill_backend, dict):
        return {}
    details: dict[str, object] = {
        "prefill_backend_configured_backend": prefill_backend.get(
            "configured_backend"
        ),
        "prefill_backend_effective_backend": prefill_backend.get(
            "effective_backend"
        ),
        "prefill_backend_warnings": prefill_backend.get("warnings"),
    }
    capability = prefill_backend.get("capability")
    if not isinstance(capability, dict):
        return {key: value for key, value in details.items() if value is not None}
    fields = (
        "recommended_backend",
        "host_probe_requested",
        "host_probe_path",
        "host_probe_ran",
        "host_probe_ok",
        "host_probe_error",
        "prefill_backend_probe_timeout_seconds",
        "mps_graph_runtime_available",
        "mps_graph_probe_requested",
        "mps_graph_probe_ran",
        "mps_graph_probe_ok",
        "metal4_ml_runtime_available",
        "mpp_runtime_available",
        "mpp_compile_probe_requested",
        "mpp_compile_probe_ran",
        "mpp_compile_probe_ok",
        "mpp_compile_variant",
        "mpp_compile_error",
        "mpp_run_probe_requested",
        "mpp_run_probe_ran",
        "mpp_run_probe_ok",
        "mpp_run_probe_error",
        "mpp_run_probe_max_abs_error",
        "mpp_run_probe_kernel_variant",
        "mpp_run_probe_shape",
        "mpp_run_probe_dtype",
        "mpp_run_probe_execution_path",
        "prefill_acceleration_runtimes",
        "selectable_accelerated_prefill_backends",
        "validated_accelerated_prefill_backends",
        "prefill_acceleration_runtime_gaps",
        "prefill_neural_accelerator_status",
        "selectable_prefill_acceleration_available",
        "validated_prefill_acceleration_available",
        "reasons",
    )
    details.update(
        {
            field: value
            for field in fields
            if (value := capability.get(field)) is not None
        }
    )
    return {key: value for key, value in details.items() if value is not None}


def _prepare_expert_pack_heap_source_from_prepared(
    prepared: PreparedManifest,
) -> dict[str, object]:
    return {
        "expert_quantization": prepared.expert_quantization,
        "expert_group_size": prepared.expert_group_size,
        "expert_layout_quantization": prepared.expert_layout_quantization,
        "expert_layout_group_size": prepared.expert_layout_group_size,
        "prepare_expert_pack_chunk_size_bytes": (
            prepared.prepare_expert_pack_chunk_size_bytes
        ),
        "prepare_expert_pack_estimated_peak_heap_bytes": (
            prepared.prepare_expert_pack_estimated_peak_heap_bytes
        ),
        "prepare_expert_pack_max_heap_bytes": prepared.prepare_expert_pack_max_heap_bytes,
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
    }


def _prepare_resident_alias_rewrite_source_from_prepared(
    prepared: PreparedManifest,
) -> dict[str, object]:
    return {
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
    }


def _prepare_expert_pack_heap_audit_details(
    source: object,
) -> dict[str, object]:
    if not isinstance(source, dict):
        source = {}
    manifest_expert_quantization = source.get("expert_quantization")
    layout_expert_quantization = source.get("expert_layout_quantization")
    effective_expert_quantization = (
        manifest_expert_quantization or layout_expert_quantization
    )
    values = {
        field: source.get(field)
        for field in _PREPARE_EXPERT_PACK_HEAP_AUDIT_FIELDS
    }
    required = effective_expert_quantization == "largerlm-affine-int4"
    evidence_present = any(value is not None for value in values.values())
    needs_validation = required or evidence_present
    missing_fields: list[str] = []
    invalid_fields: list[str] = []
    positive_fields = (
        "prepare_expert_pack_chunk_size_bytes",
        "prepare_expert_pack_estimated_peak_heap_bytes",
        "prepare_expert_pack_max_heap_bytes",
    )
    nonnegative_fields = tuple(
        field
        for field in _PREPARE_EXPERT_PACK_HEAP_AUDIT_FIELDS
        if field not in positive_fields
    )
    if needs_validation:
        for field in _PREPARE_EXPERT_PACK_HEAP_AUDIT_FIELDS:
            if values[field] is None:
                missing_fields.append(field)
        for field in positive_fields:
            value = values[field]
            if type(value) is not int or value <= 0:
                invalid_fields.append(field)
        for field in nonnegative_fields:
            value = values[field]
            if type(value) is not int or value < 0:
                invalid_fields.append(field)
    estimated_peak = values["prepare_expert_pack_estimated_peak_heap_bytes"]
    max_heap = values["prepare_expert_pack_max_heap_bytes"]
    within_heap = not needs_validation or (
        type(estimated_peak) is int
        and type(max_heap) is int
        and estimated_peak <= max_heap
    )
    raw_byte_fields = (
        "prepare_raw_quantization_extra_heap_bytes",
        "prepare_raw_quantization_max_source_block_bytes",
        "prepare_raw_quantization_max_output_block_bytes",
    )
    raw_bytes_present = any(
        type(values[field]) is int and values[field] > 0
        for field in raw_byte_fields
    )
    raw_rows = values["prepare_raw_quantization_max_rows_per_block"]
    rows_cover_raw = not raw_bytes_present or (type(raw_rows) is int and raw_rows > 0)
    all_fields_present = not missing_fields
    ok = (
        (not required and not evidence_present)
        or (
            all_fields_present
            and not invalid_fields
            and within_heap
            and rows_cover_raw
        )
    )
    return {
        "required": required,
        "evidence_present": evidence_present,
        "all_fields_present": all_fields_present,
        "missing_fields": tuple(missing_fields),
        "invalid_fields": tuple(invalid_fields),
        "within_pack_heap_limit": within_heap,
        "raw_quantization_rows_cover_bytes": rows_cover_raw,
        "expert_quantization": manifest_expert_quantization,
        "expert_layout_quantization": layout_expert_quantization,
        "effective_expert_quantization": effective_expert_quantization,
        "expert_group_size": source.get("expert_group_size"),
        "expert_layout_group_size": source.get("expert_layout_group_size"),
        **values,
        "ok": ok,
    }


def _prepare_resident_alias_rewrite_audit_details(
    source: object,
) -> dict[str, object]:
    if not isinstance(source, dict):
        source = {}
        source_is_dict = False
    else:
        source_is_dict = True
    values = {
        field: source.get(field)
        for field in _PREPARE_RESIDENT_ALIAS_REWRITE_AUDIT_FIELDS
    }
    invalid_fields = tuple(
        field
        for field, value in values.items()
        if value is not None and (type(value) is not int or value < 0)
    )

    def nonzero(field: str) -> bool:
        value = values[field]
        return type(value) is int and value > 0

    alias_fields = (
        "prepare_resident_component_alias_source_tensor_count",
        "prepare_resident_component_alias_renamed_tensor_count",
        "prepare_resident_component_alias_bytes",
    )
    fused_fields = (
        "prepare_resident_fused_gate_up_source_tensor_count",
        "prepare_resident_fused_gate_up_expanded_tensor_count",
        "prepare_resident_fused_gate_up_expanded_bytes",
    )
    alias_required = any(nonzero(field) for field in alias_fields)
    fused_required = any(nonzero(field) for field in fused_fields)
    alias_missing = tuple(
        field for field in alias_fields if alias_required and values[field] is None
    )
    fused_missing = tuple(
        field for field in fused_fields if fused_required and values[field] is None
    )
    alias_source = values["prepare_resident_component_alias_source_tensor_count"]
    alias_renamed = values["prepare_resident_component_alias_renamed_tensor_count"]
    alias_bytes = values["prepare_resident_component_alias_bytes"]
    alias_counts_match = not alias_required or (
        type(alias_source) is int
        and alias_source > 0
        and type(alias_renamed) is int
        and alias_renamed == alias_source
    )
    alias_bytes_present = not alias_required or (
        type(alias_bytes) is int and alias_bytes > 0
    )
    fused_source = values["prepare_resident_fused_gate_up_source_tensor_count"]
    fused_expanded = values["prepare_resident_fused_gate_up_expanded_tensor_count"]
    fused_bytes = values["prepare_resident_fused_gate_up_expanded_bytes"]
    fused_counts_match = not fused_required or (
        type(fused_source) is int
        and fused_source > 0
        and type(fused_expanded) is int
        and fused_expanded == fused_source * 2
    )
    fused_bytes_present = not fused_required or (
        type(fused_bytes) is int and fused_bytes > 0
    )
    ok = (
        source_is_dict
        and not invalid_fields
        and not alias_missing
        and not fused_missing
        and alias_counts_match
        and alias_bytes_present
        and fused_counts_match
        and fused_bytes_present
    )
    return {
        "evidence_present": any(value is not None for value in values.values()),
        "rewrite_present": alias_required or fused_required,
        "invalid_fields": invalid_fields,
        "alias_required": alias_required,
        "alias_missing_fields": alias_missing,
        "alias_counts_match": alias_counts_match,
        "alias_bytes_present": alias_bytes_present,
        "fused_required": fused_required,
        "fused_missing_fields": fused_missing,
        "fused_counts_match": fused_counts_match,
        "fused_bytes_present": fused_bytes_present,
        **values,
        "ok": ok,
    }


def _sum_optional_ints(*values: int | None) -> int | None:
    if not all(isinstance(value, int) for value in values):
        return None
    return sum(int(value) for value in values)


def _prepared_runtime_profile_audit_details(
    profile: object,
) -> dict[str, object]:
    if not isinstance(profile, dict):
        return {}
    return {
        field: value
        for field in _PREPARED_RUNTIME_PROFILE_AUDIT_FIELDS
        if (value := profile.get(field)) is not None
    }


def _prepared_runtime_profile_audit_details_from_prepared(
    prepared: PreparedManifest,
) -> dict[str, object]:
    return {
        "prepare_effective_unified_memory_bytes": (
            prepared.prepare_effective_unified_memory_bytes
        ),
        "prepare_effective_unified_memory_source": (
            prepared.prepare_effective_unified_memory_source
        ),
        "prepare_system_reserve_bytes": prepared.prepare_system_reserve_bytes,
        "prepared_recommended_max_live_working_set_bytes": (
            prepared.recommended_max_live_working_set_bytes
        ),
        "prepared_recommended_min_free_unified_memory_bytes": (
            prepared.recommended_min_free_unified_memory_bytes
        ),
        "prepared_recommended_required_available_memory_bytes": (
            _sum_optional_ints(
                prepared.recommended_max_live_working_set_bytes,
                prepared.recommended_min_free_unified_memory_bytes,
            )
        ),
    }


def _prepared_ssd_read_audit_details(
    flags: object,
    storage: object,
) -> dict[str, object]:
    flag_source = flags if isinstance(flags, dict) else {}
    storage_source = storage if isinstance(storage, dict) else {}
    values: dict[str, object] = {
        "source": flag_source.get("source"),
        "prefill_ssd_read_gib_per_second": flag_source.get(
            "prefill_ssd_read_gib_per_second"
        ),
        "matches_prepare_cold_read": flag_source.get("matches_prepare_cold_read"),
    }
    for field in _PREPARED_SSD_READ_MANIFEST_AUDIT_FIELDS:
        values[field] = storage_source.get(field)
    evidence_present = any(value is not None for value in values.values())
    invalid_fields: list[str] = []
    speed = values.get("prefill_ssd_read_gib_per_second")
    if speed is not None and not _audit_finite_number(speed):
        invalid_fields.append("prefill_ssd_read_gib_per_second")
    manifest_speed = values.get("prepare_cold_read_gib_per_second")
    if manifest_speed is not None and not _audit_finite_number(manifest_speed):
        invalid_fields.append("prepare_cold_read_gib_per_second")
    elapsed = values.get("prepare_cold_read_benchmark_elapsed_seconds")
    if elapsed is not None and not _audit_finite_number(elapsed):
        invalid_fields.append("prepare_cold_read_benchmark_elapsed_seconds")
    for field in (
        "prepare_cold_read_benchmark_requested_bytes",
        "prepare_cold_read_benchmark_measured_bytes",
    ):
        value = values.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            invalid_fields.append(field)
    source = values.get("source")
    if source is not None and not isinstance(source, str):
        invalid_fields.append("source")
    cold_source = values.get("prepare_cold_read_source")
    if cold_source is not None and not isinstance(cold_source, str):
        invalid_fields.append("prepare_cold_read_source")
    benchmark_path = values.get("prepare_cold_read_benchmark_path")
    if benchmark_path is not None and not isinstance(benchmark_path, str):
        invalid_fields.append("prepare_cold_read_benchmark_path")
    matches = values.get("matches_prepare_cold_read")
    if matches is not None and not isinstance(matches, bool):
        invalid_fields.append("matches_prepare_cold_read")
    return {
        "evidence_present": evidence_present,
        "invalid_fields": tuple(invalid_fields),
        **values,
        "ok": not invalid_fields,
    }


def _prepared_ssd_read_audit_details_from_prepared(
    prepared: PreparedManifest,
) -> dict[str, object]:
    return {
        "prepare_cold_read_gib_per_second": prepared.prepare_cold_read_gib_per_second,
        "prepare_cold_read_source": prepared.prepare_cold_read_source,
        "prepare_cold_read_benchmark_path": prepared.prepare_cold_read_benchmark_path,
        "prepare_cold_read_benchmark_requested_bytes": (
            prepared.prepare_cold_read_benchmark_requested_bytes
        ),
        "prepare_cold_read_benchmark_measured_bytes": (
            prepared.prepare_cold_read_benchmark_measured_bytes
        ),
        "prepare_cold_read_benchmark_elapsed_seconds": (
            prepared.prepare_cold_read_benchmark_elapsed_seconds
        ),
    }


def _launch_audit_from_health(
    args: argparse.Namespace,
    health: dict[str, Any],
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def add(code: str, ok: bool, message: str, **details: object) -> None:
        check: dict[str, object] = {
            "code": code,
            "ok": bool(ok),
            "message": message,
        }
        for key, value in details.items():
            if value is not None:
                check[key] = value
        checks.append(check)

    applied = health.get("applied_launch_profile")
    applied_locked = (
        isinstance(applied, dict)
        and applied.get("locked") is True
        and isinstance(applied.get("profile_flag_count"), int)
        and int(applied["profile_flag_count"]) > 0
    )
    add(
        "locked_launch_profile",
        applied_locked,
        "an applied launch profile must be exact-replay locked",
        path=applied.get("path") if isinstance(applied, dict) else None,
        profile_flag_count=(
            applied.get("profile_flag_count") if isinstance(applied, dict) else None
        ),
    )

    launch_profile = health.get("suggested_launch_profile")
    prepared_target = (
        launch_profile.get("prepared")
        if isinstance(launch_profile, dict)
        else None
    )
    identity_strength = (
        prepared_target.get("identity_strength")
        if isinstance(prepared_target, dict)
        else None
    )
    identity_warnings = (
        prepared_target.get("identity_warnings")
        if isinstance(prepared_target, dict)
        else None
    )
    add(
        "prepared_identity_strong",
        identity_strength == "strong",
        "prepared launch profile identity must be hash-bound",
        identity_strength=identity_strength,
        identity_warnings=identity_warnings,
    )

    storage = health.get("prepared_storage")
    storage_validated = (
        isinstance(storage, dict)
        and storage.get("prepared_storage_validated") is True
        and storage.get("expert_layout_backing_validated") is True
        and storage.get("resident_layout_backing_validated") is True
        and storage.get("decode_cache_file_exact_size") is True
    )
    add(
        "prepared_storage_validated",
        storage_validated,
        "prepared expert/resident/cache backing files must pass loader validation",
        expert_layout_backing_validated=(
            storage.get("expert_layout_backing_validated")
            if isinstance(storage, dict)
            else None
        ),
        resident_layout_backing_validated=(
            storage.get("resident_layout_backing_validated")
            if isinstance(storage, dict)
            else None
        ),
        decode_cache_file_exact_size=(
            storage.get("decode_cache_file_exact_size")
            if isinstance(storage, dict)
            else None
        ),
        decode_cache_file_extra_bytes=(
            storage.get("decode_cache_file_extra_bytes")
            if isinstance(storage, dict)
            else None
        ),
        expert_layout_bytes=(
            storage.get("expert_layout_bytes") if isinstance(storage, dict) else None
        ),
        resident_layout_bytes=(
            storage.get("resident_layout_bytes") if isinstance(storage, dict) else None
        ),
        decode_cache_layout_bytes=(
            storage.get("decode_cache_layout_bytes")
            if isinstance(storage, dict)
            else None
        ),
        decode_cache_file_bytes=(
            storage.get("decode_cache_file_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_expert_pack_chunk_size_bytes=(
            storage.get("prepare_expert_pack_chunk_size_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_expert_pack_estimated_peak_heap_bytes=(
            storage.get("prepare_expert_pack_estimated_peak_heap_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_expert_pack_max_heap_bytes=(
            storage.get("prepare_expert_pack_max_heap_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_raw_quantization_extra_heap_bytes=(
            storage.get("prepare_raw_quantization_extra_heap_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_raw_quantization_max_source_block_bytes=(
            storage.get("prepare_raw_quantization_max_source_block_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_raw_quantization_max_output_block_bytes=(
            storage.get("prepare_raw_quantization_max_output_block_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_raw_quantization_max_rows_per_block=(
            storage.get("prepare_raw_quantization_max_rows_per_block")
            if isinstance(storage, dict)
            else None
        ),
        prepare_resident_component_alias_source_tensor_count=(
            storage.get("prepare_resident_component_alias_source_tensor_count")
            if isinstance(storage, dict)
            else None
        ),
        prepare_resident_component_alias_renamed_tensor_count=(
            storage.get("prepare_resident_component_alias_renamed_tensor_count")
            if isinstance(storage, dict)
            else None
        ),
        prepare_resident_component_alias_bytes=(
            storage.get("prepare_resident_component_alias_bytes")
            if isinstance(storage, dict)
            else None
        ),
        prepare_resident_fused_gate_up_source_tensor_count=(
            storage.get("prepare_resident_fused_gate_up_source_tensor_count")
            if isinstance(storage, dict)
            else None
        ),
        prepare_resident_fused_gate_up_expanded_tensor_count=(
            storage.get("prepare_resident_fused_gate_up_expanded_tensor_count")
            if isinstance(storage, dict)
            else None
        ),
        prepare_resident_fused_gate_up_expanded_bytes=(
            storage.get("prepare_resident_fused_gate_up_expanded_bytes")
            if isinstance(storage, dict)
            else None
        ),
    )
    pack_heap = _prepare_expert_pack_heap_audit_details(storage)
    add(
        "prepare_expert_pack_heap_envelope_ok",
        pack_heap.get("ok") is True,
        "raw affine-int4 expert preparation heap envelope must be complete and safe",
        **{key: value for key, value in pack_heap.items() if key != "ok"},
    )
    resident_rewrite = _prepare_resident_alias_rewrite_audit_details(storage)
    add(
        "prepare_resident_alias_rewrite_ok",
        resident_rewrite.get("ok") is True,
        "resident alias rewrites must be complete and self-consistent",
        **{key: value for key, value in resident_rewrite.items() if key != "ok"},
    )

    memory_required = _prepared_memory_profile_required_by_args(args)
    add(
        "prepared_memory_profile_required",
        memory_required,
        "--require-prepared-memory-profile must be enabled",
    )
    memory_requirement = health.get("prepared_memory_profile_requirement")
    add(
        "prepared_memory_profile_ok",
        (
            isinstance(memory_requirement, dict)
            and memory_requirement.get("ok") is True
        ),
        "prepared package must carry a verified memory profile",
        error=(
            memory_requirement.get("error")
            if isinstance(memory_requirement, dict)
            else None
        ),
    )

    context_requirement = health.get("prepared_context_budget_requirement")
    add(
        "prepared_context_budget_profile_ok",
        (
            isinstance(context_requirement, dict)
            and context_requirement.get("ok") is True
        ),
        "prepared package must carry a verified decode-cache context budget",
        error=(
            context_requirement.get("error")
            if isinstance(context_requirement, dict)
            else None
        ),
        missing_fields=(
            context_requirement.get("missing_fields")
            if isinstance(context_requirement, dict)
            else None
        ),
        resolved_max_context_tokens=(
            context_requirement.get("prepare_resolved_max_context_tokens")
            if isinstance(context_requirement, dict)
            else None
        ),
        decode_cache_budget_bytes=(
            context_requirement.get("prepare_decode_cache_budget_bytes")
            if isinstance(context_requirement, dict)
            else None
        ),
        decode_cache_safe_context_tokens=(
            context_requirement.get("prepare_decode_cache_safe_context_tokens")
            if isinstance(context_requirement, dict)
            else None
        ),
    )
    flags_provenance = _prepared_flags_provenance_requirement(
        health.get("prepared_storage")
    )
    if flags_provenance is not None:
        add(
            "prepare_flags_provenance_ok",
            flags_provenance.get("ok") is True,
            "plan-derived prepare flags provenance must be complete",
            error=flags_provenance.get("error"),
            prepare_flags_source=flags_provenance.get("prepare_flags_source"),
            prepare_flags_path=flags_provenance.get("prepare_flags_path"),
            prepare_flags_sha256=flags_provenance.get("prepare_flags_sha256"),
        )

    runtime_profile = health.get("prepared_runtime_profile")
    runtime_profile_details = _prepared_runtime_profile_audit_details(runtime_profile)
    add(
        "prepared_runtime_profile_ok",
        isinstance(runtime_profile, dict) and runtime_profile.get("profile_ok") is True,
        "current machine memory must satisfy the prepared runtime profile",
        profile_ok=runtime_profile_details.get("profile_ok"),
        **{
            key: value
            for key, value in runtime_profile_details.items()
            if key != "profile_ok"
        },
    )
    ssd_read = _prepared_ssd_read_audit_details(
        health.get("suggested_prepared_ssd_read_flags"),
        storage,
    )
    add(
        "prepared_ssd_read_profile_valid",
        ssd_read.get("ok") is True,
        "prepared SSD read-speed profile must be internally valid",
        **{key: value for key, value in ssd_read.items() if key != "ok"},
    )

    memory_guard = health.get("memory_guard")
    add(
        "memory_guard_available_ok",
        isinstance(memory_guard, dict) and memory_guard.get("available_ok") is True,
        "current free memory must satisfy the configured launch memory guard",
        available_ok=(
            memory_guard.get("available_ok") if isinstance(memory_guard, dict) else None
        ),
    )
    configured_min_free = (
        memory_guard.get("configured_min_free_unified_memory_bytes")
        if isinstance(memory_guard, dict)
        else None
    )
    add(
        "memory_guard_free_reserve",
        isinstance(configured_min_free, int) and configured_min_free > 0,
        "launch memory guard must reserve non-zero free unified memory",
        configured_min_free_unified_memory_bytes=configured_min_free,
    )

    readiness = health.get("glm_4bit_readiness")
    glm_required = bool(getattr(args, "require_glm_4bit", False))
    public_shape_required = bool(getattr(args, "require_public_glm_5_2_shape", False))
    add(
        "glm_4bit_required",
        glm_required,
        "--require-glm-4bit must be enabled",
    )
    add(
        "glm_4bit_ready",
        isinstance(readiness, dict) and readiness.get("ok") is True,
        "prepared package must pass GLM 4-bit readiness",
        **_glm_4bit_readiness_audit_details(readiness),
    )
    add(
        "public_glm_5_2_shape_required",
        public_shape_required,
        "--require-public-glm-5-2-shape must be enabled",
    )
    add(
        "public_glm_5_2_shape_ok",
        (
            isinstance(readiness, dict)
            and readiness.get("matches_public_glm_5_2_shape") is True
        ),
        "prepared config must match the public GLM-5.2 shape",
        matches_public_glm_5_2_shape=(
            readiness.get("matches_public_glm_5_2_shape")
            if isinstance(readiness, dict)
            else None
        ),
        **_public_glm_5_2_shape_audit_details(readiness),
    )

    prefill_required = _prefill_acceleration_required(args)
    allow_non_accelerated_prefill_audit = bool(
        getattr(args, "allow_non_accelerated_prefill_launch_audit", False)
    )
    add(
        "prefill_acceleration_required",
        prefill_required or allow_non_accelerated_prefill_audit,
        "--require-prefill-acceleration or a positive "
        "--prefill-min-accelerated-flop-fraction must be enabled",
        required=prefill_required,
        allow_non_accelerated_prefill_launch_audit=(
            allow_non_accelerated_prefill_audit
        ),
    )
    prefill_gate = health.get("prefill_acceleration_requirement")
    prefill_backend = health.get("prefill_backend")
    prefill_gate_ok = (
        isinstance(prefill_gate, dict) and prefill_gate.get("ok") is True
    )
    add(
        "prefill_acceleration_gate_ok",
        prefill_gate_ok
        or (allow_non_accelerated_prefill_audit and not prefill_required),
        "configured prefill acceleration backend must be available",
        required=prefill_required,
        allow_non_accelerated_prefill_launch_audit=(
            allow_non_accelerated_prefill_audit
        ),
        reason=(
            prefill_gate.get("reason") if isinstance(prefill_gate, dict) else None
        ),
        reason_code=(
            prefill_gate.get("reason_code")
            if isinstance(prefill_gate, dict)
            else None
        ),
        **_prefill_acceleration_audit_details(prefill_backend),
    )
    capability = (
        prefill_backend.get("capability")
        if isinstance(prefill_backend, dict)
        else None
    )
    selectable_accel = (
        tuple(capability.get("selectable_accelerated_prefill_backends") or ())
        if isinstance(capability, dict)
        else ()
    )
    validated_accel = (
        tuple(capability.get("validated_accelerated_prefill_backends") or ())
        if isinstance(capability, dict)
        else ()
    )
    validated_accel_set = {str(item) for item in validated_accel}
    if prefill_required and "mpsgraph-f32" in selectable_accel:
        probe_ok = (
            "mpsgraph-f32" in validated_accel_set
            and capability.get("mps_graph_probe_requested") is True
            and capability.get("mps_graph_probe_ran") is True
            and capability.get("mps_graph_probe_ok") is True
        )
    elif prefill_required:
        probe_ok = bool(validated_accel_set)
    else:
        probe_ok = True
    add(
        "prefill_acceleration_probe_ok",
        probe_ok,
        "required selectable prefill acceleration backend must have a passing runtime probe",
        selectable_accelerated_prefill_backends=selectable_accel,
        validated_accelerated_prefill_backends=validated_accel,
        validated_prefill_acceleration_available=(
            capability.get("validated_prefill_acceleration_available")
            if isinstance(capability, dict)
            else None
        ),
        mps_graph_probe_requested=(
            capability.get("mps_graph_probe_requested")
            if isinstance(capability, dict)
            else None
        ),
        mps_graph_probe_ran=(
            capability.get("mps_graph_probe_ran")
            if isinstance(capability, dict)
            else None
        ),
        mps_graph_probe_ok=(
            capability.get("mps_graph_probe_ok")
            if isinstance(capability, dict)
            else None
        ),
    )
    applied_profile_flags = (
        set(_launch_profile_flags(applied)) if isinstance(applied, dict) else set()
    )
    acceleration_suggestion = (
        capability.get("suggested_prefill_acceleration_flags")
        if isinstance(capability, dict)
        else None
    )
    profile_probe_required = prefill_required and "mpsgraph-f32" in selectable_accel
    profile_replays_probe = (
        not profile_probe_required
        or "--run-mpsgraph-probe" in applied_profile_flags
    )
    add(
        "prefill_acceleration_profile_replays_probe",
        profile_replays_probe,
        "required MPSGraph acceleration launch profile must replay the runtime probe",
        required=profile_probe_required,
        has_run_mpsgraph_probe=("--run-mpsgraph-probe" in applied_profile_flags),
        runtime_probe_required=(
            acceleration_suggestion.get("runtime_probe_required")
            if isinstance(acceleration_suggestion, dict)
            else None
        ),
        runtime_probe_satisfied=(
            acceleration_suggestion.get("runtime_probe_satisfied")
            if isinstance(acceleration_suggestion, dict)
            else None
        ),
    )

    request_check = health.get("request_check")
    add(
        "request_check_requested",
        _inspect_request_check_requested(args),
        "a prompt/chat request check must be supplied",
    )
    add(
        "request_check_ok",
        isinstance(request_check, dict) and request_check.get("ok") is True,
        "request admission must pass",
        error=request_check.get("error") if isinstance(request_check, dict) else None,
    )
    runtime = (
        request_check.get("runtime_preflight")
        if isinstance(request_check, dict)
        else None
    )
    add(
        "request_runtime_preflight_ran",
        isinstance(runtime, dict) and runtime.get("ran") is True,
        "--check-runtime-preflight must run for the checked request",
        reason=runtime.get("reason") if isinstance(runtime, dict) else None,
    )
    add(
        "request_runtime_memory_ok",
        isinstance(runtime, dict) and runtime.get("available_memory_ok") is True,
        "request runtime preflight must fit current free memory",
        available_memory_ok=(
            runtime.get("available_memory_ok") if isinstance(runtime, dict) else None
        ),
        requested_context_tokens=(
            runtime.get("requested_context_tokens") if isinstance(runtime, dict) else None
        ),
        max_layer_peak_bytes=(
            runtime.get("max_layer_peak_bytes") if isinstance(runtime, dict) else None
        ),
        max_layer_cache_read_bytes=(
            runtime.get("max_layer_cache_read_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        read_bytes_per_token=(
            runtime.get("read_bytes_per_token") if isinstance(runtime, dict) else None
        ),
        final_logits_peak_bytes=(
            runtime.get("final_logits_peak_bytes") if isinstance(runtime, dict) else None
        ),
        embedding_row_bytes=(
            runtime.get("embedding_row_bytes") if isinstance(runtime, dict) else None
        ),
        embedding_output_bytes=(
            runtime.get("embedding_output_bytes") if isinstance(runtime, dict) else None
        ),
        live_working_set_bytes=(
            runtime.get("live_working_set_bytes") if isinstance(runtime, dict) else None
        ),
        resident_backing_bytes=(
            runtime.get("resident_backing_bytes") if isinstance(runtime, dict) else None
        ),
        nonresident_peak_bytes=(
            runtime.get("nonresident_peak_bytes") if isinstance(runtime, dict) else None
        ),
        extra_live_working_set_bytes=(
            runtime.get("extra_live_working_set_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        max_live_working_set_bytes=(
            runtime.get("max_live_working_set_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        min_available_memory_bytes=(
            runtime.get("min_available_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        required_available_memory_bytes=(
            runtime.get("required_available_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        system_available_memory_bytes=(
            runtime.get("system_available_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        system_total_memory_bytes=(
            runtime.get("system_total_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        system_memory_source=(
            runtime.get("system_memory_source") if isinstance(runtime, dict) else None
        ),
        prefill_live_memory=(
            runtime.get("prefill_live_memory") if isinstance(runtime, dict) else None
        ),
    )
    batch_prefill_required = (
        isinstance(request_check, dict)
        and request_check.get("batch_prefill_prompt") is True
    )
    prompt_chunk_plan = (
        request_check.get("prefill_prompt_chunk_plan")
        if isinstance(request_check, dict)
        else None
    )
    prompt_chunk_tokens = (
        request_check.get("prefill_prompt_chunk_tokens")
        if isinstance(request_check, dict)
        else None
    )
    audited_prompt_token_count = (
        request_check.get("prompt_token_count")
        if isinstance(request_check, dict)
        else None
    )
    prompt_chunk_plan_errors = (
        _prefill_prompt_chunk_plan_errors(
            prompt_chunk_plan,
            prompt_token_count=(
                audited_prompt_token_count
                if type(audited_prompt_token_count) is int
                else None
            ),
            chunk_tokens=prompt_chunk_tokens,
        )
        if isinstance(prompt_chunk_plan, dict)
        else ()
    )
    request_profile = health.get("request_launch_profile")
    request_profile_sections = (
        request_profile.get("sections")
        if isinstance(request_profile, dict)
        else None
    )
    request_profile_prompt_chunk_plan = (
        request_profile_sections.get("prefill_prompt_chunk_plan")
        if isinstance(request_profile_sections, dict)
        else None
    )
    request_profile_plan_matches = (
        isinstance(prompt_chunk_plan, dict)
        and isinstance(request_profile_prompt_chunk_plan, dict)
        and request_profile_prompt_chunk_plan == prompt_chunk_plan
    )
    prompt_chunk_plan_drift = (
        request_check.get("prefill_prompt_chunk_plan_drift")
        if isinstance(request_check, dict)
        else None
    )
    prompt_chunk_plan_drift_status = (
        prompt_chunk_plan_drift.get("status")
        if isinstance(prompt_chunk_plan_drift, dict)
        else None
    )
    prompt_chunk_plan_drift_ok = (
        not isinstance(prompt_chunk_plan_drift, dict)
        or prompt_chunk_plan_drift_status not in _PREFILL_CHUNK_PLAN_ADMISSION_FAILURES
    )
    add(
        "request_prefill_prompt_chunk_plan_ok",
        (
            not batch_prefill_required
            or (
                isinstance(prompt_chunk_plan, dict)
                and not prompt_chunk_plan_errors
                and request_profile_plan_matches
                and prompt_chunk_plan_drift_ok
            )
        ),
        "checked request must bind the max-safe prompt chunk plan and matrix scratch evidence",
        required=batch_prefill_required,
        evidence_present=isinstance(prompt_chunk_plan, dict),
        errors=prompt_chunk_plan_errors if prompt_chunk_plan_errors else None,
        request_profile_evidence_present=isinstance(
            request_profile_prompt_chunk_plan,
            dict,
        ),
        request_profile_plan_matches=request_profile_plan_matches,
        prefill_prompt_chunk_tokens=prompt_chunk_tokens,
        prefill_prompt_chunk_plan=(
            prompt_chunk_plan if isinstance(prompt_chunk_plan, dict) else None
        ),
        prefill_prompt_chunk_plan_drift=(
            prompt_chunk_plan_drift
            if isinstance(prompt_chunk_plan_drift, dict)
            else None
        ),
        prefill_prompt_chunk_plan_drift_status=prompt_chunk_plan_drift_status,
    )
    request_linear = (
        request_check.get("prefill_linear_backend")
        if isinstance(request_check, dict)
        else None
    )
    backend_effective = (
        prefill_backend.get("effective_backend")
        if isinstance(prefill_backend, dict)
        else None
    )
    request_effective = (
        request_linear.get("effective") if isinstance(request_linear, dict) else None
    )
    request_prefill_backend_effective_ok = (
        not batch_prefill_required
        or (
            isinstance(request_linear, dict)
            and request_linear.get("analyzed") is True
            and request_linear.get("configured") is not None
            and isinstance(request_effective, str)
            and bool(request_effective)
            and request_effective == backend_effective
        )
    )
    add(
        "request_prefill_backend_effective_ok",
        request_prefill_backend_effective_ok,
        "checked request must bind the runtime-resolved prefill backend",
        required=batch_prefill_required,
        configured=(
            request_linear.get("configured") if isinstance(request_linear, dict) else None
        ),
        effective=request_effective,
        health_effective_backend=backend_effective,
        analyzed=(
            request_linear.get("analyzed") if isinstance(request_linear, dict) else None
        ),
    )
    applied_source = applied.get("source") if isinstance(applied, dict) else None
    applied_prefill_actual = (
        applied.get("prefill_actual_read_time")
        if isinstance(applied, dict)
        else None
    )
    prefill_actual_required = (
        batch_prefill_required and applied_source == "benchmark_actual"
    )
    prefill_actual_errors = (
        _prefill_actual_read_time_errors(applied_prefill_actual)
        if isinstance(applied_prefill_actual, dict)
        else ()
    )
    prefill_actual_ok = (
        (not prefill_actual_required or isinstance(applied_prefill_actual, dict))
        and not prefill_actual_errors
    )
    prefill_actual_details = (
        {
            field: applied_prefill_actual.get(field)
            for field in _PREFILL_ACTUAL_READ_TIME_AUDIT_FIELDS
        }
        if isinstance(applied_prefill_actual, dict)
        else {}
    )
    add(
        "applied_prefill_actual_read_time_ok",
        prefill_actual_ok,
        "applied benchmark profile must preserve passing cumulative prefill read-time evidence",
        required=prefill_actual_required,
        applied_profile_source=applied_source,
        evidence_present=isinstance(applied_prefill_actual, dict),
        errors=prefill_actual_errors if prefill_actual_errors else None,
        **prefill_actual_details,
    )
    applied_prefill_actual_coverage = (
        applied.get("prefill_actual_acceleration_coverage")
        if isinstance(applied, dict)
        else None
    )
    prefill_actual_coverage_required = (
        batch_prefill_required
        and applied_source == "benchmark_actual"
        and (
            bool(getattr(args, "require_prefill_acceleration", False))
            or float(getattr(args, "prefill_min_accelerated_flop_fraction", 0.0) or 0.0)
            > 0.0
        )
    )
    prefill_actual_coverage_errors = (
        _request_prefill_acceleration_coverage_errors(
            applied_prefill_actual_coverage
        )
        if isinstance(applied_prefill_actual_coverage, dict)
        else ()
    )
    prefill_actual_coverage_ok = (
        (
            not prefill_actual_coverage_required
            or isinstance(applied_prefill_actual_coverage, dict)
        )
        and not prefill_actual_coverage_errors
        and (
            not isinstance(applied_prefill_actual_coverage, dict)
            or applied_prefill_actual_coverage.get("ok") is True
        )
    )
    prefill_actual_coverage_details = (
        {
            field: applied_prefill_actual_coverage.get(field)
            for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS
            if field not in {"ok", "required"}
        }
        if isinstance(applied_prefill_actual_coverage, dict)
        else {}
    )
    add(
        "applied_prefill_actual_acceleration_coverage_valid",
        prefill_actual_coverage_ok,
        "applied benchmark profile must preserve actual prefill acceleration coverage when acceleration is required",
        required=prefill_actual_coverage_required,
        applied_profile_source=applied_source,
        evidence_present=isinstance(applied_prefill_actual_coverage, dict),
        errors=(
            prefill_actual_coverage_errors
            if prefill_actual_coverage_errors
            else None
        ),
        **prefill_actual_coverage_details,
    )
    applied_prefill_linear_actual = (
        applied.get("prefill_actual_linear_backend")
        if isinstance(applied, dict)
        else None
    )
    prefill_linear_actual_errors = (
        _prefill_actual_linear_backend_errors(applied_prefill_linear_actual)
        if isinstance(applied_prefill_linear_actual, dict)
        else ()
    )
    prefill_linear_actual_details = (
        {
            field: applied_prefill_linear_actual.get(field)
            for field in _PREFILL_ACTUAL_LINEAR_BACKEND_AUDIT_FIELDS
        }
        if isinstance(applied_prefill_linear_actual, dict)
        else {}
    )
    add(
        "applied_prefill_actual_linear_backend_valid",
        not prefill_linear_actual_errors,
        "applied benchmark profile prefill linear backend evidence must be well-formed when present",
        required=False,
        applied_profile_source=applied_source,
        evidence_present=isinstance(applied_prefill_linear_actual, dict),
        errors=(
            prefill_linear_actual_errors
            if prefill_linear_actual_errors
            else None
        ),
        **prefill_linear_actual_details,
    )
    applied_decode_actual = (
        applied.get("decode_actual_read_time")
        if isinstance(applied, dict)
        else None
    )
    decode_actual_errors = (
        _decode_actual_read_time_errors(applied_decode_actual)
        if isinstance(applied_decode_actual, dict)
        else ()
    )
    decode_actual_ok = not decode_actual_errors
    decode_actual_details = (
        {
            field: applied_decode_actual.get(field)
            for field in _DECODE_ACTUAL_READ_TIME_AUDIT_FIELDS
        }
        if isinstance(applied_decode_actual, dict)
        else {}
    )
    add(
        "applied_decode_actual_read_time_ok",
        decode_actual_ok,
        "applied benchmark profile decode read-time evidence must be passing when present",
        required=False,
        applied_profile_source=applied_source,
        evidence_present=isinstance(applied_decode_actual, dict),
        errors=decode_actual_errors if decode_actual_errors else None,
        **decode_actual_details,
    )
    routed_read = (
        request_check.get("prefill_routed_expert_read")
        if isinstance(request_check, dict)
        else None
    )
    routed_read_ssd = (
        routed_read.get("ssd_read_gib_per_second")
        if isinstance(routed_read, dict)
        else None
    )
    routed_read_max_seconds = (
        routed_read.get("max_read_seconds") if isinstance(routed_read, dict) else None
    )
    routed_read_budget_ok = (
        not batch_prefill_required
        or (
            isinstance(routed_read, dict)
            and routed_read.get("analyzed") is True
            and routed_read.get("within_limit") is True
            and isinstance(routed_read_ssd, (int, float))
            and not isinstance(routed_read_ssd, bool)
            and routed_read_ssd > 0
            and isinstance(routed_read_max_seconds, (int, float))
            and not isinstance(routed_read_max_seconds, bool)
            and routed_read_max_seconds > 0
            and routed_read.get("planned_read_seconds") is not None
            and routed_read.get("within_seconds_limit") is True
        )
    )
    add(
        "request_prefill_routed_read_budget_ok",
        routed_read_budget_ok,
        "checked request must bind routed expert SSD reads to a measured speed and seconds cap",
        required=batch_prefill_required,
        analyzed=(
            routed_read.get("analyzed") if isinstance(routed_read, dict) else None
        ),
        within_limit=(
            routed_read.get("within_limit") if isinstance(routed_read, dict) else None
        ),
        baseline_read_bytes=(
            routed_read.get("baseline_read_bytes")
            if isinstance(routed_read, dict)
            else None
        ),
        planned_read_bytes=(
            routed_read.get("planned_read_bytes")
            if isinstance(routed_read, dict)
            else None
        ),
        extra_read_bytes=(
            routed_read.get("extra_read_bytes") if isinstance(routed_read, dict) else None
        ),
        read_amplification=(
            routed_read.get("read_amplification")
            if isinstance(routed_read, dict)
            else None
        ),
        max_read_amplification=(
            routed_read.get("max_read_amplification")
            if isinstance(routed_read, dict)
            else None
        ),
        within_amplification_limit=(
            routed_read.get("within_amplification_limit")
            if isinstance(routed_read, dict)
            else None
        ),
        max_planned_read_bytes=(
            routed_read.get("max_planned_read_bytes")
            if isinstance(routed_read, dict)
            else None
        ),
        within_planned_read_limit=(
            routed_read.get("within_planned_read_limit")
            if isinstance(routed_read, dict)
            else None
        ),
        ssd_read_gib_per_second=routed_read_ssd,
        planned_read_seconds=(
            routed_read.get("planned_read_seconds")
            if isinstance(routed_read, dict)
            else None
        ),
        max_read_seconds=routed_read_max_seconds,
        within_seconds_limit=(
            routed_read.get("within_seconds_limit")
            if isinstance(routed_read, dict)
            else None
        ),
        minimum_chunk_tokens_for_limits=(
            routed_read.get("minimum_chunk_tokens_for_limits")
            if isinstance(routed_read, dict)
            else None
        ),
    )
    request_decode_required = False
    if isinstance(request_check, dict):
        request_max_new = request_check.get("max_new_tokens")
        request_decode_required = (
            not isinstance(request_max_new, bool)
            and isinstance(request_max_new, int)
            and request_max_new > 0
        )
    decode_read = (
        request_check.get("decode_routed_expert_read")
        if isinstance(request_check, dict)
        else None
    )
    decode_read_ssd = (
        decode_read.get("ssd_read_gib_per_second")
        if isinstance(decode_read, dict)
        else None
    )
    decode_read_max_seconds = (
        decode_read.get("max_read_seconds_per_token")
        if isinstance(decode_read, dict)
        else None
    )
    decode_read_max_bytes = (
        decode_read.get("max_read_bytes_per_token")
        if isinstance(decode_read, dict)
        else None
    )
    decode_routed_read_budget_ok = (
        not request_decode_required
        or (
            isinstance(decode_read, dict)
            and decode_read.get("analyzed") is True
            and decode_read.get("within_limit") is True
            and decode_read.get("within_read_limit") is True
            and isinstance(decode_read_max_bytes, int)
            and not isinstance(decode_read_max_bytes, bool)
            and decode_read_max_bytes > 0
            and isinstance(decode_read_ssd, (int, float))
            and not isinstance(decode_read_ssd, bool)
            and decode_read_ssd > 0
            and isinstance(decode_read_max_seconds, (int, float))
            and not isinstance(decode_read_max_seconds, bool)
            and decode_read_max_seconds > 0
            and decode_read.get("planned_read_seconds_per_token") is not None
            and decode_read.get("within_seconds_limit") is True
        )
    )
    add(
        "request_decode_routed_read_budget_ok",
        decode_routed_read_budget_ok,
        "checked request must bind decode routed expert SSD reads to bytes/token and seconds/token caps",
        required=request_decode_required,
        analyzed=(
            decode_read.get("analyzed") if isinstance(decode_read, dict) else None
        ),
        read_bytes_per_token=(
            decode_read.get("read_bytes_per_token")
            if isinstance(decode_read, dict)
            else None
        ),
        max_read_bytes_per_token=decode_read_max_bytes,
        within_read_limit=(
            decode_read.get("within_read_limit")
            if isinstance(decode_read, dict)
            else None
        ),
        ssd_read_gib_per_second=decode_read_ssd,
        planned_read_seconds_per_token=(
            decode_read.get("planned_read_seconds_per_token")
            if isinstance(decode_read, dict)
            else None
        ),
        max_read_seconds_per_token=decode_read_max_seconds,
        within_seconds_limit=(
            decode_read.get("within_seconds_limit")
            if isinstance(decode_read, dict)
            else None
        ),
        within_limit=(
            decode_read.get("within_limit") if isinstance(decode_read, dict) else None
        ),
    )
    stage_temp = (
        request_check.get("prefill_routed_stage_temp_disk")
        if isinstance(request_check, dict)
        else None
    )
    stage_temp_errors = (
        _stage_temp_evidence_errors(
            stage_temp,
            required_fields=_REQUEST_PREFILL_STAGE_TEMP_AUDIT_FIELDS,
        )
        if isinstance(stage_temp, dict)
        else ()
    )
    add(
        "request_prefill_stage_temp_limit_ok",
        (
            not batch_prefill_required
            or (
                isinstance(stage_temp, dict)
                and stage_temp.get("analyzed") is True
                and stage_temp.get("within_limit") is True
                and not stage_temp_errors
            )
        ),
        "checked request must fit configured routed stage temp caps",
        required=batch_prefill_required,
        within_limit=(
            stage_temp.get("within_limit") if isinstance(stage_temp, dict) else None
        ),
        max_stage_bytes=(
            stage_temp.get("max_stage_bytes") if isinstance(stage_temp, dict) else None
        ),
        max_stage_limit_bytes=(
            stage_temp.get("max_stage_limit_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_compact_stage_bytes=(
            stage_temp.get("max_compact_stage_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_compact_stage_limit_bytes=(
            stage_temp.get("max_compact_stage_limit_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_stage_raw_ranges=(
            stage_temp.get("max_stage_raw_ranges")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_stage_raw_range_limit=(
            stage_temp.get("max_stage_raw_range_limit")
            if isinstance(stage_temp, dict)
            else None
        ),
        within_stage_raw_range_limit=(
            stage_temp.get("within_stage_raw_range_limit")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_stage_coalesced_ranges=(
            stage_temp.get("max_stage_coalesced_ranges")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_stage_coalesced_range_limit=(
            stage_temp.get("max_stage_coalesced_range_limit")
            if isinstance(stage_temp, dict)
            else None
        ),
        within_stage_coalesced_range_limit=(
            stage_temp.get("within_stage_coalesced_range_limit")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_stage_plus_compact_bytes=(
            stage_temp.get("max_stage_plus_compact_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        total_stage_plus_compact_bytes=(
            stage_temp.get("total_stage_plus_compact_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_static_capacity_binary_bytes=(
            stage_temp.get("max_static_capacity_binary_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        total_static_capacity_binary_bytes=(
            stage_temp.get("total_static_capacity_binary_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        max_stage_plus_compact_plus_static_bytes=(
            stage_temp.get("max_stage_plus_compact_plus_static_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        total_stage_plus_compact_plus_static_bytes=(
            stage_temp.get("total_stage_plus_compact_plus_static_bytes")
            if isinstance(stage_temp, dict)
            else None
        ),
        static_capacity_per_expert=(
            stage_temp.get("static_capacity_per_expert")
            if isinstance(stage_temp, dict)
            else None
        ),
        allow_static_capacity_overflow=(
            stage_temp.get("allow_static_capacity_overflow")
            if isinstance(stage_temp, dict)
            else None
        ),
        errors=stage_temp_errors if stage_temp_errors else None,
    )
    stage_free = (
        request_check.get("prefill_stage_temp_disk_free")
        if isinstance(request_check, dict)
        else None
    )
    add(
        "request_prefill_stage_temp_disk_ok",
        (
            not batch_prefill_required
            or (
                isinstance(stage_free, dict)
                and stage_free.get("analyzed") is True
                and stage_free.get("within_free_space") is True
            )
        ),
        "checked request must have enough prompt prefill temp disk space",
        required=batch_prefill_required,
        within_free_space=(
            stage_free.get("within_free_space")
            if isinstance(stage_free, dict)
            else None
        ),
        required_free_bytes=(
            stage_free.get("required_free_bytes")
            if isinstance(stage_free, dict)
            else None
        ),
        free_bytes=(
            stage_free.get("free_bytes") if isinstance(stage_free, dict) else None
        ),
        path=stage_free.get("path") if isinstance(stage_free, dict) else None,
    )
    cache_io = (
        request_check.get("prefill_cache_io")
        if isinstance(request_check, dict)
        else None
    )
    cache_io_errors = (
        _prefill_cache_io_errors(cache_io)
        if isinstance(cache_io, dict)
        else ()
    )
    cache_io_details = (
        {
            field: cache_io.get(field)
            for field in _REQUEST_PREFILL_CACHE_IO_AUDIT_FIELDS
        }
        if isinstance(cache_io, dict)
        else {}
    )
    add(
        "request_prefill_cache_io_valid",
        not cache_io_errors,
        "checked request prefill cache I/O evidence must be well-formed when present",
        required=False,
        evidence_present=isinstance(cache_io, dict),
        errors=cache_io_errors if cache_io_errors else None,
        **cache_io_details,
    )
    routed_frontier = (
        request_check.get("prefill_routed_chunk_frontier")
        if isinstance(request_check, dict)
        else None
    )
    routed_frontier_errors = (
        _prefill_routed_chunk_frontier_errors(routed_frontier)
        if isinstance(routed_frontier, dict)
        else ()
    )
    routed_frontier_details = (
        {
            field: routed_frontier.get(field)
            for field in _REQUEST_PREFILL_ROUTED_CHUNK_FRONTIER_AUDIT_FIELDS
        }
        if isinstance(routed_frontier, dict)
        else {}
    )
    add(
        "request_prefill_routed_chunk_frontier_valid",
        not routed_frontier_errors,
        "checked request routed chunk frontier evidence must be well-formed when present",
        required=False,
        evidence_present=isinstance(routed_frontier, dict),
        errors=routed_frontier_errors if routed_frontier_errors else None,
        **routed_frontier_details,
    )
    coverage = (
        request_check.get("prefill_acceleration_coverage")
        if isinstance(request_check, dict)
        else None
    )
    coverage_errors = (
        _request_prefill_acceleration_coverage_errors(coverage)
        if isinstance(coverage, dict)
        else ()
    )
    coverage_details = (
        {
            field: coverage.get(field)
            for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS
            if field != "ok"
        }
        if isinstance(coverage, dict)
        else {}
    )
    add(
        "request_prefill_acceleration_coverage_ok",
        (
            isinstance(coverage, dict)
            and coverage.get("ok") is True
            and not coverage_errors
        ),
        "checked request must resolve to the required accelerated prefill coverage",
        evidence_present=isinstance(coverage, dict),
        errors=coverage_errors if coverage_errors else None,
        **coverage_details,
    )
    request_profile = health.get("request_launch_profile")
    add(
        "request_launch_profile_safe",
        (
            isinstance(request_profile, dict)
            and request_profile.get("argv_safe_to_replay") is True
        ),
        "checked request must produce a safe replay launch profile",
    )

    failures = tuple(check["code"] for check in checks if check.get("ok") is not True)
    return {
        "required": True,
        "ok": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "checks": tuple(checks),
    }


def _launch_audit_artifact_from_health(health: dict[str, Any]) -> dict[str, object]:
    launch_profile = health.get("suggested_launch_profile")
    prepared_target = (
        launch_profile.get("prepared")
        if isinstance(launch_profile, dict)
        else None
    )
    return {
        "schema": "largerlm.launch_audit.v1",
        "source": "inspect_prepared",
        "prepared": prepared_target,
        "applied_launch_profile": health.get("applied_launch_profile"),
        "launch_audit": health.get("launch_audit"),
        "request_check": health.get("request_check"),
        "request_launch_profile": health.get("request_launch_profile"),
    }


def _write_launch_audit_file(
    path_arg: str | Path,
    health: dict[str, Any],
) -> None:
    artifact = _launch_audit_artifact_from_health(health)
    audit = artifact.get("launch_audit")
    if not isinstance(audit, dict):
        raise CliArgumentError("no launch audit is available to write")
    path = Path(path_arg)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(artifact, default=_json_default, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise CliArgumentError(f"failed to write launch audit {path}: {exc}") from exc


def _load_launch_audit_artifact(path_arg: str | Path) -> dict[str, object]:
    path = Path(path_arg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliArgumentError(f"failed to read launch audit {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliArgumentError(f"failed to parse launch audit {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CliArgumentError("launch audit must be a JSON object")
    if payload.get("schema") != "largerlm.launch_audit.v1":
        raise CliArgumentError("launch audit schema is not largerlm.launch_audit.v1")
    return payload


def _require_launch_audit_checks(audit: dict[str, object]) -> None:
    raw_checks = audit.get("checks")
    if not isinstance(raw_checks, (list, tuple)):
        raise CliArgumentError("launch audit is missing required checks")
    checks_by_code: dict[str, dict[str, object]] = {}
    for check in raw_checks:
        if not isinstance(check, dict):
            continue
        code = check.get("code")
        if isinstance(code, str):
            checks_by_code[code] = check
    missing = tuple(
        code
        for code in _REQUIRED_LAUNCH_AUDIT_CHECK_CODES
        if code not in checks_by_code
    )
    if missing:
        raise CliArgumentError(
            "launch audit is missing required checks: "
            + ", ".join(missing)
        )
    failed = tuple(
        code
        for code in _REQUIRED_LAUNCH_AUDIT_CHECK_CODES
        if checks_by_code[code].get("ok") is not True
    )
    if failed:
        raise CliArgumentError(
            "launch audit required checks did not pass: "
            + ", ".join(failed)
        )


def _launch_audit_check_by_code(
    audit: dict[str, object],
    code: str,
) -> dict[str, object] | None:
    raw_checks = audit.get("checks")
    if not isinstance(raw_checks, (list, tuple)):
        return None
    for check in raw_checks:
        if isinstance(check, dict) and check.get("code") == code:
            return check
    return None


def _require_launch_audit_glm_envelope_matches_prepared(
    audit: dict[str, object],
    prepared: PreparedManifest,
) -> None:
    check = _launch_audit_check_by_code(audit, "glm_4bit_ready")
    if not isinstance(check, dict):
        return
    missing_fields = tuple(
        field for field in _GLM_4BIT_AUDIT_ENVELOPE_FIELDS if field not in check
    )
    if missing_fields:
        raise CliArgumentError(
            "launch audit GLM 4-bit envelope is missing required fields: "
            + ", ".join(missing_fields)
        )
    audited_fields = {
        field: check[field]
        for field in _GLM_4BIT_AUDIT_ENVELOPE_FIELDS
    }
    try:
        readiness = prepared_glm_4bit_readiness(prepared)
    except (ConfigError, PreparedServerError, PreparedManifestError) as exc:
        raise CliArgumentError(
            "failed to revalidate launch audit GLM 4-bit envelope: " + str(exc)
        ) from exc
    if readiness.get("ok") is not True:
        raise CliArgumentError(
            "current prepared package no longer passes GLM 4-bit readiness"
        )
    current = _glm_4bit_readiness_audit_details(readiness)
    for field, audited in audited_fields.items():
        if current.get(field) == audited:
            continue
        raise CliArgumentError(
            "launch audit GLM 4-bit envelope does not match this prepared package: "
            f"{field} audit={audited!r} current={current.get(field)!r}"
        )


def _require_launch_audit_prepare_pack_heap_matches_prepared(
    audit: dict[str, object],
    prepared: PreparedManifest,
) -> None:
    check = _launch_audit_check_by_code(
        audit,
        "prepare_expert_pack_heap_envelope_ok",
    )
    if not isinstance(check, dict):
        return
    current = _prepare_expert_pack_heap_audit_details(
        _prepare_expert_pack_heap_source_from_prepared(prepared)
    )
    current_has_evidence = (
        current.get("required") is True
        or current.get("evidence_present") is True
    )
    audited_has_evidence = (
        check.get("required") is True
        or check.get("evidence_present") is True
    )
    if not current_has_evidence and not audited_has_evidence:
        return
    required_fields = (
        "required",
        "evidence_present",
        "all_fields_present",
        "within_pack_heap_limit",
        "raw_quantization_rows_cover_bytes",
        "expert_quantization",
        "expert_group_size",
        *_PREPARE_EXPERT_PACK_HEAP_AUDIT_FIELDS,
    )
    missing_fields = tuple(field for field in required_fields if field not in check)
    if missing_fields:
        raise CliArgumentError(
            "launch audit prepare expert-pack heap envelope is missing required "
            "fields: "
            + ", ".join(missing_fields)
        )
    if current.get("ok") is not True:
        raise CliArgumentError(
            "current prepared package does not carry a safe prepare expert-pack "
            "heap envelope"
        )
    for field in required_fields:
        if check.get(field) == current.get(field):
            continue
        raise CliArgumentError(
            "launch audit prepare expert-pack heap envelope does not match this "
            f"prepared package: {field} audit={check.get(field)!r} "
            f"current={current.get(field)!r}"
        )


def _require_launch_audit_resident_alias_rewrite_matches_prepared(
    audit: dict[str, object],
    prepared: PreparedManifest,
) -> None:
    check = _launch_audit_check_by_code(
        audit,
        "prepare_resident_alias_rewrite_ok",
    )
    if not isinstance(check, dict):
        return
    current = _prepare_resident_alias_rewrite_audit_details(
        _prepare_resident_alias_rewrite_source_from_prepared(prepared)
    )
    current_has_evidence = current.get("evidence_present") is True
    audited_has_evidence = check.get("evidence_present") is True
    if not current_has_evidence and not audited_has_evidence:
        return
    required_base_fields = (
        "evidence_present",
        "rewrite_present",
        "invalid_fields",
        "alias_required",
        "alias_missing_fields",
        "alias_counts_match",
        "alias_bytes_present",
        "fused_required",
        "fused_missing_fields",
        "fused_counts_match",
        "fused_bytes_present",
    )
    required_value_fields = tuple(
        field
        for field in _PREPARE_RESIDENT_ALIAS_REWRITE_AUDIT_FIELDS
        if current.get(field) is not None or check.get(field) is not None
    )
    required_fields = (*required_base_fields, *required_value_fields)
    missing_fields = tuple(field for field in required_fields if field not in check)
    if missing_fields:
        raise CliArgumentError(
            "launch audit resident alias rewrite evidence is missing required "
            "fields: "
            + ", ".join(missing_fields)
        )
    if current.get("ok") is not True:
        raise CliArgumentError(
            "current prepared package does not carry self-consistent resident "
            "alias rewrite evidence"
        )
    for field in required_fields:
        audit_value = check.get(field)
        current_value = current.get(field)
        if isinstance(audit_value, (list, tuple)) and isinstance(
            current_value,
            (list, tuple),
        ):
            matches = tuple(audit_value) == tuple(current_value)
        else:
            matches = audit_value == current_value
        if matches:
            continue
        raise CliArgumentError(
            "launch audit resident alias rewrite evidence does not match this "
            f"prepared package: {field} audit={audit_value!r} "
            f"current={current_value!r}"
        )


def _require_launch_audit_prepared_runtime_profile_matches_prepared(
    audit: dict[str, object],
    prepared: PreparedManifest,
) -> None:
    check = _launch_audit_check_by_code(audit, "prepared_runtime_profile_ok")
    if not isinstance(check, dict):
        return
    if not any(
        field in check for field in _PREPARED_RUNTIME_PROFILE_MANIFEST_AUDIT_FIELDS
    ):
        return
    current = _prepared_runtime_profile_audit_details_from_prepared(prepared)
    missing_fields = tuple(
        field
        for field in _PREPARED_RUNTIME_PROFILE_MANIFEST_AUDIT_FIELDS
        if current.get(field) is not None and field not in check
    )
    if missing_fields:
        raise CliArgumentError(
            "launch audit prepared runtime profile envelope is missing required "
            "fields: "
            + ", ".join(missing_fields)
        )
    for field in _PREPARED_RUNTIME_PROFILE_MANIFEST_AUDIT_FIELDS:
        audited = check.get(field)
        current_value = current.get(field)
        if audited == current_value:
            continue
        raise CliArgumentError(
            "launch audit prepared runtime profile envelope does not match this "
            f"prepared package: {field} audit={audited!r} "
            f"current={current_value!r}"
        )


def _audit_values_match(audited: object, current: object) -> bool:
    if _audit_finite_number(audited) and _audit_finite_number(current):
        return math.isclose(
            float(audited),
            float(current),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    return audited == current


def _require_launch_audit_prepared_ssd_read_matches_prepared(
    audit: dict[str, object],
    prepared: PreparedManifest,
) -> None:
    check = _launch_audit_check_by_code(audit, "prepared_ssd_read_profile_valid")
    if not isinstance(check, dict):
        return
    if not any(field in check for field in _PREPARED_SSD_READ_MANIFEST_AUDIT_FIELDS):
        return
    if check.get("ok") is not True:
        raise CliArgumentError(
            "launch audit prepared SSD read-speed profile check is not passing: "
            f"ok={check.get('ok')!r}"
        )
    current = _prepared_ssd_read_audit_details_from_prepared(prepared)
    missing_fields = tuple(
        field
        for field in _PREPARED_SSD_READ_MANIFEST_AUDIT_FIELDS
        if current.get(field) is not None and field not in check
    )
    if missing_fields:
        raise CliArgumentError(
            "launch audit prepared SSD read-speed envelope is missing required "
            "fields: "
            + ", ".join(missing_fields)
        )
    for field in _PREPARED_SSD_READ_MANIFEST_AUDIT_FIELDS:
        audited = check.get(field)
        current_value = current.get(field)
        if field not in check and current_value is None:
            continue
        if _audit_values_match(audited, current_value):
            continue
        raise CliArgumentError(
            "launch audit prepared SSD read-speed envelope does not match this "
            f"prepared package: {field} audit={audited!r} "
            f"current={current_value!r}"
        )


def _require_launch_audit_prefill_acceleration_evidence(
    audit: dict[str, object],
) -> None:
    check = _launch_audit_check_by_code(audit, "prefill_acceleration_gate_ok")
    if not isinstance(check, dict):
        return
    if check.get("required") is False:
        if check.get("allow_non_accelerated_prefill_launch_audit") is True:
            return
        raise CliArgumentError(
            "launch audit prefill acceleration evidence is non-required without "
            "allow_non_accelerated_prefill_launch_audit=true"
        )
    missing_fields = tuple(
        field for field in _PREFILL_ACCELERATION_AUDIT_FIELDS if field not in check
    )
    if missing_fields:
        raise CliArgumentError(
            "launch audit prefill acceleration evidence is missing required fields: "
            + ", ".join(missing_fields)
        )
    neural_status = check.get("prefill_neural_accelerator_status")
    if not isinstance(neural_status, dict):
        raise CliArgumentError(
            "launch audit prefill acceleration evidence is missing "
            "prefill_neural_accelerator_status"
        )
    missing_neural_fields = tuple(
        field
        for field in _PREFILL_NEURAL_ACCELERATOR_AUDIT_FIELDS
        if field not in neural_status
    )
    if missing_neural_fields:
        raise CliArgumentError(
            "launch audit prefill neural accelerator status is missing required "
            "fields: "
            + ", ".join(missing_neural_fields)
        )
    if neural_status.get("mpp_run_probe_ok") is True:
        missing_run_fields = tuple(
            field
            for field in _PREFILL_NEURAL_ACCELERATOR_RUN_PROBE_AUDIT_FIELDS
            if field not in neural_status
        )
        if missing_run_fields:
            raise CliArgumentError(
                "launch audit prefill neural accelerator MPP run probe evidence "
                "is missing required fields: "
                + ", ".join(missing_run_fields)
            )
        bad_run_fields: list[str] = []
        kernel_variant = neural_status.get("mpp_run_probe_kernel_variant")
        if not isinstance(kernel_variant, str) or not kernel_variant:
            bad_run_fields.append(
                f"mpp_run_probe_kernel_variant={kernel_variant!r}"
            )
        shape = neural_status.get("mpp_run_probe_shape")
        if shape != "32x32x32":
            bad_run_fields.append(f"mpp_run_probe_shape={shape!r}")
        dtype = neural_status.get("mpp_run_probe_dtype")
        if dtype != "half":
            bad_run_fields.append(f"mpp_run_probe_dtype={dtype!r}")
        execution_path = neural_status.get("mpp_run_probe_execution_path")
        if execution_path != "mpp::tensor_ops::matmul2d":
            bad_run_fields.append(
                f"mpp_run_probe_execution_path={execution_path!r}"
            )
        if bad_run_fields:
            raise CliArgumentError(
                "launch audit prefill neural accelerator MPP run probe evidence "
                "is not passing: "
                + ", ".join(bad_run_fields)
            )
    if check.get("reason_code") != "ok":
        raise CliArgumentError(
            "launch audit prefill acceleration evidence is not passing: "
            f"reason_code={check.get('reason_code')!r}"
        )
    bad_host_probe_fields: list[str] = []
    if check.get("host_probe_requested") is not True:
        bad_host_probe_fields.append(
            f"host_probe_requested={check.get('host_probe_requested')!r}"
        )
    host_probe_path = check.get("host_probe_path")
    if not isinstance(host_probe_path, str) or not host_probe_path:
        bad_host_probe_fields.append(f"host_probe_path={host_probe_path!r}")
    if check.get("host_probe_ran") is not True:
        bad_host_probe_fields.append(f"host_probe_ran={check.get('host_probe_ran')!r}")
    if check.get("host_probe_ok") is not True:
        bad_host_probe_fields.append(f"host_probe_ok={check.get('host_probe_ok')!r}")
    probe_timeout = check.get("prefill_backend_probe_timeout_seconds")
    if not _audit_finite_number(probe_timeout) or float(probe_timeout) <= 0.0:
        bad_host_probe_fields.append(
            "prefill_backend_probe_timeout_seconds="
            f"{probe_timeout!r}"
        )
    if bad_host_probe_fields:
        raise CliArgumentError(
            "launch audit prefill backend host probe evidence is not passing: "
            + ", ".join(bad_host_probe_fields)
        )
    selectable = check.get("selectable_accelerated_prefill_backends")
    selectable_set = (
        {str(item) for item in selectable}
        if isinstance(selectable, (list, tuple))
        else set()
    )
    if not selectable_set or check.get("selectable_prefill_acceleration_available") is not True:
        raise CliArgumentError(
            "launch audit prefill acceleration evidence has no selectable "
            "accelerated backend"
        )
    validated = check.get("validated_accelerated_prefill_backends")
    validated_set = (
        {str(item) for item in validated}
        if isinstance(validated, (list, tuple))
        else set()
    )
    if (
        not validated_set
        or check.get("validated_prefill_acceleration_available") is not True
    ):
        raise CliArgumentError(
            "launch audit prefill acceleration evidence has no validated "
            "accelerated backend"
        )
    unexpected_validated = tuple(sorted(validated_set - selectable_set))
    if unexpected_validated:
        raise CliArgumentError(
            "launch audit prefill acceleration evidence has validated backends "
            "that are not selectable: "
            + ", ".join(unexpected_validated)
        )
    if "mpsgraph-f32" in selectable_set:
        for field in (
            "mps_graph_runtime_available",
            "mps_graph_probe_requested",
            "mps_graph_probe_ran",
            "mps_graph_probe_ok",
        ):
            if check.get(field) is True:
                continue
            raise CliArgumentError(
                "launch audit MPSGraph prefill evidence is not passing: "
                f"{field}={check.get(field)!r}"
            )
    if (
        neural_status.get("selectable") is True
        and neural_status.get("ready_for_generation") is not True
    ):
        raise CliArgumentError(
            "launch audit prefill neural accelerator status is inconsistent: "
            "selectable requires ready_for_generation"
        )


def _audit_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _audit_nonnegative_int_mapping_errors(
    payload: dict[str, object],
    field: str,
    *,
    require_nonempty: bool,
) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, dict):
        return (f"{field}={value!r}",)
    if require_nonempty and not value:
        return (f"{field}=empty",)
    errors: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{field} key={key!r}")
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            errors.append(f"{field}[{key!r}]={item!r}")
    return tuple(errors)


def _audit_nonnegative_float_mapping_errors(
    payload: dict[str, object],
    field: str,
    *,
    require_nonempty: bool,
) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, dict):
        return (f"{field}={value!r}",)
    if require_nonempty and not value:
        return (f"{field}=empty",)
    errors: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{field} key={key!r}")
            continue
        if not _audit_finite_number(item) or float(item) < 0:
            errors.append(f"{field}[{key!r}]={item!r}")
    return tuple(errors)


def _audit_nonnegative_int_mapping_values(
    payload: dict[str, object],
    field: str,
) -> dict[str, int] | None:
    value = payload.get(field)
    if not isinstance(value, dict):
        return None
    converted: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            return None
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        converted[key] = item
    return converted


def _prefill_live_memory_errors(
    live: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_LIVE_MEMORY_AUDIT_FIELDS
        if field not in live
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    values: dict[str, int] = {}
    for field in _REQUEST_PREFILL_LIVE_MEMORY_AUDIT_FIELDS:
        value = live.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        values[field] = value
    if len(values) == len(_REQUEST_PREFILL_LIVE_MEMORY_AUDIT_FIELDS):
        expected = max(
            values["prompt_batch_bytes"] + values["runner_scratch_bytes"],
            values["cache_read_bytes"] + values["runner_scratch_bytes"],
            values["cache_write_bytes"] + values["runner_scratch_bytes"],
            values["stage_copy_bytes"] + values["runner_scratch_bytes"],
        )
        actual = values["estimated_live_working_set_bytes"]
        if actual != expected:
            bad_fields.append(
                "estimated_live_working_set_bytes="
                f"{actual!r} expected={expected!r}"
            )
    return tuple(bad_fields)


def _prefill_prompt_chunk_cap_errors(
    cap: object,
    *,
    source: str,
) -> tuple[str, ...]:
    if not isinstance(cap, dict):
        return (f"{source}={cap!r}",)
    bad_fields: list[str] = []
    name = cap.get("name")
    if not isinstance(name, str) or not name:
        bad_fields.append(f"{source}.name={name!r}")
    tokens = cap.get("tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        bad_fields.append(f"{source}.tokens={tokens!r}")
    bytes_available = cap.get("bytes_available")
    if bytes_available is not None and (
        isinstance(bytes_available, bool)
        or not isinstance(bytes_available, int)
        or bytes_available < 0
    ):
        bad_fields.append(f"{source}.bytes_available={bytes_available!r}")
    bytes_per_token = cap.get("bytes_per_token")
    if bytes_per_token is not None and (
        isinstance(bytes_per_token, bool)
        or not isinstance(bytes_per_token, int)
        or bytes_per_token <= 0
    ):
        bad_fields.append(f"{source}.bytes_per_token={bytes_per_token!r}")
    detail = cap.get("detail")
    if detail is not None and not isinstance(detail, str):
        bad_fields.append(f"{source}.detail={detail!r}")
    return tuple(bad_fields)


def _prefill_prompt_chunk_plan_summary_errors(
    summary: object,
    *,
    source: str,
    prompt_token_count: int | None,
) -> tuple[str, ...]:
    if not isinstance(summary, dict):
        return (f"{source}={summary!r}",)
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_PROMPT_CHUNK_PLAN_SUMMARY_FIELDS
        if field not in summary
    )
    bad_fields = [f"{source}.missing={field}" for field in missing_fields]
    values: dict[str, int] = {}
    positive_fields = (
        "prompt_tokens",
        "raw_tokens",
        "chunk_tokens",
        "tile_tokens",
        "limiting_cap_tokens",
        "hidden_dim",
        "per_token_activation_bytes",
        "per_token_disk_bytes",
    )
    nonnegative_fields = (
        "start_position",
        "max_matrix_scratch_bytes",
        "usable_disk_bytes",
    )
    for field in positive_fields:
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{source}.{field}={value!r}")
            continue
        values[field] = value
    for field in nonnegative_fields:
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{source}.{field}={value!r}")
            continue
        values[field] = value
    next_scratch = summary.get("next_token_matrix_scratch_bytes")
    if next_scratch is not None and (
        isinstance(next_scratch, bool)
        or not isinstance(next_scratch, int)
        or next_scratch < 0
    ):
        bad_fields.append(
            f"{source}.next_token_matrix_scratch_bytes={next_scratch!r}"
        )
    prompt_tokens = values.get("prompt_tokens")
    raw_tokens = values.get("raw_tokens")
    chunk_tokens = values.get("chunk_tokens")
    limiting_cap_tokens = values.get("limiting_cap_tokens")
    if (
        prompt_token_count is not None
        and prompt_tokens is not None
        and prompt_tokens != prompt_token_count
    ):
        bad_fields.append(
            f"{source}.prompt_tokens={prompt_tokens!r} "
            f"expected={prompt_token_count!r}"
        )
    if (
        chunk_tokens is not None
        and prompt_tokens is not None
        and chunk_tokens > prompt_tokens
    ):
        bad_fields.append(
            f"{source}.chunk_tokens={chunk_tokens!r} exceeds prompt_tokens"
        )
    if (
        chunk_tokens is not None
        and raw_tokens is not None
        and chunk_tokens > raw_tokens
    ):
        bad_fields.append(f"{source}.chunk_tokens={chunk_tokens!r} exceeds raw_tokens")
    if (
        limiting_cap_tokens is not None
        and raw_tokens is not None
        and limiting_cap_tokens != raw_tokens
    ):
        bad_fields.append(
            f"{source}.limiting_cap_tokens={limiting_cap_tokens!r} "
            f"expected={raw_tokens!r}"
        )
    if chunk_tokens is not None and prompt_tokens is not None:
        if chunk_tokens < prompt_tokens and next_scratch is None:
            bad_fields.append(
                f"{source}.next_token_matrix_scratch_bytes=None before prompt end"
            )
        if chunk_tokens >= prompt_tokens and next_scratch is not None:
            bad_fields.append(
                f"{source}.next_token_matrix_scratch_bytes={next_scratch!r} "
                "after prompt end"
            )
    for list_field in ("caps", "limiting_caps"):
        raw_items = summary.get(list_field)
        if not isinstance(raw_items, (list, tuple)) or not raw_items:
            bad_fields.append(f"{source}.{list_field}={raw_items!r}")
            continue
        for index, item in enumerate(raw_items):
            bad_fields.extend(
                _prefill_prompt_chunk_cap_errors(
                    item,
                    source=f"{source}.{list_field}[{index}]",
                )
            )
    return tuple(bad_fields)


def _prefill_prompt_chunk_token_errors(
    chunk_tokens: object,
    *,
    source: str,
    plan: dict[str, object],
) -> tuple[str, ...]:
    if not isinstance(chunk_tokens, dict):
        return (f"{source}={chunk_tokens!r}",)
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_PROMPT_CHUNK_TOKEN_FIELDS
        if field not in chunk_tokens
    )
    bad_fields = [f"{source}.missing={field}" for field in missing_fields]
    values: dict[str, int] = {}
    configured = chunk_tokens.get("configured")
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured < 0
    ):
        bad_fields.append(f"{source}.configured={configured!r}")
    else:
        values["configured"] = configured
    for field in ("resolved", "max_safe"):
        value = chunk_tokens.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{source}.{field}={value!r}")
            continue
        values[field] = value
    max_safe_plan = plan.get("max_safe")
    max_safe_plan_tokens = (
        max_safe_plan.get("chunk_tokens")
        if isinstance(max_safe_plan, dict)
        else None
    )
    auto_plan = plan.get("auto")
    auto_plan_tokens = (
        auto_plan.get("chunk_tokens") if isinstance(auto_plan, dict) else None
    )
    configured_is_auto = plan.get("configured_is_auto")
    resolved = values.get("resolved")
    max_safe = values.get("max_safe")
    if resolved is not None and max_safe is not None and resolved > max_safe:
        bad_fields.append(f"{source}.resolved={resolved!r} exceeds max_safe")
    if (
        max_safe is not None
        and type(max_safe_plan_tokens) is int
        and max_safe != max_safe_plan_tokens
    ):
        bad_fields.append(
            f"{source}.max_safe={max_safe!r} "
            f"expected={max_safe_plan_tokens!r}"
        )
    if configured_is_auto is True and resolved is not None:
        if type(auto_plan_tokens) is not int or resolved != auto_plan_tokens:
            bad_fields.append(
                f"{source}.resolved={resolved!r} "
                f"expected_auto={auto_plan_tokens!r}"
            )
    if (
        configured_is_auto is False
        and resolved is not None
        and "configured" in values
        and resolved != values["configured"]
    ):
        bad_fields.append(
            f"{source}.resolved={resolved!r} "
            f"expected_configured={values['configured']!r}"
        )
    return tuple(bad_fields)


def _prefill_prompt_chunk_plan_errors(
    plan: object,
    *,
    prompt_token_count: int | None = None,
    chunk_tokens: object | None = None,
) -> tuple[str, ...]:
    if not isinstance(plan, dict):
        return (f"prefill_prompt_chunk_plan={plan!r}",)
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_PROMPT_CHUNK_PLAN_FIELDS
        if field not in plan
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    source = plan.get("source")
    if source != "prepared_request_check":
        bad_fields.append(f"source={source!r}")
    configured_is_auto = plan.get("configured_is_auto")
    if not isinstance(configured_is_auto, bool):
        bad_fields.append(f"configured_is_auto={configured_is_auto!r}")
    max_safe = plan.get("max_safe")
    bad_fields.extend(
        _prefill_prompt_chunk_plan_summary_errors(
            max_safe,
            source="max_safe",
            prompt_token_count=prompt_token_count,
        )
    )
    auto = plan.get("auto")
    if configured_is_auto is True:
        bad_fields.extend(
            _prefill_prompt_chunk_plan_summary_errors(
                auto,
                source="auto",
                prompt_token_count=prompt_token_count,
            )
        )
    elif auto is not None:
        bad_fields.extend(
            _prefill_prompt_chunk_plan_summary_errors(
                auto,
                source="auto",
                prompt_token_count=prompt_token_count,
            )
        )
    if chunk_tokens is not None:
        bad_fields.extend(
            _prefill_prompt_chunk_token_errors(
                chunk_tokens,
                source="prefill_prompt_chunk_tokens",
                plan=plan,
            )
        )
    return tuple(bad_fields)


def _optional_positive_int_field_errors(
    payload: dict[str, object],
    field: str,
) -> tuple[str, ...]:
    value = payload.get(field)
    if value is None:
        return ()
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return (f"{field}={value!r}",)
    return ()


def _prefill_cache_io_errors(
    cache_io: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_CACHE_IO_AUDIT_FIELDS
        if field not in cache_io
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    positive_fields = ("dtype_bytes", "mla_cache_width")
    nonnegative_fields = (
        "indexed_attention_layers",
        "full_attention_layers",
        "dsa_full_indexer_layers",
        "causal_rows_per_layer",
        "indexed_rows_per_layer",
        "mla_cache_read_bytes",
        "dsa_index_cache_read_bytes",
        "total_cache_read_bytes",
        "mla_cache_write_bytes",
        "dsa_index_cache_write_bytes",
        "total_cache_write_bytes",
    )
    int_values: dict[str, int] = {}
    for field in positive_fields:
        value = cache_io.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    for field in nonnegative_fields:
        value = cache_io.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    bad_fields.extend(
        _optional_positive_int_field_errors(cache_io, "dsa_index_head_dim")
    )
    bad_fields.extend(
        _optional_positive_int_field_errors(cache_io, "dsa_index_topk")
    )
    required_sum_fields = {
        "mla_cache_read_bytes",
        "dsa_index_cache_read_bytes",
        "total_cache_read_bytes",
        "mla_cache_write_bytes",
        "dsa_index_cache_write_bytes",
        "total_cache_write_bytes",
    }
    if required_sum_fields.issubset(int_values):
        expected_read = (
            int_values["mla_cache_read_bytes"]
            + int_values["dsa_index_cache_read_bytes"]
        )
        if int_values["total_cache_read_bytes"] != expected_read:
            bad_fields.append(
                "total_cache_read_bytes="
                f"{int_values['total_cache_read_bytes']!r} "
                f"expected={expected_read!r}"
            )
        expected_write = (
            int_values["mla_cache_write_bytes"]
            + int_values["dsa_index_cache_write_bytes"]
        )
        if int_values["total_cache_write_bytes"] != expected_write:
            bad_fields.append(
                "total_cache_write_bytes="
                f"{int_values['total_cache_write_bytes']!r} "
                f"expected={expected_write!r}"
            )
    dsa_index_head_dim = cache_io.get("dsa_index_head_dim")
    dsa_index_topk = cache_io.get("dsa_index_topk")
    dsa_full_layers = int_values.get("dsa_full_indexer_layers")
    indexed_layers = int_values.get("indexed_attention_layers")
    if dsa_index_topk is None:
        indexed_rows = int_values.get("indexed_rows_per_layer")
        dsa_index_read = int_values.get("dsa_index_cache_read_bytes")
        if indexed_rows not in (None, 0):
            bad_fields.append(
                f"indexed_rows_per_layer={indexed_rows!r} without dsa_index_topk"
            )
        if dsa_index_read not in (None, 0):
            bad_fields.append(
                f"dsa_index_cache_read_bytes={dsa_index_read!r} "
                "without dsa_index_topk"
            )
    if dsa_index_head_dim is None:
        dsa_index_write = int_values.get("dsa_index_cache_write_bytes")
        if dsa_index_write not in (None, 0):
            bad_fields.append(
                f"dsa_index_cache_write_bytes={dsa_index_write!r} "
                "without dsa_index_head_dim"
            )
    if dsa_full_layers == 0:
        for field in (
            "dsa_index_cache_read_bytes",
            "dsa_index_cache_write_bytes",
        ):
            value = int_values.get(field)
            if value not in (None, 0):
                bad_fields.append(f"{field}={value!r} without DSA full layers")
    if indexed_layers == 0:
        value = int_values.get("indexed_rows_per_layer")
        if value not in (None, 0):
            bad_fields.append(
                f"indexed_rows_per_layer={value!r} without indexed layers"
            )
    return tuple(bad_fields)


def _frontier_static_capacity_valid(value: object) -> bool:
    return (
        value is None
        or value == "auto"
        or (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value > 0
        )
    )


def _prefill_routed_chunk_candidate_errors(
    candidate: dict[str, object],
    *,
    prompt_token_count: int,
    baseline_read_bytes: int,
    saturation_chunk_tokens: int | None,
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_ROUTED_CHUNK_CANDIDATE_AUDIT_FIELDS
        if field not in candidate
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    positive_fields = ("prompt_chunk_tokens", "chunks_per_prompt")
    nonnegative_fields = (
        "planned_read_bytes",
        "extra_read_bytes",
        "max_layer_planned_read_bytes",
        "max_stage_plus_compact_bytes",
        "max_chunk_stage_plus_compact_bytes",
        "total_stage_plus_compact_bytes",
        "max_static_capacity_binary_bytes",
        "max_chunk_static_capacity_binary_bytes",
        "total_static_capacity_binary_bytes",
        "max_stage_plus_compact_plus_static_bytes",
        "max_chunk_stage_plus_compact_plus_static_bytes",
        "total_stage_plus_compact_plus_static_bytes",
    )
    int_values: dict[str, int] = {}
    for field in positive_fields:
        value = candidate.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    for field in nonnegative_fields:
        value = candidate.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    prompt_chunk = int_values.get("prompt_chunk_tokens")
    if prompt_chunk is not None and prompt_chunk > prompt_token_count:
        bad_fields.append(
            f"prompt_chunk_tokens={prompt_chunk!r} exceeds prompt_token_count"
        )
    saturates = candidate.get("saturates_all_experts_per_layer")
    if not isinstance(saturates, bool):
        bad_fields.append(f"saturates_all_experts_per_layer={saturates!r}")
    elif prompt_chunk is not None and saturation_chunk_tokens is not None:
        expected = prompt_chunk >= saturation_chunk_tokens
        if saturates != expected:
            bad_fields.append(
                "saturates_all_experts_per_layer="
                f"{saturates!r} expected={expected!r}"
            )
    amplification = candidate.get("read_amplification")
    if not _audit_finite_number(amplification) or float(amplification) < 0:
        bad_fields.append(f"read_amplification={amplification!r}")
    planned_seconds = candidate.get("planned_read_seconds")
    if planned_seconds is not None and (
        not _audit_finite_number(planned_seconds)
        or float(planned_seconds) < 0
    ):
        bad_fields.append(f"planned_read_seconds={planned_seconds!r}")
    planned = int_values.get("planned_read_bytes")
    extra = int_values.get("extra_read_bytes")
    if planned is not None and extra is not None:
        if baseline_read_bytes > 0:
            expected_extra = planned - baseline_read_bytes
            if expected_extra < 0:
                bad_fields.append(
                    f"planned_read_bytes={planned!r} below baseline_read_bytes"
                )
            elif extra != expected_extra:
                bad_fields.append(
                    f"extra_read_bytes={extra!r} expected={expected_extra!r}"
                )
            if _audit_finite_number(amplification):
                expected_amp = planned / baseline_read_bytes
                if not math.isclose(
                    float(amplification),
                    expected_amp,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    bad_fields.append(
                        f"read_amplification={amplification!r} "
                        f"expected={expected_amp!r}"
                    )
        elif planned != 0 or extra != 0:
            bad_fields.append(
                "planned/extra read bytes must be zero when baseline_read_bytes=0"
            )
    for total_field, left_field, right_field in (
        (
            "max_stage_plus_compact_plus_static_bytes",
            "max_stage_plus_compact_bytes",
            "max_static_capacity_binary_bytes",
        ),
        (
            "max_chunk_stage_plus_compact_plus_static_bytes",
            "max_chunk_stage_plus_compact_bytes",
            "max_chunk_static_capacity_binary_bytes",
        ),
        (
            "total_stage_plus_compact_plus_static_bytes",
            "total_stage_plus_compact_bytes",
            "total_static_capacity_binary_bytes",
        ),
    ):
        total = int_values.get(total_field)
        left = int_values.get(left_field)
        right = int_values.get(right_field)
        if total is None or left is None or right is None:
            continue
        expected = left + right
        if total != expected:
            bad_fields.append(f"{total_field}={total!r} expected={expected!r}")
    return tuple(bad_fields)


def _prefill_routed_chunk_frontier_errors(
    frontier: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_ROUTED_CHUNK_FRONTIER_AUDIT_FIELDS
        if field not in frontier
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    if frontier.get("analyzed") is not True:
        bad_fields.append(f"analyzed={frontier.get('analyzed')!r}")
    positive_fields = (
        "prompt_token_count",
        "resolved_prompt_chunk_tokens",
        "top_k",
        "stage_align_bytes",
    )
    nonnegative_fields = ("layers", "baseline_read_bytes")
    int_values: dict[str, int] = {}
    for field in positive_fields:
        value = frontier.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    for field in nonnegative_fields:
        value = frontier.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    for field in ("max_safe_prompt_chunk_tokens", "saturation_chunk_tokens"):
        value = frontier.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            bad_fields.append(f"{field}={value!r}")
            continue
        if isinstance(value, int):
            int_values[field] = value
    if not _frontier_static_capacity_valid(
        frontier.get("static_capacity_per_expert")
    ):
        bad_fields.append(
            "static_capacity_per_expert="
            f"{frontier.get('static_capacity_per_expert')!r}"
        )
    if not isinstance(frontier.get("allow_static_capacity_overflow"), bool):
        bad_fields.append(
            "allow_static_capacity_overflow="
            f"{frontier.get('allow_static_capacity_overflow')!r}"
        )
    prompt_count = int_values.get("prompt_token_count")
    resolved = int_values.get("resolved_prompt_chunk_tokens")
    max_safe = int_values.get("max_safe_prompt_chunk_tokens")
    saturation = int_values.get("saturation_chunk_tokens")
    baseline = int_values.get("baseline_read_bytes")
    if prompt_count is not None:
        for field, value in (
            ("resolved_prompt_chunk_tokens", resolved),
            ("max_safe_prompt_chunk_tokens", max_safe),
        ):
            if value is not None and value > prompt_count:
                bad_fields.append(f"{field}={value!r} exceeds prompt_token_count")
    candidates = frontier.get("candidates")
    if not isinstance(candidates, (list, tuple)) or not candidates:
        bad_fields.append(f"candidates={candidates!r}")
        return tuple(bad_fields)
    if prompt_count is None or baseline is None:
        return tuple(bad_fields)
    seen_chunks: set[int] = set()
    previous_chunk = 0
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            bad_fields.append(f"candidates[{index}]={item!r}")
            continue
        candidate_errors = _prefill_routed_chunk_candidate_errors(
            item,
            prompt_token_count=prompt_count,
            baseline_read_bytes=baseline,
            saturation_chunk_tokens=saturation,
        )
        bad_fields.extend(f"candidates[{index}].{error}" for error in candidate_errors)
        chunk = item.get("prompt_chunk_tokens")
        if not isinstance(chunk, bool) and isinstance(chunk, int):
            if chunk in seen_chunks:
                bad_fields.append(f"candidates[{index}].prompt_chunk_tokens duplicate={chunk!r}")
            if chunk <= previous_chunk:
                bad_fields.append(
                    f"candidates[{index}].prompt_chunk_tokens={chunk!r} not sorted"
                )
            seen_chunks.add(chunk)
            previous_chunk = chunk
    for field, value in (
        ("resolved_prompt_chunk_tokens", resolved),
        ("max_safe_prompt_chunk_tokens", max_safe),
        ("prompt_token_count", prompt_count),
    ):
        if value is not None and value not in seen_chunks:
            bad_fields.append(f"candidates missing {field}={value!r}")
    return tuple(bad_fields)


def _prefill_actual_read_time_errors(
    actual: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _PREFILL_ACTUAL_READ_TIME_AUDIT_FIELDS
        if field not in actual
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    if actual.get("source") != "benchmark_actual_prefill":
        bad_fields.append(f"source={actual.get('source')!r}")
    planned_bytes = actual.get("total_expert_stage_planned_read_bytes")
    if (
        isinstance(planned_bytes, bool)
        or not isinstance(planned_bytes, int)
        or planned_bytes < 0
    ):
        bad_fields.append(
            f"total_expert_stage_planned_read_bytes={planned_bytes!r}"
        )
    planned_seconds = actual.get("total_expert_stage_planned_read_seconds")
    if not _audit_finite_number(planned_seconds) or float(planned_seconds) < 0:
        bad_fields.append(
            f"total_expert_stage_planned_read_seconds={planned_seconds!r}"
        )
    ssd = actual.get("prefill_ssd_read_gib_per_second")
    if not _audit_finite_number(ssd) or float(ssd) <= 0:
        bad_fields.append(f"prefill_ssd_read_gib_per_second={ssd!r}")
    max_seconds = actual.get("prefill_max_routed_read_seconds")
    if not _audit_finite_number(max_seconds) or float(max_seconds) <= 0:
        bad_fields.append(f"prefill_max_routed_read_seconds={max_seconds!r}")
    if actual.get("total_expert_stage_read_seconds_ok") is not True:
        bad_fields.append(
            "total_expert_stage_read_seconds_ok="
            f"{actual.get('total_expert_stage_read_seconds_ok')!r}"
        )
    if actual.get("total_expert_stage_copy_seconds_ok") is not True:
        bad_fields.append(
            "total_expert_stage_copy_seconds_ok="
            f"{actual.get('total_expert_stage_copy_seconds_ok')!r}"
        )
    range_int_fields = (
        "prefill_max_stage_raw_ranges",
        "prefill_max_stage_coalesced_ranges",
        "total_expert_stage_raw_ranges",
        "total_expert_stage_coalesced_ranges",
        "max_expert_stage_raw_ranges",
        "max_expert_stage_coalesced_ranges",
    )
    range_values: dict[str, int] = {}
    for field in range_int_fields:
        value = actual.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        range_values[field] = value
    for field in (
        "total_expert_stage_raw_ranges_ok",
        "total_expert_stage_coalesced_ranges_ok",
    ):
        if field in actual and actual.get(field) is not True:
            bad_fields.append(f"{field}={actual.get(field)!r}")
    raw_cap = range_values.get("prefill_max_stage_raw_ranges")
    raw_max = range_values.get("max_expert_stage_raw_ranges")
    raw_total = range_values.get("total_expert_stage_raw_ranges")
    if raw_cap is not None and raw_max is not None and raw_cap > 0 and raw_max > raw_cap:
        bad_fields.append(
            f"max_expert_stage_raw_ranges={raw_max!r} exceeds cap={raw_cap!r}"
        )
    if raw_total is not None and raw_max is not None and raw_max > raw_total:
        bad_fields.append(
            f"max_expert_stage_raw_ranges={raw_max!r} exceeds total={raw_total!r}"
        )
    coalesced_cap = range_values.get("prefill_max_stage_coalesced_ranges")
    coalesced_max = range_values.get("max_expert_stage_coalesced_ranges")
    coalesced_total = range_values.get("total_expert_stage_coalesced_ranges")
    if (
        coalesced_cap is not None
        and coalesced_max is not None
        and coalesced_cap > 0
        and coalesced_max > coalesced_cap
    ):
        bad_fields.append(
            "max_expert_stage_coalesced_ranges="
            f"{coalesced_max!r} exceeds cap={coalesced_cap!r}"
        )
    if (
        coalesced_total is not None
        and coalesced_max is not None
        and coalesced_max > coalesced_total
    ):
        bad_fields.append(
            "max_expert_stage_coalesced_ranges="
            f"{coalesced_max!r} exceeds total={coalesced_total!r}"
        )
    copy_elapsed_field = "total_expert_stage_copy_elapsed_seconds"
    copy_throughput_field = "total_expert_stage_copy_throughput_gib_per_second"
    copy_elapsed = actual.get(copy_elapsed_field)
    if (
        copy_elapsed_field in actual
        and (
            not _audit_finite_number(copy_elapsed)
            or float(copy_elapsed) < 0
        )
    ):
        bad_fields.append(
            f"{copy_elapsed_field}={copy_elapsed!r}"
        )
    copy_throughput = actual.get(copy_throughput_field)
    if copy_throughput_field in actual:
        if copy_elapsed_field not in actual:
            bad_fields.append(
                f"{copy_throughput_field}=present without {copy_elapsed_field}"
            )
        if (
            not _audit_finite_number(copy_throughput)
            or float(copy_throughput) <= 0
        ):
            bad_fields.append(
                f"{copy_throughput_field}={copy_throughput!r}"
            )
    if _audit_finite_number(planned_seconds) and _audit_finite_number(max_seconds):
        tolerance = max(abs(float(max_seconds)) * 1e-6, 1e-12)
        if float(planned_seconds) > float(max_seconds) + tolerance:
            bad_fields.append(
                "total_expert_stage_planned_read_seconds="
                f"{planned_seconds!r} exceeds cap={max_seconds!r}"
            )
    return tuple(bad_fields)


def _require_passing_prefill_actual_read_time_evidence(
    actual: dict[str, object],
    *,
    source: str,
) -> None:
    errors = _prefill_actual_read_time_errors(actual)
    if errors:
        raise CliArgumentError(
            f"{source} is not passing: " + ", ".join(errors)
        )


def _prefill_actual_linear_backend_errors(
    actual: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _PREFILL_ACTUAL_LINEAR_BACKEND_AUDIT_FIELDS
        if field not in actual
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    if actual.get("source") != "benchmark_actual_prefill":
        bad_fields.append(f"source={actual.get('source')!r}")
    configured = actual.get("configured_backend")
    if not isinstance(configured, str) or not configured:
        bad_fields.append(f"configured_backend={configured!r}")
    auto_policy = actual.get("auto_policy")
    if not isinstance(auto_policy, dict):
        bad_fields.append(f"auto_policy={auto_policy!r}")
    else:
        for field in ("mpsgraph_min_batch_tokens", "mpsgraph_min_matrix_dim"):
            value = auto_policy.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                bad_fields.append(f"auto_policy.{field}={value!r}")
    bad_fields.extend(
        _audit_nonnegative_int_mapping_errors(
            actual,
            "linear_backend_counts",
            require_nonempty=True,
        )
    )
    bad_fields.extend(
        _audit_nonnegative_int_mapping_errors(
            actual,
            "linear_backend_flops",
            require_nonempty=True,
        )
    )
    bad_fields.extend(
        _audit_nonnegative_float_mapping_errors(
            actual,
            "linear_backend_elapsed_seconds",
            require_nonempty=True,
        )
    )
    bad_fields.extend(
        _audit_nonnegative_float_mapping_errors(
            actual,
            "linear_backend_estimated_tflops",
            require_nonempty=False,
        )
    )
    elapsed = actual.get("linear_backend_elapsed_seconds")
    tflops = actual.get("linear_backend_estimated_tflops")
    if isinstance(elapsed, dict) and isinstance(tflops, dict):
        elapsed_keys = {key for key in elapsed if isinstance(key, str)}
        for key in tflops:
            if isinstance(key, str) and key in elapsed_keys:
                continue
            bad_fields.append(
                "linear_backend_estimated_tflops"
                f"[{key!r}]=present without elapsed seconds"
            )
    for field in (
        "total_linear_estimated_flops",
        "accelerated_linear_estimated_flops",
        "custom_linear_estimated_flops",
        "unsupported_linear_estimated_flops",
    ):
        value = actual.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
    fraction = actual.get("accelerated_linear_flop_fraction")
    if (
        not _audit_finite_number(fraction)
        or float(fraction) < 0
        or float(fraction) > 1
    ):
        bad_fields.append(f"accelerated_linear_flop_fraction={fraction!r}")
    flops = _audit_nonnegative_int_mapping_values(actual, "linear_backend_flops")
    if flops is not None:
        total_flops = sum(flops.values())
        accelerated_flops = sum(
            flops.get(backend, 0) for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        )
        custom_flops = flops.get("custom-metal", 0)
        unsupported_flops = flops.get("unsupported-mpsgraph", 0)
        expected_values = {
            "total_linear_estimated_flops": total_flops,
            "accelerated_linear_estimated_flops": accelerated_flops,
            "custom_linear_estimated_flops": custom_flops,
            "unsupported_linear_estimated_flops": unsupported_flops,
        }
        for field, expected in expected_values.items():
            value = actual.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value != expected:
                bad_fields.append(f"{field}={value!r} expected={expected!r}")
        if _audit_finite_number(fraction):
            expected_fraction = (
                accelerated_flops / total_flops if total_flops > 0 else 0.0
            )
            if not math.isclose(
                float(fraction),
                expected_fraction,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                bad_fields.append(
                    "accelerated_linear_flop_fraction="
                    f"{fraction!r} expected={expected_fraction!r}"
                )
    return tuple(bad_fields)


def _prefill_actual_linear_backend_coverage_errors(
    *,
    actual_linear: dict[str, object],
    actual_coverage: dict[str, object],
) -> tuple[str, ...]:
    counts = _audit_nonnegative_int_mapping_values(
        actual_linear,
        "linear_backend_counts",
    )
    flops = _audit_nonnegative_int_mapping_values(
        actual_linear,
        "linear_backend_flops",
    )
    if counts is None or flops is None:
        return ()
    total_count = sum(counts.values())
    accelerated_count = sum(
        counts.get(backend, 0) for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
    )
    expected_counts = {
        "matrix_count": total_count,
        "accelerated_matrix_count": accelerated_count,
        "mpsgraph_matrix_count": counts.get("mpsgraph-f32", 0),
        "custom_metal_matrix_count": counts.get("custom-metal", 0),
        "unsupported_mpsgraph_matrix_count": counts.get(
            "unsupported-mpsgraph",
            0,
        ),
    }
    total_flops = actual_linear.get("total_linear_estimated_flops")
    accelerated_flops = actual_linear.get("accelerated_linear_estimated_flops")
    custom_flops = actual_linear.get("custom_linear_estimated_flops")
    unsupported_flops = actual_linear.get("unsupported_linear_estimated_flops")
    expected_flops = {
        "total_estimated_flops": total_flops,
        "accelerated_estimated_flops": accelerated_flops,
        "custom_metal_estimated_flops": custom_flops,
        "unsupported_mpsgraph_estimated_flops": unsupported_flops,
    }
    errors: list[str] = []
    for field, expected in expected_counts.items():
        value = actual_coverage.get(field)
        if value == expected:
            continue
        errors.append(
            f"{field}={value!r} expected_from_actual_linear={expected!r}"
        )
    for field, expected in expected_flops.items():
        value = actual_coverage.get(field)
        if value == expected:
            continue
        errors.append(
            f"{field}={value!r} expected_from_actual_linear={expected!r}"
        )
    fraction = actual_coverage.get("accelerated_flop_fraction")
    expected_fraction = actual_linear.get("accelerated_linear_flop_fraction")
    if _audit_finite_number(fraction) and _audit_finite_number(expected_fraction):
        if not math.isclose(
            float(fraction),
            float(expected_fraction),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            errors.append(
                "accelerated_flop_fraction="
                f"{fraction!r} expected_from_actual_linear={expected_fraction!r}"
            )
    expected_backends = tuple(
        backend
        for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        if counts.get(backend, 0) > 0
    )
    backends = actual_coverage.get("accelerated_backends")
    if isinstance(backends, (list, tuple)) and tuple(backends) != expected_backends:
        errors.append(
            "accelerated_backends="
            f"{tuple(backends)!r} expected_from_actual_linear={expected_backends!r}"
        )
    return tuple(errors)


def _require_valid_prefill_actual_linear_backend_evidence(
    actual: dict[str, object],
    *,
    source: str,
) -> None:
    errors = _prefill_actual_linear_backend_errors(actual)
    if errors:
        raise CliArgumentError(f"{source} is malformed: " + ", ".join(errors))


def _request_prefill_acceleration_coverage_has_audit_fields(
    coverage: dict[str, object],
) -> bool:
    return all(
        field in coverage
        for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS
    )


def _request_prefill_acceleration_coverage_errors(
    coverage: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field
        for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS
        if field not in coverage
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    bool_fields = (
        "required",
        "analyzed",
        "ok",
        "dominant_resident_flops_accelerated",
        "any_resident_matrix_accelerated",
        "all_resident_matrices_accelerated",
    )
    for field in bool_fields:
        if field in coverage and not isinstance(coverage.get(field), bool):
            bad_fields.append(f"{field}={coverage.get(field)!r}")
    if (
        "accelerated_router_gate_only" in coverage
        and not isinstance(coverage.get("accelerated_router_gate_only"), bool)
    ):
        bad_fields.append(
            f"accelerated_router_gate_only={coverage.get('accelerated_router_gate_only')!r}"
        )
    if (
        "allow_router_gate_only_acceleration" in coverage
        and not isinstance(coverage.get("allow_router_gate_only_acceleration"), bool)
    ):
        bad_fields.append(
            "allow_router_gate_only_acceleration="
            f"{coverage.get('allow_router_gate_only_acceleration')!r}"
        )
    int_fields = (
        "matrix_count",
        "accelerated_matrix_count",
        "mpsgraph_matrix_count",
        "custom_metal_matrix_count",
        "unsupported_mpsgraph_matrix_count",
        "mpp_tensor_ops_candidate_matrix_count",
        "streamed_routed_expert_layer_count",
        "streamed_routed_expert_matrix_count",
        "streamed_routed_expert_assignments",
        "streamed_routed_expert_mpp_candidate_matrix_count",
        "total_estimated_flops",
        "accelerated_estimated_flops",
        "custom_metal_estimated_flops",
        "unsupported_mpsgraph_estimated_flops",
        "mpp_tensor_ops_candidate_estimated_flops",
        "streamed_routed_expert_estimated_flops",
        "streamed_routed_expert_mpp_candidate_estimated_flops",
        "other_estimated_flops",
    )
    int_values: dict[str, int] = {}
    for field in int_fields:
        value = coverage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    optional_int_fields = (
        "router_gate_matrix_count",
        "router_gate_estimated_flops",
        "router_gate_accelerated_matrix_count",
        "router_gate_accelerated_estimated_flops",
        "non_router_accelerated_matrix_count",
        "non_router_accelerated_estimated_flops",
    )
    optional_int_values: dict[str, int] = {}
    for field in optional_int_fields:
        if field not in coverage:
            continue
        value = coverage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        optional_int_values[field] = value
    for field in (
        "min_accelerated_flop_fraction",
        "accelerated_flop_fraction",
        "mpp_tensor_ops_candidate_flop_fraction",
    ):
        value = coverage.get(field)
        if (
            not _audit_finite_number(value)
            or float(value) < 0
                or float(value) > 1
        ):
            bad_fields.append(f"{field}={value!r}")
    if "accelerated_router_gate_flop_share" in coverage:
        value = coverage.get("accelerated_router_gate_flop_share")
        if (
            not _audit_finite_number(value)
            or float(value) < 0
            or float(value) > 1
        ):
            bad_fields.append(f"accelerated_router_gate_flop_share={value!r}")
    backends = coverage.get("accelerated_backends")
    if not isinstance(backends, (list, tuple)):
        bad_fields.append(f"accelerated_backends={backends!r}")
    else:
        seen_backends: set[str] = set()
        for item in backends:
            if not isinstance(item, str) or not item:
                bad_fields.append(f"accelerated_backends[]={item!r}")
                continue
            if item in seen_backends:
                bad_fields.append(f"accelerated_backends duplicate={item!r}")
            seen_backends.add(item)
    reason = coverage.get("reason")
    if not isinstance(reason, str):
        bad_fields.append(f"reason={reason!r}")
    mpp_policy = coverage.get("mpp_candidate_policy")
    if not isinstance(mpp_policy, dict):
        bad_fields.append(f"mpp_candidate_policy={mpp_policy!r}")
    else:
        if mpp_policy.get("candidate_backend") != "mpp_tensor_ops_prefill":
            bad_fields.append(
                "mpp_candidate_policy.candidate_backend="
                f"{mpp_policy.get('candidate_backend')!r}"
            )
        if mpp_policy.get("execution_path") != "mpp_tensor_ops_gpu_neural_accelerator":
            bad_fields.append(
                "mpp_candidate_policy.execution_path="
                f"{mpp_policy.get('execution_path')!r}"
            )
        for field in (
            "mpp_tensor_ops_min_batch_tokens",
            "mpp_tensor_ops_min_matrix_dim",
        ):
            value = mpp_policy.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                bad_fields.append(f"mpp_candidate_policy.{field}={value!r}")
        selectable = mpp_policy.get("selectable_prefill_backend")
        if not isinstance(selectable, bool):
            bad_fields.append(
                "mpp_candidate_policy.selectable_prefill_backend="
                f"{selectable!r}"
            )
    if len(int_values) == len(int_fields):
        matrix_count = int_values["matrix_count"]
        accelerated_count = int_values["accelerated_matrix_count"]
        mpsgraph_count = int_values["mpsgraph_matrix_count"]
        custom_count = int_values["custom_metal_matrix_count"]
        unsupported_count = int_values["unsupported_mpsgraph_matrix_count"]
        mpp_candidate_count = int_values["mpp_tensor_ops_candidate_matrix_count"]
        total_flops = int_values["total_estimated_flops"]
        accelerated_flops = int_values["accelerated_estimated_flops"]
        custom_flops = int_values["custom_metal_estimated_flops"]
        unsupported_flops = int_values["unsupported_mpsgraph_estimated_flops"]
        mpp_candidate_flops = int_values[
            "mpp_tensor_ops_candidate_estimated_flops"
        ]
        other_flops = int_values["other_estimated_flops"]
        if accelerated_count > matrix_count:
            bad_fields.append(
                "accelerated_matrix_count exceeds matrix_count: "
                f"{accelerated_count}>{matrix_count}"
            )
        if mpsgraph_count > accelerated_count:
            bad_fields.append(
                "mpsgraph_matrix_count exceeds accelerated_matrix_count: "
                f"{mpsgraph_count}>{accelerated_count}"
            )
        if custom_count + unsupported_count + mpsgraph_count > matrix_count:
            bad_fields.append(
                "matrix backend counts exceed matrix_count: "
                f"{mpsgraph_count + custom_count + unsupported_count}>{matrix_count}"
            )
        if mpp_candidate_count > matrix_count:
            bad_fields.append(
                "mpp_tensor_ops_candidate_matrix_count exceeds matrix_count: "
                f"{mpp_candidate_count}>{matrix_count}"
            )
        if accelerated_flops > total_flops:
            bad_fields.append(
                "accelerated_estimated_flops exceeds total_estimated_flops: "
                f"{accelerated_flops}>{total_flops}"
            )
        if mpp_candidate_flops > total_flops:
            bad_fields.append(
                "mpp_tensor_ops_candidate_estimated_flops exceeds "
                "total_estimated_flops: "
                f"{mpp_candidate_flops}>{total_flops}"
            )
        if accelerated_flops + custom_flops + unsupported_flops + other_flops != total_flops:
            bad_fields.append(
                "estimated FLOP buckets do not sum to total_estimated_flops: "
                f"{accelerated_flops}+{custom_flops}+{unsupported_flops}"
                f"+{other_flops}!={total_flops}"
            )
        router_gate_count = optional_int_values.get("router_gate_matrix_count")
        router_gate_flops = optional_int_values.get("router_gate_estimated_flops")
        router_gate_accelerated_count = optional_int_values.get(
            "router_gate_accelerated_matrix_count"
        )
        router_gate_accelerated_flops = optional_int_values.get(
            "router_gate_accelerated_estimated_flops"
        )
        non_router_accelerated_count = optional_int_values.get(
            "non_router_accelerated_matrix_count"
        )
        non_router_accelerated_flops = optional_int_values.get(
            "non_router_accelerated_estimated_flops"
        )
        if router_gate_count is not None and router_gate_count > matrix_count:
            bad_fields.append(
                "router_gate_matrix_count exceeds matrix_count: "
                f"{router_gate_count}>{matrix_count}"
            )
        if router_gate_flops is not None and router_gate_flops > total_flops:
            bad_fields.append(
                "router_gate_estimated_flops exceeds total_estimated_flops: "
                f"{router_gate_flops}>{total_flops}"
            )
        if (
            router_gate_accelerated_count is not None
            and router_gate_accelerated_count > accelerated_count
        ):
            bad_fields.append(
                "router_gate_accelerated_matrix_count exceeds "
                "accelerated_matrix_count: "
                f"{router_gate_accelerated_count}>{accelerated_count}"
            )
        if (
            router_gate_count is not None
            and router_gate_accelerated_count is not None
            and router_gate_accelerated_count > router_gate_count
        ):
            bad_fields.append(
                "router_gate_accelerated_matrix_count exceeds "
                "router_gate_matrix_count: "
                f"{router_gate_accelerated_count}>{router_gate_count}"
            )
        if (
            router_gate_accelerated_flops is not None
            and router_gate_accelerated_flops > accelerated_flops
        ):
            bad_fields.append(
                "router_gate_accelerated_estimated_flops exceeds "
                "accelerated_estimated_flops: "
                f"{router_gate_accelerated_flops}>{accelerated_flops}"
            )
        if (
            router_gate_flops is not None
            and router_gate_accelerated_flops is not None
            and router_gate_accelerated_flops > router_gate_flops
        ):
            bad_fields.append(
                "router_gate_accelerated_estimated_flops exceeds "
                "router_gate_estimated_flops: "
                f"{router_gate_accelerated_flops}>{router_gate_flops}"
            )
        if non_router_accelerated_count is not None:
            expected_non_router = max(
                0,
                accelerated_count - (router_gate_accelerated_count or 0),
            )
            if non_router_accelerated_count != expected_non_router:
                bad_fields.append(
                    "non_router_accelerated_matrix_count="
                    f"{non_router_accelerated_count!r} "
                    f"expected={expected_non_router!r}"
                )
        if non_router_accelerated_flops is not None:
            expected_non_router_flops = max(
                0,
                accelerated_flops - (router_gate_accelerated_flops or 0),
            )
            if non_router_accelerated_flops != expected_non_router_flops:
                bad_fields.append(
                    "non_router_accelerated_estimated_flops="
                    f"{non_router_accelerated_flops!r} "
                    f"expected={expected_non_router_flops!r}"
                )
        router_gate_share = coverage.get("accelerated_router_gate_flop_share")
        if (
            router_gate_accelerated_flops is not None
            and _audit_finite_number(router_gate_share)
        ):
            expected_share = (
                router_gate_accelerated_flops / accelerated_flops
                if accelerated_flops > 0
                else 0.0
            )
            if not math.isclose(
                float(router_gate_share),
                expected_share,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                bad_fields.append(
                    "accelerated_router_gate_flop_share="
                    f"{router_gate_share!r} expected={expected_share!r}"
                )
        router_gate_only = coverage.get("accelerated_router_gate_only")
        if (
            isinstance(router_gate_only, bool)
            and router_gate_accelerated_count is not None
            and router_gate_accelerated_flops is not None
        ):
            expected_only = (
                accelerated_count > 0
                and router_gate_accelerated_count == accelerated_count
                and router_gate_accelerated_flops == accelerated_flops
            )
            if router_gate_only != expected_only:
                bad_fields.append(
                    "accelerated_router_gate_only="
                    f"{router_gate_only!r} expected={expected_only!r}"
                )
        fraction = coverage.get("accelerated_flop_fraction")
        if _audit_finite_number(fraction):
            expected_fraction = (
                accelerated_flops / total_flops if total_flops > 0 else 0.0
            )
            if not math.isclose(
                float(fraction),
                expected_fraction,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                bad_fields.append(
                    "accelerated_flop_fraction="
                    f"{fraction!r} expected={expected_fraction!r}"
                )
        mpp_fraction = coverage.get("mpp_tensor_ops_candidate_flop_fraction")
        if _audit_finite_number(mpp_fraction):
            expected_mpp_fraction = (
                mpp_candidate_flops / total_flops if total_flops > 0 else 0.0
            )
            if not math.isclose(
                float(mpp_fraction),
                expected_mpp_fraction,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                bad_fields.append(
                    "mpp_tensor_ops_candidate_flop_fraction="
                    f"{mpp_fraction!r} expected={expected_mpp_fraction!r}"
                )
        any_accelerated = coverage.get("any_resident_matrix_accelerated")
        if isinstance(any_accelerated, bool) and any_accelerated != (
            accelerated_count > 0
        ):
            bad_fields.append(
                "any_resident_matrix_accelerated="
                f"{any_accelerated!r} expected={accelerated_count > 0!r}"
            )
        dominant = coverage.get("dominant_resident_flops_accelerated")
        expected_dominant = (
            total_flops > 0 and accelerated_flops * 2 >= total_flops
        )
        if isinstance(dominant, bool) and dominant != expected_dominant:
            bad_fields.append(
                "dominant_resident_flops_accelerated="
                f"{dominant!r} expected={expected_dominant!r}"
            )
        all_accelerated = coverage.get("all_resident_matrices_accelerated")
        expected_all = (
            matrix_count > 0
            and accelerated_count == matrix_count
            and custom_count == 0
            and unsupported_count == 0
        )
        if isinstance(all_accelerated, bool) and all_accelerated != expected_all:
            bad_fields.append(
                "all_resident_matrices_accelerated="
                f"{all_accelerated!r} expected={expected_all!r}"
            )
    if coverage.get("ok") is not True and not reason:
        bad_fields.append("reason=''")
    if (
        coverage.get("required") is True
        and coverage.get("ok") is True
        and coverage.get("any_resident_matrix_accelerated") is not True
    ):
        bad_fields.append(
            "ok=True requires any_resident_matrix_accelerated=True "
            "when required=True"
        )
    if (
        coverage.get("required") is True
        and coverage.get("ok") is True
        and int_values.get("accelerated_matrix_count") == 0
    ):
        bad_fields.append(
            "ok=True requires accelerated_matrix_count>0 when required=True"
        )
    if (
        coverage.get("required") is True
        and coverage.get("ok") is True
        and _audit_finite_number(coverage.get("accelerated_flop_fraction"))
        and _audit_finite_number(coverage.get("min_accelerated_flop_fraction"))
        and float(coverage["accelerated_flop_fraction"])
        < float(coverage["min_accelerated_flop_fraction"])
    ):
        bad_fields.append(
            "ok=True below min_accelerated_flop_fraction: "
            f"{coverage.get('accelerated_flop_fraction')!r}<"
            f"{coverage.get('min_accelerated_flop_fraction')!r}"
        )
    return tuple(bad_fields)


def _require_valid_request_prefill_acceleration_coverage_evidence(
    coverage: dict[str, object],
    *,
    source: str,
) -> None:
    errors = _request_prefill_acceleration_coverage_errors(coverage)
    if errors:
        raise CliArgumentError(f"{source} is malformed: " + ", ".join(errors))


def _require_valid_prefill_actual_acceleration_coverage_evidence(
    coverage: dict[str, object],
    *,
    source: str,
) -> None:
    errors = _request_prefill_acceleration_coverage_errors(coverage)
    if errors:
        raise CliArgumentError(f"{source} is malformed: " + ", ".join(errors))


def _decode_actual_read_time_errors(
    actual: dict[str, object],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field for field in _DECODE_ACTUAL_READ_TIME_AUDIT_FIELDS if field not in actual
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    if actual.get("source") != "benchmark_actual_decode":
        bad_fields.append(f"source={actual.get('source')!r}")
    if actual.get("actual_decode_routed_read_bytes_ok") is not True:
        bad_fields.append(
            "actual_decode_routed_read_bytes_ok="
            f"{actual.get('actual_decode_routed_read_bytes_ok')!r}"
        )
    for field in (
        "decode_step_count",
        "decode_read_bytes_per_token",
    ):
        value = actual.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
    for field in (
        "planned_decode_routed_read_bytes",
        "actual_decode_routed_read_bytes",
    ):
        value = actual.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
    for field in (
        "planned_decode_routed_read_seconds",
        "actual_decode_routed_read_seconds",
    ):
        value = actual.get(field)
        if not _audit_finite_number(value) or float(value) < 0:
            bad_fields.append(f"{field}={value!r}")
    for field in (
        "prefill_ssd_read_gib_per_second",
        "decode_max_routed_read_seconds_per_token",
        "total_decode_max_routed_read_seconds",
    ):
        value = actual.get(field)
        if not _audit_finite_number(value) or float(value) <= 0:
            bad_fields.append(f"{field}={value!r}")
    if actual.get("total_decode_routed_read_seconds_ok") is not True:
        bad_fields.append(
            "total_decode_routed_read_seconds_ok="
            f"{actual.get('total_decode_routed_read_seconds_ok')!r}"
        )
    actual_seconds = actual.get("actual_decode_routed_read_seconds")
    max_seconds = actual.get("total_decode_max_routed_read_seconds")
    actual_bytes = actual.get("actual_decode_routed_read_bytes")
    planned_bytes = actual.get("planned_decode_routed_read_bytes")
    if (
        isinstance(actual_bytes, int)
        and not isinstance(actual_bytes, bool)
        and isinstance(planned_bytes, int)
        and not isinstance(planned_bytes, bool)
        and actual_bytes > planned_bytes
    ):
        bad_fields.append(
            "actual_decode_routed_read_bytes="
            f"{actual_bytes!r} exceeds planned={planned_bytes!r}"
        )
    if _audit_finite_number(actual_seconds) and _audit_finite_number(max_seconds):
        tolerance = max(abs(float(max_seconds)) * 1e-6, 1e-12)
        if float(actual_seconds) > float(max_seconds) + tolerance:
            bad_fields.append(
                "actual_decode_routed_read_seconds="
                f"{actual_seconds!r} exceeds cap={max_seconds!r}"
            )
    return tuple(bad_fields)


def _require_passing_decode_actual_read_time_evidence(
    actual: dict[str, object],
    *,
    source: str,
) -> None:
    errors = _decode_actual_read_time_errors(actual)
    if errors:
        raise CliArgumentError(
            f"{source} is not passing: " + ", ".join(errors)
        )


def _require_passing_routed_read_evidence(
    routed_read: dict[str, object],
    *,
    source: str,
    required_fields: tuple[str, ...],
) -> None:
    missing_fields = tuple(
        field for field in required_fields if field not in routed_read
    )
    if missing_fields:
        raise CliArgumentError(
            f"{source} is missing required fields: "
            + ", ".join(missing_fields)
        )
    bad_fields: list[str] = []
    for field in (
        "analyzed",
        "within_limit",
        "within_amplification_limit",
        "within_planned_read_limit",
        "within_seconds_limit",
    ):
        if routed_read.get(field) is not True:
            bad_fields.append(f"{field}={routed_read.get(field)!r}")
    for field in (
        "baseline_read_bytes",
        "planned_read_bytes",
        "extra_read_bytes",
        "max_planned_read_bytes",
    ):
        value = routed_read.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
    for field in (
        "read_amplification",
        "max_read_amplification",
        "planned_read_seconds",
    ):
        value = routed_read.get(field)
        if not _audit_finite_number(value) or float(value) < 0:
            bad_fields.append(f"{field}={value!r}")
    for field in ("ssd_read_gib_per_second", "max_read_seconds"):
        value = routed_read.get(field)
        if not _audit_finite_number(value) or float(value) <= 0:
            bad_fields.append(f"{field}={value!r}")
    if "prompt_chunk_tokens" in required_fields:
        prompt_chunk = routed_read.get("prompt_chunk_tokens")
        if (
            isinstance(prompt_chunk, bool)
            or not isinstance(prompt_chunk, int)
            or prompt_chunk <= 0
        ):
            bad_fields.append(f"prompt_chunk_tokens={prompt_chunk!r}")
    if "top_k" in required_fields:
        top_k = routed_read.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            bad_fields.append(f"top_k={top_k!r}")
    minimum_chunk = routed_read.get("minimum_chunk_tokens_for_limits")
    if minimum_chunk is not None and (
        isinstance(minimum_chunk, bool)
        or not isinstance(minimum_chunk, int)
        or minimum_chunk <= 0
    ):
        bad_fields.append(f"minimum_chunk_tokens_for_limits={minimum_chunk!r}")
    if bad_fields:
        raise CliArgumentError(
            f"{source} is not passing: " + ", ".join(bad_fields)
        )


def _require_launch_audit_request_routed_read_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    audited_prompt, _ = _launch_audit_request_limits(artifact)
    if audited_prompt <= 1:
        return
    request = artifact.get("request_check")
    if not isinstance(request, dict):
        raise CliArgumentError("launch audit is missing a passing request check")
    if request.get("batch_prefill_prompt") is not True:
        raise CliArgumentError(
            "launch audit request is missing batch_prefill_prompt=true for "
            "multi-token prompt"
        )
    routed_read = request.get("prefill_routed_expert_read")
    if not isinstance(routed_read, dict):
        raise CliArgumentError(
            "launch audit request is missing prefill_routed_expert_read evidence"
        )
    _require_passing_routed_read_evidence(
        routed_read,
        source="launch audit request routed-read evidence",
        required_fields=_REQUEST_ROUTED_READ_AUDIT_FIELDS,
    )
    budget_check = _launch_audit_check_by_code(
        audit,
        "request_prefill_routed_read_budget_ok",
    )
    if not isinstance(budget_check, dict):
        return
    if budget_check.get("required") is not True:
        raise CliArgumentError(
            "launch audit routed-read budget check is inconsistent: "
            f"required={budget_check.get('required')!r}"
        )
    _require_passing_routed_read_evidence(
        budget_check,
        source="launch audit routed-read budget check",
        required_fields=_REQUEST_ROUTED_READ_LAUNCH_CHECK_FIELDS,
    )
    mismatches = tuple(
        field
        for field in _REQUEST_ROUTED_READ_LAUNCH_CHECK_FIELDS
        if budget_check.get(field) != routed_read.get(field)
    )
    if mismatches:
        field = mismatches[0]
        raise CliArgumentError(
            "launch audit routed-read budget check does not match request "
            "evidence: "
            f"{field} audit={budget_check.get(field)!r} "
            f"request={routed_read.get(field)!r}"
        )


def _require_passing_decode_routed_read_evidence(
    decode_read: dict[str, object],
    *,
    source: str,
) -> None:
    missing_fields = tuple(
        field
        for field in _REQUEST_DECODE_ROUTED_READ_AUDIT_FIELDS
        if field not in decode_read
    )
    if missing_fields:
        raise CliArgumentError(
            f"{source} is missing required fields: "
            + ", ".join(missing_fields)
        )
    bad_fields: list[str] = []
    for field in (
        "analyzed",
        "within_read_limit",
        "within_seconds_limit",
        "within_limit",
    ):
        if decode_read.get(field) is not True:
            bad_fields.append(f"{field}={decode_read.get(field)!r}")
    read_bytes = decode_read.get("read_bytes_per_token")
    if isinstance(read_bytes, bool) or not isinstance(read_bytes, int) or read_bytes < 0:
        bad_fields.append(f"read_bytes_per_token={read_bytes!r}")
    max_bytes = decode_read.get("max_read_bytes_per_token")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        bad_fields.append(f"max_read_bytes_per_token={max_bytes!r}")
    for field in (
        "ssd_read_gib_per_second",
        "planned_read_seconds_per_token",
        "max_read_seconds_per_token",
    ):
        value = decode_read.get(field)
        if not _audit_finite_number(value) or float(value) <= 0:
            bad_fields.append(f"{field}={value!r}")
    if bad_fields:
        raise CliArgumentError(
            f"{source} is not passing: " + ", ".join(bad_fields)
        )


def _require_launch_audit_request_decode_routed_read_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    _, audited_new = _launch_audit_request_limits(artifact)
    if audited_new <= 0:
        return
    request = artifact.get("request_check")
    if not isinstance(request, dict):
        raise CliArgumentError("launch audit is missing a passing request check")
    decode_read = request.get("decode_routed_expert_read")
    if not isinstance(decode_read, dict):
        raise CliArgumentError(
            "launch audit request is missing decode_routed_expert_read evidence"
        )
    _require_passing_decode_routed_read_evidence(
        decode_read,
        source="launch audit request decode routed-read evidence",
    )
    budget_check = _launch_audit_check_by_code(
        audit,
        "request_decode_routed_read_budget_ok",
    )
    if not isinstance(budget_check, dict):
        return
    if budget_check.get("required") is not True:
        raise CliArgumentError(
            "launch audit decode routed-read budget check is inconsistent: "
            f"required={budget_check.get('required')!r}"
        )
    _require_passing_decode_routed_read_evidence(
        budget_check,
        source="launch audit decode routed-read budget check",
    )
    mismatches = tuple(
        field
        for field in _REQUEST_DECODE_ROUTED_READ_AUDIT_FIELDS
        if budget_check.get(field) != decode_read.get(field)
    )
    if mismatches:
        field = mismatches[0]
        raise CliArgumentError(
            "launch audit decode routed-read budget check does not match request "
            "evidence: "
            f"{field} audit={budget_check.get(field)!r} "
            f"request={decode_read.get(field)!r}"
        )


def _require_launch_audit_runtime_preflight_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    request = artifact.get("request_check")
    if not isinstance(request, dict) or request.get("ok") is not True:
        raise CliArgumentError("launch audit is missing a passing request check")
    runtime = request.get("runtime_preflight")
    if not isinstance(runtime, dict):
        raise CliArgumentError(
            "launch audit request is missing runtime_preflight evidence"
        )
    missing_fields = tuple(
        field for field in _REQUEST_RUNTIME_PREFLIGHT_AUDIT_FIELDS if field not in runtime
    )
    if missing_fields:
        raise CliArgumentError(
            "launch audit runtime preflight evidence is missing required fields: "
            + ", ".join(missing_fields)
        )
    bad_fields: list[str] = []
    if runtime.get("ran") is not True:
        bad_fields.append(f"ran={runtime.get('ran')!r}")
    if runtime.get("available_memory_ok") is not True:
        bad_fields.append(
            f"available_memory_ok={runtime.get('available_memory_ok')!r}"
        )
    positive_fields = (
        "requested_context_tokens",
        "live_working_set_bytes",
        "max_live_working_set_bytes",
        "required_available_memory_bytes",
        "system_available_memory_bytes",
        "system_total_memory_bytes",
    )
    nonnegative_fields = (
        "max_layer_peak_bytes",
        "max_layer_cache_read_bytes",
        "read_bytes_per_token",
        "final_logits_peak_bytes",
        "embedding_row_bytes",
        "embedding_output_bytes",
        "resident_backing_bytes",
        "nonresident_peak_bytes",
        "extra_live_working_set_bytes",
        "min_available_memory_bytes",
    )
    for field in positive_fields:
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
    for field in nonnegative_fields:
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
    source = runtime.get("system_memory_source")
    if not isinstance(source, str) or not source:
        bad_fields.append(f"system_memory_source={source!r}")
    prefill_live = runtime.get("prefill_live_memory")
    if prefill_live is not None:
        if not isinstance(prefill_live, dict):
            bad_fields.append(f"prefill_live_memory={prefill_live!r}")
        else:
            bad_fields.extend(
                f"prefill_live_memory.{field}"
                for field in _prefill_live_memory_errors(prefill_live)
            )
    required_available = runtime.get("required_available_memory_bytes")
    system_available = runtime.get("system_available_memory_bytes")
    system_total = runtime.get("system_total_memory_bytes")
    live_working_set = runtime.get("live_working_set_bytes")
    resident_backing = runtime.get("resident_backing_bytes")
    nonresident_peak = runtime.get("nonresident_peak_bytes")
    extra_live_working_set = runtime.get("extra_live_working_set_bytes")
    min_available = runtime.get("min_available_memory_bytes")
    prefill_live_bytes = (
        prefill_live.get("estimated_live_working_set_bytes")
        if isinstance(prefill_live, dict)
        else None
    )
    if (
        isinstance(required_available, int)
        and isinstance(live_working_set, int)
        and isinstance(min_available, int)
        and required_available != live_working_set + min_available
    ):
        bad_fields.append(
            "required_available_memory_bytes="
            f"{required_available!r} does not equal live+reserve"
        )
    if (
        isinstance(system_available, int)
        and isinstance(required_available, int)
        and system_available < required_available
    ):
        bad_fields.append(
            "system_available_memory_bytes="
            f"{system_available!r} below required={required_available!r}"
        )
    if (
        isinstance(system_total, int)
        and isinstance(system_available, int)
        and system_total < system_available
    ):
        bad_fields.append(
            "system_total_memory_bytes="
            f"{system_total!r} below available={system_available!r}"
        )
    if (
        isinstance(live_working_set, int)
        and isinstance(resident_backing, int)
        and isinstance(nonresident_peak, int)
        and live_working_set != resident_backing + nonresident_peak
    ):
        bad_fields.append(
            "live_working_set_bytes="
            f"{live_working_set!r} does not equal resident+nonresident"
        )
    if (
        isinstance(extra_live_working_set, int)
        and isinstance(nonresident_peak, int)
        and nonresident_peak < extra_live_working_set
    ):
        bad_fields.append(
            "nonresident_peak_bytes="
            f"{nonresident_peak!r} below extra_live_working_set_bytes="
            f"{extra_live_working_set!r}"
        )
    if (
        isinstance(prefill_live_bytes, int)
        and isinstance(live_working_set, int)
        and live_working_set < prefill_live_bytes
    ):
        bad_fields.append(
            "live_working_set_bytes="
            f"{live_working_set!r} below prefill_live_memory="
            f"{prefill_live_bytes!r}"
        )
    if bad_fields:
        raise CliArgumentError(
            "launch audit runtime preflight evidence is not passing: "
            + ", ".join(bad_fields)
        )
    check = _launch_audit_check_by_code(audit, "request_runtime_memory_ok")
    if not isinstance(check, dict):
        return
    if check.get("available_memory_ok") is not True:
        raise CliArgumentError(
            "launch audit runtime memory check is not passing: "
            f"available_memory_ok={check.get('available_memory_ok')!r}"
        )
    for field in _REQUEST_RUNTIME_PREFLIGHT_AUDIT_FIELDS:
        if field == "ran":
            continue
        if check.get(field) == runtime.get(field):
            continue
        raise CliArgumentError(
            "launch audit runtime memory check does not match request evidence: "
            f"{field} audit={check.get(field)!r} request={runtime.get(field)!r}"
        )
    if "prefill_live_memory" in check and check.get("prefill_live_memory") != prefill_live:
        raise CliArgumentError(
            "launch audit runtime memory check does not match request evidence: "
            "prefill_live_memory "
            f"audit={check.get('prefill_live_memory')!r} request={prefill_live!r}"
        )


def _request_profile_prefill_prompt_chunk_plan(
    request_profile: object,
) -> object | None:
    sections = (
        request_profile.get("sections")
        if isinstance(request_profile, dict)
        else None
    )
    if not isinstance(sections, dict):
        return None
    return sections.get("prefill_prompt_chunk_plan")


def _launch_profiles_have_same_prefill_prompt_chunk_plan(
    current_profile: object,
    audited_profile: object,
) -> bool:
    audited_plan = _request_profile_prefill_prompt_chunk_plan(audited_profile)
    if audited_plan is None:
        return True
    return _request_profile_prefill_prompt_chunk_plan(current_profile) == audited_plan


def _require_launch_audit_request_prefill_prompt_chunk_plan_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    audited_prompt, _ = _launch_audit_request_limits(artifact)
    request = artifact.get("request_check")
    if not isinstance(request, dict):
        raise CliArgumentError("launch audit is missing a passing request check")
    plan = request.get("prefill_prompt_chunk_plan")
    chunk_tokens = request.get("prefill_prompt_chunk_tokens")
    check = _launch_audit_check_by_code(
        audit,
        "request_prefill_prompt_chunk_plan_ok",
    )
    if audited_prompt <= 1:
        if isinstance(plan, dict):
            errors = _prefill_prompt_chunk_plan_errors(
                plan,
                prompt_token_count=audited_prompt,
                chunk_tokens=chunk_tokens,
            )
            if errors:
                raise CliArgumentError(
                    "launch audit request prefill prompt chunk-plan evidence "
                    "is malformed: "
                    + ", ".join(errors)
                )
        return
    if request.get("batch_prefill_prompt") is not True:
        raise CliArgumentError(
            "launch audit request is missing batch_prefill_prompt=true for "
            "multi-token prompt"
        )
    if not isinstance(plan, dict):
        raise CliArgumentError(
            "launch audit request is missing prefill_prompt_chunk_plan evidence"
        )
    errors = _prefill_prompt_chunk_plan_errors(
        plan,
        prompt_token_count=audited_prompt,
        chunk_tokens=chunk_tokens,
    )
    if errors:
        raise CliArgumentError(
            "launch audit request prefill prompt chunk-plan evidence is "
            "malformed: "
            + ", ".join(errors)
        )
    request_profile = artifact.get("request_launch_profile")
    if not isinstance(request_profile, dict):
        raise CliArgumentError("launch audit is missing request launch profile")
    profile_plan = _request_profile_prefill_prompt_chunk_plan(request_profile)
    if not isinstance(profile_plan, dict):
        raise CliArgumentError(
            "launch audit request profile is missing prefill_prompt_chunk_plan "
            "section"
        )
    if profile_plan != plan:
        raise CliArgumentError(
            "launch audit request profile prefill_prompt_chunk_plan does not "
            "match request evidence"
        )
    drift = request.get("prefill_prompt_chunk_plan_drift")
    drift_status = drift.get("status") if isinstance(drift, dict) else None
    if (
        isinstance(drift_status, str)
        and drift_status in _PREFILL_CHUNK_PLAN_ADMISSION_FAILURES
    ):
        raise CliArgumentError(
            "launch audit request prefill prompt chunk-plan drift is not "
            f"passing: status={drift_status!r}"
        )
    if not isinstance(check, dict):
        return
    if check.get("required") is not True:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan check is inconsistent: "
            f"required={check.get('required')!r}"
        )
    if check.get("ok") is not True:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan check is not passing: "
            f"ok={check.get('ok')!r}"
        )
    if check.get("evidence_present") is not True:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan check is inconsistent: "
            f"evidence_present={check.get('evidence_present')!r}"
        )
    if check.get("request_profile_evidence_present") is not True:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan check is inconsistent: "
            "request_profile_evidence_present="
            f"{check.get('request_profile_evidence_present')!r}"
        )
    if check.get("request_profile_plan_matches") is not True:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan check is inconsistent: "
            f"request_profile_plan_matches={check.get('request_profile_plan_matches')!r}"
        )
    if check.get("prefill_prompt_chunk_plan") != plan:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan check does not match "
            "request evidence"
        )
    if check.get("prefill_prompt_chunk_tokens") != chunk_tokens:
        raise CliArgumentError(
            "launch audit prefill prompt chunk-token check does not match "
            "request evidence"
        )
    if (
        "prefill_prompt_chunk_plan_drift" in check
        and check.get("prefill_prompt_chunk_plan_drift") != drift
    ):
        raise CliArgumentError(
            "launch audit prefill prompt chunk-plan drift check does not "
            "match request evidence"
        )


def _require_current_runtime_preflight_for_launch_audit(
    args: argparse.Namespace,
    artifact: dict[str, object],
) -> None:
    request = artifact.get("request_check")
    runtime = request.get("runtime_preflight") if isinstance(request, dict) else None
    if not isinstance(runtime, dict) or runtime.get("ran") is not True:
        return
    if getattr(args, "preflight_runtime", None) is True:
        return
    raise CliArgumentError(
        "required launch audit needs runtime preflight on this command; "
        "remove --no-runtime-preflight so current memory is checked before generation"
    )


def _stage_temp_evidence_errors(
    stage_temp: dict[str, object],
    *,
    required_fields: tuple[str, ...],
) -> tuple[str, ...]:
    missing_fields = tuple(
        field for field in required_fields if field not in stage_temp
    )
    bad_fields = [f"missing={field}" for field in missing_fields]
    if "analyzed" in required_fields and stage_temp.get("analyzed") is not True:
        bad_fields.append(f"analyzed={stage_temp.get('analyzed')!r}")
    if "within_limit" in required_fields and stage_temp.get("within_limit") is not True:
        bad_fields.append(f"within_limit={stage_temp.get('within_limit')!r}")
    for field in ("max_stage_bytes", "max_compact_stage_bytes"):
        value = stage_temp.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
    for field in ("max_stage_limit_bytes", "max_compact_stage_limit_bytes"):
        value = stage_temp.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
    for field in (
        "within_stage_limit",
        "within_compact_stage_limit",
        "within_stage_raw_range_limit",
        "within_stage_coalesced_range_limit",
    ):
        if field in required_fields and stage_temp.get(field) is not True:
            bad_fields.append(f"{field}={stage_temp.get(field)!r}")
    if "prompt_chunk_tokens" in required_fields:
        prompt_chunk = stage_temp.get("prompt_chunk_tokens")
        if (
            isinstance(prompt_chunk, bool)
            or not isinstance(prompt_chunk, int)
            or prompt_chunk <= 0
        ):
            bad_fields.append(f"prompt_chunk_tokens={prompt_chunk!r}")
    if "top_k" in required_fields:
        top_k = stage_temp.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            bad_fields.append(f"top_k={top_k!r}")
    for field in ("chunks_per_prompt", "layers", "stage_align_bytes"):
        if field not in required_fields:
            continue
        value = stage_temp.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            bad_fields.append(f"{field}={value!r}")
    if (
        "static_capacity_per_expert" in required_fields
        and not _frontier_static_capacity_valid(
            stage_temp.get("static_capacity_per_expert")
        )
    ):
        bad_fields.append(
            "static_capacity_per_expert="
            f"{stage_temp.get('static_capacity_per_expert')!r}"
        )
    if (
        "allow_static_capacity_overflow" in required_fields
        and not isinstance(stage_temp.get("allow_static_capacity_overflow"), bool)
    ):
        bad_fields.append(
            "allow_static_capacity_overflow="
            f"{stage_temp.get('allow_static_capacity_overflow')!r}"
        )
    if (
        "static_capacity_strict_overflow_safe" in required_fields
        and not isinstance(stage_temp.get("static_capacity_strict_overflow_safe"), bool)
    ):
        bad_fields.append(
            "static_capacity_strict_overflow_safe="
            f"{stage_temp.get('static_capacity_strict_overflow_safe')!r}"
        )
    nonnegative_fields = (
        "max_static_capacity_per_expert",
        "max_stage_plus_compact_bytes",
        "total_stage_plus_compact_bytes",
        "max_static_capacity_binary_bytes",
        "total_static_capacity_binary_bytes",
        "max_stage_plus_compact_plus_static_bytes",
        "total_stage_plus_compact_plus_static_bytes",
        "max_stage_raw_ranges",
        "max_stage_raw_range_limit",
        "max_stage_coalesced_ranges",
        "max_stage_coalesced_range_limit",
    )
    int_values: dict[str, int] = {}
    for field in nonnegative_fields:
        if field not in required_fields:
            continue
        value = stage_temp.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
            continue
        int_values[field] = value
    for field in ("max_stage_bytes", "max_compact_stage_bytes"):
        value = stage_temp.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        int_values[field] = value
    for total_field, left_field, right_field in (
        (
            "max_stage_plus_compact_bytes",
            "max_stage_bytes",
            "max_compact_stage_bytes",
        ),
        (
            "max_stage_plus_compact_plus_static_bytes",
            "max_stage_plus_compact_bytes",
            "max_static_capacity_binary_bytes",
        ),
        (
            "total_stage_plus_compact_plus_static_bytes",
            "total_stage_plus_compact_bytes",
            "total_static_capacity_binary_bytes",
        ),
    ):
        total = int_values.get(total_field)
        left = int_values.get(left_field)
        right = int_values.get(right_field)
        if total is None or left is None or right is None:
            continue
        expected = left + right
        if total != expected:
            bad_fields.append(f"{total_field}={total!r} expected={expected!r}")
    return tuple(bad_fields)


def _require_passing_stage_temp_evidence(
    stage_temp: dict[str, object],
    *,
    source: str,
    required_fields: tuple[str, ...],
) -> None:
    bad_fields = _stage_temp_evidence_errors(
        stage_temp,
        required_fields=required_fields,
    )
    if bad_fields:
        raise CliArgumentError(
            f"{source} is not passing: " + ", ".join(bad_fields)
        )


def _require_passing_stage_temp_disk_evidence(
    stage_disk: dict[str, object],
    *,
    source: str,
    required_fields: tuple[str, ...] = _REQUEST_PREFILL_STAGE_TEMP_DISK_AUDIT_FIELDS,
) -> None:
    missing_fields = tuple(
        field
        for field in required_fields
        if field not in stage_disk
    )
    if missing_fields:
        raise CliArgumentError(
            f"{source} is missing required fields: "
            + ", ".join(missing_fields)
        )
    bad_fields: list[str] = []
    if "analyzed" in required_fields and stage_disk.get("analyzed") is not True:
        bad_fields.append(f"analyzed={stage_disk.get('analyzed')!r}")
    if (
        "within_free_space" in required_fields
        and stage_disk.get("within_free_space") is not True
    ):
        bad_fields.append(
            f"within_free_space={stage_disk.get('within_free_space')!r}"
        )
    for field in (
        "required_stage_temp_bytes",
        "disk_safety_margin_bytes",
        "required_free_bytes",
        "free_bytes",
    ):
        if field not in required_fields:
            continue
        value = stage_disk.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_fields.append(f"{field}={value!r}")
    path = stage_disk.get("path")
    if "path" in required_fields and (not isinstance(path, str) or not path):
        bad_fields.append(f"path={path!r}")
    if bad_fields:
        raise CliArgumentError(
            f"{source} is not passing: " + ", ".join(bad_fields)
        )


def _require_launch_audit_request_stage_temp_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    audited_prompt, _ = _launch_audit_request_limits(artifact)
    if audited_prompt <= 1:
        return
    request = artifact.get("request_check")
    if not isinstance(request, dict):
        raise CliArgumentError("launch audit is missing a passing request check")
    stage_temp = request.get("prefill_routed_stage_temp_disk")
    if not isinstance(stage_temp, dict):
        raise CliArgumentError(
            "launch audit request is missing prefill_routed_stage_temp_disk evidence"
        )
    _require_passing_stage_temp_evidence(
        stage_temp,
        source="launch audit request stage-temp evidence",
        required_fields=_REQUEST_PREFILL_STAGE_TEMP_AUDIT_FIELDS,
    )
    limit_check = _launch_audit_check_by_code(
        audit,
        "request_prefill_stage_temp_limit_ok",
    )
    if isinstance(limit_check, dict):
        if limit_check.get("required") is not True:
            raise CliArgumentError(
                "launch audit stage-temp limit check is inconsistent: "
                f"required={limit_check.get('required')!r}"
            )
        _require_passing_stage_temp_evidence(
            limit_check,
            source="launch audit stage-temp limit check",
            required_fields=(
                "within_limit",
                "max_stage_bytes",
                "max_stage_limit_bytes",
                "max_compact_stage_bytes",
                "max_compact_stage_limit_bytes",
                "max_stage_raw_ranges",
                "max_stage_raw_range_limit",
                "max_stage_coalesced_ranges",
                "max_stage_coalesced_range_limit",
                "within_stage_raw_range_limit",
                "within_stage_coalesced_range_limit",
                "max_stage_plus_compact_bytes",
                "total_stage_plus_compact_bytes",
                "max_static_capacity_binary_bytes",
                "total_static_capacity_binary_bytes",
                "max_stage_plus_compact_plus_static_bytes",
                "total_stage_plus_compact_plus_static_bytes",
                "static_capacity_per_expert",
                "allow_static_capacity_overflow",
            ),
        )
        for field in (
            "within_limit",
            "max_stage_bytes",
            "max_stage_limit_bytes",
            "max_compact_stage_bytes",
            "max_compact_stage_limit_bytes",
            "max_stage_raw_ranges",
            "max_stage_raw_range_limit",
            "max_stage_coalesced_ranges",
            "max_stage_coalesced_range_limit",
            "within_stage_raw_range_limit",
            "within_stage_coalesced_range_limit",
            "max_stage_plus_compact_bytes",
            "total_stage_plus_compact_bytes",
            "max_static_capacity_binary_bytes",
            "total_static_capacity_binary_bytes",
            "max_stage_plus_compact_plus_static_bytes",
            "total_stage_plus_compact_plus_static_bytes",
            "static_capacity_per_expert",
            "allow_static_capacity_overflow",
        ):
            if limit_check.get(field) == stage_temp.get(field):
                continue
            raise CliArgumentError(
                "launch audit stage-temp limit check does not match request "
                "evidence: "
                f"{field} audit={limit_check.get(field)!r} "
                f"request={stage_temp.get(field)!r}"
            )
    stage_disk = request.get("prefill_stage_temp_disk_free")
    if not isinstance(stage_disk, dict):
        raise CliArgumentError(
            "launch audit request is missing prefill_stage_temp_disk_free evidence"
        )
    _require_passing_stage_temp_disk_evidence(
        stage_disk,
        source="launch audit request stage-temp disk evidence",
    )
    disk_check = _launch_audit_check_by_code(
        audit,
        "request_prefill_stage_temp_disk_ok",
    )
    if isinstance(disk_check, dict):
        if disk_check.get("required") is not True:
            raise CliArgumentError(
                "launch audit stage-temp disk check is inconsistent: "
                f"required={disk_check.get('required')!r}"
            )
        _require_passing_stage_temp_disk_evidence(
            disk_check,
            source="launch audit stage-temp disk check",
            required_fields=(
                "within_free_space",
                "required_free_bytes",
                "free_bytes",
                "path",
            ),
        )
        for field in ("within_free_space", "required_free_bytes", "free_bytes", "path"):
            if disk_check.get(field) == stage_disk.get(field):
                continue
            raise CliArgumentError(
                "launch audit stage-temp disk check does not match request "
                "evidence: "
                f"{field} audit={disk_check.get(field)!r} "
                f"request={stage_disk.get(field)!r}"
            )


def _require_launch_audit_request_prefill_acceleration_coverage_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    request = artifact.get("request_check")
    coverage = (
        request.get("prefill_acceleration_coverage")
        if isinstance(request, dict)
        else None
    )
    check = _launch_audit_check_by_code(
        audit,
        "request_prefill_acceleration_coverage_ok",
    )
    check_requires_evidence = (
        isinstance(check, dict) and check.get("evidence_present") is True
    )
    coverage_has_audit_fields = (
        isinstance(coverage, dict)
        and _request_prefill_acceleration_coverage_has_audit_fields(coverage)
    )
    if check_requires_evidence and not isinstance(coverage, dict):
        raise CliArgumentError(
            "launch audit is missing request prefill acceleration coverage evidence"
        )
    if coverage_has_audit_fields or check_requires_evidence:
        if not isinstance(coverage, dict):
            raise CliArgumentError(
                "launch audit is missing request prefill acceleration coverage evidence"
            )
        _require_valid_request_prefill_acceleration_coverage_evidence(
            coverage,
            source="launch audit request prefill acceleration coverage evidence",
        )
    if not isinstance(check, dict) or not check_requires_evidence:
        return
    if check.get("ok") is not True:
        raise CliArgumentError(
            "launch audit request prefill acceleration coverage check is not "
            f"passing: ok={check.get('ok')!r}"
        )
    for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS:
        if check.get(field) == coverage.get(field):
            continue
        raise CliArgumentError(
            "launch audit request prefill acceleration coverage check does "
            "not match request evidence: "
            f"{field} audit={check.get(field)!r} "
            f"request={coverage.get(field)!r}"
        )


def _require_launch_audit_request_prefill_cache_io_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    request = artifact.get("request_check")
    cache_io = (
        request.get("prefill_cache_io")
        if isinstance(request, dict)
        else None
    )
    if isinstance(cache_io, dict):
        errors = _prefill_cache_io_errors(cache_io)
        if errors:
            raise CliArgumentError(
                "launch audit request prefill cache I/O evidence is malformed: "
                + ", ".join(errors)
            )
    check = _launch_audit_check_by_code(
        audit,
        "request_prefill_cache_io_valid",
    )
    if not isinstance(check, dict):
        return
    if isinstance(cache_io, dict):
        if check.get("ok") is not True:
            raise CliArgumentError(
                "launch audit request prefill cache I/O check is not passing: "
                f"ok={check.get('ok')!r}"
            )
        if check.get("evidence_present") is not True:
            raise CliArgumentError(
                "launch audit request prefill cache I/O check is inconsistent: "
                f"evidence_present={check.get('evidence_present')!r}"
            )
        for field in _REQUEST_PREFILL_CACHE_IO_AUDIT_FIELDS:
            if check.get(field) == cache_io.get(field):
                continue
            raise CliArgumentError(
                "launch audit request prefill cache I/O check does not match "
                "request evidence: "
                f"{field} audit={check.get(field)!r} "
                f"request={cache_io.get(field)!r}"
            )


def _require_launch_audit_request_prefill_routed_chunk_frontier_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    request = artifact.get("request_check")
    frontier = (
        request.get("prefill_routed_chunk_frontier")
        if isinstance(request, dict)
        else None
    )
    if isinstance(frontier, dict):
        errors = _prefill_routed_chunk_frontier_errors(frontier)
        if errors:
            raise CliArgumentError(
                "launch audit request routed chunk frontier evidence is malformed: "
                + ", ".join(errors)
            )
    check = _launch_audit_check_by_code(
        audit,
        "request_prefill_routed_chunk_frontier_valid",
    )
    if not isinstance(check, dict):
        return
    if isinstance(frontier, dict):
        if check.get("ok") is not True:
            raise CliArgumentError(
                "launch audit request routed chunk frontier check is not passing: "
                f"ok={check.get('ok')!r}"
            )
        if check.get("evidence_present") is not True:
            raise CliArgumentError(
                "launch audit request routed chunk frontier check is inconsistent: "
                f"evidence_present={check.get('evidence_present')!r}"
            )
        for field in _REQUEST_PREFILL_ROUTED_CHUNK_FRONTIER_AUDIT_FIELDS:
            if check.get(field) == frontier.get(field):
                continue
            raise CliArgumentError(
                "launch audit request routed chunk frontier check does not "
                "match request evidence: "
                f"{field} audit={check.get(field)!r} "
                f"request={frontier.get(field)!r}"
            )


def _require_launch_audit_prefill_actual_read_time_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    audited_prompt, _ = _launch_audit_request_limits(artifact)
    applied = artifact.get("applied_launch_profile")
    if not isinstance(applied, dict):
        return
    actual = applied.get("prefill_actual_read_time")
    required = applied.get("source") == "benchmark_actual" and audited_prompt > 1
    if required and not isinstance(actual, dict):
        raise CliArgumentError(
            "launch audit applied benchmark profile is missing "
            "prefill_actual_read_time evidence"
        )
    if isinstance(actual, dict):
        _require_passing_prefill_actual_read_time_evidence(
            actual,
            source="launch audit applied prefill read-time evidence",
        )
    check = _launch_audit_check_by_code(
        audit,
        "applied_prefill_actual_read_time_ok",
    )
    if not isinstance(check, dict):
        return
    if required and check.get("required") is not True:
        raise CliArgumentError(
            "launch audit prefill read-time check is inconsistent: "
            f"required={check.get('required')!r}"
        )
    if isinstance(actual, dict):
        if check.get("ok") is not True:
            raise CliArgumentError(
                "launch audit prefill read-time check is not passing: "
                f"ok={check.get('ok')!r}"
            )
        for field in _PREFILL_ACTUAL_READ_TIME_AUDIT_FIELDS:
            if check.get(field) == actual.get(field):
                continue
            raise CliArgumentError(
                "launch audit prefill read-time check does not match applied "
                "profile evidence: "
                f"{field} audit={check.get(field)!r} "
                f"profile={actual.get(field)!r}"
            )


def _require_launch_audit_prefill_actual_acceleration_coverage_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    applied = artifact.get("applied_launch_profile")
    actual = (
        applied.get("prefill_actual_acceleration_coverage")
        if isinstance(applied, dict)
        else None
    )
    check = _launch_audit_check_by_code(
        audit,
        "applied_prefill_actual_acceleration_coverage_valid",
    )
    required = isinstance(check, dict) and check.get("required") is True
    if required and not isinstance(actual, dict):
        raise CliArgumentError(
            "launch audit applied benchmark profile is missing "
            "prefill_actual_acceleration_coverage evidence"
        )
    if isinstance(actual, dict):
        _require_valid_prefill_actual_acceleration_coverage_evidence(
            actual,
            source="launch audit applied prefill acceleration coverage evidence",
        )
    if not isinstance(check, dict):
        return
    if isinstance(actual, dict):
        if check.get("ok") is not True:
            raise CliArgumentError(
                "launch audit prefill acceleration coverage check is not "
                f"passing: ok={check.get('ok')!r}"
            )
        if check.get("evidence_present") is not True:
            raise CliArgumentError(
                "launch audit prefill acceleration coverage check is "
                f"inconsistent: evidence_present={check.get('evidence_present')!r}"
            )
        for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS:
            if check.get(field) == actual.get(field):
                continue
            raise CliArgumentError(
                "launch audit prefill acceleration coverage check does not "
                "match applied profile evidence: "
                f"{field} audit={check.get(field)!r} "
                f"profile={actual.get(field)!r}"
            )


def _require_launch_audit_prefill_actual_linear_backend_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    applied = artifact.get("applied_launch_profile")
    actual = (
        applied.get("prefill_actual_linear_backend")
        if isinstance(applied, dict)
        else None
    )
    actual_coverage = (
        applied.get("prefill_actual_acceleration_coverage")
        if isinstance(applied, dict)
        else None
    )
    if isinstance(actual, dict):
        _require_valid_prefill_actual_linear_backend_evidence(
            actual,
            source="launch audit applied prefill linear-backend evidence",
        )
    if isinstance(actual, dict) and isinstance(actual_coverage, dict):
        cross_errors = _prefill_actual_linear_backend_coverage_errors(
            actual_linear=actual,
            actual_coverage=actual_coverage,
        )
        if cross_errors:
            raise CliArgumentError(
                "launch audit applied prefill acceleration coverage does not "
                "match prefill linear-backend evidence: "
                + ", ".join(cross_errors)
            )
    check = _launch_audit_check_by_code(
        audit,
        "applied_prefill_actual_linear_backend_valid",
    )
    if not isinstance(check, dict):
        return
    if isinstance(actual, dict):
        if check.get("ok") is not True:
            raise CliArgumentError(
                "launch audit prefill linear-backend check is not passing: "
                f"ok={check.get('ok')!r}"
            )
        if check.get("evidence_present") is not True:
            raise CliArgumentError(
                "launch audit prefill linear-backend check is inconsistent: "
                f"evidence_present={check.get('evidence_present')!r}"
            )
        for field in _PREFILL_ACTUAL_LINEAR_BACKEND_AUDIT_FIELDS:
            if check.get(field) == actual.get(field):
                continue
            raise CliArgumentError(
                "launch audit prefill linear-backend check does not match "
                "applied profile evidence: "
                f"{field} audit={check.get(field)!r} "
                f"profile={actual.get(field)!r}"
            )


def _require_launch_audit_decode_actual_read_time_evidence(
    artifact: dict[str, object],
    audit: dict[str, object],
) -> None:
    applied = artifact.get("applied_launch_profile")
    actual = (
        applied.get("decode_actual_read_time")
        if isinstance(applied, dict)
        else None
    )
    if isinstance(actual, dict):
        _require_passing_decode_actual_read_time_evidence(
            actual,
            source="launch audit applied decode read-time evidence",
        )
    check = _launch_audit_check_by_code(
        audit,
        "applied_decode_actual_read_time_ok",
    )
    if not isinstance(check, dict):
        return
    if isinstance(actual, dict):
        if check.get("ok") is not True:
            raise CliArgumentError(
                "launch audit decode read-time check is not passing: "
                f"ok={check.get('ok')!r}"
            )
        for field in _DECODE_ACTUAL_READ_TIME_AUDIT_FIELDS:
            if check.get(field) == actual.get(field):
                continue
            raise CliArgumentError(
                "launch audit decode read-time check does not match applied "
                "profile evidence: "
                f"{field} audit={check.get(field)!r} "
                f"profile={actual.get(field)!r}"
            )


def _launch_profile_pair_map(
    request_profile: dict[str, object],
) -> dict[str, str | None]:
    return {flag: value for flag, value in _launch_profile_argv_pairs(request_profile)}


def _require_profile_int_equals(
    values: dict[str, str | None],
    *,
    flag: str,
    expected: int,
    source: str,
) -> None:
    raw = values.get(flag)
    if flag == "--prefill-prompt-chunk-tokens" and raw == "auto" and expected > 0:
        return
    try:
        actual = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(
            f"{source} has invalid {flag} value: {raw!r}"
        ) from exc
    if actual != expected:
        raise CliArgumentError(
            f"{source} does not match audited request evidence: "
            f"{flag} profile={actual!r} expected={expected!r}"
        )


def _require_profile_int_at_most(
    values: dict[str, str | None],
    *,
    flag: str,
    maximum: int,
    source: str,
) -> None:
    raw = values.get(flag)
    try:
        actual = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(
            f"{source} has invalid {flag} value: {raw!r}"
        ) from exc
    if actual <= 0 or actual > maximum:
        raise CliArgumentError(
            f"{source} is wider than audited request evidence: "
            f"{flag} profile={actual!r} maximum={maximum!r}"
        )


def _require_profile_float_close(
    values: dict[str, str | None],
    *,
    flag: str,
    expected: float,
    source: str,
) -> None:
    raw = values.get(flag)
    try:
        actual = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(
            f"{source} has invalid {flag} value: {raw!r}"
        ) from exc
    if not math.isfinite(actual) or not math.isclose(
        actual,
        expected,
        rel_tol=1e-6,
        abs_tol=1e-12,
    ):
        raise CliArgumentError(
            f"{source} does not match audited request evidence: "
            f"{flag} profile={actual!r} expected={expected!r}"
        )


def _require_profile_float_at_most(
    values: dict[str, str | None],
    *,
    flag: str,
    maximum: float,
    source: str,
) -> None:
    raw = values.get(flag)
    try:
        actual = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise CliArgumentError(
            f"{source} has invalid {flag} value: {raw!r}"
        ) from exc
    tolerance = max(abs(maximum) * 1e-5, 1e-12)
    if not math.isfinite(actual) or actual <= 0 or actual > maximum + tolerance:
        raise CliArgumentError(
            f"{source} is wider than audited request evidence: "
            f"{flag} profile={actual!r} maximum={maximum!r}"
        )


def _require_launch_audit_request_profile_binds_read_budgets(
    artifact: dict[str, object],
    audit: dict[str, object],
    request_profile: dict[str, object],
) -> None:
    profile_flags = set(_launch_profile_flags(request_profile))
    profile_values = _launch_profile_pair_map(request_profile)
    request = artifact.get("request_check")
    request = request if isinstance(request, dict) else {}
    prefill_check = _launch_audit_check_by_code(
        audit,
        "request_prefill_routed_read_budget_ok",
    )
    if isinstance(prefill_check, dict) and prefill_check.get("required") is True:
        missing_prefill = tuple(
            flag
            for flag in _PREFILL_ROUTED_READ_PROFILE_FLAGS
            if flag not in profile_flags
        )
        if missing_prefill:
            raise CliArgumentError(
                "launch audit request profile is missing prefill routed-read "
                "guard flags: "
                + ", ".join(missing_prefill)
            )
        routed_read = request.get("prefill_routed_expert_read")
        if isinstance(routed_read, dict):
            prompt_chunk = routed_read.get("prompt_chunk_tokens")
            if not isinstance(prompt_chunk, bool) and isinstance(prompt_chunk, int):
                _require_profile_int_equals(
                    profile_values,
                    flag="--prefill-prompt-chunk-tokens",
                    expected=prompt_chunk,
                    source="launch audit request profile prefill routed-read guard",
                )
            planned_bytes = routed_read.get("planned_read_bytes")
            if not isinstance(planned_bytes, bool) and isinstance(planned_bytes, int):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--prefill-max-routed-read-gib",
                    maximum=planned_bytes / 1024**3 * 1.05,
                    source="launch audit request profile prefill routed-read guard",
                )
            read_amp = routed_read.get("read_amplification")
            if _audit_finite_number(read_amp):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--prefill-max-routed-read-amplification",
                    maximum=max(1.0, float(read_amp)) * 1.05,
                    source="launch audit request profile prefill routed-read guard",
                )
            ssd = routed_read.get("ssd_read_gib_per_second")
            if _audit_finite_number(ssd):
                _require_profile_float_close(
                    profile_values,
                    flag="--prefill-ssd-read-gib-s",
                    expected=float(ssd),
                    source="launch audit request profile prefill routed-read guard",
                )
            planned_seconds = routed_read.get("planned_read_seconds")
            if _audit_finite_number(planned_seconds):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--prefill-max-routed-read-seconds",
                    maximum=float(planned_seconds) * 1.05,
                    source="launch audit request profile prefill routed-read guard",
                )
    decode_check = _launch_audit_check_by_code(
        audit,
        "request_decode_routed_read_budget_ok",
    )
    if isinstance(decode_check, dict) and decode_check.get("required") is True:
        missing_decode = tuple(
            flag
            for flag in _DECODE_ROUTED_READ_PROFILE_FLAGS
            if flag not in profile_flags
        )
        if missing_decode:
            raise CliArgumentError(
                "launch audit request profile is missing decode routed-read "
                "guard flags: "
                + ", ".join(missing_decode)
            )
        decode_read = request.get("decode_routed_expert_read")
        if isinstance(decode_read, dict):
            read_bytes = decode_read.get("read_bytes_per_token")
            if not isinstance(read_bytes, bool) and isinstance(read_bytes, int):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--decode-max-routed-read-gib-per-token",
                    maximum=read_bytes / 1024**3 * 1.05,
                    source="launch audit request profile decode routed-read guard",
                )
            ssd = decode_read.get("ssd_read_gib_per_second")
            if _audit_finite_number(ssd):
                _require_profile_float_close(
                    profile_values,
                    flag="--prefill-ssd-read-gib-s",
                    expected=float(ssd),
                    source="launch audit request profile decode routed-read guard",
                )
            planned_seconds = decode_read.get("planned_read_seconds_per_token")
            if _audit_finite_number(planned_seconds):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--decode-max-routed-read-seconds-per-token",
                    maximum=float(planned_seconds) * 1.05,
                    source="launch audit request profile decode routed-read guard",
                )
    stage_check = _launch_audit_check_by_code(
        audit,
        "request_prefill_stage_temp_limit_ok",
    )
    if isinstance(stage_check, dict) and stage_check.get("required") is True:
        missing_stage = tuple(
            flag
            for flag in _PREFILL_STAGE_TEMP_PROFILE_FLAGS
            if flag not in profile_flags
        )
        if missing_stage:
            raise CliArgumentError(
                "launch audit request profile is missing stage-temp guard flags: "
                + ", ".join(missing_stage)
            )
        stage_temp = request.get("prefill_routed_stage_temp_disk")
        if isinstance(stage_temp, dict):
            stage_tiling = stage_temp.get("expert_stage_tiling") is True
            if (
                stage_tiling
                and "--prefill-expert-stage-tiling" not in profile_flags
            ):
                raise CliArgumentError(
                    "launch audit request profile is missing stage-temp guard "
                    "flags: --prefill-expert-stage-tiling"
                )
            max_stage = (
                stage_temp.get("effective_max_stage_bytes")
                if stage_tiling
                else stage_temp.get("max_stage_bytes")
            )
            if not isinstance(max_stage, bool) and isinstance(max_stage, int):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--prefill-max-stage-mib",
                    maximum=max_stage / 1024**2 * 1.05,
                    source="launch audit request profile stage-temp guard",
                )
            max_compact = (
                stage_temp.get("effective_max_compact_stage_bytes")
                if stage_tiling
                else stage_temp.get("max_compact_stage_bytes")
            )
            if not isinstance(max_compact, bool) and isinstance(max_compact, int):
                _require_profile_float_at_most(
                    profile_values,
                    flag="--prefill-max-compact-stage-mib",
                    maximum=max_compact / 1024**2 * 1.05,
                    source="launch audit request profile stage-temp guard",
                )
            max_raw_ranges = (
                stage_temp.get("effective_max_stage_raw_ranges")
                if stage_tiling
                else stage_temp.get("max_stage_raw_ranges")
            )
            if not isinstance(max_raw_ranges, bool) and isinstance(max_raw_ranges, int):
                _require_profile_int_at_most(
                    profile_values,
                    flag="--prefill-max-stage-raw-ranges",
                    maximum=max(1, math.ceil(max_raw_ranges * 1.05)),
                    source="launch audit request profile stage-temp guard",
                )
            max_coalesced_ranges = (
                stage_temp.get("effective_max_stage_coalesced_ranges")
                if stage_tiling
                else stage_temp.get("max_stage_coalesced_ranges")
            )
            if (
                not isinstance(max_coalesced_ranges, bool)
                and isinstance(max_coalesced_ranges, int)
            ):
                _require_profile_int_at_most(
                    profile_values,
                    flag="--prefill-max-stage-coalesced-ranges",
                    maximum=max(1, math.ceil(max_coalesced_ranges * 1.05)),
                    source="launch audit request profile stage-temp guard",
                )
            static_capacity = stage_temp.get("static_capacity_per_expert")
            if static_capacity is not None:
                raw_static = profile_values.get("--prefill-static-capacity-per-expert")
                if raw_static is None:
                    raise CliArgumentError(
                        "launch audit request profile is missing "
                        "--prefill-static-capacity-per-expert"
                    )
                if str(raw_static) != str(static_capacity):
                    raise CliArgumentError(
                        "launch audit request profile stage-temp guard does not "
                        "match audited request evidence: "
                        "--prefill-static-capacity-per-expert "
                        f"profile={raw_static!r} expected={static_capacity!r}"
                    )


def _require_launch_audit_artifact(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> dict[str, object] | None:
    path_arg = getattr(args, "require_launch_audit", None)
    if path_arg is None:
        return None
    artifact = _load_launch_audit_artifact(path_arg)
    audit = artifact.get("launch_audit")
    if not isinstance(audit, dict) or audit.get("ok") is not True:
        failures = audit.get("failures") if isinstance(audit, dict) else None
        raise CliArgumentError(
            "required launch audit did not pass"
            + (f": failures={failures}" if failures else "")
        )
    _require_launch_audit_checks(audit)
    _require_launch_audit_runtime_preflight_evidence(artifact, audit)
    _require_launch_audit_request_routed_read_evidence(artifact, audit)
    _require_launch_audit_request_decode_routed_read_evidence(artifact, audit)
    _require_launch_audit_request_prefill_prompt_chunk_plan_evidence(
        artifact,
        audit,
    )
    _require_launch_audit_request_stage_temp_evidence(artifact, audit)
    _require_launch_audit_request_prefill_cache_io_evidence(artifact, audit)
    _require_launch_audit_request_prefill_routed_chunk_frontier_evidence(
        artifact,
        audit,
    )
    _require_launch_audit_request_prefill_acceleration_coverage_evidence(
        artifact,
        audit,
    )
    _require_launch_audit_prefill_actual_read_time_evidence(artifact, audit)
    _require_launch_audit_prefill_actual_acceleration_coverage_evidence(
        artifact,
        audit,
    )
    _require_launch_audit_prefill_actual_linear_backend_evidence(artifact, audit)
    _require_launch_audit_decode_actual_read_time_evidence(artifact, audit)
    target = artifact.get("prepared")
    if not isinstance(target, dict):
        raise CliArgumentError("launch audit is missing prepared identity")
    current = prepared_launch_profile_target(prepared)
    if target.get("identity_strength") != "strong":
        raise CliArgumentError(
            "launch audit prepared identity is weak; regenerate it from a "
            "hash-bound prepared package"
        )
    if current.get("identity_strength") != "strong":
        raise CliArgumentError(
            "current prepared package identity is weak; regenerate it with "
            "prepare-glm so model_config_sha256 is recorded"
        )
    request_profile = artifact.get("request_launch_profile")
    if not isinstance(request_profile, dict):
        raise CliArgumentError("launch audit is missing request launch profile")
    if request_profile.get("argv_safe_to_replay") is not True:
        raise CliArgumentError("launch audit request profile is not safe to replay")
    if not _launch_profile_argv_pairs(request_profile):
        raise CliArgumentError(
            "launch audit request profile has no replayable guard flags"
        )
    request_target = request_profile.get("prepared")
    if not isinstance(request_target, dict):
        raise CliArgumentError(
            "launch audit request profile is missing prepared identity"
        )
    if request_target.get("identity_strength") != "strong":
        raise CliArgumentError(
            "launch audit request profile prepared identity is weak; "
            "regenerate it from a hash-bound prepared package"
        )
    if not _launch_profile_targets_match(request_target, target):
        raise CliArgumentError(
            "launch audit request profile does not match audit prepared identity"
        )
    _require_launch_audit_request_profile_binds_read_budgets(
        artifact,
        audit,
        request_profile,
    )
    for field in _LAUNCH_PROFILE_TARGET_FIELDS:
        if not _launch_profile_target_values_equal(
            target.get(field),
            current.get(field),
        ):
            raise CliArgumentError(
                "launch audit does not match this prepared package: "
                f"{field} audit={target.get(field)!r} "
                f"current={current.get(field)!r}"
            )
    for field in _LAUNCH_PROFILE_CONDITIONAL_TARGET_FIELDS:
        if not _launch_profile_conditional_target_field_applies(
            target,
            current,
            field,
        ):
            continue
        if not _launch_profile_target_values_equal(
            target.get(field),
            current.get(field),
        ):
            raise CliArgumentError(
                "launch audit does not match this prepared package: "
                f"{field} audit={target.get(field)!r} "
                f"current={current.get(field)!r}"
            )
    for field in _LAUNCH_PROFILE_OPTIONAL_TARGET_FIELDS:
        if field in target and not _launch_profile_target_values_equal(
            target.get(field),
            current.get(field),
        ):
            raise CliArgumentError(
                "launch audit does not match this prepared package: "
                f"{field} audit={target.get(field)!r} "
                f"current={current.get(field)!r}"
            )
    _require_launch_audit_prepared_runtime_profile_matches_prepared(audit, prepared)
    _require_launch_audit_prepared_ssd_read_matches_prepared(audit, prepared)
    _require_launch_audit_prepare_pack_heap_matches_prepared(audit, prepared)
    _require_launch_audit_resident_alias_rewrite_matches_prepared(audit, prepared)
    _require_launch_audit_glm_envelope_matches_prepared(audit, prepared)
    _require_launch_audit_prefill_acceleration_evidence(audit)
    audited_profile = artifact.get("applied_launch_profile")
    current_profile = _applied_launch_profile_summary(args, prepared)
    if not isinstance(audited_profile, dict):
        raise CliArgumentError("launch audit is missing applied launch profile")
    if not isinstance(current_profile, dict):
        raise CliArgumentError(
            "required launch audit needs --apply-launch-profile on this command"
        )
    if current_profile.get("locked") is not True:
        raise CliArgumentError(
            "required launch audit needs --lock-launch-profile on this command"
        )
    current_profile_payload: dict[str, object] | None = None
    current_profile_path = getattr(args, "apply_launch_profile", None)
    if current_profile_path is not None:
        current_profile_payload = _load_launch_profile(current_profile_path)
    current_is_audited_applied_profile = (
        current_profile.get("sha256") == audited_profile.get("sha256")
    )
    current_is_audited_request_profile = (
        isinstance(current_profile_payload, dict)
        and isinstance(request_profile, dict)
        and _launch_profiles_have_same_argv(current_profile_payload, request_profile)
        and _launch_profile_targets_match(
            current_profile_payload.get("prepared"),
            request_profile.get("prepared"),
        )
        and _launch_profiles_have_same_prefill_prompt_chunk_plan(
            current_profile_payload,
            request_profile,
        )
    )
    if not current_is_audited_applied_profile and not current_is_audited_request_profile:
        raise CliArgumentError(
            "launch audit was created for a different launch profile: "
            f"audit_sha={audited_profile.get('sha256')!r} "
            f"current_sha={current_profile.get('sha256')!r}; "
            "apply the audited launch profile or the artifact's request_launch_profile"
        )
    return artifact


def _launch_audit_request_limits(
    artifact: dict[str, object],
) -> tuple[int, int]:
    request = artifact.get("request_check")
    if not isinstance(request, dict) or request.get("ok") is not True:
        raise CliArgumentError("launch audit is missing a passing request check")
    audited_prompt = request.get("prompt_token_count")
    audited_new = request.get("max_new_tokens")
    if (
        isinstance(audited_prompt, bool)
        or not isinstance(audited_prompt, int)
        or audited_prompt <= 0
    ):
        raise CliArgumentError("launch audit request prompt_token_count is invalid")
    if (
        isinstance(audited_new, bool)
        or not isinstance(audited_new, int)
        or audited_new < 0
    ):
        raise CliArgumentError("launch audit request max_new_tokens is invalid")
    return audited_prompt, audited_new


def _require_launch_audit_server_applied_profile_matches_current(
    args: argparse.Namespace,
    prepared: PreparedManifest,
    artifact: dict[str, object],
) -> None:
    audited_profile = artifact.get("applied_launch_profile")
    if not isinstance(audited_profile, dict):
        raise CliArgumentError("launch audit is missing applied launch profile")
    current_profile = _applied_launch_profile_summary(args, prepared)
    if not isinstance(current_profile, dict):
        raise CliArgumentError(
            "server launch audit requires --apply-launch-profile on this command"
        )
    audit_sha = audited_profile.get("sha256")
    current_sha = current_profile.get("sha256")
    if audit_sha != current_sha:
        raise CliArgumentError(
            "server launch audit was created for a different applied launch "
            "profile: "
            f"audit_sha={audit_sha!r} current_sha={current_sha!r}; "
            "regenerate the launch audit with the exact server launch profile"
        )


def _require_launch_audit_request_profile_args(
    args: argparse.Namespace,
    artifact: dict[str, object],
) -> None:
    request_profile = artifact.get("request_launch_profile")
    if not isinstance(request_profile, dict):
        return
    if request_profile.get("argv_safe_to_replay") is False:
        raise CliArgumentError("launch audit request profile is not safe to replay")
    request_check = artifact.get("request_check")
    request_linear_backend = (
        request_check.get("prefill_linear_backend")
        if isinstance(request_check, dict)
        else None
    )
    request_uses_auto_backend_policy = _request_prefill_backend_is_auto(
        request_linear_backend
    )
    pairs = _launch_profile_argv_pairs(request_profile)
    if not pairs:
        return
    for flag, profile_value in pairs:
        attr = _LAUNCH_PROFILE_FLAG_ATTRS.get(flag)
        if attr is None:
            raise CliArgumentError(
                f"launch audit request profile flag {flag!r} is not supported"
            )
        actual = getattr(args, attr, None)
        if _launch_profile_value_matches(
            flag=flag,
            profile_value=profile_value,
            actual=actual,
        ):
            continue
        if _launch_audit_request_value_is_no_wider(
            flag=flag,
            profile_value=profile_value,
            actual=actual,
        ):
            continue
        if (
            flag == "--prefill-linear-backend"
            and actual == "auto"
            and request_uses_auto_backend_policy
        ):
            continue
        expected = "enabled" if profile_value is None else profile_value
        raise CliArgumentError(
            "current command does not replay launch audit request profile: "
            f"{flag} profile={expected!r} current={actual!r}"
        )


def _require_launch_audit_request_envelope(
    args: argparse.Namespace,
    prepared: PreparedManifest,
    *,
    prompt_token_count: int,
    max_new_tokens: int,
) -> None:
    artifact = _require_launch_audit_artifact(args, prepared)
    if artifact is None:
        return
    audited_prompt, audited_new = _launch_audit_request_limits(artifact)
    _require_current_runtime_preflight_for_launch_audit(args, artifact)
    _require_launch_audit_request_profile_args(args, artifact)
    if prompt_token_count > audited_prompt:
        raise CliArgumentError(
            "current request exceeds launch audit prompt envelope: "
            f"prompt_tokens={prompt_token_count} audited_prompt_tokens={audited_prompt}"
        )
    if max_new_tokens > audited_new:
        raise CliArgumentError(
            "current request exceeds launch audit generation envelope: "
            f"max_new_tokens={max_new_tokens} audited_max_new_tokens={audited_new}"
        )


def _require_current_server_memory_guard_for_launch_audit(
    args: argparse.Namespace,
) -> dict[str, object]:
    max_live_mib = getattr(args, "max_live_working_set_mib", None)
    min_free_gib = getattr(args, "min_free_unified_memory_gib", None)
    max_live_bytes = (
        int(float(max_live_mib) * 1024**2)
        if max_live_mib is not None
        else None
    )
    min_free_bytes = (
        int(float(min_free_gib) * 1024**3)
        if min_free_gib is not None
        else 0
    )
    required_available = (
        None if max_live_bytes is None else max_live_bytes + min_free_bytes
    )
    snapshot = system_memory_snapshot()
    if snapshot is None:
        raise CliArgumentError(
            "required launch audit could not verify current server memory guard"
        )
    available = snapshot.available_bytes
    total = snapshot.total_bytes
    source = snapshot.source
    if (
        required_available is not None
        and (available is None or available < required_available)
    ):
        raise CliArgumentError(
            "current server memory is below launch audit guard: "
            f"available={available!r} required={required_available!r}"
        )
    return {
        "server_configured_max_live_working_set_bytes": max_live_bytes,
        "server_configured_min_free_unified_memory_bytes": min_free_bytes,
        "server_configured_required_available_memory_bytes": required_available,
        "server_system_available_memory_bytes": available,
        "server_system_total_memory_bytes": total,
        "server_system_memory_source": source,
        "server_memory_guard_ok": True,
    }


def _require_launch_audit_server_envelope(
    args: argparse.Namespace,
    prepared: PreparedManifest,
) -> dict[str, object] | None:
    artifact = _require_launch_audit_artifact(args, prepared)
    if artifact is None:
        return None
    _require_launch_audit_server_applied_profile_matches_current(
        args,
        prepared,
        artifact,
    )
    _apply_prepared_memory_guard_defaults(args, prepared)
    audited_prompt, audited_new = _launch_audit_request_limits(artifact)
    _require_launch_audit_request_profile_args(args, artifact)
    current_memory_guard = _require_current_server_memory_guard_for_launch_audit(args)
    prompt_cap = int(getattr(args, "max_prompt_tokens", 0))
    max_new_cap = int(getattr(args, "max_new_tokens_cap", 0))
    if prompt_cap > audited_prompt:
        raise CliArgumentError(
            "server prompt token cap exceeds launch audit prompt envelope: "
            f"max_prompt_tokens={prompt_cap} "
            f"audited_prompt_tokens={audited_prompt}"
        )
    if max_new_cap > audited_new:
        raise CliArgumentError(
            "server max_new_tokens cap exceeds launch audit generation envelope: "
            f"max_new_tokens_cap={max_new_cap} "
            f"audited_max_new_tokens={audited_new}"
        )
    request = artifact.get("request_check")
    audited_context = (
        request.get("required_context_tokens")
        if isinstance(request, dict)
        else None
    )
    if (
        isinstance(audited_context, bool)
        or not isinstance(audited_context, int)
        or audited_context <= 0
    ):
        audited_context = audited_prompt + audited_new
    runtime = (
        request.get("runtime_preflight")
        if isinstance(request, dict)
        else None
    )
    audited_profile = artifact.get("applied_launch_profile")
    return {
        "schema": "largerlm.launch_audit_server_envelope.v1",
        "artifact_path": str(getattr(args, "require_launch_audit", "")),
        "artifact_source": artifact.get("source"),
        "applied_launch_profile_sha256": (
            audited_profile.get("sha256")
            if isinstance(audited_profile, dict)
            else None
        ),
        "audited_prompt_token_count": audited_prompt,
        "audited_max_new_tokens": audited_new,
        "audited_required_context_tokens": audited_context,
        "audited_runtime_required_available_memory_bytes": (
            runtime.get("required_available_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        "audited_runtime_system_available_memory_bytes": (
            runtime.get("system_available_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        "audited_runtime_system_total_memory_bytes": (
            runtime.get("system_total_memory_bytes")
            if isinstance(runtime, dict)
            else None
        ),
        "audited_runtime_system_memory_source": (
            runtime.get("system_memory_source")
            if isinstance(runtime, dict)
            else None
        ),
        "server_max_prompt_tokens": prompt_cap,
        "server_max_new_tokens_cap": max_new_cap,
        "server_caps_within_envelope": True,
        **current_memory_guard,
    }


def _write_launch_profile_file(
    path_arg: str | Path,
    profile: dict[str, object] | None,
) -> None:
    if not isinstance(profile, dict):
        raise CliArgumentError("no suggested launch profile is available to write")
    path = Path(path_arg)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(profile, default=_json_default, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise CliArgumentError(f"failed to write launch profile {path}: {exc}") from exc


def _write_prefill_calibration_flags_file(
    path_arg: str | Path,
    flags: dict[str, object] | None,
) -> None:
    if not isinstance(flags, dict):
        raise CliArgumentError("no suggested prefill calibration flags are available to write")
    path = Path(path_arg)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(flags, default=_json_default, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise CliArgumentError(
            f"failed to write prefill calibration flags {path}: {exc}"
        ) from exc


def _write_plan_prepare_flags_file(
    path_arg: str | Path,
    flags: dict[str, object] | None,
) -> None:
    if not isinstance(flags, dict):
        raise CliArgumentError("no suggested prepare flags are available to write")
    path = Path(path_arg)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(flags, default=_json_default, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise CliArgumentError(f"failed to write prepare flags {path}: {exc}") from exc


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _read_check_text_file_bounded(
    path_arg: str | Path,
    *,
    max_bytes: int,
    label: str,
    option_name: str = "--max-request-bytes",
) -> str:
    if max_bytes <= 0:
        raise TextGenerationError(f"{option_name} must be positive")
    path = Path(path_arg)
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise TextGenerationError(
            f"failed to read {label} file {path}: {exc}"
        ) from exc
    if len(data) > max_bytes:
        raise TextGenerationError(
            f"{label} file {path} exceeds {option_name} ({max_bytes} bytes)"
        )
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TextGenerationError(
            f"{label} file {path} is not valid UTF-8: {exc}"
        ) from exc


def _read_check_prompt_arg(
    *,
    prompt: str | None,
    prompt_file: str | Path | None,
    max_bytes: int,
) -> str:
    if prompt is not None and prompt_file is not None:
        raise TextGenerationError(
            "--check-prompt and --check-prompt-file are mutually exclusive"
        )
    if prompt_file is not None:
        return _read_check_text_file_bounded(
            prompt_file,
            max_bytes=max_bytes,
            label="prompt",
        )
    if prompt is None:
        raise TextGenerationError("--check-prompt or --check-prompt-file is required")
    if _utf8_len(prompt) > max_bytes:
        raise TextGenerationError(
            f"--check-prompt exceeds --max-request-bytes ({max_bytes} bytes)"
        )
    return prompt


def _read_generation_prompt_arg(
    *,
    prompt: str | None,
    prompt_file: str | Path | None,
    max_bytes: int,
) -> str:
    if prompt is not None and prompt_file is not None:
        raise TextGenerationError("--prompt and --prompt-file are mutually exclusive")
    if prompt_file is not None:
        return _read_check_text_file_bounded(
            prompt_file,
            max_bytes=max_bytes,
            label="prompt",
            option_name="--max-prompt-bytes",
        )
    if prompt is None:
        raise TextGenerationError("--prompt or --prompt-file is required")
    if _utf8_len(prompt) > max_bytes:
        raise TextGenerationError(
            f"--prompt exceeds --max-prompt-bytes ({max_bytes} bytes)"
        )
    return prompt


def _read_check_chat_messages_arg(
    *,
    messages_json: str | None,
    messages_file: str | None,
    max_bytes: int,
) -> list[dict[str, str]]:
    if messages_json is not None and messages_file is not None:
        raise TextGenerationError(
            "--check-chat-messages and --check-chat-messages-file are mutually exclusive"
        )
    if messages_file is not None:
        raw_text = _read_check_text_file_bounded(
            messages_file,
            max_bytes=max_bytes,
            label="chat messages",
        )
    elif messages_json is not None:
        if _utf8_len(messages_json) > max_bytes:
            raise TextGenerationError(
                "--check-chat-messages exceeds "
                f"--max-request-bytes ({max_bytes} bytes)"
            )
        raw_text = messages_json
    else:
        raise TextGenerationError(
            "--check-chat-messages or --check-chat-messages-file is required"
        )
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TextGenerationError(f"failed to parse chat messages JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise TextGenerationError("chat messages must be a non-empty JSON array")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TextGenerationError(f"chat messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise TextGenerationError(
                f"chat messages[{index}].role must be a non-empty string"
            )
        if not isinstance(content, str):
            raise TextGenerationError(
                f"chat messages[{index}].content must be a string"
            )
        messages.append({"role": role, "content": content})
    return messages


def _read_generation_chat_messages_arg(
    *,
    messages_json: str | None,
    messages_file: str | None,
    max_bytes: int,
) -> list[dict[str, str]]:
    if messages_json is not None and messages_file is not None:
        raise TextGenerationError(
            "--chat-messages and --chat-messages-file are mutually exclusive"
        )
    if messages_file is not None:
        raw_text = _read_check_text_file_bounded(
            messages_file,
            max_bytes=max_bytes,
            label="chat messages",
            option_name="--max-prompt-bytes",
        )
    elif messages_json is not None:
        if _utf8_len(messages_json) > max_bytes:
            raise TextGenerationError(
                "--chat-messages exceeds "
                f"--max-prompt-bytes ({max_bytes} bytes)"
            )
        raw_text = messages_json
    else:
        raise TextGenerationError("--chat-messages or --chat-messages-file is required")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise TextGenerationError(f"failed to parse chat messages JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise TextGenerationError("chat messages must be a non-empty JSON array")
    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TextGenerationError(f"chat messages[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role:
            raise TextGenerationError(
                f"chat messages[{index}].role must be a non-empty string"
            )
        if not isinstance(content, str):
            raise TextGenerationError(
                f"chat messages[{index}].content must be a string"
            )
        messages.append({"role": role, "content": content})
    return messages


def _generate_prepared_token_ids(args: argparse.Namespace) -> int:
    prepared = load_prepared_manifest(args.prepared)
    _require_launch_profile_matches_prepared(args, prepared)
    _apply_prepared_memory_guard_defaults(args, prepared)
    if args.model_config is None:
        args.model_config = str(prepared.model_dir)
    validate_layout_model_config_sha256(
        model_config_path=args.model_config,
        expert_layout_path=prepared.experts_layout,
        resident_layout_path=prepared.resident_layout,
    )
    _require_prepared_glm_4bit_if_requested(args, prepared)
    _require_prepared_public_glm_5_2_shape_if_requested(args, prepared)
    _require_prepared_runtime_profile(
        prepared,
        require_memory_profile=_prepared_memory_profile_required_by_args(args),
    )
    _require_prefill_acceleration_if_requested(args)
    prompt_token_ids = _parse_token_ids(args.prompt_token_ids)
    if (
        not args.batch_prefill_prompt
        and not args.no_batch_prefill_prompt
        and len(prompt_token_ids) > 1
    ):
        args.batch_prefill_prompt = True
    if args.batch_prefill_prompt and not args.no_batch_prefill_prompt:
        _default_prepared_static_capacity(args)
    kwargs = _generation_kwargs(args)
    _require_prepared_prefill_acceleration_coverage_if_requested(
        args,
        prompt_token_count=len(prompt_token_ids),
        generation_overrides=kwargs,
    )
    runtime_prefill_linear_backend = _require_prepared_request_admission(
        args,
        prompt_token_count=len(prompt_token_ids),
        generation_overrides=kwargs,
        runtime_preflight=bool(getattr(args, "preflight_runtime", False)),
    )
    _apply_prepared_runtime_prefill_backend(
        args,
        kwargs,
        runtime_prefill_linear_backend,
    )
    _require_launch_audit_request_envelope(
        args,
        prepared,
        prompt_token_count=len(prompt_token_ids),
        max_new_tokens=args.max_new_tokens,
    )
    prepared_lock = _acquire_direct_prepared_generation_lock(prepared)
    try:
        result = generate_token_ids(
            runner_path=args.runner,
            expert_layout_path=prepared.experts_layout,
            resident_layout_path=prepared.resident_layout,
            cache_layout_path=prepared.decode_cache_layout,
            cache_file_path=prepared.decode_cache_file,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=args.eos_token_id,
            **kwargs,
        )
    finally:
        if prepared_lock is not None:
            prepared_lock.close()
    applied_launch_profile = _applied_launch_profile_summary(args, prepared)
    result = _attach_applied_profile_to_token_result(result, applied_launch_profile)
    _require_actual_prepared_prefill_acceleration_if_requested(args, result)
    if args.write_result is not None:
        _write_json_file_atomic(
            args.write_result,
            _prepared_token_generation_result_payload(
                prepared=prepared,
                max_new_tokens=args.max_new_tokens,
                launch_audit_path=getattr(args, "require_launch_audit", None),
                prefill_mla_kv_b_cache_dir=getattr(
                    args,
                    "prefill_mla_kv_b_cache_dir",
                    None,
                ),
                result=result,
            ),
        )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_token_generation_result(result)
    return 0


def _generate_prepared_text(args: argparse.Namespace) -> int:
    prepared = load_prepared_manifest(args.prepared)
    _require_launch_profile_matches_prepared(args, prepared)
    _apply_prepared_memory_guard_defaults(args, prepared)
    if args.model_config is None:
        args.model_config = str(prepared.model_dir)
    validate_layout_model_config_sha256(
        model_config_path=args.model_config,
        expert_layout_path=prepared.experts_layout,
        resident_layout_path=prepared.resident_layout,
    )
    _require_prepared_glm_4bit_if_requested(args, prepared)
    _require_prepared_public_glm_5_2_shape_if_requested(args, prepared)
    _require_prepared_runtime_profile(
        prepared,
        require_memory_profile=_prepared_memory_profile_required_by_args(args),
    )
    _require_prefill_acceleration_if_requested(args)
    tokenizer = args.tokenizer or str(prepared.model_dir)
    prompt = _read_generation_prompt_arg(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        max_bytes=args.max_prompt_bytes,
    )
    prompt_token_count: int | None = None
    try:
        encoded = encode_prompt(
            tokenizer,
            prompt,
            backend=args.tokenizer_backend,
            trust_remote_code=args.trust_remote_code,
            add_special_tokens=not args.no_add_special_tokens,
        )
        prompt_token_count = len(encoded.token_ids)
    except TokenizerError as exc:
        raise TextGenerationError(
            "prepared text prompt tokenization failed before request admission: "
            f"{exc}"
        ) from exc
    if prompt_token_count is not None:
        if (
            args.max_prompt_tokens is not None
            and prompt_token_count > args.max_prompt_tokens
        ):
            raise TextGenerationError(
                f"prompt token length {prompt_token_count} exceeds limit "
                f"{args.max_prompt_tokens}"
            )
        if (
            not args.no_batch_prefill_prompt
            and not args.batch_prefill_prompt
            and prompt_token_count > 1
        ):
            args.batch_prefill_prompt = True
    if not args.no_batch_prefill_prompt:
        _default_prepared_static_capacity(args)
    if prompt_token_count is not None:
        kwargs = _generation_kwargs(args)
        _require_prepared_prefill_acceleration_coverage_if_requested(
            args,
            prompt_token_count=prompt_token_count,
            generation_overrides=kwargs,
        )
        runtime_prefill_linear_backend = _require_prepared_request_admission(
            args,
            prompt_token_count=prompt_token_count,
            generation_overrides=kwargs,
            runtime_preflight=bool(getattr(args, "preflight_runtime", False)),
        )
        _apply_prepared_runtime_prefill_backend(
            args,
            kwargs,
            runtime_prefill_linear_backend,
        )
        _require_launch_audit_request_envelope(
            args,
            prepared,
            prompt_token_count=prompt_token_count,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        if getattr(args, "require_launch_audit", None) is not None:
            raise CliArgumentError(
                "required launch audit needs prompt token count before generation"
            )
        kwargs = _generation_kwargs(args)
    if not args.no_batch_prefill_prompt:
        kwargs["auto_batch_prefill_prompt"] = True
    prepared_lock = _acquire_direct_prepared_generation_lock(prepared)
    try:
        result = generate_text(
            tokenizer_path=tokenizer,
            tokenizer_backend=args.tokenizer_backend,
            trust_remote_code=args.trust_remote_code,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            add_special_tokens=not args.no_add_special_tokens,
            skip_special_tokens=not args.no_skip_special_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            eos_token_id=args.eos_token_id,
            use_tokenizer_eos=not args.ignore_tokenizer_eos,
            runner_path=args.runner,
            expert_layout_path=prepared.experts_layout,
            resident_layout_path=prepared.resident_layout,
            cache_layout_path=prepared.decode_cache_layout,
            cache_file_path=prepared.decode_cache_file,
            **kwargs,
        )
    finally:
        if prepared_lock is not None:
            prepared_lock.close()
    applied_launch_profile = _applied_launch_profile_summary(args, prepared)
    if applied_launch_profile is not None:
        token_result = _attach_applied_profile_to_token_result(
            result.token_result,
            applied_launch_profile,
        )
        result = replace(
            result,
            token_result=token_result,
            applied_launch_profile=applied_launch_profile,
        )
    _require_actual_prepared_prefill_acceleration_if_requested(
        args,
        result.token_result,
    )
    if args.write_result is not None:
        _write_json_file_atomic(
            args.write_result,
            _prepared_text_generation_result_payload(
                prepared=prepared,
                max_new_tokens=args.max_new_tokens,
                launch_audit_path=getattr(args, "require_launch_audit", None),
                prefill_mla_kv_b_cache_dir=getattr(
                    args,
                    "prefill_mla_kv_b_cache_dir",
                    None,
                ),
                result=result,
            ),
        )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_text_generation_result(result)
    return 0


def _bench_prepared_token_ids(args: argparse.Namespace) -> int:
    prepared = load_prepared_manifest(args.prepared)
    _require_launch_profile_matches_prepared(args, prepared)
    _apply_prepared_memory_guard_defaults(args, prepared)
    if args.model_config is None:
        args.model_config = str(prepared.model_dir)
    validate_layout_model_config_sha256(
        model_config_path=args.model_config,
        expert_layout_path=prepared.experts_layout,
        resident_layout_path=prepared.resident_layout,
    )
    _require_prepared_glm_4bit_if_requested(args, prepared)
    _require_prepared_public_glm_5_2_shape_if_requested(args, prepared)
    _require_prepared_runtime_profile(
        prepared,
        require_memory_profile=_prepared_memory_profile_required_by_args(args),
    )
    _require_prefill_acceleration_if_requested(args)
    prompt_token_ids = _parse_token_ids(args.prompt_token_ids)
    if (
        not args.batch_prefill_prompt
        and not args.no_batch_prefill_prompt
        and len(prompt_token_ids) > 1
    ):
        args.batch_prefill_prompt = True
    if args.batch_prefill_prompt and not args.no_batch_prefill_prompt:
        _default_prepared_static_capacity(args)
    kwargs = _generation_kwargs(args)
    _require_prepared_prefill_acceleration_coverage_if_requested(
        args,
        prompt_token_count=len(prompt_token_ids),
        generation_overrides=kwargs,
    )
    _require_launch_audit_request_envelope(
        args,
        prepared,
        prompt_token_count=len(prompt_token_ids),
        max_new_tokens=args.max_new_tokens,
    )
    result = benchmark_prepared_token_ids(
        prepared.manifest_path,
        runner_path=args.runner,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        require_prepared_memory_profile=_prepared_memory_profile_required_by_args(args),
        require_glm_4bit=bool(args.require_glm_4bit),
        require_public_glm_5_2_shape=bool(args.require_public_glm_5_2_shape),
        compile_mpp_probe=bool(getattr(args, "compile_mpp_probe", False)),
        run_mpp_probe=bool(getattr(args, "run_mpp_probe", False)),
        run_mpsgraph_probe=bool(getattr(args, "run_mpsgraph_probe", False)),
        eos_token_id=args.eos_token_id,
        **kwargs,
    )
    applied_launch_profile = _applied_launch_profile_summary(args, prepared)
    if applied_launch_profile is not None:
        token_result = _attach_applied_profile_to_token_result(
            result.token_result,
            applied_launch_profile,
        )
        result = replace(
            result,
            token_result=token_result,
            applied_launch_profile=applied_launch_profile,
        )
    _require_actual_prepared_prefill_acceleration_if_requested(
        args,
        result.token_result,
    )
    if args.write_launch_profile is not None:
        _write_launch_profile_file(
            args.write_launch_profile,
            result.suggested_launch_profile,
        )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_benchmark_result(result)
    return 0


def _inspect_prepared(args: argparse.Namespace) -> int:
    config = _prepared_server_config_from_args(args)
    if (
        config.require_glm_4bit
        or config.require_public_glm_5_2_shape
        or config.require_prefill_acceleration
        or config.prefill_min_accelerated_flop_fraction > 0.0
    ):
        config = replace(
            config,
            require_glm_4bit=False,
            require_public_glm_5_2_shape=False,
            enforce_prefill_acceleration_probe=False,
        )
    app = PreparedGenerationApp(config)
    health = app.health()
    request_ok = True
    profile = health.get("prepared_runtime_profile")
    profile_ok = not (
        isinstance(profile, dict) and profile.get("profile_ok") is False
    )
    memory_profile_ok = True
    if _prepared_memory_profile_required_by_args(args):
        reason = prepared_memory_profile_failure_reason(app.state.prepared)
        memory_profile_ok = reason is None
        health["prepared_memory_profile_requirement"] = {
            "ok": memory_profile_ok,
            "error": reason,
        }
    if bool(getattr(args, "require_launch_audit", False)) or getattr(
        args,
        "write_launch_audit",
        None,
    ) is not None:
        health["prepared_context_budget_requirement"] = (
            _prepared_context_budget_requirement(
                health.get("prepared_storage"),
            )
        )
    glm_readiness = health.get("glm_4bit_readiness")
    glm_4bit_ok = not bool(getattr(args, "require_glm_4bit", False)) or (
        isinstance(glm_readiness, dict) and glm_readiness.get("ok") is True
    )
    public_glm_5_2_ok = not bool(
        getattr(args, "require_public_glm_5_2_shape", False)
    ) or (
        isinstance(glm_readiness, dict)
        and glm_readiness.get("ok") is True
        and glm_readiness.get("matches_public_glm_5_2_shape") is True
    )
    prefill_accel_ok = True
    if _prefill_acceleration_required(args):
        backend = health.get("prefill_backend")
        capability = backend.get("capability") if isinstance(backend, dict) else None
        gate = _prefill_acceleration_gate(
            configured_backend=getattr(args, "prefill_linear_backend", "auto"),
            mps_graph_runtime_available=(
                capability.get("mps_graph_runtime_available")
                if isinstance(capability, dict)
                else None
            ),
            mpp_runtime_available=(
                capability.get("mpp_runtime_available")
                if isinstance(capability, dict)
                else None
            ),
            mps_graph_probe_requested=(
                capability.get("mps_graph_probe_requested")
                if isinstance(capability, dict)
                else None
            ),
            mps_graph_probe_ran=(
                capability.get("mps_graph_probe_ran")
                if isinstance(capability, dict)
                else None
            ),
            mps_graph_probe_ok=(
                capability.get("mps_graph_probe_ok")
                if isinstance(capability, dict)
                else None
            ),
            selectable_backends=(
                tuple(capability.get("selectable_accelerated_prefill_backends") or ())
                if isinstance(capability, dict)
                else ()
            ),
            acceleration_runtimes=(
                tuple(capability.get("prefill_acceleration_runtimes") or ())
                if isinstance(capability, dict)
                else ()
            ),
        )
        health["prefill_acceleration_requirement"] = gate
        prefill_accel_ok = gate.get("ok") is True
    prompt_sources = [
        args.check_prompt_tokens is not None,
        args.check_prompt is not None,
        args.check_prompt_file is not None,
        args.check_chat_messages is not None,
        args.check_chat_messages_file is not None,
    ]
    if sum(1 for item in prompt_sources if item) > 1:
        request_ok = False
        health["request_check"] = {
            "ok": False,
            "prompt_token_count": args.check_prompt_tokens,
            "max_new_tokens": args.check_max_new_tokens,
            "error": (
                "--check-prompt-tokens, --check-prompt, --check-prompt-file, "
                "--check-chat-messages, and --check-chat-messages-file are "
                "mutually exclusive"
            ),
        }
    elif any(prompt_sources):
        payload = {
            "max_new_tokens": args.check_max_new_tokens,
            "logits_top_k": args.check_logits_top_k,
            "temperature": args.check_temperature,
            "top_p": args.check_top_p,
            "metal_final_logits": (
                args.check_metal_final_logits
                or bool(getattr(args, "metal_final_logits", False))
            ),
        }
        prompt_token_count = args.check_prompt_tokens
        request_metadata: dict[str, Any] = {
            "prompt_source": "tokens" if prompt_token_count is not None else None,
        }
        try:
            if prompt_token_count is None:
                tokenizer_path = args.tokenizer or str(app.state.prepared.model_dir)
                if (
                    args.check_chat_messages is not None
                    or args.check_chat_messages_file is not None
                ):
                    messages = _read_check_chat_messages_arg(
                        messages_json=args.check_chat_messages,
                        messages_file=args.check_chat_messages_file,
                        max_bytes=args.max_request_bytes,
                    )
                    rendered = render_chat_prompt(
                        tokenizer_path,
                        messages,
                        backend=args.tokenizer_backend,
                        trust_remote_code=args.trust_remote_code,
                        add_generation_prompt=(
                            not args.check_no_add_generation_prompt
                        ),
                    )
                    encoded = encode_prompt(
                        rendered.tokenizer_path,
                        rendered.text,
                        backend=rendered.tokenizer_backend or rendered.backend,
                        trust_remote_code=args.trust_remote_code,
                        add_special_tokens=False,
                    )
                    prompt_token_count = len(encoded.token_ids)
                    request_metadata = {
                        "prompt_source": (
                            "chat_file"
                            if args.check_chat_messages_file is not None
                            else "chat"
                        ),
                        "chat_template_backend": rendered.backend,
                        "tokenizer": {
                            "backend": encoded.backend,
                            "path": str(encoded.tokenizer_path),
                            "add_special_tokens": False,
                            "add_generation_prompt": (
                                not args.check_no_add_generation_prompt
                            ),
                        },
                    }
                else:
                    prompt = _read_check_prompt_arg(
                        prompt=args.check_prompt,
                        prompt_file=args.check_prompt_file,
                        max_bytes=args.max_request_bytes,
                    )
                    encoded = encode_prompt(
                        tokenizer_path,
                        prompt,
                        backend=args.tokenizer_backend,
                        trust_remote_code=args.trust_remote_code,
                        add_special_tokens=not args.check_no_add_special_tokens,
                    )
                    prompt_token_count = len(encoded.token_ids)
                    request_metadata = {
                        "prompt_source": (
                            "file" if args.check_prompt_file is not None else "text"
                        ),
                        "tokenizer": {
                            "backend": encoded.backend,
                            "path": str(encoded.tokenizer_path),
                            "add_special_tokens": (
                                not args.check_no_add_special_tokens
                            ),
                        },
                    }
            health["request_check"] = app.inspect_token_request(
                prompt_token_count=prompt_token_count,
                payload=payload,
                runtime_preflight=args.check_runtime_preflight,
            )
            health["request_check"].update(
                {key: value for key, value in request_metadata.items() if value}
            )
            health["request_launch_profile"] = _request_launch_profile_from_health(
                health,
                health["request_check"],
            )
        except (
            PreparedServerError,
            TextGenerationError,
            TokenGeneratorError,
            TokenizerError,
        ) as exc:
            request_ok = False
            health["request_check"] = {
                "ok": False,
                "prompt_token_count": prompt_token_count,
                "max_new_tokens": args.check_max_new_tokens,
                "error": str(exc),
            }
            if isinstance(exc, PreparedRequestCheckError):
                health["request_check"].update(exc.payload)
            health["request_check"].update(
                {key: value for key, value in request_metadata.items() if value}
            )
    launch_audit_ok = True
    write_launch_audit = getattr(args, "write_launch_audit", None)
    if bool(getattr(args, "require_launch_audit", False)) or write_launch_audit is not None:
        launch_audit = _launch_audit_from_health(args, health)
        health["launch_audit"] = launch_audit
        launch_audit_ok = launch_audit.get("ok") is True
    if args.write_launch_profile is not None:
        request_profile = health.get("request_launch_profile")
        profile = (
            request_profile
            if isinstance(request_profile, dict)
            else health.get("suggested_launch_profile")
        )
        _write_launch_profile_file(args.write_launch_profile, profile)
    if write_launch_audit is not None:
        _write_launch_audit_file(write_launch_audit, health)
    if args.json:
        print(json.dumps(health, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_prepared_health(health)
    return (
        0
        if (
            request_ok
            and profile_ok
            and memory_profile_ok
            and glm_4bit_ok
            and public_glm_5_2_ok
            and prefill_accel_ok
            and launch_audit_ok
        )
        else 1
    )


def _serve_prepared(args: argparse.Namespace) -> int:
    prepared = load_prepared_manifest(args.prepared)
    _require_launch_profile_matches_prepared(args, prepared)
    launch_audit_envelope = _require_launch_audit_server_envelope(args, prepared)
    _require_prepared_glm_4bit_if_requested(args, prepared)
    _require_prepared_public_glm_5_2_shape_if_requested(args, prepared)
    _require_prepared_runtime_profile(
        prepared,
        require_memory_profile=_prepared_memory_profile_required_by_args(args),
    )
    _require_prefill_acceleration_if_requested(args)
    _serve_prepared_ssd_read_speed_check(args, prepared)
    config = _prepared_server_config_from_args(args)
    if launch_audit_envelope is not None:
        config = replace(config, launch_audit_envelope=launch_audit_envelope)
    run_prepared_server(config)
    return 0


def _export_mlx_baseline(args: argparse.Namespace) -> int:
    output = args.output or str(Path(args.model) / "baseline_mlx")
    max_model_load_bytes = int(args.max_model_load_gib * 1024**3)
    if not args.execute:
        report = preflight_mlx_baseline(
            args.model,
            output,
            max_model_load_bytes=max_model_load_bytes,
            allow_unknown_size=args.allow_unknown_size,
        )
        if args.json:
            print(json.dumps(report, default=_json_default, indent=2, sort_keys=True))
        else:
            _print_mlx_preflight(report)
        return 0

    result = export_mlx_baseline(
        args.model,
        output,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_tensor_bytes=int(args.max_tensor_mib * 1024**2),
        max_model_load_bytes=max_model_load_bytes,
        allow_unknown_size=args.allow_unknown_size,
    )
    if args.json:
        print(json.dumps(result, default=_json_default, indent=2, sort_keys=True))
    else:
        _print_mlx_result(result)
    return 0


def _add_expert_read_advise_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expert-read-advise-merge-gap-kib",
        type=int,
        default=0,
        help="ask the runner to advise merged expert read ranges across gaps up to this size",
    )
    parser.add_argument(
        "--expert-read-advise-align-kib",
        type=int,
        default=0,
        help="ask the runner to align expert read-advice ranges to this size; 0 disables hints",
    )


def _parse_prefill_prompt_chunk_tokens(value: str) -> int:
    if value.lower() == "auto":
        return 0
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "prefill prompt chunk tokens must be a positive integer or 'auto'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "prefill prompt chunk tokens must be positive; use 'auto' for automatic sizing"
        )
    return parsed


def _parse_positive_int_csv(value: str) -> tuple[int, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated positive integer list")
    parsed: list[int] = []
    for item in items:
        try:
            number = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "expected a comma-separated positive integer list"
            ) from exc
        if number <= 0:
            raise argparse.ArgumentTypeError("integer list values must be positive")
        parsed.append(number)
    return tuple(parsed)


def _parse_matrix_shape_csv(value: str) -> tuple[tuple[int, int], ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected comma-separated INxOUT shapes")
    shapes: list[tuple[int, int]] = []
    for item in items:
        parts = item.lower().split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("matrix shapes must look like INxOUT")
        try:
            in_dim = int(parts[0])
            out_dim = int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("matrix shapes must look like INxOUT") from exc
        if in_dim <= 0 or out_dim <= 0:
            raise argparse.ArgumentTypeError("matrix shape dimensions must be positive")
        shapes.append((in_dim, out_dim))
    return tuple(shapes)


def _parse_moe_token_block(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "MoE token block must be a positive integer or 'auto'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "MoE token block must be positive; use 'auto' for automatic sizing"
        )
    return parsed


def _add_batch_prefill_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-missing-dsa-indexer",
        action="store_true",
        help=(
            "debug only: bypass the DSA prompt-prefill schedule guard when "
            "dsa_index cache segments exist"
        ),
    )
    parser.add_argument(
        "--batch-prefill-prompt",
        action="store_true",
        help="use chunked batch prefill for prompt tokens before single-token decode",
    )
    parser.add_argument(
        "--no-batch-prefill-prompt",
        action="store_true",
        help="prepared commands only: disable the automatic batch prompt prefill default",
    )
    parser.add_argument(
        "--prefill-prompt-chunk-tokens",
        type=_parse_prefill_prompt_chunk_tokens,
        default=0,
        metavar="N|auto",
        help="prompt tokens per batch prefill chunk; default auto sizes from safety caps",
    )
    parser.add_argument("--prefill-max-prompt-batch-mib", type=float, default=1024.0)
    parser.add_argument("--prefill-max-cache-write-mib", type=float, default=4096.0)
    parser.add_argument(
        "--prefill-mla-kv-b-cache-dir",
        default=None,
        help=(
            "optional directory for per-layer f32 MLA kv_b caches materialized "
            "from absorbed attention aliases during batch prompt prefill"
        ),
    )
    parser.add_argument(
        "--prefill-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during batch prompt prefill",
    )
    parser.add_argument(
        "--decode-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during single-token decode",
    )
    parser.add_argument("--disable-dsa-indexer", action="store_true")
    parser.add_argument("--dsa-index-topk", type=int, default=None)
    parser.add_argument("--dsa-index-n-heads", type=int, default=None)
    parser.add_argument("--dsa-index-head-dim", type=int, default=None)
    parser.add_argument("--dsa-qk-rope-dim", type=int, default=None)
    parser.add_argument("--dsa-rope-interleave", action="store_true")
    parser.add_argument("--dsa-layer-norm-eps", type=float, default=1e-6)
    parser.add_argument("--prefill-expert-stage-merge-gap-kib", type=float, default=0.0)
    parser.add_argument("--prefill-expert-stage-align-kib", type=float, default=4.0)
    parser.add_argument("--prefill-max-stage-mib", type=float, default=4096.0)
    parser.add_argument("--prefill-max-compact-stage-mib", type=float, default=4096.0)
    parser.add_argument(
        "--prefill-max-stage-raw-ranges",
        type=int,
        default=0,
        help="reject prompt prefill stage plans with more raw expert read ranges; 0 disables",
    )
    parser.add_argument(
        "--prefill-max-stage-coalesced-ranges",
        type=int,
        default=0,
        help="reject prompt prefill stage plans with more coalesced expert read ranges; 0 disables",
    )
    parser.add_argument(
        "--prefill-expert-stage-tiling",
        action="store_true",
        help="split prompt prefill routed experts into bounded stage/compact tiles",
    )
    parser.add_argument(
        "--prefill-persistent-moe-plan-server",
        action="store_true",
        help=(
            "keep one Metal routed-MoE plan runner alive across prompt prefill "
            "routed layers and expert-stage tiles"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-resident-linear-server",
        action="store_true",
        help=(
            "keep one Metal resident-linear runner alive across prompt prefill "
            "resident batch-linear requests"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-attention-projection-server",
        action="store_true",
        help=(
            "keep one Metal attention-projection runner alive across prompt "
            "prefill fused projection requests"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-attention-output-server",
        action="store_true",
        help=(
            "keep one Metal attention-output runner alive across prompt "
            "prefill fused output-projection requests"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-shared-expert-server",
        action="store_true",
        help=(
            "keep one Metal shared-expert runner alive across prompt "
            "prefill fused shared-expert requests"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-rope-split-server",
        action="store_true",
        help=(
            "keep one Metal RoPE split runner alive across prompt prefill "
            "fused RoPE split requests"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-mla-attention-server",
        action="store_true",
        help=(
            "keep one Metal MLA attention runner alive across prompt prefill "
            "batch MLA attention requests"
        ),
    )
    parser.add_argument(
        "--prefill-persistent-rmsnorm-server",
        action="store_true",
        help=(
            "keep one Metal RMSNorm runner alive across prompt prefill "
            "resident RMSNorm batch requests"
        ),
    )
    parser.add_argument("--prefill-copy-chunk-mib", type=float, default=8.0)
    parser.add_argument(
        "--prefill-moe-token-block",
        type=_parse_moe_token_block,
        default="auto",
        metavar="N|auto",
        help="MoE token block for batch prompt prefill; default auto fits runner scratch",
    )
    parser.add_argument(
        "--prefill-moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
        help=(
            "control staged MoE output accumulation during batch prompt prefill; "
            "env preserves LARGERLM_MOE_BATCH_ACCUMULATOR, file/memory pin it"
        ),
    )
    parser.add_argument(
        "--prefill-static-capacity-per-expert",
        default=None,
        metavar="N|auto",
        help=(
            "write LLMSCAP1 binary routes for batch prompt MoE; "
            "auto uses the prompt chunk size"
        ),
    )
    parser.add_argument(
        "--prefill-allow-static-capacity-overflow",
        action="store_true",
        help="allow prompt static-capacity route artifacts to contain overflow records",
    )
    parser.add_argument(
        "--prefill-stage-disk-margin-mib",
        type=float,
        default=0.0,
        help="extra free disk margin required before writing prompt prefill work/stage files",
    )
    parser.add_argument(
        "--prefill-max-routed-read-amplification",
        type=float,
        default=0.0,
        help=(
            "reject batch prefill when prompt chunking would exceed this routed "
            "expert read amplification; <=0 disables the guard"
        ),
    )
    parser.add_argument(
        "--prefill-max-routed-read-gib",
        type=float,
        default=0.0,
        help=(
            "reject batch prefill when planned routed expert SSD reads exceed "
            "this GiB cap; <=0 disables the guard"
        ),
    )
    parser.add_argument(
        "--prefill-ssd-read-gib-s",
        dest="prefill_ssd_read_gib_per_second",
        type=float,
        default=0.0,
        help=(
            "measured SSD read throughput used to estimate routed expert read "
            "seconds; <=0 disables time estimates"
        ),
    )
    parser.add_argument(
        "--prefill-max-routed-read-seconds",
        type=float,
        default=0.0,
        help=(
            "reject batch prefill when estimated routed expert SSD read time "
            "exceeds this cap; requires --prefill-ssd-read-gib-s"
        ),
    )
    parser.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        help="minimum prompt batch tokens before auto uses MPSGraph resident GEMMs",
    )
    parser.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
        help="minimum resident matrix dimension before auto uses MPSGraph",
    )
    parser.add_argument(
        "--prefill-router-hybrid-margin-threshold",
        type=float,
        default=0.0,
        help=(
            "when >0 and auto selects MPSGraph for router gates, first run the "
            "custom-Metal router and use it when the min effective margin "
            "exceeds this threshold; otherwise fall back to MPSGraph"
        ),
    )


def _add_generation_memory_guard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-live-working-set-mib",
        type=float,
        default=None,
        help=(
            "reject generation when estimated live runner/prompt/logits working "
            "set exceeds this cap; <=0 disables this deterministic guard; "
            "prepared commands default to the prepared manifest recommendation, "
            "otherwise 8192"
        ),
    )
    parser.add_argument(
        "--min-free-unified-memory-gib",
        type=float,
        default=None,
        help=(
            "optionally require this much additional available unified memory "
            "at admission and before prompt/decode stages; 0 disables live "
            "system probing; prepared commands default to the prepared manifest "
            "recommendation, otherwise 0"
        ),
    )
    parser.add_argument(
        "--decode-max-routed-read-gib-per-token",
        type=float,
        default=0.0,
        help=(
            "reject generation when estimated decode routed expert SSD reads "
            "exceed this GiB/token cap; <=0 disables the guard"
        ),
    )
    parser.add_argument(
        "--decode-max-routed-read-seconds-per-token",
        type=float,
        default=0.0,
        help=(
            "reject generation when estimated decode routed expert SSD read time "
            "exceeds this seconds/token cap; requires --prefill-ssd-read-gib-s "
            "or a prepared SSD read-speed default"
        ),
    )


def _add_selected_replay_ssd_read_check_args(
    parser: argparse.ArgumentParser,
    *,
    subject: str = "selected replay",
) -> None:
    parser.add_argument(
        "--check-ssd-read-speed",
        action="store_true",
        default=None,
        help=(
            f"before a {subject}, run a bounded sequential read benchmark "
            "against the prepared manifest's cold-read benchmark file"
        ),
    )
    parser.add_argument(
        "--ssd-read-speed-min-ratio",
        type=float,
        default=None,
        help=(
            "minimum current/baseline GiB/s ratio required by "
            "--check-ssd-read-speed"
        ),
    )
    parser.add_argument(
        "--ssd-read-speed-bytes-mib",
        type=float,
        default=None,
        help="MiB to read for --check-ssd-read-speed",
    )
    parser.add_argument(
        "--ssd-read-speed-chunk-mib",
        type=float,
        default=None,
        help="pread chunk MiB for --check-ssd-read-speed",
    )
    parser.add_argument(
        "--ssd-read-speed-max-chunk-mib",
        type=float,
        default=None,
        help="safety cap for --ssd-read-speed-chunk-mib",
    )


def _add_prepared_memory_profile_requirement_arg(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--require-prepared-memory-profile",
        action="store_true",
        help=(
            "fail unless the prepared manifest records the full prepare-time "
            "memory envelope and the current runtime profile can be verified"
        ),
    )


def _add_prepared_glm_4bit_requirement_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-glm-4bit",
        action="store_true",
        help=(
            "return non-zero unless the prepared expert layout is a "
            "config-consistent affine-int4 GLM MoE package"
        ),
    )
    parser.add_argument(
        "--require-public-glm-5-2-shape",
        action="store_true",
        help=(
            "return non-zero unless GLM 4bit readiness passes and the prepared "
            "config matches the public GLM-5.2 model shape"
        ),
    )


def _add_prepared_ssd_read_default_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-prepared-ssd-read-default",
        action="store_true",
        help=(
            "do not inherit prepare_cold_read_gib_per_second from the prepared "
            "manifest as the default --prefill-ssd-read-gib-s"
        ),
    )


def _add_prefill_acceleration_requirement_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-prefill-acceleration",
        action="store_true",
        help=(
            "return non-zero unless the selected prepared prefill backend has "
            "a selectable accelerated runtime and, for MPSGraph, a passing "
            "--run-mpsgraph-probe check"
        ),
    )
    parser.add_argument(
        "--prefill-min-accelerated-flop-fraction",
        type=float,
        default=0.0,
        help=(
            "require at least this fraction of estimated prefill linear FLOPs to resolve "
            "to an accelerated backend; values >0 imply --require-prefill-acceleration"
        ),
    )
    parser.add_argument(
        "--allow-router-gate-only-prefill-acceleration",
        action="store_true",
        help=(
            "allow --require-prefill-acceleration to be satisfied when all "
            "accelerated resident prefill work is MoE router-gate projection; "
            "intended only for explicit routing-drift experiments"
        ),
    )
    parser.add_argument(
        "--allow-non-accelerated-prefill-launch-audit",
        action="store_true",
        help=(
            "allow launch-audit generation for explicit custom-metal or other "
            "non-accelerated prefill backend experiments; memory, SSD, GLM, "
            "request, and runtime-preflight audit checks still apply"
        ),
    )


def _add_prefill_backend_probe_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compile-mpp-probe",
        action="store_true",
        help=(
            "run the local MPP tensor-ops compile probe during prepared backend "
            "inspection; useful for M5/Metal 4 bring-up"
        ),
    )
    parser.add_argument(
        "--run-mpp-probe",
        action="store_true",
        help=(
            "run a tiny MPP tensor-ops matmul during prepared backend inspection; "
            "useful for M5/Metal 4 execution bring-up without loading weights"
        ),
    )
    parser.add_argument(
        "--run-mpsgraph-probe",
        action="store_true",
        help=(
            "run a tiny MPSGraph matmul during backend inspection; useful for "
            "verifying M5 prefill acceleration without loading model weights"
        ),
    )
    parser.add_argument(
        "--prefill-backend-probe-timeout-seconds",
        type=float,
        default=DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
        help=(
            "timeout for prepared Metal/MPSGraph/MPP backend probes; increase "
            "on first-run MPSGraph/Metal compiler cold starts"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="largerlm")
    sub = parser.add_subparsers(dest="command")

    plan = sub.add_parser("plan", help="estimate memory and I/O for an MoE checkpoint")
    plan.add_argument("model", help="checkpoint directory or config.json")
    plan.add_argument("--quant-bits", type=int, default=4, choices=(2, 3, 4, 8))
    plan.add_argument("--group-size", type=int, default=64)
    plan.add_argument("--scan-safetensors", action="store_true")
    plan.add_argument("--unified-memory-gib", type=float, default=None)
    plan.add_argument(
        "--system-reserve-gib",
        type=float,
        default=None,
        help="override automatic live unified-memory reserve in GiB",
    )
    plan.add_argument("--runtime-buffer-gib", type=float, default=8.0)
    plan.add_argument("--page-cache-fraction", type=float, default=0.60)
    plan.add_argument("--cold-read-gib-s", type=float, default=None)
    plan.add_argument("--max-context-tokens", type=int, default=None)
    plan.add_argument("--max-cache-gib", type=float, default=None)
    plan.add_argument(
        "--write-launch-profile",
        default=None,
        help="write suggested memory and decode guard launch profile JSON to this path",
    )
    plan.add_argument(
        "--write-prepare-flags",
        default=None,
        help="write suggested prepare-glm argv flags JSON to this path",
    )
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(func=_plan)

    prefill_plan = sub.add_parser(
        "prefill-plan",
        help="estimate GLM prefill GEMM shapes and MPP tensor-op candidates",
    )
    prefill_plan.add_argument("model", help="checkpoint directory or config.json")
    prefill_plan.add_argument("--prompt-tokens", type=int, required=True)
    prefill_plan.add_argument("--dtype-bits", type=int, default=16, choices=(16, 32))
    prefill_plan.add_argument("--expert-bits", type=int, default=4, choices=(2, 3, 4, 8))
    prefill_plan.add_argument("--group-size", type=int, default=64)
    prefill_plan.add_argument("--mpp-min-tokens", type=int, default=128)
    prefill_plan.add_argument("--simdgroup-tile-m", type=int, default=32)
    prefill_plan.add_argument("--simdgroup-tile-n", type=int, default=32)
    prefill_plan.add_argument("--simdgroups-m", type=int, default=2)
    prefill_plan.add_argument("--simdgroups-n", type=int, default=2)
    prefill_plan.add_argument("--k-tile", type=int, default=128)
    prefill_plan.add_argument(
        "--max-prefill-activation-mib",
        type=float,
        default=None,
        help="recommend prompt chunking to keep per-op prefill activations under this cap",
    )
    prefill_plan.add_argument(
        "--max-runner-scratch-mib",
        type=float,
        default=None,
        help="check staged MoE runner token-block scratch against this cap",
    )
    prefill_plan.add_argument(
        "--expert-stage-align-kib",
        type=float,
        default=4.0,
        help="expert stage read alignment in KiB for stage-temp guard suggestions",
    )
    prefill_plan.add_argument(
        "--prefill-static-capacity-per-expert",
        default="auto",
        metavar="N|auto|none",
        help="include the prompt MoE static-capacity route mode in the launch profile",
    )
    prefill_plan.add_argument(
        "--ssd-read-gib-s",
        type=float,
        default=None,
        help="estimate routed expert read seconds using measured sequential SSD GiB/s",
    )
    prefill_plan.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="auto",
        help="include this resident GEMM backend policy in the launch profile",
    )
    prefill_plan.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        help="include this MPSGraph auto batch-token threshold in the launch profile",
    )
    prefill_plan.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
        help="include this MPSGraph auto matrix-dimension threshold in the launch profile",
    )
    prefill_plan.add_argument(
        "--prefill-min-accelerated-flop-fraction",
        type=float,
        default=0.0,
        help="include this accelerated resident-GEMM FLOP coverage gate in the launch profile",
    )
    prefill_plan.add_argument(
        "--require-prefill-acceleration",
        action="store_true",
        help="include the prefill acceleration requirement in the launch profile",
    )
    prefill_plan.add_argument(
        "--require-public-glm-5-2-shape",
        action="store_true",
        help="fail unless config.json matches the public GLM-5.2 architecture shape",
    )
    prefill_plan.add_argument(
        "--inspect-backend",
        action="store_true",
        help="attach local Metal 4/MPP prefill backend capability to the plan",
    )
    prefill_plan.add_argument("--sdk-path", default=None)
    prefill_plan.add_argument("--probe-binary", default=None)
    prefill_plan.add_argument("--no-host-probe", action="store_true")
    prefill_plan.add_argument("--compile-mpp-probe", action="store_true")
    prefill_plan.add_argument("--run-mpp-probe", action="store_true")
    prefill_plan.add_argument("--run-mpsgraph-probe", action="store_true")
    prefill_plan.add_argument("--probe-timeout-seconds", type=float, default=5.0)
    prefill_plan.add_argument(
        "--write-launch-profile",
        default=None,
        help="write the suggested prefill launch profile JSON to this path",
    )
    prefill_plan.add_argument(
        "--write-calibration-flags",
        default=None,
        help="write the suggested prefill-linear-calibrate argv flags JSON to this path",
    )
    prefill_plan.add_argument("--json", action="store_true")
    prefill_plan.set_defaults(func=_prefill_plan)

    prefill_plan_calibrate = sub.add_parser(
        "prefill-plan-calibrate",
        help="run bounded calibration using GLM shapes from prefill-plan",
    )
    prefill_plan_calibrate.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_plan_calibrate.add_argument("model", help="checkpoint directory or config.json")
    prefill_plan_calibrate.add_argument("--prompt-tokens", type=int, required=True)
    prefill_plan_calibrate.add_argument("--dtype-bits", type=int, default=16, choices=(16, 32))
    prefill_plan_calibrate.add_argument("--expert-bits", type=int, default=4, choices=(2, 3, 4, 8))
    prefill_plan_calibrate.add_argument("--group-size", type=int, default=64)
    prefill_plan_calibrate.add_argument("--mpp-min-tokens", type=int, default=128)
    prefill_plan_calibrate.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="auto",
        help="include this resident GEMM backend policy in the merged launch profile",
    )
    prefill_plan_calibrate.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        help="fallback MPSGraph auto batch-token threshold when calibration has no safe threshold",
    )
    prefill_plan_calibrate.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
        help="fallback MPSGraph auto matrix-dimension threshold when calibration has no safe threshold",
    )
    prefill_plan_calibrate.add_argument(
        "--prefill-min-accelerated-flop-fraction",
        type=float,
        default=0.0,
        help="preserve this accelerated resident-GEMM FLOP coverage gate in the merged profile",
    )
    prefill_plan_calibrate.add_argument(
        "--require-prefill-acceleration",
        action="store_true",
        help="preserve the prefill acceleration requirement in the merged profile",
    )
    prefill_plan_calibrate.add_argument(
        "--require-public-glm-5-2-shape",
        action="store_true",
        help="fail unless config.json matches the public GLM-5.2 architecture shape",
    )
    prefill_plan_calibrate.add_argument(
        "--expert-stage-align-kib",
        type=float,
        default=4.0,
        help="expert stage read alignment in KiB for merged stage-temp guards",
    )
    prefill_plan_calibrate.add_argument(
        "--prefill-static-capacity-per-expert",
        default="auto",
        metavar="N|auto|none",
        help="include the prompt MoE static-capacity route mode in the merged profile",
    )
    prefill_plan_calibrate.add_argument(
        "--ssd-read-gib-s",
        type=float,
        default=None,
        help="estimate routed expert read seconds and include SSD-time guards in the merged profile",
    )
    prefill_plan_calibrate.add_argument(
        "--max-prefill-activation-mib",
        type=float,
        default=None,
        help="use the planner's chunked prompt size when deriving calibration batch tokens",
    )
    prefill_plan_calibrate.add_argument(
        "--plan-max-runner-scratch-mib",
        type=float,
        default=None,
        help="optional staged-MoE scratch cap passed only to prefill-plan",
    )
    prefill_plan_calibrate.add_argument(
        "--matrix-dtype",
        choices=("F32", "BF16"),
        default="F32",
        help="dtype for synthesized resident GEMM calibration matrices",
    )
    prefill_plan_calibrate.add_argument("--repeats", type=int, default=1)
    prefill_plan_calibrate.add_argument(
        "--min-mpsgraph-speedup",
        type=float,
        default=1.0,
        help="minimum custom/MPSGraph elapsed ratio required for auto-policy coverage",
    )
    prefill_plan_calibrate.add_argument(
        "--max-calibration-case-mib",
        type=int,
        default=None,
        help="override the planner-derived calibration case cap in MiB",
    )
    prefill_plan_calibrate.add_argument(
        "--max-resident-matrix-mib",
        type=int,
        default=None,
        help="override the planner-derived resident matrix cap in MiB",
    )
    prefill_plan_calibrate.add_argument(
        "--max-runner-scratch-mib",
        type=int,
        default=None,
        help="override the planner-derived runner scratch cap in MiB",
    )
    prefill_plan_calibrate.add_argument(
        "--max-auto-calibration-case-mib",
        type=int,
        default=2048,
        help="fail before running if the planner-derived case cap exceeds this MiB limit",
    )
    prefill_plan_calibrate.add_argument(
        "--max-auto-resident-matrix-mib",
        type=int,
        default=1024,
        help="fail before running if the planner-derived matrix cap exceeds this MiB limit",
    )
    prefill_plan_calibrate.add_argument(
        "--max-auto-runner-scratch-mib",
        type=int,
        default=2048,
        help="fail before running if the planner-derived scratch cap exceeds this MiB limit",
    )
    prefill_plan_calibrate.add_argument(
        "--max-calibration-work-dir-mib",
        type=int,
        default=8192,
        help="fail before running if planned calibration temp files exceed this MiB limit",
    )
    prefill_plan_calibrate.add_argument(
        "--calibration-work-dir-free-margin-mib",
        type=int,
        default=512,
        help="require this many free MiB beyond estimated calibration temp files",
    )
    prefill_plan_calibrate.add_argument("--work-dir", default=None)
    prefill_plan_calibrate.add_argument("--keep-work-dir", action="store_true")
    prefill_plan_calibrate.add_argument("--echo-runner-output", action="store_true")
    prefill_plan_calibrate.add_argument(
        "--write-calibration-flags",
        default=None,
        help="write the applied prefill-linear-calibrate argv flags JSON",
    )
    prefill_plan_calibrate.add_argument(
        "--write-launch-profile",
        default=None,
        help="write the merged prefill launch profile JSON",
    )
    prefill_plan_calibrate.add_argument("--json", action="store_true")
    prefill_plan_calibrate.set_defaults(func=_prefill_plan_calibrate)

    disk_read = sub.add_parser(
        "disk-read-benchmark",
        help="measure bounded sequential pread throughput for an existing file",
    )
    disk_read.add_argument("path", help="file to read, such as a packed expert layer")
    disk_read.add_argument(
        "--bytes-mib",
        type=float,
        default=None,
        help="maximum MiB to read; defaults to the rest of the file",
    )
    disk_read.add_argument(
        "--chunk-mib",
        type=float,
        default=8.0,
        help="bounded pread chunk size in MiB",
    )
    disk_read.add_argument(
        "--max-chunk-mib",
        type=float,
        default=MAX_SEQUENTIAL_READ_CHUNK_BYTES / 1024**2,
        help="maximum single pread chunk MiB accepted by the safety guard",
    )
    disk_read.add_argument(
        "--offset-mib",
        type=float,
        default=0.0,
        help="starting offset in MiB",
    )
    disk_read.add_argument("--json", action="store_true")
    disk_read.set_defaults(func=_disk_read_benchmark)

    prefill_backend = sub.add_parser(
        "prefill-backend",
        help="inspect local Metal 4/MPP prefill backend capability",
    )
    prefill_backend.add_argument("--sdk-path", default=None)
    prefill_backend.add_argument("--probe-binary", default=None)
    prefill_backend.add_argument("--no-host-probe", action="store_true")
    prefill_backend.add_argument("--compile-mpp-probe", action="store_true")
    prefill_backend.add_argument("--run-mpp-probe", action="store_true")
    prefill_backend.add_argument("--run-mpsgraph-probe", action="store_true")
    prefill_backend.add_argument("--probe-timeout-seconds", type=float, default=5.0)
    prefill_backend.add_argument(
        "--write-report",
        default=None,
        help="atomically write the prefill backend capability JSON to this path",
    )
    prefill_backend.add_argument("--json", action="store_true")
    prefill_backend.set_defaults(func=_prefill_backend)

    prefill_linear = sub.add_parser(
        "prefill-linear-batch",
        help="run one bounded resident projection over a prompt-token batch",
    )
    prefill_linear.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_linear.add_argument("resident_layout", help="resident/layout.json")
    prefill_linear.add_argument("--layer", type=int, required=True)
    prefill_linear.add_argument("--tensor-suffix", required=True)
    prefill_linear.add_argument("--input-f32", required=True)
    prefill_linear.add_argument("--output-f32", required=True)
    prefill_linear.add_argument("--batch-tokens", type=int, required=True)
    prefill_linear.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_linear.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_linear.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_linear.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        help="minimum batch tokens before auto uses MPSGraph",
    )
    prefill_linear.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
        help="minimum matrix dimension before auto uses MPSGraph",
    )
    prefill_linear.add_argument("--quiet-runner", action="store_true")
    prefill_linear.add_argument("--json", action="store_true")
    prefill_linear.set_defaults(func=_prefill_linear_batch)

    prefill_linear_calibrate = sub.add_parser(
        "prefill-linear-calibrate",
        help=(
            "run a bounded custom-Metal, MPSGraph, and MPSMatrix resident GEMM "
            "calibration sweep"
        ),
    )
    prefill_linear_calibrate.add_argument(
        "runner",
        help="path to metal/largerlm-runner",
    )
    prefill_linear_calibrate.add_argument(
        "--batch-tokens",
        type=_parse_positive_int_csv,
        default=(32, 64, 128),
        help="comma-separated prompt-token batch sizes to sweep",
    )
    prefill_linear_calibrate.add_argument(
        "--matrix-dims",
        type=_parse_positive_int_csv,
        default=(16, 32, 64),
        help="comma-separated square resident GEMM dimensions to sweep",
    )
    prefill_linear_calibrate.add_argument(
        "--matrix-shapes",
        type=_parse_matrix_shape_csv,
        default=None,
        help=(
            "comma-separated INxOUT resident GEMM shapes to sweep; overrides "
            "--matrix-dims"
        ),
    )
    prefill_linear_calibrate.add_argument(
        "--matrix-dtype",
        choices=("F32", "BF16"),
        default="F32",
        help="dtype for synthesized resident GEMM calibration matrices",
    )
    prefill_linear_calibrate.add_argument("--repeats", type=int, default=1)
    prefill_linear_calibrate.add_argument(
        "--min-mpsgraph-speedup",
        type=float,
        default=1.0,
        help="minimum custom/MPSGraph elapsed ratio required for auto-policy coverage",
    )
    prefill_linear_calibrate.add_argument(
        "--max-calibration-case-mib",
        type=int,
        default=64,
        help="maximum synthesized matrix+input+output bytes per case in MiB",
    )
    prefill_linear_calibrate.add_argument(
        "--max-resident-matrix-mib",
        type=int,
        default=64,
    )
    prefill_linear_calibrate.add_argument(
        "--max-runner-scratch-mib",
        type=int,
        default=64,
    )
    prefill_linear_calibrate.add_argument(
        "--max-calibration-work-dir-mib",
        type=int,
        default=8192,
        help="maximum synthesized calibration temp files in MiB",
    )
    prefill_linear_calibrate.add_argument(
        "--calibration-work-dir-free-margin-mib",
        type=int,
        default=512,
        help="require this many free MiB beyond estimated calibration temp files",
    )
    prefill_linear_calibrate.add_argument("--work-dir", default=None)
    prefill_linear_calibrate.add_argument("--keep-work-dir", action="store_true")
    prefill_linear_calibrate.add_argument("--echo-runner-output", action="store_true")
    prefill_linear_calibrate.add_argument(
        "--write-launch-profile",
        default=None,
        help="write the suggested prefill runtime-policy launch profile JSON",
    )
    prefill_linear_calibrate.add_argument("--json", action="store_true")
    prefill_linear_calibrate.set_defaults(func=_prefill_linear_calibrate)

    prefill_rmsnorm = sub.add_parser(
        "prefill-rmsnorm-batch",
        help="run one bounded resident RMSNorm over a prompt-token batch",
    )
    prefill_rmsnorm.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_rmsnorm.add_argument("resident_layout", help="resident/layout.json")
    prefill_rmsnorm.add_argument("--layer", type=int, required=True)
    prefill_rmsnorm.add_argument("--norm-suffix", required=True)
    prefill_rmsnorm.add_argument("--input-f32", required=True)
    prefill_rmsnorm.add_argument("--output-f32", required=True)
    prefill_rmsnorm.add_argument("--batch-tokens", type=int, required=True)
    prefill_rmsnorm.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_rmsnorm.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_rmsnorm.add_argument("--quiet-runner", action="store_true")
    prefill_rmsnorm.add_argument("--json", action="store_true")
    prefill_rmsnorm.set_defaults(func=_prefill_rmsnorm_batch)

    prefill_prefix = sub.add_parser(
        "prefill-attention-prefix",
        help="run bounded GLM input RMSNorm plus q_a/kv_a resident projections",
    )
    prefill_prefix.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_prefix.add_argument("resident_layout", help="resident/layout.json")
    prefill_prefix.add_argument("--layer", type=int, required=True)
    prefill_prefix.add_argument("--input-f32", required=True)
    prefill_prefix.add_argument("--output-dir", required=True)
    prefill_prefix.add_argument("--batch-tokens", type=int, required=True)
    prefill_prefix.add_argument(
        "--norm-suffix",
        default=".input_layernorm.weight",
        help="resident vector suffix for the layer input RMSNorm",
    )
    prefill_prefix.add_argument(
        "--q-a-suffix",
        default=".self_attn.q_a_proj.weight",
        help="resident matrix suffix for q_a_proj",
    )
    prefill_prefix.add_argument(
        "--kv-a-suffix",
        default=".self_attn.kv_a_proj_with_mqa.weight",
        help="resident matrix suffix for kv_a_proj_with_mqa",
    )
    prefill_prefix.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_prefix.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_prefix.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_prefix.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_prefix.add_argument("--quiet-runner", action="store_true")
    prefill_prefix.add_argument("--json", action="store_true")
    prefill_prefix.set_defaults(func=_prefill_attention_prefix)

    prefill_projections = sub.add_parser(
        "prefill-attention-projections",
        help="run bounded GLM batch attention projections through q_b and kv_b",
    )
    prefill_projections.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_projections.add_argument("resident_layout", help="resident/layout.json")
    prefill_projections.add_argument("--layer", type=int, required=True)
    prefill_projections.add_argument("--input-f32", required=True)
    prefill_projections.add_argument("--output-dir", required=True)
    prefill_projections.add_argument("--batch-tokens", type=int, required=True)
    prefill_projections.add_argument(
        "--norm-suffix",
        default=".input_layernorm.weight",
        help="resident vector suffix for the layer input RMSNorm",
    )
    prefill_projections.add_argument(
        "--q-a-suffix",
        default=".self_attn.q_a_proj.weight",
        help="resident matrix suffix for q_a_proj",
    )
    prefill_projections.add_argument(
        "--q-a-norm-suffix",
        default=".self_attn.q_a_layernorm.weight",
        help="resident vector suffix for q_a_layernorm",
    )
    prefill_projections.add_argument(
        "--q-b-suffix",
        default=".self_attn.q_b_proj.weight",
        help="resident matrix suffix for q_b_proj",
    )
    prefill_projections.add_argument(
        "--kv-a-suffix",
        default=".self_attn.kv_a_proj_with_mqa.weight",
        help="resident matrix suffix for kv_a_proj_with_mqa",
    )
    prefill_projections.add_argument(
        "--kv-a-norm-suffix",
        default=".self_attn.kv_a_layernorm.weight",
        help="resident vector suffix for kv_a_layernorm",
    )
    prefill_projections.add_argument(
        "--kv-b-suffix",
        default=".self_attn.kv_b_proj.weight",
        help="resident matrix suffix for kv_b_proj",
    )
    prefill_projections.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_projections.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_projections.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_projections.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_projections.add_argument("--quiet-runner", action="store_true")
    prefill_projections.add_argument("--json", action="store_true")
    prefill_projections.set_defaults(func=_prefill_attention_projections)

    prefill_cache = sub.add_parser(
        "prefill-cache-write",
        help="stream a batch of f32 KV-A rows into an MLA decode cache segment",
    )
    prefill_cache.add_argument("cache_layout", help="decode cache layout JSON")
    prefill_cache.add_argument("cache_file", help="decode cache backing file")
    prefill_cache.add_argument("--layer", type=int, required=True)
    prefill_cache.add_argument("--input-f32", required=True)
    prefill_cache.add_argument("--start-position", type=int, required=True)
    prefill_cache.add_argument("--batch-tokens", type=int, required=True)
    prefill_cache.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    prefill_cache.add_argument("--max-cache-write-mib", type=float, default=4096.0)
    prefill_cache.add_argument("--json", action="store_true")
    prefill_cache.set_defaults(func=_prefill_cache_write)

    dsa_indexer = sub.add_parser(
        "dsa-indexer-batch",
        help="write DSA index cache rows and compute per-query top-k indices",
    )
    dsa_indexer.add_argument("resident_layout", help="resident/layout.json")
    dsa_indexer.add_argument("cache_layout", help="decode cache layout JSON")
    dsa_indexer.add_argument("cache_file", help="decode cache backing file")
    dsa_indexer.add_argument("--layer", type=int, required=True)
    dsa_indexer.add_argument("--hidden-f32", required=True)
    dsa_indexer.add_argument("--q-resid-f32", required=True)
    dsa_indexer.add_argument("--output-indices-json", default=None)
    dsa_indexer.add_argument("--output-indices-u32", default=None)
    dsa_indexer.add_argument("--start-position", type=int, required=True)
    dsa_indexer.add_argument("--batch-tokens", type=int, required=True)
    dsa_indexer.add_argument("--context-length", type=int, required=True)
    dsa_indexer.add_argument("--index-topk", type=int, required=True)
    dsa_indexer.add_argument("--index-n-heads", type=int, required=True)
    dsa_indexer.add_argument("--qk-rope-dim", type=int, required=True)
    dsa_indexer.add_argument("--rope-theta", type=float, default=10000.0)
    dsa_indexer.add_argument("--rope-interleave", action="store_true")
    dsa_indexer.add_argument("--layer-norm-eps", type=float, default=1e-6)
    dsa_indexer.add_argument("--max-resident-matrix-mib", type=int, default=512)
    dsa_indexer.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    dsa_indexer.add_argument("--max-cache-write-mib", type=float, default=4096.0)
    dsa_indexer.add_argument("--max-cache-read-mib", type=float, default=4096.0)
    dsa_indexer.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    dsa_indexer.add_argument("--include-topk-in-json", action="store_true")
    dsa_indexer.add_argument("--json", action="store_true")
    dsa_indexer.set_defaults(func=_dsa_indexer_batch)

    prefill_rope = sub.add_parser(
        "prefill-rope-batch",
        help="split q_b batch and run Metal RoPE over q_rope/k_rope rows",
    )
    prefill_rope.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_rope.add_argument("--q-b-f32", required=True)
    prefill_rope.add_argument("--k-rope-f32", required=True)
    prefill_rope.add_argument("--output-dir", required=True)
    prefill_rope.add_argument("--batch-tokens", type=int, required=True)
    prefill_rope.add_argument("--num-heads", type=int, required=True)
    prefill_rope.add_argument("--qk-nope-dim", type=int, required=True)
    prefill_rope.add_argument("--rope-dim", type=int, required=True)
    prefill_rope.add_argument("--start-position", type=int, required=True)
    prefill_rope.add_argument("--rope-theta", type=float, default=10000.0)
    prefill_rope.add_argument("--rope-interleave", action="store_true")
    prefill_rope.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_rope.add_argument("--quiet-runner", action="store_true")
    prefill_rope.add_argument("--json", action="store_true")
    prefill_rope.set_defaults(func=_prefill_rope_batch)

    prefill_mla = sub.add_parser(
        "prefill-mla-attention-batch",
        help="run bounded causal batch MLA attention for a prompt chunk",
    )
    prefill_mla.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_mla.add_argument("resident_layout", help="resident/layout.json")
    prefill_mla.add_argument("cache_layout", help="decode cache layout JSON")
    prefill_mla.add_argument("cache_file", help="decode cache backing file")
    prefill_mla.add_argument("--layer", type=int, required=True)
    prefill_mla.add_argument("--q-nope-f32", required=True)
    prefill_mla.add_argument("--q-rope-f32", required=True)
    prefill_mla.add_argument("--indices-u32", default=None)
    prefill_mla.add_argument("--output-f32", required=True)
    prefill_mla.add_argument("--context-length", type=int, required=True)
    prefill_mla.add_argument("--start-position", type=int, required=True)
    prefill_mla.add_argument("--batch-tokens", type=int, required=True)
    prefill_mla.add_argument("--index-topk", type=int, default=None)
    prefill_mla.add_argument("--num-heads", type=int, required=True)
    prefill_mla.add_argument("--qk-nope-dim", type=int, required=True)
    prefill_mla.add_argument("--rope-dim", type=int, required=True)
    prefill_mla.add_argument("--v-head-dim", type=int, required=True)
    prefill_mla.add_argument("--kv-lora-dim", type=int, default=None)
    prefill_mla.add_argument("--cache-position-offset", type=int, default=0)
    prefill_mla.add_argument("--attention-scale", type=float, default=None)
    prefill_mla.add_argument("--rope-theta", type=float, default=10000.0)
    prefill_mla.add_argument("--rope-interleave", action="store_true")
    prefill_mla.add_argument(
        "--mla-kv-b-cache-dir",
        default=None,
        help="optional directory for per-layer f32 MLA kv_b caches",
    )
    prefill_mla.add_argument(
        "--mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache for non-indexed batch prefill",
    )
    prefill_mla.add_argument(
        "--no-mla-value-cache",
        dest="mla_value_cache",
        action="store_false",
        help="disable the default MLA value-side cache for non-indexed batch prefill",
    )
    prefill_mla.set_defaults(mla_value_cache=True)
    prefill_mla.add_argument("--max-cache-file-mib", type=int, default=32768)
    prefill_mla.add_argument("--max-cache-read-mib", type=int, default=256)
    prefill_mla.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_mla.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_mla.add_argument("--quiet-runner", action="store_true")
    prefill_mla.add_argument("--json", action="store_true")
    prefill_mla.set_defaults(func=_prefill_mla_attention_batch)

    prefill_attn_out = sub.add_parser(
        "prefill-attention-output-batch",
        help="run bounded batch o_proj over attention values and add residual hidden",
    )
    prefill_attn_out.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_attn_out.add_argument("resident_layout", help="resident/layout.json")
    prefill_attn_out.add_argument("--layer", type=int, required=True)
    prefill_attn_out.add_argument("--attn-value-f32", required=True)
    prefill_attn_out.add_argument("--residual-f32", required=True)
    prefill_attn_out.add_argument("--output-f32", required=True)
    prefill_attn_out.add_argument("--batch-tokens", type=int, required=True)
    prefill_attn_out.add_argument(
        "--o-proj-suffix",
        default=".self_attn.o_proj.weight",
        help="resident matrix suffix for attention output projection",
    )
    prefill_attn_out.add_argument(
        "--projection-f32",
        default=None,
        help="optional path for the intermediate o_proj output",
    )
    prefill_attn_out.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_attn_out.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_attn_out.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_attn_out.add_argument("--quiet-runner", action="store_true")
    prefill_attn_out.add_argument("--json", action="store_true")
    prefill_attn_out.set_defaults(func=_prefill_attention_output_batch)

    prefill_attn_block = sub.add_parser(
        "prefill-attention-block-batch",
        help="run bounded batch attention projections, cache write, MLA, o_proj, and residual",
    )
    prefill_attn_block.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_attn_block.add_argument("resident_layout", help="resident/layout.json")
    prefill_attn_block.add_argument("cache_layout", help="decode cache layout JSON")
    prefill_attn_block.add_argument("cache_file", help="decode cache backing file")
    prefill_attn_block.add_argument("--layer", type=int, required=True)
    prefill_attn_block.add_argument("--input-f32", required=True)
    prefill_attn_block.add_argument("--output-dir", required=True)
    prefill_attn_block.add_argument("--output-f32", required=True)
    prefill_attn_block.add_argument("--context-length", type=int, default=None)
    prefill_attn_block.add_argument("--start-position", type=int, required=True)
    prefill_attn_block.add_argument("--batch-tokens", type=int, required=True)
    prefill_attn_block.add_argument("--num-heads", type=int, required=True)
    prefill_attn_block.add_argument("--qk-nope-dim", type=int, required=True)
    prefill_attn_block.add_argument("--rope-dim", type=int, required=True)
    prefill_attn_block.add_argument("--v-head-dim", type=int, required=True)
    prefill_attn_block.add_argument("--kv-lora-dim", type=int, default=None)
    prefill_attn_block.add_argument("--cache-position-offset", type=int, default=0)
    prefill_attn_block.add_argument("--attention-scale", type=float, default=None)
    prefill_attn_block.add_argument("--rope-theta", type=float, default=10000.0)
    prefill_attn_block.add_argument("--rope-interleave", action="store_true")
    prefill_attn_block.add_argument(
        "--mla-kv-b-cache-dir",
        default=None,
        help="optional directory for per-layer f32 MLA kv_b caches",
    )
    prefill_attn_block.add_argument(
        "--mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache for non-indexed batch prefill",
    )
    prefill_attn_block.add_argument(
        "--no-mla-value-cache",
        dest="mla_value_cache",
        action="store_false",
        help="disable the default MLA value-side cache for non-indexed batch prefill",
    )
    prefill_attn_block.set_defaults(mla_value_cache=True)
    prefill_attn_block.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_attn_block.add_argument("--max-cache-file-mib", type=int, default=32768)
    prefill_attn_block.add_argument("--max-cache-write-mib", type=int, default=4096)
    prefill_attn_block.add_argument("--max-cache-read-mib", type=int, default=256)
    prefill_attn_block.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_attn_block.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_attn_block.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_attn_block.add_argument("--quiet-runner", action="store_true")
    prefill_attn_block.add_argument("--json", action="store_true")
    prefill_attn_block.set_defaults(func=_prefill_attention_block_batch)

    prefill_dense_mlp = sub.add_parser(
        "prefill-dense-mlp-block-batch",
        help="run bounded batch dense MLP block and residual add for prompt chunks",
    )
    prefill_dense_mlp.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_dense_mlp.add_argument("resident_layout", help="resident/layout.json")
    prefill_dense_mlp.add_argument("--layer", type=int, required=True)
    prefill_dense_mlp.add_argument("--input-f32", required=True)
    prefill_dense_mlp.add_argument("--output-dir", required=True)
    prefill_dense_mlp.add_argument("--output-f32", required=True)
    prefill_dense_mlp.add_argument("--batch-tokens", type=int, required=True)
    prefill_dense_mlp.add_argument(
        "--norm-suffix",
        default=".post_attention_layernorm.weight",
        help="resident vector suffix for post-attention RMSNorm",
    )
    prefill_dense_mlp.add_argument(
        "--gate-suffix",
        default=".mlp.gate_proj.weight",
        help="resident matrix suffix for dense MLP gate projection",
    )
    prefill_dense_mlp.add_argument(
        "--up-suffix",
        default=".mlp.up_proj.weight",
        help="resident matrix suffix for dense MLP up projection",
    )
    prefill_dense_mlp.add_argument(
        "--down-suffix",
        default=".mlp.down_proj.weight",
        help="resident matrix suffix for dense MLP down projection",
    )
    prefill_dense_mlp.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_dense_mlp.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_dense_mlp.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_dense_mlp.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_dense_mlp.add_argument("--quiet-runner", action="store_true")
    prefill_dense_mlp.add_argument("--json", action="store_true")
    prefill_dense_mlp.set_defaults(func=_prefill_dense_mlp_block_batch)

    prefill_routed_mlp = sub.add_parser(
        "prefill-routed-mlp-block-batch",
        help="run safe serial routed/shared MoE MLP block over a prompt chunk",
    )
    prefill_routed_mlp.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_routed_mlp.add_argument("expert_layout", help="experts/layout.json")
    prefill_routed_mlp.add_argument("resident_layout", help="resident/layout.json")
    prefill_routed_mlp.add_argument("--layer", type=int, required=True)
    prefill_routed_mlp.add_argument("--input-f32", required=True)
    prefill_routed_mlp.add_argument("--output-dir", required=True)
    prefill_routed_mlp.add_argument("--output-f32", required=True)
    prefill_routed_mlp.add_argument("--batch-tokens", type=int, required=True)
    prefill_routed_mlp.add_argument("--top-k", type=int, default=8)
    prefill_routed_mlp.add_argument("--max-k", type=int, default=8)
    prefill_routed_mlp.add_argument(
        "--router-score",
        choices=("sigmoid", "softmax", "raw"),
        default="sigmoid",
    )
    prefill_routed_mlp.add_argument("--routed-scaling-factor", type=float, default=None)
    prefill_routed_mlp.add_argument("--norm-topk-prob", action="store_true")
    prefill_routed_mlp.add_argument("--no-norm-topk-prob", action="store_true")
    prefill_routed_mlp.add_argument("--router-n-group", type=int, default=None)
    prefill_routed_mlp.add_argument("--router-topk-group", type=int, default=None)
    prefill_routed_mlp.add_argument("--ignore-router-bias", action="store_true")
    prefill_routed_mlp.add_argument("--include-shared-expert", action="store_true")
    prefill_routed_mlp.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_routed_mlp.add_argument("--max-slot-mib", type=int, default=256)
    prefill_routed_mlp.add_argument("--max-router-mib", type=int, default=64)
    prefill_routed_mlp.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_routed_mlp.add_argument(
        "--expert-read-advise-merge-gap-kib",
        type=int,
        default=0,
    )
    prefill_routed_mlp.add_argument(
        "--expert-read-advise-align-kib",
        type=int,
        default=0,
    )
    prefill_routed_mlp.add_argument("--router-json-dir", default=None)
    prefill_routed_mlp.add_argument("--keep-token-files", action="store_true")
    prefill_routed_mlp.add_argument("--quiet-runner", action="store_true")
    prefill_routed_mlp.add_argument("--json", action="store_true")
    prefill_routed_mlp.set_defaults(func=_prefill_routed_mlp_block_batch)

    prefill_staged_routed_mlp = sub.add_parser(
        "prefill-staged-routed-mlp-block-batch",
        help="run prefill routed MLP through staged unique expert slots",
    )
    prefill_staged_routed_mlp.add_argument("runner", help="path to metal/largerlm-runner")
    prefill_staged_routed_mlp.add_argument("expert_layout", help="experts/layout.json")
    prefill_staged_routed_mlp.add_argument("resident_layout", help="resident/layout.json")
    prefill_staged_routed_mlp.add_argument("--layer", type=int, required=True)
    prefill_staged_routed_mlp.add_argument("--input-f32", required=True)
    prefill_staged_routed_mlp.add_argument("--output-dir", required=True)
    prefill_staged_routed_mlp.add_argument("--output-f32", required=True)
    prefill_staged_routed_mlp.add_argument("--batch-tokens", type=int, required=True)
    prefill_staged_routed_mlp.add_argument("--top-k", type=int, default=8)
    prefill_staged_routed_mlp.add_argument("--max-k", type=int, default=8)
    prefill_staged_routed_mlp.add_argument(
        "--router-score",
        choices=("sigmoid", "softmax", "raw"),
        default="sigmoid",
    )
    prefill_staged_routed_mlp.add_argument("--routed-scaling-factor", type=float, default=None)
    prefill_staged_routed_mlp.add_argument("--norm-topk-prob", action="store_true")
    prefill_staged_routed_mlp.add_argument("--no-norm-topk-prob", action="store_true")
    prefill_staged_routed_mlp.add_argument("--router-n-group", type=int, default=None)
    prefill_staged_routed_mlp.add_argument("--router-topk-group", type=int, default=None)
    prefill_staged_routed_mlp.add_argument("--ignore-router-bias", action="store_true")
    prefill_staged_routed_mlp.add_argument("--include-shared-expert", action="store_true")
    prefill_staged_routed_mlp.add_argument("--rms-norm-eps", type=float, default=1e-5)
    prefill_staged_routed_mlp.add_argument("--max-resident-matrix-mib", type=int, default=512)
    prefill_staged_routed_mlp.add_argument("--max-slot-mib", type=int, default=256)
    prefill_staged_routed_mlp.add_argument("--max-router-mib", type=int, default=64)
    prefill_staged_routed_mlp.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    prefill_staged_routed_mlp.add_argument("--moe-token-block", default="auto")
    prefill_staged_routed_mlp.add_argument(
        "--moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
    )
    prefill_staged_routed_mlp.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="custom-metal",
    )
    prefill_staged_routed_mlp.add_argument(
        "--prefill-router-hybrid-margin-threshold",
        type=float,
        default=0.0,
        help=(
            "when >0 and auto selects MPSGraph for router gates, keep custom "
            "Metal router output when its min effective margin exceeds this "
            "threshold"
        ),
    )
    prefill_staged_routed_mlp.add_argument("--expert-stage-merge-gap-kib", type=float, default=0.0)
    prefill_staged_routed_mlp.add_argument("--expert-stage-align-kib", type=float, default=4.0)
    prefill_staged_routed_mlp.add_argument("--max-stage-mib", type=float, default=4096.0)
    prefill_staged_routed_mlp.add_argument(
        "--max-compact-stage-mib",
        type=float,
        default=4096.0,
    )
    prefill_staged_routed_mlp.add_argument("--copy-chunk-mib", type=float, default=8.0)
    prefill_staged_routed_mlp.add_argument(
        "--stage-disk-margin-mib",
        type=float,
        default=0.0,
    )
    prefill_staged_routed_mlp.add_argument(
        "--prefill-ssd-read-gib-s",
        dest="prefill_ssd_read_gib_per_second",
        type=float,
        default=0.0,
        help="measured SSD read GiB/s for staged expert read-time estimates",
    )
    prefill_staged_routed_mlp.add_argument(
        "--prefill-max-routed-read-seconds",
        type=float,
        default=0.0,
        help="fail before staging when planned expert reads exceed this many seconds",
    )
    prefill_staged_routed_mlp.add_argument(
        "--expert-stage-max-raw-ranges",
        type=int,
        default=0,
        help="fail before staging when raw expert read ranges exceed this count; 0 disables",
    )
    prefill_staged_routed_mlp.add_argument(
        "--expert-stage-max-coalesced-ranges",
        type=int,
        default=0,
        help="fail before staging when coalesced expert read ranges exceed this count; 0 disables",
    )
    prefill_staged_routed_mlp.add_argument(
        "--expert-stage-tiling",
        action="store_true",
        help="split selected experts into bounded stage/compact tiles before MoE",
    )
    prefill_staged_routed_mlp.add_argument("--static-capacity-per-expert", type=int, default=None)
    prefill_staged_routed_mlp.add_argument("--static-capacity-output-json", default=None)
    prefill_staged_routed_mlp.add_argument("--static-capacity-output-bin", default=None)
    prefill_staged_routed_mlp.add_argument(
        "--no-static-capacity-json",
        action="store_true",
        help="skip the debug static-capacity JSON artifact and write only the binary route table",
    )
    prefill_staged_routed_mlp.add_argument("--allow-static-capacity-overflow", action="store_true")
    prefill_staged_routed_mlp.add_argument("--keep-token-files", action="store_true")
    prefill_staged_routed_mlp.add_argument("--quiet-runner", action="store_true")
    prefill_staged_routed_mlp.add_argument("--json", action="store_true")
    prefill_staged_routed_mlp.set_defaults(func=_prefill_staged_routed_mlp_block_batch)

    expert_io = sub.add_parser(
        "plan-expert-io",
        help="plan coalesced slot read ranges for selected routed experts",
    )
    expert_io.add_argument("expert_layout", help="experts/layout.json")
    expert_io.add_argument("--layer", type=int, required=True)
    expert_io.add_argument("--experts", required=True, help="comma-separated expert ids")
    expert_io.add_argument(
        "--merge-gap-kib",
        type=float,
        default=0.0,
        help="merge neighboring ranges when the skipped gap is at most this size",
    )
    expert_io.add_argument(
        "--align-kib",
        type=float,
        default=4.0,
        help="align planned read ranges to this byte multiple in KiB",
    )
    expert_io.add_argument("--json", action="store_true")
    expert_io.set_defaults(func=_plan_expert_io)

    batch_expert_io = sub.add_parser(
        "plan-batch-expert-io",
        help="aggregate per-token router JSON into a coalesced batch expert I/O plan",
    )
    batch_expert_io.add_argument("expert_layout", help="experts/layout.json")
    batch_expert_io.add_argument("--layer", type=int, required=True)
    batch_expert_io.add_argument("--router-json-dir", required=True)
    batch_expert_io.add_argument("--router-json-glob", default="*.router.json")
    batch_expert_io.add_argument(
        "--merge-gap-kib",
        type=float,
        default=0.0,
        help="merge neighboring ranges when the skipped gap is at most this size",
    )
    batch_expert_io.add_argument(
        "--align-kib",
        type=float,
        default=4.0,
        help="align planned read ranges to this byte multiple in KiB",
    )
    batch_expert_io.add_argument(
        "--static-capacity-per-expert",
        type=int,
        default=None,
        help="also plan fixed expert token slots with this capacity per expert",
    )
    batch_expert_io.add_argument(
        "--static-capacity-output-json",
        default=None,
        help="write fixed expert token slots to this JSON artifact",
    )
    batch_expert_io.add_argument(
        "--static-capacity-output-bin",
        default=None,
        help="write fixed expert token slots to this compact binary artifact",
    )
    batch_expert_io.add_argument(
        "--ssd-read-gib-s",
        type=float,
        default=0.0,
        help="estimate planned batch expert read time with this SSD GiB/s",
    )
    batch_expert_io.add_argument(
        "--tile-max-stage-mib",
        type=float,
        default=None,
        help="also plan selected-expert stage tiles under this per-tile stage cap",
    )
    batch_expert_io.add_argument(
        "--tile-max-compact-stage-mib",
        type=float,
        default=None,
        help="also require each selected-expert tile to fit this compact-stage cap",
    )
    batch_expert_io.add_argument(
        "--allow-static-capacity-overflow",
        action="store_true",
        help="allow writing a static capacity artifact that contains overflow assignments",
    )
    batch_expert_io.add_argument("--json", action="store_true")
    batch_expert_io.set_defaults(func=_plan_batch_expert_io)

    static_capacity_bin = sub.add_parser(
        "validate-static-capacity-bin",
        help="validate an LLMSCAP1 static-capacity binary route table",
    )
    static_capacity_bin.add_argument("path", help="static_capacity.bin")
    static_capacity_bin.add_argument("--expect-batch-tokens", type=int, default=None)
    static_capacity_bin.add_argument("--expect-expert-count", type=int, default=None)
    static_capacity_bin.add_argument(
        "--expect-capacity-per-expert",
        type=int,
        default=None,
    )
    static_capacity_bin.add_argument(
        "--expect-total-assignments",
        type=int,
        default=None,
    )
    static_capacity_bin.add_argument("--expect-used-slots", type=int, default=None)
    static_capacity_bin.add_argument(
        "--expect-overflow-records",
        type=int,
        default=None,
    )
    static_capacity_bin.add_argument("--json", action="store_true")
    static_capacity_bin.set_defaults(func=_validate_static_capacity_bin)

    stage_batch = sub.add_parser(
        "stage-batch-experts",
        help="copy coalesced routed expert ranges into a bounded stage file",
    )
    stage_batch.add_argument("expert_layout", help="experts/layout.json")
    stage_batch.add_argument("--layer", type=int, required=True)
    stage_batch.add_argument("--router-json-dir", required=True)
    stage_batch.add_argument("--router-json-glob", default="*.router.json")
    stage_batch.add_argument("--stage-file", required=True)
    stage_batch.add_argument("--manifest", default=None)
    stage_batch.add_argument(
        "--merge-gap-kib",
        type=float,
        default=0.0,
        help="merge neighboring ranges when the skipped gap is at most this size",
    )
    stage_batch.add_argument(
        "--align-kib",
        type=float,
        default=4.0,
        help="align staged read ranges to this byte multiple in KiB",
    )
    stage_batch.add_argument("--max-stage-mib", type=float, default=4096.0)
    stage_batch.add_argument("--copy-chunk-mib", type=float, default=8.0)
    stage_batch.add_argument("--stage-disk-margin-mib", type=float, default=0.0)
    stage_batch.add_argument(
        "--ssd-read-gib-s",
        type=float,
        default=0.0,
        help="estimate planned stage read time with this SSD GiB/s",
    )
    stage_batch.add_argument(
        "--max-read-seconds",
        type=float,
        default=0.0,
        help="fail before staging if planned expert reads exceed this many seconds",
    )
    stage_batch.add_argument(
        "--max-raw-ranges",
        type=int,
        default=0,
        help="fail before staging if raw expert read ranges exceed this count; 0 disables",
    )
    stage_batch.add_argument(
        "--max-coalesced-ranges",
        type=int,
        default=0,
        help="fail before staging if coalesced expert read ranges exceed this count; 0 disables",
    )
    stage_batch.add_argument("--json", action="store_true")
    stage_batch.set_defaults(func=_stage_batch_experts)

    staged_moe = sub.add_parser(
        "run-staged-routed-moe-batch",
        help="run routed MoE over a staged batch expert file with bounded scratch",
    )
    staged_moe.add_argument("runner", help="path to metal/largerlm-runner")
    staged_moe.add_argument("--stage-manifest", required=True)
    staged_moe.add_argument("--input-f32", required=True)
    staged_moe.add_argument("--output-dir", required=True)
    staged_moe.add_argument("--output-f32", required=True)
    staged_moe.add_argument("--max-compact-stage-mib", type=float, default=4096.0)
    staged_moe.add_argument("--copy-chunk-mib", type=float, default=8.0)
    staged_moe.add_argument(
        "--compact-stage-disk-margin-mib",
        type=float,
        default=0.0,
    )
    staged_moe.add_argument("--max-slot-mib", type=int, default=256)
    staged_moe.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    staged_moe.add_argument("--moe-token-block", default="auto")
    staged_moe.add_argument(
        "--moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
    )
    staged_moe.add_argument("--static-capacity-per-expert", type=int, default=None)
    staged_moe.add_argument("--static-capacity-output-json", default=None)
    staged_moe.add_argument("--static-capacity-output-bin", default=None)
    staged_moe.add_argument(
        "--no-static-capacity-json",
        action="store_true",
        help="skip the debug static-capacity JSON artifact and write only the binary route table",
    )
    staged_moe.add_argument("--allow-static-capacity-overflow", action="store_true")
    staged_moe.add_argument("--keep-token-files", action="store_true")
    staged_moe.add_argument("--quiet-runner", action="store_true")
    staged_moe.add_argument("--json", action="store_true")
    staged_moe.set_defaults(func=_run_staged_routed_moe_batch)

    staged_moe_plan = sub.add_parser(
        "run-staged-routed-moe-batch-plan",
        help="run multiple materialized staged routed MoE batch jobs in one runner process",
    )
    staged_moe_plan.add_argument("runner", help="path to metal/largerlm-runner")
    staged_moe_plan.add_argument("--plan-json", required=True)
    staged_moe_plan.add_argument(
        "--moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
    )
    staged_moe_plan.add_argument("--quiet-runner", action="store_true")
    staged_moe_plan.add_argument("--json", action="store_true")
    staged_moe_plan.set_defaults(func=_run_staged_routed_moe_batch_plan)

    staged_moe_plan_server = sub.add_parser(
        "run-staged-routed-moe-batch-plan-server",
        help="run staged routed MoE batch plans through one JSONL runner process",
    )
    staged_moe_plan_server.add_argument("runner", help="path to metal/largerlm-runner")
    staged_moe_plan_server.add_argument(
        "--plan-json",
        action="append",
        required=True,
        help="materialized staged routed MoE batch plan; may be passed more than once",
    )
    staged_moe_plan_server.add_argument(
        "--moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
    )
    staged_moe_plan_server.add_argument("--quiet-runner", action="store_true")
    staged_moe_plan_server.add_argument("--json", action="store_true")
    staged_moe_plan_server.set_defaults(
        func=_run_staged_routed_moe_batch_plan_server
    )

    tiled_staged_moe = sub.add_parser(
        "run-tiled-staged-routed-moe-batch",
        help="run routed MoE by staging selected experts in safe tiles",
    )
    tiled_staged_moe.add_argument("runner", help="path to metal/largerlm-runner")
    tiled_staged_moe.add_argument("expert_layout", help="experts/layout.json")
    tiled_staged_moe.add_argument("--layer", type=int, required=True)
    tiled_staged_moe.add_argument("--router-json-dir", required=True)
    tiled_staged_moe.add_argument("--router-json-glob", default="*.router.json")
    tiled_staged_moe.add_argument("--input-f32", required=True)
    tiled_staged_moe.add_argument("--output-dir", required=True)
    tiled_staged_moe.add_argument("--output-f32", required=True)
    tiled_staged_moe.add_argument("--merge-gap-kib", type=float, default=0.0)
    tiled_staged_moe.add_argument("--align-kib", type=float, default=4.0)
    tiled_staged_moe.add_argument("--max-stage-mib", type=float, default=4096.0)
    tiled_staged_moe.add_argument("--max-compact-stage-mib", type=float, default=4096.0)
    tiled_staged_moe.add_argument("--copy-chunk-mib", type=float, default=8.0)
    tiled_staged_moe.add_argument("--stage-disk-margin-mib", type=float, default=0.0)
    tiled_staged_moe.add_argument("--ssd-read-gib-s", type=float, default=0.0)
    tiled_staged_moe.add_argument("--max-read-seconds", type=float, default=0.0)
    tiled_staged_moe.add_argument("--max-raw-ranges", type=int, default=0)
    tiled_staged_moe.add_argument("--max-coalesced-ranges", type=int, default=0)
    tiled_staged_moe.add_argument("--max-slot-mib", type=int, default=256)
    tiled_staged_moe.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    tiled_staged_moe.add_argument("--moe-token-block", default="auto")
    tiled_staged_moe.add_argument(
        "--moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
    )
    tiled_staged_moe.add_argument("--static-capacity-per-expert", type=int, default=None)
    tiled_staged_moe.add_argument(
        "--no-static-capacity-json",
        action="store_true",
        help="skip the debug static-capacity JSON artifact and write only the binary route table",
    )
    tiled_staged_moe.add_argument("--allow-static-capacity-overflow", action="store_true")
    tiled_staged_moe.add_argument("--keep-token-files", action="store_true")
    tiled_staged_moe.add_argument("--quiet-runner", action="store_true")
    tiled_staged_moe.add_argument("--json", action="store_true")
    tiled_staged_moe.set_defaults(func=_run_tiled_staged_routed_moe_batch)

    pack = sub.add_parser(
        "pack-experts",
        help="pack routed expert safetensors into fixed-offset per-layer files",
    )
    pack.add_argument("model", help="checkpoint directory")
    pack.add_argument("--output", default=None, help="output directory (default: MODEL/experts)")
    pack.add_argument("--layers", default=None, help='layer spec, e.g. "3-5,8"')
    pack.add_argument("--execute", action="store_true", help="actually write files")
    pack.add_argument("--force", action="store_true", help="overwrite existing packed files")
    pack.add_argument("--chunk-mib", type=float, default=8.0)
    pack.add_argument("--max-chunk-mib", type=float, default=64.0)
    pack.add_argument("--max-pack-heap-mib", type=float, default=512.0)
    pack.add_argument("--disk-margin-gib", type=float, default=16.0)
    pack.add_argument("--group-size", type=int, default=64)
    pack.add_argument(
        "--quantize-bf16-affine-int4",
        action="store_true",
        help="quantize raw BF16/F16/F32 expert weights into LargerLM affine int4 slots",
    )
    pack.add_argument("--json", action="store_true")
    pack.set_defaults(func=_pack_experts)

    resident = sub.add_parser(
        "pack-resident",
        help="pack non-routed resident weights into one aligned binary blob",
    )
    resident.add_argument("model", help="checkpoint directory")
    resident.add_argument("--output", default=None, help="output directory (default: MODEL/resident)")
    resident.add_argument("--execute", action="store_true", help="actually write files")
    resident.add_argument("--force", action="store_true", help="overwrite existing files")
    resident.add_argument("--chunk-mib", type=float, default=8.0)
    resident.add_argument("--max-chunk-mib", type=float, default=64.0)
    resident.add_argument("--max-pack-heap-mib", type=float, default=512.0)
    resident.add_argument("--disk-margin-gib", type=float, default=16.0)
    resident.add_argument("--alignment", type=int, default=64)
    resident.add_argument("--json", action="store_true")
    resident.set_defaults(func=_pack_resident)

    mlx_base = sub.add_parser(
        "export-mlx-baseline",
        help="safely export a small MLX logits baseline for tiny checkpoints",
    )
    mlx_base.add_argument("model", help="checkpoint directory")
    mlx_base.add_argument("--output", default=None, help="output directory")
    mlx_base.add_argument("--prompt", default="Hello", help="prompt text")
    mlx_base.add_argument("--max-tokens", type=int, default=1)
    mlx_base.add_argument("--max-prompt-tokens", type=int, default=4096)
    mlx_base.add_argument("--max-tensor-mib", type=float, default=32.0)
    mlx_base.add_argument("--max-model-load-gib", type=float, default=64.0)
    mlx_base.add_argument("--allow-unknown-size", action="store_true")
    mlx_base.add_argument("--execute", action="store_true", help="load model and write baseline")
    mlx_base.add_argument("--json", action="store_true")
    mlx_base.set_defaults(func=_export_mlx_baseline)

    validate = sub.add_parser(
        "validate-baseline",
        help="validate baseline tensor byte counts and sha256 hashes",
    )
    validate.add_argument("baseline", help="baseline directory")
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=_validate_baseline)

    status = sub.add_parser(
        "checkpoint-status",
        help="inspect checkpoint download/header readiness without loading weights",
    )
    status.add_argument("model", help="checkpoint directory")
    status.add_argument(
        "--repo",
        default=None,
        help="optional Hugging Face repo id used to print download URLs/commands",
    )
    status.add_argument("--revision", default=None)
    status.add_argument("--endpoint", default=None)
    status.add_argument(
        "--download-disk-margin-gib",
        type=float,
        default=16.0,
        help="extra free disk GiB to require beyond remaining safetensors shards",
    )
    status.add_argument(
        "--write-missing-shards-json",
        default=None,
        help="write a JSON manifest for missing/truncated shard downloads",
    )
    status.add_argument(
        "--write-missing-shards-urls",
        default=None,
        help="write one URL per missing/truncated shard for external download tools",
    )
    status.add_argument(
        "--write-external-download-json",
        default=None,
        help=(
            "write a JSON handoff manifest with missing shard URLs, target paths, "
            "and expected byte sizes"
        ),
    )
    status.add_argument(
        "--write-external-download-sh",
        default=None,
        help=(
            "write a resumable curl script for missing shards with exact byte-size "
            "checks"
        ),
    )
    status.add_argument(
        "--write-next-bringup-json",
        default=None,
        help="write the first ready bring-up step and exact argv as JSON",
    )
    status.add_argument(
        "--write-next-bringup-sh",
        default=None,
        help="write the first ready bring-up step as a quoted executable shell script",
    )
    status.add_argument(
        "--write-status-json",
        default=None,
        help="write the full checkpoint-status report JSON atomically",
    )
    status.add_argument(
        "--verify-local-headers",
        action="store_true",
        help=(
            "read each present local safetensors header and validate it against "
            "the index/header manifest without reading tensor payloads"
        ),
    )
    status.add_argument(
        "--require-complete",
        action="store_true",
        help="return non-zero unless all expected shards have manifest-proven sizes",
    )
    status.add_argument(
        "--require-download-disk-ok",
        action="store_true",
        help=(
            "return non-zero unless remaining shard bytes plus the download "
            "disk margin fit on the target volume"
        ),
    )
    status.add_argument(
        "--require-clean",
        action="store_true",
        help=(
            "return non-zero if checkpoint-status reports missing, partial, "
            "extra, corrupt, or otherwise unsafe artifact state"
        ),
    )
    status.add_argument(
        "--require-preflight-ready",
        action="store_true",
        help="return non-zero unless config and safetensors metadata are ready for preflight",
    )
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_checkpoint_status)

    result_summary = sub.add_parser(
        "result-summary",
        help="summarize a generation or prompt-prefill result JSON without loading weights",
    )
    result_summary.add_argument("result", help="result JSON written by LargerLM")
    result_summary.add_argument(
        "--top",
        type=int,
        default=12,
        help="number of top elapsed groups/records to include",
    )
    result_summary.add_argument("--json", action="store_true")
    result_summary.set_defaults(func=_result_summary)

    result_compare = sub.add_parser(
        "result-compare",
        help="compare two generation or prompt-prefill result JSON files",
    )
    result_compare.add_argument("baseline", help="baseline result JSON")
    result_compare.add_argument("candidate", help="candidate result JSON")
    result_compare.add_argument(
        "--top",
        type=int,
        default=12,
        help="number of largest elapsed changes to include",
    )
    result_compare.add_argument(
        "--system-slowdown-ratio",
        type=float,
        default=1.8,
        help="median sentinel ratio used to flag likely system-level slowdown",
    )
    result_compare.add_argument(
        "--require-candidate-promotable",
        action="store_true",
        help=(
            "return non-zero unless profile_recommendation marks the candidate "
            "result safe to promote"
        ),
    )
    result_compare.add_argument(
        "--allow-prefill-policy-change",
        action="store_true",
        help=(
            "treat matching prompt/max-new/generated-token results as comparable "
            "even when the prefill backend or routing policy changes the prefill "
            "plan signature"
        ),
    )
    result_compare.add_argument("--json", action="store_true")
    result_compare.set_defaults(func=_result_compare)

    result_bakeoff = sub.add_parser(
        "result-bakeoff",
        help="select the best promotable candidate from multiple result JSON files",
    )
    result_bakeoff.add_argument("baseline", help="baseline result JSON")
    result_bakeoff.add_argument(
        "candidates",
        nargs="+",
        help="candidate result JSON files to compare against the baseline",
    )
    result_bakeoff.add_argument(
        "--top",
        type=int,
        default=12,
        help="number of largest elapsed changes to keep per candidate comparison",
    )
    result_bakeoff.add_argument(
        "--system-slowdown-ratio",
        type=float,
        default=1.8,
        help="median sentinel ratio used to flag likely system-level slowdown",
    )
    result_bakeoff.add_argument(
        "--require-winner",
        action="store_true",
        help="return non-zero unless at least one candidate is promotable",
    )
    result_bakeoff.add_argument(
        "--allow-prefill-policy-change",
        action="store_true",
        help=(
            "allow explicit backend/routing policy experiments to be judged by "
            "matching prompt/max-new/generated tokens instead of requiring an "
            "identical prefill plan signature"
        ),
    )
    result_bakeoff.add_argument(
        "--promote-only-replay-files-ready",
        action="store_true",
        help=(
            "only select a candidate winner when its locked launch binding is "
            "replay-ready and all referenced replay files and launch audit "
            "bindings are current"
        ),
    )
    result_bakeoff.add_argument(
        "--require-selected-replay-ready",
        action="store_true",
        help=(
            "return non-zero unless the selected winner or retained baseline "
            "has a locked replay-ready launch binding"
        ),
    )
    result_bakeoff.add_argument(
        "--write-selected-replay-json",
        default=None,
        help="write the selected replay argv and launch binding as a compact JSON artifact",
    )
    result_bakeoff.add_argument(
        "--write-bakeoff-json",
        default=None,
        help="write the full result-bakeoff report as JSON",
    )
    result_bakeoff.add_argument(
        "--write-selected-replay-script",
        default=None,
        help="write an executable shell script for the selected locked replay command",
    )
    result_bakeoff.add_argument("--json", action="store_true")
    result_bakeoff.set_defaults(func=_result_bakeoff)

    selected_replay_check = sub.add_parser(
        "selected-replay-check",
        help="verify a selected replay JSON before running its heavy replay command",
    )
    selected_replay_check.add_argument(
        "selected_replay",
        help="largerlm.selected_replay.v1 JSON produced by result-bakeoff",
    )
    _add_selected_replay_ssd_read_check_args(selected_replay_check)
    selected_replay_check.add_argument("--json", action="store_true")
    selected_replay_check.set_defaults(func=_selected_replay_check)

    selected_replay_run = sub.add_parser(
        "selected-replay-run",
        help="check a selected replay JSON and then run its locked replay command",
    )
    selected_replay_run.add_argument(
        "selected_replay",
        help="largerlm.selected_replay.v1 JSON produced by result-bakeoff",
    )
    selected_replay_run.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the replay command without running it",
    )
    selected_replay_run.add_argument(
        "--write-result",
        default=None,
        help=(
            "append an output-only --write-result path to a checked "
            "generate-prepared-token-ids replay"
        ),
    )
    selected_replay_run.add_argument(
        "--quiet-runner",
        action="store_true",
        help=(
            "append --quiet-runner to the checked generate-prepared-token-ids "
            "replay without mutating the selected replay artifact"
        ),
    )
    _add_selected_replay_ssd_read_check_args(selected_replay_run)
    selected_replay_run.add_argument("--json", action="store_true")
    selected_replay_run.set_defaults(func=_selected_replay_run)

    preflight = sub.add_parser(
        "preflight-glm",
        help="header-only GLM checkpoint readiness and safety preflight",
    )
    preflight.add_argument("model", help="checkpoint directory")
    preflight.add_argument("--tokenizer", default=None, help="tokenizer directory or file")
    preflight.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    preflight.add_argument(
        "--load-tokenizer",
        action="store_true",
        help="instantiate the local tokenizer instead of only checking files",
    )
    preflight.add_argument("--trust-remote-code", action="store_true")
    preflight.add_argument("--quant-bits", type=int, default=4, choices=(2, 3, 4, 8))
    preflight.add_argument("--group-size", type=int, default=64)
    preflight.add_argument(
        "--quantize-bf16-affine-int4",
        action="store_true",
        help="preflight raw BF16/F16/F32 expert quantization coverage",
    )
    preflight.add_argument(
        "--require-public-glm-5-2-shape",
        action="store_true",
        help="return not-ready unless config.json matches the public GLM-5.2 shape",
    )
    preflight.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "prefer largerlm.safetensors.headers.json over local shard files, "
            "allowing header-only preflight while shards are still downloading"
        ),
    )
    preflight.add_argument("--max-context-tokens", type=int, default=None)
    preflight.add_argument("--max-cache-gib", type=float, default=None)
    preflight.add_argument("--output-dir", default=None)
    preflight.add_argument("--disk-margin-gib", type=float, default=16.0)
    preflight.add_argument("--unified-memory-gib", type=float, default=None)
    preflight.add_argument(
        "--system-reserve-gib",
        type=float,
        default=None,
        help="override automatic live unified-memory reserve in GiB",
    )
    preflight.add_argument("--runtime-buffer-gib", type=float, default=8.0)
    preflight.add_argument("--page-cache-fraction", type=float, default=0.60)
    preflight.add_argument("--cold-read-gib-s", type=float, default=None)
    preflight.add_argument(
        "--write-report",
        default=None,
        help="atomically write the preflight JSON report to this path",
    )
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(func=_preflight_glm)

    prepare = sub.add_parser(
        "prepare-glm",
        help="dry-run or execute safe GLM packing plus sparse decode-cache setup",
    )
    prepare.add_argument("model", help="checkpoint directory")
    prepare.add_argument(
        "--apply-prepare-flags",
        default=None,
        help="apply a plan --write-prepare-flags JSON artifact before local args",
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--max-context-tokens", type=int, default=None)
    prepare.add_argument(
        "--auto-context-from-budget",
        action="store_true",
        help="choose the largest decode-cache context that fits the modeled cache budget",
    )
    prepare.add_argument("--execute", action="store_true", help="actually write packed files")
    prepare.add_argument("--force", action="store_true", help="overwrite existing outputs")
    prepare.add_argument("--tokenizer", default=None, help="tokenizer directory or file")
    prepare.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    prepare.add_argument("--load-tokenizer", action="store_true")
    prepare.add_argument("--trust-remote-code", action="store_true")
    prepare.add_argument("--quant-bits", type=int, default=4, choices=(2, 3, 4, 8))
    prepare.add_argument("--group-size", type=int, default=64)
    prepare.add_argument(
        "--quantize-bf16-affine-int4",
        action="store_true",
        help="quantize raw BF16/F16/F32 expert weights into LargerLM int4 slots",
    )
    prepare.add_argument(
        "--require-public-glm-5-2-shape",
        action="store_true",
        help="fail unless config.json matches the public GLM-5.2 checkpoint shape",
    )
    prepare.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "prefer largerlm.safetensors.headers.json for dry-run layout/budget "
            "checks; incompatible with --execute"
        ),
    )
    prepare.add_argument("--cache-dtype", choices=("BF16", "F16", "F32"), default="BF16")
    prepare.add_argument("--cache-alignment", type=int, default=64)
    prepare.add_argument("--max-cache-gib", type=float, default=None)
    prepare.add_argument("--disk-margin-gib", type=float, default=16.0)
    prepare.add_argument("--chunk-mib", type=float, default=8.0)
    prepare.add_argument("--max-chunk-mib", type=float, default=64.0)
    prepare.add_argument("--max-pack-heap-mib", type=float, default=512.0)
    prepare.add_argument("--unified-memory-gib", type=float, default=None)
    prepare.add_argument(
        "--system-reserve-gib",
        type=float,
        default=None,
        help="override automatic live unified-memory reserve in GiB",
    )
    prepare.add_argument("--runtime-buffer-gib", type=float, default=8.0)
    prepare.add_argument("--page-cache-fraction", type=float, default=0.60)
    prepare.add_argument("--cold-read-gib-s", type=float, default=None)
    prepare.add_argument(
        "--auto-cold-read-benchmark",
        action="store_true",
        help=(
            "after --execute, benchmark the largest packed expert layer file "
            "and write the measured GiB/s to the prepared manifest"
        ),
    )
    prepare.add_argument("--cold-read-benchmark-mib", type=float, default=1024.0)
    prepare.add_argument("--cold-read-benchmark-chunk-mib", type=float, default=8.0)
    prepare.add_argument(
        "--write-report",
        default=None,
        help="atomically write the prepare-glm JSON report to this path",
    )
    prepare.add_argument("--json", action="store_true")
    prepare.set_defaults(func=_prepare_glm)

    cache = sub.add_parser(
        "plan-cache",
        help="build a non-allocating GLM MLA/DSA decode cache layout",
    )
    cache.add_argument("model", help="checkpoint directory or config.json")
    cache.add_argument("--max-context-tokens", type=int, required=True)
    cache.add_argument("--dtype", choices=("BF16", "F16", "F32"), default="BF16")
    cache.add_argument("--alignment", type=int, default=64)
    cache.add_argument("--max-cache-gib", type=float, default=None)
    cache.add_argument("--output", default=None, help="optional layout JSON path")
    cache.add_argument("--json", action="store_true")
    cache.set_defaults(func=_plan_decode_cache)

    init_cache = sub.add_parser(
        "init-cache",
        help="create a sparse backing file from a decode cache layout",
    )
    init_cache.add_argument("layout", help="decode cache layout JSON")
    init_cache.add_argument("output", help="output cache data file")
    init_cache.add_argument("--force", action="store_true", help="overwrite existing file")
    init_cache.add_argument("--max-cache-gib", type=float, default=None)
    init_cache.add_argument("--disk-margin-gib", type=float, default=16.0)
    init_cache.add_argument("--json", action="store_true")
    init_cache.set_defaults(func=_init_decode_cache)

    context1_cache = sub.add_parser(
        "context1-o-proj-cache",
        help="plan or build the GLM context=1 collapsed o_proj*B_v cache",
    )
    context1_cache.add_argument("prepared_dir", help="prepared LargerLM directory")
    context1_cache.add_argument(
        "--output-dir",
        default=None,
        help="output cache directory; defaults inside the prepared directory",
    )
    context1_cache.add_argument("--dtype", choices=("BF16", "F32"), default="BF16")
    context1_cache.add_argument(
        "--execute",
        action="store_true",
        help="actually build the cache; omitted means safe dry-run only",
    )
    context1_cache.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing cache/progress directory when executing",
    )
    context1_cache.add_argument(
        "--backend",
        choices=("reference", "metal"),
        default="reference",
        help="builder backend; reference is for tiny fixtures, metal calls glm_moe_infer per layer",
    )
    context1_cache.add_argument(
        "--metal-binary",
        default=None,
        help="glm_moe_infer binary for --backend metal; defaults to the repo build",
    )
    context1_cache.add_argument(
        "--max-cache-gib",
        type=float,
        default=None,
        help="fail if the collapsed cache would exceed this size",
    )
    context1_build_cap = context1_cache.add_mutually_exclusive_group()
    context1_build_cap.add_argument(
        "--max-build-fma",
        type=int,
        default=50_000_000,
        help="execute-mode safety cap for any backend",
    )
    context1_build_cap.add_argument(
        "--max-build-gfma",
        type=float,
        default=None,
        help="execute-mode safety cap in billions of FMA; alternative to --max-build-fma",
    )
    context1_cache.add_argument(
        "--disk-margin-gib",
        type=float,
        default=16.0,
        help="execute-mode free disk margin to keep after the collapsed cache",
    )
    context1_cache.add_argument(
        "--max-metal-builder-live-mib",
        type=float,
        default=512.0,
        help="execute-mode live-memory cap for each Metal cache-builder layer",
    )
    context1_cache.add_argument(
        "--row-tile",
        type=int,
        default=8,
        help="hidden rows per reference-backend tile",
    )
    context1_cache.add_argument(
        "--layer",
        type=int,
        action="append",
        default=None,
        help="plan a cache layout containing only this layer; may be repeated",
    )
    context1_cache.add_argument(
        "--build-layers",
        default=None,
        help=(
            "execute only these comma/range layers while keeping the planned "
            "cache layout intact, e.g. 0,7-9"
        ),
    )
    context1_cache.add_argument(
        "--build-next-layers",
        type=int,
        default=None,
        help=(
            "select the next N incomplete layers from progress; mutually "
            "exclusive with --build-layers"
        ),
    )
    context1_cache.add_argument(
        "--write-report",
        default=None,
        help="write the dry-run/build report JSON",
    )
    context1_cache.add_argument("--json", action="store_true")
    context1_cache.set_defaults(func=_context1_o_proj_cache)

    context1_validate = sub.add_parser(
        "validate-context1-o-proj-cache",
        help="validate a GLM context=1 collapsed o_proj*B_v cache layout",
    )
    context1_validate.add_argument("layout", help="context1 o_proj cache layout JSON")
    context1_validate.add_argument(
        "--prepared-dir",
        default=None,
        help="optional prepared directory to cross-check dims, layers, bytes, and config hash",
    )
    context1_validate.add_argument(
        "--no-cache-file",
        action="store_true",
        help="validate metadata only; do not require the cache backing file to exist",
    )
    context1_validate.add_argument(
        "--allow-incomplete-progress",
        action="store_true",
        help="report incomplete builder progress instead of failing validation",
    )
    context1_validate.add_argument("--json", action="store_true")
    context1_validate.set_defaults(func=_validate_context1_o_proj_cache)

    runtime = sub.add_parser(
        "check-runtime",
        help="check one packed layer against runner memory and I/O limits",
    )
    runtime.add_argument("expert_layout", help="experts/layout.json")
    runtime.add_argument("resident_layout", help="resident/layout.json")
    runtime.add_argument("--layer", type=int, required=True)
    runtime.add_argument("--top-k", type=int, default=8)
    runtime.add_argument("--max-k", type=int, default=8)
    runtime.add_argument("--dense-mlp", action="store_true")
    runtime.add_argument("--max-slot-mib", type=float, default=256.0)
    runtime.add_argument("--max-router-mib", type=float, default=64.0)
    runtime.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    runtime.add_argument("--max-cache-read-mib", type=float, default=256.0)
    runtime.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    runtime.add_argument("--include-shared-expert", action="store_true")
    runtime.add_argument("--include-attention-projections", action="store_true")
    runtime.add_argument("--include-decoder-layer", action="store_true")
    runtime.add_argument("--context-length", type=int, default=None)
    runtime.add_argument("--num-heads", type=int, default=None)
    runtime.add_argument("--qk-nope-dim", type=int, default=None)
    runtime.add_argument("--rope-dim", type=int, default=None)
    runtime.add_argument("--v-head-dim", type=int, default=None)
    runtime.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    runtime.add_argument("--json", action="store_true")
    runtime.set_defaults(func=_check_runtime)

    decode = sub.add_parser(
        "decode-layers",
        help="run a bounded multi-layer hidden-state decode pass through packed layers",
    )
    decode.add_argument("expert_layout", help="experts/layout.json")
    decode.add_argument("resident_layout", help="resident/layout.json")
    decode.add_argument("cache_layout", help="decode cache layout JSON")
    decode.add_argument("cache_file", help="decode cache backing file")
    decode.add_argument("--model-config", default=None, help="checkpoint dir or config.json")
    decode.add_argument("--runner", default="metal/largerlm-runner")
    decode.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    decode.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    decode.add_argument("--input-f32", required=True)
    decode.add_argument("--output-f32", required=True)
    decode.add_argument("--position", type=int, required=True)
    decode.add_argument("--context-length", type=int, required=True)
    decode.add_argument("--num-heads", type=int, default=None)
    decode.add_argument("--qk-nope-dim", type=int, default=None)
    decode.add_argument("--rope-dim", type=int, default=None)
    decode.add_argument("--v-head-dim", type=int, default=None)
    decode.add_argument("--kv-lora-dim", type=int, default=None)
    decode.add_argument(
        "--mla-kv-b-cache-dir",
        default=None,
        help="optional directory for per-layer f32 MLA kv_b caches during decode",
    )
    decode.add_argument("--cache-position-offset", type=int, default=0)
    decode.add_argument("--attention-scale", type=float, default=None)
    decode.add_argument("--rope-theta", type=float, default=None)
    decode.add_argument("--rope-interleave", action="store_true")
    decode.add_argument("--top-k", type=int, default=None)
    decode.add_argument("--max-k", type=int, default=None)
    decode.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    decode.add_argument("--routed-scaling-factor", type=float, default=None)
    decode.add_argument("--norm-topk-prob", action="store_true")
    decode.add_argument("--no-norm-topk-prob", action="store_true")
    decode.add_argument("--router-n-group", type=int, default=None)
    decode.add_argument("--router-topk-group", type=int, default=None)
    decode.add_argument("--ignore-router-bias", action="store_true")
    decode.add_argument("--include-shared-expert", action="store_true")
    decode.add_argument("--no-shared-expert", action="store_true")
    decode.add_argument("--rms-norm-eps", type=float, default=None)
    decode.add_argument("--max-slot-mib", type=float, default=256.0)
    decode.add_argument("--max-router-mib", type=float, default=64.0)
    decode.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    decode.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    decode.add_argument("--max-cache-write-mib", type=float, default=4096.0)
    decode.add_argument("--max-cache-read-mib", type=float, default=256.0)
    decode.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    decode.add_argument("--disable-dsa-indexer", action="store_true")
    decode.add_argument("--dsa-index-topk", type=int, default=None)
    decode.add_argument("--dsa-index-n-heads", type=int, default=None)
    decode.add_argument("--dsa-index-head-dim", type=int, default=None)
    decode.add_argument("--dsa-qk-rope-dim", type=int, default=None)
    decode.add_argument("--dsa-rope-interleave", action="store_true")
    decode.add_argument("--dsa-layer-norm-eps", type=float, default=1e-6)
    _add_expert_read_advise_args(decode)
    decode.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    decode.add_argument("--work-dir", default=None)
    decode.add_argument("--keep-work-dir", action="store_true")
    decode.add_argument("--quiet-runner", action="store_true")
    decode.add_argument("--json", action="store_true")
    decode.set_defaults(func=_decode_layers)

    logits = sub.add_parser(
        "final-logits",
        help="stream final RMSNorm + lm_head/top-k from resident weights",
    )
    logits.add_argument("resident_layout", help="resident/layout.json")
    logits.add_argument("--input-f32", required=True, help="hidden state f32 vector")
    logits.add_argument("--runner", default=None, help="optional Metal runner for chunked top-k")
    logits.add_argument("--output-logits-f32", default=None)
    logits.add_argument("--output-topk-json", default=None)
    logits.add_argument("--top-k", type=int, default=1)
    logits.add_argument("--rms-norm-eps", type=float, default=1e-5)
    logits.add_argument("--chunk-rows", type=int, default=None)
    logits.add_argument("--max-chunk-mib", type=float, default=64.0)
    logits.add_argument("--max-output-logits-mib", type=float, default=64.0)
    logits.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    logits.add_argument("--no-tied-embeddings", action="store_true")
    logits.add_argument("--skip-final-norm", action="store_true")
    logits.add_argument("--quiet-runner", action="store_true")
    logits.add_argument("--json", action="store_true")
    logits.set_defaults(func=_final_logits)

    embed = sub.add_parser(
        "embed-token",
        help="stream one token embedding row from resident weights",
    )
    embed.add_argument("resident_layout", help="resident/layout.json")
    embed.add_argument("--token-id", type=int, required=True)
    embed.add_argument("--output-f32", required=True)
    embed.add_argument("--max-row-mib", type=float, default=64.0)
    embed.add_argument("--json", action="store_true")
    embed.set_defaults(func=_embed_token)

    embed_batch = sub.add_parser(
        "embed-tokens-batch",
        help="stream multiple token embedding rows into one f32 prompt batch",
    )
    embed_batch.add_argument("resident_layout", help="resident/layout.json")
    embed_batch.add_argument("--token-ids", default=None, help="comma/space-separated token ids")
    embed_batch.add_argument("--token-ids-file", default=None, help="JSON array or text token ids")
    embed_batch.add_argument("--output-f32", required=True)
    embed_batch.add_argument("--max-row-mib", type=float, default=64.0)
    embed_batch.add_argument("--max-output-mib", type=float, default=4096.0)
    embed_batch.add_argument("--json", action="store_true")
    embed_batch.set_defaults(func=_embed_tokens_batch)

    prefill_prompt = sub.add_parser(
        "prefill-prompt",
        help="run a bounded chunked prompt prefill over packed GLM layers",
    )
    prefill_prompt.add_argument("expert_layout", help="experts/layout.json")
    prefill_prompt.add_argument("resident_layout", help="resident/layout.json")
    prefill_prompt.add_argument("cache_layout", help="decode cache layout JSON")
    prefill_prompt.add_argument("cache_file", help="decode cache backing file")
    prefill_prompt.add_argument("--model-config", default=None, help="checkpoint dir or config.json")
    prefill_prompt.add_argument("--runner", default="metal/largerlm-runner")
    prefill_prompt.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    prefill_prompt.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    prefill_prompt.add_argument("--prompt-token-ids", required=True, help="comma-separated token ids")
    prefill_prompt.add_argument("--output-last-hidden-f32", required=True)
    prefill_prompt.add_argument("--output-final-chunk-f32", default=None)
    prefill_prompt.add_argument("--start-position", type=int, default=0)
    prefill_prompt.add_argument(
        "--prompt-chunk-tokens",
        type=_parse_prefill_prompt_chunk_tokens,
        default=64,
        metavar="N|auto",
        help="prompt tokens per chunk; use auto to size from safety caps",
    )
    prefill_prompt.add_argument("--max-prompt-batch-mib", type=float, default=1024.0)
    prefill_prompt.add_argument("--num-heads", type=int, default=None)
    prefill_prompt.add_argument("--qk-nope-dim", type=int, default=None)
    prefill_prompt.add_argument("--rope-dim", type=int, default=None)
    prefill_prompt.add_argument("--v-head-dim", type=int, default=None)
    prefill_prompt.add_argument("--kv-lora-dim", type=int, default=None)
    prefill_prompt.add_argument("--cache-position-offset", type=int, default=0)
    prefill_prompt.add_argument("--attention-scale", type=float, default=None)
    prefill_prompt.add_argument("--rope-theta", type=float, default=None)
    prefill_prompt.add_argument("--rope-interleave", action="store_true")
    prefill_prompt.add_argument("--disable-dsa-indexer", action="store_true")
    prefill_prompt.add_argument("--dsa-index-topk", type=int, default=None)
    prefill_prompt.add_argument("--dsa-index-n-heads", type=int, default=None)
    prefill_prompt.add_argument("--dsa-qk-rope-dim", type=int, default=None)
    prefill_prompt.add_argument("--dsa-rope-interleave", action="store_true")
    prefill_prompt.add_argument("--dsa-layer-norm-eps", type=float, default=1e-6)
    prefill_prompt.add_argument("--top-k", type=int, default=None)
    prefill_prompt.add_argument("--max-k", type=int, default=None)
    prefill_prompt.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    prefill_prompt.add_argument("--routed-scaling-factor", type=float, default=None)
    prefill_prompt.add_argument("--norm-topk-prob", action="store_true")
    prefill_prompt.add_argument("--no-norm-topk-prob", action="store_true")
    prefill_prompt.add_argument("--router-n-group", type=int, default=None)
    prefill_prompt.add_argument("--router-topk-group", type=int, default=None)
    prefill_prompt.add_argument("--ignore-router-bias", action="store_true")
    prefill_prompt.add_argument("--include-shared-expert", action="store_true")
    prefill_prompt.add_argument("--no-shared-expert", action="store_true")
    prefill_prompt.add_argument("--rms-norm-eps", type=float, default=None)
    prefill_prompt.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="auto",
    )
    prefill_prompt.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    )
    prefill_prompt.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
    )
    prefill_prompt.add_argument("--max-embedding-row-mib", type=float, default=64.0)
    prefill_prompt.add_argument("--max-slot-mib", type=float, default=256.0)
    prefill_prompt.add_argument("--max-router-mib", type=float, default=64.0)
    prefill_prompt.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    prefill_prompt.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    prefill_prompt.add_argument("--max-cache-write-mib", type=float, default=4096.0)
    prefill_prompt.add_argument("--max-cache-read-mib", type=float, default=256.0)
    prefill_prompt.add_argument(
        "--prefill-mla-kv-b-cache-dir",
        default=None,
        help="optional directory for per-layer f32 MLA kv_b caches",
    )
    prefill_prompt.add_argument(
        "--prefill-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during batch prompt prefill",
    )
    prefill_prompt.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_generation_memory_guard_args(prefill_prompt)
    prefill_prompt.add_argument(
        "--moe-token-block",
        default="auto",
        metavar="N|auto",
        help="MoE token block for staged routed MLP calls; default auto fits runner scratch",
    )
    prefill_prompt.add_argument(
        "--moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
        help=(
            "control staged MoE output accumulation; env preserves "
            "LARGERLM_MOE_BATCH_ACCUMULATOR, file/memory pin it"
        ),
    )
    prefill_prompt.add_argument(
        "--static-capacity-per-expert",
        default=None,
        metavar="N|auto",
        help="write LLMSCAP1 binary routes for prompt MoE; auto uses each chunk size",
    )
    prefill_prompt.add_argument(
        "--allow-static-capacity-overflow",
        action="store_true",
        help="allow prompt static-capacity route artifacts to contain overflow records",
    )
    prefill_prompt.add_argument("--expert-stage-merge-gap-kib", type=float, default=0.0)
    prefill_prompt.add_argument("--expert-stage-align-kib", type=float, default=4.0)
    prefill_prompt.add_argument("--max-stage-mib", type=float, default=4096.0)
    prefill_prompt.add_argument("--max-compact-stage-mib", type=float, default=4096.0)
    prefill_prompt.add_argument("--expert-stage-max-raw-ranges", type=int, default=0)
    prefill_prompt.add_argument(
        "--expert-stage-max-coalesced-ranges",
        type=int,
        default=0,
    )
    prefill_prompt.add_argument(
        "--expert-stage-tiling",
        action="store_true",
        help="split selected experts into bounded stage/compact tiles during prompt prefill",
    )
    prefill_prompt.add_argument(
        "--persistent-moe-plan-server",
        action="store_true",
        help=(
            "keep one Metal routed-MoE plan runner alive across prompt prefill "
            "routed layers and expert-stage tiles"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-resident-linear-server",
        action="store_true",
        help=(
            "keep one Metal resident-linear runner alive across prompt prefill "
            "resident batch-linear requests"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-attention-projection-server",
        action="store_true",
        help=(
            "keep one Metal attention-projection runner alive across prompt "
            "prefill fused projection requests"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-attention-output-server",
        action="store_true",
        help=(
            "keep one Metal attention-output runner alive across prompt "
            "prefill fused output-projection requests"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-shared-expert-server",
        action="store_true",
        help=(
            "keep one Metal shared-expert runner alive across prompt "
            "prefill fused shared-expert requests"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-rope-split-server",
        action="store_true",
        help=(
            "keep one Metal RoPE split runner alive across prompt prefill "
            "fused RoPE split requests"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-mla-attention-server",
        action="store_true",
        help=(
            "keep one Metal MLA attention runner alive across prompt prefill "
            "batch MLA attention requests"
        ),
    )
    prefill_prompt.add_argument(
        "--persistent-rmsnorm-server",
        action="store_true",
        help=(
            "keep one Metal RMSNorm runner alive across prompt prefill "
            "resident RMSNorm batch requests"
        ),
    )
    prefill_prompt.add_argument("--copy-chunk-mib", type=float, default=8.0)
    prefill_prompt.add_argument(
        "--stage-disk-margin-mib",
        type=float,
        default=0.0,
        help="extra free disk margin required before writing prompt work/stage files",
    )
    prefill_prompt.add_argument(
        "--prefill-ssd-read-gib-s",
        dest="prefill_ssd_read_gib_per_second",
        type=float,
        default=0.0,
        help="measured SSD read GiB/s for staged expert read-time estimates",
    )
    prefill_prompt.add_argument(
        "--prefill-max-routed-read-seconds",
        type=float,
        default=0.0,
        help="fail before staging when planned expert reads exceed this many seconds",
    )
    prefill_prompt.add_argument("--work-dir", default=None)
    prefill_prompt.add_argument("--keep-work-dir", action="store_true")
    prefill_prompt.add_argument("--quiet-runner", action="store_true")
    prefill_prompt.add_argument("--json", action="store_true")
    prefill_prompt.set_defaults(func=_prompt_prefill)

    gen = sub.add_parser(
        "generate-token-ids",
        help="run a greedy or top-k sampled token-id loop over packed weights",
    )
    gen.add_argument("expert_layout", help="experts/layout.json")
    gen.add_argument("resident_layout", help="resident/layout.json")
    gen.add_argument("cache_layout", help="decode cache layout JSON")
    gen.add_argument("cache_file", help="decode cache backing file")
    gen.add_argument("--model-config", default=None, help="checkpoint dir or config.json")
    gen.add_argument("--runner", default="metal/largerlm-runner")
    gen.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    gen.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    gen.add_argument("--prompt-token-ids", required=True, help="comma-separated token ids")
    gen.add_argument("--max-new-tokens", type=int, required=True)
    gen.add_argument("--eos-token-id", type=int, default=None)
    gen.add_argument("--num-heads", type=int, default=None)
    gen.add_argument("--qk-nope-dim", type=int, default=None)
    gen.add_argument("--rope-dim", type=int, default=None)
    gen.add_argument("--v-head-dim", type=int, default=None)
    gen.add_argument("--kv-lora-dim", type=int, default=None)
    gen.add_argument("--cache-position-offset", type=int, default=0)
    gen.add_argument("--attention-scale", type=float, default=None)
    gen.add_argument("--rope-theta", type=float, default=None)
    gen.add_argument("--rope-interleave", action="store_true")
    gen.add_argument("--top-k", type=int, default=None)
    gen.add_argument("--max-k", type=int, default=None)
    gen.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    gen.add_argument("--routed-scaling-factor", type=float, default=None)
    gen.add_argument("--norm-topk-prob", action="store_true")
    gen.add_argument("--no-norm-topk-prob", action="store_true")
    gen.add_argument("--router-n-group", type=int, default=None)
    gen.add_argument("--router-topk-group", type=int, default=None)
    gen.add_argument("--ignore-router-bias", action="store_true")
    gen.add_argument("--include-shared-expert", action="store_true")
    gen.add_argument("--no-shared-expert", action="store_true")
    gen.add_argument("--rms-norm-eps", type=float, default=None)
    gen.add_argument("--logits-top-k", type=int, default=1)
    gen.add_argument("--logits-chunk-rows", type=int, default=None)
    gen.add_argument("--logits-max-chunk-mib", type=float, default=64.0)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--metal-final-logits", action="store_true")
    gen.add_argument("--no-runtime-preflight", dest="preflight_runtime", action="store_false")
    gen.set_defaults(preflight_runtime=True)
    gen.add_argument("--max-embedding-row-mib", type=float, default=64.0)
    gen.add_argument("--max-slot-mib", type=float, default=256.0)
    gen.add_argument("--max-router-mib", type=float, default=64.0)
    gen.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    gen.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    gen.add_argument("--max-cache-read-mib", type=float, default=256.0)
    gen.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_generation_memory_guard_args(gen)
    _add_expert_read_advise_args(gen)
    _add_batch_prefill_prompt_args(gen)
    gen.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    gen.add_argument("--work-dir", default=None)
    gen.add_argument("--keep-work-dir", action="store_true")
    gen.add_argument("--quiet-runner", action="store_true")
    gen.add_argument("--json", action="store_true")
    gen.set_defaults(func=_generate_token_ids)

    metal_gen = sub.add_parser(
        "generate-metal-token-ids",
        help="run the single-process glm_moe_infer greedy decode loop",
    )
    metal_gen.add_argument("prepared_dir", help="prepared LargerLM package root")
    metal_gen.add_argument("--binary", default="metal/glm_moe_infer")
    metal_gen.add_argument(
        "--expert-pin-plan",
        default=None,
        help="quality-preserving expert residency plan for glm_moe_infer",
    )
    metal_gen.add_argument(
        "--max-adaptive-expert-cache-gib",
        type=float,
        default=0.0,
        help="override the plan's evictable expert cache budget",
    )
    metal_gen.add_argument("--prompt-token-ids", required=True, help="comma-separated token ids")
    metal_gen.add_argument("--max-new-tokens", type=int, required=True)
    metal_gen.add_argument("--top-k", type=int, default=None)
    metal_gen.add_argument("--logits-top-k", type=int, default=8)
    metal_gen.add_argument("--max-live-working-set-mib", type=int, default=768)
    metal_gen.add_argument("--max-cache-file-mib", type=float, default=64.0)
    metal_gen.add_argument("--max-cache-read-mib", type=float, default=1.0)
    metal_gen.add_argument("--logits-max-chunk-mib", type=float, default=16.0)
    metal_gen.add_argument(
        "--mmap-final-logits",
        action="store_true",
        help="mmap lm_head ranges as Metal buffers for final logits",
    )
    metal_gen.add_argument("--max-embedding-row-mib", type=float, default=1.0)
    metal_gen.add_argument(
        "--cache-mla-kv-b-f32",
        action="store_true",
        help="cache absorbed MLA KV-B F32 views in the glm_moe_infer process",
    )
    metal_gen.add_argument(
        "--max-mla-kv-b-cache-mib",
        type=float,
        default=0.0,
        help="required positive cache cap when --cache-mla-kv-b-f32 is enabled",
    )
    metal_gen.add_argument(
        "--context1-o-proj-cache-layout",
        default=None,
        help="validated context=1 o_proj*B_v collapsed-cache layout for glm_moe_infer",
    )
    metal_gen.add_argument(
        "--context1-o-proj-cache-file",
        default=None,
        help="optional override backing file for --context1-o-proj-cache-layout",
    )
    metal_gen.add_argument(
        "--min-free-unified-memory-gib",
        type=float,
        default=None,
        help="free unified-memory reserve for glm_moe_infer; defaults to prepared recommendation",
    )
    metal_gen.add_argument(
        "--allow-decode-only-multi-token-prompt",
        action="store_true",
        help="run anyway when the prompt has multiple tokens; only the last token is consumed",
    )
    metal_gen.add_argument(
        "--prefill-prompt",
        action="store_true",
        help="consume multi-token prompts through runtime prompt_token_ids prefill",
    )
    metal_gen.add_argument("--prefill-runner", default="metal/largerlm-runner")
    metal_gen.add_argument("--prefill-prompt-chunk-tokens", type=int, default=64)
    metal_gen.add_argument("--prefill-max-prompt-batch-mib", type=float, default=1024.0)
    metal_gen.add_argument("--prefill-max-cache-write-mib", type=float, default=4096.0)
    metal_gen.add_argument(
        "--prefill-max-runner-scratch-mib",
        type=float,
        default=4096.0,
    )
    metal_gen.add_argument(
        "--prefill-max-live-working-set-mib",
        type=float,
        default=None,
        help="prompt-prefill bridge live cap; defaults to --max-live-working-set-mib",
    )
    metal_gen.add_argument(
        "--no-generate-server-jsonl",
        action="store_true",
        help="use the legacy direct --generate-request-json entry instead of the JSONL service",
    )
    metal_gen.add_argument(
        "--python-prefill-bridge",
        action="store_true",
        help="use the older Python prompt-prefill bridge instead of runtime prompt_token_ids prefill",
    )
    metal_gen.add_argument("--work-dir", default=None)
    metal_gen.add_argument("--keep-work-dir", action="store_true")
    metal_gen.add_argument("--quiet-runner", action="store_true")
    metal_gen.add_argument("--json", action="store_true")
    metal_gen.set_defaults(func=_generate_metal_token_ids)

    metal_text = sub.add_parser(
        "generate-metal-text",
        help="tokenize text and run the single-process glm_moe_infer greedy decode loop",
    )
    metal_text.add_argument("prepared_dir", help="prepared LargerLM package root")
    metal_text.add_argument("--tokenizer", default=None, help="local tokenizer path")
    metal_text.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    metal_text.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow transformers AutoTokenizer local custom code",
    )
    metal_text.add_argument("--prompt", default=None)
    metal_text.add_argument("--prompt-file", default=None)
    metal_text.add_argument(
        "--chat-messages",
        default=None,
        help='JSON array, e.g. [{"role":"user","content":"hi"}]',
    )
    metal_text.add_argument("--chat-messages-file", default=None)
    metal_text.add_argument("--no-add-generation-prompt", action="store_true")
    metal_text.add_argument("--max-prompt-bytes", type=int, default=16 * 1024 * 1024)
    metal_text.add_argument("--max-prompt-tokens", type=int, default=4096)
    metal_text.add_argument("--no-add-special-tokens", action="store_true")
    metal_text.add_argument("--no-skip-special-tokens", action="store_true")
    metal_text.add_argument("--binary", default="metal/glm_moe_infer")
    metal_text.add_argument("--expert-pin-plan", default=None)
    metal_text.add_argument(
        "--max-adaptive-expert-cache-gib",
        type=float,
        default=0.0,
    )
    metal_text.add_argument("--max-new-tokens", type=int, required=True)
    metal_text.add_argument("--top-k", type=int, default=None)
    metal_text.add_argument("--logits-top-k", type=int, default=8)
    metal_text.add_argument("--max-live-working-set-mib", type=int, default=768)
    metal_text.add_argument("--max-cache-file-mib", type=float, default=64.0)
    metal_text.add_argument("--max-cache-read-mib", type=float, default=1.0)
    metal_text.add_argument("--logits-max-chunk-mib", type=float, default=16.0)
    metal_text.add_argument(
        "--mmap-final-logits",
        action="store_true",
        help="mmap lm_head ranges as Metal buffers for final logits",
    )
    metal_text.add_argument("--max-embedding-row-mib", type=float, default=1.0)
    metal_text.add_argument(
        "--cache-mla-kv-b-f32",
        action="store_true",
        help="cache absorbed MLA KV-B F32 views in the glm_moe_infer process",
    )
    metal_text.add_argument(
        "--max-mla-kv-b-cache-mib",
        type=float,
        default=0.0,
        help="required positive cache cap when --cache-mla-kv-b-f32 is enabled",
    )
    metal_text.add_argument(
        "--context1-o-proj-cache-layout",
        default=None,
        help="validated context=1 o_proj*B_v collapsed-cache layout for glm_moe_infer",
    )
    metal_text.add_argument(
        "--context1-o-proj-cache-file",
        default=None,
        help="optional override backing file for --context1-o-proj-cache-layout",
    )
    metal_text.add_argument(
        "--min-free-unified-memory-gib",
        type=float,
        default=None,
        help="free unified-memory reserve for glm_moe_infer; defaults to prepared recommendation",
    )
    metal_text.add_argument(
        "--allow-decode-only-multi-token-prompt",
        action="store_true",
        help="run anyway when the prompt has multiple tokens; only the last token is consumed",
    )
    metal_text.add_argument(
        "--prefill-prompt",
        action="store_true",
        help="consume multi-token prompts through runtime prompt_token_ids prefill",
    )
    metal_text.add_argument("--prefill-runner", default="metal/largerlm-runner")
    metal_text.add_argument("--prefill-prompt-chunk-tokens", type=int, default=64)
    metal_text.add_argument("--prefill-max-prompt-batch-mib", type=float, default=1024.0)
    metal_text.add_argument("--prefill-max-cache-write-mib", type=float, default=4096.0)
    metal_text.add_argument(
        "--prefill-max-runner-scratch-mib",
        type=float,
        default=4096.0,
    )
    metal_text.add_argument(
        "--prefill-max-live-working-set-mib",
        type=float,
        default=None,
        help="prompt-prefill bridge live cap; defaults to --max-live-working-set-mib",
    )
    metal_text.add_argument(
        "--no-generate-server-jsonl",
        action="store_true",
        help="use the legacy direct --generate-request-json entry instead of the JSONL service",
    )
    metal_text.add_argument(
        "--python-prefill-bridge",
        action="store_true",
        help="use the older Python prompt-prefill bridge instead of runtime prompt_token_ids prefill",
    )
    metal_text.add_argument("--work-dir", default=None)
    metal_text.add_argument("--keep-work-dir", action="store_true")
    metal_text.add_argument("--quiet-runner", action="store_true")
    metal_text.add_argument("--json", action="store_true")
    metal_text.set_defaults(func=_generate_metal_text)

    metal_text_batch = sub.add_parser(
        "generate-metal-text-batch",
        help="run multiple text prompts through one persistent glm_moe_infer runtime",
    )
    metal_text_batch.add_argument("prepared_dir", help="prepared LargerLM package root")
    metal_text_batch.add_argument(
        "--prompts-jsonl",
        required=True,
        help='JSONL file with one {"prompt": "..."} object per line',
    )
    metal_text_batch.add_argument(
        "--max-prompts-jsonl-bytes",
        type=int,
        default=16 * 1024 * 1024,
    )
    metal_text_batch.add_argument("--tokenizer", default=None, help="local tokenizer path")
    metal_text_batch.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    metal_text_batch.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow transformers AutoTokenizer local custom code",
    )
    metal_text_batch.add_argument("--max-prompt-tokens", type=int, default=4096)
    metal_text_batch.add_argument("--no-add-special-tokens", action="store_true")
    metal_text_batch.add_argument("--no-skip-special-tokens", action="store_true")
    metal_text_batch.add_argument("--binary", default="metal/glm_moe_infer")
    metal_text_batch.add_argument("--expert-pin-plan", default=None)
    metal_text_batch.add_argument(
        "--max-adaptive-expert-cache-gib",
        type=float,
        default=0.0,
    )
    metal_text_batch.add_argument("--max-new-tokens", type=int, required=True)
    metal_text_batch.add_argument("--top-k", type=int, default=None)
    metal_text_batch.add_argument("--logits-top-k", type=int, default=8)
    metal_text_batch.add_argument("--max-live-working-set-mib", type=int, default=768)
    metal_text_batch.add_argument("--max-cache-file-mib", type=float, default=64.0)
    metal_text_batch.add_argument("--max-cache-read-mib", type=float, default=1.0)
    metal_text_batch.add_argument("--logits-max-chunk-mib", type=float, default=16.0)
    metal_text_batch.add_argument(
        "--mmap-final-logits",
        action="store_true",
        help="mmap lm_head ranges as Metal buffers for final logits",
    )
    metal_text_batch.add_argument("--max-embedding-row-mib", type=float, default=1.0)
    metal_text_batch.add_argument(
        "--context1-o-proj-cache-layout",
        default=None,
        help="validated context=1 o_proj*B_v collapsed-cache layout for glm_moe_infer",
    )
    metal_text_batch.add_argument(
        "--context1-o-proj-cache-file",
        default=None,
        help="optional override backing file for --context1-o-proj-cache-layout",
    )
    metal_text_batch.add_argument(
        "--min-free-unified-memory-gib",
        type=float,
        default=None,
        help="free unified-memory reserve for glm_moe_infer; defaults to prepared recommendation",
    )
    metal_text_batch.add_argument(
        "--allow-decode-only-multi-token-prompt",
        action="store_true",
        help="run anyway when a prompt has multiple tokens; only the last token is consumed",
    )
    metal_text_batch.add_argument(
        "--prefill-prompt",
        action="store_true",
        help="consume multi-token prompts through runtime prompt_token_ids prefill",
    )
    metal_text_batch.add_argument("--prefill-runner", default="metal/largerlm-runner")
    metal_text_batch.add_argument("--prefill-prompt-chunk-tokens", type=int, default=64)
    metal_text_batch.add_argument(
        "--prefill-max-prompt-batch-mib",
        type=float,
        default=1024.0,
    )
    metal_text_batch.add_argument(
        "--prefill-max-cache-write-mib",
        type=float,
        default=4096.0,
    )
    metal_text_batch.add_argument(
        "--prefill-max-runner-scratch-mib",
        type=float,
        default=4096.0,
    )
    metal_text_batch.add_argument(
        "--prefill-max-live-working-set-mib",
        type=float,
        default=None,
        help="prompt-prefill bridge live cap; defaults to --max-live-working-set-mib",
    )
    metal_text_batch.add_argument(
        "--no-generate-server-jsonl",
        action="store_true",
        help="use the legacy direct --generate-request-json entry instead of the JSONL service",
    )
    metal_text_batch.add_argument(
        "--python-prefill-bridge",
        action="store_true",
        help="use the older Python prompt-prefill bridge instead of runtime prompt_token_ids prefill",
    )
    metal_text_batch.add_argument("--keep-work-dir", action="store_true")
    metal_text_batch.add_argument("--quiet-runner", action="store_true")
    metal_text_batch.add_argument("--json", action="store_true")
    metal_text_batch.set_defaults(func=_generate_metal_text_batch)

    text = sub.add_parser(
        "generate-text",
        help="encode text locally, run packed generation, and decode generated text",
    )
    text.add_argument("expert_layout", help="experts/layout.json")
    text.add_argument("resident_layout", help="resident/layout.json")
    text.add_argument("cache_layout", help="decode cache layout JSON")
    text.add_argument("cache_file", help="decode cache backing file")
    text.add_argument("--tokenizer", required=True, help="local tokenizer directory or file")
    text.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    text.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow transformers AutoTokenizer local custom code",
    )
    text.add_argument("--prompt", default=None)
    text.add_argument("--prompt-file", default=None)
    text.add_argument(
        "--max-prompt-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="cap --prompt/--prompt-file UTF-8 bytes before tokenization",
    )
    text.add_argument("--max-prompt-tokens", type=int, default=4096)
    text.add_argument("--no-add-special-tokens", action="store_true")
    text.add_argument("--no-skip-special-tokens", action="store_true")
    text.add_argument("--ignore-tokenizer-eos", action="store_true")
    text.add_argument("--model-config", default=None, help="checkpoint dir or config.json")
    text.add_argument("--runner", default="metal/largerlm-runner")
    text.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    text.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    text.add_argument("--max-new-tokens", type=int, required=True)
    text.add_argument("--eos-token-id", type=int, default=None)
    text.add_argument("--num-heads", type=int, default=None)
    text.add_argument("--qk-nope-dim", type=int, default=None)
    text.add_argument("--rope-dim", type=int, default=None)
    text.add_argument("--v-head-dim", type=int, default=None)
    text.add_argument("--kv-lora-dim", type=int, default=None)
    text.add_argument("--cache-position-offset", type=int, default=0)
    text.add_argument("--attention-scale", type=float, default=None)
    text.add_argument("--rope-theta", type=float, default=None)
    text.add_argument("--rope-interleave", action="store_true")
    text.add_argument("--top-k", type=int, default=None)
    text.add_argument("--max-k", type=int, default=None)
    text.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    text.add_argument("--routed-scaling-factor", type=float, default=None)
    text.add_argument("--norm-topk-prob", action="store_true")
    text.add_argument("--no-norm-topk-prob", action="store_true")
    text.add_argument("--router-n-group", type=int, default=None)
    text.add_argument("--router-topk-group", type=int, default=None)
    text.add_argument("--ignore-router-bias", action="store_true")
    text.add_argument("--include-shared-expert", action="store_true")
    text.add_argument("--no-shared-expert", action="store_true")
    text.add_argument("--rms-norm-eps", type=float, default=None)
    text.add_argument("--logits-top-k", type=int, default=1)
    text.add_argument("--logits-chunk-rows", type=int, default=None)
    text.add_argument("--logits-max-chunk-mib", type=float, default=64.0)
    text.add_argument("--temperature", type=float, default=0.0)
    text.add_argument("--top-p", type=float, default=1.0)
    text.add_argument("--seed", type=int, default=None)
    text.add_argument("--metal-final-logits", action="store_true")
    text.add_argument("--no-runtime-preflight", dest="preflight_runtime", action="store_false")
    text.set_defaults(preflight_runtime=True)
    text.add_argument("--max-embedding-row-mib", type=float, default=64.0)
    text.add_argument("--max-slot-mib", type=float, default=256.0)
    text.add_argument("--max-router-mib", type=float, default=64.0)
    text.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    text.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    text.add_argument("--max-cache-read-mib", type=float, default=256.0)
    text.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_generation_memory_guard_args(text)
    _add_expert_read_advise_args(text)
    _add_batch_prefill_prompt_args(text)
    text.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    text.add_argument("--work-dir", default=None)
    text.add_argument("--keep-work-dir", action="store_true")
    text.add_argument("--quiet-runner", action="store_true")
    text.add_argument("--json", action="store_true")
    text.set_defaults(func=_generate_text)

    pgen = sub.add_parser(
        "generate-prepared-token-ids",
        help="run token-id generation from a prepare-glm manifest or output dir",
    )
    pgen.add_argument("prepared", help="prepared output directory or manifest.json")
    _add_launch_profile_arg(pgen)
    pgen.add_argument(
        "--require-launch-audit",
        default=None,
        metavar="PATH",
        help=(
            "require a passing inspect-prepared --write-launch-audit artifact "
            "for this prepared package, locked launch profile, and request envelope"
        ),
    )
    pgen.add_argument("--model-config", default=None, help="override model config path")
    _add_prepared_memory_profile_requirement_arg(pgen)
    _add_prepared_glm_4bit_requirement_arg(pgen)
    _add_prepared_ssd_read_default_arg(pgen)
    _add_prefill_acceleration_requirement_arg(pgen)
    _add_prefill_backend_probe_arg(pgen)
    pgen.add_argument("--runner", default="metal/largerlm-runner")
    pgen.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    pgen.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    pgen.add_argument("--prompt-token-ids", required=True, help="comma-separated token ids")
    pgen.add_argument("--max-new-tokens", type=int, required=True)
    pgen.add_argument("--eos-token-id", type=int, default=None)
    pgen.add_argument("--num-heads", type=int, default=None)
    pgen.add_argument("--qk-nope-dim", type=int, default=None)
    pgen.add_argument("--rope-dim", type=int, default=None)
    pgen.add_argument("--v-head-dim", type=int, default=None)
    pgen.add_argument("--kv-lora-dim", type=int, default=None)
    pgen.add_argument("--cache-position-offset", type=int, default=0)
    pgen.add_argument("--attention-scale", type=float, default=None)
    pgen.add_argument("--rope-theta", type=float, default=None)
    pgen.add_argument("--rope-interleave", action="store_true")
    pgen.add_argument("--top-k", type=int, default=None)
    pgen.add_argument("--max-k", type=int, default=None)
    pgen.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    pgen.add_argument("--routed-scaling-factor", type=float, default=None)
    pgen.add_argument("--norm-topk-prob", action="store_true")
    pgen.add_argument("--no-norm-topk-prob", action="store_true")
    pgen.add_argument("--router-n-group", type=int, default=None)
    pgen.add_argument("--router-topk-group", type=int, default=None)
    pgen.add_argument("--ignore-router-bias", action="store_true")
    pgen.add_argument("--include-shared-expert", action="store_true")
    pgen.add_argument("--no-shared-expert", action="store_true")
    pgen.add_argument("--rms-norm-eps", type=float, default=None)
    pgen.add_argument("--logits-top-k", type=int, default=1)
    pgen.add_argument("--logits-chunk-rows", type=int, default=None)
    pgen.add_argument("--logits-max-chunk-mib", type=float, default=64.0)
    pgen.add_argument("--temperature", type=float, default=0.0)
    pgen.add_argument("--top-p", type=float, default=1.0)
    pgen.add_argument("--seed", type=int, default=None)
    pgen.add_argument("--metal-final-logits", action="store_true")
    pgen.add_argument("--no-runtime-preflight", dest="preflight_runtime", action="store_false")
    pgen.set_defaults(preflight_runtime=True)
    pgen.add_argument("--max-embedding-row-mib", type=float, default=64.0)
    pgen.add_argument("--max-slot-mib", type=float, default=256.0)
    pgen.add_argument("--max-router-mib", type=float, default=64.0)
    pgen.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    pgen.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    pgen.add_argument("--max-cache-read-mib", type=float, default=256.0)
    pgen.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_generation_memory_guard_args(pgen)
    _add_expert_read_advise_args(pgen)
    _add_batch_prefill_prompt_args(pgen)
    pgen.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    pgen.add_argument("--work-dir", default=None)
    pgen.add_argument("--keep-work-dir", action="store_true")
    pgen.add_argument("--quiet-runner", action="store_true")
    pgen.add_argument(
        "--write-result",
        default=None,
        help="atomically write a schema-tagged prepared token generation JSON result",
    )
    pgen.add_argument("--json", action="store_true")
    pgen.set_defaults(func=_generate_prepared_token_ids)

    ptext = sub.add_parser(
        "generate-prepared-text",
        help="run text generation from a prepare-glm manifest or output dir",
    )
    ptext.add_argument("prepared", help="prepared output directory or manifest.json")
    _add_launch_profile_arg(ptext)
    ptext.add_argument(
        "--require-launch-audit",
        default=None,
        metavar="PATH",
        help=(
            "require a passing inspect-prepared --write-launch-audit artifact "
            "for this prepared package, locked launch profile, and request envelope"
        ),
    )
    ptext.add_argument("--tokenizer", default=None, help="override tokenizer directory or file")
    ptext.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    ptext.add_argument("--trust-remote-code", action="store_true")
    ptext.add_argument("--prompt", default=None)
    ptext.add_argument("--prompt-file", default=None)
    ptext.add_argument(
        "--max-prompt-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="cap --prompt/--prompt-file UTF-8 bytes before tokenization",
    )
    ptext.add_argument("--max-prompt-tokens", type=int, default=4096)
    ptext.add_argument("--no-add-special-tokens", action="store_true")
    ptext.add_argument("--no-skip-special-tokens", action="store_true")
    ptext.add_argument("--ignore-tokenizer-eos", action="store_true")
    ptext.add_argument("--model-config", default=None, help="override model config path")
    _add_prepared_memory_profile_requirement_arg(ptext)
    _add_prepared_glm_4bit_requirement_arg(ptext)
    _add_prepared_ssd_read_default_arg(ptext)
    _add_prefill_acceleration_requirement_arg(ptext)
    _add_prefill_backend_probe_arg(ptext)
    ptext.add_argument("--runner", default="metal/largerlm-runner")
    ptext.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    ptext.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    ptext.add_argument("--max-new-tokens", type=int, required=True)
    ptext.add_argument("--eos-token-id", type=int, default=None)
    ptext.add_argument("--num-heads", type=int, default=None)
    ptext.add_argument("--qk-nope-dim", type=int, default=None)
    ptext.add_argument("--rope-dim", type=int, default=None)
    ptext.add_argument("--v-head-dim", type=int, default=None)
    ptext.add_argument("--kv-lora-dim", type=int, default=None)
    ptext.add_argument("--cache-position-offset", type=int, default=0)
    ptext.add_argument("--attention-scale", type=float, default=None)
    ptext.add_argument("--rope-theta", type=float, default=None)
    ptext.add_argument("--rope-interleave", action="store_true")
    ptext.add_argument("--top-k", type=int, default=None)
    ptext.add_argument("--max-k", type=int, default=None)
    ptext.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    ptext.add_argument("--routed-scaling-factor", type=float, default=None)
    ptext.add_argument("--norm-topk-prob", action="store_true")
    ptext.add_argument("--no-norm-topk-prob", action="store_true")
    ptext.add_argument("--router-n-group", type=int, default=None)
    ptext.add_argument("--router-topk-group", type=int, default=None)
    ptext.add_argument("--ignore-router-bias", action="store_true")
    ptext.add_argument("--include-shared-expert", action="store_true")
    ptext.add_argument("--no-shared-expert", action="store_true")
    ptext.add_argument("--rms-norm-eps", type=float, default=None)
    ptext.add_argument("--logits-top-k", type=int, default=1)
    ptext.add_argument("--logits-chunk-rows", type=int, default=None)
    ptext.add_argument("--logits-max-chunk-mib", type=float, default=64.0)
    ptext.add_argument("--temperature", type=float, default=0.0)
    ptext.add_argument("--top-p", type=float, default=1.0)
    ptext.add_argument("--seed", type=int, default=None)
    ptext.add_argument("--metal-final-logits", action="store_true")
    ptext.add_argument("--no-runtime-preflight", dest="preflight_runtime", action="store_false")
    ptext.set_defaults(preflight_runtime=True)
    ptext.add_argument("--max-embedding-row-mib", type=float, default=64.0)
    ptext.add_argument("--max-slot-mib", type=float, default=256.0)
    ptext.add_argument("--max-router-mib", type=float, default=64.0)
    ptext.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    ptext.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    ptext.add_argument("--max-cache-read-mib", type=float, default=256.0)
    ptext.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_generation_memory_guard_args(ptext)
    _add_expert_read_advise_args(ptext)
    _add_batch_prefill_prompt_args(ptext)
    ptext.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    ptext.add_argument("--work-dir", default=None)
    ptext.add_argument("--keep-work-dir", action="store_true")
    ptext.add_argument("--quiet-runner", action="store_true")
    ptext.add_argument(
        "--write-result",
        default=None,
        help="atomically write a schema-tagged prepared text generation JSON result",
    )
    ptext.add_argument("--json", action="store_true")
    ptext.set_defaults(func=_generate_prepared_text)

    bench = sub.add_parser(
        "bench-prepared-token-ids",
        help="benchmark token-id generation from a prepare-glm manifest",
    )
    bench.add_argument("prepared", help="prepared output directory or manifest.json")
    _add_launch_profile_arg(bench)
    bench.add_argument(
        "--require-launch-audit",
        default=None,
        metavar="PATH",
        help=(
            "require a passing inspect-prepared --write-launch-audit artifact "
            "for this prepared package, locked launch profile, and benchmark envelope"
        ),
    )
    bench.add_argument("--model-config", default=None, help="override model config path")
    _add_prepared_memory_profile_requirement_arg(bench)
    _add_prepared_glm_4bit_requirement_arg(bench)
    _add_prepared_ssd_read_default_arg(bench)
    _add_prefill_acceleration_requirement_arg(bench)
    _add_prefill_backend_probe_arg(bench)
    bench.add_argument("--runner", default="metal/largerlm-runner")
    bench.add_argument(
        "--write-launch-profile",
        default=None,
        help="write the benchmark-derived suggested launch profile JSON",
    )
    bench.add_argument("--layers", default=None, help='layer spec, e.g. "0-3,7"')
    bench.add_argument("--dense-layers", default=None, help='dense MLP layer spec, e.g. "0-2"')
    bench.add_argument("--prompt-token-ids", required=True, help="comma-separated token ids")
    bench.add_argument("--max-new-tokens", type=int, required=True)
    bench.add_argument("--eos-token-id", type=int, default=None)
    bench.add_argument("--num-heads", type=int, default=None)
    bench.add_argument("--qk-nope-dim", type=int, default=None)
    bench.add_argument("--rope-dim", type=int, default=None)
    bench.add_argument("--v-head-dim", type=int, default=None)
    bench.add_argument("--kv-lora-dim", type=int, default=None)
    bench.add_argument("--cache-position-offset", type=int, default=0)
    bench.add_argument("--attention-scale", type=float, default=None)
    bench.add_argument("--rope-theta", type=float, default=None)
    bench.add_argument("--rope-interleave", action="store_true")
    bench.add_argument("--top-k", type=int, default=None)
    bench.add_argument("--max-k", type=int, default=None)
    bench.add_argument("--router-score", choices=("sigmoid", "softmax", "raw"), default=None)
    bench.add_argument("--routed-scaling-factor", type=float, default=None)
    bench.add_argument("--norm-topk-prob", action="store_true")
    bench.add_argument("--no-norm-topk-prob", action="store_true")
    bench.add_argument("--router-n-group", type=int, default=None)
    bench.add_argument("--router-topk-group", type=int, default=None)
    bench.add_argument("--ignore-router-bias", action="store_true")
    bench.add_argument("--include-shared-expert", action="store_true")
    bench.add_argument("--no-shared-expert", action="store_true")
    bench.add_argument("--rms-norm-eps", type=float, default=None)
    bench.add_argument("--logits-top-k", type=int, default=1)
    bench.add_argument("--logits-chunk-rows", type=int, default=None)
    bench.add_argument("--logits-max-chunk-mib", type=float, default=64.0)
    bench.add_argument("--temperature", type=float, default=0.0)
    bench.add_argument("--top-p", type=float, default=1.0)
    bench.add_argument("--seed", type=int, default=None)
    bench.add_argument("--metal-final-logits", action="store_true")
    bench.add_argument("--no-runtime-preflight", dest="preflight_runtime", action="store_false")
    bench.set_defaults(preflight_runtime=True)
    bench.add_argument("--max-embedding-row-mib", type=float, default=64.0)
    bench.add_argument("--max-slot-mib", type=float, default=256.0)
    bench.add_argument("--max-router-mib", type=float, default=64.0)
    bench.add_argument("--max-resident-matrix-mib", type=float, default=512.0)
    bench.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    bench.add_argument("--max-cache-read-mib", type=float, default=256.0)
    bench.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_generation_memory_guard_args(bench)
    _add_expert_read_advise_args(bench)
    _add_batch_prefill_prompt_args(bench)
    bench.add_argument("--cache-dtype-bytes", type=int, default=2, choices=(2, 4))
    bench.add_argument("--work-dir", default=None)
    bench.add_argument("--keep-work-dir", action="store_true")
    bench.add_argument("--quiet-runner", action="store_true")
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=_bench_prepared_token_ids)

    inspect_prepared = sub.add_parser(
        "inspect-prepared",
        help="validate a prepared manifest and print serving/generation health without starting a server",
    )
    inspect_prepared.add_argument(
        "prepared",
        help="prepared output directory or manifest.json",
    )
    _add_launch_profile_arg(inspect_prepared)
    inspect_prepared.add_argument(
        "--write-launch-profile",
        default=None,
        help=(
            "write the best available suggested launch profile JSON; request "
            "profile is preferred after a successful --check-* request"
        ),
    )
    inspect_prepared.add_argument("--runner", default="metal/largerlm-runner")
    inspect_prepared.add_argument("--model-config", default=None, help="override model config path")
    _add_prepared_memory_profile_requirement_arg(inspect_prepared)
    inspect_prepared.add_argument("--tokenizer", default=None, help="override tokenizer directory or file")
    inspect_prepared.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    inspect_prepared.add_argument("--trust-remote-code", action="store_true")
    inspect_prepared.add_argument("--served-model-name", default="largerlm-prepared")
    inspect_prepared.add_argument("--max-new-tokens-cap", type=int, default=256)
    inspect_prepared.add_argument("--max-prompt-tokens", type=int, default=4096)
    inspect_prepared.add_argument(
        "--max-request-bytes",
        type=int,
        default=1024 * 1024,
        help="cap offline prompt/chat check input bytes and prepared server request bodies",
    )
    inspect_prepared.add_argument("--logits-top-k-cap", type=int, default=64)
    _add_prepared_ssd_read_default_arg(inspect_prepared)
    _add_prefill_acceleration_requirement_arg(inspect_prepared)
    _add_prefill_backend_probe_arg(inspect_prepared)
    inspect_prepared.add_argument(
        "--no-batch-prefill-prompt",
        action="store_true",
        help="disable automatic chunked prompt prefill for multi-token requests",
    )
    inspect_prepared.add_argument(
        "--allow-missing-dsa-indexer",
        action="store_true",
        help="debug only: bypass the DSA cache schedule guard",
    )
    _add_prepared_glm_4bit_requirement_arg(inspect_prepared)
    inspect_prepared.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    inspect_prepared.add_argument("--max-cache-read-mib", type=float, default=256.0)
    inspect_prepared.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    inspect_prepared.add_argument(
        "--prefill-prompt-chunk-tokens",
        type=_parse_prefill_prompt_chunk_tokens,
        default=0,
        metavar="N|auto",
        help="prompt tokens per batch prefill chunk; default auto sizes from safety caps",
    )
    inspect_prepared.add_argument("--prefill-max-prompt-batch-mib", type=float, default=1024.0)
    inspect_prepared.add_argument("--prefill-max-cache-write-mib", type=float, default=4096.0)
    inspect_prepared.add_argument("--prefill-max-stage-mib", type=float, default=4096.0)
    inspect_prepared.add_argument("--prefill-max-compact-stage-mib", type=float, default=4096.0)
    inspect_prepared.add_argument(
        "--prefill-max-stage-raw-ranges",
        type=int,
        default=0,
        help="reject prompt prefill stage plans with more raw expert read ranges; 0 disables",
    )
    inspect_prepared.add_argument(
        "--prefill-max-stage-coalesced-ranges",
        type=int,
        default=0,
        help="reject prompt prefill stage plans with more coalesced expert read ranges; 0 disables",
    )
    inspect_prepared.add_argument(
        "--prefill-expert-stage-tiling",
        action="store_true",
        help="split checked prompt prefill routed experts into bounded stage/compact tiles",
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-moe-plan-server",
        action="store_true",
        help=(
            "keep one Metal routed-MoE plan runner alive across checked prompt "
            "prefill routed layers and expert-stage tiles"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-resident-linear-server",
        action="store_true",
        help=(
            "keep one Metal resident-linear runner alive across checked prompt "
            "prefill resident batch-linear requests"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-attention-projection-server",
        action="store_true",
        help=(
            "keep one Metal attention-projection runner alive across checked "
            "prompt prefill fused projection requests"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-attention-output-server",
        action="store_true",
        help=(
            "keep one Metal attention-output runner alive across checked "
            "prompt prefill fused output-projection requests"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-shared-expert-server",
        action="store_true",
        help=(
            "keep one Metal shared-expert runner alive across checked prompt "
            "prefill fused shared-expert requests"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-rope-split-server",
        action="store_true",
        help=(
            "keep one Metal RoPE split runner alive across checked prompt "
            "prefill fused RoPE split requests"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-mla-attention-server",
        action="store_true",
        help=(
            "keep one Metal MLA attention runner alive across checked prompt "
            "prefill batch MLA attention requests"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-persistent-rmsnorm-server",
        action="store_true",
        help=(
            "keep one Metal RMSNorm runner alive across checked prompt "
            "prefill resident RMSNorm batch requests"
        ),
    )
    inspect_prepared.add_argument("--prefill-copy-chunk-mib", type=float, default=8.0)
    inspect_prepared.add_argument(
        "--prefill-stage-disk-margin-mib",
        type=float,
        default=0.0,
        help="extra free disk margin required before prompt prefill stage files",
    )
    inspect_prepared.add_argument(
        "--prefill-max-routed-read-amplification",
        type=float,
        default=0.0,
        help=(
            "reject request checks/generation when prompt chunking would exceed "
            "this routed expert read amplification; <=0 disables the guard"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-max-routed-read-gib",
        type=float,
        default=0.0,
        help=(
            "reject request checks/generation when planned routed expert SSD "
            "reads exceed this GiB cap; <=0 disables the guard"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-ssd-read-gib-s",
        dest="prefill_ssd_read_gib_per_second",
        type=float,
        default=0.0,
        help="measured SSD read GiB/s for routed expert read-time estimates",
    )
    inspect_prepared.add_argument(
        "--prefill-max-routed-read-seconds",
        type=float,
        default=0.0,
        help=(
            "reject checks/generation when estimated routed expert SSD read "
            "time exceeds this cap; requires --prefill-ssd-read-gib-s"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-moe-token-block",
        type=_parse_moe_token_block,
        default="auto",
        metavar="N|auto",
        help="MoE token block for batch prompt prefill; default auto fits runner scratch",
    )
    inspect_prepared.add_argument(
        "--prefill-moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
        help=(
            "control checked prompt MoE output accumulation; env preserves "
            "LARGERLM_MOE_BATCH_ACCUMULATOR, file/memory pin it"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-static-capacity-per-expert",
        default="auto",
        metavar="N|auto|none",
        help=(
            "write LLMSCAP1 binary routes for checked prompt MoE; "
            "auto uses the prompt chunk size"
        ),
    )
    inspect_prepared.add_argument(
        "--prefill-allow-static-capacity-overflow",
        action="store_true",
        help="allow checked prompt static-capacity route artifacts to contain overflow records",
    )
    inspect_prepared.add_argument(
        "--prefill-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during checked prompt prefill",
    )
    inspect_prepared.add_argument(
        "--decode-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during checked single-token decode",
    )
    inspect_prepared.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="auto",
    )
    inspect_prepared.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        help="minimum prompt batch tokens before auto uses MPSGraph resident GEMMs",
    )
    inspect_prepared.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
        help="minimum resident matrix dimension before auto uses MPSGraph",
    )
    inspect_prepared.add_argument(
        "--prefill-router-hybrid-margin-threshold",
        type=float,
        default=0.0,
        help=(
            "when >0 and auto selects MPSGraph for router gates, keep custom "
            "Metal router output when its min effective margin exceeds this "
            "threshold"
        ),
    )
    _add_generation_memory_guard_args(inspect_prepared)
    inspect_prepared.add_argument(
        "--check-prompt-tokens",
        type=int,
        default=None,
        help="optionally check whether a token-id request with this prompt length is admitted",
    )
    inspect_prepared.add_argument(
        "--check-prompt",
        default=None,
        help="optionally tokenize this prompt text and check request admission",
    )
    inspect_prepared.add_argument(
        "--check-prompt-file",
        default=None,
        help="optionally tokenize this prompt file and check request admission",
    )
    inspect_prepared.add_argument(
        "--check-chat-messages",
        default=None,
        help="optionally render this JSON chat messages array and check admission",
    )
    inspect_prepared.add_argument(
        "--check-chat-messages-file",
        default=None,
        help="optionally render this JSON chat messages file and check admission",
    )
    inspect_prepared.add_argument(
        "--check-no-add-special-tokens",
        action="store_true",
        help="do not add tokenizer special tokens when using --check-prompt",
    )
    inspect_prepared.add_argument(
        "--check-no-add-generation-prompt",
        action="store_true",
        help="do not add the chat generation prompt when rendering chat messages",
    )
    inspect_prepared.add_argument(
        "--check-max-new-tokens",
        type=int,
        default=1,
        help="max_new_tokens value for the optional request check",
    )
    inspect_prepared.add_argument(
        "--check-logits-top-k",
        type=int,
        default=1,
        help="logits_top_k value for the optional request check",
    )
    inspect_prepared.add_argument(
        "--check-temperature",
        type=float,
        default=0.0,
        help="temperature value for the optional request check",
    )
    inspect_prepared.add_argument(
        "--check-top-p",
        type=float,
        default=1.0,
        help="top_p value for the optional request check",
    )
    inspect_prepared.add_argument(
        "--check-metal-final-logits",
        action="store_true",
        help="include metal_final_logits=true in the optional request check",
    )
    inspect_prepared.add_argument(
        "--metal-final-logits",
        dest="metal_final_logits",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    inspect_prepared.add_argument(
        "--check-runtime-preflight",
        action="store_true",
        help=(
            "with an optional request check, also run the deterministic "
            "generation runtime/live-memory budget check"
        ),
    )
    inspect_prepared.add_argument(
        "--require-launch-audit",
        action="store_true",
        help=(
            "fail unless the inspection proves a locked profile, prepared "
            "memory profile, GLM-5.2 readiness, prefill acceleration, request "
            "admission, and runtime memory preflight are all closed"
        ),
    )
    inspect_prepared.add_argument(
        "--write-launch-audit",
        default=None,
        help="write the launch_audit evidence JSON for later serve-prepared gating",
    )
    inspect_prepared.add_argument("--json", action="store_true")
    inspect_prepared.set_defaults(func=_inspect_prepared)

    serve = sub.add_parser(
        "serve-prepared",
        help="serve a prepared manifest through a small local JSON HTTP API",
    )
    serve.add_argument("prepared", help="prepared output directory or manifest.json")
    _add_launch_profile_arg(serve)
    serve.add_argument("--runner", default="metal/largerlm-runner")
    serve.add_argument("--model-config", default=None, help="override model config path")
    serve.add_argument("--tokenizer", default=None, help="override tokenizer directory or file")
    serve.add_argument(
        "--tokenizer-backend",
        choices=("auto", "simple", "tokenizers", "transformers", "sentencepiece"),
        default="auto",
    )
    serve.add_argument("--trust-remote-code", action="store_true")
    serve.add_argument("--served-model-name", default="largerlm-prepared")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-new-tokens-cap", type=int, default=256)
    serve.add_argument("--max-prompt-tokens", type=int, default=4096)
    serve.add_argument("--max-request-bytes", type=int, default=1024 * 1024)
    serve.add_argument("--logits-top-k-cap", type=int, default=64)
    serve.add_argument(
        "--require-launch-audit",
        default=None,
        metavar="PATH",
        help=(
            "require a passing inspect-prepared --write-launch-audit artifact "
            "for this prepared package and applied locked launch profile before "
            "starting the server"
        ),
    )
    _add_selected_replay_ssd_read_check_args(serve, subject="prepared server")
    _add_prepared_memory_profile_requirement_arg(serve)
    serve.add_argument(
        "--no-batch-prefill-prompt",
        action="store_true",
        help="disable automatic chunked prompt prefill for multi-token requests",
    )
    serve.add_argument(
        "--allow-missing-dsa-indexer",
        action="store_true",
        help="debug only: bypass the DSA cache schedule guard",
    )
    _add_prepared_glm_4bit_requirement_arg(serve)
    _add_prepared_ssd_read_default_arg(serve)
    _add_prefill_acceleration_requirement_arg(serve)
    _add_prefill_backend_probe_arg(serve)
    serve.add_argument("--max-cache-file-mib", type=float, default=32768.0)
    serve.add_argument("--max-cache-read-mib", type=float, default=256.0)
    serve.add_argument("--max-runner-scratch-mib", type=float, default=4096.0)
    _add_expert_read_advise_args(serve)
    serve.add_argument(
        "--prefill-prompt-chunk-tokens",
        type=_parse_prefill_prompt_chunk_tokens,
        default=0,
        metavar="N|auto",
        help="prompt tokens per server batch prefill chunk; default auto sizes from safety caps",
    )
    serve.add_argument("--prefill-max-prompt-batch-mib", type=float, default=1024.0)
    serve.add_argument("--prefill-max-cache-write-mib", type=float, default=4096.0)
    serve.add_argument("--prefill-max-stage-mib", type=float, default=4096.0)
    serve.add_argument("--prefill-max-compact-stage-mib", type=float, default=4096.0)
    serve.add_argument(
        "--prefill-max-stage-raw-ranges",
        type=int,
        default=0,
        help="reject server prompt prefill stage plans with more raw expert read ranges; 0 disables",
    )
    serve.add_argument(
        "--prefill-max-stage-coalesced-ranges",
        type=int,
        default=0,
        help="reject server prompt prefill stage plans with more coalesced expert read ranges; 0 disables",
    )
    serve.add_argument(
        "--prefill-expert-stage-tiling",
        action="store_true",
        help="split server prompt prefill routed experts into bounded stage/compact tiles",
    )
    serve.add_argument(
        "--prefill-persistent-moe-plan-server",
        action="store_true",
        help=(
            "keep one Metal routed-MoE plan runner alive across server prompt "
            "prefill routed layers and expert-stage tiles"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-resident-linear-server",
        action="store_true",
        help=(
            "keep one Metal resident-linear runner alive across server prompt "
            "prefill resident batch-linear requests"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-attention-projection-server",
        action="store_true",
        help=(
            "keep one Metal attention-projection runner alive across server "
            "prompt prefill fused projection requests"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-attention-output-server",
        action="store_true",
        help=(
            "keep one Metal attention-output runner alive across server prompt "
            "prefill fused output-projection requests"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-shared-expert-server",
        action="store_true",
        help=(
            "keep one Metal shared-expert runner alive across server prompt "
            "prefill fused shared-expert requests"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-rope-split-server",
        action="store_true",
        help=(
            "keep one Metal RoPE split runner alive across server prompt "
            "prefill fused RoPE split requests"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-mla-attention-server",
        action="store_true",
        help=(
            "keep one Metal MLA attention runner alive across server prompt "
            "prefill batch MLA attention requests"
        ),
    )
    serve.add_argument(
        "--prefill-persistent-rmsnorm-server",
        action="store_true",
        help=(
            "keep one Metal RMSNorm runner alive across server prompt "
            "prefill resident RMSNorm batch requests"
        ),
    )
    serve.add_argument("--prefill-copy-chunk-mib", type=float, default=8.0)
    serve.add_argument(
        "--prefill-stage-disk-margin-mib",
        type=float,
        default=0.0,
        help="extra free disk margin required before server prompt prefill stage files",
    )
    serve.add_argument(
        "--prefill-max-routed-read-amplification",
        type=float,
        default=0.0,
        help=(
            "reject generation when prompt chunking would exceed this routed "
            "expert read amplification; <=0 disables the guard"
        ),
    )
    serve.add_argument(
        "--prefill-max-routed-read-gib",
        type=float,
        default=0.0,
        help=(
            "reject generation when planned routed expert SSD reads exceed "
            "this GiB cap; <=0 disables the guard"
        ),
    )
    serve.add_argument(
        "--prefill-ssd-read-gib-s",
        dest="prefill_ssd_read_gib_per_second",
        type=float,
        default=0.0,
        help="measured SSD read GiB/s for routed expert read-time estimates",
    )
    serve.add_argument(
        "--prefill-max-routed-read-seconds",
        type=float,
        default=0.0,
        help=(
            "reject generation when estimated routed expert SSD read time "
            "exceeds this cap; requires --prefill-ssd-read-gib-s"
        ),
    )
    serve.add_argument(
        "--prefill-moe-token-block",
        type=_parse_moe_token_block,
        default="auto",
        metavar="N|auto",
        help="MoE token block for server batch prompt prefill; default auto fits runner scratch",
    )
    serve.add_argument(
        "--prefill-moe-output-accumulator",
        choices=("env", "file", "memory"),
        default="env",
        help=(
            "control server staged MoE output accumulation; env preserves "
            "LARGERLM_MOE_BATCH_ACCUMULATOR, file/memory pin it"
        ),
    )
    serve.add_argument(
        "--prefill-static-capacity-per-expert",
        default="auto",
        metavar="N|auto|none",
        help=(
            "write LLMSCAP1 binary routes for server batch prompt MoE; "
            "auto uses the prompt chunk size"
        ),
    )
    serve.add_argument(
        "--prefill-allow-static-capacity-overflow",
        action="store_true",
        help="allow server prompt static-capacity route artifacts to contain overflow records",
    )
    serve.add_argument(
        "--prefill-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during server prompt prefill",
    )
    serve.add_argument(
        "--decode-mla-key-cache",
        action="store_true",
        help="request opt-in MLA K_nope score-side cache during server single-token decode",
    )
    serve.add_argument(
        "--prefill-mla-kv-b-cache-dir",
        default=None,
        help="optional directory for per-layer f32 MLA kv_b caches during server prompt prefill",
    )
    serve.add_argument(
        "--prefill-linear-backend",
        choices=("custom-metal", "mpp-f32", "mpsgraph-f32", "mps-matrix-f32", "auto"),
        default="auto",
    )
    serve.add_argument(
        "--prefill-mpsgraph-min-batch-tokens",
        type=int,
        default=AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
        help="minimum prompt batch tokens before auto uses MPSGraph resident GEMMs",
    )
    serve.add_argument(
        "--prefill-mpsgraph-min-matrix-dim",
        type=int,
        default=AUTO_MPSGRAPH_MIN_DIM,
        help="minimum resident matrix dimension before auto uses MPSGraph",
    )
    serve.add_argument(
        "--prefill-router-hybrid-margin-threshold",
        type=float,
        default=0.0,
        help=(
            "when >0 and auto selects MPSGraph for router gates, keep custom "
            "Metal router output when its min effective margin exceeds this "
            "threshold"
        ),
    )
    _add_generation_memory_guard_args(serve)
    serve.add_argument(
        "--metal-final-logits",
        action="store_true",
        help="use Metal final logits by default for server generation requests",
    )
    serve.add_argument(
        "--metal-runtime-generation",
        action="store_true",
        help=(
            "opt in to glm_moe_infer JSONL runtime generation for server "
            "requests; currently greedy-only"
        ),
    )
    serve.add_argument(
        "--metal-runtime-cache-mla-kv-b-f32",
        action="store_true",
        help=(
            "cache absorbed MLA KV-B F32 views inside the persistent "
            "glm_moe_infer runtime"
        ),
    )
    serve.add_argument(
        "--metal-runtime-max-mla-kv-b-cache-mib",
        type=float,
        default=0.0,
        help=(
            "required positive cache cap when "
            "--metal-runtime-cache-mla-kv-b-f32 is enabled"
        ),
    )
    serve.add_argument(
        "--metal-runtime-expert-pin-plan",
        default=None,
        help="quality-preserving hot-expert plan for the persistent Metal runtime",
    )
    serve.add_argument(
        "--metal-runtime-max-adaptive-expert-cache-gib",
        type=float,
        default=0.0,
        help="override the plan's evictable expert cache budget",
    )
    serve.add_argument(
        "--metal-runtime-mmap-final-logits",
        action="store_true",
        help="mmap lm_head ranges as Metal buffers for runtime final logits",
    )
    serve.add_argument(
        "--metal-runtime-context1-o-proj-cache-layout",
        default=None,
        help=(
            "validated context=1 o_proj*B_v collapsed-cache layout for the "
            "persistent glm_moe_infer runtime"
        ),
    )
    serve.add_argument(
        "--metal-runtime-context1-o-proj-cache-file",
        default=None,
        help=(
            "optional override backing file for "
            "--metal-runtime-context1-o-proj-cache-layout"
        ),
    )
    serve.add_argument("--metal-binary", default="metal/glm_moe_infer")
    serve.add_argument("--quiet-runner", action="store_true")
    serve.set_defaults(func=_serve_prepared)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        raw_argv = _expand_launch_profile_args(raw_argv)
        raw_argv = _expand_prepare_flags_args(raw_argv)
    except CliArgumentError as exc:
        print(f"largerlm: {exc}", file=sys.stderr)
        return 1
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except (
        ArtifactStatusError,
        BaselineError,
        BenchmarkError,
        CliArgumentError,
        ConfigError,
        Context1OProjCacheError,
        DecodeCacheError,
        DecodeDriverError,
        DiskBenchmarkError,
        DSAIndexerError,
        EmbeddingError,
        ExpertIOPlanError,
        FinalLogitsError,
        MetalGenerateError,
        MetalTextGenerationError,
        MlxBaselineError,
        PackerError,
        PlannerError,
        PreparedRunLockError,
        PrefillBackendError,
        PrefillExecuteError,
        PrefillPlanError,
        PromptPrefillError,
        PreflightError,
        PrepareError,
        PreparedManifestError,
        PreparedServerError,
        ResidentPackerError,
        RuntimeCheckError,
        SafetensorsError,
        SafetyError,
        StagedMoEError,
        TextGenerationError,
        TokenGeneratorError,
        TokenizerError,
        OSError,
    ) as exc:
        print(f"largerlm: {exc}", file=sys.stderr)
        return 1
