from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from largerlm.cli import _merged_prefill_runtime_policy_flags, main as cli_main
from largerlm.prefill_backend import PrefillBackendCapability
from largerlm.prefill_execute import ResidentLinearCalibrationResult
from largerlm.prefill_plan import PrefillPlanError, build_prefill_plan


FIXTURES = Path(__file__).parent / "fixtures"


def test_prefill_plan_estimates_glm_gemm_shapes() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        dtype_bits=16,
    )

    by_name = {op.name: op for op in plan.ops}
    assert plan.num_layers == 6
    assert plan.public_glm_5_2_shape["matches"] is False
    assert "hidden_size" in plan.public_glm_5_2_shape["mismatched_fields"]
    assert plan.suggested_public_glm_5_2_shape_guard_flags is None
    assert plan.moe_layers == 5
    assert plan.dense_layers == 1
    assert len(plan.ops) == 15
    assert plan.mpp_candidate_ops == 11
    assert plan.total_flops > 0
    assert plan.total_weight_bytes > 0
    assert plan.routed_expert_backend_hint == "ssd_full_layer_expert_sweep"
    assert by_name["attention.q_a_proj"].tile_plan is not None
    assert by_name["attention.q_a_proj"].tile_plan.threadgroup_tile_m == 64
    assert by_name["attention.q_a_proj"].tile_plan.threadgroup_tile_n == 64
    assert by_name["attention.q_a_proj"].tile_plan.grid_m == 2
    assert by_name["attention.q_a_proj"].tile_plan.grid_n == 1
    assert by_name["attention.q_a_proj"].tile_plan.edge_threadgroup_tiles == 2
    assert by_name["attention.q_a_proj"].layers == 6
    assert by_name["attention.q_a_proj"].m_tokens == 128
    assert by_name["attention.q_a_proj"].k_in == 128
    assert by_name["attention.q_a_proj"].n_out == 32
    assert by_name["moe.router"].backend_hint == "custom_metal_or_small_gemm"
    assert by_name["dsa.index_wk"].layers == 3
    assert by_name["dsa.index_wk"].k_in == 128
    assert by_name["dsa.index_wk"].n_out == 16
    assert by_name["dsa.index_wq_b"].layers == 3
    assert by_name["dsa.index_wq_b"].k_in == 32
    assert by_name["dsa.index_wq_b"].n_out == 64
    assert by_name["dsa.index_weights_proj"].layers == 3
    assert by_name["dsa.index_weights_proj"].k_in == 128
    assert by_name["dsa.index_weights_proj"].n_out == 4


def test_prefill_plan_estimates_glm_5_2_routed_expert_streaming() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_5_2_config.json",
        prompt_tokens=4096,
        dtype_bits=16,
        expert_bits=4,
        group_size=64,
        max_runner_scratch_bytes=4096 * 1024 * 1024,
    )
    by_name = {op.name: op for op in plan.ops}

    assert plan.public_glm_5_2_shape["matches"] is True
    assert plan.public_glm_5_2_shape["mismatched_fields"] == ()
    assert plan.suggested_public_glm_5_2_shape_guard_flags == {
        "source": "prefill_plan",
        "require_public_glm_5_2_shape": True,
        "argv": ("--require-public-glm-5-2-shape",),
    }
    assert plan.suggested_launch_profile is not None
    assert (
        plan.suggested_launch_profile["sections"][
            "public_glm_5_2_shape_guard_flags"
        ]
        == plan.suggested_public_glm_5_2_shape_guard_flags
    )
    assert "--require-public-glm-5-2-shape" in plan.suggested_launch_profile["argv"]
    assert plan.moe_layers == 75
    assert plan.dense_layers == 3
    assert plan.mpp_candidate_ops == 15
    assert len(plan.prefill_backend_candidates) == plan.mpp_candidate_ops
    assert [candidate.rank for candidate in plan.prefill_backend_candidates] == list(
        range(1, plan.mpp_candidate_ops + 1)
    )
    assert plan.prefill_backend_candidates[0].op_name == "attention.o_proj"
    assert plan.prefill_backend_candidates[0].total_flops == by_name[
        "attention.o_proj"
    ].total_flops
    assert plan.prefill_backend_candidates[0].execution_path == (
        "mpp_tensor_ops_gpu_neural_accelerator"
    )
    assert plan.prefill_backend_candidates[0].availability == "not_inspected"
    assert plan.prefill_backend_candidates[-1].op_name == "dsa.index_weights_proj"
    assert len(plan.prefill_linear_calibration_shapes) == 8
    calibration_shape = plan.prefill_linear_calibration_shapes[0]
    assert calibration_shape.candidate_rank == 1
    assert calibration_shape.op_name == "attention.o_proj"
    assert calibration_shape.batch_tokens == 4096
    assert calibration_shape.in_dim == 16384
    assert calibration_shape.out_dim == 6144
    assert calibration_shape.matrix_shape_arg == "16384x6144"
    assert calibration_shape.calibration_case_bytes == (
        16384 * 6144 * 4 + 4096 * 16384 * 4 + 4096 * 6144 * 4
    )
    assert calibration_shape.candidate_count == 1
    assert calibration_shape.candidate_ranks == (1,)
    assert calibration_shape.candidate_op_names == ("attention.o_proj",)
    assert calibration_shape.candidate_total_flops == by_name[
        "attention.o_proj"
    ].total_flops
    coverage = plan.prefill_linear_calibration_candidate_coverage
    assert coverage is not None
    assert coverage["source"] == "prefill_plan"
    assert coverage["candidate_count"] == plan.mpp_candidate_ops
    assert coverage["covered_candidate_count"] == 11
    assert coverage["covered_candidate_ranks"] == tuple(range(1, 12))
    assert coverage["uncovered_candidate_ranks"] == (12, 13, 14, 15)
    assert coverage["unique_shape_count"] == 8
    assert coverage["coverage_truncated"] is True
    assert coverage["covered_candidate_flop_fraction"] > 0.98
    assert coverage["shapes"][3]["matrix_shape_arg"] == "6144x2048"
    assert coverage["shapes"][3]["candidate_ranks"] == (4, 6, 7)
    assert coverage["shapes"][3]["candidate_op_names"] == (
        "attention.q_a_proj",
        "moe.shared_gate_proj",
        "moe.shared_up_proj",
    )
    assert plan.suggested_prefill_linear_calibration_flags is not None
    assert plan.suggested_prefill_linear_calibration_flags["command"] == (
        "prefill-linear-calibrate"
    )
    assert plan.suggested_prefill_linear_calibration_flags["argv"][:4] == (
        "--batch-tokens",
        "4096",
        "--matrix-shapes",
        (
            "16384x6144,2048x16384,512x28672,6144x2048,"
            "2048x6144,6144x576,12288x6144,6144x12288"
        ),
    )
    assert plan.suggested_prefill_linear_calibration_flags[
        "max_calibration_case_mib"
    ] == 810
    assert plan.routed_expert_slot_bytes == 21_233_664
    assert plan.routed_expert_assignments_per_moe_layer == 32_768
    assert plan.routed_expert_unique_per_moe_layer == 256
    assert plan.routed_expert_read_bytes == 407_686_348_800
    assert plan.routed_expert_chunked_read_bytes == plan.routed_expert_read_bytes
    assert plan.routed_expert_read_chunks_per_prompt == 1
    assert plan.routed_expert_read_cost_plan is not None
    assert plan.routed_expert_read_cost_plan.baseline_read_bytes == 407_686_348_800
    assert plan.routed_expert_read_cost_plan.planned_read_bytes == 407_686_348_800
    assert plan.routed_expert_read_cost_plan.extra_read_bytes == 0
    assert plan.routed_expert_read_cost_plan.read_amplification == 1.0
    assert plan.routed_expert_read_cost_plan.planned_read_seconds is None
    assert plan.routed_expert_backend_hint == "ssd_full_layer_expert_sweep"
    assert plan.cache_io_plan is not None
    causal_rows = 4096 * 4097 // 2
    indexed_rows = 2048 * 2049 // 2 + (4096 - 2048) * 2048
    assert plan.cache_io_plan.mla_cache_width == 576
    assert plan.cache_io_plan.dsa_index_head_dim == 128
    assert plan.cache_io_plan.dsa_index_topk == 2048
    assert plan.cache_io_plan.indexed_attention_layers == 78
    assert plan.cache_io_plan.full_attention_layers == 0
    assert plan.cache_io_plan.dsa_full_indexer_layers == 21
    assert plan.cache_io_plan.causal_rows_per_layer == causal_rows
    assert plan.cache_io_plan.indexed_rows_per_layer == indexed_rows
    assert plan.cache_io_plan.mla_cache_read_bytes == 78 * indexed_rows * 576 * 2
    assert plan.cache_io_plan.dsa_index_cache_read_bytes == 21 * causal_rows * 128 * 2
    assert (
        plan.cache_io_plan.total_cache_read_bytes
        == plan.cache_io_plan.mla_cache_read_bytes
        + plan.cache_io_plan.dsa_index_cache_read_bytes
    )
    assert plan.cache_io_plan.mla_cache_write_bytes == 78 * 4096 * 576 * 2
    assert plan.cache_io_plan.dsa_index_cache_write_bytes == 21 * 4096 * 128 * 2
    assert (
        plan.cache_io_plan.total_cache_write_bytes
        == plan.cache_io_plan.mla_cache_write_bytes
        + plan.cache_io_plan.dsa_index_cache_write_bytes
    )
    assert plan.routed_expert_capacity_plan is not None
    assert plan.routed_expert_capacity_plan.capacity_tokens == 4096
    assert plan.routed_expert_capacity_plan.chunks_per_prompt == 1
    assert plan.routed_expert_capacity_plan.assignments_per_capacity_chunk == 32_768
    assert plan.routed_expert_capacity_plan.unique_experts_per_capacity_chunk == 256
    assert plan.routed_expert_capacity_plan.balanced_capacity_per_expert == 128
    assert (
        plan.routed_expert_capacity_plan.balanced_capacity_assignments_per_capacity_chunk
        == 32_768
    )
    assert (
        plan.routed_expert_capacity_plan.balanced_capacity_overprovision_assignments_per_capacity_chunk
        == 0
    )
    assert plan.routed_expert_capacity_plan.balanced_capacity_utilization == 1.0
    assert (
        plan.routed_expert_capacity_plan.balanced_capacity_activation_bytes_per_moe_layer
        == 2_147_483_648
    )
    assert plan.routed_expert_capacity_plan.spill_free_capacity_per_expert == 4096
    assert (
        plan.routed_expert_capacity_plan.spill_free_capacity_assignments_per_capacity_chunk
        == 1_048_576
    )
    assert (
        plan.routed_expert_capacity_plan.spill_free_capacity_overprovision_assignments_per_capacity_chunk
        == 1_015_808
    )
    assert plan.routed_expert_capacity_plan.spill_free_capacity_utilization == pytest.approx(0.03125)
    assert (
        plan.routed_expert_capacity_plan.spill_free_capacity_activation_bytes_per_moe_layer
        == 68_719_476_736
    )
    assert plan.routed_expert_capacity_plan.requires_overflow_path_for_balanced_capacity
    assert (
        plan.routed_expert_capacity_plan.backend_hint
        == "static_capacity_full_expert_sweep_candidate"
    )
    assert plan.staged_moe_runner_scratch_plan is not None
    assert plan.staged_moe_runner_scratch_plan.assignment_table_bytes_per_moe_layer == 524_288
    assert plan.staged_moe_runner_scratch_plan.token_seen_bytes_per_moe_layer == 4_096
    assert plan.staged_moe_runner_scratch_plan.max_expert_tokens_per_capacity_chunk == 4_096
    assert plan.staged_moe_runner_scratch_plan.auto_token_block == 4_096
    assert (
        plan.staged_moe_runner_scratch_plan.token_block_buffer_bytes_per_moe_layer
        == 402_669_568
    )
    assert plan.staged_moe_runner_scratch_plan.expert_slot_alloc_bytes == 23_068_672
    assert (
        plan.staged_moe_runner_scratch_plan.estimated_peak_bytes_per_moe_layer
        == 426_266_624
    )
    assert plan.staged_moe_runner_scratch_plan.fits_runner_scratch is True
    assert plan.total_weight_bytes > plan.resident_gemm_weight_bytes
    assert by_name["attention.q_a_proj"].k_in == 6144
    assert by_name["attention.q_a_proj"].n_out == 2048
    assert by_name["attention.q_a_proj"].tile_plan is not None
    assert by_name["attention.q_a_proj"].tile_plan.grid_m == 64
    assert by_name["attention.q_a_proj"].tile_plan.grid_n == 32
    assert by_name["attention.q_a_proj"].tile_plan.edge_threadgroup_tiles == 0
    assert by_name["attention.q_a_proj"].tile_plan.static_extent_full_tiles
    assert by_name["attention.o_proj"].k_in == 16384
    assert by_name["dsa.index_wk"].layers == 21
    assert by_name["dsa.index_wk"].k_in == 6144
    assert by_name["dsa.index_wk"].n_out == 128
    assert by_name["dsa.index_wq_b"].layers == 21
    assert by_name["dsa.index_wq_b"].k_in == 2048
    assert by_name["dsa.index_wq_b"].n_out == 4096
    assert by_name["dsa.index_weights_proj"].layers == 21
    assert by_name["dsa.index_weights_proj"].k_in == 6144
    assert by_name["dsa.index_weights_proj"].n_out == 32


def test_prefill_plan_public_glm_5_2_shape_guard_requires_4bit_suggestion() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_5_2_config.json",
        prompt_tokens=128,
        dtype_bits=16,
        expert_bits=8,
        group_size=64,
    )

    assert plan.public_glm_5_2_shape["matches"] is True
    assert plan.suggested_public_glm_5_2_shape_guard_flags is None
    if plan.suggested_launch_profile is not None:
        assert "public_glm_5_2_shape_guard_flags" not in (
            plan.suggested_launch_profile["sections"]
        )
        assert "--require-public-glm-5-2-shape" not in (
            plan.suggested_launch_profile["argv"]
        )


def test_prefill_plan_rejects_invalid_prompt_tokens() -> None:
    with pytest.raises(PrefillPlanError, match="prompt_tokens"):
        build_prefill_plan(FIXTURES / "glm_moe_dsa_config.json", prompt_tokens=0)


def test_prefill_plan_rejects_prompt_above_model_max_position() -> None:
    with pytest.raises(PrefillPlanError, match="max_position_embeddings"):
        build_prefill_plan(FIXTURES / "glm_moe_dsa_config.json", prompt_tokens=1025)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"prompt_tokens": True}, "prompt_tokens must be an integer"),
        ({"prompt_tokens": 1.5}, "prompt_tokens must be an integer"),
        ({"dtype_bits": True}, "dtype_bits must be an integer"),
        ({"expert_bits": False}, "expert_bits must be an integer"),
        ({"group_size": 1.5}, "group_size must be an integer"),
        ({"mpp_min_tokens": True}, "mpp_min_tokens must be an integer"),
        ({"compile_mpp_probe": 1}, "compile_mpp_probe must be a boolean"),
        ({"simdgroup_tile_m": False}, "simdgroup_tile_m must be an integer"),
        ({"simdgroup_tile_n": 1.5}, "simdgroup_tile_n must be an integer"),
        ({"simdgroups_m": False}, "simdgroups_m must be an integer"),
        ({"simdgroups_n": 1.5}, "simdgroups_n must be an integer"),
        ({"k_tile": True}, "k_tile must be an integer"),
        (
            {"max_prefill_activation_bytes": False},
            "max_prefill_activation_bytes must be an integer",
        ),
        (
            {"max_runner_scratch_bytes": 1.5},
            "max_runner_scratch_bytes must be an integer",
        ),
        (
            {"expert_stage_align_bytes": True},
            "expert_stage_align_bytes must be an integer",
        ),
        (
            {"probe_timeout_seconds": 0},
            "probe_timeout_seconds must be positive",
        ),
        (
            {"probe_timeout_seconds": True},
            "probe_timeout_seconds must be positive",
        ),
    ),
)
def test_prefill_plan_rejects_non_integer_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    params: dict[str, object] = {"prompt_tokens": 128}
    params.update(kwargs)

    with pytest.raises(PrefillPlanError, match=message):
        build_prefill_plan(FIXTURES / "glm_moe_dsa_config.json", **params)


def test_prefill_plan_rejects_non_positive_stage_alignment() -> None:
    with pytest.raises(PrefillPlanError, match="expert_stage_align_bytes"):
        build_prefill_plan(
            FIXTURES / "glm_moe_dsa_config.json",
            prompt_tokens=128,
            expert_stage_align_bytes=0,
        )


@pytest.mark.parametrize("value", (True, 0, -1, float("nan"), "fast"))
def test_prefill_plan_rejects_invalid_ssd_read_bandwidth(value: object) -> None:
    with pytest.raises(PrefillPlanError, match="ssd_read_gib_per_second"):
        build_prefill_plan(
            FIXTURES / "glm_moe_dsa_config.json",
            prompt_tokens=128,
            ssd_read_gib_per_second=value,
        )


def test_prefill_plan_counts_effective_metal4_candidates() -> None:
    capability = PrefillBackendCapability(
        sdk_path=None,
        metal_headers_available=True,
        metal4_headers_available=True,
        metal_tensor_headers_available=True,
        metal_tensor_int4_declared=True,
        metal4_machine_learning_declared=True,
        mps_graph_matmul_declared=True,
        mpp_tensor_ops_symbol_declared=False,
        host_probe_ran=True,
        host_probe_ok=True,
        mpp_compile_probe_ran=False,
        mpp_compile_probe_ok=None,
        mpp_compile_variant=None,
        mpp_compile_error=None,
        device_name="Apple M5 Max",
        supports_metal4_family=True,
        responds_new_mtl4_command_queue=True,
        responds_new_tensor=True,
        responds_tensor_size_align=True,
        responds_new_compiler=True,
        can_allocate_tiny_ml_tensor=True,
        tensor_error=None,
        recommended_backend="mpsgraph_prefill_fallback",
        reasons=(),
    )

    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        backend_capability=capability,
    )

    assert plan.mpp_candidate_ops == 11
    assert plan.effective_metal4_candidate_ops == 11
    assert plan.effective_mpp_candidate_ops == 0
    assert plan.prefill_backend_candidates[0].preferred_backend == (
        "mpsgraph_prefill_fallback"
    )
    assert plan.prefill_backend_candidates[0].execution_path == "mpsgraph_gpu_matmul"
    assert plan.prefill_backend_candidates[0].availability == "fallback"
    assert "mpp::tensor_ops" in plan.prefill_backend_candidates[0].reason
    profile = plan.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_acceleration_flags"] == {
        "source": "prefill_plan",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_acceleration_runtimes": ("mpsgraph-f32",),
        "selectable_accelerated_prefill_backends": ("mpsgraph-f32",),
        "validated_accelerated_prefill_backends": (),
        "prefill_acceleration_runtime_gaps": (),
        "prefill_neural_accelerator_status": (
            capability.prefill_neural_accelerator_status
        ),
        "runtime_probe_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": False,
        "runtime_probe_argv": ("--run-mpsgraph-probe",),
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }
    assert "--prefill-linear-backend" in profile["argv"]
    assert "--require-prefill-acceleration" in profile["argv"]
    assert "--run-mpsgraph-probe" in profile["argv"]


def test_prefill_plan_uses_custom_path_when_mpsgraph_runtime_fails() -> None:
    capability = PrefillBackendCapability(
        sdk_path=None,
        metal_headers_available=True,
        metal4_headers_available=True,
        metal_tensor_headers_available=True,
        metal_tensor_int4_declared=True,
        metal4_machine_learning_declared=True,
        mps_graph_matmul_declared=True,
        mpp_tensor_ops_symbol_declared=False,
        host_probe_ran=True,
        host_probe_ok=False,
        mpp_compile_probe_ran=False,
        mpp_compile_probe_ok=None,
        mpp_compile_variant=None,
        mpp_compile_error=None,
        device_name=None,
        supports_metal4_family=False,
        responds_new_mtl4_command_queue=False,
        responds_new_tensor=False,
        responds_tensor_size_align=False,
        responds_new_compiler=False,
        can_allocate_tiny_ml_tensor=False,
        tensor_error="no Metal device",
        recommended_backend="custom_metal_prefill_fallback",
        reasons=("host probe could not create a default Metal device",),
        host_probe_requested=True,
    )

    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        backend_capability=capability,
    )

    assert not capability.mps_graph_runtime_available
    assert plan.prefill_backend_candidates[0].preferred_backend == "custom_metal_tile"
    assert plan.prefill_backend_candidates[0].execution_path == (
        "custom_metal_gpu_fallback"
    )
    assert plan.prefill_backend_candidates[0].availability == "fallback"
    assert "default Metal device" in plan.prefill_backend_candidates[0].reason


def test_prefill_plan_launch_profile_keeps_compile_mpp_probe_flag() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        compile_mpp_probe=True,
    )

    profile = plan.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_backend_probe_flags"] == {
        "source": "prefill_plan",
        "compile_mpp_probe": True,
        "argv": ("--compile-mpp-probe",),
    }
    assert "--compile-mpp-probe" in profile["argv"]


def test_prefill_plan_launch_profile_keeps_mpsgraph_probe_flag() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        run_mpsgraph_probe=True,
    )

    profile = plan.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_backend_probe_flags"] == {
        "source": "prefill_plan",
        "run_mpsgraph_probe": True,
        "argv": ("--run-mpsgraph-probe",),
    }
    assert "--run-mpsgraph-probe" in profile["argv"]


def test_prefill_plan_launch_profile_keeps_run_mpp_probe_flag() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        run_mpp_probe=True,
    )

    profile = plan.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_backend_probe_flags"] == {
        "source": "prefill_plan",
        "run_mpp_probe": True,
        "argv": ("--run-mpp-probe",),
    }
    assert "--run-mpp-probe" in profile["argv"]


def test_prefill_plan_launch_profile_keeps_nondefault_probe_timeout() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        run_mpsgraph_probe=True,
        probe_timeout_seconds=12.5,
    )

    profile = plan.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_backend_probe_flags"] == {
        "source": "prefill_plan",
        "run_mpsgraph_probe": True,
        "prefill_backend_probe_timeout_seconds": 12.5,
        "argv": (
            "--run-mpsgraph-probe",
            "--prefill-backend-probe-timeout-seconds",
            "12.5",
        ),
    }
    assert "--run-mpsgraph-probe" in profile["argv"]
    assert "--prefill-backend-probe-timeout-seconds" in profile["argv"]


def test_prefill_plan_launch_profile_keeps_prefill_runtime_policy() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        prefill_linear_backend="mpsgraph-f32",
        prefill_mpsgraph_min_batch_tokens=64,
        prefill_mpsgraph_min_matrix_dim=16,
        prefill_min_accelerated_flop_fraction=0.5,
    )

    policy = plan.suggested_prefill_runtime_policy_flags
    assert policy == {
        "source": "prefill_plan",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_mpsgraph_min_batch_tokens": 64,
        "prefill_mpsgraph_min_matrix_dim": 16,
        "prefill_min_accelerated_flop_fraction": 0.5,
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
        ),
    }
    profile = plan.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_runtime_policy_flags"] == policy
    assert "--prefill-linear-backend" in profile["argv"]
    assert "--prefill-min-accelerated-flop-fraction" in profile["argv"]


def test_prefill_plan_calibration_merge_uses_calibrated_backend_when_plan_auto() -> None:
    merged = _merged_prefill_runtime_policy_flags(
        plan_policy={
            "source": "prefill_plan",
            "prefill_linear_backend": "auto",
            "prefill_mpsgraph_min_batch_tokens": 2048,
            "prefill_mpsgraph_min_matrix_dim": 4096,
        },
        calibration_policy={
            "source": "prefill_linear_calibration",
            "prefill_linear_backend": "mps-matrix-f32",
            "prefill_mpsgraph_min_batch_tokens": 128,
            "prefill_mpsgraph_min_matrix_dim": 32,
        },
    )

    assert merged == {
        "source": "prefill_plan_calibration",
        "plan_source": "prefill_plan",
        "calibration_source": "prefill_linear_calibration",
        "prefill_linear_backend": "mps-matrix-f32",
        "prefill_mpsgraph_min_batch_tokens": 128,
        "prefill_mpsgraph_min_matrix_dim": 32,
        "argv": (
            "--prefill-linear-backend",
            "mps-matrix-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
        ),
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"prefill_linear_backend": "ane"}, "prefill_linear_backend"),
        ({"prefill_mpsgraph_min_batch_tokens": 0}, "prefill_mpsgraph_min_batch_tokens"),
        ({"prefill_mpsgraph_min_matrix_dim": False}, "prefill_mpsgraph_min_matrix_dim"),
        (
            {"prefill_min_accelerated_flop_fraction": 1.5},
            "prefill_min_accelerated_flop_fraction",
        ),
        (
            {"prefill_static_capacity_per_expert": 0},
            "prefill_static_capacity_per_expert",
        ),
        ({"require_prefill_acceleration": 1}, "require_prefill_acceleration"),
        ({"require_public_glm_5_2_shape": 1}, "require_public_glm_5_2_shape"),
    ),
)
def test_prefill_plan_rejects_invalid_runtime_policy(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(PrefillPlanError, match=message):
        build_prefill_plan(
            FIXTURES / "glm_moe_dsa_config.json",
            prompt_tokens=128,
            **kwargs,
        )


def test_prefill_plan_can_disable_static_capacity_profile_flag() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_moe_dsa_config.json",
        prompt_tokens=128,
        prefill_static_capacity_per_expert="none",
    )

    assert plan.routed_stage_temp_plan is not None
    assert plan.routed_stage_temp_plan.static_capacity_per_expert is None
    assert plan.routed_stage_temp_plan.max_static_capacity_binary_bytes == 0
    assert plan.routed_stage_temp_plan.max_stage_raw_ranges == 4
    assert plan.routed_stage_temp_plan.max_stage_coalesced_ranges == 4
    assert plan.suggested_stage_temp_guard_flags is not None
    assert plan.suggested_stage_temp_guard_flags["prefill_max_stage_raw_ranges"] == 5
    assert (
        plan.suggested_stage_temp_guard_flags["prefill_max_stage_coalesced_ranges"]
        == 5
    )
    assert "prefill_static_capacity_per_expert" not in (
        plan.suggested_stage_temp_guard_flags
    )
    assert "--prefill-static-capacity-per-expert" not in (
        plan.suggested_stage_temp_guard_flags["argv"]
    )


def test_prefill_plan_require_public_glm_5_2_shape_rejects_non_public() -> None:
    with pytest.raises(
        PrefillPlanError,
        match="config does not match the public GLM-5.2 shape",
    ) as exc_info:
        build_prefill_plan(
            FIXTURES / "glm_moe_dsa_config.json",
            prompt_tokens=128,
            require_public_glm_5_2_shape=True,
        )

    assert "hidden_size" in str(exc_info.value)


def test_prefill_plan_require_public_glm_5_2_shape_rejects_non_4bit() -> None:
    with pytest.raises(
        PrefillPlanError,
        match="public GLM-5.2 prefill planning requires expert_bits=4",
    ):
        build_prefill_plan(
            FIXTURES / "glm_5_2_config.json",
            prompt_tokens=128,
            expert_bits=8,
            require_public_glm_5_2_shape=True,
        )


def test_prefill_plan_recommends_activation_chunking() -> None:
    plan = build_prefill_plan(
        FIXTURES / "glm_5_2_config.json",
        prompt_tokens=4096,
        max_prefill_activation_bytes=128 * 1024 * 1024,
        ssd_read_gib_per_second=14.0,
    )

    assert plan.chunk_plan is not None
    assert plan.chunk_plan.recommended_chunk_tokens == 2240
    assert plan.chunk_plan.chunks == 2
    assert plan.chunk_plan.tile_aligned
    assert plan.chunk_plan.activation_limited
    assert plan.chunk_plan.estimated_peak_activation_bytes <= 128 * 1024 * 1024
    assert plan.routed_expert_chunked_read_bytes == 815_372_697_600
    assert plan.routed_expert_read_chunks_per_prompt == 2
    assert plan.routed_expert_read_cost_plan is not None
    assert plan.routed_expert_read_cost_plan.baseline_read_bytes == 407_686_348_800
    assert plan.routed_expert_read_cost_plan.planned_read_bytes == 815_372_697_600
    assert plan.routed_expert_read_cost_plan.extra_read_bytes == 407_686_348_800
    assert plan.routed_expert_read_cost_plan.read_amplification == 2.0
    assert plan.routed_expert_read_cost_plan.chunks_per_prompt == 2
    assert plan.routed_expert_read_cost_plan.ssd_read_bytes_per_second == 14.0 * 1024**3
    assert plan.routed_expert_read_cost_plan.planned_read_seconds == pytest.approx(
        815_372_697_600 / (14.0 * 1024**3)
    )
    assert plan.routed_expert_read_cost_plan.extra_read_seconds == pytest.approx(
        407_686_348_800 / (14.0 * 1024**3)
    )
    assert plan.suggested_guard_flags is not None
    assert plan.suggested_guard_flags["source"] == "prefill_plan"
    assert plan.suggested_guard_flags["prefill_prompt_chunk_tokens"] == 2240
    assert plan.suggested_guard_flags["prefill_max_routed_read_amplification"] == 2.1
    assert plan.suggested_guard_flags["prefill_max_routed_read_gib"] == pytest.approx(
        815_372_697_600 / 1024**3 * 1.05
    )
    assert plan.suggested_guard_flags["prefill_ssd_read_gib_per_second"] == 14.0
    assert plan.suggested_guard_flags["prefill_max_routed_read_seconds"] == pytest.approx(
        815_372_697_600 / (14.0 * 1024**3) * 1.05
    )
    assert plan.suggested_guard_flags["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "2240",
        "--prefill-max-routed-read-amplification",
        "2.1",
        "--prefill-max-routed-read-gib",
        "797.344",
        "--prefill-ssd-read-gib-s",
        "14",
        "--prefill-max-routed-read-seconds",
        "56.9531",
    )
    assert plan.suggested_stage_temp_guard_flags is not None
    assert plan.suggested_stage_temp_guard_flags["source"] == "prefill_plan"
    assert plan.suggested_stage_temp_guard_flags["prefill_prompt_chunk_tokens"] == 2240
    assert plan.routed_stage_temp_plan is not None
    assert plan.routed_stage_temp_plan.static_capacity_per_expert == "auto"
    assert plan.routed_stage_temp_plan.max_static_capacity_per_expert == 2240
    assert plan.routed_stage_temp_plan.static_capacity_strict_overflow_safe is True
    assert plan.routed_stage_temp_plan.max_stage_raw_ranges == 256
    assert plan.routed_stage_temp_plan.max_stage_coalesced_ranges == 256
    assert (
        plan.suggested_stage_temp_guard_flags["profile_max_stage_bytes"]
        == 5_436_866_560
    )
    assert (
        plan.suggested_stage_temp_guard_flags["profile_max_compact_stage_bytes"]
        == 5_435_817_984
    )
    assert (
        plan.suggested_stage_temp_guard_flags["profile_max_static_capacity_binary_bytes"]
        == 6_882_344
    )
    assert (
        plan.suggested_stage_temp_guard_flags[
            "profile_total_static_capacity_binary_bytes"
        ]
        == 943_878_000
    )
    assert (
        plan.suggested_stage_temp_guard_flags[
            "profile_max_stage_plus_compact_plus_static_bytes"
        ]
        == 10_879_566_888
    )
    assert plan.suggested_stage_temp_guard_flags["profile_max_stage_raw_ranges"] == 256
    assert (
        plan.suggested_stage_temp_guard_flags["prefill_max_stage_raw_ranges"] == 269
    )
    assert (
        plan.suggested_stage_temp_guard_flags["profile_max_stage_coalesced_ranges"]
        == 256
    )
    assert (
        plan.suggested_stage_temp_guard_flags[
            "prefill_max_stage_coalesced_ranges"
        ]
        == 269
    )
    assert plan.suggested_stage_temp_guard_flags["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "2240",
        "--prefill-max-stage-mib",
        "5444.25",
        "--prefill-max-compact-stage-mib",
        "5443.2",
        "--prefill-static-capacity-per-expert",
        "auto",
        "--prefill-max-stage-raw-ranges",
        "269",
        "--prefill-max-stage-coalesced-ranges",
        "269",
    )
    assert plan.suggested_prefill_guard_flags is not None
    assert plan.suggested_prefill_guard_flags["source"] == "prefill_plan"
    assert plan.suggested_prefill_guard_flags["prefill_prompt_chunk_tokens"] == 2240
    combined_argv = plan.suggested_prefill_guard_flags["argv"]
    assert combined_argv.count("--prefill-prompt-chunk-tokens") == 1
    assert combined_argv == (
        "--prefill-prompt-chunk-tokens",
        "2240",
        "--prefill-max-routed-read-amplification",
        "2.1",
        "--prefill-max-routed-read-gib",
        "797.344",
        "--prefill-ssd-read-gib-s",
        "14",
        "--prefill-max-routed-read-seconds",
        "56.9531",
        "--prefill-max-stage-mib",
        "5444.25",
        "--prefill-max-compact-stage-mib",
        "5443.2",
        "--prefill-static-capacity-per-expert",
        "auto",
        "--prefill-max-stage-raw-ranges",
        "269",
        "--prefill-max-stage-coalesced-ranges",
        "269",
    )
    assert plan.suggested_prefill_guard_flags["routed_read_guard"] == (
        plan.suggested_guard_flags
    )
    assert plan.suggested_prefill_guard_flags["stage_temp_guard"] == (
        plan.suggested_stage_temp_guard_flags
    )
    assert plan.suggested_launch_profile is not None
    assert plan.suggested_launch_profile["source"] == "prefill_plan"
    assert plan.suggested_launch_profile["argv_safe_to_replay"] is True
    assert plan.suggested_launch_profile["sections"]["prefill_guard_flags"] == (
        plan.suggested_prefill_guard_flags
    )
    assert plan.suggested_launch_profile["sections"][
        "public_glm_5_2_shape_guard_flags"
    ] == plan.suggested_public_glm_5_2_shape_guard_flags
    assert plan.suggested_launch_profile["argv"] == combined_argv + (
        "--require-public-glm-5-2-shape",
    )
    assert (
        plan.total_weight_bytes
        == plan.resident_gemm_weight_bytes + plan.routed_expert_chunked_read_bytes
    )
    assert plan.routed_expert_capacity_plan is not None
    assert plan.routed_expert_capacity_plan.capacity_tokens == 2240
    assert plan.routed_expert_capacity_plan.chunks_per_prompt == 2
    assert plan.routed_expert_capacity_plan.balanced_capacity_per_expert == 70
    assert (
        plan.routed_expert_capacity_plan.balanced_capacity_activation_bytes_per_moe_layer
        == 1_174_405_120
    )
    assert plan.routed_expert_capacity_plan.spill_free_capacity_per_expert == 2240
    assert (
        plan.routed_expert_capacity_plan.spill_free_capacity_activation_bytes_per_moe_layer
        == 37_580_963_840
    )


def test_prefill_plan_cli_json(capsys: pytest.CaptureFixture[str]) -> None:
    status = cli_main(
        [
            "prefill-plan",
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--max-prefill-activation-mib",
            "8",
            "--max-runner-scratch-mib",
            "4096",
            "--ssd-read-gib-s",
            "16",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_type"] == "glm_moe_dsa"
    assert payload["public_glm_5_2_shape"]["matches"] is False
    assert "hidden_size" in payload["public_glm_5_2_shape"]["mismatched_fields"]
    assert payload["suggested_public_glm_5_2_shape_guard_flags"] is None
    assert payload["mpp_candidate_ops"] == 11
    assert payload["chunk_plan"]["recommended_chunk_tokens"] == 128
    assert payload["cache_io_plan"]["mla_cache_width"] == 32
    assert payload["cache_io_plan"]["dsa_index_head_dim"] == 16
    assert payload["cache_io_plan"]["dsa_index_topk"] == 4
    assert payload["cache_io_plan"]["indexed_attention_layers"] == 6
    assert payload["cache_io_plan"]["full_attention_layers"] == 0
    assert payload["cache_io_plan"]["dsa_full_indexer_layers"] == 3
    assert payload["cache_io_plan"]["causal_rows_per_layer"] == 8256
    assert payload["cache_io_plan"]["indexed_rows_per_layer"] == 506
    assert payload["cache_io_plan"]["mla_cache_read_bytes"] == 194_304
    assert payload["cache_io_plan"]["dsa_index_cache_read_bytes"] == 792_576
    assert payload["cache_io_plan"]["total_cache_read_bytes"] == 986_880
    assert payload["cache_io_plan"]["mla_cache_write_bytes"] == 49_152
    assert payload["cache_io_plan"]["dsa_index_cache_write_bytes"] == 12_288
    assert payload["cache_io_plan"]["total_cache_write_bytes"] == 61_440
    assert payload["routed_expert_capacity_plan"]["balanced_capacity_per_expert"] == 64
    assert payload["routed_expert_capacity_plan"]["capacity_tokens"] == 128
    assert payload["routed_expert_capacity_plan"]["chunks_per_prompt"] == 1
    assert payload["routed_expert_capacity_plan"]["assignments_per_capacity_chunk"] == 256
    assert payload["routed_expert_capacity_plan"]["unique_experts_per_capacity_chunk"] == 4
    assert payload["routed_expert_capacity_plan"]["spill_free_capacity_per_expert"] == 128
    assert payload["routed_expert_read_cost_plan"]["read_amplification"] == 1.0
    assert payload["routed_expert_read_cost_plan"]["ssd_read_bytes_per_second"] == 16 * 1024**3
    assert payload["routed_expert_read_cost_plan"]["planned_read_seconds"] == pytest.approx(
        payload["routed_expert_read_cost_plan"]["planned_read_bytes"] / (16 * 1024**3)
    )
    assert payload["suggested_guard_flags"]["source"] == "prefill_plan"
    assert payload["suggested_guard_flags"]["prefill_prompt_chunk_tokens"] == 128
    assert payload["suggested_guard_flags"]["argv"][:4] == [
        "--prefill-prompt-chunk-tokens",
        "128",
        "--prefill-max-routed-read-amplification",
        "1.05",
    ]
    assert "--prefill-max-routed-read-seconds" in payload["suggested_guard_flags"]["argv"]
    assert payload["suggested_stage_temp_guard_flags"]["source"] == "prefill_plan"
    assert payload["suggested_stage_temp_guard_flags"]["prefill_prompt_chunk_tokens"] == 128
    assert payload["routed_stage_temp_plan"]["static_capacity_per_expert"] == "auto"
    assert payload["routed_stage_temp_plan"]["max_static_capacity_per_expert"] == 128
    assert payload["routed_stage_temp_plan"]["max_stage_raw_ranges"] == 4
    assert payload["routed_stage_temp_plan"]["max_stage_coalesced_ranges"] == 4
    assert payload["routed_stage_temp_plan"]["max_static_capacity_binary_bytes"] == 6200
    assert (
        payload["suggested_stage_temp_guard_flags"][
            "profile_total_static_capacity_binary_bytes"
        ]
        == 31_000
    )
    assert (
        payload["suggested_stage_temp_guard_flags"]["profile_max_stage_raw_ranges"]
        == 4
    )
    assert (
        payload["suggested_stage_temp_guard_flags"]["prefill_max_stage_raw_ranges"]
        == 5
    )
    assert (
        payload["suggested_stage_temp_guard_flags"][
            "profile_max_stage_coalesced_ranges"
        ]
        == 4
    )
    assert (
        payload["suggested_stage_temp_guard_flags"][
            "prefill_max_stage_coalesced_ranges"
        ]
        == 5
    )
    assert payload["suggested_stage_temp_guard_flags"]["argv"] == [
        "--prefill-prompt-chunk-tokens",
        "128",
        "--prefill-max-stage-mib",
        "0.237891",
        "--prefill-max-compact-stage-mib",
        "0.221484",
        "--prefill-static-capacity-per-expert",
        "auto",
        "--prefill-max-stage-raw-ranges",
        "5",
        "--prefill-max-stage-coalesced-ranges",
        "5",
    ]
    assert payload["suggested_prefill_guard_flags"]["source"] == "prefill_plan"
    assert payload["suggested_prefill_guard_flags"]["prefill_prompt_chunk_tokens"] == 128
    assert payload["suggested_prefill_guard_flags"]["argv"].count(
        "--prefill-prompt-chunk-tokens"
    ) == 1
    assert "--prefill-max-routed-read-gib" in payload["suggested_prefill_guard_flags"]["argv"]
    assert "--prefill-max-stage-mib" in payload["suggested_prefill_guard_flags"]["argv"]
    assert "--prefill-max-compact-stage-mib" in payload["suggested_prefill_guard_flags"]["argv"]
    assert payload["suggested_launch_profile"]["source"] == "prefill_plan"
    assert payload["suggested_launch_profile"]["argv_safe_to_replay"] is True
    assert payload["suggested_launch_profile"]["sections"]["prefill_guard_flags"] == (
        payload["suggested_prefill_guard_flags"]
    )
    assert payload["suggested_launch_profile"]["argv"] == payload[
        "suggested_prefill_guard_flags"
    ]["argv"]
    assert "--matrix-shapes" not in payload["suggested_launch_profile"]["argv"]
    assert payload["prefill_linear_calibration_shapes"][0]["rank"] == 1
    assert payload["prefill_linear_calibration_shapes"][0]["candidate_rank"] == 1
    assert payload["prefill_linear_calibration_shapes"][0]["op_name"] == (
        "moe.shared_down_proj"
    )
    assert payload["prefill_linear_calibration_shapes"][0]["batch_tokens"] == 128
    assert payload["prefill_linear_calibration_shapes"][0]["matrix_shape_arg"] == (
        "256x128"
    )
    assert payload["prefill_linear_calibration_shapes"][0]["candidate_count"] == 2
    assert payload["prefill_linear_calibration_shapes"][0]["candidate_ranks"] == [1, 5]
    assert payload["prefill_linear_calibration_shapes"][0]["candidate_op_names"] == [
        "moe.shared_down_proj",
        "dense_mlp.down_proj",
    ]
    assert payload["prefill_linear_calibration_candidate_coverage"]["source"] == (
        "prefill_plan"
    )
    assert payload["prefill_linear_calibration_candidate_coverage"][
        "covered_candidate_count"
    ] == 11
    assert payload["prefill_linear_calibration_candidate_coverage"][
        "unique_shape_count"
    ] == 6
    assert payload["prefill_linear_calibration_candidate_coverage"][
        "coverage_truncated"
    ] is False
    assert payload["suggested_prefill_linear_calibration_flags"]["source"] == (
        "prefill_plan"
    )
    assert payload["suggested_prefill_linear_calibration_flags"]["command"] == (
        "prefill-linear-calibrate"
    )
    assert payload["suggested_prefill_linear_calibration_flags"]["argv"] == [
        "--batch-tokens",
        "128",
        "--matrix-shapes",
        "256x128,128x256,64x128,128x32,32x128,32x64",
        "--max-calibration-case-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "3",
    ]
    assert payload["routed_expert_capacity_plan"]["balanced_capacity_utilization"] == 1.0
    assert payload["routed_expert_capacity_plan"]["spill_free_capacity_utilization"] == 0.5
    assert payload["routed_expert_capacity_plan"]["balanced_capacity_activation_bytes_per_moe_layer"] == 786_432
    assert (
        payload["routed_expert_capacity_plan"]["spill_free_capacity_activation_bytes_per_moe_layer"]
        == 1_572_864
    )
    assert payload["staged_moe_runner_scratch_plan"]["auto_token_block"] == 128
    assert payload["staged_moe_runner_scratch_plan"]["fits_runner_scratch"] is True
    assert (
        payload["staged_moe_runner_scratch_plan"]["estimated_peak_bytes_per_moe_layer"]
        == 2_691_712
    )
    assert payload["ops"][0]["name"] == "attention.q_a_proj"
    assert payload["ops"][0]["tile_plan"]["threadgroup_tile_m"] == 64
    assert payload["prefill_backend_candidates"][0]["rank"] == 1
    assert payload["prefill_backend_candidates"][0]["op_name"] == "moe.shared_down_proj"
    assert payload["prefill_backend_candidates"][0]["availability"] == "not_inspected"


def test_prefill_plan_cli_require_public_glm_5_2_shape_rejects_non_public(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli_main(
        [
            "prefill-plan",
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--require-public-glm-5-2-shape",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "config does not match the public GLM-5.2 shape" in captured.err
    assert "hidden_size" in captured.err
    assert captured.out == ""


def test_prefill_plan_cli_require_public_glm_5_2_shape_rejects_non_4bit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = cli_main(
        [
            "prefill-plan",
            str(FIXTURES / "glm_5_2_config.json"),
            "--prompt-tokens",
            "128",
            "--expert-bits",
            "8",
            "--require-public-glm-5-2-shape",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "public GLM-5.2 prefill planning requires expert_bits=4" in captured.err
    assert captured.out == ""


def test_prefill_plan_cli_writes_launch_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "prefill-launch-profile.json"
    calibration_flags_path = tmp_path / "prefill-calibration-flags.json"

    status = cli_main(
        [
            "prefill-plan",
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--max-prefill-activation-mib",
            "8",
            "--ssd-read-gib-s",
            "16",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--write-launch-profile",
            str(profile_path),
            "--write-calibration-flags",
            str(calibration_flags_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    calibration_flags = json.loads(calibration_flags_path.read_text(encoding="utf-8"))
    assert profile == payload["suggested_launch_profile"]
    assert calibration_flags == payload["suggested_prefill_linear_calibration_flags"]
    assert profile["source"] == "prefill_plan"
    assert profile["argv_safe_to_replay"] is True
    assert "--prefill-prompt-chunk-tokens" in profile["argv"]
    assert "--prefill-max-routed-read-gib" in profile["argv"]
    assert "--prefill-mpsgraph-min-batch-tokens" in profile["argv"]
    assert "--prefill-min-accelerated-flop-fraction" in profile["argv"]
    assert "--matrix-shapes" not in profile["argv"]
    assert calibration_flags["command"] == "prefill-linear-calibrate"
    assert "--matrix-shapes" in calibration_flags["argv"]
    assert profile["sections"]["prefill_runtime_policy_flags"] == payload[
        "suggested_prefill_runtime_policy_flags"
    ]


def test_prefill_plan_calibrate_cli_uses_planner_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_shapes = (
        (256, 128),
        (128, 256),
        (64, 128),
        (128, 32),
        (32, 128),
        (32, 64),
    )
    runtime_policy = {
        "source": "prefill_linear_calibration",
        "prefill_linear_backend": "mps-matrix-f32",
        "prefill_mpsgraph_min_batch_tokens": 128,
        "prefill_mpsgraph_min_matrix_dim": 32,
        "argv": (
            "--prefill-linear-backend",
            "mps-matrix-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
        ),
    }
    launch_profile = {
        "source": "prefill_linear_calibration",
        "argv_safe_to_replay": True,
        "sections": {"prefill_runtime_policy_flags": runtime_policy},
        "argv": runtime_policy["argv"],
    }

    def fake_calibration(**kwargs):
        assert kwargs["batch_token_values"] == (128,)
        assert kwargs["matrix_dim_values"] == (32, 64, 128)
        assert kwargs["matrix_shapes"] == expected_shapes
        assert kwargs["repeats"] == 2
        assert kwargs["min_mpsgraph_speedup"] == 1.25
        assert kwargs["max_calibration_case_mib"] == 1
        assert kwargs["max_resident_matrix_mib"] == 1
        assert kwargs["max_runner_scratch_mib"] == 3
        assert kwargs["max_calibration_work_dir_mib"] == 8192
        assert kwargs["calibration_work_dir_free_margin_mib"] == 512
        assert kwargs["matrix_dtype"] == "BF16"
        assert kwargs["work_dir"] == str(tmp_path / "work")
        assert kwargs["keep_work_dir"] is True
        return ResidentLinearCalibrationResult(
            runner_path=Path(kwargs["runner_path"]),
            work_dir=tmp_path / "work",
            kept_work_dir=True,
            batch_token_values=(128,),
            matrix_dim_values=(32, 64, 128),
            repeats=2,
            min_mpsgraph_speedup=1.25,
            max_calibration_case_bytes=1024 * 1024,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=3,
            recommended_prefill_mpsgraph_min_batch_tokens=128,
            recommended_prefill_mpsgraph_min_matrix_dim=32,
            suggested_prefill_runtime_policy_flags=runtime_policy,
            suggested_launch_profile=launch_profile,
            cases=(),
            matrix_shapes=expected_shapes,
            matrix_dtype="BF16",
        )

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fake_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    flags_path = tmp_path / "calibration-flags.json"
    profile_path = tmp_path / "profile.json"

    status = cli_main(
        [
            "prefill-plan-calibrate",
            str(runner),
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--max-prefill-activation-mib",
            "8",
            "--repeats",
            "2",
            "--min-mpsgraph-speedup",
            "1.25",
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--require-prefill-acceleration",
            "--ssd-read-gib-s",
            "16",
            "--matrix-dtype",
            "BF16",
            "--work-dir",
            str(tmp_path / "work"),
            "--keep-work-dir",
            "--write-calibration-flags",
            str(flags_path),
            "--write-launch-profile",
            str(profile_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prefill_plan"]["model_type"] == "glm_moe_dsa"
    matrix_bytes = sum(in_dim * out_dim * 2 for in_dim, out_dim in expected_shapes)
    input_bytes = sum(128 * in_dim * 4 for in_dim, _out_dim in expected_shapes)
    output_bytes = sum(128 * out_dim * 4 for _in_dim, out_dim in expected_shapes)
    assert payload["calibration_work_dir_budget"] == {
        "source": "prefill_plan_calibration",
        "shape_count": 6,
        "repeats": 2,
        "matrix_dtype": "BF16",
        "backend_count": 3,
        "calibrated_backends": [
            "custom-metal",
            "mpsgraph-f32",
            "mps-matrix-f32",
        ],
        "backend_output_file_count": 36,
        "matrix_bytes": matrix_bytes,
        "input_bytes": input_bytes,
        "single_backend_output_bytes": output_bytes,
        "total_backend_output_bytes": 6 * output_bytes,
        "single_case_bytes": matrix_bytes + input_bytes + output_bytes,
        "estimated_work_dir_bytes": matrix_bytes + input_bytes + 6 * output_bytes,
        "max_calibration_work_dir_bytes": 8192 * 1024 * 1024,
        "max_calibration_work_dir_mib": 8192,
    }
    assert payload["applied_calibration_flags"]["argv"] == [
        "--batch-tokens",
        "128",
        "--matrix-shapes",
        "256x128,128x256,64x128,128x32,32x128,32x64",
        "--matrix-dtype",
        "BF16",
        "--max-calibration-case-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "3",
    ]
    assert payload["applied_calibration_flags"]["max_calibration_case_bytes"] == (
        1 * 1024 * 1024
    )
    assert payload["applied_calibration_flags"]["max_resident_matrix_bytes"] == (
        1 * 1024 * 1024
    )
    assert payload["applied_calibration_flags"]["max_runner_scratch_bytes"] == (
        3 * 1024 * 1024
    )
    assert payload["calibration"]["matrix_shapes"] == [
        [256, 128],
        [128, 256],
        [64, 128],
        [128, 32],
        [32, 128],
        [32, 64],
    ]
    assert payload["calibration"]["matrix_dtype"] == "BF16"
    coverage = payload["calibration_candidate_coverage"]
    assert coverage == payload["prefill_plan"][
        "prefill_linear_calibration_candidate_coverage"
    ]
    assert coverage["source"] == "prefill_plan"
    assert coverage["candidate_count"] == len(
        payload["prefill_plan"]["prefill_backend_candidates"]
    )
    assert coverage["covered_candidate_count"] == 11
    assert coverage["covered_candidate_ranks"] == list(range(1, 12))
    assert coverage["uncovered_candidate_ranks"] == []
    assert coverage["unique_shape_count"] == 6
    assert coverage["coverage_truncated"] is False
    assert coverage["covered_candidate_flop_fraction"] == 1.0
    assert coverage["shapes"][0]["matrix_shape_arg"] == "256x128"
    assert coverage["shapes"][0]["candidate_ranks"] == [1, 5]
    assert coverage["shapes"][0]["candidate_op_names"] == [
        "moe.shared_down_proj",
        "dense_mlp.down_proj",
    ]
    assert coverage["shapes"][1]["candidate_count"] == 4
    assert coverage["shapes"][1]["candidate_ranks"] == [2, 3, 6, 7]
    assert json.loads(flags_path.read_text(encoding="utf-8")) == payload[
        "applied_calibration_flags"
    ]
    written_profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert written_profile == payload["combined_launch_profile"]
    assert written_profile["source"] == "prefill_plan_calibration"
    assert written_profile["argv_safe_to_replay"] is True
    assert "--matrix-shapes" not in written_profile["argv"]
    assert "--prefill-prompt-chunk-tokens" in written_profile["argv"]
    assert "--prefill-ssd-read-gib-s" in written_profile["argv"]
    assert "--prefill-max-routed-read-seconds" in written_profile["argv"]
    assert "--prefill-mpsgraph-min-batch-tokens" in written_profile["argv"]
    assert "--prefill-min-accelerated-flop-fraction" in written_profile["argv"]
    assert "--require-prefill-acceleration" in written_profile["argv"]
    assert written_profile["sections"]["prefill_runtime_policy_flags"] == {
        "argv": [
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--require-prefill-acceleration",
        ],
        "calibration_source": "prefill_linear_calibration",
        "plan_source": "prefill_plan",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_min_accelerated_flop_fraction": 0.5,
        "prefill_mpsgraph_min_batch_tokens": 128,
        "prefill_mpsgraph_min_matrix_dim": 32,
        "require_prefill_acceleration": True,
        "source": "prefill_plan_calibration",
    }
    guard = written_profile["sections"]["prefill_guard_flags"]
    assert "--prefill-max-routed-read-seconds" in guard["argv"]
    assert guard["routed_read_guard"]["prefill_ssd_read_gib_per_second"] == 16.0
    assert "--prefill-max-routed-read-seconds" in guard["routed_read_guard"]["argv"]


def test_prefill_plan_calibrate_cli_bf16_raises_runner_scratch_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    max_matrix_values = 16384 * 6144
    expected_runner_scratch_mib = math.ceil(
        (
            max_matrix_values * 6
            + 128 * 16384 * 4
            + 128 * 6144 * 4
        )
        * 1.10
        / (1024 * 1024)
    )

    def fake_calibration(**kwargs):
        assert kwargs["matrix_dtype"] == "BF16"
        assert kwargs["max_runner_scratch_mib"] == expected_runner_scratch_mib
        return ResidentLinearCalibrationResult(
            runner_path=Path(kwargs["runner_path"]),
            work_dir=tmp_path / "work",
            kept_work_dir=False,
            batch_token_values=(128,),
            matrix_dim_values=(512, 576, 2048, 6144),
            repeats=1,
            min_mpsgraph_speedup=1.0,
            max_calibration_case_bytes=435 * 1024 * 1024,
            max_resident_matrix_mib=423,
            max_runner_scratch_mib=expected_runner_scratch_mib,
            recommended_prefill_mpsgraph_min_batch_tokens=None,
            recommended_prefill_mpsgraph_min_matrix_dim=None,
            suggested_prefill_runtime_policy_flags=None,
            suggested_launch_profile=None,
            cases=(),
            matrix_dtype="BF16",
        )

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fake_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    status = cli_main(
        [
            "prefill-plan-calibrate",
            str(runner),
            str(FIXTURES / "glm_5_2_config.json"),
            "--prompt-tokens",
            "128",
            "--matrix-dtype",
            "BF16",
            "--require-public-glm-5-2-shape",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied_calibration_flags"]["max_runner_scratch_mib"] == (
        expected_runner_scratch_mib
    )
    assert payload["applied_calibration_flags"]["max_runner_scratch_bytes"] == (
        expected_runner_scratch_mib * 1024 * 1024
    )


def test_prefill_plan_calibrate_cli_require_public_glm_5_2_shape_rejects_non_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_calibration(**kwargs):
        pytest.fail("calibration should be blocked before runner dispatch")

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fail_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    status = cli_main(
        [
            "prefill-plan-calibrate",
            str(runner),
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--require-public-glm-5-2-shape",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "config does not match the public GLM-5.2 shape" in captured.err
    assert "hidden_size" in captured.err
    assert captured.out == ""


def test_prefill_plan_calibrate_cli_require_public_glm_5_2_shape_rejects_non_4bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_calibration(**kwargs):
        pytest.fail("calibration should be blocked before runner dispatch")

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fail_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    status = cli_main(
        [
            "prefill-plan-calibrate",
            str(runner),
            str(FIXTURES / "glm_5_2_config.json"),
            "--prompt-tokens",
            "128",
            "--expert-bits",
            "8",
            "--require-public-glm-5-2-shape",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "public GLM-5.2 prefill planning requires expert_bits=4" in captured.err
    assert captured.out == ""


def test_prefill_plan_calibrate_cli_rejects_auto_cap_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_calibration(**kwargs):
        pytest.fail("calibration should be blocked before runner dispatch")

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fail_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    status = cli_main(
        [
            "prefill-plan-calibrate",
            str(runner),
            str(FIXTURES / "glm_5_2_config.json"),
            "--prompt-tokens",
            "4096",
            "--max-auto-calibration-case-mib",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "planned --max-calibration-case-mib 810 MiB exceeds" in captured.err
    assert captured.out == ""


def test_prefill_plan_calibrate_cli_rejects_work_dir_budget_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_calibration(**kwargs):
        pytest.fail("calibration should be blocked before runner dispatch")

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fail_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    status = cli_main(
        [
            "prefill-plan-calibrate",
            str(runner),
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--max-prefill-activation-mib",
            "8",
            "--max-calibration-work-dir-mib",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "planned calibration work dir" in captured.err
    assert "--max-calibration-work-dir-mib 1 MiB" in captured.err
    assert captured.out == ""
