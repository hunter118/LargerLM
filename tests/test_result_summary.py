from __future__ import annotations

import hashlib
import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest

from largerlm.cli import main as cli_main
from largerlm.disk_benchmark import SequentialReadBenchmark
from largerlm.generation_guard import SystemMemorySnapshot
from largerlm.prepared_lock import PREPARED_RUN_LOCK_ENV
from largerlm.result_summary import (
    compare_result_summaries,
    format_result_bakeoff_text,
    format_result_comparison_text,
    format_result_summary_text,
    result_bakeoff_files,
    summarize_result_file,
    summarize_result_payload,
)


def _test_memory_snapshot(
    *,
    available_bytes: int = 1024**3,
    total_bytes: int = 2 * 1024**3,
) -> SystemMemorySnapshot:
    return SystemMemorySnapshot(
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        page_size=4096,
        source="test",
    )


def _test_disk_usage(
    *,
    free_bytes: int = 1024**3,
    total_bytes: int = 2 * 1024**3,
):
    def usage(path_arg: object) -> dict[str, object]:
        return {
            "path": str(path_arg),
            "probe_path": str(path_arg),
            "total_bytes": total_bytes,
            "used_bytes": total_bytes - free_bytes,
            "free_bytes": free_bytes,
        }

    return usage


def _test_sequential_read_benchmark(gib_per_second: float):
    calls: list[dict[str, object]] = []

    def benchmark(
        path: str | Path,
        *,
        bytes_to_read: int | None = None,
        chunk_bytes: int = 8 * 1024**2,
        offset_bytes: int = 0,
        max_chunk_bytes: int | None = 512 * 1024**2,
    ) -> SequentialReadBenchmark:
        calls.append(
            {
                "path": Path(path),
                "bytes_to_read": bytes_to_read,
                "chunk_bytes": chunk_bytes,
                "offset_bytes": offset_bytes,
                "max_chunk_bytes": max_chunk_bytes,
            }
        )
        measured = int(bytes_to_read or 1024)
        return SequentialReadBenchmark(
            path=Path(path),
            file_size_bytes=max(measured, 1),
            offset_bytes=offset_bytes,
            requested_bytes=measured,
            measured_bytes=measured,
            chunk_bytes=chunk_bytes,
            elapsed_seconds=1.0,
            gib_per_second=gib_per_second,
            short_read=False,
        )

    benchmark.calls = calls  # type: ignore[attr-defined]
    return benchmark


def _replay_prefill_acceleration_coverage() -> dict[str, object]:
    return {
        "ok": True,
        "required": True,
        "analyzed": True,
        "min_accelerated_flop_fraction": 0.0,
        "matrix_count": 1,
        "accelerated_matrix_count": 1,
        "mpsgraph_matrix_count": 1,
        "custom_metal_matrix_count": 0,
        "unsupported_mpsgraph_matrix_count": 0,
        "total_estimated_flops": 1024,
        "accelerated_estimated_flops": 1024,
        "custom_metal_estimated_flops": 0,
        "unsupported_mpsgraph_estimated_flops": 0,
        "other_estimated_flops": 0,
        "router_gate_matrix_count": 0,
        "router_gate_estimated_flops": 0,
        "router_gate_accelerated_matrix_count": 0,
        "router_gate_accelerated_estimated_flops": 0,
        "non_router_matrix_count": 1,
        "non_router_estimated_flops": 1024,
        "non_router_accelerated_matrix_count": 1,
        "non_router_accelerated_estimated_flops": 1024,
        "non_router_unaccelerated_matrix_count": 0,
        "non_router_unaccelerated_estimated_flops": 0,
        "non_router_unaccelerated_flop_fraction": 0.0,
        "non_router_unaccelerated_streamed_routed_expert_matrix_count": 0,
        "non_router_unaccelerated_streamed_routed_expert_estimated_flops": 0,
        "non_router_unaccelerated_non_streamed_matrix_count": 0,
        "non_router_unaccelerated_non_streamed_estimated_flops": 0,
        "unaccelerated_backend_matrix_counts": {},
        "unaccelerated_backend_estimated_flops": {},
        "accelerated_router_gate_flop_share": 0.0,
        "accelerated_router_gate_only": False,
        "mpp_candidate_policy": {
            "candidate_backend": "mpp_tensor_ops_prefill",
            "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
            "mpp_tensor_ops_min_batch_tokens": 128,
            "mpp_tensor_ops_min_matrix_dim": 32,
            "selectable_prefill_backend": False,
        },
        "mpp_tensor_ops_candidate_matrix_count": 0,
        "mpp_tensor_ops_candidate_estimated_flops": 0,
        "mpp_tensor_ops_candidate_flop_fraction": 0.0,
        "streamed_routed_expert_layer_count": 0,
        "streamed_routed_expert_matrix_count": 0,
        "streamed_routed_expert_assignments": 0,
        "streamed_routed_expert_estimated_flops": 0,
        "streamed_routed_expert_mpp_candidate_matrix_count": 0,
        "streamed_routed_expert_mpp_candidate_estimated_flops": 0,
        "accelerated_flop_fraction": 1.0,
        "dominant_resident_flops_accelerated": True,
        "accelerated_backends": ["mpsgraph-f32"],
        "any_resident_matrix_accelerated": True,
        "all_resident_matrices_accelerated": True,
        "reason": "",
    }


def _write_replay_binding_files(
    root: Path,
    *,
    prefill_moe_output_accumulator: str | None = None,
) -> Path:
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    benchmark_path = prepared / "benchmark.bin"
    benchmark_path.write_bytes(b"0" * 4096)
    for relative in (
        "experts/layout.json",
        "resident/layout.json",
        "decode_cache_layout.json",
        "decode_cache.bin",
    ):
        target = prepared / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_dir": str(prepared),
                "experts_layout": "experts/layout.json",
                "resident_layout": "resident/layout.json",
                "decode_cache_layout": "decode_cache_layout.json",
                "decode_cache_file": "decode_cache.bin",
                "max_context_tokens": 1,
                "prepare_cold_read_gib_per_second": 4.0,
                "prepare_cold_read_source": "test",
                "prepare_cold_read_benchmark_path": str(benchmark_path),
                "prepare_cold_read_benchmark_requested_bytes": 1024,
                "prepare_cold_read_benchmark_measured_bytes": 1024,
                "prepare_cold_read_benchmark_elapsed_seconds": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile_path = prepared / "launch-profile.json"
    profile_argv = [
        "--run-mpsgraph-probe",
        "--prefill-mpsgraph-min-batch-tokens",
        "16",
        "--prefill-mpsgraph-min-matrix-dim",
        "32",
        "--require-prefill-acceleration",
    ]
    profile_sections: dict[str, object] = {}
    if prefill_moe_output_accumulator is not None:
        profile_argv.extend(
            ["--prefill-moe-output-accumulator", prefill_moe_output_accumulator]
        )
        profile_sections["prefill_moe_output_accumulator_flags"] = {
            "source": "test",
            "prefill_moe_output_accumulator": prefill_moe_output_accumulator,
            "argv": (
                "--prefill-moe-output-accumulator",
                prefill_moe_output_accumulator,
            ),
        }
    profile_path.write_text(
        json.dumps(
            {
                "argv": profile_argv,
                "sections": profile_sections,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    coverage = _replay_prefill_acceleration_coverage()
    (prepared / "launch-audit.json").write_text(
        json.dumps(
            {
                "schema": "largerlm.launch_audit.v1",
                "applied_launch_profile": {
                    "path": str(profile_path),
                    "sha256": profile_sha256,
                },
                "launch_audit": {
                    "ok": True,
                    "checks": [
                        {"code": "prefill_acceleration_required", "ok": True},
                        {
                            "code": "prefill_acceleration_gate_ok",
                            "ok": True,
                            "prefill_acceleration_runtimes": ["mpsgraph-f32"],
                            "selectable_accelerated_prefill_backends": [
                                "mpsgraph-f32"
                            ],
                            "validated_prefill_acceleration_available": True,
                        },
                        {
                            "code": "prefill_acceleration_probe_ok",
                            "ok": True,
                            "mps_graph_probe_requested": True,
                            "mps_graph_probe_ran": True,
                            "mps_graph_probe_ok": True,
                            "validated_accelerated_prefill_backends": [
                                "mpsgraph-f32"
                            ],
                        },
                        {
                            "code": "prefill_acceleration_profile_replays_probe",
                            "ok": True,
                            "required": True,
                            "runtime_probe_required": True,
                            "runtime_probe_satisfied": True,
                            "has_run_mpsgraph_probe": True,
                        },
                        {
                            "code": "request_prefill_acceleration_coverage_ok",
                            "evidence_present": True,
                            "ok": True,
                            **coverage,
                        },
                    ],
                },
                "request_check": {
                    "ok": True,
                    "prefill_acceleration_coverage": coverage,
                    "runtime_preflight": {
                        "ran": True,
                        "available_memory_ok": True,
                        "required_available_memory_bytes": 4096,
                        "system_available_memory_bytes": 8192,
                        "system_memory_source": "test",
                    },
                    "prefill_stage_temp_disk_free": {
                        "analyzed": True,
                        "within_free_space": True,
                        "path": str(prepared),
                        "required_free_bytes": 8192,
                        "required_stage_temp_bytes": 8192,
                        "free_bytes": 16384,
                        "total_bytes": 32768,
                        "used_bytes": 16384,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return prepared


def _sample_result(
    *,
    include_prompt_elapsed: bool = False,
    prepared_root: str | Path = "/tmp/prepared",
) -> dict[str, object]:
    prepared_path = Path(prepared_root)
    manifest_path = prepared_path / "manifest.json"
    profile_path = prepared_path / "launch-profile.json"
    audit_path = prepared_path / "launch-audit.json"
    profile_sha256 = (
        "a" * 64
        if str(prepared_path) == "/tmp/prepared"
        else hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if profile_path.exists()
        else "a" * 64
    )
    prompt_prefill: dict[str, object] = {
        "chunk_count": 1,
        "chunk_tokens": 2,
        "total_expert_stage_planned_read_bytes": 3 * 1024**3,
        "total_expert_stage_copy_elapsed_seconds": 1.5,
        "total_expert_stage_copy_throughput_gib_per_second": 2.0,
        "linear_backend_counts": {"custom-metal": 2, "mpsgraph-f32": 1},
        "linear_backend_elapsed_seconds": {
            "custom-metal": 2.0,
            "mpsgraph-f32": 0.5,
        },
        "linear_backend_flops": {"custom-metal": 200, "mpsgraph-f32": 50},
        "linear_backend_estimated_tflops": {
            "custom-metal": 0.1,
            "mpsgraph-f32": 0.2,
        },
        "accelerated_linear_flop_fraction": 0.2,
        "chunks": [
            {
                "layers": [
                    {
                        "attention": {
                            "mla_key_cache": True,
                            "mla_key_cache_bytes": 16,
                            "mla_value_cache": True,
                            "mla_value_cache_bytes": 32,
                            "projections_elapsed_seconds": 0.08,
                            "mla_attention_elapsed_seconds": 1.25,
                            "attention_output": {
                                "o_proj": {
                                    "elapsed_seconds": 0.7,
                                    "backend": "custom-metal",
                                    "layer": 0,
                                    "tensor": "model.layers.0.self_attn.o_proj.weight",
                                }
                            }
                        },
                        "staged_mlp": {
                            "batch_tokens": 2,
                            "compact_stage_bytes": 128,
                            "compact_stage_materialized_bytes": 0,
                            "compact_stage_storage": "hardlink",
                            "copy_chunk_bytes": 8 * 1024 * 1024,
                            "effective_moe_token_block": 2,
                            "layer": 0,
                            "max_k": 2,
                            "moe_batch_buffer_bytes": 64,
                            "moe_estimated_peak_bytes": 512,
                            "moe_max_expert_tokens": 2,
                            "moe_output_accumulator": "memory",
                            "moe_output_accumulator_bytes": 32,
                            "routed_moe_elapsed_seconds": 0.25,
                            "stage_copy_elapsed_seconds": 0.125,
                            "stage_copy_throughput_gib_per_second": 1.0,
                            "stage_plus_compact_bytes": 384,
                            "stage_plus_compact_materialized_bytes": 256,
                            "static_capacity_per_expert": 2,
                            "static_capacity_total_slots": 6,
                            "static_capacity_used_slots": 4,
                            "top_k": 2,
                            "router_margin_summary": {
                                "token_count": 2,
                                "min_effective_score_margin": 0.0002,
                                "mean_effective_score_margin": 0.00035,
                                "effective_near_tie_counts": {
                                    "le_1e-06": 0,
                                    "le_1e-05": 0,
                                    "le_1e-04": 0,
                                    "le_1e-03": 2,
                                },
                                "min_topk_score_margin": 0.0003,
                                "mean_topk_score_margin": 0.0004,
                                "topk_near_tie_counts": {
                                    "le_1e-06": 0,
                                    "le_1e-05": 0,
                                    "le_1e-04": 0,
                                    "le_1e-03": 2,
                                },
                                "min_group_score_margin": 0.0002,
                                "mean_group_score_margin": 0.0003,
                                "group_near_tie_counts": {
                                    "le_1e-06": 0,
                                    "le_1e-05": 0,
                                    "le_1e-04": 0,
                                    "le_1e-03": 2,
                                },
                                "weakest_token": {
                                    "token_index": 0,
                                    "effective_score_margin": 0.0002,
                                    "topk_score_margin": 0.0003,
                                    "group_score_margin": 0.0002,
                                },
                                "router_gate_policy": {
                                    "mode": "custom-first-mpsgraph-fallback",
                                    "margin_threshold": 0.001,
                                    "decision": "mpsgraph-f32-fallback",
                                    "command_count": 2,
                                    "custom_elapsed_seconds": 0.05,
                                    "fallback_elapsed_seconds": 0.5,
                                    "custom_min_effective_score_margin": 0.0002,
                                    "fallback_min_effective_score_margin": 0.0003,
                                },
                            },
                            "stage_result": {
                                "batch_plan": {
                                    "batch_tokens": 2,
                                    "planned_read_bytes": 256,
                                    "raw_range_count": 2,
                                    "coalesced_range_count": 1,
                                    "selected_experts": [3, 5],
                                    "total_assignments": 4,
                                    "unique_requested_bytes": 256,
                                },
                                "io_summary": {
                                    "selected_expert_count": 2,
                                    "staged_bytes": 256,
                                    "waste_bytes": 0,
                                },
                                "planned_read_bytes": 256,
                                "copy_chunk_bytes": 8 * 1024 * 1024,
                                "selected_experts": [3, 5],
                                "staged_bytes": 256,
                            },
                            "staged_moe": {
                                "effective_moe_token_block": 2,
                                "moe_batch_buffer_bytes": 64,
                                "moe_estimated_peak_bytes": 512,
                                "moe_max_expert_tokens": 2,
                                "moe_timing_sort_seconds": 0.001,
                                "moe_timing_setup_seconds": 0.002,
                                "moe_timing_expert_read_seconds": 0.03,
                                "moe_timing_input_read_seconds": 0.01,
                                "moe_timing_output_read_seconds": 0.02,
                                "moe_timing_kernel_seconds": 0.16,
                                "moe_timing_mxfp4_swiglu_kernel_seconds": 0.11,
                                "moe_timing_mxfp4_down_add_kernel_seconds": 0.04,
                                "moe_timing_output_write_seconds": 0.015,
                                "moe_timing_final_read_seconds": 0.002,
                                "moe_timing_total_seconds": 0.24,
                                "selected_experts": [3, 5],
                                "stage_planned_read_bytes": 256,
                                "stage_staged_bytes": 256,
                                "stage_unique_requested_bytes": 256,
                                "stage_waste_bytes": 0,
                                "static_capacity_total_slots": 6,
                                "static_capacity_used_slots": 4,
                            },
                            "router_gate_proj": {
                                "elapsed_seconds": 0.5,
                                "backend": "mpsgraph-f32",
                                "layer": 0,
                                "tensor": "model.layers.0.mlp.gate.weight",
                            }
                        },
                    }
                ]
            }
        ],
    }
    if include_prompt_elapsed:
        prompt_prefill["elapsed_seconds"] = 8.0
    return {
        "prepared_manifest": str(manifest_path),
        "request": {
            "prompt_tokens": 2,
            "prompt_token_ids": [1, 2],
            "max_new_tokens": 1,
            "launch_audit_path": str(audit_path),
        },
        "token_result": {
            "applied_launch_profile": {
                "argv_safe_to_replay": True,
                "current_prepared_manifest": str(manifest_path),
                "lock_required": True,
                "locked": True,
                "matches_prepared": True,
                "path": str(profile_path),
                "sha256": profile_sha256,
            },
            "elapsed_seconds": 10.0,
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [7],
            "prefill_actual_read_time": {
                "source": "generation_actual_prefill",
                "total_expert_stage_serial_read_bytes": 6 * 1024**3,
                "total_expert_stage_unique_requested_bytes": 2 * 1024**3,
                "total_expert_stage_planned_read_bytes": 3 * 1024**3,
                "total_expert_stage_waste_bytes": 1 * 1024**3,
                "total_expert_stage_coalesced_savings_bytes": 3 * 1024**3,
                "total_expert_stage_raw_ranges": 4,
                "total_expert_stage_coalesced_ranges": 2,
                "total_expert_stage_read_advice_attempted_ranges": 2,
                "total_expert_stage_read_advice_calls": 2,
                "total_expert_stage_read_advice_bytes": 3 * 1024**3,
                "total_expert_stage_read_advice_failures": 0,
                "total_expert_stage_copy_read_calls": 3,
                "total_expert_stage_copy_write_calls": 3,
                "total_expert_stage_copy_average_read_bytes": float(1024**3),
                "total_expert_stage_copy_average_write_bytes": float(1024**3),
                "total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib": {
                    "8": 6,
                    "16": 4,
                    "32": 3,
                },
                "total_expert_stage_assignment_read_amplification": 0.5,
                "total_expert_stage_unique_read_amplification": 1.5,
                "max_expert_stage_unique_read_amplification": 2.0,
                "max_expert_stage_stage_budget_utilization": 0.75,
                "expert_stage_io_stage_count": 1,
                "expert_stage_copy_hotspots": [
                    {
                        "chunk_index": 0,
                        "layer": 0,
                        "tile_index": 0,
                        "selected_experts": [3, 5],
                        "raw_range_count": 2,
                        "coalesced_range_count": 1,
                        "planned_read_bytes": 256,
                        "copy_elapsed_seconds": 0.125,
                        "copy_read_calls": 3,
                    }
                ],
                "expert_stage_range_hotspots": [
                    {
                        "chunk_index": 0,
                        "layer": 0,
                        "tile_index": 0,
                        "selected_experts": [3, 5],
                        "raw_range_count": 2,
                        "coalesced_range_count": 1,
                        "planned_read_bytes": 256,
                        "copy_elapsed_seconds": 0.125,
                        "copy_read_calls": 3,
                    }
                ],
            },
            "steps": [
                {
                    "elapsed_seconds": 10.0,
                    "logits_elapsed_seconds": 0.25,
                    "selected_token_id": 7,
                }
            ],
            "prefill_actual_acceleration_coverage": {
                "source": "generation_actual_prefill",
                "ok": True,
                "accelerated_flop_fraction": 0.2,
                "matrix_count": 3,
                "total_estimated_flops": 250,
                "custom_metal_matrix_count": 2,
                "custom_metal_estimated_flops": 200,
                "unsupported_mpsgraph_matrix_count": 0,
                "unsupported_mpsgraph_estimated_flops": 0,
                "other_matrix_count": 0,
                "other_estimated_flops": 0,
                "accelerated_estimated_flops": 50,
                "streamed_routed_expert_matrix_count": 1,
                "streamed_routed_expert_estimated_flops": 128,
                "router_gate_matrix_count": 1,
                "router_gate_estimated_flops": 50,
                "router_gate_accelerated_matrix_count": 1,
                "router_gate_accelerated_estimated_flops": 50,
                "non_router_matrix_count": 2,
                "non_router_estimated_flops": 200,
                "non_router_accelerated_matrix_count": 0,
                "non_router_accelerated_estimated_flops": 0,
                "non_router_unaccelerated_matrix_count": 2,
                "non_router_unaccelerated_estimated_flops": 200,
                "non_router_unaccelerated_flop_fraction": 0.8,
                "non_router_unaccelerated_streamed_routed_expert_matrix_count": 1,
                "non_router_unaccelerated_streamed_routed_expert_estimated_flops": 128,
                "non_router_unaccelerated_non_streamed_matrix_count": 1,
                "non_router_unaccelerated_non_streamed_estimated_flops": 72,
                "unaccelerated_backend_matrix_counts": {"custom-metal": 2},
                "unaccelerated_backend_estimated_flops": {"custom-metal": 200},
                "accelerated_router_gate_flop_share": 1.0,
                "accelerated_router_gate_only": True,
            },
            "prefill_actual_acceleration_frontier": {
                "source": "generation_actual_prefill",
                "suggested_guard_flags": {
                    "argv": ["--prefill-prompt-chunk-tokens", "2"],
                },
            },
            "prefill_actual_linear_backend": {
                "source": "generation_actual_prefill",
                "configured_backend": "auto",
                "linear_backend_counts": {
                    "custom-metal": 2,
                    "mpsgraph-f32": 1,
                },
                "linear_backend_elapsed_seconds": {
                    "custom-metal": 2.0,
                    "mpsgraph-f32": 0.5,
                },
                "linear_backend_flops": {
                    "custom-metal": 200,
                    "mpsgraph-f32": 50,
                },
                "linear_backend_estimated_tflops": {
                    "custom-metal": 0.1,
                    "mpsgraph-f32": 0.2,
                },
                "linear_backend_component_stats": {
                    "attention.o_proj": {
                        "linear_backend_counts": {"custom-metal": 1},
                        "linear_backend_elapsed_seconds": {"custom-metal": 0.7},
                        "linear_backend_flops": {"custom-metal": 160},
                        "linear_backend_estimated_tflops": {
                            "custom-metal": 160 / 0.7 / 1e12,
                        },
                    },
                    "moe.router_gate_proj": {
                        "linear_backend_counts": {"mpsgraph-f32": 1},
                        "linear_backend_elapsed_seconds": {"mpsgraph-f32": 0.5},
                        "linear_backend_flops": {"mpsgraph-f32": 50},
                        "linear_backend_estimated_tflops": {
                            "mpsgraph-f32": 50 / 0.5 / 1e12,
                        },
                    },
                },
                "accelerated_linear_flop_fraction": 0.2,
            },
            "prompt_prefill": prompt_prefill,
        },
    }


def _sample_text_result(
    *,
    include_prompt_elapsed: bool = False,
    prepared_root: str | Path = "/tmp/prepared",
) -> dict[str, object]:
    result = _sample_result(
        include_prompt_elapsed=include_prompt_elapsed,
        prepared_root=prepared_root,
    )
    token_result = result["token_result"]
    assert isinstance(token_result, dict)
    return {
        "schema": "largerlm.prepared_text_generation_result.v1",
        "source": "generate_prepared_text",
        "prepared_manifest": result["prepared_manifest"],
        "request": result["request"],
        "text_result": {
            "prompt": "AB",
            "generated_text": "C",
            "full_text": "ABC",
            "tokenizer_backend": "simple",
            "tokenizer_path": "/tmp/prepared/../model/simple_tokenizer.json",
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [7],
            "eos_token_id": None,
            "token_result": token_result,
            "applied_launch_profile": token_result["applied_launch_profile"],
        },
    }


def test_result_summary_estimates_unattributed_without_prompt_elapsed(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    profile_sha = hashlib.sha256(
        (prepared / "launch-profile.json").read_bytes()
    ).hexdigest()
    summary = summarize_result_payload(_sample_result(prepared_root=prepared))

    assert summary["total_elapsed_seconds"] == 10.0
    assert summary["launch_binding"] == {
        "prepared_manifest": str(prepared / "manifest.json"),
        "prepared_dir": str(prepared),
        "prepared_manifest_exists": True,
        "current_prepared_manifest": str(prepared / "manifest.json"),
        "prepared_manifest_matches": True,
        "launch_profile_path": str(prepared / "launch-profile.json"),
        "launch_profile_exists": True,
        "launch_profile_file_sha256": profile_sha,
        "launch_profile_sha256_matches_file": True,
        "launch_audit_path": str(prepared / "launch-audit.json"),
        "launch_audit_exists": True,
        "launch_audit_json_valid": True,
        "launch_audit_schema": "largerlm.launch_audit.v1",
        "launch_audit_ok": True,
        "launch_audit_profile_path": str(prepared / "launch-profile.json"),
        "launch_audit_profile_path_matches": True,
        "launch_audit_profile_sha256": profile_sha,
        "launch_audit_profile_sha256_matches": True,
        "launch_audit_binding_matches": True,
        "applied_launch_profile_sha256": profile_sha,
        "applied_launch_profile_locked": True,
        "applied_launch_profile_lock_required": True,
        "applied_launch_profile_matches_prepared": True,
        "applied_launch_profile_argv_safe_to_replay": True,
        "safe_to_replay": True,
        "replay_ready": True,
        "replay_files_ready": True,
        "replay_prompt_token_count": 2,
        "replay_max_new_tokens": 1,
        "replay_mla_kv_b_cache_dir": None,
        "replay_mla_kv_b_cache_dir_exists": None,
        "replay_mla_kv_b_cache_expected_file_count": None,
        "replay_mla_kv_b_cache_current_file_count": None,
        "replay_mla_kv_b_cache_expected_total_bytes": None,
        "replay_mla_kv_b_cache_current_total_bytes": None,
        "replay_mla_kv_b_cache_ready": None,
        "replay_generate_token_ids_argv": [
            "python",
            "-m",
            "largerlm",
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(prepared / "launch-profile.json"),
            "--lock-launch-profile",
            "--require-locked-launch-profile",
            "--require-launch-audit",
            str(prepared / "launch-audit.json"),
            str(prepared),
            "--prompt-token-ids",
            "1,2",
            "--max-new-tokens",
            "1",
        ],
        "replay_generate_token_ids_command": " ".join(
            [
                "python",
                "-m",
                "largerlm",
                "generate-prepared-token-ids",
                "--apply-launch-profile",
                str(prepared / "launch-profile.json"),
                "--lock-launch-profile",
                "--require-locked-launch-profile",
                "--require-launch-audit",
                str(prepared / "launch-audit.json"),
                str(prepared),
                "--prompt-token-ids",
                "1,2",
                "--max-new-tokens",
                "1",
            ]
        ),
    }
    assert summary["known_subphase_elapsed_seconds"] == pytest.approx(4.25)
    assert summary["unattributed_elapsed_seconds"] == pytest.approx(5.75)
    prompt = summary["prompt_prefill"]
    assert prompt["expert_stage_planned_read_gib"] == pytest.approx(3.0)
    assert prompt["expert_stage_io"] == {
        "source": "generation_actual_prefill",
        "serial_read_bytes": 6 * 1024**3,
        "unique_requested_bytes": 2 * 1024**3,
        "planned_read_bytes": 3 * 1024**3,
        "waste_bytes": 1 * 1024**3,
        "coalesced_savings_bytes": 3 * 1024**3,
        "raw_ranges": 4,
        "coalesced_ranges": 2,
        "read_advice_attempted_ranges": 2,
        "read_advice_calls": 2,
        "read_advice_bytes": 3 * 1024**3,
        "read_advice_failures": 0,
        "copy_read_calls": 3,
        "copy_write_calls": 3,
        "copy_average_read_bytes": float(1024**3),
        "copy_average_write_bytes": float(1024**3),
        "copy_read_call_counterfactuals_by_chunk_mib": {
            "8": 6,
            "16": 4,
            "32": 3,
        },
        "assignment_read_amplification": 0.5,
        "unique_read_amplification": 1.5,
        "max_unique_read_amplification": 2.0,
        "max_stage_budget_utilization": 0.75,
        "stage_count": 1,
        "copy_hotspots": [
            {
                "chunk_index": 0,
                "layer": 0,
                "tile_index": 0,
                "selected_experts": [3, 5],
                "raw_range_count": 2,
                "coalesced_range_count": 1,
                "planned_read_bytes": 256,
                "copy_elapsed_seconds": 0.125,
                "copy_read_calls": 3,
            }
        ],
        "range_hotspots": [
            {
                "chunk_index": 0,
                "layer": 0,
                "tile_index": 0,
                "selected_experts": [3, 5],
                "raw_range_count": 2,
                "coalesced_range_count": 1,
                "planned_read_bytes": 256,
                "copy_elapsed_seconds": 0.125,
                "copy_read_calls": 3,
            }
        ],
    }
    assert prompt["linear_elapsed_seconds"] == pytest.approx(2.5)
    assert prompt["top_linear_components"][0]["component"] == "attention.o_proj"
    assert prompt["top_linear_components"][0]["elapsed_seconds"] == pytest.approx(0.7)
    assert prompt["top_linear_components"][0]["estimated_flops"] == 160
    assert prompt["top_linear_components"][1]["component"] == "moe.router_gate_proj"
    assert prompt["mla_key_cache"] == {
        "observed": True,
        "layer_count": 1,
        "enabled_layer_count": 1,
        "disabled_layer_count": 0,
        "all_layers_enabled": True,
        "total_mla_key_cache_bytes": 16,
    }
    assert prompt["mla_value_cache"] == {
        "observed": True,
        "layer_count": 1,
        "enabled_layer_count": 1,
        "disabled_layer_count": 0,
        "all_layers_enabled": True,
        "total_mla_value_cache_bytes": 32,
    }
    assert prompt["mla_attention_layers"]["count"] == 1
    assert prompt["mla_attention_layers"]["elapsed_seconds"] == pytest.approx(1.25)
    assert prompt["mla_attention_layers"]["top_slowest_layers"][0]["layer"] == 0
    assert prompt["mla_attention_layers"]["top_slowest_layers"][0][
        "elapsed_seconds"
    ] == pytest.approx(1.25)
    signature = summary["prefill_plan_signature"]
    assert signature["present"] is True
    assert signature["chunk_count"] == 1
    assert signature["chunk_tokens"] == 2
    assert signature["linear_backend_counts"] == {
        "custom-metal": 2,
        "mpsgraph-f32": 1,
    }

    assert signature["linear_backend_flops"] == {
        "custom-metal": 200,
        "mpsgraph-f32": 50,
    }
    assert signature["expert_stage_planned_read_bytes"] == 3 * 1024**3
    assert signature["mla_key_cache"]["all_layers_enabled"] is True
    assert signature["mla_value_cache"]["all_layers_enabled"] is True
    actual = summary["prefill_actual"]
    assert actual["accelerated_flop_fraction"] == pytest.approx(0.2)
    assert actual["total_estimated_flops"] == 250
    assert actual["accelerated_estimated_flops"] == 50
    assert actual["streamed_routed_expert_estimated_flops"] == 128
    assert actual["streamed_routed_expert_matrix_count"] == 1
    assert actual["router_gate_acceleration_analyzed"] is True
    assert actual["router_gate_accelerated_matrix_count"] == 1
    assert actual["router_gate_accelerated_estimated_flops"] == 50
    assert actual["non_router_matrix_count"] == 2
    assert actual["non_router_estimated_flops"] == 200
    assert actual["non_router_accelerated_matrix_count"] == 0
    assert actual["non_router_accelerated_estimated_flops"] == 0
    assert actual["non_router_unaccelerated_matrix_count"] == 2
    assert actual["non_router_unaccelerated_estimated_flops"] == 200
    assert actual["non_router_unaccelerated_flop_fraction"] == pytest.approx(0.8)
    assert (
        actual["non_router_unaccelerated_streamed_routed_expert_matrix_count"]
        == 1
    )
    assert (
        actual["non_router_unaccelerated_streamed_routed_expert_estimated_flops"]
        == 128
    )
    assert actual["non_router_unaccelerated_non_streamed_matrix_count"] == 1
    assert actual["non_router_unaccelerated_non_streamed_estimated_flops"] == 72
    assert actual["unaccelerated_backend_matrix_counts"] == {"custom-metal": 2}
    assert actual["unaccelerated_backend_estimated_flops"] == {"custom-metal": 200}
    assert actual["accelerated_router_gate_flop_share"] == pytest.approx(1.0)
    assert actual["accelerated_router_gate_only"] is True
    assert actual["routed_moe_elapsed_seconds"] == pytest.approx(0.25)
    assert actual["routed_moe_estimated_tflops"] == pytest.approx(
        128 / 0.25 / 1e12
    )
    assert actual["routed_moe_custom_elapsed_fraction"] == pytest.approx(0.125)
    routed_layers = actual["routed_moe_layers"]
    assert routed_layers["count"] == 1
    assert routed_layers["elapsed_seconds"] == pytest.approx(0.25)
    assert routed_layers["total_assignments"] == 4
    assert routed_layers["total_selected_expert_slots"] == 2
    assert routed_layers["total_stage_planned_read_bytes"] == 256
    assert routed_layers["total_stage_plus_compact_bytes"] == 384
    assert routed_layers["total_stage_plus_compact_materialized_bytes"] == 256
    assert routed_layers["total_compact_stage_materialized_bytes"] == 0
    assert routed_layers["total_static_capacity_used_slots"] == 4
    assert routed_layers["total_static_capacity_slots"] == 6
    assert routed_layers["static_capacity_utilization"] == pytest.approx(4 / 6)
    assert routed_layers["max_effective_moe_token_block"] == 2
    assert routed_layers["max_moe_estimated_peak_bytes"] == 512
    assert routed_layers["max_copy_chunk_bytes"] == 8 * 1024 * 1024
    assert routed_layers["moe_timing_elapsed_seconds"] == {
        "expert_read": pytest.approx(0.03),
        "final_read": pytest.approx(0.002),
        "input_read": pytest.approx(0.01),
        "kernel": pytest.approx(0.16),
        "mxfp4_down_add_kernel": pytest.approx(0.04),
        "mxfp4_swiglu_kernel": pytest.approx(0.11),
        "output_read": pytest.approx(0.02),
        "output_write": pytest.approx(0.015),
        "setup": pytest.approx(0.002),
        "sort": pytest.approx(0.001),
        "total": pytest.approx(0.24),
    }
    assert routed_layers["moe_timing_elapsed_fraction"]["kernel"] == pytest.approx(
        0.16 / 0.25
    )
    assert routed_layers["moe_runner_total_elapsed_seconds"] == pytest.approx(0.24)
    assert routed_layers["moe_runner_total_elapsed_fraction"] == pytest.approx(
        0.24 / 0.25
    )
    assert routed_layers["moe_non_runner_elapsed_seconds"] == pytest.approx(0.01)
    assert routed_layers["moe_non_runner_elapsed_fraction"] == pytest.approx(
        0.01 / 0.25
    )
    assert routed_layers["moe_timing_top_phase"] == "kernel"
    assert routed_layers["moe_timing_top_phase_seconds"] == pytest.approx(0.16)
    assert routed_layers["moe_timing_top_phase_fraction"] == pytest.approx(
        0.16 / 0.25
    )
    assert routed_layers["moe_mxfp4_kernel_split_elapsed_seconds"] == {
        "down_add": pytest.approx(0.04),
        "swiglu": pytest.approx(0.11),
        "total": pytest.approx(0.15),
    }
    assert routed_layers["moe_mxfp4_kernel_split_fraction"] == {
        "down_add": pytest.approx(0.04 / 0.16),
        "swiglu": pytest.approx(0.11 / 0.16),
        "total": pytest.approx(0.15 / 0.16),
    }
    assert routed_layers["moe_mxfp4_kernel_split_top_phase"] == "swiglu"
    assert routed_layers["moe_mxfp4_kernel_split_top_phase_seconds"] == pytest.approx(
        0.11
    )
    assert routed_layers["moe_mxfp4_kernel_split_top_phase_fraction"] == pytest.approx(
        0.11 / 0.16
    )
    assert routed_layers["router_margin_layer_count"] == 1
    assert routed_layers["router_min_effective_score_margin"] == pytest.approx(
        0.0002
    )
    assert routed_layers["router_min_topk_score_margin"] == pytest.approx(0.0003)
    assert routed_layers["router_min_group_score_margin"] == pytest.approx(0.0002)
    assert routed_layers["router_effective_near_tie_counts"] == {
        "le_1e-06": 0,
        "le_1e-05": 0,
        "le_1e-04": 0,
        "le_1e-03": 2,
    }
    assert routed_layers["router_gate_policy"] == {
        "layer_count": 1,
        "decision_counts": {"mpsgraph-f32-fallback": 1},
        "mode_counts": {"custom-first-mpsgraph-fallback": 1},
        "margin_threshold": pytest.approx(0.001),
        "margin_threshold_min": pytest.approx(0.001),
        "margin_threshold_max": pytest.approx(0.001),
        "command_count": 2,
        "custom_elapsed_seconds": pytest.approx(0.05),
        "fallback_elapsed_seconds": pytest.approx(0.5),
        "total_elapsed_seconds": pytest.approx(0.55),
        "extra_custom_probe_elapsed_seconds": pytest.approx(0.05),
    }
    hints = routed_layers["bottleneck_hints"]
    assert hints["token_block_observed_layer_count"] == 1
    assert hints["token_block_limited_layer_count"] == 0
    assert hints["token_block_status"] == "saturates_expert_fanout"
    assert hints["stage_copy_to_routed_elapsed_ratio"] == pytest.approx(0.5)
    assert hints["stage_copy_throughput_gib_per_second"] == pytest.approx(
        (256 / 1024**3) / 0.125
    )
    assert hints["top_stage_plus_compact_layer_share"] == pytest.approx(1.0)
    assert hints["moe_mxfp4_kernel_split_top_phase"] == "swiglu"
    assert hints["moe_mxfp4_kernel_split_top_phase_fraction"] == pytest.approx(
        0.11 / 0.16
    )
    assert [item["argv"] for item in hints["suggested_stage_copy_experiments"]] == [
        ["--prefill-copy-chunk-mib", "32"],
        ["--prefill-copy-chunk-mib", "64"],
    ]
    assert all(
        item["requires_bakeoff"] is True
        for item in hints["suggested_stage_copy_experiments"]
    )
    assert {
        item["promotion_gate"]
        for item in hints["suggested_stage_copy_experiments"]
    } == {"result-bakeoff-total-latency-win"}
    assert hints["moe_timing_top_phase"] == "kernel"
    assert hints["moe_timing_top_phase_fraction"] == pytest.approx(0.16 / 0.25)
    assert hints["moe_runner_total_elapsed_fraction"] == pytest.approx(0.24 / 0.25)
    assert hints["moe_non_runner_elapsed_seconds"] == pytest.approx(0.01)
    assert hints["moe_non_runner_elapsed_fraction"] == pytest.approx(0.01 / 0.25)
    assert hints["moe_timing_io_elapsed_fraction"] == pytest.approx(
        (0.03 + 0.01 + 0.02 + 0.015 + 0.002) / 0.25
    )
    assert hints["moe_timing_output_accumulator_elapsed_fraction"] == pytest.approx(
        (0.02 + 0.015) / 0.25
    )
    assert hints["suggested_moe_kernel_experiments"] == [
        {
            "env": {"LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING": "1"},
            "argv": [
                "python",
                "scripts/glm_moe_tile_sweep.py",
                "--repeat",
                "6",
                "--order",
                "interleave",
                "--layer",
                "0",
                "--batch-tokens",
                "2",
                "--experts",
                "3,5",
                "--tiles",
                "1",
                "--vector-swiglu-modes",
                "off,on",
                "--group32-modes",
                "auto,off",
            ],
            "scope": "bounded-layer-microbench",
            "requires_bakeoff": True,
            "promotion_gate": (
                "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
            ),
            "reason": (
                "split timing says the gate/up/SwiGLU side dominates; run a "
                "bounded vector-SwiGLU A/B against the default group32/auto "
                "path on the hottest routed layer, then promote only after "
                "numerical agreement and a full replay bakeoff win"
            ),
        }
    ]
    assert routed_layers["output_accumulator_counts"] == {"memory": 1}
    assert len(hints["notes"]) == 6
    assert any("gate/up/SwiGLU" in note for note in hints["notes"])
    assert routed_layers["top_slowest_layers"] == [
        {
            "assignments": 4,
            "batch_tokens": 2,
            "chunk_index": 0,
            "compact_stage_bytes": 128,
            "compact_stage_materialized_bytes": 0,
            "compact_stage_storage": "hardlink",
            "copy_chunk_bytes": 8 * 1024 * 1024,
            "effective_moe_token_block": 2,
            "elapsed_seconds": 0.25,
            "layer": 0,
            "layer_index": 0,
            "max_k": 2,
            "moe_batch_buffer_bytes": 64,
            "moe_estimated_peak_bytes": 512,
            "moe_max_expert_tokens": 2,
            "moe_output_accumulator": "memory",
            "moe_output_accumulator_bytes": 32,
            "moe_timing_expert_read_seconds": 0.03,
            "moe_timing_final_read_seconds": 0.002,
            "moe_timing_input_read_seconds": 0.01,
            "moe_timing_kernel_seconds": 0.16,
            "moe_timing_mxfp4_down_add_kernel_seconds": 0.04,
            "moe_timing_mxfp4_swiglu_kernel_seconds": 0.11,
            "moe_timing_output_read_seconds": 0.02,
            "moe_timing_output_write_seconds": 0.015,
            "moe_timing_setup_seconds": 0.002,
            "moe_timing_sort_seconds": 0.001,
            "moe_timing_total_seconds": 0.24,
            "moe_non_runner_elapsed_seconds": pytest.approx(0.01),
            "moe_non_runner_elapsed_fraction": pytest.approx(0.01 / 0.25),
            "moe_runner_total_elapsed_fraction": pytest.approx(0.24 / 0.25),
            "router_margin_summary": {
                "token_count": 2,
                "min_effective_score_margin": 0.0002,
                "mean_effective_score_margin": 0.00035,
                "effective_near_tie_counts": {
                    "le_1e-06": 0,
                    "le_1e-05": 0,
                    "le_1e-04": 0,
                    "le_1e-03": 2,
                },
                "min_topk_score_margin": 0.0003,
                "mean_topk_score_margin": 0.0004,
                "topk_near_tie_counts": {
                    "le_1e-06": 0,
                    "le_1e-05": 0,
                    "le_1e-04": 0,
                    "le_1e-03": 2,
                },
                "min_group_score_margin": 0.0002,
                "mean_group_score_margin": 0.0003,
                "group_near_tie_counts": {
                    "le_1e-06": 0,
                    "le_1e-05": 0,
                    "le_1e-04": 0,
                    "le_1e-03": 2,
                },
                "weakest_token": {
                    "token_index": 0,
                    "effective_score_margin": 0.0002,
                    "topk_score_margin": 0.0003,
                    "group_score_margin": 0.0002,
                },
                "router_gate_policy": {
                    "mode": "custom-first-mpsgraph-fallback",
                    "margin_threshold": 0.001,
                    "decision": "mpsgraph-f32-fallback",
                    "command_count": 2,
                    "custom_elapsed_seconds": 0.05,
                    "fallback_elapsed_seconds": 0.5,
                    "custom_min_effective_score_margin": 0.0002,
                    "fallback_min_effective_score_margin": 0.0003,
                },
            },
            "router_min_effective_score_margin": pytest.approx(0.0002),
            "router_min_group_score_margin": pytest.approx(0.0002),
            "router_min_topk_score_margin": pytest.approx(0.0003),
            "selected_experts": [3, 5],
            "selected_expert_count": 2,
            "stage_coalesced_range_count": 1,
            "stage_copy_elapsed_seconds": 0.125,
            "stage_copy_throughput_gib_per_second": 1.0,
            "stage_planned_read_bytes": 256,
            "stage_plus_compact_bytes": 384,
            "stage_plus_compact_materialized_bytes": 256,
            "stage_raw_range_count": 2,
            "stage_staged_bytes": 256,
            "stage_unique_requested_bytes": 256,
            "stage_waste_bytes": 0,
            "static_capacity_per_expert": 2,
            "static_capacity_total_slots": 6,
            "static_capacity_used_slots": 4,
            "static_capacity_utilization": pytest.approx(4 / 6),
            "top_k": 2,
        }
    ]
    assert actual["acceleration_coverage"]["source"] == "generation_actual_prefill"
    assert actual["linear_backend"]["source"] == "generation_actual_prefill"
    assert summary["elapsed_records"]["backend_counts"] == {
        "custom-metal": 1,
        "mpsgraph-f32": 1,
    }
    tensor_groups = summary["elapsed_records"]["tensor_suffix_groups"]
    assert len(tensor_groups) == 2
    assert tensor_groups[0]["tensor_suffix"] == "self_attn.o_proj.weight"
    assert tensor_groups[0]["count"] == 1
    assert tensor_groups[0]["elapsed_seconds"] == pytest.approx(0.7)
    assert tensor_groups[0]["backend_counts"] == {"custom-metal": 1}
    assert tensor_groups[0]["backend_elapsed_seconds"] == {
        "custom-metal": pytest.approx(0.7)
    }
    assert tensor_groups[0]["top_record"]["tensor"] == (
        "model.layers.0.self_attn.o_proj.weight"
    )
    assert tensor_groups[1]["tensor_suffix"] == "mlp.gate.weight"
    assert tensor_groups[1]["backend_counts"] == {"mpsgraph-f32": 1}
    named = summary["named_elapsed_fields"]
    assert named["field_elapsed_seconds"]["mla_attention_elapsed_seconds"] == 1.25
    assert named["field_top_records"]["projections_elapsed_seconds"] == {
        "field": "projections_elapsed_seconds",
        "elapsed_seconds": pytest.approx(0.08),
        "path": (
            "token_result.prompt_prefill.chunks[0].layers[0]."
            "attention.projections_elapsed_seconds"
        ),
    }
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    assert targets["prefill_non_router_acceleration_gap"]["rank_score"] == pytest.approx(
        8.0
    )
    assert targets["routed_moe"]["kind"] == "streamed_moe_kernel"
    assert targets["routed_moe"]["elapsed_seconds"] == pytest.approx(0.25)
    routed_experiments = targets["routed_moe"]["suggested_experiments"]
    assert routed_experiments[0]["scope"] == "bounded-layer-microbench"
    assert routed_experiments[0]["requires_bakeoff"] is True
    assert routed_experiments[0]["argv"][:4] == [
        "python",
        "scripts/glm_moe_tile_sweep.py",
        "--repeat",
        "6",
    ]
    assert targets["expert_stage_copy"]["kind"] == "ssd_streaming"
    assert targets["expert_stage_copy"]["elapsed_seconds"] == pytest.approx(1.5)
    stage_io = summary["prompt_prefill"]["expert_stage_io"]
    assert stage_io["stage_count"] == 1
    assert stage_io["copy_hotspots"][0]["layer"] == 0
    assert stage_io["range_hotspots"][0]["coalesced_range_count"] == 1
    text = format_result_summary_text(summary)
    assert "launch binding:" in text
    assert "safe_to_replay=True" in text
    assert "replay_ready=True" in text
    assert "files_ready=True" in text
    assert (
        "prefill non-router accel gap: matrices=2 flops=200 fraction=80.0% "
        "streamed=128 non_streamed=72 backends=custom-metal:200"
        in text
    )
    assert "routed moe slow layers:" in text
    assert "linear component hot spots:" in text
    assert "attention.o_proj" in text
    assert "MLA key cache: enabled=1/1 bytes=16B" in text
    assert (
        "expert stage io: serial=6.000GiB unique=2.000GiB "
        "planned=3.000GiB waste=1.000GiB savings=3.000GiB "
        "unique_amp=1.500x serial_ratio=0.500x ranges=4/2 "
        "advice=2/2 failures=0 copy_calls=3/3 "
        "copy_avg=1.000GiB/1.000GiB copy_calls_if=8MiB=6,16MiB=4,32MiB=3 "
        "stage_budget=75.0%"
    ) in text
    assert (
        "expert stage hotspots: "
        "copy=c0/L0/t0(0.125s,2/1r,256B,3calls) "
        "ranges=c0/L0/t0(0.125s,2/1r,256B,3calls)"
    ) in text
    assert "optimization targets:" in text
    assert "prefill_non_router_acceleration_gap[prefill_acceleration_gap]" in text
    assert "optimization target experiments (bounded; require bakeoff):" in text
    assert "routed_moe: LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1" in text
    assert "routed moe: elapsed=0.250s 5.12e-10 TFLOP/s custom_share=12.5% non_runner=4.0%" in text
    assert "expert_stage_copy[ssd_streaming]=1.500s/15.0%" in text
    assert "tensor hot spots:" in text
    assert "0.700s x1 custom-metal=1 self_attn.o_proj.weight" in text
    assert "token_block=saturates_expert_fanout" in text
    assert "moe_top=kernel:64.0%" in text
    assert "non_runner=4.0%" in text
    assert (
        "routed moe MXFP4 split: swiglu=0.110s down_add=0.040s "
        "total=0.150s top=swiglu:68.8%"
    ) in text
    assert (
        "routed moe kernel experiments "
        "(microbench first; require result-bakeoff win): "
        "LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING=1 "
        "python scripts/glm_moe_tile_sweep.py --repeat 6 --order interleave "
        "--layer 0 --batch-tokens 2 --experts 3,5 --tiles 1 "
        "--vector-swiglu-modes off,on --group32-modes auto,off"
    ) in text
    assert "router margins: layers=1 effective_min=0.0002" in text
    assert (
        "router hybrid: layers=1 threshold=0.001 "
        "decisions=mpsgraph-f32-fallback:1 elapsed=0.550s "
        "fallback_custom_probe=0.050s"
    ) in text
    assert "router_margin=0.0002" in text
    assert (
        "routed moe copy experiments (A/B only; require result-bakeoff win): "
        "--prefill-copy-chunk-mib 32"
    ) in text
    assert "compact=hardlink" in text
    assert "accum=memory" in text
    assert "layer=0" in text
    assert "assignments=4" in text
    assert (
        "timing=kernel=0.160s,swiglu=0.110s,down_add=0.040s,"
        "expert_read=0.030s"
    ) in text


def test_result_summary_suggests_bounded_moe_target_experiment_without_split_hint() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    staged = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "staged_mlp"
    ]["staged_moe"]
    assert isinstance(staged, dict)
    staged["moe_timing_mxfp4_swiglu_kernel_seconds"] = 0.04
    staged["moe_timing_mxfp4_down_add_kernel_seconds"] = 0.04

    summary = summarize_result_payload(payload)
    routed_layers = summary["prefill_actual"]["routed_moe_layers"]
    hints = routed_layers["bottleneck_hints"]
    assert hints["suggested_moe_kernel_experiments"] == []

    targets = {item["target"]: item for item in summary["optimization_targets"]}
    routed = targets["routed_moe"]
    experiments = routed["suggested_experiments"]
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["scope"] == "bounded-layer-microbench"
    assert experiment["requires_bakeoff"] is True
    assert experiment["promotion_gate"] == (
        "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
    )
    assert experiment["env"] == {"LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING": "1"}
    assert experiment["argv"] == [
        "python",
        "scripts/glm_moe_tile_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--layer",
        "0",
        "--batch-tokens",
        "2",
        "--experts",
        "3,5",
        "--tiles",
        "1,2",
        "--vector-swiglu-modes",
        "off,on",
        "--group32-modes",
        "auto,off",
        "--max-stage-mib",
        "64",
        "--max-compact-stage-mib",
        "64",
        "--max-runner-scratch-mib",
        "256",
        "--copy-chunk-mib",
        "8",
        "--write-result",
        (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            "glm-moe-layer0-optimization-target-2tok.json"
        ),
    ]
    assert experiment["safety"] == {
        "source": "routed_moe.top_slowest_layers[0]",
        "layer": 0,
        "batch_tokens": 2,
        "selected_expert_count": 2,
        "selected_experts": [3, 5],
        "max_stage_mib": 64,
        "max_compact_stage_mib": 64,
        "max_runner_scratch_mib": 256,
        "observed_moe_estimated_peak_bytes": 512,
        "observed_stage_planned_read_bytes": 256,
    }

    text = format_result_summary_text(summary)
    assert "--max-runner-scratch-mib 256" in text
    assert "glm-moe-layer0-optimization-target-2tok.json" in text


def test_result_summary_promotes_routed_moe_process_boundary_when_non_runner_dominates() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    staged = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "staged_mlp"
    ]
    assert isinstance(staged, dict)
    staged["routed_moe_elapsed_seconds"] = 1.0
    staged["moe_timing_total_seconds"] = 0.2
    staged["moe_timing_kernel_seconds"] = 0.1
    staged_moe = staged["staged_moe"]
    assert isinstance(staged_moe, dict)
    staged_moe.update(
        {
            "wall_total_elapsed_seconds": 0.30,
            "wall_compact_stage_elapsed_seconds": 0.04,
            "wall_routes_elapsed_seconds": 0.03,
            "wall_static_capacity_elapsed_seconds": 0.02,
            "wall_runner_elapsed_seconds": 0.20,
            "wall_output_validation_elapsed_seconds": 0.01,
        }
    )

    summary = summarize_result_payload(payload)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    routed = targets["routed_moe"]
    routed_layers = summary["prefill_actual"]["routed_moe_layers"]
    hints = routed_layers["bottleneck_hints"]

    assert routed["kind"] == "process_boundary"
    assert routed["evidence"]["non_runner_elapsed_fraction"] == pytest.approx(0.8)
    assert routed["evidence"]["runner_total_elapsed_fraction"] == pytest.approx(0.2)
    assert "persistent routed-MoE runner" in routed["suggested_next_step"]
    assert routed_layers["moe_wall_residual_elapsed_seconds"] == pytest.approx(0.70)
    assert routed_layers["moe_wall_residual_elapsed_fraction"] == pytest.approx(0.70)
    assert hints["moe_wall_residual_elapsed_seconds"] == pytest.approx(0.70)
    assert hints["moe_wall_residual_elapsed_fraction"] == pytest.approx(0.70)
    assert any(
        "large residual outside the recorded wall phase split" in note
        for note in hints["notes"]
    )
    assert "residual=0.700s" in format_result_summary_text(summary)


def test_result_summary_reports_routed_moe_wall_phase_breakdown() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    staged_moe = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "staged_mlp"
    ]["staged_moe"]
    assert isinstance(staged_moe, dict)
    staged_moe.update(
        {
            "wall_total_elapsed_seconds": 0.30,
            "wall_compact_stage_elapsed_seconds": 0.04,
            "wall_routes_elapsed_seconds": 0.03,
            "wall_static_capacity_elapsed_seconds": 0.02,
            "wall_runner_elapsed_seconds": 0.20,
            "wall_output_validation_elapsed_seconds": 0.01,
        }
    )

    summary = summarize_result_payload(payload)
    routed = summary["prefill_actual"]["routed_moe_layers"]

    assert routed["moe_wall_elapsed_seconds"] == {
        "compact_stage": pytest.approx(0.04),
        "output_validation": pytest.approx(0.01),
        "routes": pytest.approx(0.03),
        "runner": pytest.approx(0.20),
        "static_capacity": pytest.approx(0.02),
        "total": pytest.approx(0.30),
    }
    assert routed["moe_wall_elapsed_fraction"]["runner"] == pytest.approx(0.20 / 0.25)
    assert routed["moe_wall_residual_elapsed_seconds"] == pytest.approx(0.0)
    assert routed["moe_wall_residual_elapsed_fraction"] == pytest.approx(0.0)
    text = format_result_summary_text(summary)
    assert (
        "routed moe wall: total=0.300s runner_wall=0.200s "
        "compact=0.040s routes=0.030s static=0.020s validate=0.010s "
        "residual=0.000s"
    ) in text


def test_result_summary_suggests_bounded_attention_output_target_experiment() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    attention = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]
    assert isinstance(attention, dict)
    attention["attention_output_elapsed_seconds"] = 0.7

    summary = summarize_result_payload(payload)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    attention_output = targets["attention_output"]
    experiments = attention_output["suggested_experiments"]
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["scope"] == "bounded-resident-projection-microbench"
    assert experiment["requires_bakeoff"] is True
    assert experiment["argv"] == [
        "python",
        "scripts/glm_resident_mxfp4_group32_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--layer",
        "0",
        "--batch-tokens",
        "2",
        "--group32-modes",
        "off,auto",
        "--max-resident-matrix-mib",
        "256",
        "--max-runner-scratch-mib",
        "256",
        "--write-result",
        (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            "glm-resident-o-proj-layer0-optimization-target-2tok.json"
        ),
    ]
    assert experiment["safety"] == {
        "source": "elapsed_records.tensor_suffix_groups.self_attn.o_proj.weight",
        "layer": 0,
        "batch_tokens": 2,
        "max_resident_matrix_mib": 256,
        "max_runner_scratch_mib": 256,
        "observed_group_elapsed_seconds": pytest.approx(0.7),
        "observed_top_record_elapsed_seconds": pytest.approx(0.7),
    }


def test_result_summary_suggests_bounded_attention_projection_fusion_experiment() -> None:
    payload = _sample_result(include_prompt_elapsed=True)

    summary = summarize_result_payload(payload)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    attention_projections = targets["attention_projections"]
    experiments = attention_projections["suggested_experiments"]
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["scope"] == "bounded-attention-projection-fusion-microbench"
    assert experiment["requires_bakeoff"] is True
    assert experiment["write_result"] == (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        "glm-attn-proj-layer0-fusion-sweep-2tok.json"
    )
    assert experiment["argv"] == [
        "python",
        "scripts/glm_attention_projection_fusion_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--layer",
        "0",
        "--batch-tokens",
        "2",
        "--modes",
        "fused,separate",
        "--max-resident-matrix-mib",
        "256",
        "--max-runner-scratch-mib",
        "256",
        "--write-result",
        (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            "glm-attn-proj-layer0-fusion-sweep-2tok.json"
        ),
    ]
    assert experiment["safety"] == {
        "source": (
            "named_elapsed_fields.field_top_records."
            "projections_elapsed_seconds"
        ),
        "layer": 0,
        "batch_tokens": 2,
        "max_resident_matrix_mib": 256,
        "max_runner_scratch_mib": 256,
        "observed_top_record_elapsed_seconds": pytest.approx(0.08),
    }


def test_result_summary_suggests_bounded_rope_split_fusion_experiment() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    attention = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]
    assert isinstance(attention, dict)
    attention["rope_elapsed_seconds"] = 0.2
    attention["mla_attention"] = {
        "layer": 0,
        "batch_tokens": 2,
        "context_length": 2,
        "start_position": 0,
        "num_heads": 64,
        "qk_nope_dim": 192,
        "rope_dim": 64,
        "v_head_dim": 256,
        "kv_lora_dim": 512,
        "rope_theta": 8000000.0,
        "rope_interleave": True,
    }

    summary = summarize_result_payload(payload)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    rope = targets["rope_split"]
    experiments = rope["suggested_experiments"]
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["scope"] == "bounded-rope-split-fusion-microbench"
    assert experiment["requires_bakeoff"] is True
    assert experiment["write_result"] == (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        "glm-rope-split-layer0-fusion-sweep-2tok.json"
    )
    assert experiment["argv"] == [
        "python",
        "scripts/glm_rope_split_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--batch-tokens",
        "2",
        "--num-heads",
        "64",
        "--qk-nope-dim",
        "192",
        "--rope-dim",
        "64",
        "--start-position",
        "0",
        "--max-runner-scratch-mib",
        "256",
        "--rope-theta",
        "8000000",
        "--rope-interleave",
        "--write-result",
        (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            "glm-rope-split-layer0-fusion-sweep-2tok.json"
        ),
    ]
    assert experiment["safety"] == {
        "source": "named_elapsed_fields.field_top_records.rope_elapsed_seconds",
        "layer": 0,
        "batch_tokens": 2,
        "num_heads": 64,
        "qk_nope_dim": 192,
        "rope_dim": 64,
        "max_runner_scratch_mib": 256,
        "observed_top_record_elapsed_seconds": pytest.approx(0.2),
    }


def test_result_summary_suggests_bounded_cache_write_chunk_experiment() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    attention = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]
    assert isinstance(attention, dict)
    attention["cache_write_elapsed_seconds"] = 0.3

    summary = summarize_result_payload(payload)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    cache_write = targets["cache_write"]
    experiments = cache_write["suggested_experiments"]
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["scope"] == "bounded-cache-write-chunk-sweep"
    assert experiment["requires_bakeoff"] is True
    assert experiment["write_result"] == (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        "glm-cache-write-layer0-chunk-sweep-2tok.json"
    )
    assert experiment["argv"] == [
        "python",
        "scripts/glm_prefill_cache_write_sweep.py",
        "--prepared-dir",
        "/tmp/prepared",
        "--layer",
        "0",
        "--batch-tokens",
        "2",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--chunk-modes",
        "default,one-row,16KiB,256KiB,1MiB",
        "--max-cache-file-mib",
        "256",
        "--max-cache-write-mib",
        "256",
        "--write-result",
        (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            "glm-cache-write-layer0-chunk-sweep-2tok.json"
        ),
    ]
    assert experiment["safety"] == {
        "source": "named_elapsed_fields.field_top_records.cache_write_elapsed_seconds",
        "layer": 0,
        "batch_tokens": 2,
        "prepared_dir": "/tmp/prepared",
        "synthetic_cache_layout": True,
        "max_cache_file_mib": 256,
        "max_cache_write_mib": 256,
        "observed_top_record_elapsed_seconds": pytest.approx(0.3),
    }


def test_result_summary_suggests_bounded_mla_attention_cache_experiment() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    attention = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]
    assert isinstance(attention, dict)
    attention["mla_attention"] = {
        "layer": 0,
        "batch_tokens": 2,
        "context_length": 2,
        "start_position": 0,
        "num_heads": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 3,
        "kv_lora_dim": 4,
        "rope_theta": 10000.0,
        "rope_interleave": True,
        "cache_read_bytes": 16,
        "estimated_peak_bytes": 1024,
        "mla_key_cache": True,
        "mla_key_cache_bytes": 16,
        "mla_value_cache": True,
        "mla_value_cache_bytes": 32,
        "mla_timing_total_elapsed_seconds": 0.9,
        "mla_timing_value_read_elapsed_seconds": 0.3,
        "mla_timing_kernel_elapsed_seconds": 0.4,
    }

    summary = summarize_result_payload(payload)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    mla_target = targets["mla_attention"]
    experiments = mla_target["suggested_experiments"]
    assert len(experiments) == 1
    experiment = experiments[0]
    assert experiment["scope"] == "bounded-mla-attention-cache-microbench"
    assert experiment["requires_bakeoff"] is True
    assert experiment["argv"] == [
        "python",
        "scripts/glm_mla_attention_cache_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--layer",
        "0",
        "--context-length",
        "2",
        "--start-position",
        "0",
        "--batch-tokens",
        "2",
        "--num-heads",
        "2",
        "--qk-nope-dim",
        "1",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "3",
        "--kv-lora-dim",
        "4",
        "--cache-modes",
        "key-value,key-only,value-only,none",
        "--max-cache-read-mib",
        "256",
        "--max-resident-matrix-mib",
        "256",
        "--max-runner-scratch-mib",
        "256",
        "--rope-theta",
        "10000",
        "--rope-interleave",
        "--write-result",
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/glm-mla-layer0-cache-sweep-2tok.json",
    ]
    assert experiment["safety"] == {
        "source": "prompt_prefill.mla_attention_layers.top_slowest_layers[0]",
        "layer": 0,
        "context_length": 2,
        "batch_tokens": 2,
        "max_cache_read_mib": 256,
        "max_resident_matrix_mib": 256,
        "max_runner_scratch_mib": 256,
        "observed_elapsed_seconds": pytest.approx(1.25),
        "observed_timing_total_seconds": pytest.approx(0.9),
        "observed_value_read_seconds": pytest.approx(0.3),
        "observed_kernel_seconds": pytest.approx(0.4),
        "observed_estimated_peak_bytes": 1024,
        "observed_mla_key_cache_bytes": 16,
        "observed_mla_value_cache_bytes": 32,
        "observed_cache_read_bytes": 16,
    }


def test_result_summary_file_attaches_suggested_experiment_result(
    tmp_path: Path,
) -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    staged = payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "staged_mlp"
    ]["staged_moe"]
    assert isinstance(staged, dict)
    staged["moe_timing_mxfp4_swiglu_kernel_seconds"] = 0.04
    staged["moe_timing_mxfp4_down_add_kernel_seconds"] = 0.04
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    experiment_path = (
        tmp_path
        / "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        / "glm-moe-layer0-optimization-target-2tok.json"
    )
    experiment_path.parent.mkdir(parents=True)
    experiment_path.write_text(
        json.dumps(
            {
                "schema": "largerlm.glm_moe_tile_sweep.v2",
                "config_comparison": {
                    "baseline_config": "tile1_auto_silu",
                    "candidate_config": None,
                    "candidate_for_full_replay": False,
                    "fastest_kernel_config": "tile1_auto_silu",
                    "requires_full_replay_bakeoff": False,
                    "max_promotion_drift": 1e-5,
                    "min_promotion_speedup_ratio": 0.98,
                    "min_promotion_sample_count": 3,
                    "reasons": [
                        "no_candidate_met_kernel_speedup_and_drift_policy",
                        "baseline_has_fastest_kernel_mean",
                    ],
                    "rows": [{"config": "tile1_auto_silu"}],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_result_file(result_path)
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    experiment = targets["routed_moe"]["suggested_experiments"][0]
    assert experiment["write_result"] == (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        "glm-moe-layer0-optimization-target-2tok.json"
    )
    attached = experiment["result"]
    assert attached["present"] is True
    assert attached["schema"] == "largerlm.glm_moe_tile_sweep.v2"
    assert attached["resolved_path"] == str(experiment_path)
    comparison = attached["config_comparison"]
    assert comparison["baseline"] == "tile1_auto_silu"
    assert comparison["fastest"] == "tile1_auto_silu"
    assert comparison["candidate_for_full_replay"] is False
    assert comparison["requires_full_replay_bakeoff"] is False
    assert comparison["row_count"] == 1
    assert comparison["reasons"] == [
        "no_candidate_met_kernel_speedup_and_drift_policy",
        "baseline_has_fastest_kernel_mean",
    ]

    text = format_result_summary_text(summary)
    assert "optimization target experiment results:" in text
    assert (
        "routed_moe: present schema=largerlm.glm_moe_tile_sweep.v2 "
        "candidate=False baseline=tile1_auto_silu fastest=tile1_auto_silu"
    ) in text


def test_result_summary_reports_mpp_frontier_status() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    token = payload["token_result"]
    assert isinstance(token, dict)
    coverage = token["prefill_actual_acceleration_coverage"]
    assert isinstance(coverage, dict)
    coverage.update(
        {
            "matrix_count": 4,
            "mpp_candidate_policy": {
                "candidate_backend": "mpp_tensor_ops_prefill",
                "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
                "mpp_tensor_ops_min_batch_tokens": 128,
                "mpp_tensor_ops_min_matrix_dim": 32,
                "selectable_prefill_backend": False,
            },
            "mpp_tensor_ops_candidate_matrix_count": 3,
            "mpp_tensor_ops_candidate_estimated_flops": 750,
            "mpp_tensor_ops_candidate_flop_fraction": 0.75,
            "mpp_tensor_ops_candidate_backend_counts": {
                "custom-metal": 2,
                "fused-metal": 1,
            },
            "mpp_tensor_ops_candidate_backend_flops": {
                "custom-metal": 500,
                "fused-metal": 250,
            },
            "streamed_routed_expert_mpp_candidate_matrix_count": 1,
            "streamed_routed_expert_mpp_candidate_estimated_flops": 250,
            "total_estimated_flops": 1000,
        }
    )
    frontier = token["prefill_actual_acceleration_frontier"]
    assert isinstance(frontier, dict)
    frontier.update(
        {
            "analyzed": True,
            "configured_backend": "custom-metal",
            "prompt_token_count": 128,
            "resolved_prompt_chunk_tokens": 128,
            "reason": "effective prefill backend is custom-metal",
            "mpp_candidate_policy": coverage["mpp_candidate_policy"],
            "candidates": [
                {
                    "prompt_chunk_tokens": 128,
                    "is_resolved": True,
                    "viable_for_request": True,
                    "analyzed": True,
                    "matrix_count": 4,
                    "mpp_tensor_ops_candidate_matrix_count": 3,
                    "mpp_tensor_ops_candidate_estimated_flops": 750,
                    "mpp_tensor_ops_candidate_flop_fraction": 0.75,
                    "mpp_tensor_ops_candidate_backend_counts": {
                        "custom-metal": 2,
                        "fused-metal": 1,
                    },
                }
            ],
        }
    )

    summary = summarize_result_payload(payload)
    status = summary["prefill_actual"]["acceleration_frontier_status"]

    assert status["status"] == "mpp_backend_not_selectable"
    assert status["mpp_tensor_ops_candidate_matrix_count"] == 3
    assert status["matrix_count"] == 4
    assert status["mpp_tensor_ops_candidate_flop_fraction"] == pytest.approx(0.75)
    assert status["mpp_tensor_ops_candidate_backend_counts"] == {
        "custom-metal": 2,
        "fused-metal": 1,
    }
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    mpp_target = targets["mpp_tensor_ops_prefill"]
    assert mpp_target["kind"] == "prefill_acceleration_frontier"
    assert mpp_target["rank_score"] == pytest.approx(7.5)
    assert mpp_target["evidence"]["frontier_status"] == "mpp_backend_not_selectable"
    assert mpp_target["evidence"]["candidate_matrix_count"] == 3
    assert mpp_target["evidence"]["candidate_flop_fraction"] == pytest.approx(0.75)

    text = format_result_summary_text(summary)
    assert (
        "prefill MPP frontier: status=mpp_backend_not_selectable "
        "candidate=3/4 flops=75.0% selectable=False policy=tokens>=128,dim>=32"
    ) in text
    assert "prefill MPP backends: custom-metal=2,fused-metal=1" in text
    assert (
        "prefill MPP blocker: candidate shapes are present but "
        "mpp_tensor_ops_prefill is not selectable in this build"
    ) in text
    assert (
        "mpp_tensor_ops_prefill[prefill_acceleration_frontier]=75.0%flops"
    ) in text


def test_result_summary_reports_decode_layer_attention_hotspots() -> None:
    payload = _sample_result(include_prompt_elapsed=True)
    token = payload["token_result"]
    assert isinstance(token, dict)
    token["generated_token_ids"] = [7, 8]
    token["elapsed_seconds"] = 16.0
    steps = token["steps"]
    assert isinstance(steps, list)
    steps.append(
        {
            "position": 2,
            "input_token_id": 7,
            "selected_token_id": 8,
            "elapsed_seconds": 6.0,
            "logits_elapsed_seconds": 0.2,
            "expert_read_bytes": 1024,
            "cache_read_bytes": 64,
            "logits_read_bytes": 128,
            "decode_layers": [
                {
                    "layer": 0,
                    "elapsed_seconds": 1.2,
                    "attention_elapsed_seconds": 1.0,
                    "mlp_elapsed_seconds": 0.2,
                    "mlp_stage_elapsed_seconds": {
                        "expert_kernel": 0.08,
                        "expert_read": 0.03,
                        "router": 0.02,
                        "shared": 0.04,
                        "total": 0.2,
                    },
                    "mlp_diagnostics": {
                        "preload_selected_enabled": True,
                        "preload_selected_bytes": 8192,
                        "mxfp4_fused_decode_enabled": True,
                    },
                    "mla_attention_timing_elapsed_seconds": {
                        "kernel": 0.8,
                        "kernel_weights": 0.55,
                        "kernel_values": 0.25,
                        "total": 0.9,
                        "value_read": 0.05,
                    },
                },
                {
                    "layer": 1,
                    "elapsed_seconds": 0.7,
                    "attention_elapsed_seconds": 0.4,
                    "mlp_elapsed_seconds": 0.3,
                    "mlp_stage_elapsed_seconds": {
                        "expert_kernel": 0.12,
                        "expert_read": 0.04,
                        "router": 0.03,
                        "shared": 0.05,
                        "total": 0.3,
                    },
                    "mlp_diagnostics": {
                        "preload_selected_enabled": False,
                        "preload_selected_bytes": 0,
                        "mxfp4_fused_decode_enabled": False,
                    },
                    "mla_attention_timing_elapsed_seconds": {
                        "kernel": 0.3,
                        "kernel_weights": 0.2,
                        "kernel_values": 0.1,
                        "total": 0.35,
                        "cache_read": 0.01,
                    },
                },
            ],
        }
    )

    summary = summarize_result_payload(payload)

    decode = summary["decode_steps"]
    assert decode["present"] is True
    assert decode["generated_step_count"] == 2
    assert decode["step_count_with_decode_layers"] == 1
    assert decode["decode_layer_count"] == 2
    assert decode["decode_layer_elapsed_seconds"] == pytest.approx(1.9)
    assert decode["attention_elapsed_seconds"] == pytest.approx(1.4)
    assert decode["mlp_elapsed_seconds"] == pytest.approx(0.5)
    assert decode["mla_attention_timing_elapsed_seconds"] == {
        "cache_read": pytest.approx(0.01),
        "kernel": pytest.approx(1.1),
        "kernel_values": pytest.approx(0.35),
        "kernel_weights": pytest.approx(0.75),
        "total": pytest.approx(1.25),
        "value_read": pytest.approx(0.05),
    }
    assert decode["mlp_stage_elapsed_seconds"] == {
        "expert_kernel": pytest.approx(0.2),
        "expert_read": pytest.approx(0.07),
        "router": pytest.approx(0.05),
        "shared": pytest.approx(0.09),
        "total": pytest.approx(0.5),
    }
    assert decode["mlp_preload_selected_enabled_count"] == 1
    assert decode["mlp_preload_selected_bytes"] == 8192
    assert decode["mlp_mxfp4_fused_decode_enabled_count"] == 1
    assert decode["top_slowest_steps"][0]["position"] == 2
    assert decode["top_attention_layers"][0]["layer"] == 0
    assert decode["top_attention_layers"][0]["attention_elapsed_seconds"] == 1.0
    assert decode["top_mlp_layers"][0]["layer"] == 1
    assert decode["top_mlp_layers"][0]["mlp_elapsed_seconds"] == 0.3
    assert decode["top_mla_kernel_layers"][0]["layer"] == 0
    assert decode["top_mla_kernel_layers"][0]["mla_kernel_elapsed_seconds"] == 0.8

    text = format_result_summary_text(summary)
    assert "decode layers: steps=1/2 layers=2 elapsed=1.900s" in text
    assert "attention=1.400s mlp=0.500s" in text
    assert "top decode attention layers: L0=1.000s, L1=0.400s" in text
    assert "top decode MLP layers: L1=0.300s, L0=0.200s" in text
    assert "decode MLP timing: expert_kernel=0.200s expert_read=0.070s" in text
    assert "decode MLP selected preload: enabled=1/2 bytes=8.000KiB" in text
    assert "decode MLP MXFP4 fused decode: enabled=1/2" in text
    assert "decode MLA timing: kernel=1.100s weights=0.750s values=0.350s total=1.250s" in text
    assert "top decode MLA kernel layers: L0=0.800s, L1=0.300s" in text


def test_result_summary_replays_prefill_mla_kv_b_cache_dir(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    cache_dir = prepared / "mla-kv-b-cache"
    cache_dir.mkdir()
    (cache_dir / "layer-0-kvb-f32.bin").write_bytes(b"0" * 16)
    result = json.loads(json.dumps(_sample_result(prepared_root=prepared)))
    request = result["request"]
    assert isinstance(request, dict)
    request["prefill_mla_kv_b_cache_dir"] = str(cache_dir)
    request["prefill_mla_kv_b_cache_file_count"] = 1
    request["prefill_mla_kv_b_cache_total_bytes"] = 16

    summary = summarize_result_payload(result)

    binding = summary["launch_binding"]
    assert binding["replay_files_ready"] is True
    assert binding["replay_mla_kv_b_cache_dir"] == str(cache_dir)
    assert binding["replay_mla_kv_b_cache_dir_exists"] is True
    assert binding["replay_mla_kv_b_cache_expected_file_count"] == 1
    assert binding["replay_mla_kv_b_cache_current_file_count"] == 1
    assert binding["replay_mla_kv_b_cache_expected_total_bytes"] == 16
    assert binding["replay_mla_kv_b_cache_current_total_bytes"] == 16
    assert binding["replay_mla_kv_b_cache_ready"] is True
    argv = binding["replay_generate_token_ids_argv"]
    assert argv is not None
    assert argv[-6:] == [
        "--prefill-mla-kv-b-cache-dir",
        str(cache_dir),
        "--prompt-token-ids",
        "1,2",
        "--max-new-tokens",
        "1",
    ]


def test_result_summary_uses_compact_mla_cache_summary_without_chunks() -> None:
    result = json.loads(json.dumps(_sample_result()))
    prompt = result["token_result"]["prompt_prefill"]
    assert isinstance(prompt, dict)
    prompt.pop("chunks")
    prompt["mla_key_cache"] = {
        "observed": True,
        "layer_count": 1,
        "enabled_layer_count": 1,
        "disabled_layer_count": 0,
        "all_layers_enabled": True,
        "total_mla_key_cache_bytes": 16,
    }
    prompt["mla_value_cache"] = {
        "observed": True,
        "layer_count": 1,
        "enabled_layer_count": 1,
        "disabled_layer_count": 0,
        "all_layers_enabled": True,
        "total_mla_value_cache_bytes": 32,
    }

    summary = summarize_result_payload(result)

    signature = summary["prefill_plan_signature"]
    assert signature["mla_key_cache"]["enabled_layer_count"] == 1
    assert signature["mla_value_cache"]["enabled_layer_count"] == 1
    text = format_result_summary_text(summary)
    assert "MLA key cache: enabled=1/1 bytes=16B" in text


def test_result_summary_marks_missing_prefill_mla_kv_b_cache_not_files_ready(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    cache_dir = prepared / "missing-mla-kv-b-cache"
    result = json.loads(json.dumps(_sample_result(prepared_root=prepared)))
    request = result["request"]
    assert isinstance(request, dict)
    request["prefill_mla_kv_b_cache_dir"] = str(cache_dir)
    request["prefill_mla_kv_b_cache_file_count"] = 1
    request["prefill_mla_kv_b_cache_total_bytes"] = 16

    summary = summarize_result_payload(result)

    binding = summary["launch_binding"]
    assert binding["replay_ready"] is True
    assert binding["replay_files_ready"] is False
    assert binding["replay_mla_kv_b_cache_dir_exists"] is False
    assert binding["replay_mla_kv_b_cache_current_file_count"] is None
    assert binding["replay_mla_kv_b_cache_current_total_bytes"] is None
    assert binding["replay_mla_kv_b_cache_ready"] is False


def test_result_summary_handles_prepared_text_result_schema(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    summary = summarize_result_payload(
        _sample_text_result(
            include_prompt_elapsed=True,
            prepared_root=prepared,
        )
    )

    assert summary["prompt_tokens"] == 2
    assert summary["generated_token_ids"] == [7]
    assert summary["total_elapsed_seconds"] == 10.0
    assert summary["launch_binding"]["launch_profile_path"] == str(
        prepared / "launch-profile.json"
    )
    assert summary["launch_binding"]["replay_ready"] is True
    text = format_result_summary_text(summary)
    assert "tokens: prompt=2 generated=1" in text


def test_result_summary_suggests_next_copy_chunk_after_64_mib() -> None:
    result = json.loads(json.dumps(_sample_result()))
    staged_mlp = result["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "staged_mlp"
    ]
    staged_mlp["copy_chunk_bytes"] = 64 * 1024 * 1024
    staged_mlp["stage_result"]["copy_chunk_bytes"] = 64 * 1024 * 1024

    summary = summarize_result_payload(result)

    hints = summary["prefill_actual"]["routed_moe_layers"]["bottleneck_hints"]
    assert [item["argv"] for item in hints["suggested_stage_copy_experiments"]] == [
        ["--prefill-copy-chunk-mib", "128"],
    ]
    assert hints["suggested_stage_copy_experiments"][0]["requires_bakeoff"] is True
    assert (
        hints["suggested_stage_copy_experiments"][0]["promotion_gate"]
        == "result-bakeoff-total-latency-win"
    )


def test_result_summary_suggests_memory_accumulator_ab_for_file_output_io() -> None:
    result = json.loads(json.dumps(_sample_result()))
    staged_mlp = result["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "staged_mlp"
    ]
    staged_mlp["routed_moe_elapsed_seconds"] = 1.0
    staged_mlp["moe_output_accumulator"] = "file"
    staged_mlp["moe_output_accumulator_bytes"] = 0
    staged_moe = staged_mlp["staged_moe"]
    staged_moe["moe_output_accumulator"] = "file"
    staged_moe["moe_output_accumulator_bytes"] = 0
    staged_moe["moe_timing_output_read_seconds"] = 0.2
    staged_moe["moe_timing_output_write_seconds"] = 0.1
    staged_moe["moe_timing_total_seconds"] = 0.95

    summary = summarize_result_payload(result)

    routed_layers = summary["prefill_actual"]["routed_moe_layers"]
    hints = routed_layers["bottleneck_hints"]
    assert routed_layers["output_accumulator_counts"] == {"file": 1}
    assert hints["suggested_output_accumulator_experiments"] == [
        {
            "env": {"LARGERLM_MOE_BATCH_ACCUMULATOR": "memory"},
            "requires_bakeoff": True,
            "promotion_gate": "result-bakeoff-total-latency-win",
            "reason": (
                "file-backed output accumulator I/O is measurable; rerun under "
                "the same launch and memory guards with the opt-in memory "
                "accumulator, then promote only if a paired result bakeoff "
                "shows a total-latency win"
            ),
        }
    ]
    text = format_result_summary_text(summary)
    assert (
        "routed moe accumulator experiments "
        "(A/B only; require result-bakeoff win): "
        "LARGERLM_MOE_BATCH_ACCUMULATOR=memory"
    ) in text


def test_result_summary_uses_prompt_elapsed_when_present() -> None:
    summary = summarize_result_payload(_sample_result(include_prompt_elapsed=True))

    assert summary["known_subphase_elapsed_seconds"] == pytest.approx(8.25)
    assert summary["unattributed_elapsed_seconds"] == pytest.approx(1.75)
    assert summary["prompt_prefill"]["elapsed_seconds"] == pytest.approx(8.0)


def test_result_summary_reports_persistent_resident_linear_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"][
        "persistent_resident_linear_server"
    ] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_resident_linear_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_linear_server=yes" in text


def test_result_summary_reports_persistent_attention_projection_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"][
        "persistent_attention_projection_server"
    ] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_attention_projection_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_attention_projection_server=yes" in text


def test_result_summary_reports_persistent_attention_output_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"][
        "persistent_attention_output_server"
    ] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_attention_output_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_attention_output_server=yes" in text


def test_result_summary_reports_persistent_shared_expert_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"][
        "persistent_shared_expert_server"
    ] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_shared_expert_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_shared_expert_server=yes" in text


def test_result_summary_reports_persistent_rope_split_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"]["persistent_rope_split_server"] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_rope_split_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_rope_split_server=yes" in text


def test_result_summary_reports_persistent_mla_attention_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"][
        "persistent_mla_attention_server"
    ] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_mla_attention_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_mla_attention_server=yes" in text


def test_result_summary_reports_persistent_rmsnorm_server() -> None:
    result = json.loads(json.dumps(_sample_result(include_prompt_elapsed=True)))
    result["token_result"]["prompt_prefill"]["persistent_rmsnorm_server"] = True

    summary = summarize_result_payload(result)

    assert summary["prompt_prefill"]["persistent_rmsnorm_server"] is True
    text = format_result_summary_text(summary)
    assert "persistent_rmsnorm_server=yes" in text


def test_result_summary_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_sample_result()), encoding="utf-8")

    status = cli_main(["result-summary", str(result), "--top", "1", "--json"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "largerlm.result_summary.v1"
    assert len(payload["elapsed_records"]["top_groups"]) == 1


def test_result_summary_reports_runner_command_records() -> None:
    result = json.loads(json.dumps(_sample_result()))
    layer = result["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0]
    layer["attention"]["attention_output"]["o_proj"]["command"] = [
        "metal/largerlm-runner",
        "--resident-layout",
        "resident/layout.json",
        "--run-resident-linear-batch",
    ]
    layer["attention"]["rope"] = {
        "command": [
            "metal/largerlm-runner",
            "--run-rope-batch",
            "--q-f32",
            "q.f32",
        ]
    }
    layer["attention"]["rope_alias_record"] = {
        "command": [
            "metal/largerlm-runner",
            "--run-rope-batch",
            "--q-f32",
            "q.f32",
        ]
    }
    layer["attention"]["singleton_rope"] = {
        "command": [
            "python-rope-singleton",
            "--q-f32",
            "q.f32",
        ]
    }

    summary = summarize_result_payload(result)

    records = summary["runner_command_records"]
    assert records["record_count"] == 4
    assert records["unique_command_count"] == 3
    assert records["duplicate_record_count"] == 1
    assert records["group_counts"] == {
        "--run-resident-linear-batch": 1,
        "--run-rope-batch": 2,
        "python-rope-singleton": 1,
    }
    assert records["unique_group_counts"] == {
        "--run-resident-linear-batch": 1,
        "--run-rope-batch": 1,
        "python-rope-singleton": 1,
    }
    text = format_result_summary_text(summary)
    assert "runner command records: unique=3 records=4 duplicate_records=1" in text
    assert "--run-resident-linear-batch=1" in text


def test_result_summary_promotes_runner_process_fusion_target_for_many_commands() -> None:
    result = json.loads(json.dumps(_sample_result()))
    layer = result["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0]
    layer["attention"]["runner_burst"] = [
        {
            "command": [
                "metal/largerlm-runner",
                "--run-attn-output-batch",
                "--input-f32",
                f"attn_{index}.f32",
            ]
        }
        for index in range(510)
    ]

    summary = summarize_result_payload(result)

    records = summary["runner_command_records"]
    assert records["unique_command_count"] == 510
    targets = {item["target"]: item for item in summary["optimization_targets"]}
    fusion = targets["runner_process_fusion"]
    assert fusion["kind"] == "runner_process_orchestration"
    assert fusion["evidence"]["unique_command_count"] == 510
    assert fusion["evidence"]["threshold_unique_command_count"] == 500
    assert fusion["evidence"]["top_unique_groups"][0] == {
        "group": "--run-attn-output-batch",
        "unique_count": 510,
        "record_count": 510,
    }
    text = format_result_summary_text(summary)
    assert "runner_process_fusion[runner_process_orchestration]=510cmds" in text


def test_result_summary_cli_rejects_invalid_top(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_sample_result()), encoding="utf-8")

    status = cli_main(["result-summary", str(result), "--top", "0"])

    assert status == 2
    assert "--top must be >= 1" in capsys.readouterr().err


def _scaled_elapsed_result(
    scale: float,
    *,
    prepared_root: str | Path = "/tmp/prepared",
) -> dict[str, object]:
    result = json.loads(
        json.dumps(
            _sample_result(
                include_prompt_elapsed=True,
                prepared_root=prepared_root,
            )
        )
    )
    token = result["token_result"]
    token["elapsed_seconds"] *= scale
    token["steps"][0]["elapsed_seconds"] *= scale
    token["steps"][0]["logits_elapsed_seconds"] *= scale
    prompt = token["prompt_prefill"]
    prompt["elapsed_seconds"] *= scale
    prompt["total_expert_stage_copy_elapsed_seconds"] *= scale
    for key in list(prompt["linear_backend_elapsed_seconds"]):
        prompt["linear_backend_elapsed_seconds"][key] *= scale
    layer = prompt["chunks"][0]["layers"][0]
    layer["attention"]["projections_elapsed_seconds"] *= scale
    layer["attention"]["mla_attention_elapsed_seconds"] *= scale
    layer["attention"]["attention_output"]["o_proj"]["elapsed_seconds"] *= scale
    layer["staged_mlp"]["router_gate_proj"]["elapsed_seconds"] *= scale
    return result


def _http_smoke_wrapper(result: dict[str, object]) -> dict[str, object]:
    token_result = result["token_result"]
    assert isinstance(token_result, dict)
    request = result["request"]
    assert isinstance(request, dict)
    return {
        "schema": "largerlm.http_smoke.v1",
        "endpoint": "/v1/chat/completions",
        "url": "http://127.0.0.1:18081/v1/chat/completions",
        "status": 200,
        "elapsed_seconds": 11.5,
        "error": None,
        "response": {
            "object": "chat.completion",
            "model": "unit-glm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": request["prompt_tokens"],
                "completion_tokens": request["max_new_tokens"],
                "total_tokens": request["prompt_tokens"] + request["max_new_tokens"],
            },
            "largerlm": {
                "chat_template_backend": "local-jinja",
                "tokenizer_backend": "simple",
                "token_result": token_result,
                "request_check": {
                    "ok": True,
                    "prompt_token_count": request["prompt_tokens"],
                    "max_new_tokens": request["max_new_tokens"],
                },
                "launch_audit_envelope": {
                    "schema": "largerlm.launch_audit_server_envelope.v1",
                    "artifact_path": request["launch_audit_path"],
                },
            },
        },
    }


def _raw_openai_chat_response(result: dict[str, object]) -> dict[str, object]:
    wrapper = _http_smoke_wrapper(result)
    response = wrapper["response"]
    assert isinstance(response, dict)
    return response


def _http_generation_smoke_wrapper(result: dict[str, object]) -> dict[str, object]:
    token_result = result["token_result"]
    assert isinstance(token_result, dict)
    request = result["request"]
    assert isinstance(request, dict)
    return {
        "schema": "largerlm.http_generation_smoke.v1",
        "endpoint": "/generate-token-ids",
        "client_elapsed_seconds": 11.5,
        "request": {
            "prompt_token_ids": request["prompt_token_ids"],
            "max_new_tokens": request["max_new_tokens"],
        },
        "response": token_result,
    }


def _http_generate_text_smoke_wrapper(result: dict[str, object]) -> dict[str, object]:
    text_result = result["text_result"]
    assert isinstance(text_result, dict)
    request = result["request"]
    assert isinstance(request, dict)
    return {
        "schema": "largerlm.http_generate_text_smoke.v1",
        "endpoint": "/generate-text",
        "client_elapsed_seconds": 11.5,
        "request": {
            "prompt": text_result["prompt"],
            "max_new_tokens": request["max_new_tokens"],
        },
        "response": {
            **text_result,
            "request_check": {
                "ok": True,
                "prompt_token_count": request["prompt_tokens"],
                "max_new_tokens": request["max_new_tokens"],
            },
            "launch_audit_envelope": {
                "schema": "largerlm.launch_audit_server_envelope.v1",
                "artifact_path": request["launch_audit_path"],
            },
        },
    }


def test_result_summary_unwraps_http_smoke_wrapper(tmp_path: Path) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    wrapper = _http_smoke_wrapper(
        _sample_result(include_prompt_elapsed=True, prepared_root=prepared)
    )

    summary = summarize_result_payload(wrapper, source="http-smoke")

    assert summary["source"] == "http-smoke"
    assert summary["result_wrapper"]["endpoint"] == "/v1/chat/completions"
    assert summary["result_wrapper"]["status"] == 200
    assert summary["result_wrapper"]["client_elapsed_seconds"] == 11.5
    assert summary["result_wrapper"]["chat_template_backend"] == "local-jinja"
    assert summary["prompt_tokens"] == 2
    assert summary["max_new_tokens"] == 1
    assert summary["total_elapsed_seconds"] == 10.0
    assert summary["prefill_plan_signature"]["linear_backend_counts"] == {
        "custom-metal": 2,
        "mpsgraph-f32": 1,
    }
    assert summary["launch_binding"]["prepared_manifest"] == str(
        prepared / "manifest.json"
    )
    assert summary["launch_binding"]["launch_audit_path"] == str(
        prepared / "launch-audit.json"
    )
    assert "wrapper: schema=largerlm.http_smoke.v1" in format_result_summary_text(
        summary
    )


def test_result_summary_unwraps_raw_openai_chat_response(tmp_path: Path) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    cache_dir = prepared / "mla-kv-b-cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "layer-0.bin"
    cache_file.write_bytes(b"cache")
    audit_path = prepared / "launch-audit.json"
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["applied_launch_profile"]["path"] = os.path.relpath(
        prepared / "launch-profile.json"
    )
    audit_path.write_text(json.dumps(audit_payload) + "\n", encoding="utf-8")
    response = _raw_openai_chat_response(
        _sample_result(include_prompt_elapsed=True, prepared_root=prepared)
    )
    largerlm = response["largerlm"]
    assert isinstance(largerlm, dict)
    request_check = largerlm["request_check"]
    assert isinstance(request_check, dict)
    request_check["prefill_mla_kv_b_cache_dir"] = str(cache_dir)
    request_check["prefill_mla_kv_b_cache_file_count"] = 1
    request_check["prefill_mla_kv_b_cache_total_bytes"] = cache_file.stat().st_size

    summary = summarize_result_payload(response, source="raw-openai-chat")

    assert summary["source"] == "raw-openai-chat"
    assert summary["result_wrapper"]["schema"] is None
    assert summary["result_wrapper"]["endpoint"] == "/v1/chat/completions"
    assert summary["result_wrapper"]["status"] is None
    assert summary["result_wrapper"]["client_elapsed_seconds"] is None
    assert summary["result_wrapper"]["model"] == "unit-glm"
    assert summary["result_wrapper"]["chat_template_backend"] == "local-jinja"
    assert summary["prompt_tokens"] == 2
    assert summary["max_new_tokens"] == 1
    assert summary["generated_token_ids"] == [7]
    assert summary["total_elapsed_seconds"] == 10.0
    assert summary["prefill_plan_signature"]["linear_backend_counts"] == {
        "custom-metal": 2,
        "mpsgraph-f32": 1,
    }
    assert summary["launch_binding"]["prepared_manifest"] == str(
        prepared / "manifest.json"
    )
    assert summary["launch_binding"]["launch_audit_path"] == str(
        prepared / "launch-audit.json"
    )
    assert summary["launch_binding"]["launch_audit_profile_path_matches"] is True
    assert summary["launch_binding"]["launch_audit_binding_matches"] is True
    assert summary["launch_binding"]["replay_mla_kv_b_cache_dir"] == str(cache_dir)
    assert summary["launch_binding"]["replay_mla_kv_b_cache_ready"] is True
    assert summary["launch_binding"]["replay_files_ready"] is True
    assert "--prefill-mla-kv-b-cache-dir" in summary["launch_binding"][
        "replay_generate_token_ids_argv"
    ]
    assert "wrapper: schema=None endpoint=/v1/chat/completions" in (
        format_result_summary_text(summary)
    )


def test_result_summary_unwraps_direct_http_generation_wrapper(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    wrapper = _http_generation_smoke_wrapper(
        _sample_result(include_prompt_elapsed=True, prepared_root=prepared)
    )

    summary = summarize_result_payload(wrapper, source="http-generation-smoke")

    assert summary["result_wrapper"]["schema"] == "largerlm.http_generation_smoke.v1"
    assert summary["result_wrapper"]["endpoint"] == "/generate-token-ids"
    assert summary["result_wrapper"]["client_elapsed_seconds"] == 11.5
    assert summary["prompt_tokens"] == 2
    assert summary["max_new_tokens"] == 1
    assert summary["generated_token_ids"] == [7]
    assert summary["total_elapsed_seconds"] == 10.0


def test_result_summary_unwraps_generate_text_http_wrapper(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    wrapper = _http_generate_text_smoke_wrapper(
        _sample_text_result(include_prompt_elapsed=True, prepared_root=prepared)
    )

    summary = summarize_result_payload(wrapper, source="http-generate-text-smoke")

    assert summary["result_wrapper"]["schema"] == (
        "largerlm.http_generate_text_smoke.v1"
    )
    assert summary["result_wrapper"]["endpoint"] == "/generate-text"
    assert summary["result_wrapper"]["client_elapsed_seconds"] == 11.5
    assert summary["prompt_tokens"] == 2
    assert summary["max_new_tokens"] == 1
    assert summary["generated_token_ids"] == [7]
    assert summary["total_elapsed_seconds"] == 10.0
    assert summary["launch_binding"]["launch_audit_binding_matches"] is True
    assert summary["launch_binding"]["safe_to_replay"] is True


def test_result_summary_binds_raw_generation_server_response(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    result = _sample_result(include_prompt_elapsed=True, prepared_root=prepared)
    request = result["request"]
    assert isinstance(request, dict)
    token_result = dict(result["token_result"])
    applied = token_result["applied_launch_profile"]
    assert isinstance(applied, dict)
    launch_envelope = {
        "schema": "largerlm.launch_audit_server_envelope.v1",
        "artifact_path": request["launch_audit_path"],
        "applied_launch_profile_sha256": applied["sha256"],
    }
    token_result["launch_audit_envelope"] = launch_envelope
    token_result["request_check"] = {
        "ok": True,
        "prompt_token_count": request["prompt_tokens"],
        "max_new_tokens": request["max_new_tokens"],
        "launch_audit_envelope": launch_envelope,
    }

    summary = summarize_result_payload(token_result, source="raw-generation-server")

    assert summary["source"] == "raw-generation-server"
    assert summary["result_wrapper"] is None
    assert summary["prompt_tokens"] == 2
    assert summary["max_new_tokens"] == 1
    assert summary["generated_token_ids"] == [7]
    assert summary["launch_binding"]["prepared_manifest"] == str(
        prepared / "manifest.json"
    )
    assert summary["launch_binding"]["launch_profile_path"] == str(
        prepared / "launch-profile.json"
    )
    assert summary["launch_binding"]["launch_audit_path"] == str(
        prepared / "launch-audit.json"
    )
    assert summary["launch_binding"]["launch_audit_binding_matches"] is True
    assert summary["launch_binding"]["safe_to_replay"] is True
    assert summary["launch_binding"]["replay_ready"] is True
    assert summary["launch_binding"]["replay_files_ready"] is True
    assert summary["launch_binding"]["replay_prompt_token_count"] == 2
    assert summary["launch_binding"]["replay_max_new_tokens"] == 1


def test_result_compare_cli_accepts_http_smoke_wrapper_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate-http.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            _http_smoke_wrapper(_scaled_elapsed_result(2.5, prepared_root=prepared))
        ),
        encoding="utf-8",
    )

    candidate_summary = summarize_result_file(candidate)
    assert candidate_summary["result_wrapper"]["endpoint"] == "/v1/chat/completions"

    status = cli_main(
        ["result-compare", str(baseline), str(candidate), "--top", "2", "--json"]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workload"]["comparable"] is True
    assert payload["possible_system_slowdown"]["detected"] is True


def test_result_compare_detects_global_system_slowdown() -> None:
    baseline = summarize_result_payload(_scaled_elapsed_result(1.0), source="base")
    candidate = summarize_result_payload(_scaled_elapsed_result(3.0), source="slow")

    comparison = compare_result_summaries(baseline, candidate)

    assert comparison["schema"] == "largerlm.result_comparison.v1"
    assert comparison["workload"]["comparable"] is True
    assert comparison["total"]["ratio"] == pytest.approx(3.0)
    slowdown = comparison["possible_system_slowdown"]
    assert slowdown["detected"] is True
    assert slowdown["median_sentinel_ratio"] == pytest.approx(3.0)
    recommendation = comparison["profile_recommendation"]
    assert recommendation["decision"] == "inconclusive"
    assert recommendation["candidate_promotable"] is False
    assert "possible_system_slowdown" in recommendation["reasons"]


def test_result_compare_rejects_prefill_plan_mismatch() -> None:
    baseline_payload = _sample_result(include_prompt_elapsed=True)
    candidate_payload = json.loads(json.dumps(baseline_payload))
    candidate_prompt = candidate_payload["token_result"]["prompt_prefill"]
    candidate_prompt["chunk_count"] += 1
    baseline = summarize_result_payload(baseline_payload, source="base")
    candidate = summarize_result_payload(candidate_payload, source="different-plan")

    comparison = compare_result_summaries(baseline, candidate)

    assert comparison["workload"]["generated_token_ids_match"] is True
    assert comparison["workload"]["prompt_tokens_match"] is True
    assert comparison["workload"]["max_new_tokens_match"] is True
    assert comparison["workload"]["prefill_plan_match"] is False
    assert comparison["workload"]["comparable"] is False
    assert comparison["profile_recommendation"]["candidate_promotable"] is False
    assert "workload_not_comparable" in comparison["profile_recommendation"]["reasons"]
    text = format_result_comparison_text(comparison)
    assert "prefill plan: baseline chunks=1x2" in text
    assert "candidate chunks=2x2" in text
    assert "mla_value_cache=1/1" in text


def test_result_compare_allows_explicit_prefill_policy_change() -> None:
    baseline_payload = _scaled_elapsed_result(1.0)
    candidate_payload = _scaled_elapsed_result(0.7)
    candidate_prompt = candidate_payload["token_result"]["prompt_prefill"]
    candidate_prompt["linear_backend_counts"] = {"custom-metal": 3}
    candidate_prompt["linear_backend_flops"] = {"custom-metal": 250}
    candidate_prompt["max_stage_plus_compact_bytes"] = 512
    baseline = summarize_result_payload(baseline_payload, source="base")
    candidate = summarize_result_payload(candidate_payload, source="policy-candidate")

    strict = compare_result_summaries(baseline, candidate)
    assert strict["workload"]["prefill_plan_match"] is False
    assert strict["workload"]["comparable"] is False
    assert strict["profile_recommendation"]["candidate_promotable"] is False

    comparison = compare_result_summaries(
        baseline,
        candidate,
        allow_prefill_policy_change=True,
    )

    assert comparison["workload"]["prefill_plan_match"] is False
    assert comparison["workload"]["request_shape_match"] is True
    assert comparison["workload"]["generated_token_ids_match"] is True
    assert comparison["workload"]["prefill_policy_change_allowed"] is True
    assert comparison["workload"]["comparison_mode"] == "prefill_policy_experiment"
    assert comparison["workload"]["comparable"] is True
    recommendation = comparison["profile_recommendation"]
    assert recommendation["candidate_promotable"] is True
    assert recommendation["decision"] == "prefer_candidate"
    assert recommendation["total_ratio"] == pytest.approx(0.7)
    text = format_result_comparison_text(comparison)
    assert "mode=prefill_policy_experiment" in text


def test_result_compare_prints_mla_cache_plan_mismatch() -> None:
    baseline_payload = _sample_result(include_prompt_elapsed=True)
    candidate_payload = json.loads(json.dumps(baseline_payload))
    attention = candidate_payload["token_result"]["prompt_prefill"]["chunks"][0][
        "layers"
    ][0]["attention"]
    attention["mla_value_cache"] = False
    attention["mla_value_cache_bytes"] = 0
    baseline = summarize_result_payload(baseline_payload, source="base")
    candidate = summarize_result_payload(candidate_payload, source="no-value-cache")

    comparison = compare_result_summaries(baseline, candidate)

    assert comparison["workload"]["prefill_plan_match"] is False
    assert comparison["workload"]["comparable"] is False
    text = format_result_comparison_text(comparison)
    assert "baseline chunks=1x2" in text
    assert "candidate chunks=1x2" in text
    assert "mla_key_cache=1/1" in text
    assert "mla_value_cache=1/1" in text
    assert "mla_value_cache=0/1" in text


def test_result_compare_keeps_targeted_mla_improvement_distinct() -> None:
    baseline_payload = _sample_result(include_prompt_elapsed=True)
    candidate_payload = json.loads(json.dumps(baseline_payload))
    candidate_payload["token_result"]["elapsed_seconds"] = 9.25
    candidate_payload["token_result"]["prompt_prefill"]["elapsed_seconds"] = 7.25
    candidate_payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]["mla_attention_elapsed_seconds"] = 0.5
    baseline = summarize_result_payload(baseline_payload, source="base")
    candidate = summarize_result_payload(candidate_payload, source="fast")

    comparison = compare_result_summaries(baseline, candidate)

    assert comparison["possible_system_slowdown"]["detected"] is False
    top_names = [item["name"] for item in comparison["changes"]["top"]]
    assert "named_elapsed_fields.mla_attention_elapsed_seconds" in top_names
    mla = next(
        item
        for item in comparison["changes"]["top"]
        if item["name"] == "named_elapsed_fields.mla_attention_elapsed_seconds"
    )
    assert mla["ratio"] == pytest.approx(0.4)
    recommendation = comparison["profile_recommendation"]
    assert recommendation["decision"] == "prefer_candidate"
    assert recommendation["candidate_promotable"] is True
    assert recommendation["reasons"] == ["candidate_total_elapsed_faster"]
    assert "profile recommendation: prefer_candidate" in format_result_comparison_text(
        comparison
    )


def test_result_compare_allows_large_total_win_despite_tensor_regressions() -> None:
    baseline_payload = _sample_result(include_prompt_elapsed=True)
    candidate_payload = json.loads(json.dumps(baseline_payload))
    candidate_payload["token_result"]["elapsed_seconds"] = 7.0
    candidate_payload["token_result"]["prompt_prefill"]["elapsed_seconds"] = 5.0
    candidate_payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]["attention_output"]["o_proj"]["elapsed_seconds"] = 1.4
    baseline = summarize_result_payload(baseline_payload, source="base")
    candidate = summarize_result_payload(candidate_payload, source="candidate")

    comparison = compare_result_summaries(baseline, candidate)

    assert comparison["possible_system_slowdown"]["detected"] is False
    recommendation = comparison["profile_recommendation"]
    assert recommendation["decision"] == "prefer_candidate"
    assert recommendation["candidate_promotable"] is True
    assert recommendation["total_ratio"] == pytest.approx(0.7)
    assert recommendation["large_tensor_regression_total_seconds"] == pytest.approx(
        0.7
    )
    assert recommendation["reasons"] == [
        "candidate_large_total_win_overrides_tensor_regressions"
    ]


def test_result_compare_reports_tensor_suffix_changes() -> None:
    baseline_payload = _sample_result(include_prompt_elapsed=True)
    candidate_payload = json.loads(json.dumps(baseline_payload))
    candidate_payload["token_result"]["prompt_prefill"]["chunks"][0]["layers"][0][
        "attention"
    ]["attention_output"]["o_proj"]["elapsed_seconds"] = 1.4
    baseline = summarize_result_payload(baseline_payload, source="base")
    candidate = summarize_result_payload(candidate_payload, source="candidate")

    comparison = compare_result_summaries(baseline, candidate)

    row = next(
        item
        for item in comparison["changes"]["top"]
        if item["name"]
        == "tensor_suffix_elapsed_seconds.self_attn.o_proj.weight"
    )
    assert row["kind"] == "tensor_suffix_elapsed"
    assert row["baseline_seconds"] == pytest.approx(0.7)
    assert row["candidate_seconds"] == pytest.approx(1.4)
    assert row["delta_seconds"] == pytest.approx(0.7)
    assert row["ratio"] == pytest.approx(2.0)
    recommendation = comparison["profile_recommendation"]
    assert recommendation["decision"] == "tie"
    assert recommendation["candidate_promotable"] is False
    assert recommendation["top_tensor_regressions"][0]["name"] == (
        "tensor_suffix_elapsed_seconds.self_attn.o_proj.weight"
    )
    assert recommendation["large_tensor_regressions"][0]["ratio"] == pytest.approx(
        2.0
    )
    assert (
        "tensor_suffix_elapsed_seconds.self_attn.o_proj.weight"
        in format_result_comparison_text(comparison)
    )


def test_result_compare_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_scaled_elapsed_result(2.5)), encoding="utf-8")

    status = cli_main(
        ["result-compare", str(baseline), str(candidate), "--top", "2", "--json"]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "largerlm.result_comparison.v1"
    assert payload["possible_system_slowdown"]["detected"] is True
    assert payload["profile_recommendation"]["candidate_promotable"] is False
    assert len(payload["changes"]["top"]) == 2


def test_result_compare_cli_require_candidate_promotable_rejects_slow_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_scaled_elapsed_result(1.25)), encoding="utf-8")

    status = cli_main(
        [
            "result-compare",
            str(baseline),
            str(candidate),
            "--require-candidate-promotable",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "profile recommendation: prefer_baseline" in captured.out
    assert "candidate result is not promotable" in captured.err


def test_result_compare_cli_require_candidate_promotable_accepts_faster_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_scaled_elapsed_result(0.75)), encoding="utf-8")

    status = cli_main(
        [
            "result-compare",
            str(baseline),
            str(candidate),
            "--require-candidate-promotable",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "profile recommendation: prefer_candidate" in captured.out
    assert captured.err == ""


def test_result_bakeoff_selects_fastest_promotable_candidate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    fast = tmp_path / "fast.json"
    faster = tmp_path / "faster.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    slow.write_text(json.dumps(_scaled_elapsed_result(1.1)), encoding="utf-8")
    fast.write_text(json.dumps(_scaled_elapsed_result(0.8)), encoding="utf-8")
    faster.write_text(json.dumps(_scaled_elapsed_result(0.7)), encoding="utf-8")

    bakeoff = result_bakeoff_files(baseline, [slow, fast, faster], top_limit=2)

    assert bakeoff["schema"] == "largerlm.result_bakeoff.v1"
    assert bakeoff["baseline_retained"] is False
    assert bakeoff["selected"]["role"] == "candidate"
    assert bakeoff["selected"]["path"] == str(faster)
    assert bakeoff["selected"]["launch_binding"]["replay_ready"] is True
    assert bakeoff["winner"]["path"] == str(faster)
    assert bakeoff["winner"]["total_ratio"] == pytest.approx(0.7)
    assert bakeoff["winner"]["launch_binding"]["launch_profile_path"] == (
        "/tmp/prepared/launch-profile.json"
    )
    assert bakeoff["winner"]["launch_binding"]["launch_audit_path"] == (
        "/tmp/prepared/launch-audit.json"
    )
    assert bakeoff["winner"]["launch_binding"]["safe_to_replay"] is True
    assert bakeoff["winner"]["launch_binding"]["replay_ready"] is True
    assert bakeoff["winner"]["launch_binding"]["replay_generate_token_ids_argv"][-4:] == [
        "--prompt-token-ids",
        "1,2",
        "--max-new-tokens",
        "1",
    ]
    assert bakeoff["winner"]["required_environment"] == {
        "LARGERLM_MOE_BATCH_ACCUMULATOR": "memory"
    }
    assert bakeoff["baseline_launch_binding"]["safe_to_replay"] is True
    assert [item["candidate_promotable"] for item in bakeoff["candidates"]] == [
        False,
        True,
        True,
    ]
    text = format_result_bakeoff_text(bakeoff)
    assert "selected: role=candidate" in text
    assert "selected launch:" in text
    assert "selected env: LARGERLM_MOE_BATCH_ACCUMULATOR=memory" in text
    assert "winner:" in text
    assert "winner env: LARGERLM_MOE_BATCH_ACCUMULATOR=memory" in text
    assert "winner launch:" in text
    assert "safe_to_replay=True" in text
    assert "replay_ready=True" in text
    assert str(faster) in text


def test_result_bakeoff_allows_explicit_prefill_policy_change(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "policy-candidate.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    candidate_payload = _scaled_elapsed_result(0.7)
    candidate_prompt = candidate_payload["token_result"]["prompt_prefill"]
    candidate_prompt["linear_backend_counts"] = {"custom-metal": 3}
    candidate_prompt["linear_backend_flops"] = {"custom-metal": 250}
    candidate_prompt["max_stage_plus_compact_bytes"] = 512
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")

    strict = result_bakeoff_files(baseline, [candidate], top_limit=2)
    assert strict["baseline_retained"] is True
    assert strict["candidates"][0]["candidate_promotable"] is False
    assert "workload_not_comparable" in strict["candidates"][0]["reasons"]

    bakeoff = result_bakeoff_files(
        baseline,
        [candidate],
        top_limit=2,
        allow_prefill_policy_change=True,
    )

    assert bakeoff["allow_prefill_policy_change"] is True
    assert bakeoff["baseline_retained"] is False
    assert bakeoff["selected"]["role"] == "candidate"
    assert bakeoff["winner"]["path"] == str(candidate)
    assert bakeoff["candidates"][0]["candidate_promotable"] is True
    workload = bakeoff["candidates"][0]["comparison"]["workload"]
    assert workload["comparison_mode"] == "prefill_policy_experiment"
    assert workload["comparable"] is True
    assert "prefill policy changes allowed: True" in format_result_bakeoff_text(
        bakeoff
    )


def test_result_bakeoff_promote_only_replay_files_ready_retains_safe_baseline(
    tmp_path: Path,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    fast_without_files = tmp_path / "fast-without-files.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    fast_without_files.write_text(
        json.dumps(_scaled_elapsed_result(0.7)),
        encoding="utf-8",
    )

    bakeoff = result_bakeoff_files(
        baseline,
        [fast_without_files],
        promote_only_replay_files_ready=True,
    )

    assert bakeoff["promote_only_replay_files_ready"] is True
    assert bakeoff["baseline_retained"] is True
    assert bakeoff["winner"] is None
    assert bakeoff["selected"]["role"] == "baseline"
    assert bakeoff["selected"]["launch_binding"]["replay_files_ready"] is True
    candidate = bakeoff["candidates"][0]
    assert candidate["performance_promotable"] is True
    assert candidate["candidate_promotable"] is False
    assert candidate["replay_ready"] is True
    assert candidate["replay_files_ready"] is False
    assert "candidate_replay_files_not_ready" in candidate["reasons"]
    text = format_result_bakeoff_text(bakeoff)
    assert "promotion requires replay files ready: True" in text
    assert "performance_promotable=True" in text
    assert "replay_files_ready=False" in text


def test_result_bakeoff_cli_require_winner_rejects_all_slow_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    slower = tmp_path / "slower.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    slow.write_text(json.dumps(_scaled_elapsed_result(1.1)), encoding="utf-8")
    slower.write_text(json.dumps(_scaled_elapsed_result(1.2)), encoding="utf-8")

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            str(slower),
            "--require-winner",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "selected: role=baseline" in captured.out
    assert "winner: none" in captured.out
    assert "baseline retained: True" in captured.out
    assert "baseline launch:" in captured.out
    assert "replay_ready=True" in captured.out
    assert "no promotable candidate result found" in captured.err


def test_result_bakeoff_cli_writes_full_bakeoff_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    fast = tmp_path / "fast.json"
    report = tmp_path / "bakeoff.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    fast.write_text(json.dumps(_scaled_elapsed_result(0.7)), encoding="utf-8")

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(fast),
            "--write-bakeoff-json",
            str(report),
            "--json",
        ]
    )

    assert status == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(report.read_text(encoding="utf-8"))
    assert file_payload["schema"] == "largerlm.result_bakeoff.v1"
    assert file_payload["winner"]["path"] == str(fast)
    assert stdout_payload["winner"]["path"] == file_payload["winner"]["path"]


def test_result_bakeoff_cli_require_selected_replay_ready_accepts_retained_baseline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--require-selected-replay-ready",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "selected: role=baseline" in captured.out
    assert "selected launch:" in captured.out
    assert captured.err == ""


def test_result_bakeoff_cli_writes_selected_replay_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    replay_script = tmp_path / "selected-replay.sh"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--write-selected-replay-json",
            str(replay_json),
            "--write-selected-replay-script",
            str(replay_script),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "selected: role=baseline" in captured.out
    assert captured.err == ""
    payload = json.loads(replay_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "largerlm.selected_replay.v1"
    assert payload["selected_role"] == "baseline"
    assert payload["selected_result"] == str(baseline)
    assert payload["launch_binding"]["replay_ready"] is True
    assert payload["launch_binding"]["replay_files_ready"] is True
    assert payload["launch_binding"]["launch_profile_sha256_matches_file"] is True
    assert payload["launch_binding"]["launch_audit_binding_matches"] is True
    assert payload["required_environment"] == {
        "LARGERLM_MOE_BATCH_ACCUMULATOR": "memory"
    }
    assert payload["selected_total_elapsed_seconds"] == pytest.approx(10.0)
    signature = payload["selected_prefill_plan_signature"]
    assert signature["present"] is True
    assert signature["chunk_count"] == 1
    assert signature["chunk_tokens"] == 2
    assert signature["linear_backend_counts"] == {
        "custom-metal": 2,
        "mpsgraph-f32": 1,
    }
    ssd_policy = payload["pre_run_checks"]["ssd_read_speed"]
    assert ssd_policy["enabled"] is False
    assert ssd_policy["bytes_mib"] == pytest.approx(1024 / 1024**2)
    assert ssd_policy["auto_enabled_min_bytes"] == 64 * 1024**2
    assert payload["argv"][-4:] == [
        "--prompt-token-ids",
        "1,2",
        "--max-new-tokens",
        "1",
    ]
    script = replay_script.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "selected_role: baseline" in script
    assert "exec python -m largerlm selected-replay-run" in script
    assert "--quiet-runner" in script
    assert "--check-ssd-read-speed" in script
    assert "export LARGERLM_MOE_BATCH_ACCUMULATOR=memory" in script
    assert str(replay_json) in script


def test_result_bakeoff_omits_accumulator_env_when_launch_profile_pins_memory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(
        tmp_path,
        prefill_moe_output_accumulator="memory",
    )
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    replay_script = tmp_path / "selected-replay.sh"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--write-selected-replay-json",
            str(replay_json),
            "--write-selected-replay-script",
            str(replay_script),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "selected env:" not in captured.out
    payload = json.loads(replay_json.read_text(encoding="utf-8"))
    assert payload["required_environment"] == {}
    profile_path = Path(payload["launch_binding"]["launch_profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert "--prefill-moe-output-accumulator" in profile["argv"]
    script = replay_script.read_text(encoding="utf-8")
    assert "export LARGERLM_MOE_BATCH_ACCUMULATOR=memory" not in script

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 0
    check_payload = json.loads(capsys.readouterr().out)
    assert check_payload["required_environment"] == {}


def test_result_bakeoff_cli_writes_cwd_independent_relative_replay_script(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    baseline = Path("baseline.json")
    slow = Path("slow.json")
    replay_json = Path("out/selected-replay.json")
    replay_script = Path("out/selected-replay.sh")
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--write-selected-replay-json",
            str(replay_json),
            "--write-selected-replay-script",
            str(replay_script),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "selected: role=baseline" in captured.out
    assert captured.err == ""
    script = replay_script.read_text(encoding="utf-8")
    assert (
        'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"'
        in script
    )
    assert (
        'exec python -m largerlm selected-replay-run "$SCRIPT_DIR"/selected-replay.json --quiet-runner --check-ssd-read-speed'
        in script
    )

    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo_root)
        if not env.get("PYTHONPATH")
        else str(repo_root) + os.pathsep + env["PYTHONPATH"]
    )
    completed = subprocess.run(
        [
            str((tmp_path / replay_script).resolve()),
            "--dry-run",
            "--ssd-read-speed-min-ratio",
            "0.000001",
            "--ssd-read-speed-bytes-mib",
            "0.001",
            "--ssd-read-speed-chunk-mib",
            "0.001",
            "--write-result",
            "out/replayed-result.json",
        ],
        cwd=other_cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "selected replay ok: True" in completed.stdout
    assert "failed checks: 0" in completed.stdout
    assert "command: python -m largerlm generate-prepared-token-ids" in completed.stdout
    assert "--quiet-runner" in completed.stdout
    assert "--write-result out/replayed-result.json" in completed.stdout
    assert completed.stderr == ""


def test_result_bakeoff_cli_writes_standalone_checked_replay_script(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_script = tmp_path / "selected-replay.sh"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--write-selected-replay-script",
            str(replay_script),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "selected: role=baseline" in captured.out
    assert captured.err == ""
    script = replay_script.read_text(encoding="utf-8")
    assert "LARGERLM_SELECTED_REPLAY_JSON" in script
    assert "selected-replay-run" in script
    assert "--quiet-runner" in script
    assert "--check-ssd-read-speed" in script
    assert "generate-prepared-token-ids" in script


def test_selected_replay_check_cli_accepts_current_replay_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json)])

    assert status == 0
    captured = capsys.readouterr()
    assert "selected replay ok: True" in captured.out
    assert "selected prefill plan: chunks=1x2" in captured.out
    assert "backends=custom-metal=2,mpsgraph-f32=1" in captured.out
    assert "selected elapsed: 10.000s" in captured.out
    assert "files_ready=True" in captured.out
    assert "audit_bound=True" in captured.out
    assert "failed checks: 0" in captured.out
    assert captured.err == ""


def test_selected_replay_check_cli_accepts_nonaccelerated_custom_metal_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    profile_path = prepared / "launch-profile.json"
    profile_path.write_text(
        json.dumps({"argv": ["--prefill-linear-backend", "custom-metal"]}) + "\n",
        encoding="utf-8",
    )
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    coverage = _replay_prefill_acceleration_coverage()
    coverage.update(
        {
            "required": False,
            "accelerated_matrix_count": 0,
            "mpsgraph_matrix_count": 0,
            "custom_metal_matrix_count": 1,
            "accelerated_estimated_flops": 0,
            "custom_metal_estimated_flops": 1024,
            "accelerated_flop_fraction": 0.0,
            "dominant_resident_flops_accelerated": False,
            "accelerated_backends": [],
            "any_resident_matrix_accelerated": False,
            "all_resident_matrices_accelerated": False,
            "reason": "non-accelerated custom-metal experiment",
        }
    )
    audit_path = prepared / "launch-audit.json"
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["applied_launch_profile"]["sha256"] = profile_sha
    audit_payload["request_check"]["prefill_acceleration_coverage"] = coverage
    for check in audit_payload["launch_audit"]["checks"]:
        if check.get("code") == "prefill_acceleration_required":
            check.update({"ok": True, "required": False})
        elif check.get("code") == "prefill_acceleration_gate_ok":
            check.update(
                {
                    "ok": True,
                    "required": False,
                    "prefill_acceleration_runtimes": ["mpsgraph-f32"],
                    "selectable_accelerated_prefill_backends": ["mpsgraph-f32"],
                    "validated_prefill_acceleration_available": False,
                }
            )
        elif check.get("code") == "prefill_acceleration_probe_ok":
            check.update(
                {
                    "ok": True,
                    "mps_graph_probe_requested": False,
                    "mps_graph_probe_ran": False,
                    "mps_graph_probe_ok": None,
                    "validated_accelerated_prefill_backends": [],
                }
            )
        elif check.get("code") == "prefill_acceleration_profile_replays_probe":
            check.update(
                {
                    "ok": True,
                    "required": False,
                    "runtime_probe_required": False,
                    "runtime_probe_satisfied": False,
                    "has_run_mpsgraph_probe": False,
                }
            )
        elif check.get("code") == "request_prefill_acceleration_coverage_ok":
            check.clear()
            check.update(
                {
                    "code": "request_prefill_acceleration_coverage_ok",
                    "evidence_present": True,
                    "ok": True,
                    **coverage,
                }
            )
    audit_path.write_text(json.dumps(audit_payload) + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    checks = {check["code"]: check for check in payload["checks"]}
    assert checks["launch_audit_prefill_acceleration_required"]["ok"] is True
    assert checks["launch_audit_prefill_acceleration_required"]["required"] is False
    assert checks["launch_profile_requires_prefill_acceleration"]["ok"] is True
    assert checks["launch_profile_replays_mpsgraph_probe"]["ok"] is True
    assert checks["launch_profile_mpsgraph_thresholds_present"]["ok"] is True
    assert captured.err == ""


def test_selected_replay_check_cli_honors_artifact_ssd_read_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    fake_benchmark = _test_sequential_read_benchmark(3.5)
    monkeypatch.setattr("largerlm.cli.benchmark_sequential_read", fake_benchmark)
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    payload = json.loads(replay_json.read_text(encoding="utf-8"))
    payload["pre_run_checks"]["ssd_read_speed"].update(
        {"enabled": True, "bytes_mib": 1.0, "chunk_mib": 1.0}
    )
    replay_json.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 0
    captured = capsys.readouterr()
    checked = json.loads(captured.out)
    check = next(
        item
        for item in checked["checks"]
        if item["code"] == "current_ssd_read_speed_ok"
    )
    assert check["ok"] is True
    assert check["actual_gib_per_second"] == 3.5
    assert check["required_gib_per_second"] == 3.0
    assert check["requested_bytes"] == 1024**2
    assert fake_benchmark.calls[0]["path"] == prepared / "benchmark.bin"  # type: ignore[attr-defined]
    assert captured.err == ""


def test_selected_replay_check_cli_can_verify_current_ssd_read_speed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    fake_benchmark = _test_sequential_read_benchmark(3.5)
    monkeypatch.setattr("largerlm.cli.benchmark_sequential_read", fake_benchmark)
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-check",
            str(replay_json),
            "--check-ssd-read-speed",
            "--ssd-read-speed-min-ratio",
            "0.75",
            "--ssd-read-speed-bytes-mib",
            "1",
            "--ssd-read-speed-chunk-mib",
            "1",
            "--json",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    check = next(
        item
        for item in payload["checks"]
        if item["code"] == "current_ssd_read_speed_ok"
    )
    assert check["ok"] is True
    assert check["baseline_gib_per_second"] == 4.0
    assert check["actual_gib_per_second"] == 3.5
    assert check["required_gib_per_second"] == 3.0
    assert check["requested_bytes"] == 1024**2
    assert check["chunk_bytes"] == 1024**2
    assert fake_benchmark.calls[0]["path"] == prepared / "benchmark.bin"  # type: ignore[attr-defined]
    assert captured.err == ""


def test_selected_replay_run_cli_dry_run_rejects_slow_current_ssd_read_speed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli.benchmark_sequential_read",
        _test_sequential_read_benchmark(2.0),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-run",
            str(replay_json),
            "--dry-run",
            "--check-ssd-read-speed",
            "--ssd-read-speed-min-ratio",
            "0.75",
            "--ssd-read-speed-bytes-mib",
            "1",
            "--ssd-read-speed-chunk-mib",
            "1",
            "--json",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    check = next(
        item
        for item in payload["checks"]
        if item["code"] == "current_ssd_read_speed_ok"
    )
    assert payload["ok"] is False
    assert check["ok"] is False
    assert check["actual_gib_per_second"] == 2.0
    assert check["required_gib_per_second"] == 3.0
    assert "run_argv" not in payload
    assert captured.err == ""


def test_selected_replay_run_cli_text_reports_slow_current_ssd_read_speed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli.benchmark_sequential_read",
        _test_sequential_read_benchmark(2.0),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-run",
            str(replay_json),
            "--dry-run",
            "--check-ssd-read-speed",
            "--ssd-read-speed-min-ratio",
            "0.75",
            "--ssd-read-speed-bytes-mib",
            "1",
            "--ssd-read-speed-chunk-mib",
            "1",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "current_ssd_read_speed_ok" in captured.out
    assert "actual=2.000GiB/s" in captured.out
    assert "required=3.000GiB/s" in captured.out
    assert "baseline=4.000GiB/s" in captured.out
    assert "ratio=0.500x" in captured.out
    assert "requested=1.00 MiB (1048576 bytes)" in captured.out
    assert "chunk=1.00 MiB (1048576 bytes)" in captured.out
    assert "command: python -m largerlm generate-prepared-token-ids" not in captured.out
    assert captured.err == ""


def test_selected_replay_run_cli_rejects_concurrent_prepared_replay_before_exec(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()
    exec_called = False

    def fake_execvp(_file: str, _argv: list[str]) -> None:
        nonlocal exec_called
        exec_called = True
        raise AssertionError("selected replay should not exec while lock is held")

    monkeypatch.setattr("largerlm.cli.os.execvp", fake_execvp)
    lock_path = prepared / ".largerlm-selected-replay.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        status = cli_main(["selected-replay-run", str(replay_json)])
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "another selected replay is already running" in captured.err
    assert str(lock_path) in captured.err
    assert exec_called is False


def test_selected_replay_run_cli_exec_sets_inherited_lock_env(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()
    captured_exec: dict[str, object] = {}

    class ExecIntercept(RuntimeError):
        pass

    def fake_execvp(file_arg: str, argv: list[str]) -> None:
        captured_exec["file"] = file_arg
        captured_exec["argv"] = list(argv)
        captured_exec["lock_env"] = os.environ.get(PREPARED_RUN_LOCK_ENV)
        captured_exec["accumulator_env"] = os.environ.get(
            "LARGERLM_MOE_BATCH_ACCUMULATOR"
        )
        raise ExecIntercept

    monkeypatch.setattr("largerlm.cli.os.execvp", fake_execvp)

    try:
        with pytest.raises(ExecIntercept):
            cli_main(["selected-replay-run", str(replay_json)])
    finally:
        os.environ.pop(PREPARED_RUN_LOCK_ENV, None)
        os.environ.pop("LARGERLM_MOE_BATCH_ACCUMULATOR", None)

    assert captured_exec["file"] == "python"
    assert "generate-prepared-token-ids" in captured_exec["argv"]
    assert captured_exec["lock_env"] == str(
        (prepared / ".largerlm-selected-replay.lock").resolve()
    )
    assert captured_exec["accumulator_env"] == "memory"
    assert capsys.readouterr().err == ""


def test_selected_replay_run_cli_dry_run_accepts_current_replay_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-run", str(replay_json), "--dry-run"])

    assert status == 0
    captured = capsys.readouterr()
    assert "selected replay ok: True" in captured.out
    assert "failed checks: 0" in captured.out
    assert "command: python -m largerlm generate-prepared-token-ids" in captured.out
    assert "--require-locked-launch-profile" in captured.out
    assert captured.err == ""


def test_selected_replay_run_cli_dry_run_can_append_write_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    output = tmp_path / "selected-replay-result.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-run",
            str(replay_json),
            "--dry-run",
            "--write-result",
            str(output),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "selected replay ok: True" in captured.out
    assert "failed checks: 0" in captured.out
    assert f"--write-result {output}" in captured.out
    assert not output.exists()
    assert captured.err == ""


def test_selected_replay_run_cli_json_dry_run_reports_appended_write_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    output = tmp_path / "selected-replay-result.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-run",
            str(replay_json),
            "--dry-run",
            "--write-result",
            str(output),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["quiet_runner"] is True
    assert payload["write_result_path"] == str(output)
    assert payload["run_argv"][-3:] == [
        "--quiet-runner",
        "--write-result",
        str(output),
    ]
    assert "--quiet-runner" in payload["run_command"]
    assert f"--write-result {output}" in payload["run_command"]
    assert "--write-result" not in payload["argv"]
    assert "--quiet-runner" not in payload["argv"]
    assert not output.exists()
    assert captured.err == ""


def test_selected_replay_run_cli_dry_run_replays_audit_request_profile_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    audit_path = prepared / "launch-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["request_launch_profile"] = {
        "argv": ["--prefill-linear-backend", "mpsgraph-f32"]
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-run",
            str(replay_json),
            "--dry-run",
            "--json",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["launch_audit_request_profile_argv"] == [
        "--prefill-linear-backend",
        "mpsgraph-f32",
    ]
    assert "--prefill-linear-backend" not in payload["argv"]
    run_argv = payload["run_argv"]
    assert run_argv[run_argv.index("--prefill-linear-backend") + 1] == "mpsgraph-f32"
    assert captured.err == ""


def test_selected_replay_run_cli_dry_run_keeps_auto_backend_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    audit_path = prepared / "launch-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["request_check"]["prefill_linear_backend"] = {
        "analyzed": True,
        "configured": "auto",
        "effective": "auto",
    }
    audit["request_launch_profile"] = {
        "argv": ["--prefill-linear-backend", "mpsgraph-f32"]
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    capsys.readouterr()

    status = cli_main(
        [
            "selected-replay-run",
            str(replay_json),
            "--dry-run",
            "--json",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert "launch_audit_request_profile_argv" not in payload
    assert "--prefill-linear-backend" not in payload["run_argv"]
    assert captured.err == ""


def test_selected_replay_check_cli_rejects_current_low_memory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failed = {check["code"]: check for check in payload["checks"] if not check["ok"]}
    assert "current_available_memory_meets_audit_requirement" in failed
    assert failed["current_available_memory_meets_audit_requirement"][
        "required_available_memory_bytes"
    ] == 4096
    assert failed["current_available_memory_meets_audit_requirement"][
        "current_system_available_memory_bytes"
    ] == 1024
    assert captured.err == ""


def test_selected_replay_check_cli_rejects_current_low_stage_temp_disk(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=4096),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failed = {check["code"]: check for check in payload["checks"] if not check["ok"]}
    assert "current_stage_temp_disk_meets_audit_requirement" in failed
    assert failed["current_stage_temp_disk_meets_audit_requirement"][
        "required_free_bytes"
    ] == 8192
    assert failed["current_stage_temp_disk_meets_audit_requirement"][
        "current_free_bytes"
    ] == 4096
    assert captured.err == ""


def test_selected_replay_check_cli_rejects_profile_without_acceleration_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    profile_path = prepared / "launch-profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "argv": [
                    "--run-mpsgraph-probe",
                    "--prefill-mpsgraph-min-batch-tokens",
                    "16",
                    "--prefill-mpsgraph-min-matrix-dim",
                    "32",
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    audit_path = prepared / "launch-audit.json"
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["applied_launch_profile"]["sha256"] = profile_sha
    audit_path.write_text(json.dumps(audit_payload) + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failed = {check["code"]: check for check in payload["checks"] if not check["ok"]}
    assert "launch_profile_requires_prefill_acceleration" in failed
    assert "launch_profile_replays_mpsgraph_probe" not in failed
    assert "launch_profile_mpsgraph_thresholds_present" not in failed
    assert captured.err == ""


def test_selected_replay_check_cli_rejects_malformed_acceleration_coverage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    audit_path = prepared / "launch-audit.json"
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    coverage = audit_payload["request_check"]["prefill_acceleration_coverage"]
    coverage.pop("streamed_routed_expert_layer_count")
    for check in audit_payload["launch_audit"]["checks"]:
        if check.get("code") == "request_prefill_acceleration_coverage_ok":
            check.pop("streamed_routed_expert_layer_count")
    audit_path.write_text(json.dumps(audit_payload) + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failed = {check["code"]: check for check in payload["checks"] if not check["ok"]}
    assert "launch_audit_request_prefill_acceleration_coverage_schema_valid" in failed
    assert "launch_audit_request_prefill_acceleration_check_schema_valid" in failed
    assert captured.err == ""


def test_selected_replay_check_cli_rejects_profile_file_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: _test_memory_snapshot(available_bytes=1024**3),
    )
    monkeypatch.setattr(
        "largerlm.cli._selected_replay_disk_usage",
        _test_disk_usage(free_bytes=1024**3),
    )
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )
    assert (
        cli_main(
            [
                "result-bakeoff",
                str(baseline),
                str(slow),
                "--write-selected-replay-json",
                str(replay_json),
            ]
        )
        == 0
    )
    capsys.readouterr()
    (prepared / "launch-profile.json").write_text(
        '{"argv": ["--changed"]}\n',
        encoding="utf-8",
    )

    status = cli_main(["selected-replay-check", str(replay_json), "--json"])

    assert status == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema"] == "largerlm.selected_replay_check.v1"
    assert payload["ok"] is False
    failed_codes = {
        check["code"] for check in payload["checks"] if check["ok"] is not True
    }
    assert "current_replay_files_ready" in failed_codes
    assert "current_launch_profile_sha256_matches_file" in failed_codes
    assert "current_launch_audit_profile_sha256_matches_file" in failed_codes
    assert "launch_profile_file_sha256_stable" in failed_codes
    assert captured.err == ""


def test_result_bakeoff_cli_write_replay_ignores_fast_candidate_without_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    baseline = tmp_path / "baseline.json"
    fast_without_files = tmp_path / "fast-without-files.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    fast_without_files.write_text(
        json.dumps(_scaled_elapsed_result(0.7)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(fast_without_files),
            "--write-selected-replay-json",
            str(replay_json),
        ]
    )

    assert status == 0
    captured = capsys.readouterr()
    assert "promotion requires replay files ready: True" in captured.out
    assert "selected: role=baseline" in captured.out
    assert "candidate_replay_files_not_ready" in captured.out
    assert captured.err == ""
    payload = json.loads(replay_json.read_text(encoding="utf-8"))
    assert payload["selected_role"] == "baseline"
    assert payload["selected_result"] == str(baseline)
    assert payload["launch_binding"]["replay_files_ready"] is True


def test_result_bakeoff_cli_require_selected_replay_ready_rejects_missing_prompt_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_payload = _scaled_elapsed_result(1.0)
    candidate_payload = _scaled_elapsed_result(1.1)
    baseline_payload["request"].pop("prompt_token_ids")
    candidate_payload["request"].pop("prompt_token_ids")
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    slow.write_text(json.dumps(candidate_payload), encoding="utf-8")

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--require-selected-replay-ready",
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "selected: role=baseline" in captured.out
    assert "replay_ready=False" in captured.out
    assert "selected result is not replay-ready" in captured.err


def test_result_bakeoff_cli_refuses_to_write_replay_artifacts_without_replay_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_payload = _scaled_elapsed_result(1.0)
    candidate_payload = _scaled_elapsed_result(1.1)
    baseline_payload["request"].pop("prompt_token_ids")
    candidate_payload["request"].pop("prompt_token_ids")
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(json.dumps(baseline_payload), encoding="utf-8")
    slow.write_text(json.dumps(candidate_payload), encoding="utf-8")

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--write-selected-replay-json",
            str(replay_json),
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "selected result is not replay-ready" in captured.err
    assert not replay_json.exists()


def test_result_bakeoff_cli_refuses_replay_artifacts_with_audit_profile_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_replay_binding_files(tmp_path)
    (prepared / "launch-audit.json").write_text(
        json.dumps(
            {
                "schema": "largerlm.launch_audit.v1",
                "applied_launch_profile": {
                    "path": str(prepared / "launch-profile.json"),
                    "sha256": "b" * 64,
                },
                "launch_audit": {
                    "ok": True,
                    "checks": [{"ok": True}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    slow = tmp_path / "slow.json"
    replay_json = tmp_path / "selected-replay.json"
    baseline.write_text(
        json.dumps(_scaled_elapsed_result(1.0, prepared_root=prepared)),
        encoding="utf-8",
    )
    slow.write_text(
        json.dumps(_scaled_elapsed_result(1.1, prepared_root=prepared)),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "result-bakeoff",
            str(baseline),
            str(slow),
            "--write-selected-replay-json",
            str(replay_json),
        ]
    )

    assert status == 1
    captured = capsys.readouterr()
    assert "replay_ready=True" in captured.out
    assert "files_ready=False" in captured.out
    assert "selected replay files are not ready" in captured.err
    assert not replay_json.exists()


def test_result_bakeoff_cli_json_reports_winner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_scaled_elapsed_result(1.0)), encoding="utf-8")
    candidate.write_text(json.dumps(_scaled_elapsed_result(0.75)), encoding="utf-8")

    status = cli_main(
        ["result-bakeoff", str(baseline), str(candidate), "--json"]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "largerlm.result_bakeoff.v1"
    assert payload["selected"]["role"] == "candidate"
    assert payload["selected"]["path"] == str(candidate)
    assert payload["selected"]["launch_binding"]["replay_ready"] is True
    assert payload["winner"]["path"] == str(candidate)
    assert payload["winner"]["launch_binding"]["safe_to_replay"] is True
    assert payload["winner"]["launch_binding"]["replay_ready"] is True
    assert payload["baseline_retained"] is False
