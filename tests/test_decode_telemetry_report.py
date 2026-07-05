from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "decode_telemetry_report.py"
_SPEC = importlib.util.spec_from_file_location(
    "decode_telemetry_report_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
reporter = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(reporter)


def test_build_report_from_probe_decode_layers_payload() -> None:
    payload = {
        "probe_decode_layers": {
            "elapsed_seconds": 4.0,
            "final_logits_elapsed_seconds": 0.5,
            "final_logits_bytes_read": 2048,
            "final_logits_lm_head_bytes_read": 2000,
            "layer_elapsed_seconds": 3.8,
            "layer_count": 78,
            "dense_layer_count": 3,
            "moe_layer_count": 75,
            "expert_bytes_read": 2 * reporter.GIB,
            "dense_mlp_bytes_read": 123,
            "expert_read_seconds": 0.25,
            "moe_mlp_output_write_seconds": 0.03,
            "moe_mlp_overhead_seconds": 0.07,
            "layer_overhead_seconds": 0.09,
            "attn_projection_command_buffer_count": 78,
            "attn_projection_synchronous_wait_count": 12,
            "attn_projection_async_submitted_count": 66,
            "rope_mla_command_buffer_count": 0,
            "attn_output_command_buffer_count": 78,
            "attn_output_context1_o_proj_cache_count": 78,
            "attn_output_resident_mmap_backed_count": 77,
            "post_attn_norm_command_buffer_count": 0,
            "router_command_buffer_count": 0,
            "post_attn_norm_router_command_buffer_count": 0,
            "dense_mlp_command_buffer_count": 3,
            "dense_mlp_synchronous_wait_count": 1,
            "dense_mlp_async_submitted_count": 2,
            "moe_mlp_command_buffer_count": 75,
            "moe_mlp_synchronous_wait_count": 12,
            "attn_output_norm_router_fused_count": 75,
            "rope_mla_attn_output_norm_router_fused_count": 75,
            "rope_mla_input_buffer_direct_count": 78,
            "attn_output_buffer_direct_count": 75,
            "moe_mlp_input_buffer_direct_count": 75,
            "layer_input_buffer_direct_count": 77,
            "command_buffer_count": 234,
            "synchronous_wait_count_estimate": 234,
            "expert_read_dispatch_count": 75,
            "expert_read_task_count": 600,
            "expert_read_max_task_count": 8,
            "expert_read_max_worker_count": 8,
            "expert_read_pool_dispatch_count": 75,
            "expert_read_serial_dispatch_count": 0,
            "attn_projection_elapsed_seconds": 0.4,
            "mla_attention_elapsed_seconds": 2.0,
            "mla_attention_value_read_seconds": 1.2,
            "mla_attention_kernel_seconds": 0.7,
            "attn_output_elapsed_seconds": 0.5,
            "attn_output_bytes_read": reporter.GIB,
            "attn_output_read_seconds": 0.11,
            "attn_output_projection_kernel_seconds": 0.22,
            "post_attn_norm_weight_bytes_read": 4096,
            "post_attn_norm_weight_read_seconds": 0.03,
            "router_bytes_read": 8192,
            "router_correction_bias_bytes_read": 1024,
            "router_read_seconds": 0.04,
            "router_kernel_seconds": 0.05,
            "mlp_elapsed_seconds": 0.7,
            "moe_mlp_kernel_seconds": 0.2,
        }
    }

    report = reporter.build_decode_telemetry_report(
        payload,
        reference_cold_read_gib_per_second=4.0,
    )

    assert report["source_kind"] == "probe_decode_layers"
    assert report["layer_count"] == 78
    assert report["moe_layer_count"] == 75
    assert report["bytes"]["expert_gib_read"] == 2.0
    assert report["bytes"]["attn_output_bytes_read"] == reporter.GIB
    assert report["bytes"]["post_attn_norm_weight_bytes_read"] == 4096
    assert report["bytes"]["router_bytes_read"] == 8192
    assert report["bytes"]["router_correction_bias_bytes_read"] == 1024
    assert report["bytes"]["final_logits_bytes_read"] == 2048
    assert report["bytes"]["final_logits_lm_head_bytes_read"] == 2000
    assert report["expert_read"]["observed_gib_per_second"] == 8.0
    assert report["expert_read"]["reference_cold_read_floor_seconds"] == 0.5
    assert report["expert_read"]["pooled_read_ok"] is True
    assert (
        report["timing"]["primary_top_level_bottleneck"]["label"]
        == "mla_attention"
    )
    nested = {
        item["label"]: item["seconds"]
        for item in report["timing"]["nested_components"]
    }
    assert nested["mla_value_read"] == 1.2
    assert nested["mla_kernel"] == 0.7
    assert nested["attn_output_read"] == 0.11
    assert nested["attn_output_projection_kernel"] == 0.22
    assert nested["post_attn_norm_weight_read"] == 0.03
    assert nested["router_read"] == 0.04
    assert nested["router_kernel"] == 0.05
    assert nested["moe_mlp_output_write"] == 0.03
    assert nested["moe_mlp_overhead"] == 0.07
    assert nested["layer_overhead"] == 0.09
    assert report["command_buffers"]["count"] == 234
    assert report["command_buffers"]["synchronous_wait_count_estimate"] == 234
    assert report["command_buffers"]["attn_projection_synchronous_wait_count"] == 12
    assert report["command_buffers"]["attn_projection_async_submitted_count"] == 66
    assert report["command_buffers"]["post_attn_norm_count"] == 0
    assert report["command_buffers"]["router_count"] == 0
    assert report["command_buffers"]["post_attn_norm_router_count"] == 0
    assert report["command_buffers"]["moe_mlp_count"] == 75
    assert report["command_buffers"]["dense_mlp_count"] == 3
    assert report["command_buffers"]["dense_mlp_synchronous_wait_count"] == 1
    assert report["command_buffers"]["dense_mlp_async_submitted_count"] == 2
    assert report["command_buffers"]["moe_mlp_synchronous_wait_count"] == 12
    assert report["command_buffers"]["attn_output_context1_o_proj_cache_count"] == 78
    assert report["command_buffers"]["attn_output_resident_mmap_backed_count"] == 77
    assert report["command_buffers"]["attn_output_norm_router_fused_count"] == 75
    assert (
        report["command_buffers"]["rope_mla_attn_output_norm_router_fused_count"]
        == 75
    )
    assert report["command_buffers"]["rope_mla_input_buffer_direct_count"] == 78
    assert report["command_buffers"]["attn_output_buffer_direct_count"] == 75
    assert report["command_buffers"]["moe_mlp_input_buffer_direct_count"] == 75
    assert report["command_buffers"]["layer_input_buffer_direct_count"] == 77
    top_level = {
        item["label"]: item["seconds"]
        for item in report["timing"]["top_level_components"]
    }
    assert top_level["rope_mla_attn_output_norm_router_fused"] == 0.5
    frontier = report["optimization_frontier"]
    assert frontier["seconds_per_token"]["attention_output_combined"] == 0.33
    assert frontier["counts_per_token"]["command_buffers"] == 234
    assert frontier["counts_per_token"]["synchronous_waits_estimate"] == 234
    assert frontier["counts_per_token"]["attn_projection_synchronous_waits"] == 12
    assert frontier["sync_waits_by_stage"][0] == {
        "label": "attn_output",
        "count": 78,
        "per_token": 78,
    }
    assert frontier["ranked_seconds"][0]["label"] == "mla_value_read"
    assert frontier["primary_recommendation"]["code"] == "attention_output_stage_first"
    assert [
        item["code"] for item in frontier["recommendations"]
    ] == [
        "attention_output_stage_first",
        "reduce_command_buffer_and_wait_count",
    ]


def test_fused_attention_output_label_can_be_primary_bottleneck() -> None:
    payload = {
        "probe_decode_layers": {
            "elapsed_seconds": 4.0,
            "layer_elapsed_seconds": 4.0,
            "attn_projection_elapsed_seconds": 0.4,
            "mla_attention_elapsed_seconds": 0.2,
            "attn_output_elapsed_seconds": 2.5,
            "mlp_elapsed_seconds": 0.7,
            "attn_output_command_buffer_count": 78,
            "attn_output_norm_router_fused_count": 75,
            "rope_mla_attn_output_norm_router_fused_count": 75,
        }
    }

    report = reporter.build_decode_telemetry_report(payload)

    primary = report["timing"]["primary_top_level_bottleneck"]
    assert primary["label"] == "rope_mla_attn_output_norm_router_fused"
    assert primary["seconds"] == 2.5


def test_layer_frontier_ranks_hot_decode_layers() -> None:
    payload = {
        "probe_decode_layers": {
            "elapsed_seconds": 2.0,
            "layer_elapsed_seconds": 2.0,
            "command_buffer_count": 12,
            "synchronous_wait_count_estimate": 9,
            "layers": [
                {
                    "layer": 3,
                    "kind": "moe",
                    "elapsed_seconds": 0.4,
                    "attn_output_read_seconds": 0.02,
                    "attn_output_projection_kernel_seconds": 0.03,
                    "expert_read_seconds": 0.04,
                    "moe_mlp_overhead_seconds": 0.01,
                    "layer_overhead_seconds": 0.005,
                    "attn_projection_command_buffer_count": 1,
                    "attn_projection_synchronous_wait_count": 0,
                    "rope_mla_command_buffer_count": 1,
                    "attn_output_command_buffer_count": 1,
                    "moe_mlp_command_buffer_count": 1,
                    "moe_mlp_synchronous_wait_count": 0,
                    "attn_output_bytes_read": 100,
                    "expert_bytes_read": 200,
                    "router_topk_backend": "metal",
                    "rope_mla_attn_output_norm_router_fused": True,
                    "moe_mlp_input_buffer_direct": True,
                },
                {
                    "layer": 4,
                    "kind": "moe",
                    "elapsed_seconds": 0.7,
                    "attn_output_read_seconds": 0.04,
                    "attn_output_projection_kernel_seconds": 0.2,
                    "expert_read_seconds": 0.03,
                    "moe_mlp_overhead_seconds": 0.08,
                    "layer_overhead_seconds": 0.04,
                    "attn_projection_command_buffer_count": 1,
                    "attn_projection_synchronous_wait_count": 1,
                    "rope_mla_command_buffer_count": 1,
                    "attn_output_command_buffer_count": 1,
                    "post_attn_norm_command_buffer_count": 1,
                    "router_command_buffer_count": 1,
                    "moe_mlp_command_buffer_count": 1,
                    "moe_mlp_synchronous_wait_count": 1,
                    "attn_output_bytes_read": 300,
                    "expert_bytes_read": 400,
                    "router_topk_backend": "cpu",
                    "rope_mla_attn_output_norm_router_fused": False,
                    "moe_mlp_input_buffer_direct": False,
                },
                {
                    "layer": 0,
                    "kind": "dense",
                    "elapsed_seconds": 0.3,
                    "attn_output_read_seconds": 0.01,
                    "attn_output_projection_kernel_seconds": 0.01,
                    "layer_overhead_seconds": 0.02,
                    "dense_mlp_command_buffer_count": 1,
                },
            ],
        }
    }

    report = reporter.build_decode_telemetry_report(payload)

    frontier = report["layer_frontier"]
    assert frontier["layer_count"] == 3
    hot = frontier["top_attention_output_layers"][0]
    assert hot["layer"] == 4
    assert hot["kind"] == "moe"
    assert abs(hot["attention_output_combined_seconds"] - 0.24) < 1e-12
    assert hot["command_buffer_count"] == 6
    assert hot["synchronous_wait_count_estimate"] == 6
    assert hot["attn_projection_synchronous_wait_count"] == 1
    assert hot["attn_projection_async_submitted"] is False
    assert hot["router_topk_backend"] == "cpu"
    assert hot["rope_mla_attn_output_norm_router_fused"] is False
    assert hot["moe_mlp_input_buffer_direct"] is False
    assert frontier["top_layer_overhead_layers"][0]["layer"] == 4
    assert frontier["top_moe_overhead_layers"][0]["layer"] == 4
    assert frontier["top_command_buffer_layers"][0]["layer"] == 4
    assert frontier["top_sync_wait_layers"][0]["layer"] == 4


def test_build_report_from_smoke_dims_payload_flags_serial_fallback() -> None:
    payload = {
        "dims": {
            "elapsed_seconds": 1.0,
            "expert_bytes_read": reporter.GIB,
            "expert_read_seconds": 0.5,
            "expert_read_task_count": 8,
            "expert_read_pool_dispatch_count": 0,
            "expert_read_serial_dispatch_count": 8,
        }
    }

    report = reporter.build_decode_telemetry_report(payload)

    assert report["source_kind"] == "smoke_dims"
    assert report["expert_read"]["observed_gib_per_second"] == 2.0
    assert report["expert_read"]["pooled_read_ok"] is False


def test_build_report_from_generate_steps_sums_step_telemetry(tmp_path: Path) -> None:
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps(
            {
                "probe_generate": {
                    "steps": [
                        {
                            "decode_elapsed_seconds": 1.0,
                            "layer_count": 4,
                            "dense_layer_count": 1,
                            "moe_layer_count": 3,
                            "expert_bytes_read": 10,
                            "expert_read_seconds": 0.1,
                            "moe_mlp_output_write_seconds": 0.01,
                            "moe_mlp_overhead_seconds": 0.02,
                            "layer_overhead_seconds": 0.03,
                            "command_buffer_count": 4,
                            "synchronous_wait_count_estimate": 4,
                            "moe_mlp_command_buffer_count": 3,
                            "moe_mlp_synchronous_wait_count": 0,
                            "post_attn_norm_router_command_buffer_count": 1,
                            "attn_output_norm_router_fused_count": 1,
                            "rope_mla_attn_output_norm_router_fused_count": 1,
                            "rope_mla_input_buffer_direct_count": 1,
                            "attn_output_buffer_direct_count": 1,
                            "moe_mlp_input_buffer_direct_count": 1,
                            "layer_input_buffer_direct_count": 1,
                            "expert_read_task_count": 4,
                            "expert_read_pool_dispatch_count": 1,
                            "expert_read_serial_dispatch_count": 0,
                            "expert_read_max_task_count": 4,
                        },
                        {
                            "decode_elapsed_seconds": 2.0,
                            "layer_count": 4,
                            "dense_layer_count": 1,
                            "moe_layer_count": 3,
                            "expert_bytes_read": 20,
                            "expert_read_seconds": 0.2,
                            "moe_mlp_output_write_seconds": 0.03,
                            "moe_mlp_overhead_seconds": 0.04,
                            "layer_overhead_seconds": 0.05,
                            "command_buffer_count": 5,
                            "synchronous_wait_count_estimate": 5,
                            "moe_mlp_command_buffer_count": 3,
                            "moe_mlp_synchronous_wait_count": 1,
                            "post_attn_norm_router_command_buffer_count": 1,
                            "attn_output_norm_router_fused_count": 1,
                            "rope_mla_attn_output_norm_router_fused_count": 1,
                            "rope_mla_input_buffer_direct_count": 1,
                            "attn_output_buffer_direct_count": 1,
                            "moe_mlp_input_buffer_direct_count": 1,
                            "layer_input_buffer_direct_count": 1,
                            "expert_read_task_count": 8,
                            "expert_read_pool_dispatch_count": 1,
                            "expert_read_serial_dispatch_count": 0,
                            "expert_read_max_task_count": 8,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    status = reporter.main([str(path), "--json"])

    assert status == 0
    report = reporter.build_decode_telemetry_report(json.loads(path.read_text()))
    assert report["source_kind"] == "probe_generate_steps"
    assert report["elapsed_seconds"] == 3.0
    assert report["layer_count"] == 4
    assert report["dense_layer_count"] == 1
    assert report["moe_layer_count"] == 3
    assert report["bytes"]["expert_bytes_read"] == 30
    assert report["expert_read"]["task_count"] == 12
    assert report["expert_read"]["max_task_count"] == 8
    nested = {
        item["label"]: item["seconds"]
        for item in report["timing"]["nested_components"]
    }
    assert nested["moe_mlp_output_write"] == 0.04
    assert nested["moe_mlp_overhead"] == 0.06
    assert nested["layer_overhead"] == 0.08
    assert report["command_buffers"]["count"] == 9
    assert report["command_buffers"]["synchronous_wait_count_estimate"] == 9
    assert report["command_buffers"]["moe_mlp_count"] == 6
    assert report["command_buffers"]["moe_mlp_synchronous_wait_count"] == 1
    assert report["command_buffers"]["post_attn_norm_router_count"] == 2
    assert report["command_buffers"]["attn_output_norm_router_fused_count"] == 2
    assert (
        report["command_buffers"]["rope_mla_attn_output_norm_router_fused_count"]
        == 2
    )
    assert report["command_buffers"]["rope_mla_input_buffer_direct_count"] == 2
    assert report["command_buffers"]["attn_output_buffer_direct_count"] == 2
    assert report["command_buffers"]["moe_mlp_input_buffer_direct_count"] == 2
    assert report["command_buffers"]["layer_input_buffer_direct_count"] == 2


def test_build_report_from_http_request_wrapper_uses_last_request() -> None:
    payload = {
        "requests": [
            {
                "payload": {
                    "steps": [
                        {
                            "decode_elapsed_seconds": 9.0,
                            "expert_bytes_read": 90,
                            "expert_read_seconds": 9.0,
                        }
                    ]
                }
            },
            {
                "payload": {
                    "steps": [
                        {
                            "decode_elapsed_seconds": 1.0,
                            "layer_count": 2,
                            "dense_layer_count": 1,
                            "moe_layer_count": 1,
                            "expert_bytes_read": reporter.GIB,
                            "expert_read_seconds": 0.25,
                            "moe_mlp_output_write_seconds": 0.01,
                            "moe_mlp_overhead_seconds": 0.02,
                            "layer_overhead_seconds": 0.03,
                            "command_buffer_count": 4,
                            "synchronous_wait_count_estimate": 4,
                            "post_attn_norm_router_command_buffer_count": 1,
                            "attn_output_norm_router_fused_count": 1,
                            "rope_mla_attn_output_norm_router_fused_count": 1,
                            "rope_mla_input_buffer_direct_count": 1,
                            "attn_output_buffer_direct_count": 1,
                            "moe_mlp_input_buffer_direct_count": 1,
                            "layer_input_buffer_direct_count": 1,
                        }
                    ]
                }
            },
        ]
    }

    report = reporter.build_decode_telemetry_report(payload)

    assert report["source_kind"] == "request_last_steps"
    assert report["elapsed_seconds"] == 1.0
    assert report["layer_count"] == 2
    assert report["dense_layer_count"] == 1
    assert report["moe_layer_count"] == 1
    assert report["bytes"]["expert_gib_read"] == 1.0
    assert report["expert_read"]["observed_gib_per_second"] == 4.0
    assert report["command_buffers"]["count"] == 4
    assert report["command_buffers"]["synchronous_wait_count_estimate"] == 4
    assert report["command_buffers"]["post_attn_norm_router_count"] == 1
    assert report["command_buffers"]["attn_output_norm_router_fused_count"] == 1
    assert (
        report["command_buffers"]["rope_mla_attn_output_norm_router_fused_count"]
        == 1
    )
    assert report["command_buffers"]["rope_mla_input_buffer_direct_count"] == 1
    assert report["command_buffers"]["attn_output_buffer_direct_count"] == 1
    assert report["command_buffers"]["moe_mlp_input_buffer_direct_count"] == 1
    assert report["command_buffers"]["layer_input_buffer_direct_count"] == 1
    nested = {
        item["label"]: item["seconds"]
        for item in report["timing"]["nested_components"]
    }
    assert nested["moe_mlp_output_write"] == 0.01
    assert nested["moe_mlp_overhead"] == 0.02
    assert nested["layer_overhead"] == 0.03


def test_build_report_from_single_http_payload_wrapper() -> None:
    payload = {
        "payload": {
            "steps": [
                {
                    "decode_elapsed_seconds": 1.0,
                    "final_logits_elapsed_seconds": 0.1,
                    "layer_count": 2,
                    "dense_layer_count": 1,
                    "moe_layer_count": 1,
                    "expert_bytes_read": reporter.GIB,
                    "expert_read_seconds": 0.25,
                    "command_buffer_count": 4,
                    "synchronous_wait_count_estimate": 3,
                    "layer_input_buffer_direct_count": 1,
                },
                {
                    "decode_elapsed_seconds": 2.0,
                    "final_logits_elapsed_seconds": 0.2,
                    "layer_count": 2,
                    "dense_layer_count": 1,
                    "moe_layer_count": 1,
                    "expert_bytes_read": reporter.GIB,
                    "expert_read_seconds": 0.5,
                    "command_buffer_count": 4,
                    "synchronous_wait_count_estimate": 3,
                    "layer_input_buffer_direct_count": 1,
                },
            ]
        }
    }

    report = reporter.build_decode_telemetry_report(payload)

    assert report["source_kind"] == "payload_steps"
    assert report["token_count"] == 2
    assert report["elapsed_seconds"] == 3.0
    assert report["final_logits_elapsed_seconds"] == 0.30000000000000004
    assert report["decode_tokens_per_second"] == 2 / 3.0
    assert report["with_final_logits_tokens_per_second"] == 2 / 3.3
    assert report["layer_count"] == 2
    assert report["bytes"]["expert_gib_read"] == 2.0
    assert report["expert_read"]["observed_gib_per_second"] == 2.6666666666666665
    assert report["command_buffers"]["count"] == 8
    assert report["command_buffers"]["synchronous_wait_count_estimate"] == 6
    assert report["command_buffers"]["layer_input_buffer_direct_count"] == 2


def test_build_report_from_flat_metal_generate_result() -> None:
    payload = {
        "generated_token_ids": [15, 11, 15],
        "elapsed_seconds": 4.5,
        "decode_elapsed_seconds": [1.0, 1.1, 1.2],
        "final_logits_elapsed_seconds": [0.1, 0.11, 0.12],
        "final_logits_bytes_read": [1000, 20, 20],
        "final_logits_lm_head_bytes_read": [960, 0, 0],
        "decode_layer_count": [78, 78, 78],
        "decode_dense_layer_count": [3, 3, 3],
        "decode_moe_layer_count": [75, 75, 75],
        "decode_layer_elapsed_seconds": [0.9, 1.0, 1.1],
        "decode_attn_projection_elapsed_seconds": [0.2, 0.21, 0.22],
        "decode_mla_attention_elapsed_seconds": [0.3, 0.31, 0.32],
        "decode_attn_output_elapsed_seconds": [0.4, 0.41, 0.42],
        "decode_attn_output_bytes_read": [10, 20, 30],
        "decode_attn_output_read_seconds": [0.04, 0.05, 0.06],
        "decode_attn_output_projection_kernel_seconds": [0.3, 0.31, 0.32],
        "decode_post_attn_norm_weight_bytes_read": [1, 2, 3],
        "decode_post_attn_norm_weight_read_seconds": [0.001, 0.002, 0.003],
        "decode_router_bytes_read": [4, 5, 6],
        "decode_router_correction_bias_bytes_read": [7, 8, 9],
        "decode_router_read_seconds": [0.004, 0.005, 0.006],
        "decode_router_kernel_seconds": [0.007, 0.008, 0.009],
        "decode_mlp_elapsed_seconds": [0.5, 0.51, 0.52],
        "decode_expert_bytes_read": [100, 200, 300],
        "decode_expert_read_seconds": [0.01, 0.02, 0.03],
        "decode_moe_mlp_command_buffer_count": [75, 75, 75],
        "decode_moe_mlp_synchronous_wait_count": [0, 0, 75],
        "decode_attn_output_context1_o_proj_cache_count": [78, 78, 78],
        "decode_attn_output_resident_mmap_backed_count": [77, 78, 78],
        "decode_command_buffer_count": [200, 200, 200],
        "decode_synchronous_wait_count_estimate": [125, 125, 200],
        "decode_rope_mla_attn_output_norm_router_fused_count": [75, 75, 75],
        "decode_layer_input_buffer_direct_count": [77, 77, 77],
        "decode_expert_read_task_count": [600, 600, 600],
        "decode_expert_read_pool_dispatch_count": [75, 75, 75],
        "decode_expert_read_serial_dispatch_count": [0, 0, 0],
    }

    report = reporter.build_decode_telemetry_report(payload)

    assert report["source_kind"] == "metal_generate_result"
    assert report["token_count"] == 3
    assert report["elapsed_seconds"] == 3.3
    assert report["final_logits_elapsed_seconds"] == 0.33
    assert report["with_final_logits_tokens_per_second"] == 3 / 3.63
    assert report["layer_count"] == 78
    assert report["dense_layer_count"] == 3
    assert report["moe_layer_count"] == 75
    assert report["bytes"]["expert_bytes_read"] == 600
    assert report["bytes"]["attn_output_bytes_read"] == 60
    assert report["bytes"]["post_attn_norm_weight_bytes_read"] == 6
    assert report["bytes"]["router_bytes_read"] == 15
    assert report["bytes"]["router_correction_bias_bytes_read"] == 24
    assert report["bytes"]["final_logits_bytes_read"] == 1040
    assert report["bytes"]["final_logits_lm_head_bytes_read"] == 960
    assert report["expert_read"]["task_count"] == 1800
    assert report["expert_read"]["pool_dispatch_count"] == 225
    assert report["command_buffers"]["count"] == 600
    assert report["command_buffers"]["synchronous_wait_count_estimate"] == 450
    assert report["command_buffers"]["moe_mlp_count"] == 225
    assert report["command_buffers"]["moe_mlp_synchronous_wait_count"] == 75
    assert report["command_buffers"]["attn_output_context1_o_proj_cache_count"] == 234
    assert report["command_buffers"]["attn_output_resident_mmap_backed_count"] == 233
    assert (
        report["command_buffers"]["rope_mla_attn_output_norm_router_fused_count"]
        == 225
    )
    assert report["command_buffers"]["layer_input_buffer_direct_count"] == 231
    nested = {
        item["label"]: item["seconds"]
        for item in report["timing"]["nested_components"]
    }
    assert abs(nested["attn_output_read"] - 0.15) < 1e-12
    assert abs(nested["attn_output_projection_kernel"] - 0.93) < 1e-12
    assert abs(nested["post_attn_norm_weight_read"] - 0.006) < 1e-12
    assert abs(nested["router_read"] - 0.015) < 1e-12
    assert abs(nested["router_kernel"] - 0.024) < 1e-12
    frontier = report["optimization_frontier"]
    assert (
        abs(frontier["seconds_per_token"]["attention_output_combined"] - 0.36)
        < 1e-12
    )
    assert frontier["counts_per_token"]["command_buffers"] == 200
    assert frontier["counts_per_token"]["synchronous_waits_estimate"] == 150
    assert frontier["counts_per_token"]["moe_mlp_synchronous_waits"] == 25
    assert frontier["primary_recommendation"]["code"] == "attention_output_stage_first"
    assert frontier["ranked_seconds"][0]["label"] == "attention_output_combined"
