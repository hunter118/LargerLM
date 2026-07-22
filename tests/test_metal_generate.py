from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from largerlm.config import ModelConfig
from largerlm import metal_generate
from largerlm.metal_generate import MetalGenerateError, generate_metal_token_ids


class _FakeLayout:
    def to_json(self) -> dict[str, object]:
        return {
            "version": 1,
            "model_type": "glm-test",
            "max_context_tokens": 2,
            "dtype": "F32",
            "segments": [],
        }


def test_append_expert_cache_launch_args() -> None:
    cmd = ["metal/glm_moe_infer"]

    metal_generate._append_expert_cache_launch_args(
        cmd,
        expert_pin_plan="/tmp/expert-pin-plan.json",
        max_adaptive_expert_cache_gib=36.0,
    )

    assert cmd == [
        "metal/glm_moe_infer",
        "--expert-pin-plan",
        "/tmp/expert-pin-plan.json",
        "--max-adaptive-expert-cache-gib",
        "36",
    ]


def test_append_expert_cache_launch_args_keeps_default_command_unchanged() -> None:
    cmd = ["metal/glm_moe_infer"]

    metal_generate._append_expert_cache_launch_args(
        cmd,
        expert_pin_plan=None,
        max_adaptive_expert_cache_gib=0.0,
    )

    assert cmd == ["metal/glm_moe_infer"]


def test_generate_metal_token_ids_keeps_auto_work_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    kept_root = tmp_path / "kept-metal-generate"

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("sparse", "sparse"),
    )

    monkeypatch.setattr(
        metal_generate,
        "load_prepared_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)
    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        lambda *args, **kwargs: _FakeLayout(),
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=64),
    )

    def fake_mkdtemp(*args, **kwargs) -> str:
        kept_root.mkdir()
        return str(kept_root)

    def fake_embed_token(*args, **kwargs) -> None:
        Path(kwargs["output_f32_path"]).write_bytes(b"\0" * 16)

    def fake_run_generate_server_jsonl_request(*, binary, prepared_dir, request, quiet):
        assert str(binary) == "/tmp/fake-glm_moe_infer"
        assert str(prepared_dir) == str(prepared)
        assert request["input_token_id"] == 0
        assert "input_f32" not in request
        assert "output_f32" not in request
        assert "output_generated_json" not in request
        assert "output_next_input_f32" not in request
        assert request["decode_layers"] == "0,1"
        assert request["generate_steps"] == 2
        assert request["position"] == 0
        assert request["context_length"] == 1
        assert request["expert_buffer_count"] == 1
        assert request["include_shared_expert"] is False
        assert request["min_free_unified_memory_gib"] == 0.0
        assert request["in_memory_decode_cache"] is True
        assert request["cache_mla_kv_b_f32"] is True
        assert request["max_mla_kv_b_cache_mib"] == 512.0
        assert request["context1_o_proj_cache_layout"] == "/tmp/context1-layout.json"
        assert request["context1_o_proj_cache_file"] == "/tmp/context1-cache.bin"
        payload = {
            "estimated_live_working_set_bytes": 1234,
            "expert_buffer_count_runtime_allocated": 1,
            "executor_mla_kv_b_cache_enabled": True,
            "executor_mla_kv_b_cache_current_bytes": 96,
            "executor_mla_kv_b_cache_live_estimate_bytes": 512 * 1024 * 1024,
            "probe_generate": {
                "ok": True,
                "entry": "generate_token_ids",
                "generated_token_ids": [5, 6],
                "steps": [
                    {
                        "decode_elapsed_seconds": 1.25,
                        "final_logits_elapsed_seconds": 0.1,
                        "final_logits_bytes_read": 1024,
                        "final_logits_lm_head_bytes_read": 1000,
                        "final_logits_read_seconds": 0.01,
                        "final_logits_kernel_seconds": 0.08,
                        "final_logits_resident_mmap_backed": False,
                        "expert_bytes_read": 128,
                        "dense_mlp_bytes_read": 16,
                        "shared_bytes_read": 12,
                        "layer_count": 78,
                        "dense_layer_count": 3,
                        "moe_layer_count": 75,
                        "layer_elapsed_seconds": 1.2,
                        "attn_projection_elapsed_seconds": 0.11,
                        "mla_attention_elapsed_seconds": 0.22,
                        "mla_attention_cache_read_seconds": 0.01,
                        "mla_attention_value_read_seconds": 0.02,
                        "mla_attention_kernel_seconds": 0.03,
                        "mla_attention_output_write_seconds": 0.04,
                        "attn_output_elapsed_seconds": 0.05,
                        "attn_output_bytes_read": 64,
                        "attn_output_read_seconds": 0.012,
                        "attn_output_projection_kernel_seconds": 0.034,
                        "post_attn_norm_weight_bytes_read": 8,
                        "post_attn_norm_weight_read_seconds": 0.001,
                        "router_bytes_read": 24,
                        "router_correction_bias_bytes_read": 4,
                        "router_read_seconds": 0.002,
                        "router_kernel_seconds": 0.003,
                        "mlp_elapsed_seconds": 0.06,
                        "dense_mlp_elapsed_seconds": 0.07,
                        "moe_mlp_elapsed_seconds": 0.08,
                        "expert_read_seconds": 0.03,
                        "shared_read_seconds": 0.01,
                        "shared_prefetch_seconds": 0.009,
                        "shared_prefetch_used_count": 75,
                        "moe_mlp_kernel_seconds": 0.04,
                        "moe_mlp_output_write_seconds": 0.005,
                        "moe_mlp_overhead_seconds": 0.006,
                        "layer_overhead_seconds": 0.009,
                        "attn_projection_command_buffer_count": 3,
                        "attn_projection_synchronous_wait_count": 1,
                        "attn_projection_async_submitted_count": 77,
                        "rope_mla_command_buffer_count": 0,
                        "attn_output_command_buffer_count": 1,
                        "attn_output_context1_o_proj_cache_count": 78,
                        "attn_output_resident_mmap_backed_count": 77,
                        "post_attn_norm_command_buffer_count": 0,
                        "router_command_buffer_count": 0,
                        "post_attn_norm_router_command_buffer_count": 0,
                        "dense_mlp_command_buffer_count": 0,
                        "dense_mlp_synchronous_wait_count": 0,
                        "dense_mlp_async_submitted_count": 0,
                        "moe_mlp_command_buffer_count": 1,
                        "moe_mlp_synchronous_wait_count": 0,
                        "attn_output_norm_router_fused_count": 1,
                        "rope_mla_attn_output_norm_router_fused_count": 1,
                        "rope_mla_input_buffer_direct_count": 1,
                        "attn_output_buffer_direct_count": 1,
                        "moe_mlp_input_buffer_direct_count": 1,
                        "layer_input_buffer_direct_count": 1,
                        "command_buffer_count": 5,
                        "synchronous_wait_count_estimate": 5,
                        "expert_read_dispatch_count": 1,
                        "expert_read_task_count": 8,
                        "expert_read_max_task_count": 8,
                        "expert_read_max_worker_count": 8,
                        "expert_read_pool_dispatch_count": 1,
                        "expert_read_serial_dispatch_count": 0,
                        "mla_value_cache_hit_count": 0,
                        "mla_value_cache_store_count": 2,
                        "mla_value_cache_bytes": 64,
                        "mla_value_cache_total_bytes": 64,
                    },
                    {
                        "decode_elapsed_seconds": 1.5,
                        "final_logits_elapsed_seconds": 0.2,
                        "final_logits_bytes_read": 24,
                        "final_logits_lm_head_bytes_read": 0,
                        "final_logits_read_seconds": 0.001,
                        "final_logits_kernel_seconds": 0.18,
                        "final_logits_resident_mmap_backed": True,
                        "expert_bytes_read": 256,
                        "dense_mlp_bytes_read": 32,
                        "shared_bytes_read": 24,
                        "layer_count": 78,
                        "dense_layer_count": 3,
                        "moe_layer_count": 75,
                        "layer_elapsed_seconds": 1.4,
                        "attn_projection_elapsed_seconds": 0.12,
                        "mla_attention_elapsed_seconds": 0.23,
                        "mla_attention_cache_read_seconds": 0.011,
                        "mla_attention_value_read_seconds": 0.021,
                        "mla_attention_kernel_seconds": 0.031,
                        "mla_attention_output_write_seconds": 0.041,
                        "attn_output_elapsed_seconds": 0.051,
                        "attn_output_bytes_read": 65,
                        "attn_output_read_seconds": 0.013,
                        "attn_output_projection_kernel_seconds": 0.035,
                        "post_attn_norm_weight_bytes_read": 9,
                        "post_attn_norm_weight_read_seconds": 0.002,
                        "router_bytes_read": 25,
                        "router_correction_bias_bytes_read": 5,
                        "router_read_seconds": 0.003,
                        "router_kernel_seconds": 0.004,
                        "mlp_elapsed_seconds": 0.061,
                        "dense_mlp_elapsed_seconds": 0.071,
                        "moe_mlp_elapsed_seconds": 0.081,
                        "expert_read_seconds": 0.05,
                        "shared_read_seconds": 0.02,
                        "shared_prefetch_seconds": 0.018,
                        "shared_prefetch_used_count": 75,
                        "moe_mlp_kernel_seconds": 0.06,
                        "moe_mlp_output_write_seconds": 0.007,
                        "moe_mlp_overhead_seconds": 0.008,
                        "layer_overhead_seconds": 0.01,
                        "attn_projection_command_buffer_count": 3,
                        "attn_projection_synchronous_wait_count": 0,
                        "attn_projection_async_submitted_count": 78,
                        "rope_mla_command_buffer_count": 0,
                        "attn_output_command_buffer_count": 1,
                        "attn_output_context1_o_proj_cache_count": 78,
                        "attn_output_resident_mmap_backed_count": 78,
                        "post_attn_norm_command_buffer_count": 0,
                        "router_command_buffer_count": 0,
                        "post_attn_norm_router_command_buffer_count": 0,
                        "dense_mlp_command_buffer_count": 0,
                        "dense_mlp_synchronous_wait_count": 0,
                        "dense_mlp_async_submitted_count": 0,
                        "moe_mlp_command_buffer_count": 1,
                        "moe_mlp_synchronous_wait_count": 1,
                        "attn_output_norm_router_fused_count": 1,
                        "rope_mla_attn_output_norm_router_fused_count": 1,
                        "rope_mla_input_buffer_direct_count": 1,
                        "attn_output_buffer_direct_count": 1,
                        "moe_mlp_input_buffer_direct_count": 1,
                        "layer_input_buffer_direct_count": 1,
                        "command_buffer_count": 5,
                        "synchronous_wait_count_estimate": 5,
                        "expert_read_dispatch_count": 1,
                        "expert_read_task_count": 8,
                        "expert_read_max_task_count": 8,
                        "expert_read_max_worker_count": 8,
                        "expert_read_pool_dispatch_count": 1,
                        "expert_read_serial_dispatch_count": 0,
                        "mla_value_cache_hit_count": 2,
                        "mla_value_cache_store_count": 0,
                        "mla_value_cache_bytes": 64,
                        "mla_value_cache_total_bytes": 96,
                    },
                ],
            },
        }
        return payload, 0.5

    monkeypatch.setattr(metal_generate.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(metal_generate, "embed_token", fake_embed_token)
    monkeypatch.setattr(
        metal_generate,
        "_run_generate_server_jsonl_request",
        fake_run_generate_server_jsonl_request,
    )

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0],
        max_new_tokens=2,
        binary="/tmp/fake-glm_moe_infer",
        keep_work_dir=True,
        logits_top_k=1,
        cache_mla_kv_b_f32=True,
        max_mla_kv_b_cache_mib=512.0,
        context1_o_proj_cache_layout="/tmp/context1-layout.json",
        context1_o_proj_cache_file="/tmp/context1-cache.bin",
        quiet=True,
    )

    assert result.work_dir == kept_root
    assert result.work_dir.is_dir()
    assert result.kept_work_dir is True
    assert result.generated_token_ids == (5, 6)
    assert result.decode_elapsed_seconds == (1.25, 1.5)
    assert result.final_logits_elapsed_seconds == (0.1, 0.2)
    assert result.final_logits_bytes_read == (1024, 24)
    assert result.final_logits_lm_head_bytes_read == (1000, 0)
    assert result.final_logits_read_seconds == (0.01, 0.001)
    assert result.final_logits_kernel_seconds == (0.08, 0.18)
    assert result.final_logits_resident_mmap_backed == (False, True)
    assert result.decode_expert_bytes_read == (128, 256)
    assert result.decode_dense_mlp_bytes_read == (16, 32)
    assert result.decode_layer_count == (78, 78)
    assert result.decode_dense_layer_count == (3, 3)
    assert result.decode_moe_layer_count == (75, 75)
    assert result.decode_layer_elapsed_seconds == (1.2, 1.4)
    assert result.decode_attn_projection_elapsed_seconds == (0.11, 0.12)
    assert result.decode_mla_attention_elapsed_seconds == (0.22, 0.23)
    assert result.decode_mla_attention_cache_read_seconds == (0.01, 0.011)
    assert result.decode_mla_attention_value_read_seconds == (0.02, 0.021)
    assert result.decode_mla_attention_kernel_seconds == (0.03, 0.031)
    assert result.decode_mla_attention_output_write_seconds == (0.04, 0.041)
    assert result.decode_attn_output_elapsed_seconds == (0.05, 0.051)
    assert result.decode_attn_output_bytes_read == (64, 65)
    assert result.decode_attn_output_read_seconds == (0.012, 0.013)
    assert result.decode_attn_output_projection_kernel_seconds == (0.034, 0.035)
    assert result.decode_post_attn_norm_weight_bytes_read == (8, 9)
    assert result.decode_post_attn_norm_weight_read_seconds == (0.001, 0.002)
    assert result.decode_router_bytes_read == (24, 25)
    assert result.decode_router_correction_bias_bytes_read == (4, 5)
    assert result.decode_router_read_seconds == (0.002, 0.003)
    assert result.decode_router_kernel_seconds == (0.003, 0.004)
    assert result.decode_mlp_elapsed_seconds == (0.06, 0.061)
    assert result.decode_dense_mlp_elapsed_seconds == (0.07, 0.071)
    assert result.decode_moe_mlp_elapsed_seconds == (0.08, 0.081)
    assert result.decode_expert_read_seconds == (0.03, 0.05)
    assert result.decode_shared_bytes_read == (12, 24)
    assert result.decode_shared_read_seconds == (0.01, 0.02)
    assert result.decode_shared_prefetch_seconds == (0.009, 0.018)
    assert result.decode_shared_prefetch_used_count == (75, 75)
    assert result.decode_moe_mlp_kernel_seconds == (0.04, 0.06)
    assert result.decode_moe_mlp_output_write_seconds == (0.005, 0.007)
    assert result.decode_moe_mlp_overhead_seconds == (0.006, 0.008)
    assert result.decode_layer_overhead_seconds == (0.009, 0.01)
    assert result.decode_attn_projection_command_buffer_count == (3, 3)
    assert result.decode_attn_projection_synchronous_wait_count == (1, 0)
    assert result.decode_attn_projection_async_submitted_count == (77, 78)
    assert result.decode_rope_mla_command_buffer_count == (0, 0)
    assert result.decode_attn_output_command_buffer_count == (1, 1)
    assert result.decode_attn_output_context1_o_proj_cache_count == (78, 78)
    assert result.decode_attn_output_resident_mmap_backed_count == (77, 78)
    assert result.decode_post_attn_norm_command_buffer_count == (0, 0)
    assert result.decode_router_command_buffer_count == (0, 0)
    assert result.decode_post_attn_norm_router_command_buffer_count == (0, 0)
    assert result.decode_dense_mlp_command_buffer_count == (0, 0)
    assert result.decode_dense_mlp_synchronous_wait_count == (0, 0)
    assert result.decode_dense_mlp_async_submitted_count == (0, 0)
    assert result.decode_moe_mlp_command_buffer_count == (1, 1)
    assert result.decode_moe_mlp_synchronous_wait_count == (0, 1)
    assert result.decode_attn_output_norm_router_fused_count == (1, 1)
    assert result.decode_rope_mla_attn_output_norm_router_fused_count == (1, 1)
    assert result.decode_rope_mla_input_buffer_direct_count == (1, 1)
    assert result.decode_attn_output_buffer_direct_count == (1, 1)
    assert result.decode_moe_mlp_input_buffer_direct_count == (1, 1)
    assert result.decode_layer_input_buffer_direct_count == (1, 1)
    assert result.decode_command_buffer_count == (5, 5)
    assert result.decode_synchronous_wait_count_estimate == (5, 5)
    assert result.decode_expert_read_dispatch_count == (1, 1)
    assert result.decode_expert_read_task_count == (8, 8)
    assert result.decode_expert_read_max_task_count == (8, 8)
    assert result.decode_expert_read_max_worker_count == (8, 8)
    assert result.decode_expert_read_pool_dispatch_count == (1, 1)
    assert result.decode_expert_read_serial_dispatch_count == (0, 0)
    assert result.decode_mla_value_cache_hit_count == (0, 2)
    assert result.decode_mla_value_cache_store_count == (2, 0)
    assert result.decode_mla_value_cache_bytes == (64, 64)
    assert result.decode_mla_value_cache_total_bytes == (64, 96)
    assert result.mla_kv_b_cache_enabled is True
    assert result.mla_kv_b_cache_current_bytes == 96
    assert result.mla_kv_b_cache_live_estimate_bytes == 512 * 1024 * 1024
    assert not result.generated_json_path.exists()
    assert result.expert_buffer_count_allocated == 1
    assert result.min_free_unified_memory_gib == 0.0


def test_generate_metal_token_ids_reuses_generate_server_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "metal-generate"

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("sparse", "sparse"),
    )

    class FakeSession:
        prepared_dir = prepared
        request_count = 0

        def request(self, request):
            self.request_count += 1
            assert request["input_token_id"] == 0
            assert request["in_memory_decode_cache"] is True
            payload = {
                "estimated_live_working_set_bytes": 1234,
                "expert_buffer_count_runtime_allocated": 1,
                "probe_generate": {
                    "ok": True,
                    "entry": "generate_token_ids",
                    "generated_token_ids": [5],
                    "steps": [
                        {
                            "decode_elapsed_seconds": 1.25,
                            "final_logits_elapsed_seconds": 0.1,
                        },
                    ],
                },
            }
            return payload, 0.25

    def fail_server(**kwargs):
        raise AssertionError("temporary JSONL server should not be launched")

    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)
    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        lambda *args, **kwargs: _FakeLayout(),
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=64),
    )
    monkeypatch.setattr(
        metal_generate,
        "_run_generate_server_jsonl_request",
        fail_server,
    )
    session = FakeSession()

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0],
        max_new_tokens=1,
        binary="/tmp/fake-glm_moe_infer",
        work_dir=root,
        logits_top_k=1,
        generate_server_session=session,
        quiet=True,
    )

    assert result.generated_token_ids == (5,)
    assert result.metal_elapsed_seconds == 0.25
    assert result.expert_buffer_count_allocated == 1
    assert session.request_count == 1


def test_generate_metal_token_ids_can_use_legacy_request_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "metal-generate"
    captured: dict[str, object] = {}

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("sparse", "sparse"),
    )

    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)
    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        lambda *args, **kwargs: _FakeLayout(),
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=64),
    )

    def fail_server(**kwargs):
        raise AssertionError("JSONL server should not be used")

    def fake_run_command(cmd, *, quiet: bool):
        captured["cmd"] = tuple(cmd)
        request_path = Path(cmd[cmd.index("--generate-request-json") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        captured["request"] = request
        Path(request["output_generated_json"]).write_text(
            '{"step_count":1}',
            encoding="utf-8",
        )
        payload = {
            "estimated_live_working_set_bytes": 1234,
            "expert_buffer_count_allocated": 1,
            "probe_generate": {
                "ok": True,
                "entry": "generate_token_ids",
                "generated_token_ids": [5],
                "steps": [
                    {
                        "decode_elapsed_seconds": 1.25,
                        "final_logits_elapsed_seconds": 0.1,
                    },
                ],
            },
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        metal_generate,
        "_run_generate_server_jsonl_request",
        fail_server,
    )
    monkeypatch.setattr(metal_generate, "_run_command", fake_run_command)

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0],
        max_new_tokens=1,
        binary="/tmp/fake-glm_moe_infer",
        work_dir=root,
        logits_top_k=1,
        use_generate_server_jsonl=False,
        quiet=True,
    )

    assert "--generate-request-json" in captured["cmd"]
    request = captured["request"]
    assert request["output_f32"] == str(root / "output.f32")
    assert request["output_generated_json"] == str(root / "generated.json")
    assert request["output_next_input_f32"] == str(root / "next_input.f32")
    assert request["in_memory_decode_cache"] is False
    assert result.generated_token_ids == (5,)
    assert result.generated_json_path.exists()


def test_generate_metal_token_ids_rejects_multi_token_prompt_by_default(
    tmp_path: Path,
) -> None:
    with pytest.raises(MetalGenerateError, match="decode-only Metal loop"):
        generate_metal_token_ids(
            tmp_path / "prepared",
            prompt_token_ids=[0, 1],
            max_new_tokens=1,
        )


def test_generate_metal_token_ids_requires_mla_kv_b_cache_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("sparse",),
    )
    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)

    with pytest.raises(MetalGenerateError, match="max_mla_kv_b_cache_mib"):
        generate_metal_token_ids(
            prepared,
            prompt_token_ids=[0],
            max_new_tokens=1,
            logits_top_k=1,
            cache_mla_kv_b_f32=True,
        )


def test_generate_metal_token_ids_rejects_context1_cache_file_without_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("sparse",),
    )
    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)

    with pytest.raises(MetalGenerateError, match="context1_o_proj_cache_file"):
        generate_metal_token_ids(
            prepared,
            prompt_token_ids=[0],
            max_new_tokens=1,
            logits_top_k=1,
            context1_o_proj_cache_file="/tmp/context1-cache.bin",
        )


def test_generate_metal_token_ids_prefill_prompt_defaults_to_runtime_jsonl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "metal-generate"
    captured: dict[str, object] = {}

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("dense", "sparse"),
        scoring_func="raw",
    )

    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)
    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        lambda *args, **kwargs: _FakeLayout(),
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=128),
    )

    def fail_prefill(**kwargs):
        raise AssertionError("Python prefill bridge should not be used")

    def fake_run_generate_server_jsonl_request(*, binary, prepared_dir, request, quiet):
        captured["server_binary"] = str(binary)
        captured["server_prepared_dir"] = str(prepared_dir)
        captured["request"] = request
        payload = {
            "estimated_live_working_set_bytes": 2048,
            "expert_buffer_count_runtime_allocated": 2,
            "probe_generate": {
                "ok": True,
                "entry": "generate_token_ids",
                "first_step_from_input_logits": True,
                "generated_token_ids": [7, 8],
                "prompt_prefill": {
                    "ok": True,
                    "token_count": 2,
                    "steps": [
                        {"decode_elapsed_seconds": 1.0},
                        {"decode_elapsed_seconds": 1.25},
                    ],
                },
                "steps": [
                    {
                        "decode_elapsed_seconds": 0.0,
                        "final_logits_elapsed_seconds": 0.25,
                    },
                    {
                        "decode_elapsed_seconds": 1.5,
                        "final_logits_elapsed_seconds": 0.2,
                    },
                ],
            },
        }
        return payload, 3.0

    monkeypatch.setattr(metal_generate, "run_prompt_prefill", fail_prefill)
    monkeypatch.setattr(
        metal_generate,
        "_run_generate_server_jsonl_request",
        fake_run_generate_server_jsonl_request,
    )

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0, 1],
        max_new_tokens=2,
        binary="/tmp/glm_moe_infer",
        work_dir=root,
        prefill_prompt=True,
        logits_top_k=1,
        quiet=True,
    )

    assert captured["server_binary"] == "/tmp/glm_moe_infer"
    assert captured["server_prepared_dir"] == str(prepared)
    request = captured["request"]
    assert request["prompt_token_ids"] == [0, 1]
    assert "input_f32" not in request
    assert "input_token_id" not in request
    assert "output_f32" not in request
    assert request["generate_steps"] == 2
    assert request["position"] == 0
    assert request["context_length"] == 1
    assert request["expert_buffer_count"] == 2
    assert request["include_shared_expert"] is True
    assert request["in_memory_decode_cache"] is True
    assert result.generated_token_ids == (7, 8)
    assert result.prompt_prefill is None
    assert result.prompt_prefill_elapsed_seconds == 2.25
    assert result.expert_buffer_count_allocated == 2
    assert "runtime prompt prefill path" in result.note


def test_generate_metal_token_ids_can_bridge_multi_token_prompt_prefill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "metal-generate"
    captured: dict[str, object] = {}

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        experts_layout=tmp_path / "experts" / "layout.json",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("dense", "sparse"),
        scoring_func="raw",
    )

    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)

    def fake_build_decode_cache_layout(*args, **kwargs):
        captured["max_context_tokens"] = kwargs["max_context_tokens"]
        return _FakeLayout()

    def fake_run_prompt_prefill(**kwargs):
        captured["prefill_kwargs"] = kwargs
        Path(kwargs["output_last_hidden_f32_path"]).write_bytes(b"\0" * 16)
        return SimpleNamespace(
            elapsed_seconds=3.0,
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=4096,
            ),
        )

    def fake_run_command(cmd, *, quiet: bool):
        captured["final_logits_cmd"] = tuple(cmd)
        payload = {
            "estimated_live_working_set_bytes": 2048,
            "probe_final_logits": {
                "ok": True,
                "elapsed_seconds": 0.25,
                "generated_token": {"token_id": 7},
            },
        }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        fake_build_decode_cache_layout,
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=128),
    )
    monkeypatch.setattr(metal_generate, "run_prompt_prefill", fake_run_prompt_prefill)
    monkeypatch.setattr(metal_generate, "_run_command", fake_run_command)

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0, 1],
        max_new_tokens=1,
        binary="/tmp/glm_moe_infer",
        work_dir=root,
        prefill_prompt=True,
        prefill_runner="/tmp/largerlm-runner",
        prefill_max_live_working_set_mib=2048.0,
        min_free_unified_memory_gib=5.0,
        use_python_prefill_bridge=True,
        logits_top_k=1,
        quiet=True,
    )

    assert captured["max_context_tokens"] == 2
    prefill_kwargs = captured["prefill_kwargs"]
    assert prefill_kwargs["prompt_token_ids"] == (0, 1)
    assert prefill_kwargs["layers"] == range(0, 2)
    assert prefill_kwargs["dense_layers"] == (0,)
    assert prefill_kwargs["runner_path"] == "/tmp/largerlm-runner"
    assert prefill_kwargs["max_live_working_set_mib"] == 2048.0
    assert prefill_kwargs["min_free_unified_memory_gib"] == 5.0
    assert "--probe-final-logits" in captured["final_logits_cmd"]
    final_logits_cmd = captured["final_logits_cmd"]
    assert "--min-free-unified-memory-gib" in final_logits_cmd
    assert (
        final_logits_cmd[
            final_logits_cmd.index("--min-free-unified-memory-gib") + 1
        ]
        == "5"
    )
    assert result.generated_token_ids == (7,)
    assert result.decode_elapsed_seconds == (0.0,)
    assert result.final_logits_elapsed_seconds == (0.25,)
    assert result.prompt_prefill is not None
    assert result.prompt_prefill_elapsed_seconds == 3.0
    assert result.prompt_prefill_estimated_live_working_set_bytes == 4096
    assert result.prefill_max_live_working_set_mib == 2048.0
    assert result.min_free_unified_memory_gib == 5.0
    assert result.metal_elapsed_seconds is not None
    assert result.metal_elapsed_seconds < result.elapsed_seconds
    assert result.prefill_final_logits_elapsed_seconds == 0.25
    assert "prefill bridge path" in result.note


def test_generate_metal_token_ids_prefill_bridge_continues_in_one_metal_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "metal-generate"
    captured: dict[str, object] = {}

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        experts_layout=tmp_path / "experts" / "layout.json",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("dense", "sparse"),
        scoring_func="raw",
    )

    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)
    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        lambda *args, **kwargs: _FakeLayout(),
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=128),
    )

    def fake_run_prompt_prefill(**kwargs):
        captured["prefill_kwargs"] = kwargs
        Path(kwargs["output_last_hidden_f32_path"]).write_bytes(b"\0" * 16)
        return SimpleNamespace(
            elapsed_seconds=3.0,
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=4096,
            ),
        )

    def fake_run_generate_server_jsonl_request(*, binary, prepared_dir, request, quiet):
        captured["server_binary"] = str(binary)
        captured["server_prepared_dir"] = str(prepared_dir)
        captured["request"] = request
        payload = {
            "estimated_live_working_set_bytes": 2048,
            "decode_cache_backend": "memory",
            "decode_cache_memory_loaded": 1,
            "probe_generate": {
                "ok": True,
                "entry": "generate_token_ids",
                "first_step_from_input_logits": True,
                "generated_token_ids": [7, 8],
                "steps": [
                    {
                        "decode_elapsed_seconds": 0.0,
                        "final_logits_elapsed_seconds": 0.25,
                    },
                    {
                        "decode_elapsed_seconds": 1.5,
                        "final_logits_elapsed_seconds": 0.2,
                    },
                ],
            },
        }
        return payload, 0.5

    monkeypatch.setattr(metal_generate, "run_prompt_prefill", fake_run_prompt_prefill)
    monkeypatch.setattr(
        metal_generate,
        "_run_generate_server_jsonl_request",
        fake_run_generate_server_jsonl_request,
    )

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0, 1],
        max_new_tokens=2,
        binary="/tmp/glm_moe_infer",
        work_dir=root,
        prefill_prompt=True,
        use_python_prefill_bridge=True,
        logits_top_k=1,
        quiet=True,
    )

    assert captured["server_binary"] == "/tmp/glm_moe_infer"
    assert captured["server_prepared_dir"] == str(prepared)
    request = captured["request"]
    assert captured["prefill_kwargs"]["min_free_unified_memory_gib"] == 0.0
    assert request["input_f32"] == str(root / "prefill_last_hidden.f32")
    assert "input_token_id" not in request
    assert "output_f32" not in request
    assert "output_generated_json" not in request
    assert "output_next_input_f32" not in request
    assert request["generate_first_from_input_logits"] is True
    assert request["generate_steps"] == 2
    assert request["position"] == 2
    assert request["context_length"] == 3
    assert request["expert_buffer_count"] == 2
    assert request["include_shared_expert"] is True
    assert request["in_memory_decode_cache"] is True
    assert request["min_free_unified_memory_gib"] == 0.0
    assert result.generated_token_ids == (7, 8)
    assert result.decode_elapsed_seconds == (0.0, 1.5)
    assert result.final_logits_elapsed_seconds == (0.25, 0.2)
    assert result.prefill_final_logits_elapsed_seconds == 0.25
    assert result.metal_elapsed_seconds is not None
    assert result.metal_elapsed_seconds < result.elapsed_seconds
    assert "persistent glm_moe_infer JSONL service" in result.note


def test_generate_metal_token_ids_defaults_min_free_from_prepared_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    root = tmp_path / "metal-generate"
    captured: dict[str, object] = {}

    manifest = SimpleNamespace(
        manifest_path=prepared / "manifest.json",
        model_dir=tmp_path / "model",
        resident_layout=tmp_path / "resident" / "layout.json",
        max_context_tokens=8,
        recommended_min_free_unified_memory_bytes=3 * 1024**3,
    )
    config = ModelConfig(
        model_type="glm-test",
        hidden_size=4,
        num_hidden_layers=2,
        vocab_size=16,
        n_shared_experts=0,
        num_experts_per_tok=1,
        num_attention_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        qk_rope_head_dim=1,
        v_head_dim=1,
        mlp_layer_types=("sparse", "sparse"),
    )

    monkeypatch.setattr(metal_generate, "load_prepared_manifest", lambda path: manifest)
    monkeypatch.setattr(metal_generate, "load_config", lambda path: config)
    monkeypatch.setattr(
        metal_generate,
        "build_decode_cache_layout",
        lambda *args, **kwargs: _FakeLayout(),
    )
    monkeypatch.setattr(
        metal_generate,
        "init_decode_cache_file",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=64),
    )

    def fake_run_generate_server_jsonl_request(*, binary, prepared_dir, request, quiet):
        captured["request"] = request
        payload = {
            "estimated_live_working_set_bytes": 1234,
            "admission_ok": 1,
            "available_unified_memory_ok": 1,
            "system_available_memory_bytes": 10 * 1024**3,
            "required_available_memory_bytes": 4 * 1024**3,
            "expert_buffer_count_runtime_allocated": 1,
            "probe_generate": {
                "ok": True,
                "entry": "generate_token_ids",
                "generated_token_ids": [5],
                "steps": [
                    {
                        "decode_elapsed_seconds": 1.25,
                        "final_logits_elapsed_seconds": 0.1,
                    },
                ],
            },
        }
        return payload, 0.5

    monkeypatch.setattr(
        metal_generate,
        "_run_generate_server_jsonl_request",
        fake_run_generate_server_jsonl_request,
    )

    result = generate_metal_token_ids(
        prepared,
        prompt_token_ids=[0],
        max_new_tokens=1,
        binary="/tmp/fake-glm_moe_infer",
        work_dir=root,
        logits_top_k=1,
        quiet=True,
    )

    assert captured["request"]["min_free_unified_memory_gib"] == 3.0
    assert result.min_free_unified_memory_gib == 3.0
    assert result.admission_ok is True
    assert result.available_unified_memory_ok is True
    assert result.system_available_memory_bytes == 10 * 1024**3
    assert result.required_available_memory_bytes == 4 * 1024**3
    assert result.expert_buffer_count_allocated == 1
