from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from largerlm.generation_guard import GenerationGuardError
from largerlm.layout import config_sha256
from largerlm.config import load_config
from largerlm.context1_o_proj_cache import Context1OProjCacheError
from largerlm.server import (
    PreparedGenerationApp,
    PreparedRequestCheckError,
    PreparedServerConfig,
    PreparedServerError,
    _PreparedHTTPServer,
    _decode_layer_record_payload,
    _estimate_mla_kv_b_f32_cache_plan,
    _is_public_glm_5_2_shape,
    _prefill_linear_backend_request_summary,
    _public_glm_5_2_shape_report,
    combine_suggested_launch_profile,
)
from largerlm.decode_driver import DecodeLayerRecord
from largerlm.text_generator import TextGenerationResult
from largerlm.metal_generate import MetalGenerateError, MetalTokenGenerationResult
from largerlm.metal_text_generator import MetalTextGenerationResult
from largerlm.token_generator import (
    AutoPrefillChunkCap,
    AutoPrefillPromptChunkPlan,
    TokenGenerationResult,
    TokenGeneratorError,
)
from largerlm.tokenizer import RenderedChatPrompt
from test_prepare import (
    _add_prepared_memory_profile,
    _add_large_bf16_prefill_matrix,
    _make_prepared_glm_4bit_ready,
    _mock_safe_system_memory,
    _set_prepared_context,
    _write_simple_model_tokenizer,
    _write_minimal_prepared_manifest,
    _write_mxfp4_glm_config,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _fake_context1_layout(
    layout_path: Path,
    *,
    default_cache_file: Path,
    total_bytes: int = 16,
    layers: tuple[int, ...] = (0,),
):
    return SimpleNamespace(
        layout_path=layout_path,
        cache_file_path=default_cache_file,
        dtype="BF16",
        dtype_bytes=2,
        hidden_dim=4,
        attention_value_dim=2,
        kv_lora_dim=2,
        total_bytes=total_bytes,
        layers=layers,
        to_json=lambda: {
            "schema": "largerlm.context1_o_proj_bv_cache.v1",
            "layout_path": str(layout_path),
            "cache_file": str(default_cache_file),
            "dtype": "BF16",
            "dtype_bytes": 2,
            "total_bytes": total_bytes,
            "layers": list(layers),
            "tensor_count": len(layers),
        },
    )


def _set_layout_config_sha256(layout_path: Path, digest: str) -> None:
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["config_sha256"] = digest
    layout_path.write_text(json.dumps(payload), encoding="utf-8")


def _token_result(
    tmp_path: Path,
    prompt: tuple[int, ...],
    *,
    prompt_prefill: object | None = None,
    auto_prefill_prompt_chunk_plan: AutoPrefillPromptChunkPlan | None = None,
    max_safe_prefill_prompt_chunk_plan: AutoPrefillPromptChunkPlan | None = None,
    prefill_prompt_chunk_plan_drift: dict[str, object] | None = None,
    prefill_actual_read_time: dict[str, object] | None = None,
) -> TokenGenerationResult:
    return TokenGenerationResult(
        prompt_token_ids=prompt,
        generated_token_ids=(2,),
        steps=(),
        work_dir=tmp_path,
        kept_work_dir=False,
        max_context_tokens=4,
        sampling_temperature=0.0,
        sampling_top_p=1.0,
        elapsed_seconds=0.5,
        estimated_read_bytes=123,
        prompt_prefill=prompt_prefill,
        auto_prefill_prompt_chunk_plan=auto_prefill_prompt_chunk_plan,
        max_safe_prefill_prompt_chunk_plan=max_safe_prefill_prompt_chunk_plan,
        prefill_prompt_chunk_plan_drift=prefill_prompt_chunk_plan_drift,
        prefill_actual_read_time=prefill_actual_read_time,
    )


def _metal_token_result(
    tmp_path: Path,
    prompt: tuple[int, ...],
    *,
    generated: tuple[int, ...] = (2,),
    mla_kv_b_cache: bool | None = None,
) -> MetalTokenGenerationResult:
    return MetalTokenGenerationResult(
        prepared_dir=tmp_path / "prepared",
        binary=tmp_path / "glm_moe_infer",
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        work_dir=tmp_path / "metal-work",
        kept_work_dir=False,
        cache_layout_path=tmp_path / "decode_cache_layout.json",
        cache_file_path=tmp_path / "decode_cache.bin",
        input_f32_path=tmp_path / "input.f32",
        output_f32_path=tmp_path / "output.f32",
        generated_json_path=tmp_path / "generated.json",
        elapsed_seconds=1.5,
        decode_elapsed_seconds=(0.0, 0.5)[: len(generated)],
        final_logits_elapsed_seconds=(0.1, 0.2)[: len(generated)],
        final_logits_bytes_read=(1024, 24)[: len(generated)],
        final_logits_lm_head_bytes_read=(1000, 0)[: len(generated)],
        final_logits_read_seconds=(0.01, 0.001)[: len(generated)],
        final_logits_kernel_seconds=(0.08, 0.18)[: len(generated)],
        final_logits_resident_mmap_backed=(False, True)[: len(generated)],
        decode_layer_count=(78, 78)[: len(generated)],
        decode_dense_layer_count=(3, 3)[: len(generated)],
        decode_moe_layer_count=(75, 75)[: len(generated)],
        decode_layer_elapsed_seconds=(0.9, 1.4)[: len(generated)],
        decode_attn_projection_elapsed_seconds=(0.11, 0.12)[: len(generated)],
        decode_mla_attention_elapsed_seconds=(0.21, 0.22)[: len(generated)],
        decode_mla_attention_cache_read_seconds=(0.01, 0.011)[: len(generated)],
        decode_mla_attention_value_read_seconds=(0.02, 0.021)[: len(generated)],
        decode_mla_attention_kernel_seconds=(0.03, 0.031)[: len(generated)],
        decode_mla_attention_output_write_seconds=(0.04, 0.041)[
            : len(generated)
        ],
        decode_attn_output_elapsed_seconds=(0.05, 0.051)[: len(generated)],
        decode_attn_output_bytes_read=(64, 65)[: len(generated)],
        decode_attn_output_read_seconds=(0.012, 0.013)[: len(generated)],
        decode_attn_output_projection_kernel_seconds=(0.034, 0.035)[
            : len(generated)
        ],
        decode_post_attn_norm_weight_bytes_read=(8, 9)[: len(generated)],
        decode_post_attn_norm_weight_read_seconds=(0.001, 0.002)[
            : len(generated)
        ],
        decode_router_bytes_read=(24, 25)[: len(generated)],
        decode_router_correction_bias_bytes_read=(4, 5)[: len(generated)],
        decode_router_read_seconds=(0.002, 0.003)[: len(generated)],
        decode_router_kernel_seconds=(0.003, 0.004)[: len(generated)],
        decode_mlp_elapsed_seconds=(0.06, 0.061)[: len(generated)],
        decode_dense_mlp_elapsed_seconds=(0.07, 0.071)[: len(generated)],
        decode_moe_mlp_elapsed_seconds=(0.08, 0.081)[: len(generated)],
        decode_shared_bytes_read=(12, 24)[: len(generated)],
        decode_shared_read_seconds=(0.01, 0.02)[: len(generated)],
        decode_shared_prefetch_seconds=(0.009, 0.018)[: len(generated)],
        decode_shared_prefetch_used_count=(75, 75)[: len(generated)],
        decode_moe_mlp_output_write_seconds=(0.005, 0.006)[: len(generated)],
        decode_moe_mlp_overhead_seconds=(0.015, 0.016)[: len(generated)],
        decode_layer_overhead_seconds=(0.025, 0.026)[: len(generated)],
        decode_attn_projection_command_buffer_count=(3, 3)[: len(generated)],
        decode_attn_projection_synchronous_wait_count=(1, 0)[: len(generated)],
        decode_attn_projection_async_submitted_count=(77, 78)[: len(generated)],
        decode_rope_mla_command_buffer_count=(0, 0)[: len(generated)],
        decode_attn_output_command_buffer_count=(1, 1)[: len(generated)],
        decode_attn_output_context1_o_proj_cache_count=(78, 78)[: len(generated)],
        decode_attn_output_resident_mmap_backed_count=(77, 78)[: len(generated)],
        decode_post_attn_norm_command_buffer_count=(0, 0)[: len(generated)],
        decode_router_command_buffer_count=(0, 0)[: len(generated)],
        decode_post_attn_norm_router_command_buffer_count=(0, 0)[: len(generated)],
        decode_dense_mlp_command_buffer_count=(0, 0)[: len(generated)],
        decode_dense_mlp_synchronous_wait_count=(0, 0)[: len(generated)],
        decode_dense_mlp_async_submitted_count=(0, 0)[: len(generated)],
        decode_moe_mlp_command_buffer_count=(1, 1)[: len(generated)],
        decode_moe_mlp_synchronous_wait_count=(0, 1)[: len(generated)],
        decode_attn_output_norm_router_fused_count=(1, 1)[: len(generated)],
        decode_rope_mla_attn_output_norm_router_fused_count=(1, 1)[
            : len(generated)
        ],
        decode_rope_mla_input_buffer_direct_count=(1, 1)[: len(generated)],
        decode_attn_output_buffer_direct_count=(1, 1)[: len(generated)],
        decode_moe_mlp_input_buffer_direct_count=(1, 1)[: len(generated)],
        decode_layer_input_buffer_direct_count=(1, 1)[: len(generated)],
        decode_command_buffer_count=(5, 5)[: len(generated)],
        decode_synchronous_wait_count_estimate=(5, 5)[: len(generated)],
        estimated_live_working_set_bytes=2048,
        max_live_working_set_mib=768,
        cache_total_bytes=64,
        note="runtime prompt prefill path",
        min_free_unified_memory_gib=1.0,
        admission_ok=True,
        available_unified_memory_ok=True,
        system_available_memory_bytes=10_000,
        required_available_memory_bytes=2_000,
        expert_buffer_count_allocated=2,
        decode_mla_value_cache_hit_count=(0, 2)[: len(generated)],
        decode_mla_value_cache_store_count=(2, 0)[: len(generated)],
        decode_mla_value_cache_bytes=(64, 64)[: len(generated)],
        decode_mla_value_cache_total_bytes=(64, 96)[: len(generated)],
        mla_kv_b_cache_enabled=mla_kv_b_cache,
        mla_kv_b_cache_current_bytes=96 if mla_kv_b_cache else 0,
        mla_kv_b_cache_live_estimate_bytes=512 * 1024**2
        if mla_kv_b_cache
        else 0,
        prompt_prefill_elapsed_seconds=0.25 if len(prompt) > 1 else None,
        metal_elapsed_seconds=1.5,
        prefill_final_logits_elapsed_seconds=0.1 if len(prompt) > 1 else None,
    )


class _FakeMetalGenerateServerSession:
    instances: list["_FakeMetalGenerateServerSession"] = []

    def __init__(
        self,
        *,
        binary: Path,
        prepared_dir: Path,
        expert_pin_plan: Path | None = None,
        max_adaptive_expert_cache_gib: float = 0.0,
        quiet: bool = True,
    ) -> None:
        self.binary = Path(binary)
        self.prepared_dir = Path(prepared_dir)
        self.expert_pin_plan = expert_pin_plan
        self.max_adaptive_expert_cache_gib = max_adaptive_expert_cache_gib
        self.quiet = quiet
        self.closed = False
        self.request_count = 0
        _FakeMetalGenerateServerSession.instances.append(self)

    def close(self) -> None:
        self.closed = True


def test_decode_layer_payload_reports_mla_kv_b_cache_flag(tmp_path: Path) -> None:
    cache_dir = tmp_path / "mla-kv-b-cache"
    record = DecodeLayerRecord(
        layer=3,
        kind="moe",
        input_path=tmp_path / "in.f32",
        output_path=tmp_path / "out.f32",
        command=(
            "metal/largerlm-runner",
            "--run-decoder-layers",
            "--mla-kv-b-cache-dir",
            str(cache_dir),
        ),
    )

    payload = _decode_layer_record_payload(record)

    assert payload["command_has_mla_kv_b_cache_dir"] is True
    assert payload["command_mla_kv_b_cache_dir"] == str(cache_dir)


def test_prepared_server_passes_expert_cache_to_persistent_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    expert_pin_plan = tmp_path / "expert-pin-plan.json"
    expert_pin_plan.write_text("{}", encoding="utf-8")
    _FakeMetalGenerateServerSession.instances = []
    monkeypatch.setattr(
        "largerlm.server.MetalGenerateServerSession",
        _FakeMetalGenerateServerSession,
    )

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            metal_runtime_generation=True,
            metal_runtime_expert_pin_plan=expert_pin_plan,
            metal_runtime_max_adaptive_expert_cache_gib=36.0,
        )
    )
    try:
        session = app._metal_runtime_session()
        health = app.health()

        assert session.expert_pin_plan == expert_pin_plan.resolve()
        assert session.max_adaptive_expert_cache_gib == 36.0
        assert health["metal_runtime_expert_pin_plan"] == str(
            expert_pin_plan.resolve()
        )
        assert health["metal_runtime_max_adaptive_expert_cache_gib"] == 36.0
    finally:
        app.close()


def _replace_kv_b_with_absorbed_aliases(prepared: Path, *, mxfp4: bool) -> None:
    cfg = load_config(prepared.parent / "model")
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    tensors = resident_layout["tensors"]
    assert isinstance(tensors, list)
    tensors[:] = [
        tensor
        for tensor in tensors
        if not tensor["name"].endswith(".self_attn.kv_b_proj.weight")
    ]
    offset = int(resident_layout["total_bytes"])
    heads = int(cfg.num_attention_heads or 0)
    kv_lora = int(cfg.kv_lora_rank or 0)
    nope = int(cfg.qk_nope_head_dim or 0)
    value = int(cfg.v_head_dim or 0)

    def add_tensor(name: str, dtype: str, shape: list[int]) -> None:
        nonlocal offset
        size = 4 if dtype == "U32" else 1 if dtype == "U8" else 4
        for dim in shape:
            size *= dim
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": size,
                "dtype": dtype,
                "shape": shape,
                "category": "attention",
            }
        )
        offset += size

    for layer in range(int(cfg.num_hidden_layers)):
        prefix = f"model.layers.{layer}.self_attn"
        if mxfp4:
            add_tensor(f"{prefix}.embed_q.weight", "U32", [heads, kv_lora, nope // 8])
            add_tensor(f"{prefix}.embed_q.scales", "U8", [heads, kv_lora, nope // 32])
            add_tensor(
                f"{prefix}.unembed_out.weight",
                "U32",
                [heads, value, kv_lora // 8],
            )
            add_tensor(f"{prefix}.unembed_out.scales", "U8", [heads, value, kv_lora // 32])
        else:
            add_tensor(f"{prefix}.embed_q.weight", "F32", [heads, kv_lora, nope])
            add_tensor(f"{prefix}.unembed_out.weight", "F32", [heads, value, kv_lora])

    resident_layout["total_bytes"] = offset
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")
    (prepared / "resident" / "resident.bin").write_bytes(b"\0" * offset)


def _applied_launch_profile(tmp_path: Path) -> dict[str, object]:
    return {
        "path": str(tmp_path / "launch-profile.json"),
        "sha256": "a" * 64,
        "source": "unit",
        "matches_prepared": True,
        "prefill_prompt_chunk_plan": {
            "source": "prepared_request_check",
            "auto": {"prompt_tokens": 2, "chunk_tokens": 2},
            "max_safe": {"prompt_tokens": 2, "chunk_tokens": 2},
        },
    }


def _server_launch_audit_envelope(
    tmp_path: Path,
    *,
    prompt_tokens: int = 2,
    max_new_tokens: int = 1,
) -> dict[str, object]:
    return {
        "schema": "largerlm.launch_audit_server_envelope.v1",
        "artifact_path": str(tmp_path / "launch-audit.json"),
        "artifact_source": "unit",
        "applied_launch_profile_sha256": "a" * 64,
        "audited_prompt_token_count": prompt_tokens,
        "audited_max_new_tokens": max_new_tokens,
        "audited_required_context_tokens": prompt_tokens + max_new_tokens,
        "server_max_prompt_tokens": prompt_tokens,
        "server_max_new_tokens_cap": max_new_tokens,
        "server_caps_within_envelope": True,
    }


def _fake_auto_prefill_chunk_plan(chunk_tokens: int) -> AutoPrefillPromptChunkPlan:
    cap = AutoPrefillChunkCap(name="test_cap", tokens=chunk_tokens)
    return AutoPrefillPromptChunkPlan(
        prompt_tokens=chunk_tokens,
        start_position=0,
        raw_tokens=chunk_tokens,
        chunk_tokens=chunk_tokens,
        tile_tokens=1,
        limiting_cap_tokens=chunk_tokens,
        limiting_caps=(cap,),
        caps=(cap,),
        hidden_dim=8,
        per_token_activation_bytes=32,
        max_matrix_scratch_bytes=2 * 1024 * 1024,
        next_token_matrix_scratch_bytes=None,
        usable_disk_bytes=1024**3,
        per_token_disk_bytes=32,
    )


def _fake_runtime_guard(**kwargs):
    requested = int(kwargs.get("requested_context_tokens", 1))
    live = SimpleNamespace(
        estimated_live_working_set_bytes=4096,
        max_live_working_set_bytes=kwargs.get("max_live_working_set_mib")
        and int(float(kwargs["max_live_working_set_mib"]) * 1024**2),
        min_available_memory_bytes=int(
            float(kwargs.get("min_free_unified_memory_mib", 0.0)) * 1024**2
        ),
        system_available_bytes=128 * 1024**3,
        system_total_bytes=128 * 1024**3,
        system_source="test",
    )
    return SimpleNamespace(
        requested_context_tokens=requested,
        layers=(0,),
        dense_layers=(),
        max_layer_peak_bytes=2048,
        max_layer_cache_read_bytes=24,
        read_bytes_per_token=384,
        final_logits_budget=SimpleNamespace(estimated_peak_bytes=512),
        embedding_budget=SimpleNamespace(row_bytes=16, output_bytes=16),
        live_memory_budget=live,
    )


def _mpsgraph_probe_ready_backend(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "sdk_path": tmp_path / "MacOSX.sdk",
        "recommended_backend": "mpsgraph_prefill_fallback",
        "host_probe_requested": True,
        "host_probe_path": tmp_path / "prefill-backend-probe",
        "host_probe_ran": True,
        "host_probe_ok": True,
        "host_probe_error": None,
        "probe_timeout_seconds": 5.0,
        "mps_graph_runtime_available": True,
        "mps_graph_matmul_declared": True,
        "mps_graph_probe_requested": True,
        "mps_graph_probe_ran": True,
        "mps_graph_probe_ok": True,
        "mps_graph_probe_error": None,
        "metal4_ml_runtime_available": False,
        "mpp_tensor_ops_symbol_declared": False,
        "mpp_runtime_available": False,
        "mpp_compile_probe_requested": False,
        "mpp_compile_probe_ran": False,
        "mpp_compile_probe_ok": None,
        "reasons": (),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


@pytest.fixture(autouse=True)
def _default_server_runtime_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("largerlm.server.check_generation_runtime", _fake_runtime_guard)


def test_combine_suggested_launch_profile_deduplicates_and_marks_conflicts() -> None:
    launch = {
        "source": "launch",
        "argv": (
            "--max-live-working-set-mib",
            "40960",
            "--prefill-ssd-read-gib-s",
            "14",
        ),
    }
    prefill = {
        "source": "prefill",
        "argv": (
            "--prefill-prompt-chunk-tokens",
            "2240",
            "--prefill-ssd-read-gib-s",
            "14",
        ),
    }
    decode = {
        "source": "decode",
        "argv": (
            "--decode-max-routed-read-gib-per-token",
            "1.2",
            "--prefill-ssd-read-gib-s",
            "16",
        ),
    }
    accel = {
        "source": "accel",
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }
    probe = {
        "source": "probe",
        "argv": ("--run-mpsgraph-probe",),
    }

    profile = combine_suggested_launch_profile(
        launch_guard_flags=launch,
        prefill_backend_probe_flags=probe,
        prefill_acceleration_flags=accel,
        prefill_guard_flags=prefill,
        decode_guard_flags=decode,
        source="unit",
    )

    assert profile is not None
    assert profile["source"] == "unit"
    assert profile["argv_safe_to_replay"] is False
    assert profile["sections"]["launch_guard_flags"] == launch
    assert profile["sections"]["prefill_backend_probe_flags"] == probe
    assert profile["sections"]["prefill_acceleration_flags"] == accel
    assert profile["sections"]["prefill_guard_flags"] == prefill
    assert profile["sections"]["decode_guard_flags"] == decode
    assert profile["argv"].count("--prefill-ssd-read-gib-s") == 1
    assert profile["argv"].count("--run-mpsgraph-probe") == 1
    assert "--require-prefill-acceleration" in profile["argv"]
    assert profile["argv_conflicts"] == (
        {
            "section": "decode_guard_flags",
            "flag": "--prefill-ssd-read-gib-s",
            "kept_value": "14",
            "dropped_value": "16",
        },
    )


def test_prepared_server_http_models_endpoint(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            served_model_name="glm-http-local",
        )
    )
    try:
        httpd = _PreparedHTTPServer(("127.0.0.1", 0), app)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{httpd.server_port}/v1/models",
            timeout=5,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert payload["data"][0]["id"] == "glm-http-local"


def test_prepared_server_http_generation_error_is_bad_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fake_generate_token_ids(**kwargs):
        del kwargs
        raise TokenGeneratorError("live memory guard failed")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            served_model_name="glm-http-local",
        )
    )
    try:
        httpd = _PreparedHTTPServer(("127.0.0.1", 0), app)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/generate-token-ids",
            data=json.dumps(
                {"prompt_token_ids": [0], "max_new_tokens": 1}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert exc_info.value.code == 400
    assert payload["error"] == "live memory guard failed"


def test_prepared_server_http_request_check_error_includes_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_check_generation_runtime(**kwargs):
        del kwargs
        raise GenerationGuardError(
            "available unified memory is below test reserve",
            payload={
                "code": "available_unified_memory_below_required",
                "required_available_memory_bytes": 2048,
                "system_available_memory_bytes": 1024,
                "available_memory_ok": False,
            },
        )

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fail_check_generation_runtime,
    )
    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            served_model_name="glm-http-local",
        )
    )
    try:
        httpd = _PreparedHTTPServer(("127.0.0.1", 0), app)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this sandbox: {exc}")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/generate-token-ids",
            data=json.dumps(
                {"prompt_token_ids": [0], "max_new_tokens": 1}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        payload = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    assert exc_info.value.code == 400
    assert payload["error"] == "available unified memory is below test reserve"
    request_check = payload["request_check"]
    assert request_check["ok"] is False
    assert request_check["prompt_token_count"] == 1
    runtime = request_check["runtime_preflight"]
    assert runtime["ran"] is True
    assert runtime["code"] == "available_unified_memory_below_required"
    assert runtime["required_available_memory_bytes"] == 2048
    assert runtime["system_available_memory_bytes"] == 1024
    assert runtime["available_memory_ok"] is False


def test_prepared_server_token_ids_uses_safe_batch_prefill_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    cache_dir = tmp_path / "mla-kv-b-cache"
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_f32_prefill",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            expert_read_advise_merge_gap_kib=128,
            expert_read_advise_align_kib=4,
            prefill_mla_kv_b_cache_dir=cache_dir,
            decode_mla_key_cache=True,
            prefill_router_hybrid_margin_threshold=1e-5,
        )
    )

    payload = app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 2,
            "logits_top_k": 2,
        }
    )

    assert payload["generated_token_ids"] == [2]
    assert captured["batch_prefill_prompt"] is True
    assert captured["prefill_prompt_chunk_tokens"] == 0
    assert captured["prefill_static_capacity_per_expert"] == "auto"
    assert captured["prefill_moe_token_block"] == "auto"
    assert captured["prefill_moe_output_accumulator"] == "env"
    assert captured["prefill_mla_kv_b_cache_dir"] == cache_dir
    assert captured["decode_mla_key_cache"] is True
    assert captured["prefill_router_hybrid_margin_threshold"] == 1e-5
    assert captured["prefill_linear_backend"] == "auto"
    assert captured["preflight_runtime"] is True
    assert captured["max_live_working_set_mib"] == 8192.0
    assert captured["min_free_unified_memory_gib"] == 0.0
    assert captured["expert_read_advise_merge_gap_kib"] == 128
    assert captured["expert_read_advise_align_kib"] == 4
    assert captured["logits_top_k"] == 2
    health = app.health()
    assert health["expert_read_advise_merge_gap_kib"] == 128
    assert health["expert_read_advise_align_kib"] == 4
    assert health["prefill_moe_output_accumulator"] == "env"
    assert health["prefill_mla_kv_b_cache_dir"] == str(cache_dir)
    assert health["decode_mla_key_cache"] is True
    assert health["prefill_router_hybrid_margin_threshold"] == 1e-5
    assert payload["request_check"]["prefill_mla_kv_b_cache_dir"] == str(cache_dir)
    assert payload["request_check"]["decode_mla_key_cache"] is True
    assert (
        payload["request_check"]["prefill_router_hybrid_margin_threshold"] == 1e-5
    )


def test_prepared_server_token_ids_can_use_metal_runtime_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}
    context1_layout = tmp_path / "context1-layout.json"
    context1_cache = tmp_path / "context1-cache.bin"
    context1_cache.write_bytes(b"\0" * 16)
    _FakeMetalGenerateServerSession.instances = []

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        captured["prepared_dir"] = Path(prepared_dir)
        captured.update(kwargs)
        kwargs["generate_server_session"].request_count += 1
        return _metal_token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            generated=(7, 8),
            mla_kv_b_cache=True,
        )

    monkeypatch.setattr(
        "largerlm.server.MetalGenerateServerSession",
        _FakeMetalGenerateServerSession,
    )
    monkeypatch.setattr(
        "largerlm.server.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )

    def fake_load_context1_layout(layout_path, **kwargs):
        assert Path(layout_path) == context1_layout
        assert kwargs["prepared_dir"] == prepared
        assert kwargs["require_cache_file"] is False
        return _fake_context1_layout(
            context1_layout,
            default_cache_file=tmp_path / "context1-default-cache.bin",
        )

    monkeypatch.setattr(
        "largerlm.server.load_context1_o_proj_cache_layout",
        fake_load_context1_layout,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            metal_runtime_generation=True,
            metal_binary_path=Path("/tmp/glm_moe_infer"),
            max_live_working_set_mib=512.0,
            min_free_unified_memory_gib=3.0,
            metal_runtime_cache_mla_kv_b_f32=True,
            metal_runtime_max_mla_kv_b_cache_mib=512.0,
            metal_runtime_mmap_final_logits=True,
            metal_runtime_context1_o_proj_cache_layout=context1_layout,
            metal_runtime_context1_o_proj_cache_file=context1_cache,
        )
    )

    payload = app.generate_token_ids(
        {"prompt_token_ids": [0, 1], "max_new_tokens": 2}
    )

    assert payload["runtime"] == "glm_moe_infer"
    assert payload["generated_token_ids"] == [7, 8]
    assert payload["prompt_prefill"]["source"] == "runtime_prompt_token_ids"
    assert payload["request_check"]["metal_runtime_generation"] is True
    assert payload["request_check"]["batch_prefill_prompt"] is False
    assert captured["prepared_dir"] == prepared
    assert captured["binary"] == Path("/tmp/glm_moe_infer")
    assert captured["prompt_token_ids"] == (0, 1)
    assert captured["prefill_prompt"] is True
    assert captured["use_generate_server_jsonl"] is True
    assert captured["generate_server_session"] is _FakeMetalGenerateServerSession.instances[0]
    assert captured["max_live_working_set_mib"] == 512
    assert captured["min_free_unified_memory_gib"] == 3.0
    assert captured["cache_mla_kv_b_f32"] is True
    assert captured["max_mla_kv_b_cache_mib"] == 512.0
    assert captured["mmap_final_logits"] is True
    assert captured["context1_o_proj_cache_layout"] == context1_layout
    assert captured["context1_o_proj_cache_file"] == context1_cache
    assert captured["logits_top_k"] == captured["top_k"]
    assert payload["mla_kv_b_cache_enabled"] is True
    assert payload["mla_kv_b_cache_current_bytes"] == 96
    assert payload["mla_kv_b_cache_live_estimate_bytes"] == 512 * 1024**2
    assert payload["steps"][0]["final_logits_bytes_read"] == 1024
    assert payload["steps"][0]["final_logits_lm_head_bytes_read"] == 1000
    assert payload["steps"][0]["final_logits_read_seconds"] == 0.01
    assert payload["steps"][0]["final_logits_kernel_seconds"] == 0.08
    assert payload["steps"][0]["final_logits_resident_mmap_backed"] is False
    assert payload["steps"][1]["final_logits_lm_head_bytes_read"] == 0
    assert payload["steps"][1]["final_logits_resident_mmap_backed"] is True
    assert payload["steps"][0]["layer_elapsed_seconds"] == 0.9
    assert payload["steps"][0]["layer_count"] == 78
    assert payload["steps"][0]["dense_layer_count"] == 3
    assert payload["steps"][0]["moe_layer_count"] == 75
    assert payload["steps"][0]["attn_projection_elapsed_seconds"] == 0.11
    assert payload["steps"][0]["mla_attention_elapsed_seconds"] == 0.21
    assert payload["steps"][0]["mla_attention_cache_read_seconds"] == 0.01
    assert payload["steps"][0]["mla_attention_value_read_seconds"] == 0.02
    assert payload["steps"][0]["mla_attention_kernel_seconds"] == 0.03
    assert payload["steps"][0]["mla_attention_output_write_seconds"] == 0.04
    assert payload["steps"][0]["attn_output_elapsed_seconds"] == 0.05
    assert payload["steps"][0]["attn_output_bytes_read"] == 64
    assert payload["steps"][0]["attn_output_read_seconds"] == 0.012
    assert payload["steps"][0]["attn_output_projection_kernel_seconds"] == 0.034
    assert payload["steps"][0]["post_attn_norm_weight_bytes_read"] == 8
    assert payload["steps"][0]["post_attn_norm_weight_read_seconds"] == 0.001
    assert payload["steps"][0]["router_bytes_read"] == 24
    assert payload["steps"][0]["router_correction_bias_bytes_read"] == 4
    assert payload["steps"][0]["router_read_seconds"] == 0.002
    assert payload["steps"][0]["router_kernel_seconds"] == 0.003
    assert payload["steps"][0]["mlp_elapsed_seconds"] == 0.06
    assert payload["steps"][0]["dense_mlp_elapsed_seconds"] == 0.07
    assert payload["steps"][0]["moe_mlp_elapsed_seconds"] == 0.08
    assert payload["steps"][0]["shared_bytes_read"] == 12
    assert payload["steps"][0]["shared_read_seconds"] == 0.01
    assert payload["steps"][0]["shared_prefetch_seconds"] == 0.009
    assert payload["steps"][0]["shared_prefetch_used_count"] == 75
    assert payload["steps"][0]["moe_mlp_output_write_seconds"] == 0.005
    assert payload["steps"][0]["moe_mlp_overhead_seconds"] == 0.015
    assert payload["steps"][0]["layer_overhead_seconds"] == 0.025
    assert payload["steps"][0]["attn_projection_command_buffer_count"] == 3
    assert payload["steps"][0]["attn_projection_synchronous_wait_count"] == 1
    assert payload["steps"][0]["attn_projection_async_submitted_count"] == 77
    assert payload["steps"][0]["rope_mla_command_buffer_count"] == 0
    assert payload["steps"][0]["attn_output_command_buffer_count"] == 1
    assert payload["steps"][0]["attn_output_context1_o_proj_cache_count"] == 78
    assert payload["steps"][0]["attn_output_resident_mmap_backed_count"] == 77
    assert payload["steps"][0]["post_attn_norm_command_buffer_count"] == 0
    assert payload["steps"][0]["router_command_buffer_count"] == 0
    assert payload["steps"][0]["post_attn_norm_router_command_buffer_count"] == 0
    assert payload["steps"][0]["dense_mlp_command_buffer_count"] == 0
    assert payload["steps"][0]["dense_mlp_synchronous_wait_count"] == 0
    assert payload["steps"][0]["dense_mlp_async_submitted_count"] == 0
    assert payload["steps"][0]["moe_mlp_command_buffer_count"] == 1
    assert payload["steps"][0]["moe_mlp_synchronous_wait_count"] == 0
    assert payload["steps"][0]["attn_output_norm_router_fused_count"] == 1
    assert payload["steps"][0]["rope_mla_attn_output_norm_router_fused_count"] == 1
    assert payload["steps"][0]["rope_mla_input_buffer_direct_count"] == 1
    assert payload["steps"][0]["attn_output_buffer_direct_count"] == 1
    assert payload["steps"][0]["moe_mlp_input_buffer_direct_count"] == 1
    assert payload["steps"][0]["layer_input_buffer_direct_count"] == 1
    assert payload["steps"][0]["command_buffer_count"] == 5
    assert payload["steps"][0]["synchronous_wait_count_estimate"] == 5
    assert payload["steps"][0]["mla_value_cache_store_count"] == 2
    assert payload["steps"][1]["mla_value_cache_hit_count"] == 2
    app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})
    assert len(_FakeMetalGenerateServerSession.instances) == 1
    health = app.health()
    assert health["metal_runtime_generation"] is True
    assert health["metal_binary_path"] == "/tmp/glm_moe_infer"
    assert health["metal_runtime_cache_mla_kv_b_f32"] is True
    assert health["metal_runtime_max_mla_kv_b_cache_mib"] == 512.0
    assert health["metal_runtime_mmap_final_logits"] is True
    assert health["metal_runtime_context1_o_proj_cache_layout"] == str(context1_layout)
    assert health["metal_runtime_context1_o_proj_cache_file"] == str(context1_cache)
    assert health["metal_runtime_context1_o_proj_cache"]["ok"] is True
    assert health["metal_runtime_context1_o_proj_cache"]["cache_file_override"] is True
    assert (
        health["metal_runtime_context1_o_proj_cache"]["runtime_cache_file"]
        == str(context1_cache)
    )
    assert (
        health["metal_runtime_context1_o_proj_cache"]["runtime_cache_file_bytes"]
        == 16
    )
    assert health["suggested_metal_runtime_context1_o_proj_cache_flags"] == {
        "source": "prepared_health",
        "metal_runtime_context1_o_proj_cache": True,
        "metal_runtime_context1_o_proj_cache_layout": str(context1_layout),
        "metal_runtime_context1_o_proj_cache_file": str(context1_cache),
        "validated": True,
        "cache_file_override": True,
        "runtime_cache_file": str(context1_cache),
        "runtime_cache_file_bytes": 16,
        "argv": (
            "--metal-runtime-context1-o-proj-cache-layout",
            str(context1_layout),
            "--metal-runtime-context1-o-proj-cache-file",
            str(context1_cache),
        ),
    }
    assert health["suggested_metal_runtime_mla_kv_b_cache_flags"] == {
        "source": "prepared_health",
        "metal_runtime_cache_mla_kv_b_f32": True,
        "metal_runtime_max_mla_kv_b_cache_mib": 512.0,
        "argv": (
            "--metal-runtime-cache-mla-kv-b-f32",
            "--metal-runtime-max-mla-kv-b-cache-mib",
            "512",
        ),
    }
    assert health["suggested_metal_runtime_mmap_final_logits_flags"] == {
        "source": "prepared_health",
        "metal_runtime_mmap_final_logits": True,
        "argv": ("--metal-runtime-mmap-final-logits",),
    }
    assert "--metal-runtime-cache-mla-kv-b-f32" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--metal-runtime-mmap-final-logits" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert health["suggested_launch_profile"]["sections"][
        "metal_runtime_context1_o_proj_cache_flags"
    ] == health["suggested_metal_runtime_context1_o_proj_cache_flags"]
    assert "--metal-runtime-context1-o-proj-cache-layout" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert str(context1_layout) in health["suggested_launch_profile"]["argv"]
    assert "--metal-runtime-context1-o-proj-cache-file" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert str(context1_cache) in health["suggested_launch_profile"]["argv"]
    assert health["metal_runtime_session_started"] is True
    assert health["metal_runtime_session_request_count"] == 2
    app.close()
    assert _FakeMetalGenerateServerSession.instances[0].closed is True


def test_prepared_server_health_recommends_mla_kv_b_cache_for_metal_runtime(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            metal_runtime_generation=True,
        )
    )

    health = app.health()

    assert health["metal_runtime_cache_mla_kv_b_f32"] is False
    assert health["metal_runtime_mla_kv_b_cache_plan"] == {
        "source": "model_config",
        "num_hidden_layers": 2,
        "kv_lora_rank": 2,
        "attention_kv_b_output_dim": 4,
        "per_layer_bytes": 32,
        "estimated_full_cache_bytes": 64,
        "estimated_full_cache_mib": 64 / 1024**2,
        "recommended_max_cache_mib": 1.0,
    }
    assert health["suggested_metal_runtime_mla_kv_b_cache_flags"] == {
        "source": "prepared_health",
        "recommended": True,
        "metal_runtime_cache_mla_kv_b_f32": True,
        "metal_runtime_max_mla_kv_b_cache_mib": 1.0,
        "estimated_full_cache_bytes": 64,
        "estimated_full_cache_mib": 64 / 1024**2,
        "argv": (
            "--metal-runtime-cache-mla-kv-b-f32",
            "--metal-runtime-max-mla-kv-b-cache-mib",
            "1",
        ),
    }
    assert health["suggested_metal_runtime_mmap_final_logits_flags"] == {
        "source": "prepared_health",
        "recommended": True,
        "metal_runtime_mmap_final_logits": True,
        "argv": ("--metal-runtime-mmap-final-logits",),
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["metal_runtime_mla_kv_b_cache_flags"] == health[
        "suggested_metal_runtime_mla_kv_b_cache_flags"
    ]
    assert profile["sections"]["metal_runtime_mmap_final_logits_flags"] == health[
        "suggested_metal_runtime_mmap_final_logits_flags"
    ]
    assert "--metal-runtime-cache-mla-kv-b-f32" in profile["argv"]
    assert "--metal-runtime-mmap-final-logits" in profile["argv"]


def test_prepared_server_metal_runtime_rejects_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _FakeMetalGenerateServerSession.instances = []

    def fail_generate_metal_token_ids(*args, **kwargs):
        raise AssertionError("metal runtime should not be called")

    monkeypatch.setattr(
        "largerlm.server.generate_metal_token_ids",
        fail_generate_metal_token_ids,
    )
    monkeypatch.setattr(
        "largerlm.server.MetalGenerateServerSession",
        _FakeMetalGenerateServerSession,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            metal_runtime_generation=True,
        )
    )

    with pytest.raises(PreparedServerError, match="greedy generation only"):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 1,
                "temperature": 0.7,
            }
        )
    assert _FakeMetalGenerateServerSession.instances == []


def test_prepared_server_metal_runtime_resets_failed_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _FakeMetalGenerateServerSession.instances = []
    calls = 0

    def fake_generate_metal_token_ids(prepared_dir, **kwargs):
        nonlocal calls
        calls += 1
        session = kwargs["generate_server_session"]
        session.request_count += 1
        if calls == 1:
            raise MetalGenerateError("jsonl pipe broke")
        return _metal_token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            generated=(9,),
        )

    monkeypatch.setattr(
        "largerlm.server.MetalGenerateServerSession",
        _FakeMetalGenerateServerSession,
    )
    monkeypatch.setattr(
        "largerlm.server.generate_metal_token_ids",
        fake_generate_metal_token_ids,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            metal_runtime_generation=True,
        )
    )

    with pytest.raises(PreparedServerError, match="jsonl pipe broke"):
        app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})

    assert len(_FakeMetalGenerateServerSession.instances) == 1
    assert _FakeMetalGenerateServerSession.instances[0].closed is True
    assert app.health()["metal_runtime_session_started"] is False

    payload = app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})

    assert payload["generated_token_ids"] == [9]
    assert len(_FakeMetalGenerateServerSession.instances) == 2
    assert _FakeMetalGenerateServerSession.instances[1].closed is False
    assert app.health()["metal_runtime_session_request_count"] == 1


def test_prepared_server_metal_runtime_rejects_prefill_acceleration_requirement(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(PreparedServerError, match="prefill acceleration"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                metal_runtime_generation=True,
                require_prefill_acceleration=True,
                enforce_prefill_acceleration_probe=False,
            )
        )


def test_prepared_server_token_ids_runs_request_admission_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=999,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill_prompt_chunk_tokens 999 exceeds safety-capped maximum",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_rejected_prefill_chunk_reports_chunk_plan(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=999,
        )
    )

    with pytest.raises(PreparedRequestCheckError) as exc_info:
        app.inspect_token_request(
            prompt_token_count=2,
            payload={"max_new_tokens": 1},
        )

    payload = exc_info.value.payload
    assert payload["ok"] is False
    assert payload["error"] == str(exc_info.value)
    assert payload["batch_prefill_prompt"] is True
    chunk = payload["prefill_prompt_chunk_tokens"]
    assert chunk["configured"] == 999
    assert chunk["resolved"] == 999
    assert chunk["max_safe"] >= 1
    plan = payload["prefill_prompt_chunk_plan"]
    assert plan["source"] == "prepared_request_check"
    assert plan["configured_is_auto"] is False
    assert plan["auto"] is None
    assert plan["max_safe"]["chunk_tokens"] == chunk["max_safe"]
    assert "mpp_tensor_ops_candidate_reachable_under_caps" in plan["max_safe"]
    assert "mpp_tensor_ops_candidate_blocking_cap_summary" in plan["max_safe"]


def test_prepared_server_token_ids_runs_runtime_preflight_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_check_generation_runtime(**kwargs):
        del kwargs
        raise GenerationGuardError(
            "available unified memory is below test reserve",
            payload={
                "code": "available_unified_memory_below_required",
                "required_available_memory_bytes": 2048,
                "system_available_memory_bytes": 1024,
                "available_memory_ok": False,
            },
        )

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fail_check_generation_runtime,
    )
    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    with pytest.raises(PreparedRequestCheckError, match="below test reserve") as exc_info:
        app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})
    runtime = exc_info.value.payload["runtime_preflight"]
    assert runtime["ran"] is True
    assert runtime["code"] == "available_unified_memory_below_required"
    assert runtime["required_available_memory_bytes"] == 2048
    assert runtime["system_available_memory_bytes"] == 1024
    assert runtime["available_memory_ok"] is False


def test_prepared_server_token_payload_reports_decode_layer_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    step = SimpleNamespace(
        position=0,
        input_token_id=0,
        selected_token_id=2,
        elapsed_seconds=0.25,
        embedding_read_bytes=32,
        expert_read_bytes=384,
        cache_read_bytes=24,
        logits_read_bytes=16,
        estimated_read_bytes=456,
        topk=(SimpleNamespace(token_id=2, logit=1.5),),
        decode_layers=(
            SimpleNamespace(
                layer=1,
                kind="moe",
                composed=False,
                expert_read_bytes=384,
                attention_read_bytes=128,
                cache_read_bytes=24,
                dsa_index_cache_read_bytes=0,
                mla_cache_read_bytes=24,
                estimated_peak_bytes=4096,
                router_stage_peak_bytes=512,
                moe_stage_peak_bytes=2048,
                input_in_memory=True,
                output_in_memory=False,
                attention_stage_elapsed_seconds={
                    "projections": 0.1,
                    "mla_attention": 0.2,
                },
                mlp_stage_elapsed_seconds={
                    "expert_kernel": 0.11,
                    "expert_read": 0.03,
                    "router": 0.02,
                    "shared": 0.04,
                },
                mlp_diagnostics={
                    "preload_selected_enabled": True,
                    "preload_selected_bytes": 8192,
                    "mxfp4_fused_decode_enabled": True,
                },
                mla_attention_timing_elapsed_seconds={
                    "kernel": 0.18,
                    "kernel_weights": 0.11,
                    "kernel_values": 0.07,
                    "total": 0.21,
                },
                mla_attention_diagnostics={
                    "weights_bytes": 64,
                    "estimated_peak_bytes": 4096,
                    "singleton_kernel": False,
                    "split_kernel_timing_enabled": True,
                },
                dsa_indexer_mode="none",
                dsa_rope_interleave=False,
            ),
        ),
    )

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(step,),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=0.25,
            estimated_read_bytes=456,
            estimated_embedding_read_bytes=32,
            estimated_expert_read_bytes=384,
            estimated_cache_read_bytes=24,
            estimated_logits_read_bytes=16,
            decode_actual_read_time={
                "source": "generation_actual_decode",
                "decode_step_count": 1,
                "decode_read_bytes_per_token": 384,
                "planned_decode_routed_read_bytes": 384,
                "actual_decode_routed_read_bytes": 384,
                "actual_decode_routed_read_bytes_ok": True,
                "planned_decode_routed_read_seconds": 0.25,
                "actual_decode_routed_read_seconds": 0.25,
                "prefill_ssd_read_gib_per_second": 16.0,
                "decode_max_routed_read_seconds_per_token": 1.0,
                "total_decode_max_routed_read_seconds": 1.0,
                "total_decode_routed_read_seconds_ok": True,
            },
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    payload = app.generate_token_ids(
        {
            "prompt_token_ids": [0],
            "max_new_tokens": 1,
        }
    )

    assert payload["steps"][0]["estimated_read_bytes"] == 456
    assert payload["steps"][0]["embedding_read_bytes"] == 32
    assert payload["steps"][0]["topk"] == [{"token_id": 2, "logit": 1.5}]
    assert payload["estimated_read_bytes"] == 456
    assert payload["estimated_embedding_read_bytes"] == 32
    assert payload["estimated_expert_read_bytes"] == 384
    assert payload["estimated_cache_read_bytes"] == 24
    assert payload["estimated_logits_read_bytes"] == 16
    assert payload["decode_actual_read_time"] == {
        "source": "generation_actual_decode",
        "decode_step_count": 1,
        "decode_read_bytes_per_token": 384,
        "planned_decode_routed_read_bytes": 384,
        "actual_decode_routed_read_bytes": 384,
        "actual_decode_routed_read_bytes_ok": True,
        "planned_decode_routed_read_seconds": 0.25,
        "actual_decode_routed_read_seconds": 0.25,
        "prefill_ssd_read_gib_per_second": 16.0,
        "decode_max_routed_read_seconds_per_token": 1.0,
        "total_decode_max_routed_read_seconds": 1.0,
        "total_decode_routed_read_seconds_ok": True,
    }
    decode_layer = payload["steps"][0]["decode_layers"][0]
    assert decode_layer["layer"] == 1
    assert decode_layer["kind"] == "moe"
    assert decode_layer["expert_read_bytes"] == 384
    assert decode_layer["cache_read_bytes"] == 24
    assert decode_layer["estimated_peak_bytes"] == 4096
    assert decode_layer["input_in_memory"] is True
    assert decode_layer["output_in_memory"] is False
    assert decode_layer["attention_stage_elapsed_seconds"] == {
        "projections": 0.1,
        "mla_attention": 0.2,
    }
    assert decode_layer["mlp_stage_elapsed_seconds"] == {
        "expert_kernel": 0.11,
        "expert_read": 0.03,
        "router": 0.02,
        "shared": 0.04,
    }
    assert decode_layer["mlp_diagnostics"] == {
        "preload_selected_enabled": True,
        "preload_selected_bytes": 8192,
        "mxfp4_fused_decode_enabled": True,
    }
    assert decode_layer["mla_attention_timing_elapsed_seconds"] == {
        "kernel": 0.18,
        "kernel_weights": 0.11,
        "kernel_values": 0.07,
        "total": 0.21,
    }
    assert decode_layer["mla_attention_diagnostics"] == {
        "weights_bytes": 64,
        "estimated_peak_bytes": 4096,
        "singleton_kernel": False,
        "split_kernel_timing_enabled": True,
    }


def test_prepared_server_passes_config_tie_embedding_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = prepared.parent / "model" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tie_word_embeddings"] = False
    config["vocab_size"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    app.generate_token_ids(
        {
            "prompt_token_ids": [0],
            "max_new_tokens": 1,
        }
    )

    assert captured["allow_tied_embeddings"] is False
    assert captured["expected_vocab_size"] == 4
    assert captured["expected_hidden_size"] == 8


def test_prepared_server_auto_prefill_backend_falls_back_without_mpsgraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_prefill_fallback",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("MPSGraph matmul headers are missing",),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_linear_backend="auto",
        )
    )

    health = app.health()
    app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )
    check = app.inspect_token_request(
        prompt_token_count=2,
        payload={"max_new_tokens": 1},
    )

    assert health["prefill_linear_backend"] == "auto"
    assert health["runtime_prefill_linear_backend"] == "custom-metal"
    assert health["prefill_backend"]["configured_backend"] == "auto"
    assert health["prefill_backend"]["effective_backend"] == "custom-metal"
    assert captured["prefill_linear_backend"] == "custom-metal"
    assert check["prefill_linear_backend"]["configured"] == "auto"
    assert check["prefill_linear_backend"]["effective"] == "custom-metal"
    chunk_plan = check["prefill_prompt_chunk_plan"]
    assert chunk_plan["source"] == "prepared_request_check"
    assert chunk_plan["configured_is_auto"] is True
    assert chunk_plan["auto"]["chunk_tokens"] == (
        check["prefill_prompt_chunk_tokens"]["resolved"]
    )
    assert chunk_plan["max_safe"]["chunk_tokens"] == (
        check["prefill_prompt_chunk_tokens"]["max_safe"]
    )


def test_prepared_server_request_check_reports_prefill_chunk_plan_drift_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server._auto_prefill_prompt_chunk_plan",
        lambda **kwargs: _fake_auto_prefill_chunk_plan(2),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            applied_launch_profile=_applied_launch_profile(tmp_path),
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=2,
        payload={"max_new_tokens": 1},
    )

    drift = check["prefill_prompt_chunk_plan_drift"]
    assert drift["source"] == "applied_launch_profile"
    assert drift["status"] == "ok"
    assert drift["profile_max_safe_chunk_tokens"] == 2
    assert drift["actual_prompt_chunk_tokens"] == 2
    assert drift["actual_auto_chunk_tokens"] == 2
    assert drift["actual_max_safe_chunk_tokens"] == 2


def test_prepared_server_generation_rejects_prefill_chunk_plan_drift_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    applied_profile = _applied_launch_profile(tmp_path)
    applied_profile["prefill_prompt_chunk_plan"]["max_safe"]["chunk_tokens"] = 3
    monkeypatch.setattr(
        "largerlm.server._auto_prefill_prompt_chunk_plan",
        lambda **kwargs: _fake_auto_prefill_chunk_plan(2),
    )
    called = False
    expected_drift = {
        "status": "shorter_prompt",
        "profile_max_safe_prompt_tokens": 4,
        "actual_max_safe_prompt_tokens": 2,
        "current_max_safe_at_least_profile": False,
        "current_request_below_profile_prompt_tokens": True,
    }

    def fake_generate_token_ids(**kwargs):
        nonlocal called
        called = True
        return _token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            auto_prefill_prompt_chunk_plan=_fake_auto_prefill_chunk_plan(2),
            max_safe_prefill_prompt_chunk_plan=_fake_auto_prefill_chunk_plan(2),
            prefill_prompt_chunk_plan_drift=expected_drift,
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            applied_launch_profile=applied_profile,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill chunk-plan drift current_max_safe_below_profile",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1],
                "max_new_tokens": 1,
            }
        )

    assert called is False


def test_prepared_server_generation_allows_shorter_prompt_than_profile_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    applied_profile = _applied_launch_profile(tmp_path)
    applied_profile["prefill_prompt_chunk_plan"]["auto"] = {
        "prompt_tokens": 4,
        "chunk_tokens": 4,
    }
    applied_profile["prefill_prompt_chunk_plan"]["max_safe"] = {
        "prompt_tokens": 4,
        "chunk_tokens": 4,
    }
    monkeypatch.setattr(
        "largerlm.server._auto_prefill_prompt_chunk_plan",
        lambda **kwargs: _fake_auto_prefill_chunk_plan(2),
    )
    called = False
    expected_drift = {
        "status": "shorter_prompt",
        "profile_max_safe_prompt_tokens": 4,
        "actual_max_safe_prompt_tokens": 2,
        "current_max_safe_at_least_profile": False,
        "current_request_below_profile_prompt_tokens": True,
    }

    def fake_generate_token_ids(**kwargs):
        nonlocal called
        called = True
        return _token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            auto_prefill_prompt_chunk_plan=_fake_auto_prefill_chunk_plan(2),
            max_safe_prefill_prompt_chunk_plan=_fake_auto_prefill_chunk_plan(2),
            prefill_prompt_chunk_plan_drift=expected_drift,
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            applied_launch_profile=applied_profile,
        )
    )
    check = app.inspect_token_request(
        prompt_token_count=2,
        payload={"max_new_tokens": 1},
    )
    drift = check["prefill_prompt_chunk_plan_drift"]
    assert drift["status"] == "shorter_prompt"
    assert drift["profile_max_safe_prompt_tokens"] == 4
    assert drift["actual_max_safe_prompt_tokens"] == 2
    assert drift["current_max_safe_at_least_profile"] is False
    assert drift["current_request_below_profile_prompt_tokens"] is True

    payload = app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )

    drift = payload["prefill_prompt_chunk_plan_drift"]
    assert drift["status"] == "shorter_prompt"
    assert drift["profile_max_safe_prompt_tokens"] == 4
    assert drift["actual_max_safe_prompt_tokens"] == 2
    assert drift["current_max_safe_at_least_profile"] is False
    assert drift["current_request_below_profile_prompt_tokens"] is True
    assert called is True


def test_prepared_server_generation_rejects_nonpassing_admission_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    called = False

    def fake_inspect_token_request(self, **kwargs):
        del self, kwargs
        return {
            "ok": False,
            "reason": "synthetic live-memory guard failure",
        }

    def fake_generate_token_ids(**kwargs):
        nonlocal called
        called = True
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        fake_inspect_token_request,
    )
    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="synthetic live-memory guard failure",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1],
                "max_new_tokens": 1,
            }
        )

    assert called is False


def test_prepared_server_auto_prefill_backend_falls_back_without_mpsgraph_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_prefill_fallback",
            mps_graph_matmul_declared=True,
            mps_graph_runtime_available=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("host probe could not create a default Metal device",),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_linear_backend="auto",
        )
    )

    health = app.health()
    app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )

    assert health["runtime_prefill_linear_backend"] == "custom-metal"
    assert health["prefill_backend"]["effective_backend"] == "custom-metal"
    assert (
        health["prefill_backend"]["capability"]["mps_graph_runtime_available"]
        is False
    )
    assert health["prefill_backend"]["warnings"] == (
        "auto prefill will fall back to custom-metal resident GEMMs because "
        "the MPSGraph runtime probe did not succeed",
    )
    assert captured["prefill_linear_backend"] == "custom-metal"


def test_prepared_server_backend_health_reports_selectable_acceleration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(
            tmp_path,
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpp_tensor_ops_prefill",
            metal4_ml_runtime_available=True,
            mpp_runtime_available=True,
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_min_accelerated_flop_fraction=0.5,
            prefill_mpsgraph_min_batch_tokens=64,
            prefill_mpsgraph_min_matrix_dim=16,
            prefill_router_hybrid_margin_threshold=1e-5,
            prefill_run_mpsgraph_probe=True,
        )
    )

    health = app.health()
    capability = health["prefill_backend"]["capability"]

    assert capability["prefill_acceleration_runtimes"] == (
        "mpp_tensor_ops_prefill",
        "mpsgraph-f32",
    )
    assert capability["selectable_accelerated_prefill_backends"] == (
        "mpp-f32",
        "mpsgraph-f32",
    )
    assert capability["validated_accelerated_prefill_backends"] == ("mpsgraph-f32",)
    assert capability["prefill_acceleration_runtime_gaps"] == ()
    neural_status = capability["prefill_neural_accelerator_status"]
    assert neural_status["runtime"] == "mpp_tensor_ops_prefill"
    assert neural_status["execution_path"] == (
        "mpp_tensor_ops_gpu_neural_accelerator"
    )
    assert neural_status["status"] == "selectable"
    assert neural_status["runtime_visible"] is True
    assert neural_status["ready_for_generation"] is True
    assert neural_status["selectable"] is True
    assert capability["selectable_prefill_acceleration_available"] is True
    assert capability["validated_prefill_acceleration_available"] is True
    assert capability["suggested_prefill_acceleration_flags"] == {
        "source": "prepared_health",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_acceleration_runtimes": (
            "mpp_tensor_ops_prefill",
            "mpsgraph-f32",
        ),
        "selectable_accelerated_prefill_backends": ("mpp-f32", "mpsgraph-f32"),
        "validated_accelerated_prefill_backends": ("mpsgraph-f32",),
        "prefill_acceleration_runtime_gaps": (),
        "prefill_neural_accelerator_status": neural_status,
        "runtime_probe_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": True,
        "runtime_probe_argv": ("--run-mpsgraph-probe",),
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }
    profile = health["suggested_launch_profile"]
    assert profile["source"] == "prepared_health"
    assert profile["argv_safe_to_replay"] is True
    assert profile["prepared"]["identity_strength"] == "weak"
    assert "model_config_sha256 is unavailable" in (
        profile["prepared"]["identity_warnings"][0]
    )
    assert profile["prepared"]["expert_layout_bytes"] == 16
    assert profile["prepared"]["resident_layout_bytes"] == 16
    assert profile["sections"]["prefill_acceleration_flags"] == {
        "source": "prepared_health",
        "prefill_acceleration_runtimes": (
            "mpp_tensor_ops_prefill",
            "mpsgraph-f32",
        ),
        "selectable_accelerated_prefill_backends": ("mpp-f32", "mpsgraph-f32"),
        "validated_accelerated_prefill_backends": ("mpsgraph-f32",),
        "prefill_acceleration_runtime_gaps": (),
        "prefill_neural_accelerator_status": neural_status,
        "runtime_probe_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": True,
        "runtime_probe_argv": ("--run-mpsgraph-probe",),
        "argv": ("--require-prefill-acceleration", "--run-mpsgraph-probe"),
        "prefill_linear_backend_policy": "auto",
    }
    policy = health["suggested_prefill_runtime_policy_flags"]
    assert policy == {
        "source": "prepared_health",
        "prefill_mpsgraph_min_batch_tokens": 64,
        "prefill_mpsgraph_min_matrix_dim": 16,
        "prefill_min_accelerated_flop_fraction": 0.5,
        "prefill_router_hybrid_margin_threshold": 1e-5,
        "argv": (
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--prefill-router-hybrid-margin-threshold",
            "1e-05",
        ),
    }
    assert profile["sections"]["prefill_runtime_policy_flags"] == policy
    assert "--prefill-linear-backend" not in profile["argv"]
    assert "--require-prefill-acceleration" in profile["argv"]
    assert "--prefill-min-accelerated-flop-fraction" in profile["argv"]
    assert "--prefill-router-hybrid-margin-threshold" in profile["argv"]
    assert "--prefill-mpsgraph-min-batch-tokens" in profile["argv"]


def test_prepared_server_backend_health_runs_mpp_compile_probe_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend_probe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpp_tensor_ops_prefill",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=True,
            mpp_tensor_ops_symbol_declared=True,
            mpp_runtime_available=True,
            mpp_compile_probe_requested=bool(kwargs.get("compile_mpp_probe")),
            mpp_compile_probe_ran=True,
            mpp_compile_probe_ok=True,
            mpp_compile_variant="metal_mpp",
            mpp_compile_error=None,
            reasons=(),
        )

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fake_backend_probe)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_compile_mpp_probe=True,
        )
    )

    health = app.health()
    capability = health["prefill_backend"]["capability"]

    assert captured["compile_mpp_probe"] is True
    assert health["prefill_compile_mpp_probe"] is True
    assert health["prefill_backend"]["auto_policy"]["compile_mpp_probe"] is True
    assert capability["mpp_compile_probe_requested"] is True
    assert capability["mpp_compile_probe_ran"] is True
    assert capability["mpp_compile_probe_ok"] is True
    assert capability["mpp_compile_variant"] == "metal_mpp"
    neural_status = capability["prefill_neural_accelerator_status"]
    assert neural_status["status"] == "selectable"
    assert neural_status["mpp_tensor_ops_symbol_declared"] is True
    assert neural_status["mpp_compile_probe_requested"] is True
    assert neural_status["mpp_compile_probe_ran"] is True
    assert neural_status["mpp_compile_probe_ok"] is True
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "compile_mpp_probe": True,
        "argv": ("--compile-mpp-probe",),
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert "--compile-mpp-probe" in profile["argv"]


def test_prepared_server_backend_health_runs_mpp_execution_probe_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend_probe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpp_tensor_ops_prefill",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=True,
            mpp_tensor_ops_symbol_declared=True,
            mpp_runtime_available=True,
            mpp_compile_probe_requested=True,
            mpp_compile_probe_ran=True,
            mpp_compile_probe_ok=True,
            mpp_compile_variant="metal_mpp",
            mpp_compile_error=None,
            mpp_run_probe_requested=bool(kwargs.get("run_mpp_probe")),
            mpp_run_probe_ran=True,
            mpp_run_probe_ok=True,
            mpp_run_probe_error=None,
            mpp_run_probe_max_abs_error=0.0,
            reasons=(),
        )

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fake_backend_probe)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_run_mpp_probe=True,
        )
    )

    health = app.health()
    capability = health["prefill_backend"]["capability"]

    assert captured["run_mpp_probe"] is True
    assert health["prefill_run_mpp_probe"] is True
    assert health["prefill_backend"]["auto_policy"]["run_mpp_probe"] is True
    assert capability["mpp_run_probe_requested"] is True
    assert capability["mpp_run_probe_ran"] is True
    assert capability["mpp_run_probe_ok"] is True
    assert capability["mpp_run_probe_max_abs_error"] == 0.0
    assert capability["prefill_neural_accelerator_status"]["status"] == "selectable"
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "run_mpp_probe": True,
        "argv": ("--run-mpp-probe",),
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert "--run-mpp-probe" in profile["argv"]


def test_prepared_server_backend_health_runs_mpsgraph_probe_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend_probe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            mps_graph_probe_ran=True,
            mps_graph_probe_ok=True,
            mps_graph_probe_error=None,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            mpp_compile_probe_ran=False,
            mpp_compile_probe_ok=None,
            mpp_compile_variant=None,
            mpp_compile_error=None,
            reasons=(),
        )

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fake_backend_probe)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_run_mpsgraph_probe=True,
            prefill_backend_probe_timeout_seconds=12.5,
        )
    )

    health = app.health()
    capability = health["prefill_backend"]["capability"]

    assert captured["run_mpsgraph_probe"] is True
    assert captured["probe_timeout_seconds"] == 12.5
    assert health["prefill_run_mpsgraph_probe"] is True
    assert health["prefill_backend_probe_timeout_seconds"] == 12.5
    assert health["prefill_backend"]["auto_policy"]["run_mpsgraph_probe"] is True
    assert capability["mps_graph_probe_requested"] is True
    assert capability["mps_graph_probe_ran"] is True
    assert capability["mps_graph_probe_ok"] is True
    assert capability["host_probe_requested"] is True
    assert capability["host_probe_path"]
    assert capability["host_probe_ran"] is True
    assert capability["host_probe_ok"] is True
    assert capability["prefill_backend_probe_timeout_seconds"] == 12.5
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "run_mpsgraph_probe": True,
        "argv": ("--run-mpsgraph-probe",),
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert "--run-mpsgraph-probe" in profile["argv"]


def test_prepared_server_request_check_reports_routed_read_amplification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server._system_disk_usage",
        lambda path: {
            "path": str(path),
            "total_bytes": 1024**3,
            "used_bytes": 1024,
            "free_bytes": 1024**3 - 1024,
        },
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=4,
        payload={"max_new_tokens": 0},
    )

    chunk_plan = check["prefill_prompt_chunk_plan"]
    assert chunk_plan["source"] == "prepared_request_check"
    assert chunk_plan["configured_is_auto"] is False
    assert chunk_plan["auto"] is None
    assert chunk_plan["max_safe"]["chunk_tokens"] == (
        check["prefill_prompt_chunk_tokens"]["max_safe"]
    )
    assert "prompt_tokens" in chunk_plan["max_safe"]
    assert "limiting_cap_names" in chunk_plan["max_safe"]
    routed = check["prefill_routed_expert_read"]
    assert routed["analyzed"] is True
    assert routed["prompt_chunk_tokens"] == 2
    assert routed["top_k"] == 2
    assert routed["layers"] == 1
    assert routed["chunks_per_prompt"] == 2
    assert routed["baseline_read_bytes"] == 16
    assert routed["planned_read_bytes"] == 32
    assert routed["extra_read_bytes"] == 16
    assert routed["planned_read_seconds"] is None
    assert routed["read_amplification"] == 2.0
    assert routed["max_read_amplification"] == 0.0
    assert routed["max_planned_read_bytes"] == 0
    assert routed["max_read_seconds"] == 0.0
    assert routed["minimum_chunk_tokens_for_limits"] is None
    assert routed["within_amplification_limit"] is True
    assert routed["within_planned_read_limit"] is True
    assert routed["within_seconds_limit"] is True
    assert routed["within_limit"] is True
    assert routed["max_layer_baseline_read_bytes"] == 16
    assert routed["max_layer_planned_read_bytes"] == 32
    stage_temp = check["prefill_routed_stage_temp_disk"]
    assert stage_temp["analyzed"] is True
    assert stage_temp["prompt_chunk_tokens"] == 2
    assert stage_temp["top_k"] == 2
    assert stage_temp["layers"] == 1
    assert stage_temp["chunks_per_prompt"] == 2
    assert stage_temp["stage_align_bytes"] == 4096
    assert stage_temp["static_capacity_per_expert"] == "auto"
    assert stage_temp["allow_static_capacity_overflow"] is False
    assert stage_temp["max_static_capacity_per_expert"] == 2
    assert stage_temp["static_capacity_strict_overflow_safe"] is True
    assert stage_temp["max_unique_experts_per_layer"] == 1
    assert stage_temp["max_stage_raw_ranges"] == 1
    assert stage_temp["max_stage_raw_range_limit"] == 0
    assert stage_temp["within_stage_raw_range_limit"] is True
    assert stage_temp["max_stage_coalesced_ranges"] == 1
    assert stage_temp["max_stage_coalesced_range_limit"] == 0
    assert stage_temp["within_stage_coalesced_range_limit"] is True
    assert stage_temp["max_stage_bytes"] == 4112
    assert stage_temp["max_compact_stage_bytes"] == 16
    assert stage_temp["max_stage_plus_compact_bytes"] == 4128
    assert stage_temp["max_static_capacity_binary_bytes"] == 68
    assert stage_temp["max_static_capacity_overflow_records"] == 0
    assert stage_temp["max_stage_plus_compact_plus_static_bytes"] == 4196
    assert stage_temp["max_chunk_stage_plus_compact_bytes"] == 4128
    assert stage_temp["max_chunk_static_capacity_binary_bytes"] == 68
    assert stage_temp["max_chunk_stage_plus_compact_plus_static_bytes"] == 4196
    assert stage_temp["total_stage_bytes"] == 8224
    assert stage_temp["total_compact_stage_bytes"] == 32
    assert stage_temp["total_stage_plus_compact_bytes"] == 8256
    assert stage_temp["total_static_capacity_binary_bytes"] == 136
    assert stage_temp["total_static_capacity_overflow_records"] == 0
    assert stage_temp["total_stage_plus_compact_plus_static_bytes"] == 8392
    assert stage_temp["max_stage_limit_bytes"] == 4096 * 1024**2
    assert stage_temp["max_compact_stage_limit_bytes"] == 4096 * 1024**2
    assert stage_temp["within_stage_limit"] is True
    assert stage_temp["within_compact_stage_limit"] is True
    assert stage_temp["within_limit"] is True
    disk_free = check["prefill_stage_temp_disk_free"]
    assert disk_free["analyzed"] is True
    assert disk_free["path"] == "/private/tmp"
    assert disk_free["required_stage_temp_bytes"] == 4196
    assert disk_free["disk_safety_margin_bytes"] == 0
    assert disk_free["required_free_bytes"] == 4196
    assert disk_free["free_bytes"] == 1024**3 - 1024
    assert disk_free["within_free_space"] is True
    suggested = check["suggested_guard_flags"]
    assert suggested["headroom_factor"] == 1.05
    assert suggested["source"] == "prepared_request_check"
    assert suggested["prefill_prompt_chunk_tokens"] == 2
    assert suggested["prefill_max_routed_read_amplification"] == pytest.approx(2.1)
    assert suggested["prefill_max_routed_read_gib"] == pytest.approx(
        32 / 1024**3 * 1.05
    )
    assert suggested["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "2",
        "--prefill-max-routed-read-amplification",
        "2.1",
        "--prefill-max-routed-read-gib",
        f"{32 / 1024**3 * 1.05:.6g}",
    )
    stage_suggested = check["suggested_stage_temp_guard_flags"]
    assert stage_suggested["headroom_factor"] == 1.05
    assert stage_suggested["source"] == "prepared_request_check"
    assert stage_suggested["prefill_prompt_chunk_tokens"] == 2
    assert stage_suggested["prefill_max_stage_mib"] == pytest.approx(
        4112 / 1024**2 * 1.05
    )
    assert stage_suggested["prefill_max_compact_stage_mib"] == pytest.approx(
        16 / 1024**2 * 1.05
    )
    assert stage_suggested["profile_max_stage_bytes"] == 4112
    assert stage_suggested["profile_max_compact_stage_bytes"] == 16
    assert stage_suggested["profile_max_stage_plus_compact_bytes"] == 4128
    assert stage_suggested["profile_total_stage_plus_compact_bytes"] == 8256
    assert stage_suggested["profile_max_static_capacity_binary_bytes"] == 68
    assert stage_suggested["profile_total_static_capacity_binary_bytes"] == 136
    assert stage_suggested["profile_max_stage_plus_compact_plus_static_bytes"] == 4196
    assert (
        stage_suggested["profile_total_stage_plus_compact_plus_static_bytes"] == 8392
    )
    assert stage_suggested["prefill_max_stage_raw_ranges"] == 2
    assert stage_suggested["profile_max_stage_raw_ranges"] == 1
    assert stage_suggested["prefill_max_stage_coalesced_ranges"] == 2
    assert stage_suggested["profile_max_stage_coalesced_ranges"] == 1
    assert stage_suggested["prefill_static_capacity_per_expert"] == "auto"
    assert stage_suggested["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "2",
        "--prefill-max-stage-mib",
        f"{4112 / 1024**2 * 1.05:.6g}",
        "--prefill-max-compact-stage-mib",
        f"{16 / 1024**2 * 1.05:.6g}",
        "--prefill-static-capacity-per-expert",
        "auto",
        "--prefill-max-stage-raw-ranges",
        "2",
        "--prefill-max-stage-coalesced-ranges",
        "2",
    )
    combined = check["suggested_prefill_guard_flags"]
    assert combined["source"] == "prepared_request_check"
    assert combined["prefill_prompt_chunk_tokens"] == 2
    assert combined["routed_read_guard"] == suggested
    assert combined["stage_temp_guard"] == stage_suggested
    assert combined["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "2",
        "--prefill-max-routed-read-amplification",
        "2.1",
        "--prefill-max-routed-read-gib",
        f"{32 / 1024**3 * 1.05:.6g}",
        "--prefill-max-stage-mib",
        f"{4112 / 1024**2 * 1.05:.6g}",
        "--prefill-max-compact-stage-mib",
        f"{16 / 1024**2 * 1.05:.6g}",
        "--prefill-static-capacity-per-expert",
        "auto",
        "--prefill-max-stage-raw-ranges",
        "2",
        "--prefill-max-stage-coalesced-ranges",
        "2",
    )
    assert combined["argv"].count("--prefill-prompt-chunk-tokens") == 1
    frontier = check["prefill_routed_chunk_frontier"]
    assert frontier["analyzed"] is True
    assert frontier["resolved_prompt_chunk_tokens"] == 2
    assert frontier["max_safe_prompt_chunk_tokens"] == 4
    assert frontier["prompt_token_count"] == 4
    assert frontier["top_k"] == 2
    assert frontier["layers"] == 1
    assert frontier["baseline_read_bytes"] == 16
    assert frontier["saturation_chunk_tokens"] == 1
    candidates = {
        item["prompt_chunk_tokens"]: item for item in frontier["candidates"]
    }
    assert sorted(candidates) == [1, 2, 4]
    assert candidates[2]["chunks_per_prompt"] == 2
    assert candidates[2]["saturates_all_experts_per_layer"] is True
    assert candidates[2]["planned_read_bytes"] == 32
    assert candidates[2]["read_amplification"] == 2.0
    assert candidates[2]["max_stage_plus_compact_bytes"] == 4128
    assert candidates[2]["total_stage_plus_compact_bytes"] == 8256
    assert candidates[2]["max_static_capacity_binary_bytes"] == 68
    assert candidates[2]["total_static_capacity_binary_bytes"] == 136
    assert candidates[2]["max_stage_plus_compact_plus_static_bytes"] == 4196
    assert candidates[2]["total_stage_plus_compact_plus_static_bytes"] == 8392


def test_prepared_server_request_check_uses_work_dir_for_stage_temp_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    seen_paths: list[Path] = []

    def fake_disk_usage(path: Path) -> dict[str, object]:
        seen_paths.append(Path(path))
        return {
            "path": str(path),
            "total_bytes": 1024**3,
            "used_bytes": 1024,
            "free_bytes": 1024**3 - 1024,
        }

    monkeypatch.setattr("largerlm.server._system_disk_usage", fake_disk_usage)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
        )
    )
    work_dir = tmp_path / "prefill-work"

    check = app.inspect_token_request(
        prompt_token_count=4,
        payload={"max_new_tokens": 0},
        generation_overrides={"work_dir": work_dir},
    )

    disk_free = check["prefill_stage_temp_disk_free"]
    assert seen_paths == [work_dir]
    assert disk_free["path"] == str(work_dir)
    assert disk_free["within_free_space"] is True


def test_prepared_server_request_check_rejects_routed_stage_temp_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server._system_disk_usage",
        lambda path: {
            "path": str(path),
            "total_bytes": 1024**3,
            "used_bytes": 1024,
            "free_bytes": 1024**3 - 1024,
        },
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    monkeypatch.setattr(
        "largerlm.server._auto_prefill_prompt_chunk_plan",
        lambda **kwargs: _fake_auto_prefill_chunk_plan(4),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_max_stage_mib=0.001,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prompt prefill routed stage temp stage 4112 bytes exceeds cap 1048 bytes",
    ):
        app.inspect_token_request(
            prompt_token_count=4,
            payload={"max_new_tokens": 0},
        )


def test_prepared_server_request_check_accepts_tiled_routed_stage_temp_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "experts" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["num_experts"] = 2
    layout["layers"][0]["num_experts"] = 2
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    (prepared / "experts" / "layer_000.bin").write_bytes(b"\0" * 32)
    monkeypatch.setattr(
        "largerlm.server._system_disk_usage",
        lambda path: {
            "path": str(path),
            "total_bytes": 1024**3,
            "used_bytes": 1024,
            "free_bytes": 1024**3 - 1024,
        },
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    monkeypatch.setattr(
        "largerlm.server._auto_prefill_prompt_chunk_plan",
        lambda **kwargs: _fake_auto_prefill_chunk_plan(4),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_max_stage_mib=0.005,
            prefill_expert_stage_tiling=True,
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=4,
        payload={"max_new_tokens": 0},
    )

    stage_temp = check["prefill_routed_stage_temp_disk"]
    assert stage_temp["expert_stage_tiling"] is True
    assert stage_temp["within_limit"] is True
    assert stage_temp["max_stage_bytes"] == 8224
    assert stage_temp["effective_max_stage_bytes"] == 4112
    assert stage_temp["expert_stage_tiling_plan"]["total_stage_tile_count"] == 4
    assert stage_temp["expert_stage_tiling_plan"]["max_experts_per_stage_tile"] == 1
    disk_free = check["prefill_stage_temp_disk_free"]
    assert disk_free["required_stage_temp_bytes"] == (
        stage_temp["effective_max_stage_plus_compact_plus_static_bytes"]
    )
    assert check["suggested_stage_temp_guard_flags"]["prefill_expert_stage_tiling"] is True
    assert "--prefill-expert-stage-tiling" in check["suggested_prefill_guard_flags"]["argv"]


def test_prepared_server_request_check_rejects_low_prefill_stage_temp_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server._system_disk_usage",
        lambda path: {
            "path": str(path),
            "total_bytes": 8192,
            "used_bytes": 4096,
            "free_bytes": 1024,
        },
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_prompt_chunk_tokens=2,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prompt prefill temp disk free space is below",
    ):
        app.inspect_token_request(
            prompt_token_count=4,
            payload={"max_new_tokens": 0},
        )


def test_prepared_server_request_check_rejects_unverified_prefill_stage_temp_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr("largerlm.server._system_disk_usage", lambda path: None)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_prompt_chunk_tokens=2,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="could not verify prompt prefill temp disk",
    ):
        app.inspect_token_request(
            prompt_token_count=4,
            payload={"max_new_tokens": 0},
        )


def test_prepared_server_request_check_rejects_routed_read_amplification_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_max_routed_read_amplification=1.5,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match=(
            "prefill routed expert read amplification 2 exceeds cap 1.5.*"
            "prefill_prompt_chunk_tokens>=4"
        ),
    ):
        app.inspect_token_request(
            prompt_token_count=4,
            payload={"max_new_tokens": 0},
        )


def test_prepared_server_request_check_suggests_routed_read_seconds_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_ssd_read_gib_per_second=16.0,
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=4,
        payload={"max_new_tokens": 0},
    )

    routed = check["prefill_routed_expert_read"]
    expected_seconds = 32 / (16 * 1024**3)
    assert routed["planned_read_seconds"] == pytest.approx(expected_seconds)
    suggested = check["suggested_guard_flags"]
    assert suggested["source"] == "prepared_request_check"
    assert suggested["prefill_ssd_read_gib_per_second"] == 16.0
    assert suggested["planned_routed_read_seconds"] == pytest.approx(expected_seconds)
    assert suggested["prefill_max_routed_read_seconds"] == pytest.approx(
        expected_seconds * 1.05
    )
    assert "--prefill-ssd-read-gib-s" in suggested["argv"]
    assert "--prefill-max-routed-read-seconds" in suggested["argv"]
    combined = check["suggested_prefill_guard_flags"]
    assert combined["source"] == "prepared_request_check"
    assert combined["prefill_prompt_chunk_tokens"] == 2
    assert "--prefill-ssd-read-gib-s" in combined["argv"]
    assert "--prefill-max-routed-read-seconds" in combined["argv"]
    assert "--prefill-max-stage-mib" in combined["argv"]


def test_prepared_server_request_check_rejects_decode_routed_read_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fake_check_generation_runtime(**kwargs):
        return SimpleNamespace(
            requested_context_tokens=kwargs["requested_context_tokens"],
            layers=(1,),
            dense_layers=(),
            max_layer_peak_bytes=4096,
            max_layer_cache_read_bytes=24,
            read_bytes_per_token=384,
            final_logits_budget=SimpleNamespace(estimated_peak_bytes=128),
            embedding_budget=SimpleNamespace(row_bytes=32, output_bytes=32),
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=4096,
                max_live_working_set_bytes=8192,
                min_available_memory_bytes=0,
                system_available_bytes=128 * 1024**3,
                system_total_bytes=128 * 1024**3,
                system_source="test",
            ),
        )

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fake_check_generation_runtime,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            decode_max_routed_read_gib_per_token=1e-9,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="decode routed expert read .* bytes/token exceeds cap",
    ):
        app.inspect_token_request(
            prompt_token_count=1,
            payload={"max_new_tokens": 1},
            runtime_preflight=False,
        )


def test_prepared_server_request_check_reports_decode_routed_read_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_check_generation_runtime(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            requested_context_tokens=kwargs["requested_context_tokens"],
            layers=(1,),
            dense_layers=(),
            max_layer_peak_bytes=4096,
            max_layer_cache_read_bytes=24,
            read_bytes_per_token=384,
            final_logits_budget=SimpleNamespace(estimated_peak_bytes=128),
            embedding_budget=SimpleNamespace(row_bytes=32, output_bytes=32),
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=4096,
                max_live_working_set_bytes=8192,
                min_available_memory_bytes=0,
                system_available_bytes=128 * 1024**3,
                system_total_bytes=128 * 1024**3,
                system_source="test",
            ),
        )

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fake_check_generation_runtime,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_ssd_read_gib_per_second=16.0,
            decode_max_routed_read_gib_per_token=1.0,
            decode_max_routed_read_seconds_per_token=1.0,
            decode_mla_key_cache=True,
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=1,
        payload={"max_new_tokens": 1},
        runtime_preflight=False,
    )

    assert check["runtime_preflight"]["ran"] is True
    assert captured["decode_mla_key_cache"] is True
    assert (
        check["runtime_preflight"]["required_for_decode_routed_read_guard"]
        is True
    )
    decode = check["decode_routed_expert_read"]
    assert decode["read_bytes_per_token"] == 384
    assert decode["max_read_bytes_per_token"] == 1024**3
    assert decode["planned_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3)
    )
    assert decode["within_limit"] is True
    suggested = check["suggested_decode_guard_flags"]
    assert suggested["source"] == "prepared_request_check"
    assert suggested["decode_read_bytes_per_token"] == 384
    assert suggested["decode_max_routed_read_gib_per_token"] == pytest.approx(
        384 / 1024**3 * 1.05
    )
    assert suggested["decode_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3)
    )
    assert suggested["decode_max_routed_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3) * 1.05
    )
    assert suggested["decode_mla_key_cache"] is True
    assert "--decode-max-routed-read-gib-per-token" in suggested["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in suggested["argv"]
    assert "--decode-mla-key-cache" in suggested["argv"]


def test_prepared_server_runtime_preflight_runs_for_prefill_only_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_check_generation_runtime(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            requested_context_tokens=kwargs["requested_context_tokens"],
            layers=(1,),
            dense_layers=(),
            max_layer_peak_bytes=4096,
            max_layer_cache_read_bytes=24,
            read_bytes_per_token=10 * 1024**3,
            final_logits_budget=SimpleNamespace(estimated_peak_bytes=128),
            embedding_budget=SimpleNamespace(row_bytes=32, output_bytes=32),
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=kwargs[
                    "extra_live_working_set_bytes"
                ],
                max_live_working_set_bytes=128 * 1024**2,
                min_available_memory_bytes=0,
                system_available_bytes=128 * 1024**3,
                system_total_bytes=128 * 1024**3,
                system_source="test",
            ),
        )

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fake_check_generation_runtime,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_max_prompt_batch_mib=16.0,
            max_runner_scratch_mib=32.0,
            max_cache_read_mib=4.0,
            prefill_max_cache_write_mib=8.0,
            prefill_copy_chunk_mib=1.0,
            decode_max_routed_read_gib_per_token=1e-9,
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=4,
        payload={"max_new_tokens": 0},
        runtime_preflight=False,
    )

    assert check["runtime_preflight"]["ran"] is True
    assert check["runtime_preflight"]["requested_context_tokens"] == 4
    assert check["runtime_preflight"]["prefill_live_memory"] == {
        "prompt_batch_bytes": 16 * 1024**2,
        "runner_scratch_bytes": 32 * 1024**2,
        "cache_read_bytes": 4 * 1024**2,
        "cache_write_bytes": 8 * 1024**2,
        "stage_copy_bytes": 1 * 1024**2,
        "estimated_live_working_set_bytes": 48 * 1024**2,
    }
    assert captured["requested_context_tokens"] == 4
    assert captured["extra_live_working_set_bytes"] == 48 * 1024**2
    assert check["decode_routed_expert_read"] is None
    assert check["suggested_decode_guard_flags"] is None


def test_prepared_server_request_check_rejects_routed_read_gib_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_max_routed_read_amplification=3.0,
            prefill_max_routed_read_gib=1e-8,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match=(
            "prefill routed expert planned read 32 bytes exceeds cap 10 bytes.*"
            "no prompt chunk size"
        ),
    ):
        app.inspect_token_request(
            prompt_token_count=4,
            payload={"max_new_tokens": 0},
        )


def test_prepared_server_request_check_rejects_routed_read_seconds_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=2,
            prefill_ssd_read_gib_per_second=1.0,
            prefill_max_routed_read_seconds=1e-8,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match=(
            "prefill routed expert planned read time .* exceeds cap 1e-08s.*"
            "no prompt chunk size"
        ),
    ):
        app.inspect_token_request(
            prompt_token_count=4,
            payload={"max_new_tokens": 0},
        )


def test_prefill_linear_backend_summary_rejects_boolean_shape(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_bin_path = prepared / "resident" / "resident.bin"
    payload = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    offset = int(payload["total_bytes"])
    payload["tensors"].append(
        {
            "name": "model.layers.0.self_attn.q_a_proj.weight",
            "offset": offset,
            "size": 16,
            "dtype": "F32",
            "shape": [True, 4],
            "category": "attention",
        }
    )
    payload["total_bytes"] = offset + 16
    resident_layout_path.write_text(json.dumps(payload), encoding="utf-8")
    resident_bin_path.write_bytes(resident_bin_path.read_bytes() + b"\0" * 16)

    with pytest.raises(PreparedServerError, match="shape must use integer rows and cols"):
        _prefill_linear_backend_request_summary(
            resident_layout_path=resident_layout_path,
            configured_backend="auto",
            effective_backend="custom-metal",
            prompt_chunk_tokens=1,
            mpsgraph_min_batch_tokens=128,
            mpsgraph_min_matrix_dim=32,
        )


def test_prefill_linear_backend_summary_reports_mpp_tensor_ops_candidates(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)

    summary = _prefill_linear_backend_request_summary(
        resident_layout_path=prepared / "resident" / "layout.json",
        configured_backend="auto",
        effective_backend="auto",
        prompt_chunk_tokens=128,
        mpsgraph_min_batch_tokens=128,
        mpsgraph_min_matrix_dim=32,
    )

    expected_flops = 2 * 128 * 32 * 32
    assert summary["mpp_candidate_policy"] == {
        "candidate_backend": "mpp_tensor_ops_prefill",
        "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
        "mpp_tensor_ops_min_batch_tokens": 128,
        "mpp_tensor_ops_min_matrix_dim": 32,
        "selectable_prefill_backend": False,
    }
    assert summary["matrix_count"] == 1
    assert summary["mpp_tensor_ops_candidate_matrix_count"] == 1
    assert summary["mpp_tensor_ops_candidate_estimated_flops"] == expected_flops
    assert summary["mpp_tensor_ops_candidate_flop_fraction"] == 1.0
    assert summary["mpp_tensor_ops_candidate_backend_counts"] == {"mpsgraph-f32": 1}
    assert summary["mpp_tensor_ops_candidate_backend_flops"] == {
        "mpsgraph-f32": expected_flops
    }
    assert summary["total_matrix_scratch_bytes"] == 2 * 1024 * 1024 + 32 * 32 * 2
    assert summary["top_matrices"][0]["mpp_tensor_ops_candidate"] is True


def test_prefill_linear_backend_summary_excludes_router_gate_candidates(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_bin_path = prepared / "resident" / "resident.bin"
    payload = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    offset = int(payload["total_bytes"])
    matrix_size = 256 * 6144 * 2
    payload["tensors"].append(
        {
            "name": "model.layers.0.mlp.gate.weight",
            "offset": offset,
            "size": matrix_size,
            "dtype": "BF16",
            "shape": [256, 6144],
            "category": "router",
        }
    )
    payload["total_bytes"] = offset + matrix_size
    resident_layout_path.write_text(json.dumps(payload), encoding="utf-8")
    resident_bin_path.write_bytes(resident_bin_path.read_bytes() + b"\0" * matrix_size)

    summary = _prefill_linear_backend_request_summary(
        resident_layout_path=resident_layout_path,
        configured_backend="auto",
        effective_backend="auto",
        prompt_chunk_tokens=16,
        mpsgraph_min_batch_tokens=32,
        mpsgraph_min_matrix_dim=32,
    )

    assert summary["matrix_count"] == 0
    assert summary["mpsgraph_matrix_count"] == 0
    assert summary["accelerated_matrix_count"] == 0

    accelerated = _prefill_linear_backend_request_summary(
        resident_layout_path=resident_layout_path,
        configured_backend="auto",
        effective_backend="auto",
        prompt_chunk_tokens=16,
        mpsgraph_min_batch_tokens=16,
        mpsgraph_min_matrix_dim=32,
    )

    assert accelerated["matrix_count"] == 1
    assert accelerated["mpsgraph_matrix_count"] == 1
    assert accelerated["accelerated_matrix_count"] == 1


def test_prepared_server_payload_reports_prompt_prefill_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    prompt_prefill = SimpleNamespace(
        elapsed_seconds=8.0,
        chunk_count=1,
        chunk_tokens=2,
        linear_backend_counts={"mpsgraph-f32": 4},
        linear_backend_flops={"mpsgraph-f32": 8192},
        linear_backend_elapsed_seconds={"mpsgraph-f32": 0.004},
        linear_backend_estimated_tflops={"mpsgraph-f32": 8192 / 0.004 / 1e12},
        total_linear_estimated_flops=8192,
        accelerated_linear_estimated_flops=8192,
        custom_linear_estimated_flops=0,
        unsupported_linear_estimated_flops=0,
        accelerated_linear_flop_fraction=1.0,
        prefill_acceleration_coverage={
            "ok": True,
            "matrix_count": 4,
            "accelerated_matrix_count": 4,
            "total_estimated_flops": 8192,
            "accelerated_estimated_flops": 8192,
            "accelerated_flop_fraction": 1.0,
            "dominant_resident_flops_accelerated": True,
            "any_resident_matrix_accelerated": True,
        },
        prefill_acceleration_frontier={
            "source": "prompt_prefill_actual",
            "resolved_prompt_chunk_tokens": 2,
            "minimum_accelerated_prompt_chunk_tokens": 2,
        },
        chunks=(
            SimpleNamespace(
                layers=(
                    SimpleNamespace(
                        attention=SimpleNamespace(
                            mla_key_cache=True,
                            mla_key_cache_bytes=16,
                            mla_value_cache=True,
                            mla_value_cache_bytes=32,
                        )
                    ),
                    SimpleNamespace(
                        attention=SimpleNamespace(
                            mla_key_cache=False,
                            mla_key_cache_bytes=0,
                            mla_value_cache=True,
                            mla_value_cache_bytes=32,
                        )
                    ),
                )
            ),
        ),
        total_linear_matrix_scratch_bytes=4096,
        max_linear_matrix_scratch_bytes=2048,
        total_linear_matrix_f32_bytes=1024,
        total_linear_matrix_raw_conversion_bytes=512,
        total_staged_bytes=300,
        total_compact_stage_bytes=200,
        total_compact_stage_materialized_bytes=0,
        max_staged_bytes=150,
        max_compact_stage_bytes=100,
        max_compact_stage_materialized_bytes=0,
        total_stage_plus_compact_bytes=500,
        total_stage_plus_compact_materialized_bytes=300,
        max_stage_plus_compact_bytes=250,
        max_stage_plus_compact_materialized_bytes=150,
        total_expert_stage_planned_read_bytes=100,
        total_expert_stage_planned_read_seconds=0.25,
        total_expert_stage_copy_elapsed_seconds=0.05,
        total_expert_stage_copy_throughput_gib_per_second=(
            (300 / 1024**3) / 0.05
        ),
        prefill_ssd_read_gib_per_second=16.0,
        prefill_max_routed_read_seconds=1.0,
        total_expert_stage_read_seconds_ok=True,
        total_expert_stage_copy_seconds_ok=True,
        total_expert_stage_unique_requested_bytes=80,
        total_expert_stage_waste_bytes=20,
        total_expert_stage_unique_read_amplification=1.25,
        total_expert_stage_read_advice_attempted_ranges=3,
        total_expert_stage_read_advice_calls=2,
        total_expert_stage_read_advice_bytes=96,
        total_expert_stage_read_advice_failures=1,
        total_routed_expert_assignments=16,
        max_moe_estimated_peak_bytes=2048,
        static_capacity_per_expert="auto",
    )

    def fake_generate_token_ids(**kwargs):
        return _token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            prompt_prefill=prompt_prefill,
            auto_prefill_prompt_chunk_plan=_fake_auto_prefill_chunk_plan(2),
            max_safe_prefill_prompt_chunk_plan=_fake_auto_prefill_chunk_plan(3),
            prefill_actual_read_time={
                "source": "generation_actual_prefill",
                "total_expert_stage_planned_read_bytes": 100,
                "total_expert_stage_planned_read_seconds": 0.25,
                "total_expert_stage_copy_elapsed_seconds": 0.05,
                "total_expert_stage_copy_throughput_gib_per_second": (
                    (300 / 1024**3) / 0.05
                ),
                "prefill_ssd_read_gib_per_second": 16.0,
                "prefill_max_routed_read_seconds": 1.0,
                "total_expert_stage_read_seconds_ok": True,
                "total_expert_stage_copy_seconds_ok": True,
            },
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            applied_launch_profile=_applied_launch_profile(tmp_path),
        )
    )

    payload = app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )

    prefill_payload = payload["prompt_prefill"]
    assert prefill_payload["elapsed_seconds"] == 8.0
    assert prefill_payload["linear_backend_counts"] == {"mpsgraph-f32": 4}
    assert prefill_payload["linear_backend_flops"] == {"mpsgraph-f32": 8192}
    assert prefill_payload["linear_backend_elapsed_seconds"] == {
        "mpsgraph-f32": 0.004
    }
    assert prefill_payload["linear_backend_estimated_tflops"] == {
        "mpsgraph-f32": 8192 / 0.004 / 1e12
    }
    assert prefill_payload["total_linear_estimated_flops"] == 8192
    assert prefill_payload["accelerated_linear_estimated_flops"] == 8192
    assert prefill_payload["custom_linear_estimated_flops"] == 0
    assert prefill_payload["unsupported_linear_estimated_flops"] == 0
    assert prefill_payload["accelerated_linear_flop_fraction"] == 1.0
    assert prefill_payload["prefill_acceleration_coverage"] == {
        "ok": True,
        "matrix_count": 4,
        "accelerated_matrix_count": 4,
        "total_estimated_flops": 8192,
        "accelerated_estimated_flops": 8192,
        "accelerated_flop_fraction": 1.0,
        "dominant_resident_flops_accelerated": True,
        "any_resident_matrix_accelerated": True,
    }
    assert prefill_payload["prefill_acceleration_frontier"] == {
        "source": "prompt_prefill_actual",
        "resolved_prompt_chunk_tokens": 2,
        "minimum_accelerated_prompt_chunk_tokens": 2,
    }
    assert prefill_payload["mla_key_cache"] == {
        "observed": True,
        "layer_count": 2,
        "enabled_layer_count": 1,
        "disabled_layer_count": 1,
        "all_layers_enabled": False,
        "total_mla_key_cache_bytes": 16,
    }
    assert prefill_payload["mla_value_cache"] == {
        "observed": True,
        "layer_count": 2,
        "enabled_layer_count": 2,
        "disabled_layer_count": 0,
        "all_layers_enabled": True,
        "total_mla_value_cache_bytes": 64,
    }
    assert prefill_payload["total_linear_matrix_scratch_bytes"] == 4096
    assert prefill_payload["max_linear_matrix_scratch_bytes"] == 2048
    assert prefill_payload["total_linear_matrix_f32_bytes"] == 1024
    assert prefill_payload["total_linear_matrix_raw_conversion_bytes"] == 512
    assert prefill_payload["total_staged_bytes"] == 300
    assert prefill_payload["total_compact_stage_bytes"] == 200
    assert prefill_payload["total_compact_stage_materialized_bytes"] == 0
    assert prefill_payload["max_staged_bytes"] == 150
    assert prefill_payload["max_compact_stage_bytes"] == 100
    assert prefill_payload["max_compact_stage_materialized_bytes"] == 0
    assert prefill_payload["total_stage_plus_compact_bytes"] == 500
    assert prefill_payload["total_stage_plus_compact_materialized_bytes"] == 300
    assert prefill_payload["max_stage_plus_compact_bytes"] == 250
    assert prefill_payload["max_stage_plus_compact_materialized_bytes"] == 150
    assert prefill_payload["total_expert_stage_planned_read_bytes"] == 100
    assert prefill_payload["total_expert_stage_planned_read_seconds"] == 0.25
    assert prefill_payload["total_expert_stage_copy_elapsed_seconds"] == 0.05
    assert prefill_payload[
        "total_expert_stage_copy_throughput_gib_per_second"
    ] == pytest.approx((300 / 1024**3) / 0.05)
    assert prefill_payload["prefill_ssd_read_gib_per_second"] == 16.0
    assert prefill_payload["prefill_max_routed_read_seconds"] == 1.0
    assert prefill_payload["total_expert_stage_read_seconds_ok"] is True
    assert prefill_payload["total_expert_stage_copy_seconds_ok"] is True
    assert payload["prefill_actual_read_time"] == {
        "source": "generation_actual_prefill",
        "total_expert_stage_planned_read_bytes": 100,
        "total_expert_stage_planned_read_seconds": 0.25,
        "total_expert_stage_copy_elapsed_seconds": 0.05,
        "total_expert_stage_copy_throughput_gib_per_second": (
            (300 / 1024**3) / 0.05
        ),
        "prefill_ssd_read_gib_per_second": 16.0,
        "prefill_max_routed_read_seconds": 1.0,
        "total_expert_stage_read_seconds_ok": True,
        "total_expert_stage_copy_seconds_ok": True,
    }
    assert prefill_payload["total_expert_stage_read_advice_attempted_ranges"] == 3
    assert prefill_payload["total_expert_stage_read_advice_calls"] == 2
    assert prefill_payload["total_expert_stage_read_advice_bytes"] == 96
    assert prefill_payload["total_expert_stage_read_advice_failures"] == 1
    auto_plan = payload["auto_prefill_prompt_chunk_plan"]
    assert auto_plan["chunk_tokens"] == 2
    assert auto_plan["limiting_cap_names"] == ("test_cap",)
    assert auto_plan["max_matrix_scratch_bytes"] == 2 * 1024 * 1024
    max_safe_plan = payload["max_safe_prefill_prompt_chunk_plan"]
    assert max_safe_plan["chunk_tokens"] == 3
    assert max_safe_plan["limiting_cap_names"] == ("test_cap",)
    drift = payload["prefill_prompt_chunk_plan_drift"]
    assert drift["source"] == "applied_launch_profile"
    assert drift["status"] == "changed"
    assert drift["profile_max_safe_chunk_tokens"] == 2
    assert drift["actual_prompt_chunk_tokens"] == 2
    assert drift["actual_max_safe_chunk_tokens"] == 3
    assert drift["current_max_safe_at_least_profile"] is True
    assert drift["selected_chunk_within_current_max_safe"] is True


def test_prepared_server_health_reports_effective_context_limits(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    health = app.health()

    assert health["prepared_max_context_tokens"] == 4
    assert health["decode_cache_context_tokens"] == 4
    assert health["model_context_tokens"] is None
    assert health["effective_context_tokens"] == 4
    assert health["effective_max_prompt_tokens"] == 4
    assert health["prepared_storage"] == {
        "prepared_storage_validated": True,
        "expert_layout_backing_validated": True,
        "resident_layout_backing_validated": True,
        "decode_cache_file_exact_size": True,
        "expert_layout_bytes": 16,
        "resident_layout_bytes": 16,
        "decode_cache_layout_bytes": 32,
        "decode_cache_file_bytes": 32,
        "decode_cache_file_extra_bytes": 0,
        "prepared_total_layout_bytes": 64,
        "prepared_total_file_bytes": 64,
        "resident_and_cache_layout_bytes": 48,
        "recommended_max_live_working_set_bytes": None,
        "recommended_min_free_unified_memory_bytes": None,
        "recommended_required_available_memory_bytes": None,
        "expert_quantization": None,
        "expert_group_size": None,
        "expert_layout_quantization": "mlx-affine-int4",
        "expert_layout_group_size": 8,
        "prepare_hardware_chip_name": None,
        "prepare_hardware_unified_memory_bytes": None,
        "prepare_hardware_gpu_cores": None,
        "prepare_hardware_apple_silicon_generation": None,
        "prepare_hardware_apple_silicon_tier": None,
        "prepare_effective_unified_memory_bytes": None,
        "prepare_effective_unified_memory_source": None,
        "prepare_system_reserve_bytes": None,
        "prepare_auto_context_from_budget": None,
        "prepare_requested_max_context_tokens": None,
        "prepare_resolved_max_context_tokens": None,
        "prepare_decode_cache_budget_bytes": None,
        "prepare_decode_cache_safe_context_tokens": None,
        "prepare_effective_max_cache_bytes": None,
        "prepare_model_max_position_embeddings": None,
        "prepare_cache_dtype": None,
        "prepare_cache_alignment": None,
        "prepare_flags_applied": None,
        "prepare_flags_source": None,
        "prepare_flags_path": None,
        "prepare_flags_sha256": None,
        "prepare_expert_pack_chunk_size_bytes": None,
        "prepare_expert_pack_estimated_peak_heap_bytes": None,
        "prepare_expert_pack_max_heap_bytes": None,
        "prepare_raw_quantization_extra_heap_bytes": None,
        "prepare_raw_quantization_max_source_block_bytes": None,
        "prepare_raw_quantization_max_output_block_bytes": None,
        "prepare_raw_quantization_max_rows_per_block": None,
        "prepare_resident_component_alias_source_tensor_count": None,
        "prepare_resident_component_alias_renamed_tensor_count": None,
        "prepare_resident_component_alias_bytes": None,
        "prepare_resident_fused_gate_up_source_tensor_count": None,
        "prepare_resident_fused_gate_up_expanded_tensor_count": None,
        "prepare_resident_fused_gate_up_expanded_bytes": None,
        "prepare_public_glm_5_2_shape_required": None,
        "prepare_public_glm_5_2_shape_matches": None,
        "prepare_public_glm_5_2_shape_mismatched_fields": None,
        "prepare_cold_read_gib_per_second": None,
        "prepare_cold_read_source": None,
        "prepare_cold_read_benchmark_path": None,
        "prepare_cold_read_benchmark_requested_bytes": None,
        "prepare_cold_read_benchmark_measured_bytes": None,
        "prepare_cold_read_benchmark_elapsed_seconds": None,
        "model_config_sha256": None,
    }
    assert health["prepare_live_memory"]["recorded"] is False


def test_prepared_server_health_reports_prepare_live_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_live_memory_estimated_live_working_set_bytes": 1024,
            "prepare_live_memory_min_available_memory_bytes": 2048,
            "prepare_live_memory_required_available_memory_bytes": 3072,
            "prepare_live_memory_system_available_memory_bytes": 8192,
            "prepare_live_memory_system_total_bytes": 16384,
            "prepare_live_memory_system_source": "prepare-test",
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server._system_memory_health",
        lambda: {
            "available_bytes": 4096,
            "total_bytes": 16384,
            "source": "health-test",
        },
    )

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    live = app.health()["prepare_live_memory"]

    assert live == {
        "recorded": True,
        "estimated_live_working_set_bytes": 1024,
        "min_available_memory_bytes": 2048,
        "required_available_memory_bytes": 3072,
        "prepare_system_available_memory_bytes": 8192,
        "prepare_system_total_memory_bytes": 16384,
        "prepare_system_memory_source": "prepare-test",
        "current_system_available_memory_bytes": 4096,
        "current_system_memory_source": "health-test",
        "current_available_meets_prepare_live_requirement": True,
    }


def test_prepared_server_require_glm_4bit_rejects_incomplete_layout(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(
        PreparedServerError,
        match="prepared GLM 4bit readiness failed: .*expert layout layers do not match",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_allows_ready_layout(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
            prefill_ssd_read_gib_per_second=16.0,
            decode_mla_key_cache=True,
        )
    )

    health = app.health()
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is True
    glm_guard = health["suggested_glm_4bit_guard_flags"]
    assert glm_guard == {
        "source": "prepared_health",
        "require_glm_4bit": True,
        "argv": ("--require-glm-4bit",),
    }
    suggested = health["suggested_decode_guard_flags"]
    assert suggested["source"] == "prepared_health"
    assert suggested["decode_read_bytes_per_token"] == (
        readiness["expected_decode_token_routed_expert_read_bytes"]
    )
    assert suggested["decode_read_seconds_per_token"] == pytest.approx(
        readiness["expected_decode_token_routed_expert_read_bytes"] / (16 * 1024**3)
    )
    assert "--decode-max-routed-read-gib-per-token" in suggested["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in suggested["argv"]
    assert suggested["decode_mla_key_cache"] is True
    assert "--decode-mla-key-cache" in suggested["argv"]
    ssd_flags = health["suggested_prepared_ssd_read_flags"]
    assert ssd_flags == {
        "source": "configured_runtime",
        "prefill_ssd_read_gib_per_second": 16.0,
        "prepare_cold_read_gib_per_second": None,
        "prepare_cold_read_source": None,
        "matches_prepare_cold_read": False,
        "argv": ("--prefill-ssd-read-gib-s", "16"),
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prepared_ssd_read_flags"] == ssd_flags
    assert profile["sections"]["glm_4bit_guard_flags"] == glm_guard
    assert profile["sections"]["decode_guard_flags"] == suggested
    assert "--prefill-ssd-read-gib-s" in profile["argv"]
    assert "--require-glm-4bit" in profile["argv"]
    assert "--decode-mla-key-cache" in profile["argv"]


def test_prepared_server_require_glm_4bit_allows_mxfp4_layout(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_mxfp4_glm_config(prepared.parent / "model" / "config.json")
    _make_prepared_glm_4bit_ready(
        prepared,
        expert_quantization="mlx-mxfp4",
        expert_group_size=32,
    )

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
            prefill_ssd_read_gib_per_second=16.0,
        )
    )

    readiness = app.health()["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["issues"] == []
    assert readiness["expert_layout_quantization"] == "mlx-mxfp4"
    assert readiness["expert_layout_group_size"] == 32
    assert readiness["expected_expert_slot_bytes"] == 1632
    assert readiness["expected_expert_layer_bytes"] == 3264
    assert readiness["expected_total_expert_bytes"] == 6528
    assert readiness["expected_decode_token_routed_expert_read_bytes"] == 6528
    assert readiness["prepared_expert_layout_bytes"] == 6528


def test_prepared_server_require_glm_4bit_accepts_mxfp4_resident_embedding(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_mxfp4_glm_config(prepared.parent / "model" / "config.json")
    _make_prepared_glm_4bit_ready(
        prepared,
        expert_quantization="mlx-mxfp4",
        expert_group_size=32,
    )
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    embed = next(
        tensor
        for tensor in resident_layout["tensors"]
        if tensor["name"] == "model.embed_tokens.weight"
    )
    embed["dtype"] = "U32"
    embed["shape"] = [4, 4]
    embed["size"] = 64
    scales_offset = resident_layout["total_bytes"]
    resident_layout["tensors"].append(
        {
            "name": "model.embed_tokens.scales",
            "offset": scales_offset,
            "size": 4,
            "dtype": "U8",
            "shape": [4, 1],
            "category": "embedding",
        }
    )
    resident_layout["total_bytes"] = scales_offset + 4
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")
    (prepared / "resident" / "resident.bin").write_bytes(
        b"\0" * resident_layout["total_bytes"]
    )

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
        )
    )

    assert app.health()["glm_4bit_readiness"]["ok"] is True


def test_prepared_server_require_glm_4bit_accepts_absorbed_attention_aliases(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    _replace_kv_b_with_absorbed_aliases(prepared, mxfp4=False)

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
        )
    )

    assert app.health()["glm_4bit_readiness"]["ok"] is True


def test_prepared_server_require_glm_4bit_accepts_mxfp4_absorbed_attention_aliases(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_mxfp4_glm_config(prepared.parent / "model" / "config.json")
    _make_prepared_glm_4bit_ready(
        prepared,
        expert_quantization="mlx-mxfp4",
        expert_group_size=32,
    )
    _replace_kv_b_with_absorbed_aliases(prepared, mxfp4=True)

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
        )
    )

    assert app.health()["glm_4bit_readiness"]["ok"] is True


def test_prepared_server_require_glm_4bit_rejects_component_offset_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    components = expert_layout["layers"][0]["components"]
    components[1]["offset"], components[2]["offset"] = (
        components[2]["offset"],
        components[1]["offset"],
    )
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="component gate_proj\\.scales offset",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_mxfp4_component_order_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_mxfp4_glm_config(prepared.parent / "model" / "config.json")
    _make_prepared_glm_4bit_ready(
        prepared,
        expert_quantization="mlx-mxfp4",
        expert_group_size=32,
    )
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    component_order = expert_layout["component_order"]
    component_order[1], component_order[2] = component_order[2], component_order[1]
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="component_order does not match mlx-mxfp4 slot order",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_accepts_f16_affine_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    for layer in expert_layout["layers"]:
        for component in layer["components"]:
            if component["name"].endswith((".scales", ".biases")):
                component["dtype"] = "F16"
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
        )
    )

    assert app.health()["glm_4bit_readiness"]["ok"] is True


def test_prepared_server_require_glm_4bit_accepts_uint32_affine_weights(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    for layer in expert_layout["layers"]:
        for component in layer["components"]:
            if component["name"].endswith(".weight"):
                component["dtype"] = "uint32"
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_glm_4bit=True,
        )
    )

    assert app.health()["glm_4bit_readiness"]["ok"] is True


def test_prepared_server_require_glm_4bit_rejects_component_order_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    component_order = expert_layout["component_order"]
    component_order[1], component_order[2] = component_order[2], component_order[1]
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="component_order does not match affine-int4 slot order",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_expert_model_type_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    expert_layout["model_type"] = "other_model"
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="expert layout model_type",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_resident_model_type_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    resident_layout["model_type"] = "other_model"
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="resident layout model_type",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_missing_layout_config_hash(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("model_config_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for layout_path in (
        prepared / "experts" / "layout.json",
        prepared / "resident" / "layout.json",
    ):
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        layout["config_sha256"] = None
        layout_path.write_text(json.dumps(layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="expert layout missing config_sha256",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_missing_manifest_expert_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("expert_quantization")
    manifest.pop("expert_group_size")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="manifest missing expert_quantization",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_public_glm_5_2_shape_report_accepts_fixture() -> None:
    config = load_config(FIXTURES / "glm_5_2_config.json")

    report = _public_glm_5_2_shape_report(config)

    assert _is_public_glm_5_2_shape(config) is True
    assert report["matches"] is True
    assert report["mismatched_fields"] == ()
    checks = report["checks"]
    assert checks["attention_q_projection_output_dim"] == {
        "actual": 16384,
        "expected": 16384,
        "matches": True,
    }
    assert checks["weight_dtype"] == {
        "actual": "bfloat16",
        "expected": "bfloat16",
        "matches": True,
    }
    assert checks["hidden_act"] == {
        "actual": "silu",
        "expected": "silu",
        "matches": True,
    }
    assert checks["attention_bias"] == {
        "actual": False,
        "expected": False,
        "matches": True,
    }
    assert checks["attention_dropout"] == {
        "actual": 0.0,
        "expected": 0.0,
        "matches": True,
    }
    assert checks["attention_kv_b_output_dim"]["actual"] == 28672
    assert checks["indexer_types"]["matches"] is True
    assert checks["full_indexer_layers"]["actual"][:4] == (0, 1, 2, 6)
    assert checks["index_topk_freq"] == {
        "actual": 4,
        "expected": 4,
        "matches": True,
    }
    assert checks["index_skip_topk_offset"] == {
        "actual": 3,
        "expected": 3,
        "matches": True,
    }
    assert checks["num_nextn_predict_layers"] == {
        "actual": 1,
        "expected": 1,
        "matches": True,
    }
    assert report["dsa_full_indexer_layer_count"] == 21
    assert report["dsa_full_indexer_layers"][:4] == (0, 1, 2, 6)
    assert report["dsa_schedule"] == {
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "full_indexer_layer_count": 21,
        "first_full_indexer_layers": (0, 1, 2, 6, 10, 14, 18, 22),
        "last_full_indexer_layers": (62, 66, 70, 74),
    }
    assert checks["scoring_func"]["actual"] == "sigmoid"
    assert checks["routed_scaling_factor"]["actual"] == 2.5


def test_mla_kv_b_cache_plan_uses_public_glm_5_2_shape() -> None:
    config = load_config(FIXTURES / "glm_5_2_config.json")

    plan = _estimate_mla_kv_b_f32_cache_plan(config)

    assert plan == {
        "source": "model_config",
        "num_hidden_layers": 78,
        "kv_lora_rank": 512,
        "attention_kv_b_output_dim": 28672,
        "per_layer_bytes": 58_720_256,
        "estimated_full_cache_bytes": 4_580_179_968,
        "estimated_full_cache_mib": 4368.0,
        "recommended_max_cache_mib": 4608.0,
    }


def test_public_glm_5_2_shape_report_rejects_semantic_drift(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "glm_5_2_config.json").read_text(encoding="utf-8"))
    payload["q_lora_rank"] = 1024
    payload["n_group"] = 2
    payload["hidden_act"] = "gelu"
    payload["attention_bias"] = True
    payload["attention_dropout"] = 0.1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)

    report = _public_glm_5_2_shape_report(config)

    assert _is_public_glm_5_2_shape(config) is False
    assert report["matches"] is False
    assert "q_lora_rank" in report["mismatched_fields"]
    assert "n_group" in report["mismatched_fields"]
    assert "hidden_act" in report["mismatched_fields"]
    assert "attention_bias" in report["mismatched_fields"]
    assert "attention_dropout" in report["mismatched_fields"]
    assert report["checks"]["q_lora_rank"] == {
        "actual": 1024,
        "expected": 2048,
        "matches": False,
    }
    assert report["checks"]["n_group"] == {
        "actual": 2,
        "expected": 1,
        "matches": False,
    }
    assert report["checks"]["hidden_act"] == {
        "actual": "gelu",
        "expected": "silu",
        "matches": False,
    }
    assert report["checks"]["attention_bias"] == {
        "actual": True,
        "expected": False,
        "matches": False,
    }
    assert report["checks"]["attention_dropout"] == {
        "actual": 0.1,
        "expected": 0.0,
        "matches": False,
    }


def test_public_glm_5_2_shape_report_rejects_dsa_schedule_raw_drift(
    tmp_path: Path,
) -> None:
    payload = json.loads((FIXTURES / "glm_5_2_config.json").read_text(encoding="utf-8"))
    payload["index_topk_freq"] = 8
    payload["index_skip_topk_offset"] = 1
    payload["num_nextn_predict_layers"] = 2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)

    report = _public_glm_5_2_shape_report(config)

    assert _is_public_glm_5_2_shape(config) is False
    assert report["matches"] is False
    assert "index_topk_freq" in report["mismatched_fields"]
    assert "index_skip_topk_offset" in report["mismatched_fields"]
    assert "num_nextn_predict_layers" in report["mismatched_fields"]
    assert report["checks"]["index_topk_freq"] == {
        "actual": 8,
        "expected": 4,
        "matches": False,
    }
    assert report["checks"]["index_skip_topk_offset"] == {
        "actual": 1,
        "expected": 3,
        "matches": False,
    }
    assert report["checks"]["num_nextn_predict_layers"] == {
        "actual": 2,
        "expected": 1,
        "matches": False,
    }


def test_prepared_server_require_public_glm_5_2_shape_rejects_non_public_ready_layout(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)

    with pytest.raises(
        PreparedServerError,
        match="prepared config does not match the public GLM-5.2 shape",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_public_glm_5_2_shape=True,
            )
        )


def test_prepared_server_public_glm_5_2_shape_profile_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_memory_profile(prepared)
    _make_prepared_glm_4bit_ready(prepared)
    _mock_safe_system_memory(monkeypatch)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_public_glm_5_2_shape=True,
        )
    )

    health = app.health()
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["matches_public_glm_5_2_shape"] is True
    assert app.state.config.require_prepared_memory_profile is True
    assert health["require_prepared_memory_profile"] is True
    guard = health["suggested_public_glm_5_2_shape_guard_flags"]
    assert guard == {
        "source": "prepared_health",
        "require_public_glm_5_2_shape": True,
        "argv": ("--require-public-glm-5-2-shape",),
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["public_glm_5_2_shape_guard_flags"] == guard
    assert "--require-public-glm-5-2-shape" in profile["argv"]


def test_prepared_server_public_glm_5_2_shape_requires_memory_profile_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            require_public_glm_5_2_shape=True,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prepared manifest is missing required memory profile fields",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_rejects_public_glm_5_2_with_missing_dsa_indexer(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(
        PreparedServerError,
        match="require_public_glm_5_2_shape cannot be used with allow_missing_dsa_indexer",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_public_glm_5_2_shape=True,
                allow_missing_dsa_indexer=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_oversized_expert_layer_file(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    layer_file = prepared / "experts" / "layer_001.bin"
    layer_file.write_bytes(layer_file.read_bytes() + b"extra")

    with pytest.raises(
        PreparedServerError,
        match="expert layout layer 1 file bytes .* expected packed layer bytes",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_missing_lm_head_when_not_tied(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = prepared.parent / "model" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tie_word_embeddings"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _make_prepared_glm_4bit_ready(prepared)

    with pytest.raises(
        PreparedServerError,
        match="resident layout missing lm_head.weight and config tie_word_embeddings=false",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_vocab_size_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = prepared.parent / "model" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["vocab_size"] = 5
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _make_prepared_glm_4bit_ready(prepared)

    with pytest.raises(
        PreparedServerError,
        match=(
            "resident tensor embed_tokens.weight vocab rows 4 "
            "does not match config vocab_size 5"
        ),
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_missing_resident_router(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    resident_layout["tensors"] = [
        tensor
        for tensor in resident_layout["tensors"]
        if tensor.get("name") != "model.layers.1.mlp.gate.weight"
    ]
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="resident layout layer 1 missing router gate.weight",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_missing_router_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    resident_layout.pop("router")
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="resident layout missing router metadata",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_router_metadata_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    resident_layout["router"]["scoring_func"] = "sigmoid"
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")

    with pytest.raises(
        PreparedServerError,
        match="resident layout router metadata scoring_func",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_require_glm_4bit_rejects_resident_routed_expert(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    resident_layout["tensors"].append(
        {
            "name": "model.layers.1.mlp.experts.0.w1.weight",
            "offset": resident_layout["total_bytes"],
            "size": 16,
            "dtype": "BF16",
            "shape": [8],
            "category": "routed_experts",
        }
    )
    resident_layout["total_bytes"] += 16
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")
    (prepared / "resident" / "resident.bin").write_bytes(
        b"\0" * resident_layout["total_bytes"]
    )

    with pytest.raises(
        PreparedServerError,
        match="resident layout contains routed expert tensors",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                require_glm_4bit=True,
            )
        )


def test_prepared_server_health_reports_system_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    health = app.health()

    assert health["system_memory"] == {
        "total_bytes": 128 * 1024**3,
        "available_bytes": 96 * 1024**3,
        "page_size": 16 * 1024,
        "source": "test",
    }
    assert health["memory_guard"] == {
        "configured_max_live_working_set_bytes": 8192 * 1024**2,
        "configured_min_free_unified_memory_bytes": 0,
        "configured_required_available_memory_bytes": 8192 * 1024**2,
        "system_available_memory_bytes": 96 * 1024**3,
        "system_total_memory_bytes": 128 * 1024**3,
        "system_memory_source": "test",
        "available_ok": True,
    }
    assert health["suggested_launch_guard_flags"] is None
    assert health["launch_audit_envelope"] is None
    assert health["prepared_runtime_profile"] == {
        "prepare_effective_unified_memory_bytes": None,
        "prepare_effective_unified_memory_source": None,
        "prepare_system_reserve_bytes": None,
        "prepared_recommended_max_live_working_set_bytes": None,
        "prepared_recommended_min_free_unified_memory_bytes": None,
        "prepared_recommended_required_available_memory_bytes": None,
        "system_total_memory_bytes": 128 * 1024**3,
        "system_available_memory_bytes": 96 * 1024**3,
        "system_memory_source": "test",
        "system_total_meets_prepare_effective_unified_memory": None,
        "system_available_meets_prepare_system_reserve": None,
        "system_available_meets_prepared_recommended_required_available": None,
        "profile_ok": None,
        "warnings": [],
    }


def test_prepared_server_health_reports_launch_audit_envelope(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    envelope = {
        "schema": "largerlm.launch_audit_server_envelope.v1",
        "artifact_path": str(tmp_path / "launch-audit.json"),
        "artifact_source": "unit",
        "applied_launch_profile_sha256": "abc123",
        "audited_prompt_token_count": 64,
        "audited_max_new_tokens": 4,
        "audited_required_context_tokens": 68,
        "server_max_prompt_tokens": 64,
        "server_max_new_tokens_cap": 4,
        "server_caps_within_envelope": True,
    }
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_prompt_tokens=64,
            max_new_tokens_cap=4,
            launch_audit_envelope=envelope,
        )
    )

    health = app.health()

    assert health["launch_audit_envelope"] == envelope


def test_prepared_server_request_check_reports_launch_audit_envelope(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    envelope = _server_launch_audit_envelope(
        tmp_path,
        prompt_tokens=2,
        max_new_tokens=1,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_prompt_tokens=4,
            max_new_tokens_cap=4,
            launch_audit_envelope=envelope,
        )
    )

    check = app.inspect_token_request(
        prompt_token_count=1,
        payload={"max_new_tokens": 1},
    )

    request_envelope = check["launch_audit_envelope"]
    assert request_envelope["schema"] == "largerlm.launch_audit_server_envelope.v1"
    assert request_envelope["artifact_path"] == str(tmp_path / "launch-audit.json")
    assert request_envelope["audited_prompt_token_count"] == 2
    assert request_envelope["audited_max_new_tokens"] == 1
    assert request_envelope["request_prompt_token_count"] == 1
    assert request_envelope["request_max_new_tokens"] == 1
    assert request_envelope["within_envelope"] is True


def test_prepared_server_rejects_prompt_above_launch_audit_envelope_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    called = False

    def fail_generate_token_ids(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_prompt_tokens=4,
            max_new_tokens_cap=4,
            launch_audit_envelope=_server_launch_audit_envelope(
                tmp_path,
                prompt_tokens=1,
                max_new_tokens=1,
            ),
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="request exceeds launch audit prompt envelope",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1],
                "max_new_tokens": 1,
            }
        )

    assert called is False


def test_prepared_server_rejects_max_new_above_launch_audit_envelope_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    called = False

    def fail_generate_token_ids(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_prompt_tokens=4,
            max_new_tokens_cap=4,
            launch_audit_envelope=_server_launch_audit_envelope(
                tmp_path,
                prompt_tokens=2,
                max_new_tokens=1,
            ),
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="request exceeds launch audit generation envelope",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 2,
            }
        )

    assert called is False


def test_prepared_server_health_warns_when_current_memory_is_below_prepare_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    profile = app.health()["prepared_runtime_profile"]

    assert profile["prepare_effective_unified_memory_bytes"] == 128 * 1024**3
    assert profile["prepare_effective_unified_memory_source"] == "explicit"
    assert profile["prepare_system_reserve_bytes"] == 48 * 1024**3
    assert profile["prepared_recommended_max_live_working_set_bytes"] is None
    assert profile["prepared_recommended_min_free_unified_memory_bytes"] is None
    assert profile["prepared_recommended_required_available_memory_bytes"] is None
    assert profile["system_total_memory_bytes"] == 64 * 1024**3
    assert profile["system_available_memory_bytes"] == 32 * 1024**3
    assert profile["system_total_meets_prepare_effective_unified_memory"] is False
    assert profile["system_available_meets_prepare_system_reserve"] is False
    assert (
        profile["system_available_meets_prepared_recommended_required_available"]
        is None
    )
    assert profile["profile_ok"] is False
    assert profile["warnings"] == [
        "current system total memory is below the prepared effective unified-memory budget",
        "current available memory is below the prepared system reserve",
    ]


def test_prepared_runtime_profile_rejects_below_recommended_required_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_max_live_working_set_bytes"] = 40 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=48 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    health = app.health()
    profile = health["prepared_runtime_profile"]

    assert profile["prepared_recommended_required_available_memory_bytes"] == (
        56 * 1024**3
    )
    suggested = health["suggested_launch_guard_flags"]
    assert suggested == {
        "source": "prepared_manifest",
        "recommended_max_live_working_set_bytes": 40 * 1024**3,
        "max_live_working_set_mib": 40 * 1024,
        "recommended_min_free_unified_memory_bytes": 16 * 1024**3,
        "min_free_unified_memory_gib": 16.0,
        "recommended_required_available_memory_bytes": 56 * 1024**3,
        "argv": (
            "--max-live-working-set-mib",
            "40960",
            "--min-free-unified-memory-gib",
            "16",
        ),
    }
    launch_profile = health["suggested_launch_profile"]
    assert launch_profile["source"] == "prepared_health"
    assert launch_profile["argv_safe_to_replay"] is True
    assert launch_profile["prepared"]["max_context_tokens"] == 4
    assert launch_profile["prepared"]["decode_cache_file_bytes"] == 32
    assert launch_profile["sections"]["launch_guard_flags"] == suggested
    assert launch_profile["argv"][:4] == suggested["argv"]
    assert (
        profile["system_available_meets_prepared_recommended_required_available"]
        is False
    )
    assert profile["profile_ok"] is False
    assert profile["warnings"] == [
        "current available memory is below the prepared recommended required available memory",
    ]


def test_prepared_server_generation_rejects_below_prepare_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prepared runtime profile check failed: current system total memory",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_generation_rejects_unverified_prepare_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_max_live_working_set_bytes"] = 40 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("largerlm.server.system_memory_snapshot", lambda: None)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prepared runtime profile check failed: could not verify",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_health_warns_for_forced_mpsgraph_without_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_prefill_fallback",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("MPSGraph matmul headers are missing",),
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="mpsgraph-f32",
        )
    )

    health = app.health()

    assert health["prefill_backend"]["configured_backend"] == "mpsgraph-f32"
    assert health["prefill_backend"]["capability"]["mps_graph_matmul_declared"] is False
    assert health["prefill_backend"]["warnings"] == (
        "mpsgraph-f32 was forced but MPSGraph matmul headers were not found",
    )


def test_prepared_server_health_keeps_running_when_backend_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_backend_probe(**kwargs):
        del kwargs
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fail_backend_probe)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="mpsgraph-f32",
        )
    )

    health = app.health()

    assert health["prefill_backend"]["capability"] is None
    assert health["prefill_backend"]["warnings"] == (
        "prefill backend inspection failed: probe exploded",
        "mpsgraph-f32 was forced but MPSGraph support was not verified",
    )


def test_prepared_server_rejects_prompt_plus_generation_over_context(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    with pytest.raises(PreparedServerError, match="server cap 2"):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1, 2],
                "max_new_tokens": 2,
            }
        )


def test_prepared_server_model_context_tightens_effective_context(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = tmp_path / "model" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["max_position_embeddings"] = 3
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    assert app.health()["model_context_tokens"] == 3
    assert app.health()["effective_context_tokens"] == 3
    with pytest.raises(PreparedServerError, match="server cap 2"):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1, 2],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_uses_configured_prefill_linear_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="mpsgraph-f32",
        )
    )

    health = app.health()
    app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )

    assert health["prefill_linear_backend"] == "mpsgraph-f32"
    assert captured["prefill_linear_backend"] == "mpsgraph-f32"


def test_prepared_server_require_prefill_acceleration_checks_request_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)

    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(tmp_path),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_mpsgraph_min_batch_tokens=128,
            prefill_run_mpsgraph_probe=True,
            require_prefill_acceleration=True,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill acceleration coverage failed: no resident prefill matrices",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0] * 64,
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_require_prefill_acceleration_allows_decode_only_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(tmp_path),
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_mpsgraph_min_batch_tokens=2,
            prefill_run_mpsgraph_probe=True,
            require_prefill_acceleration=True,
        )
    )

    result = app.generate_token_ids(
        {
            "prompt_token_ids": [0],
            "max_new_tokens": 1,
        }
    )

    assert result["generated_token_ids"] == [2]
    assert captured["batch_prefill_prompt"] is False


def test_prepared_server_require_prefill_acceleration_requires_mpsgraph_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    with pytest.raises(PreparedServerError, match="requires --run-mpsgraph-probe"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                prefill_linear_backend="auto",
                require_prefill_acceleration=True,
            )
        )


def test_prepared_server_require_prefill_acceleration_checks_actual_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(tmp_path),
    )

    def fake_generate_token_ids(**kwargs):
        return _token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            prompt_prefill=SimpleNamespace(
                prefill_acceleration_coverage={
                    "ok": False,
                    "reason": "no resident prefill matrices used an accelerated backend",
                },
                prefill_acceleration_frontier=None,
            ),
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_mpsgraph_min_batch_tokens=2,
            prefill_mpsgraph_min_matrix_dim=32,
            prefill_run_mpsgraph_probe=True,
            require_prefill_acceleration=True,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill acceleration actual coverage failed: no resident prefill matrices",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_min_prefill_accelerated_flops_checks_actual_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(tmp_path),
    )

    def fake_generate_token_ids(**kwargs):
        return _token_result(
            tmp_path,
            tuple(kwargs["prompt_token_ids"]),
            prompt_prefill=SimpleNamespace(
                prefill_acceleration_coverage={
                    "ok": True,
                    "any_resident_matrix_accelerated": True,
                    "accelerated_flop_fraction": 0.25,
                },
                prefill_acceleration_frontier=None,
            ),
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_mpsgraph_min_batch_tokens=2,
            prefill_mpsgraph_min_matrix_dim=32,
            prefill_min_accelerated_flop_fraction=0.5,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill acceleration actual coverage failed: .*0.25.*0.5",
    ):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_uses_configured_prefill_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_cache_read_mib=128.0,
            max_cache_file_mib=2048.0,
            max_runner_scratch_mib=512.0,
            prefill_prompt_chunk_tokens=2,
            prefill_max_prompt_batch_mib=256.0,
            prefill_max_cache_write_mib=512.0,
            prefill_max_stage_mib=768.0,
            prefill_max_compact_stage_mib=384.0,
            prefill_expert_stage_tiling=True,
            prefill_persistent_moe_plan_server=True,
            prefill_persistent_resident_linear_server=True,
            prefill_persistent_attention_projection_server=True,
            prefill_persistent_attention_output_server=True,
            prefill_persistent_shared_expert_server=True,
            prefill_persistent_rope_split_server=True,
            prefill_persistent_mla_attention_server=True,
            prefill_persistent_rmsnorm_server=True,
            prefill_copy_chunk_mib=4.0,
            prefill_stage_disk_margin_mib=1024.0,
            prefill_max_routed_read_amplification=1.75,
            prefill_max_routed_read_gib=0.5,
            prefill_ssd_read_gib_per_second=16.0,
            prefill_max_routed_read_seconds=5.0,
            prefill_moe_token_block="16",
            prefill_moe_output_accumulator="memory",
            prefill_mpsgraph_min_batch_tokens=64,
            prefill_mpsgraph_min_matrix_dim=16,
        )
    )

    health = app.health()
    app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )

    assert health["prefill_prompt_chunk_tokens"] == 2
    assert health["max_cache_read_mib"] == 128.0
    assert health["max_cache_file_mib"] == 2048.0
    assert health["max_runner_scratch_mib"] == 512.0
    assert health["prefill_max_prompt_batch_mib"] == 256.0
    assert health["prefill_stage_disk_margin_mib"] == 1024.0
    assert health["prefill_max_routed_read_amplification"] == 1.75
    assert health["prefill_max_routed_read_gib"] == 0.5
    assert health["prefill_ssd_read_gib_per_second"] == 16.0
    assert health["prefill_max_routed_read_seconds"] == 5.0
    assert health["prefill_moe_token_block"] == 16
    assert health["prefill_moe_output_accumulator"] == "memory"
    assert health["prefill_expert_stage_tiling"] is True
    assert health["prefill_persistent_moe_plan_server"] is True
    assert health["prefill_persistent_resident_linear_server"] is True
    assert health["prefill_persistent_attention_projection_server"] is True
    assert health["prefill_persistent_attention_output_server"] is True
    assert health["prefill_persistent_shared_expert_server"] is True
    assert health["prefill_persistent_rope_split_server"] is True
    assert health["prefill_persistent_mla_attention_server"] is True
    assert health["prefill_persistent_rmsnorm_server"] is True
    assert health["prefill_mpsgraph_min_batch_tokens"] == 64
    assert health["prefill_mpsgraph_min_matrix_dim"] == 16
    assert health["suggested_prefill_copy_policy_flags"] == {
        "source": "prepared_health",
        "prefill_copy_chunk_mib": 4.0,
        "argv": ("--prefill-copy-chunk-mib", "4"),
    }
    assert "--prefill-persistent-moe-plan-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-persistent-resident-linear-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-persistent-attention-projection-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-persistent-attention-output-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-persistent-rope-split-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-persistent-mla-attention-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-persistent-rmsnorm-server" in health[
        "suggested_launch_profile"
    ]["argv"]
    assert "--prefill-moe-output-accumulator" in health["suggested_launch_profile"][
        "argv"
    ]
    assert "prefill_persistent_moe_plan_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_resident_linear_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_attention_projection_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_attention_output_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_shared_expert_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_rope_split_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_mla_attention_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_persistent_rmsnorm_server_flags" in health[
        "suggested_launch_profile"
    ]["sections"]
    assert "prefill_moe_output_accumulator_flags" in health["suggested_launch_profile"][
        "sections"
    ]
    assert health["suggested_launch_profile"]["sections"][
        "prefill_copy_policy_flags"
    ] == health["suggested_prefill_copy_policy_flags"]
    assert health["suggested_launch_profile"]["sections"][
        "prefill_moe_output_accumulator_flags"
    ] == {
        "source": "prepared_health",
        "prefill_moe_output_accumulator": "memory",
        "argv": ("--prefill-moe-output-accumulator", "memory"),
    }
    assert "--prefill-copy-chunk-mib" in health["suggested_launch_profile"]["argv"]
    assert captured["prefill_prompt_chunk_tokens"] == 2
    assert captured["max_cache_read_mib"] == 128.0
    assert captured["max_cache_file_mib"] == 2048.0
    assert captured["max_runner_scratch_mib"] == 512.0
    assert captured["prefill_max_prompt_batch_mib"] == 256.0
    assert captured["prefill_max_cache_write_mib"] == 512.0
    assert captured["prefill_max_stage_mib"] == 768.0
    assert captured["prefill_max_compact_stage_mib"] == 384.0
    assert captured["prefill_copy_chunk_mib"] == 4.0
    assert captured["prefill_stage_disk_margin_mib"] == 1024.0
    assert captured["prefill_max_routed_read_amplification"] == 1.75
    assert captured["prefill_max_routed_read_gib"] == 0.5
    assert captured["prefill_ssd_read_gib_per_second"] == 16.0
    assert captured["prefill_max_routed_read_seconds"] == 5.0
    assert captured["prefill_moe_token_block"] == 16
    assert captured["prefill_moe_output_accumulator"] == "memory"
    assert captured["prefill_expert_stage_tiling"] is True
    assert captured["prefill_persistent_moe_plan_server"] is True
    assert captured["prefill_persistent_resident_linear_server"] is True
    assert captured["prefill_persistent_attention_projection_server"] is True
    assert captured["prefill_persistent_attention_output_server"] is True
    assert captured["prefill_persistent_shared_expert_server"] is True
    assert captured["prefill_persistent_rope_split_server"] is True
    assert captured["prefill_persistent_mla_attention_server"] is True
    assert captured["prefill_persistent_rmsnorm_server"] is True
    assert captured["prefill_mpsgraph_min_batch_tokens"] == 64
    assert captured["prefill_mpsgraph_min_matrix_dim"] == 16


def test_prepared_server_rejects_invalid_prefill_linear_backend(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(PreparedServerError, match="prefill_linear_backend"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                prefill_linear_backend="bad-backend",
            )
        )


def test_prepared_server_rejects_invalid_prefill_caps(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(PreparedServerError, match="prefill_max_prompt_batch_mib"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                prefill_max_prompt_batch_mib=0.0,
            )
        )


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    (
        ({"max_cache_read_mib": 0.0}, "max_cache_read_mib"),
        ({"max_cache_file_mib": 0.0}, "max_cache_file_mib"),
        ({"max_runner_scratch_mib": 0.0}, "max_runner_scratch_mib"),
        ({"max_live_working_set_mib": -1.0}, "max_live_working_set_mib"),
        ({"min_free_unified_memory_gib": -1.0}, "min_free_unified_memory_gib"),
        (
            {"metal_runtime_cache_mla_kv_b_f32": True},
            "metal_runtime_max_mla_kv_b_cache_mib",
        ),
        (
            {"metal_runtime_max_mla_kv_b_cache_mib": -1.0},
            "metal_runtime_max_mla_kv_b_cache_mib",
        ),
        (
            {"metal_runtime_context1_o_proj_cache_file": Path("/tmp/cache.bin")},
            "metal_runtime_context1_o_proj_cache_file",
        ),
        (
            {"expert_read_advise_merge_gap_kib": 1.5},
            "expert_read_advise_merge_gap_kib must be an integer",
        ),
        (
            {"expert_read_advise_align_kib": True},
            "expert_read_advise_align_kib must be an integer",
        ),
        (
            {"expert_read_advise_merge_gap_kib": -1},
            "expert_read_advise_merge_gap_kib",
        ),
        ({"expert_read_advise_align_kib": -1}, "expert_read_advise_align_kib"),
        ({"prefill_copy_chunk_mib": float("nan")}, "prefill_copy_chunk_mib"),
        (
            {"prefill_max_routed_read_amplification": -1.0},
            "prefill_max_routed_read_amplification",
        ),
        ({"prefill_max_routed_read_gib": -1.0}, "prefill_max_routed_read_gib"),
        (
            {"prefill_ssd_read_gib_per_second": -1.0},
            "prefill_ssd_read_gib_per_second",
        ),
        (
            {"prefill_max_routed_read_seconds": -1.0},
            "prefill_max_routed_read_seconds",
        ),
        (
            {"prefill_max_routed_read_seconds": 1.0},
            "prefill_ssd_read_gib_per_second must be positive",
        ),
        (
            {"prefill_expert_stage_tiling": 1},
            "prefill_expert_stage_tiling must be a boolean",
        ),
        (
            {"prefill_persistent_moe_plan_server": 1},
            "prefill_persistent_moe_plan_server must be a boolean",
        ),
        (
            {"prefill_persistent_resident_linear_server": 1},
            "prefill_persistent_resident_linear_server must be a boolean",
        ),
        (
            {"prefill_persistent_attention_projection_server": 1},
            "prefill_persistent_attention_projection_server must be a boolean",
        ),
        (
            {"prefill_persistent_attention_output_server": 1},
            "prefill_persistent_attention_output_server must be a boolean",
        ),
        (
            {"prefill_persistent_shared_expert_server": 1},
            "prefill_persistent_shared_expert_server must be a boolean",
        ),
        (
            {"prefill_persistent_rope_split_server": 1},
            "prefill_persistent_rope_split_server must be a boolean",
        ),
        (
            {"prefill_persistent_mla_attention_server": 1},
            "prefill_persistent_mla_attention_server must be a boolean",
        ),
        (
            {"prefill_persistent_rmsnorm_server": 1},
            "prefill_persistent_rmsnorm_server must be a boolean",
        ),
        (
            {"prefill_backend_probe_timeout_seconds": 0.0},
            "prefill_backend_probe_timeout_seconds",
        ),
        (
            {"decode_max_routed_read_gib_per_token": -1.0},
            "decode_max_routed_read_gib_per_token",
        ),
        (
            {"decode_max_routed_read_seconds_per_token": -1.0},
            "decode_max_routed_read_seconds_per_token",
        ),
        (
            {"decode_max_routed_read_seconds_per_token": 1.0},
            "prefill_ssd_read_gib_per_second must be positive",
        ),
        ({"prefill_mpsgraph_min_batch_tokens": 0}, "prefill_mpsgraph_min_batch_tokens"),
        ({"prefill_mpsgraph_min_matrix_dim": False}, "prefill_mpsgraph_min_matrix_dim"),
        ({"prefill_moe_output_accumulator": "bad"}, "prefill_moe_output_accumulator"),
    ),
)
def test_prepared_server_rejects_invalid_runtime_caps(
    tmp_path: Path,
    config_kwargs: dict[str, object],
    message: str,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(PreparedServerError, match=message):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                **config_kwargs,
            )
        )


def test_prepared_server_rejects_invalid_context1_o_proj_cache_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = tmp_path / "context1-layout.json"

    def fake_load_context1_layout(layout_path_arg, **kwargs):
        assert Path(layout_path_arg) == layout_path
        assert kwargs["prepared_dir"] == prepared
        assert kwargs["require_cache_file"] is True
        raise Context1OProjCacheError("bad layout")

    monkeypatch.setattr(
        "largerlm.server.load_context1_o_proj_cache_layout",
        fake_load_context1_layout,
    )

    with pytest.raises(
        PreparedServerError,
        match="context1 o_proj cache validation failed: bad layout",
    ):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                metal_runtime_context1_o_proj_cache_layout=layout_path,
            )
        )


def test_prepared_server_rejects_short_context1_o_proj_cache_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = tmp_path / "context1-layout.json"
    cache_file = tmp_path / "context1-cache.bin"
    cache_file.write_bytes(b"\0" * 8)

    def fake_load_context1_layout(layout_path_arg, **kwargs):
        assert Path(layout_path_arg) == layout_path
        assert kwargs["prepared_dir"] == prepared
        assert kwargs["require_cache_file"] is False
        return _fake_context1_layout(
            layout_path,
            default_cache_file=tmp_path / "context1-default-cache.bin",
            total_bytes=16,
        )

    monkeypatch.setattr(
        "largerlm.server.load_context1_o_proj_cache_layout",
        fake_load_context1_layout,
    )

    with pytest.raises(PreparedServerError, match="cache file is smaller"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                metal_runtime_context1_o_proj_cache_layout=layout_path,
                metal_runtime_context1_o_proj_cache_file=cache_file,
            )
        )


def test_prepared_server_rejects_incomplete_context1_o_proj_cache_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = tmp_path / "context1-layout.json"
    cache_file = tmp_path / "context1-cache.bin"
    progress = tmp_path / "progress.json"
    cache_file.write_bytes(b"\0" * 32)
    progress.write_text(
        json.dumps(
            {
                "schema": "largerlm.context1_o_proj_bv_cache_progress.v1",
                "cache_schema": "largerlm.context1_o_proj_bv_cache.v1",
                "dtype": "BF16",
                "total_bytes": 32,
                "fma_total": 256,
                "backend": "metal",
                "layers": [0, 1],
                "completed_layers": [1],
                "cache_file": str(cache_file),
                "updated_at_unix": 1.0,
            }
        ),
        encoding="utf-8",
    )

    def fake_load_context1_layout(layout_path_arg, **kwargs):
        assert Path(layout_path_arg) == layout_path
        assert kwargs["prepared_dir"] == prepared
        assert kwargs["require_cache_file"] is False
        return _fake_context1_layout(
            layout_path,
            default_cache_file=cache_file,
            total_bytes=32,
            layers=(0, 1),
        )

    monkeypatch.setattr(
        "largerlm.server.load_context1_o_proj_cache_layout",
        fake_load_context1_layout,
    )

    with pytest.raises(PreparedServerError, match="progress is incomplete"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                metal_runtime_context1_o_proj_cache_layout=layout_path,
                metal_runtime_context1_o_proj_cache_file=cache_file,
            )
        )


def test_prepared_server_rejects_invalid_prefill_moe_token_block(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(PreparedServerError, match="prefill_moe_token_block"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                prefill_moe_token_block=0,
            )
        )


def test_prepared_server_rejects_model_config_override_hash_drift(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    model = tmp_path / "model"
    digest = config_sha256(model)
    assert digest is not None
    _set_layout_config_sha256(prepared / "experts" / "layout.json", digest)
    _set_layout_config_sha256(prepared / "resident" / "layout.json", digest)
    override = tmp_path / "override_config.json"
    override.write_text(
        (model / "config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload = json.loads(override.read_text(encoding="utf-8"))
    payload["routed_scaling_factor"] = 2.5
    override.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedServerError, match="config_sha256 mismatch"):
        PreparedGenerationApp(
            PreparedServerConfig(
                prepared_path=prepared,
                runner_path=Path("unused-runner"),
                model_config_path=override,
            )
        )


def test_prepared_server_inherits_manifest_memory_guard_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_max_live_working_set_bytes"] = 6 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 24 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=64 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    health = app.health()
    app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})

    assert health["max_live_working_set_mib"] == 6 * 1024
    assert health["min_free_unified_memory_gib"] == 24
    assert health["memory_guard"] == {
        "configured_max_live_working_set_bytes": 6 * 1024**3,
        "configured_min_free_unified_memory_bytes": 24 * 1024**3,
        "configured_required_available_memory_bytes": 30 * 1024**3,
        "system_available_memory_bytes": 64 * 1024**3,
        "system_total_memory_bytes": 128 * 1024**3,
        "system_memory_source": "test",
        "available_ok": True,
    }
    assert captured["max_live_working_set_mib"] == 6 * 1024
    assert captured["min_free_unified_memory_gib"] == 24


def test_prepared_server_memory_guard_options_override_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_max_live_working_set_bytes"] = 6 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 24 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_live_working_set_mib=0.0,
            min_free_unified_memory_gib=0.0,
        )
    )

    health = app.health()
    app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})

    assert health["max_live_working_set_mib"] == 0.0
    assert health["min_free_unified_memory_gib"] == 0.0
    assert captured["max_live_working_set_mib"] == 0.0
    assert captured["min_free_unified_memory_gib"] == 0.0


def test_prepared_server_rejects_generation_over_token_cap(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=2,
        )
    )

    with pytest.raises(PreparedServerError, match="max_new_tokens"):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 3,
            }
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {"prompt_token_ids": [0], "max_new_tokens": 1.5},
            "max_new_tokens must be an integer",
        ),
        (
            {"prompt_token_ids": [0], "max_new_tokens": True},
            "max_new_tokens must be an integer",
        ),
        (
            {"prompt_token_ids": [0], "max_new_tokens": 1, "logits_top_k": 1.5},
            "logits_top_k must be an integer",
        ),
        (
            {"prompt_token_ids": [0], "max_new_tokens": 1, "seed": 1.5},
            "seed must be an integer",
        ),
    ),
)
def test_prepared_server_rejects_non_integer_generation_controls(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match=message):
        app.generate_token_ids(payload)


def test_prepared_server_rejects_fractional_prompt_token_ids(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match="prompt_token_ids"):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0, 1.5],
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_rejects_zero_top_p(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match="top_p"):
        app.generate_token_ids(
            {
                "prompt_token_ids": [0],
                "max_new_tokens": 1,
                "top_p": 0.0,
            }
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {"prompt_token_ids": [0], "max_new_tokens": 1, "top_p": float("nan")},
            "top_p must be finite",
        ),
        (
            {"prompt_token_ids": [0], "max_new_tokens": 1, "temperature": math.inf},
            "temperature must be finite",
        ),
    ),
)
def test_prepared_server_rejects_non_finite_float_controls(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match=message):
        app.generate_token_ids(payload)


def test_prepared_server_token_ids_reports_applied_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    applied = _applied_launch_profile(tmp_path)

    def fake_generate_token_ids(**kwargs):
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            applied_launch_profile=applied,
        )
    )

    payload = app.generate_token_ids(
        {
            "prompt_token_ids": [0],
            "max_new_tokens": 1,
        }
    )

    assert payload["applied_launch_profile"] == applied


def test_prepared_server_text_uses_prepared_paths_and_auto_prefill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    cache_dir = tmp_path / "mla-kv-b-cache"
    captured: dict[str, object] = {}
    applied = _applied_launch_profile(tmp_path)

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        token_result = _token_result(tmp_path, (0, 1))
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="ok",
            full_text=kwargs["prompt"] + "ok",
            tokenizer_backend="simple",
            tokenizer_path=Path(kwargs["tokenizer_path"]),
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            applied_launch_profile=applied,
            prefill_mla_kv_b_cache_dir=cache_dir,
        )
    )

    payload = app.generate_text({"prompt": "AB", "max_new_tokens": 1})

    assert payload["generated_text"] == "ok"
    assert payload["applied_launch_profile"] == applied
    assert payload["token_result"]["applied_launch_profile"] == applied
    assert captured["batch_prefill_prompt"] is True
    assert captured["auto_batch_prefill_prompt"] is True
    assert captured["prefill_static_capacity_per_expert"] == "auto"
    assert captured["prefill_mla_kv_b_cache_dir"] == cache_dir
    assert captured["runner_path"] == Path("unused-runner")
    assert captured["max_prompt_tokens"] == 3
    assert payload["request_check"]["prefill_mla_kv_b_cache_dir"] == str(cache_dir)


def test_prepared_server_text_can_use_metal_runtime_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    captured: dict[str, object] = {}
    context1_layout = tmp_path / "context1-layout.json"
    context1_cache = tmp_path / "context1-cache.bin"
    context1_cache.write_bytes(b"\0" * 16)
    _FakeMetalGenerateServerSession.instances = []

    def fake_generate_metal_text(**kwargs):
        captured.update(kwargs)
        kwargs["generate_server_session"].request_count += 1
        token_result = _metal_token_result(tmp_path, (0, 1), generated=(2,))
        return MetalTextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="C",
            full_text=kwargs["prompt"] + "C",
            tokenizer_backend="simple",
            tokenizer_path=Path(kwargs["tokenizer_path"]),
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            token_result=token_result,
        )

    monkeypatch.setattr(
        "largerlm.server.MetalGenerateServerSession",
        _FakeMetalGenerateServerSession,
    )
    monkeypatch.setattr("largerlm.server.generate_metal_text", fake_generate_metal_text)

    def fake_load_context1_layout(layout_path, **kwargs):
        assert Path(layout_path) == context1_layout
        assert kwargs["prepared_dir"] == prepared
        assert kwargs["require_cache_file"] is False
        return _fake_context1_layout(
            context1_layout,
            default_cache_file=tmp_path / "context1-default-cache.bin",
        )

    monkeypatch.setattr(
        "largerlm.server.load_context1_o_proj_cache_layout",
        fake_load_context1_layout,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            metal_runtime_generation=True,
            metal_binary_path=Path("/tmp/glm_moe_infer"),
            max_new_tokens_cap=4,
            metal_runtime_cache_mla_kv_b_f32=True,
            metal_runtime_max_mla_kv_b_cache_mib=256.0,
            metal_runtime_context1_o_proj_cache_layout=context1_layout,
            metal_runtime_context1_o_proj_cache_file=context1_cache,
        )
    )

    payload = app.generate_text({"prompt": "AB", "max_new_tokens": 1})

    assert payload["runtime"] == "glm_moe_infer"
    assert payload["generated_text"] == "C"
    assert payload["token_result"]["runtime"] == "glm_moe_infer"
    assert payload["token_result"]["prompt_prefill"]["source"] == (
        "runtime_prompt_token_ids"
    )
    assert payload["request_check"]["metal_runtime_generation"] is True
    assert payload["request_check"]["batch_prefill_prompt"] is False
    assert captured["prepared_dir"] == prepared
    assert captured["binary"] == Path("/tmp/glm_moe_infer")
    assert captured["prompt"] == "AB"
    assert captured["max_new_tokens"] == 1
    assert captured["max_prompt_tokens"] == 3
    assert captured["use_generate_server_jsonl"] is True
    assert captured["generate_server_session"] is _FakeMetalGenerateServerSession.instances[0]
    assert captured["cache_mla_kv_b_f32"] is True
    assert captured["max_mla_kv_b_cache_mib"] == 256.0
    assert captured["context1_o_proj_cache_layout"] == context1_layout
    assert captured["context1_o_proj_cache_file"] == context1_cache
    health = app.health()
    assert health["metal_runtime_context1_o_proj_cache"]["ok"] is True
    assert health["metal_runtime_session_started"] is True
    assert health["metal_runtime_session_request_count"] == 1


def test_prepared_server_text_honors_batch_prefill_payload_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        token_result = _token_result(tmp_path, (0, 1))
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="ok",
            full_text=kwargs["prompt"] + "ok",
            tokenizer_backend="simple",
            tokenizer_path=Path(kwargs["tokenizer_path"]),
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    payload = app.generate_text(
        {
            "prompt": "AB",
            "max_new_tokens": 1,
            "batch_prefill_prompt": False,
        }
    )

    assert payload["generated_text"] == "ok"
    assert captured["batch_prefill_prompt"] is False
    assert captured["auto_batch_prefill_prompt"] is False
    assert captured["prefill_static_capacity_per_expert"] is None


def test_prepared_server_text_runs_request_admission_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)

    def fail_generate_text(**kwargs):
        del kwargs
        raise AssertionError("generate_text should not be called")

    monkeypatch.setattr("largerlm.server.generate_text", fail_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            prefill_prompt_chunk_tokens=999,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill_prompt_chunk_tokens 999 exceeds safety-capped maximum",
    ):
        app.generate_text(
            {
                "prompt": "AB",
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_text_rejects_nonpassing_admission_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    called = False

    def fake_inspect_token_request(self, **kwargs):
        del self, kwargs
        return {
            "ok": False,
            "reason": "synthetic live-memory guard failure",
        }

    def fake_generate_text(**kwargs):
        nonlocal called
        called = True
        token_result = _token_result(tmp_path, (0, 1))
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="ok",
            full_text=kwargs["prompt"] + "ok",
            tokenizer_backend="simple",
            tokenizer_path=Path(kwargs["tokenizer_path"]),
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        fake_inspect_token_request,
    )
    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="synthetic live-memory guard failure",
    ):
        app.generate_text(
            {
                "prompt": "AB",
                "max_new_tokens": 1,
            }
        )

    assert called is False


def test_prepared_server_text_runs_runtime_preflight_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)

    def fail_check_generation_runtime(**kwargs):
        del kwargs
        raise GenerationGuardError("available unified memory is below test reserve")

    def fail_generate_text(**kwargs):
        del kwargs
        raise AssertionError("generate_text should not be called")

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fail_check_generation_runtime,
    )
    monkeypatch.setattr("largerlm.server.generate_text", fail_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
        )
    )

    with pytest.raises(PreparedServerError, match="below test reserve"):
        app.generate_text({"prompt": "A", "max_new_tokens": 1})


def test_prepared_server_text_require_prefill_acceleration_checks_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(tmp_path),
    )

    def fail_generate_text(**kwargs):
        raise AssertionError("generate_text should not be called")

    monkeypatch.setattr("largerlm.server.generate_text", fail_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_mpsgraph_min_batch_tokens=128,
            prefill_run_mpsgraph_probe=True,
            require_prefill_acceleration=True,
        )
    )

    with pytest.raises(
        PreparedServerError,
        match="prefill acceleration coverage failed: no resident prefill matrices",
    ):
        app.generate_text(
            {
                "prompt": "A" * 64,
                "max_new_tokens": 1,
            }
        )


def test_prepared_server_text_require_prefill_acceleration_allows_decode_only_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    _set_prepared_context(prepared, 256)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_backend(tmp_path),
    )
    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        token_result = _token_result(tmp_path, (0,))
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="B",
            full_text=kwargs["prompt"] + "B",
            tokenizer_backend="tokenizers",
            tokenizer_path=prepared.parent / "model",
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="auto",
            prefill_mpsgraph_min_batch_tokens=2,
            prefill_run_mpsgraph_probe=True,
            require_prefill_acceleration=True,
        )
    )

    result = app.generate_text({"prompt": "A", "max_new_tokens": 1})

    assert result["generated_text"] == "B"
    assert captured["batch_prefill_prompt"] is False


def test_prepared_server_derives_glm_router_and_dsa_defaults_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = tmp_path / "model" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "mlp_layer_types": ["dense", "sparse"],
            "scoring_func": "sigmoid",
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "n_group": 1,
            "topk_group": 1,
            "eos_token_id": [2, 3, 2],
            "q_lora_rank": 4,
            "indexer_types": ["full", "shared"],
            "index_topk": 3,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "indexer_rope_interleave": True,
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return _token_result(tmp_path, tuple(kwargs["prompt_token_ids"]))

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            allow_missing_dsa_indexer=True,
        )
    )

    result = app.generate_token_ids(
        {
            "prompt_token_ids": [0, 1],
            "max_new_tokens": 1,
        }
    )

    assert result["generated_token_ids"] == [2]
    assert captured["dense_layers"] == (0,)
    assert captured["router_score"] == "sigmoid"
    assert captured["norm_topk_prob"] is True
    assert captured["routed_scaling_factor"] == 2.5
    assert captured["router_n_group"] == 1
    assert captured["router_topk_group"] == 1
    assert captured["eos_token_ids"] == (2, 3)
    assert captured["dsa_indexer_types"] == ("full", "shared")
    assert captured["dsa_index_topk"] == 3
    assert captured["dsa_index_n_heads"] == 2
    assert captured["dsa_index_head_dim"] == 4
    assert captured["dsa_qk_rope_dim"] == 2
    assert captured["dsa_rope_interleave"] is True
    assert captured["allow_missing_dsa_indexer"] is True


def test_prepared_server_openai_completion_maps_to_text_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    captured: dict[str, object] = {}
    applied = _applied_launch_profile(tmp_path)

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        token_result = _token_result(tmp_path, (0, 1))
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="ok",
            full_text=kwargs["prompt"] + "ok",
            tokenizer_backend="simple",
            tokenizer_path=Path(kwargs["tokenizer_path"]),
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            applied_launch_profile=applied,
        )
    )

    payload = app.openai_completion(
        {
            "model": "glm-local",
            "prompt": "AB",
            "max_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
            "echo": True,
        }
    )

    assert payload["object"] == "text_completion"
    assert payload["model"] == "glm-local"
    assert payload["choices"][0]["text"] == "ABok"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert payload["largerlm"]["applied_launch_profile"] == applied
    assert payload["largerlm"]["token_result"]["applied_launch_profile"] == applied
    assert captured["max_new_tokens"] == 2
    assert captured["max_prompt_tokens"] == 2
    assert captured["auto_batch_prefill_prompt"] is True
    assert captured["preflight_runtime"] is True


def test_prepared_server_openai_models_uses_served_model_name(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            served_model_name="glm-5.2-local",
        )
    )

    payload = app.openai_models()

    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "glm-5.2-local"
    assert app.health()["served_model_name"] == "glm-5.2-local"


def test_prepared_server_openai_completion_rejects_streaming(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match="stream"):
        app.openai_completion(
            {
                "prompt": "hi",
                "stream": True,
            }
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"prompt": "hi", "max_tokens": 1.5}, "max_tokens must be an integer"),
        (
            {"prompt": "hi", "max_completion_tokens": 1.5},
            "max_tokens must be an integer",
        ),
        ({"prompt": "hi", "n": 1.5}, "n must be an integer"),
    ),
)
def test_prepared_server_openai_completion_rejects_non_integer_controls(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match=message):
        app.openai_completion(payload)


def test_prepared_server_openai_chat_completion_uses_chat_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    captured: dict[str, object] = {}
    rendered_messages: dict[str, object] = {}
    applied = _applied_launch_profile(tmp_path)

    def fake_render_chat_prompt(
        tokenizer_path,
        messages,
        *,
        backend,
        trust_remote_code,
        add_generation_prompt,
    ):
        rendered_messages.update(
            {
                "tokenizer_path": Path(tokenizer_path),
                "messages": messages,
                "backend": backend,
                "trust_remote_code": trust_remote_code,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return RenderedChatPrompt(
            text="AB",
            backend="transformers",
            tokenizer_path=Path(tokenizer_path),
        )

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        token_result = _token_result(tmp_path, (0, 1))
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="ok",
            full_text=kwargs["prompt"] + "ok",
            tokenizer_backend="simple",
            tokenizer_path=Path(kwargs["tokenizer_path"]),
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.server.render_chat_prompt", fake_render_chat_prompt)
    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            max_new_tokens_cap=4,
            tokenizer_backend="auto",
            applied_launch_profile=applied,
        )
    )

    payload = app.openai_chat_completion(
        {
            "model": "glm-chat-local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
        }
    )

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "glm-chat-local"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "ok",
    }
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 3
    assert payload["largerlm"]["applied_launch_profile"] == applied
    assert payload["largerlm"]["token_result"]["applied_launch_profile"] == applied
    assert rendered_messages["messages"] == [{"role": "user", "content": "hi"}]
    assert rendered_messages["add_generation_prompt"] is True
    assert captured["prompt"] == "AB"
    assert captured["add_special_tokens"] is False
    assert captured["max_prompt_tokens"] == 2
    assert captured["auto_batch_prefill_prompt"] is True


def test_prepared_server_openai_chat_completion_rejects_tools(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    with pytest.raises(PreparedServerError, match="tools"):
        app.openai_chat_completion(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            }
        )
