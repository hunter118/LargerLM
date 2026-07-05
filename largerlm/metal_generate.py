from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import ModelConfig, load_config
from .decode_cache import build_decode_cache_layout, init_decode_cache_file
from .embedding import embed_token
from .prepared import PreparedManifest, load_prepared_manifest
from .prompt_prefill import PromptPrefillResult, run_prompt_prefill


class MetalGenerateError(RuntimeError):
    """Raised when the single-process Metal decode loop cannot run safely."""


@dataclass(frozen=True)
class MetalTokenGenerationResult:
    prepared_dir: Path
    binary: Path
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    work_dir: Path
    kept_work_dir: bool
    cache_layout_path: Path
    cache_file_path: Path
    input_f32_path: Path
    output_f32_path: Path
    generated_json_path: Path
    elapsed_seconds: float
    decode_elapsed_seconds: tuple[float, ...]
    final_logits_elapsed_seconds: tuple[float, ...]
    estimated_live_working_set_bytes: int
    max_live_working_set_mib: int
    cache_total_bytes: int
    note: str
    final_logits_bytes_read: tuple[int, ...] = ()
    final_logits_lm_head_bytes_read: tuple[int, ...] = ()
    final_logits_read_seconds: tuple[float, ...] = ()
    final_logits_kernel_seconds: tuple[float, ...] = ()
    final_logits_resident_mmap_backed: tuple[bool, ...] = ()
    min_free_unified_memory_gib: float = 0.0
    admission_ok: bool | None = None
    available_unified_memory_ok: bool | None = None
    system_available_memory_bytes: int | None = None
    required_available_memory_bytes: int | None = None
    expert_buffer_count_allocated: int | None = None
    decode_expert_bytes_read: tuple[int, ...] = ()
    decode_dense_mlp_bytes_read: tuple[int, ...] = ()
    decode_shared_bytes_read: tuple[int, ...] = ()
    decode_layer_count: tuple[int, ...] = ()
    decode_dense_layer_count: tuple[int, ...] = ()
    decode_moe_layer_count: tuple[int, ...] = ()
    decode_layer_elapsed_seconds: tuple[float, ...] = ()
    decode_attn_projection_elapsed_seconds: tuple[float, ...] = ()
    decode_mla_attention_elapsed_seconds: tuple[float, ...] = ()
    decode_mla_attention_cache_read_seconds: tuple[float, ...] = ()
    decode_mla_attention_value_read_seconds: tuple[float, ...] = ()
    decode_mla_attention_kernel_seconds: tuple[float, ...] = ()
    decode_mla_attention_output_write_seconds: tuple[float, ...] = ()
    decode_attn_output_elapsed_seconds: tuple[float, ...] = ()
    decode_mlp_elapsed_seconds: tuple[float, ...] = ()
    decode_dense_mlp_elapsed_seconds: tuple[float, ...] = ()
    decode_moe_mlp_elapsed_seconds: tuple[float, ...] = ()
    decode_expert_read_seconds: tuple[float, ...] = ()
    decode_shared_read_seconds: tuple[float, ...] = ()
    decode_shared_prefetch_seconds: tuple[float, ...] = ()
    decode_moe_mlp_kernel_seconds: tuple[float, ...] = ()
    decode_moe_mlp_output_write_seconds: tuple[float, ...] = ()
    decode_moe_mlp_overhead_seconds: tuple[float, ...] = ()
    decode_layer_overhead_seconds: tuple[float, ...] = ()
    decode_attn_projection_command_buffer_count: tuple[int, ...] = ()
    decode_attn_projection_synchronous_wait_count: tuple[int, ...] = ()
    decode_attn_projection_async_submitted_count: tuple[int, ...] = ()
    decode_rope_mla_command_buffer_count: tuple[int, ...] = ()
    decode_attn_output_command_buffer_count: tuple[int, ...] = ()
    decode_attn_output_context1_o_proj_cache_count: tuple[int, ...] = ()
    decode_attn_output_resident_mmap_backed_count: tuple[int, ...] = ()
    decode_post_attn_norm_command_buffer_count: tuple[int, ...] = ()
    decode_router_command_buffer_count: tuple[int, ...] = ()
    decode_post_attn_norm_router_command_buffer_count: tuple[int, ...] = ()
    decode_dense_mlp_command_buffer_count: tuple[int, ...] = ()
    decode_dense_mlp_synchronous_wait_count: tuple[int, ...] = ()
    decode_dense_mlp_async_submitted_count: tuple[int, ...] = ()
    decode_moe_mlp_command_buffer_count: tuple[int, ...] = ()
    decode_moe_mlp_synchronous_wait_count: tuple[int, ...] = ()
    decode_shared_prefetch_used_count: tuple[int, ...] = ()
    decode_attn_output_norm_router_fused_count: tuple[int, ...] = ()
    decode_rope_mla_attn_output_norm_router_fused_count: tuple[int, ...] = ()
    decode_rope_mla_input_buffer_direct_count: tuple[int, ...] = ()
    decode_attn_output_buffer_direct_count: tuple[int, ...] = ()
    decode_moe_mlp_input_buffer_direct_count: tuple[int, ...] = ()
    decode_layer_input_buffer_direct_count: tuple[int, ...] = ()
    decode_command_buffer_count: tuple[int, ...] = ()
    decode_synchronous_wait_count_estimate: tuple[int, ...] = ()
    decode_expert_read_dispatch_count: tuple[int, ...] = ()
    decode_expert_read_task_count: tuple[int, ...] = ()
    decode_expert_read_max_task_count: tuple[int, ...] = ()
    decode_expert_read_max_worker_count: tuple[int, ...] = ()
    decode_expert_read_pool_dispatch_count: tuple[int, ...] = ()
    decode_expert_read_serial_dispatch_count: tuple[int, ...] = ()
    decode_mla_value_cache_hit_count: tuple[int, ...] = ()
    decode_mla_value_cache_store_count: tuple[int, ...] = ()
    decode_mla_value_cache_bytes: tuple[int, ...] = ()
    decode_mla_value_cache_total_bytes: tuple[int, ...] = ()
    decode_attn_output_bytes_read: tuple[int, ...] = ()
    decode_attn_output_read_seconds: tuple[float, ...] = ()
    decode_attn_output_projection_kernel_seconds: tuple[float, ...] = ()
    decode_post_attn_norm_weight_bytes_read: tuple[int, ...] = ()
    decode_post_attn_norm_weight_read_seconds: tuple[float, ...] = ()
    decode_router_bytes_read: tuple[int, ...] = ()
    decode_router_correction_bias_bytes_read: tuple[int, ...] = ()
    decode_router_read_seconds: tuple[float, ...] = ()
    decode_router_kernel_seconds: tuple[float, ...] = ()
    mla_kv_b_cache_enabled: bool | None = None
    mla_kv_b_cache_current_bytes: int | None = None
    mla_kv_b_cache_live_estimate_bytes: int | None = None
    prompt_prefill: PromptPrefillResult | None = None
    prompt_prefill_elapsed_seconds: float | None = None
    prompt_prefill_estimated_live_working_set_bytes: int | None = None
    prefill_max_live_working_set_mib: float | None = None
    metal_elapsed_seconds: float | None = None
    prefill_final_logits_elapsed_seconds: float | None = None
    input_token_id: int | None = None


def _layers_csv(config: ModelConfig) -> str:
    return ",".join(str(layer) for layer in range(config.num_hidden_layers))


def _dense_layers_csv(config: ModelConfig) -> str:
    moe_layers = set(config.moe_layers)
    return ",".join(
        str(layer)
        for layer in range(config.num_hidden_layers)
        if layer not in moe_layers
    )


def _dense_layers_tuple(config: ModelConfig) -> tuple[int, ...]:
    moe_layers = set(config.moe_layers)
    return tuple(
        layer
        for layer in range(config.num_hidden_layers)
        if layer not in moe_layers
    )


def _required_int(value: int | None, label: str) -> int:
    if value is None:
        raise MetalGenerateError(f"config is missing {label}")
    return int(value)


def _run_command(cmd: Sequence[str], *, quiet: bool) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if not quiet:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    if completed.returncode != 0:
        if quiet:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="")
        raise MetalGenerateError(
            f"glm_moe_infer failed with exit code {completed.returncode}"
        )
    return completed


def _manifest_from_prepared(prepared_dir: str | Path) -> PreparedManifest:
    prepared = Path(prepared_dir)
    manifest_path = prepared / "manifest.json"
    if not manifest_path.exists():
        raise MetalGenerateError(f"prepared manifest not found: {manifest_path}")
    return load_prepared_manifest(manifest_path)


def _router_score(config: ModelConfig) -> str:
    return config.scoring_func or "sigmoid"


def _required_context_tokens(
    *,
    prompt_token_count: int,
    max_new_tokens: int,
    prefill_prompt: bool,
) -> int:
    if prefill_prompt and prompt_token_count > 1:
        return prompt_token_count + max_new_tokens - 1
    return max_new_tokens


def _prompt_prefill_estimated_live_working_set_bytes(
    result: PromptPrefillResult,
) -> int | None:
    budget = getattr(result, "live_memory_budget", None)
    value = getattr(budget, "estimated_live_working_set_bytes", None)
    return int(value) if value is not None else None


def _payload_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    return bool(value)


def _payload_bool_any(payload: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = _payload_bool(payload, key)
        if value is not None:
            return value
    return None


def _payload_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _payload_int_any(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _payload_int(payload, key)
        if value is not None:
            return value
    return None


def _step_dicts(steps: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(steps, list):
        return ()
    return tuple(step for step in steps if isinstance(step, dict))


def _step_float_tuple(steps: object, key: str) -> tuple[float, ...]:
    return tuple(float(step.get(key, 0.0)) for step in _step_dicts(steps))


def _step_int_tuple(steps: object, key: str) -> tuple[int, ...]:
    return tuple(int(step.get(key, 0)) for step in _step_dicts(steps))


def _step_bool_tuple(steps: object, key: str) -> tuple[bool, ...]:
    return tuple(bool(step.get(key, False)) for step in _step_dicts(steps))


def _final_logits_result_fields(steps: object) -> dict[str, Any]:
    return {
        "final_logits_bytes_read": _step_int_tuple(
            steps,
            "final_logits_bytes_read",
        ),
        "final_logits_lm_head_bytes_read": _step_int_tuple(
            steps,
            "final_logits_lm_head_bytes_read",
        ),
        "final_logits_read_seconds": _step_float_tuple(
            steps,
            "final_logits_read_seconds",
        ),
        "final_logits_kernel_seconds": _step_float_tuple(
            steps,
            "final_logits_kernel_seconds",
        ),
        "final_logits_resident_mmap_backed": _step_bool_tuple(
            steps,
            "final_logits_resident_mmap_backed",
        ),
    }


def _mla_kv_b_cache_result_fields(
    payload: dict[str, Any],
    steps: object,
) -> dict[str, Any]:
    return {
        "decode_mla_value_cache_hit_count": _step_int_tuple(
            steps,
            "mla_value_cache_hit_count",
        ),
        "decode_mla_value_cache_store_count": _step_int_tuple(
            steps,
            "mla_value_cache_store_count",
        ),
        "decode_mla_value_cache_bytes": _step_int_tuple(
            steps,
            "mla_value_cache_bytes",
        ),
        "decode_mla_value_cache_total_bytes": _step_int_tuple(
            steps,
            "mla_value_cache_total_bytes",
        ),
        "decode_attn_output_bytes_read": _step_int_tuple(
            steps,
            "attn_output_bytes_read",
        ),
        "decode_attn_output_read_seconds": _step_float_tuple(
            steps,
            "attn_output_read_seconds",
        ),
        "decode_attn_output_projection_kernel_seconds": _step_float_tuple(
            steps,
            "attn_output_projection_kernel_seconds",
        ),
        "decode_post_attn_norm_weight_bytes_read": _step_int_tuple(
            steps,
            "post_attn_norm_weight_bytes_read",
        ),
        "decode_post_attn_norm_weight_read_seconds": _step_float_tuple(
            steps,
            "post_attn_norm_weight_read_seconds",
        ),
        "decode_router_bytes_read": _step_int_tuple(
            steps,
            "router_bytes_read",
        ),
        "decode_router_correction_bias_bytes_read": _step_int_tuple(
            steps,
            "router_correction_bias_bytes_read",
        ),
        "decode_router_read_seconds": _step_float_tuple(
            steps,
            "router_read_seconds",
        ),
        "decode_router_kernel_seconds": _step_float_tuple(
            steps,
            "router_kernel_seconds",
        ),
        "mla_kv_b_cache_enabled": _payload_bool_any(
            payload,
            "mla_kv_b_cache_enabled",
            "executor_mla_kv_b_cache_enabled",
        ),
        "mla_kv_b_cache_current_bytes": _payload_int_any(
            payload,
            "mla_kv_b_cache_current_bytes",
            "executor_mla_kv_b_cache_current_bytes",
        ),
        "mla_kv_b_cache_live_estimate_bytes": _payload_int_any(
            payload,
            "mla_kv_b_cache_live_estimate_bytes",
            "executor_mla_kv_b_cache_live_estimate_bytes",
        ),
    }


def _decode_layer_timing_result_fields(steps: object) -> dict[str, Any]:
    return {
        "decode_layer_count": _step_int_tuple(
            steps,
            "layer_count",
        ),
        "decode_dense_layer_count": _step_int_tuple(
            steps,
            "dense_layer_count",
        ),
        "decode_moe_layer_count": _step_int_tuple(
            steps,
            "moe_layer_count",
        ),
        "decode_layer_elapsed_seconds": _step_float_tuple(
            steps,
            "layer_elapsed_seconds",
        ),
        "decode_attn_projection_elapsed_seconds": _step_float_tuple(
            steps,
            "attn_projection_elapsed_seconds",
        ),
        "decode_mla_attention_elapsed_seconds": _step_float_tuple(
            steps,
            "mla_attention_elapsed_seconds",
        ),
        "decode_mla_attention_cache_read_seconds": _step_float_tuple(
            steps,
            "mla_attention_cache_read_seconds",
        ),
        "decode_mla_attention_value_read_seconds": _step_float_tuple(
            steps,
            "mla_attention_value_read_seconds",
        ),
        "decode_mla_attention_kernel_seconds": _step_float_tuple(
            steps,
            "mla_attention_kernel_seconds",
        ),
        "decode_mla_attention_output_write_seconds": _step_float_tuple(
            steps,
            "mla_attention_output_write_seconds",
        ),
        "decode_attn_output_elapsed_seconds": _step_float_tuple(
            steps,
            "attn_output_elapsed_seconds",
        ),
        "decode_mlp_elapsed_seconds": _step_float_tuple(
            steps,
            "mlp_elapsed_seconds",
        ),
        "decode_dense_mlp_elapsed_seconds": _step_float_tuple(
            steps,
            "dense_mlp_elapsed_seconds",
        ),
        "decode_moe_mlp_elapsed_seconds": _step_float_tuple(
            steps,
            "moe_mlp_elapsed_seconds",
        ),
        "decode_shared_bytes_read": _step_int_tuple(
            steps,
            "shared_bytes_read",
        ),
        "decode_shared_read_seconds": _step_float_tuple(
            steps,
            "shared_read_seconds",
        ),
        "decode_shared_prefetch_seconds": _step_float_tuple(
            steps,
            "shared_prefetch_seconds",
        ),
        "decode_shared_prefetch_used_count": _step_int_tuple(
            steps,
            "shared_prefetch_used_count",
        ),
        "decode_moe_mlp_output_write_seconds": _step_float_tuple(
            steps,
            "moe_mlp_output_write_seconds",
        ),
        "decode_moe_mlp_overhead_seconds": _step_float_tuple(
            steps,
            "moe_mlp_overhead_seconds",
        ),
        "decode_layer_overhead_seconds": _step_float_tuple(
            steps,
            "layer_overhead_seconds",
        ),
        "decode_attn_projection_command_buffer_count": _step_int_tuple(
            steps,
            "attn_projection_command_buffer_count",
        ),
        "decode_attn_projection_synchronous_wait_count": _step_int_tuple(
            steps,
            "attn_projection_synchronous_wait_count",
        ),
        "decode_attn_projection_async_submitted_count": _step_int_tuple(
            steps,
            "attn_projection_async_submitted_count",
        ),
        "decode_rope_mla_command_buffer_count": _step_int_tuple(
            steps,
            "rope_mla_command_buffer_count",
        ),
        "decode_attn_output_command_buffer_count": _step_int_tuple(
            steps,
            "attn_output_command_buffer_count",
        ),
        "decode_attn_output_context1_o_proj_cache_count": _step_int_tuple(
            steps,
            "attn_output_context1_o_proj_cache_count",
        ),
        "decode_attn_output_resident_mmap_backed_count": _step_int_tuple(
            steps,
            "attn_output_resident_mmap_backed_count",
        ),
        "decode_post_attn_norm_command_buffer_count": _step_int_tuple(
            steps,
            "post_attn_norm_command_buffer_count",
        ),
        "decode_router_command_buffer_count": _step_int_tuple(
            steps,
            "router_command_buffer_count",
        ),
        "decode_post_attn_norm_router_command_buffer_count": _step_int_tuple(
            steps,
            "post_attn_norm_router_command_buffer_count",
        ),
        "decode_dense_mlp_command_buffer_count": _step_int_tuple(
            steps,
            "dense_mlp_command_buffer_count",
        ),
        "decode_dense_mlp_synchronous_wait_count": _step_int_tuple(
            steps,
            "dense_mlp_synchronous_wait_count",
        ),
        "decode_dense_mlp_async_submitted_count": _step_int_tuple(
            steps,
            "dense_mlp_async_submitted_count",
        ),
        "decode_moe_mlp_command_buffer_count": _step_int_tuple(
            steps,
            "moe_mlp_command_buffer_count",
        ),
        "decode_moe_mlp_synchronous_wait_count": _step_int_tuple(
            steps,
            "moe_mlp_synchronous_wait_count",
        ),
        "decode_attn_output_norm_router_fused_count": _step_int_tuple(
            steps,
            "attn_output_norm_router_fused_count",
        ),
        "decode_rope_mla_attn_output_norm_router_fused_count": _step_int_tuple(
            steps,
            "rope_mla_attn_output_norm_router_fused_count",
        ),
        "decode_rope_mla_input_buffer_direct_count": _step_int_tuple(
            steps,
            "rope_mla_input_buffer_direct_count",
        ),
        "decode_attn_output_buffer_direct_count": _step_int_tuple(
            steps,
            "attn_output_buffer_direct_count",
        ),
        "decode_moe_mlp_input_buffer_direct_count": _step_int_tuple(
            steps,
            "moe_mlp_input_buffer_direct_count",
        ),
        "decode_layer_input_buffer_direct_count": _step_int_tuple(
            steps,
            "layer_input_buffer_direct_count",
        ),
        "decode_command_buffer_count": _step_int_tuple(
            steps,
            "command_buffer_count",
        ),
        "decode_synchronous_wait_count_estimate": _step_int_tuple(
            steps,
            "synchronous_wait_count_estimate",
        ),
    }


def _build_generate_request_payload(
    *,
    decode_layers: str,
    input_f32: Path | None = None,
    input_token_id: int | None = None,
    prompt_token_ids: Sequence[int] | None = None,
    output_f32: Path | None = None,
    output_dir: Path,
    output_generated_json: Path | None = None,
    output_next_input_f32: Path | None = None,
    generate_steps: int,
    generate_first_from_input_logits: bool,
    cache_layout: Path,
    cache_file: Path,
    position: int,
    context_length: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    cache_position_offset: int,
    top_k: int,
    expert_buffer_count: int,
    rms_norm_eps: float,
    rope_theta: float,
    max_cache_file_mib: float,
    max_cache_read_mib: float,
    max_chunk_mib: float,
    mmap_final_logits: bool,
    max_embedding_row_mib: float,
    max_live_working_set_mib: int,
    min_free_unified_memory_gib: float,
    include_shared_expert: bool,
    rope_interleave: bool,
    in_memory_decode_cache: bool | None = None,
    cache_mla_kv_b_f32: bool | None = None,
    max_mla_kv_b_cache_mib: float | None = None,
    context1_o_proj_cache_layout: str | Path | None = None,
    context1_o_proj_cache_file: str | Path | None = None,
) -> dict[str, object]:
    input_source_count = sum(
        1
        for item in (input_f32, input_token_id, prompt_token_ids)
        if item is not None
    )
    if input_source_count != 1:
        raise MetalGenerateError("provide exactly one generate input source")
    request: dict[str, object] = {
        "decode_layers": decode_layers,
        "output_dir": str(output_dir),
        "generate_steps": int(generate_steps),
        "generate_first_from_input_logits": bool(generate_first_from_input_logits),
        "cache_layout": str(cache_layout),
        "cache_file": str(cache_file),
        "position": int(position),
        "context_length": int(context_length),
        "num_heads": int(num_heads),
        "kv_lora_dim": int(kv_lora_dim),
        "qk_nope_dim": int(qk_nope_dim),
        "rope_dim": int(rope_dim),
        "v_head_dim": int(v_head_dim),
        "cache_position_offset": int(cache_position_offset),
        "top_k": int(top_k),
        "expert_buffer_count": int(expert_buffer_count),
        "rms_norm_eps": float(rms_norm_eps),
        "rope_theta": float(rope_theta),
        "max_cache_file_mib": float(max_cache_file_mib),
        "max_cache_read_mib": float(max_cache_read_mib),
        "max_chunk_mib": float(max_chunk_mib),
        "mmap_final_logits": bool(mmap_final_logits),
        "max_embedding_row_mib": float(max_embedding_row_mib),
        "max_live_working_set_mib": int(max_live_working_set_mib),
        "min_free_unified_memory_gib": float(min_free_unified_memory_gib),
        "skip_debug_intermediates": True,
        "include_shared_expert": bool(include_shared_expert),
        "rope_interleave": bool(rope_interleave),
    }
    if output_f32 is not None:
        request["output_f32"] = str(output_f32)
    if output_generated_json is not None:
        request["output_generated_json"] = str(output_generated_json)
    if output_next_input_f32 is not None:
        request["output_next_input_f32"] = str(output_next_input_f32)
    if in_memory_decode_cache is not None:
        request["in_memory_decode_cache"] = bool(in_memory_decode_cache)
    if cache_mla_kv_b_f32 is not None:
        request["cache_mla_kv_b_f32"] = bool(cache_mla_kv_b_f32)
    if max_mla_kv_b_cache_mib is not None:
        request["max_mla_kv_b_cache_mib"] = float(max_mla_kv_b_cache_mib)
    if context1_o_proj_cache_layout is not None:
        request["context1_o_proj_cache_layout"] = str(context1_o_proj_cache_layout)
    if context1_o_proj_cache_file is not None:
        request["context1_o_proj_cache_file"] = str(context1_o_proj_cache_file)
    if input_f32 is not None:
        request["input_f32"] = str(input_f32)
    elif input_token_id is not None:
        request["input_token_id"] = int(input_token_id)
    else:
        prompt = tuple(int(token) for token in prompt_token_ids or ())
        if not prompt:
            raise MetalGenerateError("prompt_token_ids must not be empty")
        if any(token < 0 for token in prompt):
            raise MetalGenerateError("prompt_token_ids must be non-negative")
        request["prompt_token_ids"] = list(prompt)
    return request


def _write_generate_request_json(
    path: Path,
    **kwargs: Any,
) -> Path:
    request = _build_generate_request_payload(**kwargs)
    path.write_text(
        json.dumps(request, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _run_generate_server_jsonl_request(
    *,
    binary: str | Path,
    prepared_dir: str | Path,
    request: dict[str, object],
    quiet: bool,
) -> tuple[dict[str, Any], float]:
    stdin = (
        json.dumps(request, separators=(",", ":"))
        + "\n"
        + json.dumps({"command": "quit"}, separators=(",", ":"))
        + "\n"
    )
    cmd = [
        str(binary),
        "--prepared",
        str(prepared_dir),
        "--generate-server-jsonl",
    ]
    started = time.monotonic()
    completed = subprocess.run(cmd, input=stdin, text=True, capture_output=True)
    elapsed = time.monotonic() - started
    if not quiet:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    if completed.returncode != 0:
        if quiet:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="")
        raise MetalGenerateError(
            f"glm_moe_infer JSONL server failed with exit code {completed.returncode}"
        )
    try:
        lines = [
            json.loads(line)
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise MetalGenerateError(
            f"failed to parse glm_moe_infer JSONL server output: {exc}"
        ) from exc
    if len(lines) < 3:
        raise MetalGenerateError("glm_moe_infer JSONL server output was incomplete")
    ready = lines[0]
    response = lines[1]
    done = lines[-1]
    if ready.get("event") != "ready" or not ready.get("ok"):
        raise MetalGenerateError("glm_moe_infer JSONL server did not report ready")
    if response.get("event") != "request" or not response.get("accepted"):
        error = response.get("error") if isinstance(response, dict) else None
        raise MetalGenerateError(
            "glm_moe_infer JSONL server rejected request"
            + (f": {error}" if error else "")
        )
    if not response.get("ok") or response.get("executor_exit_code") not in (0, None):
        error = response.get("executor_payload_parse_error") or response.get("error")
        raise MetalGenerateError(
            "glm_moe_infer JSONL server execution failed"
            + (f": {error}" if error else "")
        )
    if done.get("event") != "done" or not done.get("ok"):
        raise MetalGenerateError("glm_moe_infer JSONL server did not finish cleanly")
    return response, elapsed


class MetalGenerateServerSession:
    """Persistent glm_moe_infer JSONL generation server client."""

    def __init__(
        self,
        *,
        binary: str | Path,
        prepared_dir: str | Path,
        quiet: bool = True,
    ) -> None:
        self.binary = Path(binary)
        self.prepared_dir = Path(prepared_dir)
        self.quiet = quiet
        self.request_count = 0
        self.closed = False
        cmd = [
            str(self.binary),
            "--prepared",
            str(self.prepared_dir),
            "--generate-server-jsonl",
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=(subprocess.DEVNULL if quiet else None),
            text=True,
            bufsize=1,
        )
        if self._process.stdout is None or self._process.stdin is None:
            self.close()
            raise MetalGenerateError("failed to open glm_moe_infer JSONL pipes")
        started = time.monotonic()
        line = self._process.stdout.readline()
        self.startup_elapsed_seconds = time.monotonic() - started
        if not line:
            self.close()
            raise MetalGenerateError("glm_moe_infer JSONL server exited before ready")
        try:
            ready = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise MetalGenerateError(
                f"failed to parse glm_moe_infer JSONL ready line: {exc}"
            ) from exc
        if ready.get("event") != "ready" or not ready.get("ok"):
            self.close()
            raise MetalGenerateError("glm_moe_infer JSONL server did not report ready")
        self.ready_payload = ready

    def request(self, request: dict[str, object]) -> tuple[dict[str, Any], float]:
        if self.closed:
            raise MetalGenerateError("glm_moe_infer JSONL server session is closed")
        if self._process.stdin is None or self._process.stdout is None:
            raise MetalGenerateError("glm_moe_infer JSONL server pipes are closed")
        started = time.monotonic()
        try:
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            self.close()
            raise MetalGenerateError("glm_moe_infer JSONL server pipe broke") from exc
        line = self._process.stdout.readline()
        elapsed = time.monotonic() - started
        if not line:
            code = self._process.poll()
            self.close()
            raise MetalGenerateError(
                "glm_moe_infer JSONL server exited before response"
                + (f" with code {code}" if code is not None else "")
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise MetalGenerateError(
                f"failed to parse glm_moe_infer JSONL response: {exc}"
            ) from exc
        self.request_count += 1
        if response.get("event") != "request" or not response.get("accepted"):
            error = response.get("error") if isinstance(response, dict) else None
            raise MetalGenerateError(
                "glm_moe_infer JSONL server rejected request"
                + (f": {error}" if error else "")
            )
        if not response.get("ok") or response.get("executor_exit_code") not in (0, None):
            error = response.get("executor_payload_parse_error") or response.get("error")
            raise MetalGenerateError(
                "glm_moe_infer JSONL server execution failed"
                + (f": {error}" if error else "")
            )
        return response, elapsed

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        process = getattr(self, "_process", None)
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.write(
                    json.dumps({"command": "quit"}, separators=(",", ":")) + "\n"
                )
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            if process.stdout is not None and process.poll() is None:
                process.stdout.readline()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def __enter__(self) -> "MetalGenerateServerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def generate_metal_token_ids(
    prepared_dir: str | Path,
    *,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    binary: str | Path = "metal/glm_moe_infer",
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
    top_k: int | None = None,
    logits_top_k: int = 8,
    max_live_working_set_mib: int = 768,
    max_cache_file_mib: float = 64.0,
    max_cache_read_mib: float = 1.0,
    logits_max_chunk_mib: float = 16.0,
    mmap_final_logits: bool = False,
    max_embedding_row_mib: float = 1.0,
    cache_mla_kv_b_f32: bool = False,
    max_mla_kv_b_cache_mib: float = 0.0,
    context1_o_proj_cache_layout: str | Path | None = None,
    context1_o_proj_cache_file: str | Path | None = None,
    allow_decode_only_multi_token_prompt: bool = False,
    prefill_prompt: bool = False,
    prefill_runner: str | Path = "metal/largerlm-runner",
    prefill_prompt_chunk_tokens: int = 64,
    prefill_max_prompt_batch_mib: float = 1024.0,
    prefill_max_cache_write_mib: float = 4096.0,
    prefill_max_runner_scratch_mib: float = 4096.0,
    prefill_max_live_working_set_mib: float | None = None,
    min_free_unified_memory_gib: float | None = None,
    use_generate_server_jsonl: bool = True,
    generate_server_session: MetalGenerateServerSession | None = None,
    use_python_prefill_bridge: bool = False,
    quiet: bool = True,
) -> MetalTokenGenerationResult:
    prompt = tuple(int(token) for token in prompt_token_ids)
    if not prompt:
        raise MetalGenerateError("prompt_token_ids must not be empty")
    if any(token < 0 for token in prompt):
        raise MetalGenerateError("prompt_token_ids must be non-negative")
    if (
        len(prompt) > 1
        and not allow_decode_only_multi_token_prompt
        and not prefill_prompt
    ):
        raise MetalGenerateError(
            "glm_moe_infer is currently a decode-only Metal loop and cannot "
            "consume multi-token prompts correctly until prompt prefill is "
            "folded into the runtime; pass "
            "--prefill-prompt to seed decode from the existing prompt prefill "
            "path, or pass --allow-decode-only-multi-token-prompt to run anyway "
            "using only the last prompt token embedding"
        )
    if max_new_tokens <= 0:
        raise MetalGenerateError("max_new_tokens must be positive")
    if generate_server_session is not None and not use_generate_server_jsonl:
        raise MetalGenerateError(
            "generate_server_session requires use_generate_server_jsonl=True"
        )
    if logits_top_k <= 0:
        raise MetalGenerateError("logits_top_k must be positive")
    manifest = _manifest_from_prepared(prepared_dir)
    config = load_config(manifest.model_dir)
    required_context_tokens = _required_context_tokens(
        prompt_token_count=len(prompt),
        max_new_tokens=max_new_tokens,
        prefill_prompt=prefill_prompt,
    )
    if required_context_tokens > (manifest.max_context_tokens or required_context_tokens):
        raise MetalGenerateError(
            f"required context {required_context_tokens} exceeds prepared max context "
            f"{manifest.max_context_tokens}"
        )
    top_k = int(top_k if top_k is not None else config.experts_per_token)
    if top_k <= 0:
        raise MetalGenerateError("top_k must be positive")
    if logits_top_k != top_k:
        raise MetalGenerateError(
            "glm_moe_infer currently requires logits_top_k to match router top_k"
        )
    if min_free_unified_memory_gib is None:
        recommended_min_free_bytes = getattr(
            manifest,
            "recommended_min_free_unified_memory_bytes",
            None,
        )
        resolved_min_free_unified_memory_gib = (
            recommended_min_free_bytes / 1024**3
            if recommended_min_free_bytes is not None
            else 0.0
        )
    else:
        resolved_min_free_unified_memory_gib = float(min_free_unified_memory_gib)
    if resolved_min_free_unified_memory_gib < 0.0:
        raise MetalGenerateError("min_free_unified_memory_gib must be non-negative")
    if max_mla_kv_b_cache_mib < 0.0:
        raise MetalGenerateError("max_mla_kv_b_cache_mib must be non-negative")
    if cache_mla_kv_b_f32 and max_mla_kv_b_cache_mib <= 0.0:
        raise MetalGenerateError(
            "max_mla_kv_b_cache_mib must be positive when cache_mla_kv_b_f32 is enabled"
        )
    if (
        context1_o_proj_cache_file is not None
        and context1_o_proj_cache_layout is None
    ):
        raise MetalGenerateError(
            "context1_o_proj_cache_file requires context1_o_proj_cache_layout"
        )
    shared = 1 if (config.n_shared_experts or 0) > 0 else 0
    num_heads = _required_int(config.num_attention_heads, "num_attention_heads")
    kv_lora_dim = _required_int(config.kv_lora_rank, "kv_lora_rank")
    qk_nope_dim = _required_int(config.qk_nope_head_dim, "qk_nope_head_dim")
    rope_dim = _required_int(config.qk_rope_head_dim, "qk_rope_head_dim")
    v_head_dim = _required_int(config.v_head_dim, "v_head_dim")
    rms_norm_eps = config.rms_norm_eps if config.rms_norm_eps is not None else 1e-5
    rope_theta = config.rope_theta if config.rope_theta is not None else 10000.0
    if (
        generate_server_session is not None
        and Path(generate_server_session.prepared_dir) != manifest.manifest_path.parent
    ):
        raise MetalGenerateError(
            "generate_server_session prepared_dir does not match request prepared_dir"
        )

    def run_generate_server_request(
        request: dict[str, object],
    ) -> tuple[dict[str, Any], float]:
        if generate_server_session is not None:
            return generate_server_session.request(request)
        return _run_generate_server_jsonl_request(
            binary=binary,
            prepared_dir=manifest.manifest_path.parent,
            request=request,
            quiet=quiet,
        )

    temp_root: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        if keep_work_dir:
            root = Path(tempfile.mkdtemp(prefix="largerlm-metal-generate-"))
        else:
            temp_root = tempfile.TemporaryDirectory(prefix="largerlm-metal-generate-")
            root = Path(temp_root.name)
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
    try:
        input_f32 = root / "input.f32"
        output_f32 = root / "output.f32"
        next_input_f32 = root / "next_input.f32"
        generated_json = root / "generated.json"
        runner_work = root / "glm_moe_infer_work"
        cache_layout_path = root / "decode_cache_layout.json"
        cache_file_path = root / "decode_cache.bin"
        layout = build_decode_cache_layout(
            config,
            max_context_tokens=required_context_tokens,
            dtype="F32",
            alignment=64,
            max_cache_bytes=int(max_cache_file_mib * 1024 * 1024),
        )
        cache_layout_path.write_text(
            json.dumps(layout.to_json(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cache_init = init_decode_cache_file(
            cache_layout_path,
            cache_file_path,
            force=True,
            max_cache_bytes=int(max_cache_file_mib * 1024 * 1024),
            disk_safety_margin_bytes=0,
        )
        prompt_prefill_result: PromptPrefillResult | None = None
        prefill_final_logits_elapsed: float | None = None
        generated: tuple[int, ...]
        decode_elapsed: tuple[float, ...]
        logits_elapsed: tuple[float, ...]
        payload: dict[str, Any]
        if prefill_prompt and len(prompt) > 1 and not use_python_prefill_bridge:
            request = _build_generate_request_payload(
                decode_layers=_layers_csv(config),
                prompt_token_ids=prompt,
                output_dir=runner_work,
                generate_steps=max_new_tokens,
                generate_first_from_input_logits=False,
                cache_layout=cache_layout_path,
                cache_file=cache_file_path,
                position=0,
                context_length=1,
                num_heads=num_heads,
                kv_lora_dim=kv_lora_dim,
                qk_nope_dim=qk_nope_dim,
                rope_dim=rope_dim,
                v_head_dim=v_head_dim,
                cache_position_offset=0,
                top_k=top_k,
                expert_buffer_count=top_k + shared,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                max_cache_file_mib=max_cache_file_mib,
                max_cache_read_mib=max_cache_read_mib,
                max_chunk_mib=logits_max_chunk_mib,
                mmap_final_logits=mmap_final_logits,
                max_embedding_row_mib=max_embedding_row_mib,
                max_live_working_set_mib=max_live_working_set_mib,
                min_free_unified_memory_gib=resolved_min_free_unified_memory_gib,
                include_shared_expert=bool(shared),
                rope_interleave=bool(config.rope_interleave),
                in_memory_decode_cache=True,
                cache_mla_kv_b_f32=cache_mla_kv_b_f32,
                max_mla_kv_b_cache_mib=(
                    max_mla_kv_b_cache_mib if cache_mla_kv_b_f32 else None
                ),
                context1_o_proj_cache_layout=context1_o_proj_cache_layout,
                context1_o_proj_cache_file=context1_o_proj_cache_file,
            )
            if use_generate_server_jsonl:
                payload, elapsed = run_generate_server_request(request)
            else:
                request_path = root / "runtime_prompt_generate_request.json"
                request_with_debug_outputs = dict(request)
                request_with_debug_outputs["output_f32"] = str(output_f32)
                request_with_debug_outputs["output_generated_json"] = str(
                    generated_json
                )
                request_with_debug_outputs["output_next_input_f32"] = str(
                    next_input_f32
                )
                request_with_debug_outputs["in_memory_decode_cache"] = False
                request_path.write_text(
                    json.dumps(
                        request_with_debug_outputs,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                cmd = [
                    str(binary),
                    "--prepared",
                    str(manifest.manifest_path.parent),
                    "--generate-request-json",
                    str(request_path),
                    "--json",
                ]
                started = time.monotonic()
                completed = _run_command(cmd, quiet=quiet)
                elapsed = time.monotonic() - started
                try:
                    payload = json.loads(completed.stdout)
                except json.JSONDecodeError as exc:
                    raise MetalGenerateError(
                        f"failed to parse glm_moe_infer JSON: {exc}"
                    ) from exc
            generate = payload.get("probe_generate")
            if not isinstance(generate, dict) or not generate.get("ok"):
                raise MetalGenerateError("glm_moe_infer did not report probe_generate ok")
            if generate.get("entry") != "generate_token_ids":
                raise MetalGenerateError(
                    "glm_moe_infer did not report generate_token_ids entry"
                )
            prefill_payload = generate.get("prompt_prefill")
            if not isinstance(prefill_payload, dict) or not prefill_payload.get("ok"):
                raise MetalGenerateError(
                    "glm_moe_infer did not report prompt prefill ok"
                )
            generated = tuple(
                int(item) for item in generate.get("generated_token_ids", ())
            )
            if len(generated) != max_new_tokens:
                raise MetalGenerateError(
                    f"generated {len(generated)} tokens, expected {max_new_tokens}"
                )
            steps = generate.get("steps")
            if not isinstance(steps, list):
                raise MetalGenerateError("glm_moe_infer generate steps missing")
            decode_elapsed = _step_float_tuple(steps, "decode_elapsed_seconds")
            logits_elapsed = _step_float_tuple(
                steps,
                "final_logits_elapsed_seconds",
            )
            prefill_steps = prefill_payload.get("steps")
            prompt_prefill_elapsed = None
            if isinstance(prefill_steps, list):
                prompt_prefill_elapsed = sum(
                    float(step.get("decode_elapsed_seconds", 0.0))
                    for step in prefill_steps
                    if isinstance(step, dict)
                )
            return MetalTokenGenerationResult(
                prepared_dir=manifest.manifest_path.parent,
                binary=Path(binary),
                prompt_token_ids=prompt,
                generated_token_ids=generated,
                work_dir=root,
                kept_work_dir=keep_work_dir or work_dir is not None,
                cache_layout_path=cache_layout_path,
                cache_file_path=cache_file_path,
                input_f32_path=input_f32,
                output_f32_path=output_f32,
                generated_json_path=generated_json,
                elapsed_seconds=elapsed,
                decode_elapsed_seconds=decode_elapsed,
                final_logits_elapsed_seconds=logits_elapsed,
                **_final_logits_result_fields(steps),
                estimated_live_working_set_bytes=int(
                    payload.get("estimated_live_working_set_bytes", 0)
                ),
                max_live_working_set_mib=max_live_working_set_mib,
                cache_total_bytes=cache_init.total_bytes,
                min_free_unified_memory_gib=resolved_min_free_unified_memory_gib,
                admission_ok=_payload_bool(payload, "admission_ok"),
                available_unified_memory_ok=_payload_bool(
                    payload,
                    "available_unified_memory_ok",
                ),
                system_available_memory_bytes=_payload_int(
                    payload,
                    "system_available_memory_bytes",
                ),
                required_available_memory_bytes=_payload_int(
                    payload,
                    "required_available_memory_bytes",
                ),
                expert_buffer_count_allocated=_payload_int_any(
                    payload,
                    "expert_buffer_count_allocated",
                    "expert_buffer_count_runtime_allocated",
                ),
                decode_expert_bytes_read=_step_int_tuple(
                    steps,
                    "expert_bytes_read",
                ),
                decode_dense_mlp_bytes_read=_step_int_tuple(
                    steps,
                    "dense_mlp_bytes_read",
                ),
                **_decode_layer_timing_result_fields(steps),
                decode_expert_read_seconds=_step_float_tuple(
                    steps,
                    "expert_read_seconds",
                ),
                decode_moe_mlp_kernel_seconds=_step_float_tuple(
                    steps,
                    "moe_mlp_kernel_seconds",
                ),
                decode_expert_read_dispatch_count=_step_int_tuple(
                    steps,
                    "expert_read_dispatch_count",
                ),
                decode_expert_read_task_count=_step_int_tuple(
                    steps,
                    "expert_read_task_count",
                ),
                decode_expert_read_max_task_count=_step_int_tuple(
                    steps,
                    "expert_read_max_task_count",
                ),
                decode_expert_read_max_worker_count=_step_int_tuple(
                    steps,
                    "expert_read_max_worker_count",
                ),
                decode_expert_read_pool_dispatch_count=_step_int_tuple(
                    steps,
                    "expert_read_pool_dispatch_count",
                ),
                decode_expert_read_serial_dispatch_count=_step_int_tuple(
                    steps,
                    "expert_read_serial_dispatch_count",
                ),
                **_mla_kv_b_cache_result_fields(payload, steps),
                prompt_prefill_elapsed_seconds=prompt_prefill_elapsed,
                metal_elapsed_seconds=elapsed,
                prefill_final_logits_elapsed_seconds=(
                    logits_elapsed[0] if logits_elapsed else None
                ),
                note=(
                    "runtime prompt prefill path: glm_moe_infer consumes "
                    "prompt_token_ids, writes decode cache, and generates "
                    "tokens inside the "
                    + (
                        "persistent JSONL service"
                        if use_generate_server_jsonl
                        else "direct request-json entry"
                    )
                    + "; this is sequential prompt prefill, not yet batched"
                ),
            )

        if prefill_prompt and len(prompt) > 1:
            prefill_last_hidden = root / "prefill_last_hidden.f32"
            resolved_prefill_max_live_working_set_mib = (
                float(max_live_working_set_mib)
                if prefill_max_live_working_set_mib is None
                else float(prefill_max_live_working_set_mib)
            )
            prompt_prefill_result = run_prompt_prefill(
                runner_path=prefill_runner,
                expert_layout_path=manifest.experts_layout,
                resident_layout_path=manifest.resident_layout,
                cache_layout_path=cache_layout_path,
                cache_file_path=cache_file_path,
                prompt_token_ids=prompt,
                output_last_hidden_f32_path=prefill_last_hidden,
                output_final_chunk_f32_path=(
                    root / "prefill_final_chunk.f32" if keep_work_dir else None
                ),
                layers=range(config.num_hidden_layers),
                dense_layers=_dense_layers_tuple(config),
                work_dir=root / "prefill_prompt",
                keep_work_dir=keep_work_dir,
                start_position=0,
                prompt_chunk_tokens=prefill_prompt_chunk_tokens,
                max_prompt_batch_mib=prefill_max_prompt_batch_mib,
                num_heads=num_heads,
                qk_nope_dim=qk_nope_dim,
                rope_dim=rope_dim,
                v_head_dim=v_head_dim,
                kv_lora_dim=kv_lora_dim,
                rope_theta=rope_theta,
                rope_interleave=config.rope_interleave,
                top_k=top_k,
                max_k=max(8, top_k),
                router_score=_router_score(config),
                routed_scaling_factor=config.routed_scaling_factor,
                norm_topk_prob=bool(config.norm_topk_prob),
                router_n_group=config.n_group,
                router_topk_group=config.topk_group,
                include_shared_expert=bool(shared),
                rms_norm_eps=rms_norm_eps,
                max_embedding_row_mib=max_embedding_row_mib,
                max_cache_file_mib=max_cache_file_mib,
                max_cache_write_mib=prefill_max_cache_write_mib,
                max_cache_read_mib=max_cache_read_mib,
                max_runner_scratch_mib=prefill_max_runner_scratch_mib,
                max_live_working_set_mib=resolved_prefill_max_live_working_set_mib,
                min_free_unified_memory_gib=(
                    resolved_min_free_unified_memory_gib
                ),
                expected_vocab_size=config.vocab_size,
                expected_hidden_size=config.hidden_size,
                echo_runner_output=not quiet,
            )
            if max_new_tokens > 1:
                request = _build_generate_request_payload(
                    decode_layers=_layers_csv(config),
                    input_f32=prefill_last_hidden,
                    output_dir=runner_work,
                    generate_steps=max_new_tokens,
                    generate_first_from_input_logits=True,
                    cache_layout=cache_layout_path,
                    cache_file=cache_file_path,
                    position=len(prompt),
                    context_length=len(prompt) + 1,
                    num_heads=num_heads,
                    kv_lora_dim=kv_lora_dim,
                    qk_nope_dim=qk_nope_dim,
                    rope_dim=rope_dim,
                    v_head_dim=v_head_dim,
                    cache_position_offset=0,
                    top_k=top_k,
                    expert_buffer_count=top_k + shared,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                    max_cache_file_mib=max_cache_file_mib,
                    max_cache_read_mib=max_cache_read_mib,
                    max_chunk_mib=logits_max_chunk_mib,
                    mmap_final_logits=mmap_final_logits,
                    max_embedding_row_mib=max_embedding_row_mib,
                    max_live_working_set_mib=max_live_working_set_mib,
                    min_free_unified_memory_gib=resolved_min_free_unified_memory_gib,
                    include_shared_expert=bool(shared),
                    rope_interleave=bool(config.rope_interleave),
                    in_memory_decode_cache=True,
                    cache_mla_kv_b_f32=cache_mla_kv_b_f32,
                    max_mla_kv_b_cache_mib=(
                        max_mla_kv_b_cache_mib if cache_mla_kv_b_f32 else None
                    ),
                    context1_o_proj_cache_layout=context1_o_proj_cache_layout,
                    context1_o_proj_cache_file=context1_o_proj_cache_file,
                )
                if use_generate_server_jsonl:
                    payload, metal_elapsed = run_generate_server_request(request)
                else:
                    request_path = root / "prefill_generate_request.json"
                    request_with_debug_outputs = dict(request)
                    request_with_debug_outputs["output_f32"] = str(output_f32)
                    request_with_debug_outputs["output_generated_json"] = str(
                        generated_json
                    )
                    request_with_debug_outputs["output_next_input_f32"] = str(
                        next_input_f32
                    )
                    request_with_debug_outputs["in_memory_decode_cache"] = False
                    request_path.write_text(
                        json.dumps(
                            request_with_debug_outputs,
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    cmd = [
                        str(binary),
                        "--prepared",
                        str(manifest.manifest_path.parent),
                        "--generate-request-json",
                        str(request_path),
                        "--json",
                    ]
                    started = time.monotonic()
                    completed = _run_command(cmd, quiet=quiet)
                    metal_elapsed = time.monotonic() - started
                    try:
                        payload = json.loads(completed.stdout)
                    except json.JSONDecodeError as exc:
                        raise MetalGenerateError(
                            f"failed to parse glm_moe_infer JSON: {exc}"
                        ) from exc
                elapsed = metal_elapsed + prompt_prefill_result.elapsed_seconds
                generate = payload.get("probe_generate")
                if not isinstance(generate, dict) or not generate.get("ok"):
                    raise MetalGenerateError(
                        "glm_moe_infer did not report probe_generate ok"
                    )
                if generate.get("entry") != "generate_token_ids":
                    raise MetalGenerateError(
                        "glm_moe_infer did not report generate_token_ids entry"
                    )
                if not generate.get("first_step_from_input_logits"):
                    raise MetalGenerateError(
                        "glm_moe_infer did not report prefill-continuation generate mode"
                    )
                generated = tuple(
                    int(item) for item in generate.get("generated_token_ids", ())
                )
                if len(generated) != max_new_tokens:
                    raise MetalGenerateError(
                        f"generated {len(generated)} tokens, expected {max_new_tokens}"
                    )
                steps = generate.get("steps")
                if not isinstance(steps, list):
                    raise MetalGenerateError("glm_moe_infer generate steps missing")
                decode_elapsed = _step_float_tuple(steps, "decode_elapsed_seconds")
                logits_elapsed = _step_float_tuple(
                    steps,
                    "final_logits_elapsed_seconds",
                )
                prefill_final_logits_elapsed = (
                    logits_elapsed[0] if logits_elapsed else None
                )
                return MetalTokenGenerationResult(
                    prepared_dir=manifest.manifest_path.parent,
                    binary=Path(binary),
                    prompt_token_ids=prompt,
                    generated_token_ids=generated,
                    work_dir=root,
                    kept_work_dir=keep_work_dir or work_dir is not None,
                    cache_layout_path=cache_layout_path,
                    cache_file_path=cache_file_path,
                    input_f32_path=prefill_last_hidden,
                    output_f32_path=output_f32,
                    generated_json_path=generated_json,
                    elapsed_seconds=elapsed,
                    decode_elapsed_seconds=decode_elapsed,
                    final_logits_elapsed_seconds=logits_elapsed,
                    **_final_logits_result_fields(steps),
                    estimated_live_working_set_bytes=int(
                        payload.get("estimated_live_working_set_bytes", 0)
                    ),
                    max_live_working_set_mib=max_live_working_set_mib,
                    cache_total_bytes=cache_init.total_bytes,
                    min_free_unified_memory_gib=(
                        resolved_min_free_unified_memory_gib
                    ),
                    admission_ok=_payload_bool(payload, "admission_ok"),
                    available_unified_memory_ok=_payload_bool(
                        payload,
                        "available_unified_memory_ok",
                    ),
                    system_available_memory_bytes=_payload_int(
                        payload,
                        "system_available_memory_bytes",
                    ),
                    required_available_memory_bytes=_payload_int(
                        payload,
                        "required_available_memory_bytes",
                    ),
                    expert_buffer_count_allocated=_payload_int_any(
                        payload,
                        "expert_buffer_count_allocated",
                        "expert_buffer_count_runtime_allocated",
                    ),
                    decode_expert_bytes_read=_step_int_tuple(
                        steps,
                        "expert_bytes_read",
                    ),
                    decode_dense_mlp_bytes_read=_step_int_tuple(
                        steps,
                        "dense_mlp_bytes_read",
                    ),
                    **_decode_layer_timing_result_fields(steps),
                    decode_expert_read_seconds=_step_float_tuple(
                        steps,
                        "expert_read_seconds",
                    ),
                    decode_moe_mlp_kernel_seconds=_step_float_tuple(
                        steps,
                        "moe_mlp_kernel_seconds",
                    ),
                    decode_expert_read_dispatch_count=_step_int_tuple(
                        steps,
                        "expert_read_dispatch_count",
                    ),
                    decode_expert_read_task_count=_step_int_tuple(
                        steps,
                        "expert_read_task_count",
                    ),
                    decode_expert_read_max_task_count=_step_int_tuple(
                        steps,
                        "expert_read_max_task_count",
                    ),
                    decode_expert_read_max_worker_count=_step_int_tuple(
                        steps,
                        "expert_read_max_worker_count",
                    ),
                    decode_expert_read_pool_dispatch_count=_step_int_tuple(
                        steps,
                        "expert_read_pool_dispatch_count",
                    ),
                    decode_expert_read_serial_dispatch_count=_step_int_tuple(
                        steps,
                        "expert_read_serial_dispatch_count",
                    ),
                    **_mla_kv_b_cache_result_fields(payload, steps),
                    note=(
                        "prefill bridge path: uses existing prompt prefill to seed "
                        "decode cache, then runs first-token logits and remaining "
                        "decode steps through the "
                        + (
                            "persistent glm_moe_infer JSONL service"
                            if use_generate_server_jsonl
                            else "direct request-json entry"
                        )
                        + "; prompt prefill is not yet folded into that runtime; "
                        "estimated_live_working_set_bytes reports the Metal "
                        "logits/decode stage, while the prompt_prefill payload "
                        "reports prefill work separately"
                    ),
                    prompt_prefill=prompt_prefill_result,
                    prompt_prefill_elapsed_seconds=(
                        prompt_prefill_result.elapsed_seconds
                    ),
                    prompt_prefill_estimated_live_working_set_bytes=(
                        _prompt_prefill_estimated_live_working_set_bytes(
                            prompt_prefill_result
                        )
                    ),
                    prefill_max_live_working_set_mib=(
                        resolved_prefill_max_live_working_set_mib
                    ),
                    metal_elapsed_seconds=metal_elapsed,
                    prefill_final_logits_elapsed_seconds=(
                        prefill_final_logits_elapsed
                    ),
                )
            prefill_token_json = root / "prefill_token.json"
            cmd = [
                str(binary),
                "--prepared",
                str(manifest.manifest_path.parent),
                "--probe-final-logits",
                "--input-f32",
                str(prefill_last_hidden),
                "--output-token-json",
                str(prefill_token_json),
                "--output-next-input-f32",
                str(next_input_f32),
                "--top-k",
                str(top_k),
                "--rms-norm-eps",
                f"{rms_norm_eps:.9g}",
                "--max-chunk-mib",
                f"{logits_max_chunk_mib:.9g}",
                "--max-embedding-row-mib",
                f"{max_embedding_row_mib:.9g}",
                "--max-live-working-set-mib",
                str(max_live_working_set_mib),
                "--min-free-unified-memory-gib",
                f"{resolved_min_free_unified_memory_gib:.9g}",
                "--json",
            ]
            started = time.monotonic()
            completed = _run_command(cmd, quiet=quiet)
            first_elapsed = time.monotonic() - started
            try:
                first_payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise MetalGenerateError(
                    f"failed to parse glm_moe_infer final-logits JSON: {exc}"
                ) from exc
            logits_payload = first_payload.get("probe_final_logits")
            if not isinstance(logits_payload, dict) or not logits_payload.get("ok"):
                raise MetalGenerateError(
                    "glm_moe_infer did not report probe_final_logits ok"
                )
            generated_token = logits_payload.get("generated_token")
            if not isinstance(generated_token, dict):
                raise MetalGenerateError("glm_moe_infer final logits token missing")
            first_token = int(generated_token["token_id"])
            prefill_final_logits_elapsed = float(
                logits_payload.get("elapsed_seconds", first_elapsed)
            )
            metal_elapsed = first_elapsed
            elapsed = first_elapsed + prompt_prefill_result.elapsed_seconds
            generated = (first_token,)
            decode_elapsed = (0.0,)
            logits_elapsed = (prefill_final_logits_elapsed,)
            payload = first_payload
            if "estimated_live_working_set_bytes" not in payload:
                payload["estimated_live_working_set_bytes"] = first_payload.get(
                    "estimated_live_working_set_bytes",
                    0,
                )
            return MetalTokenGenerationResult(
                prepared_dir=manifest.manifest_path.parent,
                binary=Path(binary),
                prompt_token_ids=prompt,
                generated_token_ids=generated,
                work_dir=root,
                kept_work_dir=keep_work_dir or work_dir is not None,
                cache_layout_path=cache_layout_path,
                cache_file_path=cache_file_path,
                input_f32_path=prefill_last_hidden,
                output_f32_path=output_f32,
                generated_json_path=generated_json,
                elapsed_seconds=elapsed,
                decode_elapsed_seconds=decode_elapsed,
                final_logits_elapsed_seconds=logits_elapsed,
                estimated_live_working_set_bytes=int(
                    payload.get("estimated_live_working_set_bytes", 0)
                ),
                max_live_working_set_mib=max_live_working_set_mib,
                cache_total_bytes=cache_init.total_bytes,
                min_free_unified_memory_gib=resolved_min_free_unified_memory_gib,
                admission_ok=_payload_bool(payload, "admission_ok"),
                available_unified_memory_ok=_payload_bool(
                    payload,
                    "available_unified_memory_ok",
                ),
                system_available_memory_bytes=_payload_int(
                    payload,
                    "system_available_memory_bytes",
                ),
                required_available_memory_bytes=_payload_int(
                    payload,
                    "required_available_memory_bytes",
                ),
                expert_buffer_count_allocated=_payload_int_any(
                    payload,
                    "expert_buffer_count_allocated",
                    "expert_buffer_count_runtime_allocated",
                ),
                note=(
                    "prefill bridge path: uses existing prompt prefill to seed "
                    "decode cache and first-token logits; prompt prefill is not "
                    "yet folded into the single-process glm_moe_infer loop; "
                    "estimated_live_working_set_bytes reports the Metal logits/"
                    "decode stage, while the prompt_prefill payload reports "
                    "prefill work separately"
                ),
                prompt_prefill=prompt_prefill_result,
                prompt_prefill_elapsed_seconds=prompt_prefill_result.elapsed_seconds,
                prompt_prefill_estimated_live_working_set_bytes=(
                    _prompt_prefill_estimated_live_working_set_bytes(
                        prompt_prefill_result
                    )
                ),
                prefill_max_live_working_set_mib=(
                    resolved_prefill_max_live_working_set_mib
                ),
                metal_elapsed_seconds=metal_elapsed,
                prefill_final_logits_elapsed_seconds=prefill_final_logits_elapsed,
            )
        else:
            input_token_id = prompt[-1]
        request = _build_generate_request_payload(
            decode_layers=_layers_csv(config),
            input_token_id=input_token_id,
            output_dir=runner_work,
            generate_steps=max_new_tokens,
            generate_first_from_input_logits=False,
            cache_layout=cache_layout_path,
            cache_file=cache_file_path,
            position=0,
            context_length=1,
            num_heads=num_heads,
            kv_lora_dim=kv_lora_dim,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            cache_position_offset=0,
            top_k=top_k,
            expert_buffer_count=top_k + shared,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            max_cache_file_mib=max_cache_file_mib,
            max_cache_read_mib=max_cache_read_mib,
            max_chunk_mib=logits_max_chunk_mib,
            mmap_final_logits=mmap_final_logits,
            max_embedding_row_mib=max_embedding_row_mib,
            max_live_working_set_mib=max_live_working_set_mib,
            min_free_unified_memory_gib=resolved_min_free_unified_memory_gib,
            include_shared_expert=bool(shared),
            rope_interleave=bool(config.rope_interleave),
            in_memory_decode_cache=True,
            cache_mla_kv_b_f32=cache_mla_kv_b_f32,
            max_mla_kv_b_cache_mib=(
                max_mla_kv_b_cache_mib if cache_mla_kv_b_f32 else None
            ),
            context1_o_proj_cache_layout=context1_o_proj_cache_layout,
            context1_o_proj_cache_file=context1_o_proj_cache_file,
        )
        if use_generate_server_jsonl:
            payload, elapsed = run_generate_server_request(request)
        else:
            request_path = root / "generate_request.json"
            request_with_debug_outputs = dict(request)
            request_with_debug_outputs["output_f32"] = str(output_f32)
            request_with_debug_outputs["output_generated_json"] = str(generated_json)
            request_with_debug_outputs["output_next_input_f32"] = str(next_input_f32)
            request_with_debug_outputs["in_memory_decode_cache"] = False
            request_path.write_text(
                json.dumps(request_with_debug_outputs, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            cmd = [
                str(binary),
                "--prepared",
                str(manifest.manifest_path.parent),
                "--generate-request-json",
                str(request_path),
                "--json",
            ]
            started = time.monotonic()
            completed = _run_command(cmd, quiet=quiet)
            elapsed = time.monotonic() - started
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise MetalGenerateError(
                    f"failed to parse glm_moe_infer JSON: {exc}"
                ) from exc
        generate = payload.get("probe_generate")
        if not isinstance(generate, dict) or not generate.get("ok"):
            raise MetalGenerateError("glm_moe_infer did not report probe_generate ok")
        if generate.get("entry") != "generate_token_ids":
            raise MetalGenerateError(
                "glm_moe_infer did not report generate_token_ids entry"
            )
        generated = tuple(int(item) for item in generate.get("generated_token_ids", ()))
        if len(generated) != max_new_tokens:
            raise MetalGenerateError(
                f"generated {len(generated)} tokens, expected {max_new_tokens}"
            )
        steps = generate.get("steps")
        if not isinstance(steps, list):
            raise MetalGenerateError("glm_moe_infer generate steps missing")
        decode_elapsed = _step_float_tuple(steps, "decode_elapsed_seconds")
        logits_elapsed = _step_float_tuple(steps, "final_logits_elapsed_seconds")
        return MetalTokenGenerationResult(
            prepared_dir=manifest.manifest_path.parent,
            binary=Path(binary),
            prompt_token_ids=prompt,
            generated_token_ids=generated,
            work_dir=root,
            kept_work_dir=keep_work_dir or work_dir is not None,
            cache_layout_path=cache_layout_path,
            cache_file_path=cache_file_path,
            input_f32_path=input_f32,
            output_f32_path=output_f32,
            generated_json_path=generated_json,
            elapsed_seconds=elapsed,
            decode_elapsed_seconds=decode_elapsed,
            final_logits_elapsed_seconds=logits_elapsed,
            **_final_logits_result_fields(steps),
            estimated_live_working_set_bytes=int(
                payload.get("estimated_live_working_set_bytes", 0)
            ),
            max_live_working_set_mib=max_live_working_set_mib,
            cache_total_bytes=cache_init.total_bytes,
            min_free_unified_memory_gib=resolved_min_free_unified_memory_gib,
            admission_ok=_payload_bool(payload, "admission_ok"),
            available_unified_memory_ok=_payload_bool(
                payload,
                "available_unified_memory_ok",
            ),
            system_available_memory_bytes=_payload_int(
                payload,
                "system_available_memory_bytes",
            ),
            required_available_memory_bytes=_payload_int(
                payload,
                "required_available_memory_bytes",
            ),
            expert_buffer_count_allocated=_payload_int_any(
                payload,
                "expert_buffer_count_allocated",
                "expert_buffer_count_runtime_allocated",
            ),
            decode_expert_bytes_read=_step_int_tuple(
                steps,
                "expert_bytes_read",
            ),
            decode_dense_mlp_bytes_read=_step_int_tuple(
                steps,
                "dense_mlp_bytes_read",
            ),
            **_decode_layer_timing_result_fields(steps),
            decode_expert_read_seconds=_step_float_tuple(
                steps,
                "expert_read_seconds",
            ),
            decode_moe_mlp_kernel_seconds=_step_float_tuple(
                steps,
                "moe_mlp_kernel_seconds",
            ),
            decode_expert_read_dispatch_count=_step_int_tuple(
                steps,
                "expert_read_dispatch_count",
            ),
            decode_expert_read_task_count=_step_int_tuple(
                steps,
                "expert_read_task_count",
            ),
            decode_expert_read_max_task_count=_step_int_tuple(
                steps,
                "expert_read_max_task_count",
            ),
            decode_expert_read_max_worker_count=_step_int_tuple(
                steps,
                "expert_read_max_worker_count",
            ),
            decode_expert_read_pool_dispatch_count=_step_int_tuple(
                steps,
                "expert_read_pool_dispatch_count",
            ),
            decode_expert_read_serial_dispatch_count=_step_int_tuple(
                steps,
                "expert_read_serial_dispatch_count",
            ),
            **_mla_kv_b_cache_result_fields(payload, steps),
            note=(
                "decode-only path: glm_moe_infer decodes the last prompt token "
                "embedding as the initial hidden state through the "
                + (
                    "persistent JSONL service"
                    if use_generate_server_jsonl
                    else "direct request-json entry"
                )
                + "; prompt prefill is not yet folded into glm_moe_infer"
            ),
            metal_elapsed_seconds=elapsed,
            input_token_id=input_token_id,
        )
    finally:
        if temp_root is not None:
            temp_root.cleanup()
