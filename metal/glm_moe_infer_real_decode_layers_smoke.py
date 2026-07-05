#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_moe_infer_real_attn_projections_smoke import (
    mxfp4_dims,
    tensor_map,
    vector_dim,
    write_input,
)
from glm_moe_infer_real_final_logits_smoke import compare_topk
from glm_moe_infer_real_layer_moe_smoke import assert_layer_moe_read_telemetry
from glm_moe_infer_real_mlp_block_smoke import compare_outputs
from largerlm.embedding import embed_token


def run_command(
    cmd: list[str], *, env: dict[str, str] | None = None, echo: bool = True
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if echo and completed.stdout:
        print(completed.stdout, end="")
    if echo and completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0 and not echo:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def without_options(cmd: list[str], options_with_value_counts: dict[str, int]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(cmd):
        value_count = options_with_value_counts.get(cmd[index])
        if value_count is not None:
            index += 1 + value_count
            continue
        result.append(cmd[index])
        index += 1
    return result


def align(value: int, alignment: int) -> int:
    rem = value % alignment
    return value if rem == 0 else value + (alignment - rem)


def write_f32_cache(
    root: Path,
    layers: list[int],
    width: int,
    *,
    max_context_tokens: int = 1,
) -> tuple[Path, Path, Path]:
    cache_layout = root / "cache_layout.json"
    old_cache = root / "old_decode_cache.bin"
    new_cache = root / "new_decode_cache.bin"
    segments = []
    cursor = 0
    row_bytes = width * 4
    for layer in layers:
        cursor = align(cursor, 64)
        total_bytes = row_bytes * max_context_tokens
        segments.append(
            {
                "kind": "mla_kv",
                "layer": layer,
                "offset": cursor,
                "width": width,
                "dtype": "F32",
                "dtype_bytes": 4,
                "token_stride_bytes": row_bytes,
                "max_context_tokens": max_context_tokens,
                "total_bytes": total_bytes,
            }
        )
        cursor += total_bytes
    raw = b"\0" * cursor
    old_cache.write_bytes(raw)
    new_cache.write_bytes(raw)
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": max_context_tokens,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 64,
                "total_bytes": cursor,
                "segments": segments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_layout, old_cache, new_cache


def expert_layer_ids(expert_layout: Path) -> list[int]:
    data = json.loads(expert_layout.read_text(encoding="utf-8"))
    return [int(item["layer"]) for item in data.get("layers", [])]


def compare_f32_files(old_path: Path, new_path: Path) -> dict[str, float | int]:
    old_raw = old_path.read_bytes()
    new_raw = new_path.read_bytes()
    if len(old_raw) != len(new_raw):
        raise SystemExit(f"f32 file byte length mismatch: {len(old_raw)} != {len(new_raw)}")
    if len(old_raw) % 4 != 0:
        raise SystemExit("f32 file byte length is not a multiple of 4")
    count = len(old_raw) // 4
    old = struct.unpack(f"<{count}f", old_raw)
    new = struct.unpack(f"<{count}f", new_raw)
    diffs = [abs(a - b) for a, b in zip(old, new)]
    max_index = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    return {
        "count": count,
        "max_abs_diff": diffs[max_index] if diffs else 0.0,
        "max_index": max_index,
        "old_at_max": old[max_index] if diffs else 0.0,
        "new_at_max": new[max_index] if diffs else 0.0,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer decode layer-list probe against largerlm-runner on real GLM dense->MoE layers."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--dense-layers", default="0")
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Run every prepared GLM layer, deriving dense layers from the expert layout.",
    )
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Skip the largerlm-runner oracle and only validate the new runtime safety/telemetry path.",
    )
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-interleave", action="store_true")
    parser.add_argument("--include-shared-expert", action="store_true", default=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=2e-5)
    parser.add_argument(
        "--all-layers-max-diff",
        type=float,
        default=5e-4,
        help="Hidden-vector max diff tolerance for full-layer oracle runs.",
    )
    parser.add_argument(
        "--quiet-commands",
        action="store_true",
        help="Only print child runner output if a command fails.",
    )
    parser.add_argument(
        "--wrap-resident-metal",
        action="store_true",
        help="Wrap the resident mmap as a Metal buffer and validate direct resident weight reads.",
    )
    parser.add_argument(
        "--mmap-final-logits",
        action="store_true",
        help="Mmap only final logits lm_head ranges as Metal buffers.",
    )
    parser.add_argument("--max-live-working-set-mib", type=int, default=768)
    args = parser.parse_args()

    resident_layout = args.prepared / "resident" / "layout.json"
    expert_layout = args.prepared / "experts" / "layout.json"
    if not resident_layout.exists() or not expert_layout.exists():
        print(f"prepared package missing, skipping: {args.prepared}")
        return 0

    if args.all_layers:
        expert_layers = expert_layer_ids(expert_layout)
        if not expert_layers:
            raise SystemExit("expert layout has no layers")
        max_layer = max(expert_layers)
        layers = list(range(max_layer + 1))
        expert_layer_set = set(expert_layers)
        dense_layers = [layer for layer in layers if layer not in expert_layer_set]
    else:
        layers = [int(item) for item in args.layers.split(",") if item]
        dense_layers = [int(item) for item in args.dense_layers.split(",") if item]
    if not layers:
        raise SystemExit("--layers must not be empty")
    tensors = tensor_map(resident_layout)
    prefix = f"model.layers.{layers[0]}"
    q_a_out, hidden_dim, _ = mxfp4_dims(tensors, f"{prefix}.self_attn.q_a_proj.weight")
    q_b_out, q_b_in, _ = mxfp4_dims(tensors, f"{prefix}.self_attn.q_b_proj.weight")
    kv_a_out, kv_a_in, _ = mxfp4_dims(
        tensors,
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
    )
    kv_lora_dim = vector_dim(tensors, f"{prefix}.self_attn.kv_a_layernorm.weight")
    kv_rope_dim = kv_a_out - kv_lora_dim
    embed_shape = tensors[f"{prefix}.self_attn.embed_q.weight"]["shape"]
    unembed_shape = tensors[f"{prefix}.self_attn.unembed_out.weight"]["shape"]
    num_heads = int(embed_shape[0])
    qk_nope_dim = int(embed_shape[2]) * 8
    v_head_dim = int(unembed_shape[1])
    if (
        q_b_in != q_a_out
        or kv_a_in != hidden_dim
        or int(embed_shape[1]) != kv_lora_dim
        or int(unembed_shape[0]) != num_heads
        or int(unembed_shape[2]) * 8 != kv_lora_dim
        or q_b_out != num_heads * (qk_nope_dim + kv_rope_dim)
    ):
        raise SystemExit("real GLM decode-layer dims are inconsistent")

    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-glm-decode-layers-real-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.f32"
    old_output = root / "old_decode_layers_out.f32"
    new_output = root / "new_decode_layers_out.f32"
    old_topk = root / "old_topk.json"
    new_topk = root / "new_topk.json"
    new_token = root / "new_token.json"
    new_next_input = root / "new_next_input.f32"
    expected_next_input = root / "expected_next_input.f32"
    old_report = root / "old_report.json"
    old_work = root / "old_work"
    new_work = root / "new_work"
    cache_layout, old_cache, new_cache = write_f32_cache(root, layers, kv_a_out)
    write_input(input_path, hidden_dim)
    print(f"fixture: {root}")

    runner_env = os.environ.copy()
    runner_env["LARGERLM_MOE_DECODE_MXFP4_FUSED"] = "1"
    old_cmd = [
        str(args.runner),
        "--layout",
        str(expert_layout),
        "--resident-layout",
        str(resident_layout),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(old_cache),
        "--run-decoder-layers",
        "--layers",
        ",".join(str(x) for x in layers),
        "--dense-layers",
        ",".join(str(x) for x in dense_layers),
        "--input-f32",
        str(input_path),
        "--position",
        str(args.position),
        "--context-length",
        "1",
        "--num-heads",
        str(num_heads),
        "--kv-lora-dim",
        str(kv_lora_dim),
        "--qk-nope-dim",
        str(qk_nope_dim),
        "--rope-dim",
        str(kv_rope_dim),
        "--v-head-dim",
        str(v_head_dim),
        "--cache-position-offset",
        str(args.position),
        "--top-k",
        str(args.top_k),
        "--max-k",
        str(args.top_k),
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--work-dir",
        str(old_work),
        "--output-f32",
        str(old_output),
        "--output-report-json",
        str(old_report),
        "--output-topk-json",
        str(old_topk),
        "--final-logits-top-k",
        str(args.top_k),
        "--final-logits-max-chunk-mib",
        "16",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "128",
        "--max-router-mib",
        "8",
        "--max-slot-mib",
        "64",
        "--max-runner-scratch-mib",
        "512",
        "--expert-read-advise-merge-gap-kib",
        "0",
        "--expert-read-advise-align-kib",
        "4",
        "--quiet-inner-layers",
    ]
    if args.rope_interleave:
        old_cmd.append("--rope-interleave")
    if args.include_shared_expert:
        old_cmd.append("--include-shared-expert")
    if not args.new_only:
        old_hidden_cmd = without_options(
            old_cmd,
            {
                "--output-topk-json": 1,
                "--final-logits-top-k": 1,
                "--final-logits-max-chunk-mib": 1,
            },
        )
        run_command(old_hidden_cmd, env=runner_env, echo=not args.quiet_commands)
        run_command(old_cmd, env=runner_env, echo=not args.quiet_commands)

    new_cmd = [
        str(args.binary),
        "--prepared",
        str(args.prepared),
        "--probe-decode-layers",
        "--decode-layers",
        ",".join(str(x) for x in layers),
        "--input-f32",
        str(input_path),
        "--output-f32",
        str(new_output),
        "--output-dir",
        str(new_work),
        "--output-topk-json",
        str(new_topk),
        "--output-token-json",
        str(new_token),
        "--output-next-input-f32",
        str(new_next_input),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(new_cache),
        "--position",
        str(args.position),
        "--context-length",
        "1",
        "--num-heads",
        str(num_heads),
        "--kv-lora-dim",
        str(kv_lora_dim),
        "--qk-nope-dim",
        str(qk_nope_dim),
        "--rope-dim",
        str(kv_rope_dim),
        "--v-head-dim",
        str(v_head_dim),
        "--cache-position-offset",
        str(args.position),
        "--top-k",
        str(args.top_k),
        "--expert-buffer-count",
        str(args.top_k + (1 if args.include_shared_expert else 0)),
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-chunk-mib",
        "16",
        "--max-embedding-row-mib",
        "1",
        "--max-live-working-set-mib",
        str(args.max_live_working_set_mib),
        "--skip-debug-intermediates",
        "--json",
    ]
    if args.rope_interleave:
        new_cmd.append("--rope-interleave")
    if args.include_shared_expert:
        new_cmd.append("--include-shared-expert")
    if args.wrap_resident_metal:
        new_cmd.append("--wrap-resident-metal")
    if args.mmap_final_logits:
        new_cmd.append("--mmap-final-logits")
    completed = run_command(new_cmd, echo=not args.quiet_commands)
    payload = json.loads(completed.stdout)
    decode = payload.get("probe_decode_layers") or {}
    if not payload.get("ok") or not decode.get("ok"):
        raise SystemExit(f"glm_moe_infer decode-layers probe did not report ok: {payload}")
    if args.wrap_resident_metal and not payload.get("resident_metal_wrapped"):
        raise SystemExit("decode-layers did not wrap the resident mmap as a Metal buffer")
    expected_expert_buffers = args.top_k + (1 if args.include_shared_expert else 0)
    if payload.get("expert_buffer_count") != expected_expert_buffers:
        raise SystemExit("mixed decode-layers probe activated unexpected expert buffer count")
    if decode.get("dense_layer_count") != len(dense_layers):
        raise SystemExit("dense layer count mismatch in decode-layers JSON")
    layer_payloads = decode.get("layers") or []
    if len(layer_payloads) != len(layers):
        raise SystemExit("decode-layers JSON layer count mismatch")
    if layer_payloads[0].get("input_from_memory"):
        raise SystemExit("first decode layer should still use the fixture input path")
    if any(
        not (item.get("input_from_memory") or item.get("input_buffer_direct"))
        for item in layer_payloads[1:]
    ):
        raise SystemExit("decode-layers buffer chain did not feed every later layer")
    if any(not item.get("attn_projection_fused_pre_cache") for item in layer_payloads):
        raise SystemExit("decode-layers attention projection did not use fused pre-cache path")
    if any(item.get("attn_projection_command_buffer_count") != 1 for item in layer_payloads):
        raise SystemExit("decode-layers attention projection should use one pre-cache command buffer")
    attn_projection_wait_sum = sum(
        int(item.get("attn_projection_synchronous_wait_count") or 0)
        for item in layer_payloads
    )
    attn_projection_async_sum = sum(
        1 for item in layer_payloads if item.get("attn_projection_async_submitted")
    )
    if int(decode.get("attn_projection_synchronous_wait_count") or 0) != attn_projection_wait_sum:
        raise SystemExit("decode-layers aggregate attention-projection wait mismatch")
    if int(decode.get("attn_projection_async_submitted_count") or 0) != attn_projection_async_sum:
        raise SystemExit("decode-layers aggregate attention-projection async mismatch")
    expected_memory_inputs = sum(
        1 for item in layer_payloads if item.get("input_from_memory")
    )
    expected_direct_inputs = sum(
        1 for item in layer_payloads if item.get("input_buffer_direct")
    )
    if decode.get("memory_input_layer_count") != expected_memory_inputs:
        raise SystemExit("decode-layers memory input count mismatch")
    if decode.get("layer_input_buffer_direct_count") != expected_direct_inputs:
        raise SystemExit("decode-layers direct-buffer input count mismatch")
    if not decode.get("hot_intermediate_memory_enabled"):
        raise SystemExit("decode-layers hot intermediate memory bridge is disabled")
    if decode.get("hot_intermediate_memory_bytes", 0) <= 0:
        raise SystemExit("decode-layers hot intermediate memory bytes were not reported")
    if any(item.get("hot_intermediate_tensors") != 5 for item in layer_payloads):
        raise SystemExit("decode-layers hot intermediate tensor count mismatch")
    if decode.get("debug_intermediates_written"):
        raise SystemExit("decode-layers should skip debug intermediate files")
    debug_files = [p for p in new_work.rglob("*") if p.is_file()]
    if debug_files:
        raise SystemExit(f"decode-layers wrote unexpected debug files: {debug_files[:3]}")
    if any(not item.get("attn_output_fused_matvec_add") for item in layer_payloads):
        raise SystemExit("decode-layers attention output did not use fused matvec+add")
    if any(item.get("attn_output_command_buffer_count") != 1 for item in layer_payloads):
        raise SystemExit("decode-layers attention output should use one command buffer")
    attn_output_bytes_sum = sum(
        int(item.get("attn_output_bytes_read") or 0) for item in layer_payloads
    )
    if args.wrap_resident_metal:
        moe_attn_output_bytes_sum = sum(
            int(item.get("attn_output_bytes_read") or 0)
            for item in layer_payloads
            if item.get("kind") == "moe"
        )
        if moe_attn_output_bytes_sum != 0:
            raise SystemExit(
                "resident Metal path still reported MoE attention-output resident reads"
            )
    elif attn_output_bytes_sum <= 0:
        raise SystemExit("decode-layers did not report attention-output resident bytes")
    if int(decode.get("attn_output_bytes_read") or 0) != attn_output_bytes_sum:
        raise SystemExit("decode-layers aggregate attn_output_bytes_read mismatch")
    for key in (
        "attn_output_read_seconds",
        "attn_output_projection_kernel_seconds",
    ):
        expected = sum(float(item.get(key) or 0.0) for item in layer_payloads)
        if abs(float(decode.get(key) or 0.0) - expected) > 1e-9:
            raise SystemExit(f"decode-layers aggregate {key} mismatch")
    if any(not item.get("rope_mla_fused") for item in layer_payloads):
        raise SystemExit("decode-layers RoPE+MLA path did not use fused command buffer")
    for item in layer_payloads:
        if item.get("rope_mla_command_buffer_count") not in (0, 1):
            raise SystemExit("decode-layers RoPE+MLA command buffer count mismatch")
    dense_payloads = [item for item in layer_payloads if item.get("kind") == "dense"]
    if any(not item.get("dense_mlp_fused_pipeline") for item in dense_payloads):
        raise SystemExit("decode-layers dense MLP did not use fused pipeline")
    if any(item.get("dense_mlp_command_buffer_count") != 1 for item in dense_payloads):
        raise SystemExit("decode-layers dense MLP should use one command buffer")
    expected_dense_wait_sum = 0
    expected_dense_async_sum = 0
    for index, item in enumerate(layer_payloads):
        if item.get("kind") != "dense":
            continue
        wait_count = int(item.get("dense_mlp_synchronous_wait_count") or 0)
        async_submitted = bool(item.get("dense_mlp_async_submitted"))
        if index + 1 < len(layer_payloads):
            if wait_count != 0 or not async_submitted:
                raise SystemExit("non-final dense MLP did not defer its command wait")
            expected_dense_async_sum += 1
        else:
            if wait_count != 1 or async_submitted:
                raise SystemExit("final dense MLP should still wait for output readback")
        expected_dense_wait_sum += wait_count
    if int(decode.get("dense_mlp_synchronous_wait_count") or 0) != expected_dense_wait_sum:
        raise SystemExit("decode-layers aggregate dense MLP wait count mismatch")
    if int(decode.get("dense_mlp_async_submitted_count") or 0) != expected_dense_async_sum:
        raise SystemExit("decode-layers aggregate dense MLP async count mismatch")
    moe_payloads = [item for item in layer_payloads if item.get("kind") == "moe"]
    if any(item.get("router_topk_backend") != "metal" for item in moe_payloads):
        raise SystemExit("decode-layers MoE router top-k did not use Metal backend")
    if args.include_shared_expert:
        if any(not item.get("shared_prefetch_used") for item in moe_payloads):
            raise SystemExit("decode-layers MoE shared expert was not prefetched")
        shared_bytes_sum = sum(
            int(item.get("shared_bytes_read") or 0) for item in moe_payloads
        )
        shared_read_seconds_sum = sum(
            float(item.get("shared_read_seconds") or 0.0) for item in moe_payloads
        )
        shared_prefetch_seconds_sum = sum(
            float(item.get("shared_prefetch_seconds") or 0.0)
            for item in moe_payloads
        )
        if shared_bytes_sum <= 0:
            raise SystemExit("decode-layers did not report shared expert bytes")
        if shared_read_seconds_sum <= 0.0 or shared_prefetch_seconds_sum <= 0.0:
            raise SystemExit("decode-layers did not report shared expert prefetch time")
        if int(decode.get("shared_bytes_read") or 0) != shared_bytes_sum:
            raise SystemExit("decode-layers aggregate shared_bytes_read mismatch")
        if int(decode.get("shared_prefetch_used_count") or 0) != len(moe_payloads):
            raise SystemExit("decode-layers aggregate shared prefetch count mismatch")
        if abs(float(decode.get("shared_read_seconds") or 0.0) - shared_read_seconds_sum) > 1e-9:
            raise SystemExit("decode-layers aggregate shared_read_seconds mismatch")
        if abs(float(decode.get("shared_prefetch_seconds") or 0.0) - shared_prefetch_seconds_sum) > 1e-9:
            raise SystemExit("decode-layers aggregate shared_prefetch_seconds mismatch")
    for key in (
        "post_attn_norm_weight_bytes_read",
        "router_bytes_read",
        "router_correction_bias_bytes_read",
    ):
        expected = sum(int(item.get(key) or 0) for item in moe_payloads)
        if args.wrap_resident_metal and key != "post_attn_norm_weight_bytes_read":
            if expected != 0:
                raise SystemExit(f"resident Metal path still reported {key}")
        elif expected <= 0:
            raise SystemExit(f"decode-layers did not report {key}")
        if int(decode.get(key) or 0) != expected:
            raise SystemExit(f"decode-layers aggregate {key} mismatch")
    for key in (
        "post_attn_norm_weight_read_seconds",
        "router_read_seconds",
    ):
        expected = sum(float(item.get(key) or 0.0) for item in moe_payloads)
        if abs(float(decode.get(key) or 0.0) - expected) > 1e-9:
            raise SystemExit(f"decode-layers aggregate {key} mismatch")
    if any(not item.get("moe_mlp_residual_add_fused") for item in moe_payloads):
        raise SystemExit("decode-layers MoE MLP residual add did not fuse into GPU command")
    expected_moe_cmds = 1
    if any(item.get("moe_mlp_command_buffer_count") != expected_moe_cmds for item in moe_payloads):
        raise SystemExit("decode-layers MoE MLP command buffer count mismatch")
    expected_moe_wait_sum = 0
    for index, item in enumerate(layer_payloads):
        if item.get("kind") != "moe":
            continue
        wait_count = int(item.get("moe_mlp_synchronous_wait_count") or 0)
        if index + 1 < len(layer_payloads):
            if wait_count != 0:
                raise SystemExit("non-final MoE layer did not defer its command wait")
        else:
            if wait_count != expected_moe_cmds:
                raise SystemExit("final MoE layer should still wait for output readback")
        expected_moe_wait_sum += wait_count
    if int(decode.get("moe_mlp_synchronous_wait_count") or 0) != expected_moe_wait_sum:
        raise SystemExit("decode-layers aggregate MoE wait count mismatch")
    for item in moe_payloads:
        assert_layer_moe_read_telemetry(
            item,
            route_count=args.top_k,
            active_expert_buffers=expected_expert_buffers,
        )
    aggregate_checks = {
        "expert_read_dispatch_count": sum(
            int(item.get("expert_read_dispatch_count") or 0)
            for item in moe_payloads
        ),
        "expert_read_task_count": sum(
            int(item.get("expert_read_task_count") or 0) for item in moe_payloads
        ),
        "expert_read_pool_dispatch_count": sum(
            int(item.get("expert_read_pool_dispatch_count") or 0)
            for item in moe_payloads
        ),
        "expert_read_serial_dispatch_count": sum(
            int(item.get("expert_read_serial_dispatch_count") or 0)
            for item in moe_payloads
        ),
        "expert_read_max_task_count": max(
            (int(item.get("expert_read_max_task_count") or 0) for item in moe_payloads),
            default=0,
        ),
        "expert_read_max_worker_count": max(
            (
                int(item.get("expert_read_max_worker_count") or 0)
                for item in moe_payloads
            ),
            default=0,
        ),
    }
    for key, expected in aggregate_checks.items():
        if decode.get(key) != expected:
            raise SystemExit(
                f"decode-layers aggregate {key} mismatch: "
                f"expected {expected}, got {decode.get(key)}"
            )
    expert_read_seconds_sum = sum(
        float(item.get("expert_read_seconds") or 0.0) for item in moe_payloads
    )
    if abs(float(decode.get("expert_read_seconds") or 0.0) - expert_read_seconds_sum) > 1e-9:
        raise SystemExit("decode-layers aggregate expert_read_seconds mismatch")
    moe_kernel_seconds_sum = sum(
        float(item.get("moe_mlp_kernel_seconds") or 0.0) for item in moe_payloads
    )
    if abs(float(decode.get("moe_mlp_kernel_seconds") or 0.0) - moe_kernel_seconds_sum) > 1e-9:
        raise SystemExit("decode-layers aggregate moe_mlp_kernel_seconds mismatch")
    logits_payload = payload.get("probe_final_logits") or {}
    if not logits_payload.get("ok"):
        raise SystemExit("decode-layers final logits probe did not report ok")
    if not logits_payload.get("input_from_memory"):
        raise SystemExit("decode-layers final logits should use in-memory hidden input")
    if logits_payload.get("source") != "decode_layers_memory_output":
        raise SystemExit("decode-layers final logits source should be memory output")
    if args.wrap_resident_metal or args.mmap_final_logits:
        if not logits_payload.get("resident_mmap_backed"):
            raise SystemExit("decode-layers final logits did not use resident Metal weights")
        if int(logits_payload.get("lm_head_bytes_read") or 0) != 0:
            raise SystemExit("resident Metal final logits still reported lm_head reads")
    logits_topk = logits_payload.get("topk") or []
    generated_token = logits_payload.get("generated_token") or {}
    if not logits_topk:
        raise SystemExit("decode-layers final logits top-k is empty")
    if generated_token.get("selection") != "argmax":
        raise SystemExit("decode-layers generated token should use argmax selection")
    if generated_token.get("token_id") != logits_topk[0].get("token_id"):
        raise SystemExit("decode-layers generated token does not match top-1")
    if not new_token.exists():
        raise SystemExit("decode-layers did not write generated token JSON")
    token_payload = json.loads(new_token.read_text(encoding="utf-8"))
    if token_payload.get("token_id") != generated_token.get("token_id"):
        raise SystemExit("generated token JSON does not match final-logits payload")
    next_embedding = logits_payload.get("next_input_embedding") or {}
    if not next_embedding.get("ok"):
        raise SystemExit("decode-layers did not report generated-token embedding ok")
    if next_embedding.get("token_id") != generated_token.get("token_id"):
        raise SystemExit("generated-token embedding used the wrong token id")
    if next_embedding.get("hidden_dim") != hidden_dim:
        raise SystemExit("generated-token embedding hidden dim mismatch")
    if next_embedding.get("bytes_read", 0) <= 0:
        raise SystemExit("generated-token embedding did not report bytes read")
    if not new_next_input.exists():
        raise SystemExit("decode-layers did not write generated-token embedding")
    embed_token(
        resident_layout,
        token_id=int(generated_token["token_id"]),
        output_f32_path=expected_next_input,
        max_row_bytes=1024 * 1024,
        expected_hidden_size=hidden_dim,
    )
    next_input_comparison = compare_outputs(expected_next_input, new_next_input)
    if next_input_comparison["max_abs_diff"] != 0.0:
        raise SystemExit(
            "generated-token embedding differs from Python embedding oracle"
        )

    output_comparison = None
    topk_comparison = None
    if not args.new_only:
        output_comparison = compare_outputs(old_output, new_output)
        topk_comparison = compare_topk(old_topk, new_topk, max_diff=1e-4)
    cache_comparison = None
    if not args.new_only and old_cache.read_bytes() != new_cache.read_bytes():
        cache_comparison = compare_f32_files(old_cache, new_cache)
    print(
        json.dumps(
            {
                "output": output_comparison,
                "topk": topk_comparison,
                "cache": cache_comparison,
                "next_input_embedding": next_input_comparison,
                "dims": {
                    "layers": layers,
                    "dense_layers": dense_layers,
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "kv_lora_dim": kv_lora_dim,
                    "qk_nope_dim": qk_nope_dim,
                    "rope_dim": kv_rope_dim,
                    "v_head_dim": v_head_dim,
                    "estimated_live_working_set_bytes": payload[
                        "estimated_live_working_set_bytes"
                    ],
                    "expert_buffer_count": payload["expert_buffer_count"],
                    "scratch_bytes": decode["scratch_bytes"],
                    "memory_chain_bytes": decode["memory_chain_bytes"],
                    "memory_input_layer_count": decode["memory_input_layer_count"],
                    "layer_input_buffer_direct_count": decode[
                        "layer_input_buffer_direct_count"
                    ],
                    "hot_intermediate_memory_bytes": decode[
                        "hot_intermediate_memory_bytes"
                    ],
                    "debug_intermediates_written": decode[
                        "debug_intermediates_written"
                    ],
                    "attn_projection_command_buffers": [
                        item["attn_projection_command_buffer_count"]
                        for item in layer_payloads
                    ],
                    "attn_projection_synchronous_waits": [
                        item.get("attn_projection_synchronous_wait_count", 0)
                        for item in layer_payloads
                    ],
                    "attn_projection_async_submitted": [
                        item.get("attn_projection_async_submitted", False)
                        for item in layer_payloads
                    ],
                    "attn_output_command_buffers": [
                        item["attn_output_command_buffer_count"]
                        for item in layer_payloads
                    ],
                    "rope_mla_command_buffers": [
                        item["rope_mla_command_buffer_count"] for item in layer_payloads
                    ],
                    "dense_mlp_command_buffers": [
                        item.get("dense_mlp_command_buffer_count", 0)
                        for item in dense_payloads
                    ],
                    "dense_mlp_synchronous_waits": [
                        item.get("dense_mlp_synchronous_wait_count", 0)
                        for item in dense_payloads
                    ],
                    "dense_mlp_async_submitted": [
                        item.get("dense_mlp_async_submitted", False)
                        for item in dense_payloads
                    ],
                    "moe_mlp_command_buffers": [
                        item.get("moe_mlp_command_buffer_count", 0)
                        for item in moe_payloads
                    ],
                    "moe_mlp_synchronous_waits": [
                        item.get("moe_mlp_synchronous_wait_count", 0)
                        for item in moe_payloads
                    ],
                    "moe_router_topk_backends": [
                        item.get("router_topk_backend") for item in moe_payloads
                    ],
                    "final_logits_input_from_memory": logits_payload[
                        "input_from_memory"
                    ],
                    "final_logits_source": logits_payload["source"],
                    "final_logits_resident_mmap_backed": logits_payload.get(
                        "resident_mmap_backed"
                    ),
                    "final_logits_bytes_read": logits_payload.get("bytes_read"),
                    "final_logits_lm_head_bytes_read": logits_payload.get(
                        "lm_head_bytes_read"
                    ),
                    "final_logits_read_seconds": logits_payload.get("read_seconds"),
                    "final_logits_kernel_seconds": logits_payload.get(
                        "kernel_seconds"
                    ),
                    "generated_token_id": generated_token["token_id"],
                    "next_input_embedding_bytes_read": next_embedding["bytes_read"],
                    "final_logits_elapsed_seconds": logits_payload[
                        "elapsed_seconds"
                    ],
                    "elapsed_seconds": decode["elapsed_seconds"],
                    "layer_elapsed_seconds": decode["layer_elapsed_seconds"],
                    "attn_projection_elapsed_seconds": decode[
                        "attn_projection_elapsed_seconds"
                    ],
                    "mla_attention_elapsed_seconds": decode[
                        "mla_attention_elapsed_seconds"
                    ],
                    "mla_attention_cache_read_seconds": decode[
                        "mla_attention_cache_read_seconds"
                    ],
                    "mla_attention_value_read_seconds": decode[
                        "mla_attention_value_read_seconds"
                    ],
                    "mla_attention_kernel_seconds": decode[
                        "mla_attention_kernel_seconds"
                    ],
                    "mla_attention_output_write_seconds": decode[
                        "mla_attention_output_write_seconds"
                    ],
                    "attn_output_elapsed_seconds": decode[
                        "attn_output_elapsed_seconds"
                    ],
                    "attn_output_bytes_read": decode["attn_output_bytes_read"],
                    "attn_output_read_seconds": decode[
                        "attn_output_read_seconds"
                    ],
                    "attn_output_projection_kernel_seconds": decode[
                        "attn_output_projection_kernel_seconds"
                    ],
                    "post_attn_norm_weight_bytes_read": decode[
                        "post_attn_norm_weight_bytes_read"
                    ],
                    "post_attn_norm_weight_read_seconds": decode[
                        "post_attn_norm_weight_read_seconds"
                    ],
                    "router_bytes_read": decode["router_bytes_read"],
                    "router_correction_bias_bytes_read": decode[
                        "router_correction_bias_bytes_read"
                    ],
                    "router_read_seconds": decode["router_read_seconds"],
                    "router_kernel_seconds": decode["router_kernel_seconds"],
                    "mlp_elapsed_seconds": decode["mlp_elapsed_seconds"],
                    "dense_mlp_elapsed_seconds": decode[
                        "dense_mlp_elapsed_seconds"
                    ],
                    "moe_mlp_elapsed_seconds": decode[
                        "moe_mlp_elapsed_seconds"
                    ],
                    "max_diff_threshold": (
                        args.all_layers_max_diff if args.all_layers else args.max_diff
                    ),
                    "expert_bytes_read": decode["expert_bytes_read"],
                    "dense_mlp_bytes_read": decode["dense_mlp_bytes_read"],
                    "expert_read_seconds": decode["expert_read_seconds"],
                    "moe_mlp_kernel_seconds": decode["moe_mlp_kernel_seconds"],
                    "expert_read_dispatch_count": decode[
                        "expert_read_dispatch_count"
                    ],
                    "expert_read_task_count": decode["expert_read_task_count"],
                    "expert_read_max_task_count": decode[
                        "expert_read_max_task_count"
                    ],
                    "expert_read_max_worker_count": decode[
                        "expert_read_max_worker_count"
                    ],
                    "expert_read_pool_dispatch_count": decode[
                        "expert_read_pool_dispatch_count"
                    ],
                    "expert_read_serial_dispatch_count": decode[
                        "expert_read_serial_dispatch_count"
                    ],
                    "new_only": args.new_only,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    effective_max_diff = args.all_layers_max_diff if args.all_layers else args.max_diff
    if output_comparison and output_comparison["max_abs_diff"] > effective_max_diff:
        raise SystemExit(
            f"max_abs_diff {output_comparison['max_abs_diff']:.9g} exceeds {effective_max_diff:.9g}"
        )
    if cache_comparison and cache_comparison["max_abs_diff"] > effective_max_diff:
        raise SystemExit(
            f"cache max_abs_diff {cache_comparison['max_abs_diff']:.9g} exceeds {effective_max_diff:.9g}"
        )
    print("  real decode layers smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
