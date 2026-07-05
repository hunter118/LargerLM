#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

from glm_moe_infer_real_attn_projections_smoke import (
    mxfp4_dims,
    tensor_map,
    vector_dim,
    write_input,
)
from glm_moe_infer_real_mlp_block_smoke import compare_outputs


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def write_f32_cache(root: Path, layer: int, width: int) -> tuple[Path, Path, Path]:
    cache_layout = root / "cache_layout.json"
    old_cache = root / "old_decode_cache.bin"
    new_cache = root / "new_decode_cache.bin"
    raw = b"\0" * (width * 4)
    old_cache.write_bytes(raw)
    new_cache.write_bytes(raw)
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 1,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 64,
                "total_bytes": width * 4,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": layer,
                        "offset": 0,
                        "width": width,
                        "dtype": "F32",
                        "dtype_bytes": 4,
                        "token_stride_bytes": width * 4,
                        "max_context_tokens": 1,
                        "total_bytes": width * 4,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_layout, old_cache, new_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer dense decoder-layer probe against largerlm-runner on one real GLM dense-prefix layer."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-interleave", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-5)
    args = parser.parse_args()

    resident_layout = args.prepared / "resident" / "layout.json"
    expert_layout = args.prepared / "experts" / "layout.json"
    if not resident_layout.exists() or not expert_layout.exists():
        print(f"prepared package missing, skipping: {args.prepared}")
        return 0

    tensors = tensor_map(resident_layout)
    prefix = f"model.layers.{args.layer}"
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
        raise SystemExit("real GLM dense decoder dims are inconsistent")

    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-glm-dense-decoder-real-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.f32"
    old_output = root / "old_dense_decoder_out.f32"
    new_output = root / "new_dense_decoder_out.f32"
    old_work = root / "old_work"
    new_work = root / "new_work"
    cache_layout, old_cache, new_cache = write_f32_cache(root, args.layer, kv_a_out)
    write_input(input_path, hidden_dim)
    print(f"fixture: {root}")

    old_cmd = [
        str(args.runner),
        "--resident-layout",
        str(resident_layout),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(old_cache),
        "--layer",
        str(args.layer),
        "--run-dense-decoder-layer",
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
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--work-dir",
        str(old_work),
        "--output-f32",
        str(old_output),
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "128",
        "--max-runner-scratch-mib",
        "512",
    ]
    if args.rope_interleave:
        old_cmd.append("--rope-interleave")
    run_command(old_cmd)

    new_cmd = [
        str(args.binary),
        "--prepared",
        str(args.prepared),
        "--probe-dense-decoder-layer",
        "--probe-layer",
        str(args.layer),
        "--input-f32",
        str(input_path),
        "--output-f32",
        str(new_output),
        "--output-dir",
        str(new_work),
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
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-live-working-set-mib",
        "512",
        "--json",
    ]
    if args.rope_interleave:
        new_cmd.append("--rope-interleave")
    completed = run_command(new_cmd)
    payload = json.loads(completed.stdout)
    decoder = payload.get("probe_dense_decoder_layer") or {}
    if not payload.get("ok") or not decoder.get("ok"):
        raise SystemExit(f"glm_moe_infer dense decoder-layer probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0 or payload.get("expert_files_opened") != 0:
        raise SystemExit(
            "dense decoder probe should not open expert files or allocate expert buffers"
        )

    output_comparison = compare_outputs(old_output, new_output)
    print(
        json.dumps(
            {
                "output": output_comparison,
                "dims": {
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "kv_lora_dim": kv_lora_dim,
                    "qk_nope_dim": qk_nope_dim,
                    "rope_dim": kv_rope_dim,
                    "v_head_dim": v_head_dim,
                    "estimated_live_working_set_bytes": payload[
                        "estimated_live_working_set_bytes"
                    ],
                    "scratch_bytes": decoder["scratch_bytes"],
                    "elapsed_seconds": decoder["elapsed_seconds"],
                    "mla_kernel_seconds": decoder["mla_attention_kernel_seconds"],
                    "attn_output_projection_kernel_seconds": decoder[
                        "attn_output_projection_kernel_seconds"
                    ],
                    "dense_mlp_elapsed_seconds": decoder["dense_mlp_elapsed_seconds"],
                    "dense_mlp_bytes_read": decoder["dense_mlp_bytes_read"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if output_comparison["max_abs_diff"] > args.max_diff:
        raise SystemExit(
            f"max_abs_diff {output_comparison['max_abs_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    if old_cache.read_bytes() != new_cache.read_bytes():
        raise SystemExit("dense decoder cache writes differ")
    print("  real dense decoder layer smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
