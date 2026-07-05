#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

from glm_moe_infer_real_attn_projections_smoke import (
    mxfp4_dims,
    tensor_map,
    vector_dim,
    write_dummy_expert_layout,
    write_input,
)
from glm_moe_infer_real_rope_split_smoke import slice_k_rope


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise SystemExit(f"{path} is not f32-aligned")
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def write_f32(path: Path, values: tuple[float, ...]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def write_f32_cache(root: Path, layer: int, kv_a: tuple[float, ...]) -> tuple[Path, Path]:
    width = len(kv_a)
    cache_layout = root / "cache_layout.json"
    cache_file = root / "decode_cache.bin"
    cache_file.write_bytes(struct.pack(f"<{width}f", *kv_a))
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
    return cache_layout, cache_file


def compare_output(old_path: Path, new_path: Path) -> dict[str, float | int]:
    old = read_f32(old_path)
    new = read_f32(new_path)
    if len(old) != len(new):
        raise SystemExit(f"MLA output length mismatch: {len(old)} != {len(new)}")
    diffs = [abs(a - b) for a, b in zip(old, new)]
    max_index = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    return {
        "count": len(diffs),
        "max_abs_diff": diffs[max_index] if diffs else 0.0,
        "max_index": max_index,
        "old0": old[0] if old else 0.0,
        "new0": new[0] if new else 0.0,
        "old_at_max": old[max_index] if diffs else 0.0,
        "new_at_max": new[max_index] if diffs else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer single-token MLA attention against largerlm-runner on a real GLM layer."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--position", type=int, default=19)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-interleave", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    resident_layout = args.prepared / "resident" / "layout.json"
    if not resident_layout.exists():
        print(f"prepared resident layout missing, skipping: {resident_layout}")
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
        raise SystemExit("real GLM MLA dims are inconsistent")

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-mla-attn-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    dummy_expert_layout = write_dummy_expert_layout(root)
    input_path = root / "input.f32"
    attn_dir = root / "attn"
    rope_dir = root / "rope"
    rope_dir.mkdir()
    write_input(input_path, hidden_dim)
    print(f"fixture: {root}")

    attn_completed = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(resident_layout),
            "--expert-layout",
            str(dummy_expert_layout),
            "--no-open-experts",
            "--probe-attn-projections",
            "--probe-layer",
            str(args.layer),
            "--input-f32",
            str(input_path),
            "--rms-norm-eps",
            f"{args.rms_norm_eps:.9g}",
            "--output-dir",
            str(attn_dir),
            "--max-live-working-set-mib",
            "64",
            "--json",
        ]
    )
    attn_payload = json.loads(attn_completed.stdout)
    if not attn_payload.get("ok"):
        raise SystemExit("attention projection precursor failed")

    kv_a = read_f32(attn_dir / "attn_kv_a.f32")
    cache_layout, cache_file = write_f32_cache(root, args.layer, kv_a)
    k_rope = root / "k_rope.f32"
    slice_k_rope(attn_dir / "attn_kv_a.f32", k_rope, kv_lora_dim, kv_rope_dim)

    rope_common = [
        "--q-b-f32",
        str(attn_dir / "attn_q_b.f32"),
        "--k-f32",
        str(k_rope),
        "--num-heads",
        str(num_heads),
        "--qk-nope-dim",
        str(qk_nope_dim),
        "--rope-dim",
        str(kv_rope_dim),
        "--start-position",
        str(args.position),
        "--batch-tokens",
        "1",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
    ]
    if args.rope_interleave:
        rope_common.append("--rope-interleave")

    run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(resident_layout),
            "--expert-layout",
            str(dummy_expert_layout),
            "--no-open-experts",
            "--probe-rope-split",
            *rope_common,
            "--output-q-nope-f32",
            str(rope_dir / "q_nope.f32"),
            "--output-q-rope-f32",
            str(rope_dir / "q_rope.f32"),
            "--output-q-f32",
            str(rope_dir / "q_rope_rotated.f32"),
            "--output-k-f32",
            str(rope_dir / "k_rope_rotated.f32"),
            "--max-live-working-set-mib",
            "8",
            "--json",
        ]
    )

    old_out = root / "old_mla.f32"
    new_out = root / "new_mla.f32"
    shared_mla = [
        "--resident-layout",
        str(resident_layout),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(cache_file),
        "--q-nope-f32",
        str(rope_dir / "q_nope.f32"),
        "--q-rope-f32",
        str(rope_dir / "q_rope_rotated.f32"),
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
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
    ]
    if args.rope_interleave:
        shared_mla.append("--rope-interleave")

    run_command(
        [
            str(args.runner),
            *shared_mla,
            "--layer",
            str(args.layer),
            "--run-mla-attention",
            "--output-f32",
            str(old_out),
            "--max-resident-matrix-mib",
            "128",
            "--max-runner-scratch-mib",
            "256",
        ]
    )

    completed = run_command(
        [
            str(args.binary),
            "--expert-layout",
            str(dummy_expert_layout),
            "--no-open-experts",
            "--probe-mla-attention",
            "--probe-layer",
            str(args.layer),
            *shared_mla,
            "--output-f32",
            str(new_out),
            "--max-live-working-set-mib",
            "192",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    mla = payload.get("probe_mla_attention") or {}
    if not payload.get("ok") or not mla.get("ok"):
        raise SystemExit(f"glm_moe_infer MLA attention probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("MLA attention probe should not allocate expert buffers")

    comparison = compare_output(old_out, new_out)
    print(
        json.dumps(
            {
                "comparison": comparison,
                "dims": {
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "kv_lora_dim": kv_lora_dim,
                    "qk_nope_dim": qk_nope_dim,
                    "rope_dim": kv_rope_dim,
                    "v_head_dim": v_head_dim,
                    "estimated_live_working_set_bytes": payload["estimated_live_working_set_bytes"],
                    "scratch_bytes": mla["scratch_bytes"],
                    "kernel_seconds": mla["kernel_seconds"],
                    "value_read_seconds": mla["value_read_seconds"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if float(comparison["max_abs_diff"]) > args.max_diff:
        raise SystemExit(
            f"MLA max_abs_diff {comparison['max_abs_diff']:.9g} "
            f"exceeds {args.max_diff:.9g}"
        )
    print("  real MLA attention smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
