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


FILES = (
    "q_nope.f32",
    "q_rope.f32",
    "q_rope_rotated.f32",
    "k_rope_rotated.f32",
)


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


def slice_k_rope(kv_a_path: Path, output_path: Path, kv_lora_dim: int, kv_rope_dim: int) -> None:
    values = read_f32(kv_a_path)
    expected = kv_lora_dim + kv_rope_dim
    if len(values) != expected:
        raise SystemExit(f"unexpected kv_a length {len(values)} != {expected}")
    write_f32(output_path, values[kv_lora_dim:])


def compare_file(old_path: Path, new_path: Path) -> dict[str, float | int | str]:
    old = read_f32(old_path)
    new = read_f32(new_path)
    if len(old) != len(new):
        raise SystemExit(f"{old_path.name} length mismatch: {len(old)} != {len(new)}")
    diffs = [abs(a - b) for a, b in zip(old, new)]
    max_index = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    return {
        "file": old_path.name,
        "count": len(diffs),
        "max_abs_diff": diffs[max_index] if diffs else 0.0,
        "max_index": max_index,
        "old_at_max": old[max_index] if diffs else 0.0,
        "new_at_max": new[max_index] if diffs else 0.0,
        "old0": old[0] if old else 0.0,
        "new0": new[0] if new else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer RoPE split against largerlm-runner on one real GLM layer."
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
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--qk-nope-dim", type=int, default=192)
    parser.add_argument("--rope-dim", type=int, default=64)
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
    kv_a_out, _, _ = mxfp4_dims(tensors, f"{prefix}.self_attn.kv_a_proj_with_mqa.weight")
    kv_lora_dim = vector_dim(tensors, f"{prefix}.self_attn.kv_a_layernorm.weight")
    kv_rope_dim = kv_a_out - kv_lora_dim
    if q_b_in != q_a_out:
        raise SystemExit("q_b input dim does not match q_a output dim")
    expected_q_b = args.num_heads * (args.qk_nope_dim + args.rope_dim)
    if q_b_out != expected_q_b:
        raise SystemExit(f"q_b out dim {q_b_out} != expected {expected_q_b}")
    if kv_rope_dim != args.rope_dim:
        raise SystemExit(f"kv rope dim {kv_rope_dim} != rope dim {args.rope_dim}")

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-rope-split-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    dummy_expert_layout = write_dummy_expert_layout(root)
    input_path = root / "input.f32"
    attn_dir = root / "attn"
    old_rope = root / "old_rope"
    new_rope = root / "new_rope"
    old_rope.mkdir()
    new_rope.mkdir()
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

    k_rope = root / "k_rope.f32"
    slice_k_rope(attn_dir / "attn_kv_a.f32", k_rope, kv_lora_dim, kv_rope_dim)

    common = [
        "--q-b-f32",
        str(attn_dir / "attn_q_b.f32"),
        "--k-f32",
        str(k_rope),
        "--num-heads",
        str(args.num_heads),
        "--qk-nope-dim",
        str(args.qk_nope_dim),
        "--rope-dim",
        str(args.rope_dim),
        "--start-position",
        str(args.position),
        "--batch-tokens",
        "1",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
    ]
    if args.rope_interleave:
        common.append("--rope-interleave")

    run_command(
        [
            str(args.runner),
            "--run-rope-split-batch",
            *common,
            "--output-q-nope-f32",
            str(old_rope / "q_nope.f32"),
            "--output-q-rope-f32",
            str(old_rope / "q_rope.f32"),
            "--output-q-f32",
            str(old_rope / "q_rope_rotated.f32"),
            "--output-k-f32",
            str(old_rope / "k_rope_rotated.f32"),
            "--max-runner-scratch-mib",
            "64",
        ]
    )

    completed = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(resident_layout),
            "--expert-layout",
            str(dummy_expert_layout),
            "--no-open-experts",
            "--probe-rope-split",
            *common,
            "--output-q-nope-f32",
            str(new_rope / "q_nope.f32"),
            "--output-q-rope-f32",
            str(new_rope / "q_rope.f32"),
            "--output-q-f32",
            str(new_rope / "q_rope_rotated.f32"),
            "--output-k-f32",
            str(new_rope / "k_rope_rotated.f32"),
            "--max-live-working-set-mib",
            "8",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    rope = payload.get("probe_rope_split") or {}
    if not payload.get("ok") or not rope.get("ok"):
        raise SystemExit(f"glm_moe_infer RoPE split probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("RoPE-only probe should not allocate expert buffers")
    if rope.get("q_rope_bytes") != args.num_heads * args.rope_dim * 4:
        raise SystemExit(f"unexpected q_rope bytes: {rope}")

    comparisons = [compare_file(old_rope / name, new_rope / name) for name in FILES]
    print(
        json.dumps(
            {
                "comparison": comparisons,
                "dims": {
                    "hidden_dim": hidden_dim,
                    "q_b_out": q_b_out,
                    "kv_lora_dim": kv_lora_dim,
                    "kv_rope_dim": kv_rope_dim,
                    "estimated_live_working_set_bytes": payload["estimated_live_working_set_bytes"],
                    "scratch_bytes": rope["scratch_bytes"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    worst = max(comparisons, key=lambda item: float(item["max_abs_diff"]))
    if float(worst["max_abs_diff"]) > args.max_diff:
        raise SystemExit(
            f"{worst['file']} max_abs_diff {worst['max_abs_diff']:.9g} "
            f"exceeds {args.max_diff:.9g}"
        )
    print("  real RoPE split smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
