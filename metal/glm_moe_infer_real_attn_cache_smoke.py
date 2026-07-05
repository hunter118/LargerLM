#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

from glm_moe_infer_real_attn_projections_smoke import (
    FILES,
    compare_file,
    mxfp4_dims,
    tensor_map,
    vector_dim,
    write_dummy_expert_layout,
    write_input,
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


def write_cache_fixture(
    root: Path,
    layer: int,
    width: int,
    max_context_tokens: int,
) -> tuple[Path, Path, Path]:
    token_bytes = width * 2
    total_bytes = token_bytes * max_context_tokens
    layout_path = root / "cache_layout.json"
    old_cache = root / "old_decode_cache.bin"
    new_cache = root / "new_decode_cache.bin"
    layout_path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": max_context_tokens,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": total_bytes,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": layer,
                        "offset": 0,
                        "width": width,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": token_bytes,
                        "max_context_tokens": max_context_tokens,
                        "total_bytes": total_bytes,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    old_cache.write_bytes(b"\0" * total_bytes)
    new_cache.write_bytes(b"\0" * total_bytes)
    return layout_path, old_cache, new_cache


def bf16_to_f32_tuple(raw: bytes) -> tuple[float, ...]:
    if len(raw) % 2 != 0:
        raise SystemExit("BF16 payload is not 2-byte aligned")
    values: list[float] = []
    for (word,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", (word << 16).to_bytes(4, "little"))[0])
    return tuple(values)


def compare_cache(old_cache: Path, new_cache: Path, position: int, width: int) -> dict[str, object]:
    old = old_cache.read_bytes()
    new = new_cache.read_bytes()
    if old != new:
        for index, (a, b) in enumerate(zip(old, new)):
            if a != b:
                raise SystemExit(f"cache byte mismatch at {index}: {a} != {b}")
        raise SystemExit(f"cache length mismatch: {len(old)} != {len(new)}")
    token_bytes = width * 2
    offset = position * token_bytes
    segment = old[offset : offset + token_bytes]
    if not any(segment):
        raise SystemExit("cache write segment is still all zero")
    values = bf16_to_f32_tuple(segment[: min(len(segment), 16)])
    return {
        "bytes": len(old),
        "token_bytes": token_bytes,
        "position": position,
        "offset": offset,
        "first_values": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer attention projection KV-cache append against largerlm-runner."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--position", type=int, default=2)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
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
    if q_b_in != q_a_out or kv_a_in != hidden_dim or kv_lora_dim >= kv_a_out:
        raise SystemExit("real GLM attention projection dims are inconsistent")

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-attn-cache-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    dummy_expert_layout = write_dummy_expert_layout(root)
    input_path = root / "input.f32"
    old_dir = root / "old"
    new_dir = root / "new"
    write_input(input_path, hidden_dim)
    layout_path, old_cache, new_cache = write_cache_fixture(
        root,
        args.layer,
        kv_a_out,
        max_context_tokens=4,
    )
    print(f"fixture: {root}")

    common = [
        "--resident-layout",
        str(resident_layout),
        "--layer",
        str(args.layer),
        "--input-f32",
        str(input_path),
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--cache-layout",
        str(layout_path),
        "--position",
        str(args.position),
        "--max-cache-file-mib",
        "1",
    ]
    run_command(
        [
            str(args.runner),
            *common,
            "--run-attn-projections",
            "--output-dir",
            str(old_dir),
            "--cache-file",
            str(old_cache),
            "--max-resident-matrix-mib",
            "64",
            "--max-runner-scratch-mib",
            "256",
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
            "--probe-attn-projections",
            "--probe-layer",
            str(args.layer),
            "--input-f32",
            str(input_path),
            "--rms-norm-eps",
            f"{args.rms_norm_eps:.9g}",
            "--output-dir",
            str(new_dir),
            "--cache-layout",
            str(layout_path),
            "--cache-file",
            str(new_cache),
            "--position",
            str(args.position),
            "--max-cache-file-mib",
            "1",
            "--max-live-working-set-mib",
            "64",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    attn = payload.get("probe_attn_projections") or {}
    if not payload.get("ok") or not attn.get("ok"):
        raise SystemExit(f"glm_moe_infer attention cache probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("attention cache probe should not allocate expert buffers")
    if not attn.get("cache_append"):
        raise SystemExit(f"glm_moe_infer did not report cache append: {attn}")
    expected_token_bytes = kv_a_out * 2
    if attn.get("cache_write_bytes") != expected_token_bytes:
        raise SystemExit(f"unexpected cache write bytes: {attn}")

    comparisons = [compare_file(old_dir / name, new_dir / name) for name in FILES]
    cache = compare_cache(old_cache, new_cache, args.position, kv_a_out)
    print(
        json.dumps(
            {
                "comparison": comparisons,
                "cache": cache,
                "dims": {
                    "hidden_dim": hidden_dim,
                    "q_a_out": q_a_out,
                    "q_b_out": q_b_out,
                    "kv_a_out": kv_a_out,
                    "kv_lora_dim": kv_lora_dim,
                    "estimated_live_working_set_bytes": payload["estimated_live_working_set_bytes"],
                    "cache_write_seconds": attn["cache_write_seconds"],
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
    print("  real attention cache append smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
