#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 4
Q_LORA_DIM = 2
Q_OUT_DIM = 3
KV_LORA_DIM = 2
KV_ROPE_DIM = 1
KV_A_OUT_DIM = KV_LORA_DIM + KV_ROPE_DIM


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def zeros(count: int) -> bytes:
    return b"\0" * (count * 4)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, shape: list[int], data: bytes | None = None) -> None:
        if data is None:
            count = 1
            for dim in shape:
                count *= dim
            data = zeros(count)
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": "F32",
                "shape": shape,
                "category": "attention",
            }
        )
        payload.extend(data)

    prefix = "model.layers.1"
    add(f"{prefix}.input_layernorm.weight", [HIDDEN_DIM], f32([1.0] * HIDDEN_DIM))
    add(f"{prefix}.self_attn.q_a_layernorm.weight", [Q_LORA_DIM], f32([1.0] * Q_LORA_DIM))
    add(f"{prefix}.self_attn.kv_a_layernorm.weight", [KV_LORA_DIM], f32([1.0] * KV_LORA_DIM))
    add(f"{prefix}.self_attn.q_a_proj.weight", [Q_LORA_DIM, HIDDEN_DIM], f32([0.25] * (Q_LORA_DIM * HIDDEN_DIM)))
    add(f"{prefix}.self_attn.q_b_proj.weight", [Q_OUT_DIM, Q_LORA_DIM], f32([0.5] * (Q_OUT_DIM * Q_LORA_DIM)))
    add(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight", [KV_A_OUT_DIM, HIDDEN_DIM], f32([0.125] * (KV_A_OUT_DIM * HIDDEN_DIM)))
    add(f"{prefix}.self_attn.embed_q.weight", [1, KV_LORA_DIM, 1], f32([1.0, 0.5]))
    add(f"{prefix}.self_attn.unembed_out.weight", [1, 1, KV_LORA_DIM], f32([0.75, 0.25]))

    (resident / "resident.bin").write_bytes(bytes(payload))
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(f32([1.0, 2.0, 3.0, 4.0]))
    cache_bytes = KV_A_OUT_DIM * 4
    (root / "decode_cache.bin").write_bytes(b"\0" * cache_bytes)
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 1,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 64,
                "total_bytes": cache_bytes,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": KV_A_OUT_DIM,
                        "dtype": "F32",
                        "dtype_bytes": 4,
                        "token_stride_bytes": cache_bytes,
                        "max_context_tokens": 1,
                        "total_bytes": cache_bytes,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_checked(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def run_smoke(runner: Path, root: Path) -> None:
    out_dir = root / "attn"
    stdout = run_checked(
        [
            str(runner),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--layer",
            "1",
            "--run-attn-projections",
            "--input-f32",
            str(root / "input.f32"),
            "--output-dir",
            str(out_dir),
            "--cache-layout",
            str(root / "cache_layout.json"),
            "--cache-file",
            str(root / "decode_cache.bin"),
            "--position",
            "0",
            "--rms-norm-eps",
            "0",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    if "value source:       absorbed-alias" not in stdout:
        raise SystemExit("attention projections did not report absorbed alias source")
    if "kv_b skipped:       yes" not in stdout:
        raise SystemExit("attention projections did not skip kv_b")
    for name in ("attn_q_a.f32", "attn_q_a_norm.f32", "attn_q_b.f32", "attn_kv_a.f32", "attn_kv_a_norm.f32"):
        if not (out_dir / name).exists():
            raise SystemExit(f"missing projection output {name}")
    if (out_dir / "attn_kv_b.f32").exists():
        raise SystemExit("absorbed projection unexpectedly wrote attn_kv_b.f32")
    if root.joinpath("decode_cache.bin").read_bytes() == b"\0" * (KV_A_OUT_DIM * 4):
        raise SystemExit("absorbed projection did not append KV-A cache")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run absorbed attention projection smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-attn-proj-absorbed-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    run_smoke(args.runner, root)
    print(f"fixture: {root}")
    print("  absorbed attention projections: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
