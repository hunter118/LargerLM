#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


NUM_HEADS = 1
KV_LORA_DIM = 8
QK_NOPE_DIM = 8
ROPE_DIM = 2
V_HEAD_DIM = 8
CONTEXT_LEN = 3
SCALE_E8M0_ONE = 127


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def mxfp4_tensor3d(dim0: int, dim1: int, logical_dim2: int, code: int = 2) -> tuple[bytes, bytes]:
    if logical_dim2 % 8:
        raise ValueError("logical_dim2 must be divisible by 8")
    packed_cols = logical_dim2 // 8
    weights = bytearray()
    packed = pack8(code)
    for _ in range(dim0 * dim1 * packed_cols):
        weights.extend(struct.pack("<I", packed))
    scales = bytes([SCALE_E8M0_ONE]) * (dim0 * dim1)
    return bytes(weights), scales


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, dtype: str, shape: list[int], data: bytes) -> None:
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": "attention",
            }
        )
        payload.extend(data)

    embed_w, embed_s = mxfp4_tensor3d(NUM_HEADS, KV_LORA_DIM, QK_NOPE_DIM)
    unembed_w, unembed_s = mxfp4_tensor3d(NUM_HEADS, V_HEAD_DIM, KV_LORA_DIM)
    add(
        "model.layers.1.self_attn.embed_q.weight",
        "U32",
        [NUM_HEADS, KV_LORA_DIM, QK_NOPE_DIM // 8],
        embed_w,
    )
    add(
        "model.layers.1.self_attn.embed_q.scales",
        "U8",
        [NUM_HEADS, KV_LORA_DIM, 1],
        embed_s,
    )
    add(
        "model.layers.1.self_attn.unembed_out.weight",
        "U32",
        [NUM_HEADS, V_HEAD_DIM, KV_LORA_DIM // 8],
        unembed_w,
    )
    add(
        "model.layers.1.self_attn.unembed_out.scales",
        "U8",
        [NUM_HEADS, V_HEAD_DIM, 1],
        unembed_s,
    )

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

    cache = [
        [0.125] * KV_LORA_DIM + [0.0] * ROPE_DIM,
        [0.25] * KV_LORA_DIM + [0.0] * ROPE_DIM,
        [0.0625] * KV_LORA_DIM + [0.0] * ROPE_DIM,
    ]
    cache_payload = f32([value for row in cache for value in row])
    (root / "decode_cache.bin").write_bytes(cache_payload)
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": CONTEXT_LEN,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 64,
                "total_bytes": len(cache_payload),
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": KV_LORA_DIM + ROPE_DIM,
                        "dtype": "F32",
                        "dtype_bytes": 4,
                        "token_stride_bytes": (KV_LORA_DIM + ROPE_DIM) * 4,
                        "max_context_tokens": CONTEXT_LEN,
                        "total_bytes": len(cache_payload),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (root / "q_nope_single.f32").write_bytes(
        f32([0.1, 0.2, 0.3, 0.4, 0.15, 0.05, -0.1, 0.25])
    )
    (root / "q_rope_single.f32").write_bytes(f32([0.0] * (NUM_HEADS * ROPE_DIM)))
    (root / "q_nope_batch.f32").write_bytes(
        f32(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.15,
                0.05,
                -0.1,
                0.25,
                -0.1,
                0.05,
                0.2,
                -0.3,
                0.1,
                0.05,
                0.15,
                -0.05,
            ]
        )
    )
    (root / "q_rope_batch.f32").write_bytes(f32([0.0] * (2 * NUM_HEADS * ROPE_DIM)))
    (root / "indices.u32").write_bytes(struct.pack("<6I", 2, 0, 2, 2, 1, 2))


def attention_value(latent: list[float]) -> float:
    return sum(latent[:KV_LORA_DIM])


def reference_for_context(q_nope: list[float], cache_rows: list[list[float]]) -> list[float]:
    scale = 1.0 / math.sqrt(QK_NOPE_DIM + ROPE_DIM)
    q_sum = sum(q_nope[:QK_NOPE_DIM])
    scores = [attention_value(row) * q_sum * scale for row in cache_rows]
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    denom = sum(weights)
    value = sum(w * attention_value(row) for w, row in zip(weights, cache_rows)) / denom
    return [value] * V_HEAD_DIM


def reference_outputs(root: Path) -> tuple[list[float], list[float], list[float]]:
    raw = read_f32(root / "decode_cache.bin", CONTEXT_LEN * (KV_LORA_DIM + ROPE_DIM))
    width = KV_LORA_DIM + ROPE_DIM
    cache = [list(raw[i * width : (i + 1) * width]) for i in range(CONTEXT_LEN)]
    q_single = list(read_f32(root / "q_nope_single.f32", QK_NOPE_DIM))
    q_batch = list(read_f32(root / "q_nope_batch.f32", 2 * QK_NOPE_DIM))
    single = reference_for_context(q_single, cache)
    batch = (
        reference_for_context(q_batch[:QK_NOPE_DIM], cache[:2])
        + reference_for_context(q_batch[QK_NOPE_DIM:], cache[:3])
    )
    indexed = (
        reference_for_context(q_batch[:QK_NOPE_DIM], [cache[0], cache[2]])
        + reference_for_context(q_batch[QK_NOPE_DIM:], [cache[1], cache[2]])
    )
    return single, batch, indexed


def run_checked(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if "value source:       absorbed-alias" not in completed.stdout:
        raise SystemExit("MLA attention did not report absorbed alias source")
    return completed.stdout


def common_args(runner: Path, root: Path, output: Path) -> list[str]:
    return [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(root / "decode_cache.bin"),
        "--layer",
        "1",
        "--context-length",
        str(CONTEXT_LEN),
        "--num-heads",
        str(NUM_HEADS),
        "--kv-lora-dim",
        str(KV_LORA_DIM),
        "--qk-nope-dim",
        str(QK_NOPE_DIM),
        "--rope-dim",
        str(ROPE_DIM),
        "--v-head-dim",
        str(V_HEAD_DIM),
        "--cache-position-offset",
        "0",
        "--rope-theta",
        "10000",
        "--output-f32",
        str(output),
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def assert_close(name: str, got: tuple[float, ...], expected: list[float]) -> None:
    if len(got) != len(expected):
        raise SystemExit(f"{name} length {len(got)} != expected {len(expected)}")
    if any(abs(a - b) > 5e-5 for a, b in zip(got, expected)):
        raise SystemExit(f"{name} output {got} != expected {expected}")


def run_smoke(runner: Path, root: Path) -> None:
    single_expected, batch_expected, indexed_expected = reference_outputs(root)

    single_out = root / "single.f32"
    run_checked(
        common_args(runner, root, single_out)
        + [
            "--run-mla-attention",
            "--q-nope-f32",
            str(root / "q_nope_single.f32"),
            "--q-rope-f32",
            str(root / "q_rope_single.f32"),
        ]
    )
    assert_close("single", read_f32(single_out, V_HEAD_DIM), single_expected)

    batch_out = root / "batch.f32"
    run_checked(
        common_args(runner, root, batch_out)
        + [
            "--run-mla-attention-batch",
            "--q-nope-f32",
            str(root / "q_nope_batch.f32"),
            "--q-rope-f32",
            str(root / "q_rope_batch.f32"),
            "--start-position",
            "1",
            "--batch-tokens",
            "2",
        ]
    )
    assert_close("batch", read_f32(batch_out, 2 * V_HEAD_DIM), batch_expected)

    indexed_out = root / "indexed.f32"
    run_checked(
        common_args(runner, root, indexed_out)
        + [
            "--run-mla-attention-indexed-batch",
            "--q-nope-f32",
            str(root / "q_nope_batch.f32"),
            "--q-rope-f32",
            str(root / "q_rope_batch.f32"),
            "--indices-u32",
            str(root / "indices.u32"),
            "--batch-tokens",
            "2",
            "--index-topk",
            "2",
        ]
    )
    assert_close("indexed", read_f32(indexed_out, 2 * V_HEAD_DIM), indexed_expected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run absorbed MLA attention alias smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-mla-absorbed-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    run_smoke(args.runner, root)
    print(f"fixture: {root}")
    print("  absorbed MLA attention: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
