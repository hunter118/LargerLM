#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(v) for v in values)


def rotate(values: list[float], position: int, theta: float) -> list[float]:
    dim = len(values)
    half = dim // 2
    out: list[float] = []
    for i, x in enumerate(values):
        if i < half:
            rot = -values[i + half]
            freq_idx = i
        else:
            rot = values[i - half]
            freq_idx = i - half
        angle = position / (theta ** (2.0 * freq_idx / dim))
        out.append(x * math.cos(angle) + rot * math.sin(angle))
    return out


def reference(
    q_nope: list[float],
    q_rope_rot: list[float],
    cache: list[list[float]],
    kv_b: list[list[float]],
    o_proj: list[list[float]],
    residual: list[float],
) -> list[float]:
    num_heads = 2
    kv_lora_dim = 2
    qk_nope_dim = 1
    rope_dim = 2
    v_head_dim = 1
    scale = 1.0 / math.sqrt(qk_nope_dim + rope_dim)
    values: list[float] = []
    for h in range(num_heads):
        row_base = h * (qk_nope_dim + v_head_dim)
        scores: list[float] = []
        head_values: list[float] = []
        for t, token in enumerate(cache):
            latent = token[:kv_lora_dim]
            k_rope = rotate(token[kv_lora_dim:], t, 10000.0)
            k_nope = sum(kv_b[row_base][r] * latent[r] for r in range(kv_lora_dim))
            score = q_nope[h] * k_nope
            score += sum(q_rope_rot[h * rope_dim + d] * k_rope[d] for d in range(rope_dim))
            scores.append(score * scale)
            v_row = kv_b[row_base + qk_nope_dim]
            head_values.append(sum(v_row[r] * latent[r] for r in range(kv_lora_dim)))
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        denom = sum(weights)
        values.append(sum(w * v for w, v in zip(weights, head_values)) / denom)
    projected = [sum(row[i] * values[i] for i in range(len(values))) for row in o_proj]
    return [a + b for a, b in zip(projected, residual)]


def write_fixture(root: Path) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[float]]:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    tensors = []
    payload = bytearray()

    blobs = [
        ("model.layers.1.input_layernorm.weight", "F32", [3], "norms", f32([1.0, 1.0, 1.0])),
        ("model.layers.1.self_attn.q_a_layernorm.weight", "F32", [2], "norms", f32([1.0, 1.0])),
        ("model.layers.1.self_attn.kv_a_layernorm.weight", "F32", [2], "norms", f32([1.0, 1.0])),
        (
            "model.layers.1.self_attn.q_a_proj.weight",
            "F32",
            [2, 3],
            "attention",
            f32([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
        (
            "model.layers.1.self_attn.q_b_proj.weight",
            "F32",
            [6, 2],
            "attention",
            f32([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.5, 0.5, 0.0, -1.0]),
        ),
        (
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
            "F32",
            [4, 3],
            "attention",
            f32([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
    ]
    kv_b = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, -1.0]]
    o_proj = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    blobs.extend(
        [
            (
                "model.layers.1.self_attn.kv_b_proj.weight",
                "F32",
                [4, 2],
                "attention",
                f32([v for row in kv_b for v in row]),
            ),
            (
                "model.layers.1.self_attn.o_proj.weight",
                "F32",
                [3, 2],
                "attention",
                f32([v for row in o_proj for v in row]),
            ),
        ]
    )
    for name, dtype, shape, category, data in blobs:
        offset = len(payload)
        payload.extend(data)
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": category,
            }
        )
    (resident / "resident.bin").write_bytes(payload)
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
    previous = [0.0, 1.0, -0.5, 0.25]
    (root / "decode_cache.bin").write_bytes(bf16(previous + [0.0, 0.0, 0.0, 0.0]))
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 2,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 16,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": 4,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": 8,
                        "max_context_tokens": 2,
                        "total_bytes": 16,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    residual = [1.0, 1.0, 1.0]
    (root / "input.f32").write_bytes(f32(residual))
    return [previous, [1.0, 2.0, 0.5, 1.0]], kv_b, o_proj, residual


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-attn-block-", dir="/private/tmp"))
    cache, kv_b, o_proj, residual = write_fixture(root)
    runner = str(Path(__file__).with_name("largerlm-runner"))
    out_dir = root / "proj"
    run(
        [
            runner,
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--layer",
            "1",
            "--run-attn-projections",
            "--input-f32",
            str(root / "input.f32"),
            "--rms-norm-eps",
            "0",
            "--output-dir",
            str(out_dir),
            "--cache-layout",
            str(root / "cache_layout.json"),
            "--cache-file",
            str(root / "decode_cache.bin"),
            "--position",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    q_b = list(read_f32(out_dir / "attn_q_b.f32", 6))
    q_nope = [q_b[0], q_b[3]]
    q_rot = [q_b[1], q_b[2], q_b[4], q_b[5]]
    (root / "q_nope.f32").write_bytes(f32(q_nope))
    (root / "q_rot.f32").write_bytes(f32(q_rot))
    (root / "dummy_k.f32").write_bytes(f32([0.0, 0.0]))
    run(
        [
            runner,
            "--run-rope",
            "--q-f32",
            str(root / "q_rot.f32"),
            "--k-f32",
            str(root / "dummy_k.f32"),
            "--output-q-f32",
            str(root / "q_rope.f32"),
            "--output-k-f32",
            str(root / "dummy_k_rope.f32"),
            "--num-heads",
            "2",
            "--rope-dim",
            "2",
            "--position",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    run(
        [
            runner,
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--cache-layout",
            str(root / "cache_layout.json"),
            "--cache-file",
            str(root / "decode_cache.bin"),
            "--layer",
            "1",
            "--run-mla-attention",
            "--q-nope-f32",
            str(root / "q_nope.f32"),
            "--q-rope-f32",
            str(root / "q_rope.f32"),
            "--context-length",
            "2",
            "--num-heads",
            "2",
            "--qk-nope-dim",
            "1",
            "--rope-dim",
            "2",
            "--v-head-dim",
            "1",
            "--output-f32",
            str(root / "attn_value.f32"),
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    run(
        [
            runner,
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--layer",
            "1",
            "--run-attn-output",
            "--input-f32",
            str(root / "attn_value.f32"),
            "--residual-f32",
            str(root / "input.f32"),
            "--output-f32",
            str(root / "attn_out.f32"),
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    q_rope = list(read_f32(root / "q_rope.f32", 4))
    expected = reference(q_nope, q_rope, cache, kv_b, o_proj, residual)
    got = read_f32(root / "attn_out.f32", 3)
    if any(abs(a - b) > 5e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"attention block {got} != expected {expected}")
    print(f"fixture: {root}")
    print("  attention block:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
