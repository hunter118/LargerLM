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


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(v) for v in values)


def rotate(values: list[float], position: int, theta: float, interleave: bool) -> list[float]:
    dim = len(values)
    half = dim // 2
    out: list[float] = []
    for i, x in enumerate(values):
        if interleave:
            pair = i ^ 1
            rot = values[pair] if i & 1 else -values[pair]
            freq_idx = i // 2
        else:
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
    q_rope: list[float],
    cache: list[list[float]],
    kv_b: list[list[float]],
    *,
    start_position: int,
    batch_tokens: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    position_offset: int,
    theta: float,
    interleave: bool,
) -> list[float]:
    scale = 1.0 / math.sqrt(qk_nope_dim + rope_dim)
    output: list[float] = []
    for token in range(batch_tokens):
        q_context = start_position + token + 1
        for h in range(num_heads):
            row_base = h * (qk_nope_dim + v_head_dim)
            scores: list[float] = []
            values: list[float] = []
            for t in range(q_context):
                latent = cache[t][:kv_lora_dim]
                k_rope = rotate(cache[t][kv_lora_dim:], position_offset + t, theta, interleave)
                score = 0.0
                for d in range(qk_nope_dim):
                    k_nope = sum(kv_b[row_base + d][r] * latent[r] for r in range(kv_lora_dim))
                    q_idx = token * num_heads * qk_nope_dim + h * qk_nope_dim + d
                    score += q_nope[q_idx] * k_nope
                for d in range(rope_dim):
                    q_idx = token * num_heads * rope_dim + h * rope_dim + d
                    score += q_rope[q_idx] * k_rope[d]
                scores.append(score * scale)
                v_row = kv_b[row_base + qk_nope_dim]
                values.append(sum(v_row[r] * latent[r] for r in range(kv_lora_dim)))
            max_score = max(scores)
            weights = [math.exp(score - max_score) for score in scores]
            denom = sum(weights)
            output.append(sum(w * v for w, v in zip(weights, values)) / denom)
    return output


def write_fixture(root: Path) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    kv_b = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, -1.0],
    ]
    payload = bf16([v for row in kv_b for v in row])
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
                "tensors": [
                    {
                        "name": "model.layers.1.self_attn.kv_b_proj.weight",
                        "offset": 0,
                        "size": len(payload),
                        "dtype": "BF16",
                        "shape": [4, 2],
                        "category": "attention",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cache = [
        [1.0, 2.0, 0.5, 1.0],
        [2.0, 1.0, 1.0, -0.5],
        [0.0, 3.0, -1.0, 0.25],
    ]
    cache_payload = bf16([v for row in cache for v in row])
    (root / "decode_cache.bin").write_bytes(cache_payload)
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 3,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": len(cache_payload),
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": 4,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": 8,
                        "max_context_tokens": 3,
                        "total_bytes": len(cache_payload),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    q_nope = [1.5, -0.5, -0.25, 0.75]
    q_rope = [0.3, -0.2, 1.0, 0.5, 0.4, 0.1, -0.6, 0.2]
    (root / "q_nope.f32").write_bytes(f32(q_nope))
    (root / "q_rope.f32").write_bytes(f32(q_rope))
    return q_nope, q_rope, cache, kv_b


def run_case(root: Path, expected: list[float], interleave: bool) -> None:
    output = root / ("batch_attn_interleave.f32" if interleave else "batch_attn_default.f32")
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(root / "decode_cache.bin"),
        "--layer",
        "1",
        "--run-mla-attention-batch",
        "--q-nope-f32",
        str(root / "q_nope.f32"),
        "--q-rope-f32",
        str(root / "q_rope.f32"),
        "--context-length",
        "3",
        "--start-position",
        "1",
        "--batch-tokens",
        "2",
        "--num-heads",
        "2",
        "--qk-nope-dim",
        "1",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "1",
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
    if interleave:
        cmd.append("--rope-interleave")
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    got = struct.unpack("<4f", output.read_bytes())
    if any(abs(a - b) > 3e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"interleave={interleave} output {got} != expected {expected}")


def run_server_case(root: Path, expected: list[float], interleave: bool) -> None:
    output = root / (
        "batch_attn_server_interleave.f32"
        if interleave
        else "batch_attn_server_default.f32"
    )
    request = {
        "resident_layout": str(root / "resident" / "layout.json"),
        "cache_layout": str(root / "cache_layout.json"),
        "cache_file": str(root / "decode_cache.bin"),
        "layer": 1,
        "q_nope_f32": str(root / "q_nope.f32"),
        "q_rope_f32": str(root / "q_rope.f32"),
        "output_f32": str(output),
        "context_length": 3,
        "start_position": 1,
        "batch_tokens": 2,
        "num_heads": 2,
        "kv_lora_dim": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "cache_position_offset": 0,
        "rope_theta": 10000.0,
        "rope_interleave": interleave,
        "max_cache_file_mib": 1,
        "max_cache_read_mib": 1,
        "max_resident_matrix_mib": 1,
        "max_runner_scratch_mib": 64,
    }
    payload = (
        json.dumps(request, separators=(",", ":"))
        + "\n"
        + json.dumps({"command": "quit"}, separators=(",", ":"))
        + "\n"
    )
    completed = subprocess.run(
        [str(Path(__file__).with_name("largerlm-runner")), "--run-mla-attention-batch-server-jsonl"],
        input=payload,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if "  server request:      ok" not in completed.stdout:
        raise SystemExit("MLA attention batch server did not report success")
    got = struct.unpack("<4f", output.read_bytes())
    if any(abs(a - b) > 3e-4 for a, b in zip(got, expected)):
        raise SystemExit(
            f"server interleave={interleave} output {got} != expected {expected}"
        )


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-mla-attn-batch-", dir="/private/tmp"))
    q_nope, q_rope, cache, kv_b = write_fixture(root)
    common = {
        "start_position": 1,
        "batch_tokens": 2,
        "num_heads": 2,
        "kv_lora_dim": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "position_offset": 0,
        "theta": 10000.0,
    }
    run_case(
        root,
        reference(q_nope, q_rope, cache, kv_b, interleave=False, **common),
        interleave=False,
    )
    run_server_case(
        root,
        reference(q_nope, q_rope, cache, kv_b, interleave=False, **common),
        interleave=False,
    )
    run_case(
        root,
        reference(q_nope, q_rope, cache, kv_b, interleave=True, **common),
        interleave=True,
    )
    run_server_case(
        root,
        reference(q_nope, q_rope, cache, kv_b, interleave=True, **common),
        interleave=True,
    )
    print(f"fixture: {root}")
    print("  mla attention batch: ok")
    print("  mla attention batch server: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
