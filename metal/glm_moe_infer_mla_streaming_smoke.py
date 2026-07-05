#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def parse_contexts(raw: str) -> list[int]:
    contexts: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("contexts must be positive")
        contexts.append(value)
    if not contexts:
        raise argparse.ArgumentTypeError("at least one context is required")
    return contexts


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise SystemExit(f"{path} is not f32-aligned")
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def rotate(values: list[float], position: int, theta: float, interleave: bool) -> list[float]:
    dim = len(values)
    half = dim // 2
    out: list[float] = []
    for i, x in enumerate(values):
        if interleave:
            pair = i ^ 1
            rot = values[pair] if i & 1 else -values[pair]
            freq_idx = i // 2
        elif i < half:
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
    for h in range(num_heads):
        row_base = h * (qk_nope_dim + v_head_dim)
        scores: list[float] = []
        values: list[list[float]] = []
        for t, token in enumerate(cache):
            latent = token[:kv_lora_dim]
            k_rope = rotate(token[kv_lora_dim:], position_offset + t, theta, interleave)
            score = 0.0
            for d in range(qk_nope_dim):
                row = kv_b[row_base + d]
                k_nope = sum(row[r] * latent[r] for r in range(kv_lora_dim))
                score += q_nope[h * qk_nope_dim + d] * k_nope
            score += sum(q_rope[h * rope_dim + d] * k_rope[d] for d in range(rope_dim))
            scores.append(score * scale)
            head_values: list[float] = []
            for v in range(v_head_dim):
                row = kv_b[row_base + qk_nope_dim + v]
                head_values.append(sum(row[r] * latent[r] for r in range(kv_lora_dim)))
            values.append(head_values)
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        denom = sum(weights)
        for v in range(v_head_dim):
            output.append(sum(w * row[v] for w, row in zip(weights, values)) / denom)
    return output


def write_dummy_layouts(root: Path) -> tuple[Path, Path]:
    resident = root / "resident"
    experts = root / "experts"
    resident.mkdir(parents=True)
    experts.mkdir(parents=True)
    (resident / "resident.bin").write_bytes(b"")
    resident_layout = resident / "layout.json"
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": 0,
                "tensors": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (experts / "layer_001.bin").write_bytes(b"\0")
    expert_layout = experts / "layout.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-mxfp4",
                "group_size": 32,
                "num_layers": 2,
                "num_experts": 1,
                "component_order": [],
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 1,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_001.bin",
                        "components": [],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return resident_layout, expert_layout


def fixture_values(
    *,
    context_length: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    q_nope = [
        0.17 * math.sin(0.31 * (i + 1)) + 0.03 * math.cos(0.07 * i)
        for i in range(num_heads * qk_nope_dim)
    ]
    q_rope = [
        0.13 * math.cos(0.19 * (i + 2)) - 0.02 * math.sin(0.11 * i)
        for i in range(num_heads * rope_dim)
    ]
    cache: list[list[float]] = []
    for t in range(context_length):
        row: list[float] = []
        for i in range(kv_lora_dim + rope_dim):
            row.append(0.21 * math.sin(0.17 * (t + 1) * (i + 1)) + 0.04 * math.cos(0.05 * (t + i)))
        cache.append(row)
    rows = num_heads * (qk_nope_dim + v_head_dim)
    kv_b: list[list[float]] = []
    for row in range(rows):
        kv_b.append(
            [
                0.15 * math.sin(0.23 * (row + 1) * (r + 1))
                + 0.05 * math.cos(0.13 * (row - r))
                for r in range(kv_lora_dim)
            ]
        )
    return q_nope, q_rope, cache, kv_b


def write_fixture(
    root: Path,
    *,
    context_length: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    q_nope, q_rope, cache, kv_b = fixture_values(
        context_length=context_length,
        num_heads=num_heads,
        kv_lora_dim=kv_lora_dim,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
    )
    (root / "q_nope.f32").write_bytes(f32(q_nope))
    (root / "q_rope.f32").write_bytes(f32(q_rope))
    (root / "kv_b.f32").write_bytes(f32([v for row in kv_b for v in row]))
    (root / "decode_cache.bin").write_bytes(f32([v for row in cache for v in row]))
    width = kv_lora_dim + rope_dim
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": context_length,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 64,
                "total_bytes": context_length * width * 4,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": width,
                        "dtype": "F32",
                        "dtype_bytes": 4,
                        "token_stride_bytes": width * 4,
                        "max_context_tokens": context_length,
                        "total_bytes": context_length * width * 4,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return q_nope, q_rope, cache, kv_b


def run_case(
    *,
    binary: Path,
    root: Path,
    resident_layout: Path,
    expert_layout: Path,
    expected: list[float],
    context_length: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    position_offset: int,
    theta: float,
    interleave: bool,
    max_diff: float,
) -> dict[str, object]:
    output = root / (
        f"mla_ctx{context_length}_{'interleave' if interleave else 'default'}.f32"
    )
    cmd = [
        str(binary),
        "--resident-layout",
        str(resident_layout),
        "--expert-layout",
        str(expert_layout),
        "--no-open-experts",
        "--probe-mla-attention",
        "--probe-layer",
        "1",
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(root / "decode_cache.bin"),
        "--q-nope-f32",
        str(root / "q_nope.f32"),
        "--q-rope-f32",
        str(root / "q_rope.f32"),
        "--mla-kv-b-f32",
        str(root / "kv_b.f32"),
        "--context-length",
        str(context_length),
        "--num-heads",
        str(num_heads),
        "--kv-lora-dim",
        str(kv_lora_dim),
        "--qk-nope-dim",
        str(qk_nope_dim),
        "--rope-dim",
        str(rope_dim),
        "--v-head-dim",
        str(v_head_dim),
        "--cache-position-offset",
        str(position_offset),
        "--rope-theta",
        f"{theta:.9g}",
        "--output-f32",
        str(output),
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-live-working-set-mib",
        "32",
        "--json",
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
    payload = json.loads(completed.stdout)
    mla = payload.get("probe_mla_attention") or {}
    if not payload.get("ok") or not mla.get("ok"):
        raise SystemExit(f"streaming MLA probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("direct MLA streaming smoke should not allocate expert buffers")
    if mla.get("context_length") != context_length:
        raise SystemExit(f"unexpected context length in payload: {mla}")
    got = read_f32(output)
    if len(got) != len(expected):
        raise SystemExit(f"output length {len(got)} != expected {len(expected)}")
    diffs = [abs(a - b) for a, b in zip(got, expected)]
    max_index = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    max_abs_diff = diffs[max_index] if diffs else 0.0
    if max_abs_diff > max_diff:
        raise SystemExit(
            f"interleave={interleave} max_abs_diff {max_abs_diff:.9g} "
            f"at {max_index}: got {got[max_index]:.9g}, expected {expected[max_index]:.9g}"
        )
    return {
        "context_length": context_length,
        "interleave": interleave,
        "max_abs_diff": max_abs_diff,
        "max_index": max_index,
        "kernel_seconds": mla.get("kernel_seconds"),
        "scratch_bytes": mla.get("scratch_bytes"),
        "kv_b_f32_bytes": mla.get("kv_b_f32_bytes"),
        "output0": mla.get("output0"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate glm_moe_infer streaming MLA attention on contextLength>32 synthetic tensors."
    )
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--contexts", type=parse_contexts, default=parse_contexts("9,32,33"))
    parser.add_argument("--max-diff", type=float, default=5e-5)
    args = parser.parse_args()

    max_context_length = max(args.contexts)
    num_heads = 3
    kv_lora_dim = 4
    qk_nope_dim = 3
    rope_dim = 4
    v_head_dim = 2
    position_offset = 7
    theta = 10000.0

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-mla-streaming-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    resident_layout, expert_layout = write_dummy_layouts(root)
    q_nope, q_rope, cache, kv_b = write_fixture(
        root,
        context_length=max_context_length,
        num_heads=num_heads,
        kv_lora_dim=kv_lora_dim,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
    )
    common = {
        "num_heads": num_heads,
        "kv_lora_dim": kv_lora_dim,
        "qk_nope_dim": qk_nope_dim,
        "rope_dim": rope_dim,
        "v_head_dim": v_head_dim,
        "position_offset": position_offset,
        "theta": theta,
    }
    results = []
    for interleave in (False, True):
        for context_length in args.contexts:
            expected = reference(
                q_nope,
                q_rope,
                cache[:context_length],
                kv_b,
                interleave=interleave,
                **common,
            )
            results.append(
                run_case(
                    binary=args.binary,
                    root=root,
                    resident_layout=resident_layout,
                    expert_layout=expert_layout,
                    expected=expected,
                    context_length=context_length,
                    max_diff=args.max_diff,
                    interleave=interleave,
                    **common,
                )
            )
    print(
        json.dumps(
            {
                "fixture": str(root),
                "contexts": args.contexts,
                "max_context_length": max_context_length,
                "num_heads": num_heads,
                "kv_lora_dim": kv_lora_dim,
                "qk_nope_dim": qk_nope_dim,
                "rope_dim": rope_dim,
                "v_head_dim": v_head_dim,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("  glm_moe_infer streaming MLA attention: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
