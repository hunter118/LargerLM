#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from glm_moe_infer_mla_streaming_smoke import write_dummy_layouts


def parse_contexts(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("contexts must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one context is required")
    return values


def write_f32_stream(path: Path, count: int, value_at, *, chunk: int = 16384) -> None:
    with path.open("wb") as f:
        start = 0
        while start < count:
            end = min(start + chunk, count)
            values = [float(value_at(i)) for i in range(start, end)]
            f.write(struct.pack(f"<{len(values)}f", *values))
            start = end


def deterministic_weight(i: int) -> float:
    x = (i * 1103515245 + 12345) & 0xFFFF
    return (x / 65535.0 - 0.5) * 0.04


def write_fixture(
    root: Path,
    *,
    max_context: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
) -> dict[str, int]:
    q_nope_count = num_heads * qk_nope_dim
    q_rope_count = num_heads * rope_dim
    cache_width = kv_lora_dim + rope_dim
    cache_count = max_context * cache_width
    kv_b_count = num_heads * (qk_nope_dim + v_head_dim) * kv_lora_dim

    write_f32_stream(
        root / "q_nope.f32",
        q_nope_count,
        lambda i: 0.05 * math.sin(0.0017 * (i + 1)) + deterministic_weight(i),
    )
    write_f32_stream(
        root / "q_rope.f32",
        q_rope_count,
        lambda i: 0.04 * math.cos(0.0023 * (i + 3)) + deterministic_weight(i + 17),
    )
    write_f32_stream(
        root / "decode_cache.bin",
        cache_count,
        lambda i: 0.03 * math.sin(0.0009 * (i + 5)) + deterministic_weight(i + 29),
    )
    write_f32_stream(
        root / "kv_b.f32",
        kv_b_count,
        lambda i: deterministic_weight(i + 43),
    )

    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": max_context,
                "dtype": "F32",
                "dtype_bytes": 4,
                "alignment": 64,
                "total_bytes": cache_count * 4,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": cache_width,
                        "dtype": "F32",
                        "dtype_bytes": 4,
                        "token_stride_bytes": cache_width * 4,
                        "max_context_tokens": max_context,
                        "total_bytes": cache_count * 4,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "q_nope_bytes": q_nope_count * 4,
        "q_rope_bytes": q_rope_count * 4,
        "cache_bytes": cache_count * 4,
        "kv_b_bytes": kv_b_count * 4,
        "output_bytes": num_heads * v_head_dim * 4,
    }


def run_probe(
    *,
    binary: Path,
    root: Path,
    resident_layout: Path,
    expert_layout: Path,
    context: int,
    num_heads: int,
    kv_lora_dim: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    max_live_working_set_mib: int,
) -> dict[str, object]:
    output = root / f"streaming_ctx{context}.f32"
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
        str(context),
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
        "0",
        "--rope-theta",
        "10000",
        "--output-f32",
        str(output),
        "--max-cache-file-mib",
        "8",
        "--max-cache-read-mib",
        "8",
        "--max-live-working-set-mib",
        str(max_live_working_set_mib),
        "--json",
    ]
    started = time.perf_counter()
    completed = subprocess.run(cmd, text=True, capture_output=True)
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    mla = payload.get("probe_mla_attention") or {}
    if not payload.get("ok") or not mla.get("ok"):
        raise SystemExit(f"streaming MLA microbench probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("streaming MLA microbench should not allocate expert buffers")
    if mla.get("value_source") != "direct-f32":
        raise SystemExit(f"expected direct-f32 value source: {mla}")
    return {
        "context_length": context,
        "wall_seconds": wall,
        "kernel_seconds": float(mla["kernel_seconds"]),
        "elapsed_seconds": float(mla["elapsed_seconds"]),
        "estimated_live_working_set_bytes": int(payload["estimated_live_working_set_bytes"]),
        "scratch_bytes": int(mla["scratch_bytes"]),
        "raw_cache_bytes": int(mla["raw_cache_bytes"]),
        "cache_f32_bytes": int(mla["cache_f32_bytes"]),
        "kv_b_f32_bytes": int(mla["kv_b_f32_bytes"]),
        "output0": float(mla["output0"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a GLM-shaped direct-F32 MLA microbench without reading real model weights."
    )
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--contexts", type=parse_contexts, default=parse_contexts("9,32,33,64"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--kv-lora-dim", type=int, default=512)
    parser.add_argument("--qk-nope-dim", type=int, default=192)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--v-head-dim", type=int, default=256)
    parser.add_argument("--max-live-working-set-mib", type=int, default=512)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-mla-streaming-microbench-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    resident_layout, expert_layout = write_dummy_layouts(root)
    max_context = max(args.contexts)
    fixture_started = time.perf_counter()
    sizes = write_fixture(
        root,
        max_context=max_context,
        num_heads=args.num_heads,
        kv_lora_dim=args.kv_lora_dim,
        qk_nope_dim=args.qk_nope_dim,
        rope_dim=args.rope_dim,
        v_head_dim=args.v_head_dim,
    )
    fixture_seconds = time.perf_counter() - fixture_started

    results = []
    for context in args.contexts:
        samples = [
            run_probe(
                binary=args.binary,
                root=root,
                resident_layout=resident_layout,
                expert_layout=expert_layout,
                context=context,
                num_heads=args.num_heads,
                kv_lora_dim=args.kv_lora_dim,
                qk_nope_dim=args.qk_nope_dim,
                rope_dim=args.rope_dim,
                v_head_dim=args.v_head_dim,
                max_live_working_set_mib=args.max_live_working_set_mib,
            )
            for _ in range(args.repeats)
        ]
        kernel_seconds = [float(sample["kernel_seconds"]) for sample in samples]
        elapsed_seconds = [float(sample["elapsed_seconds"]) for sample in samples]
        wall_seconds = [float(sample["wall_seconds"]) for sample in samples]
        first = samples[0]
        results.append(
            {
                "context_length": context,
                "repeats": args.repeats,
                "kernel_seconds_min": min(kernel_seconds),
                "kernel_seconds_median": statistics.median(kernel_seconds),
                "elapsed_seconds_median": statistics.median(elapsed_seconds),
                "wall_seconds_median": statistics.median(wall_seconds),
                "estimated_live_working_set_bytes": first["estimated_live_working_set_bytes"],
                "scratch_bytes": first["scratch_bytes"],
                "raw_cache_bytes": first["raw_cache_bytes"],
                "cache_f32_bytes": first["cache_f32_bytes"],
                "kv_b_f32_bytes": first["kv_b_f32_bytes"],
                "output0_first": first["output0"],
                "samples": samples,
            }
        )
    payload = {
        "fixture": str(root),
        "fixture_seconds": fixture_seconds,
        "sizes": sizes,
        "shape": {
            "num_heads": args.num_heads,
            "kv_lora_dim": args.kv_lora_dim,
            "qk_nope_dim": args.qk_nope_dim,
            "rope_dim": args.rope_dim,
            "v_head_dim": args.v_head_dim,
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
