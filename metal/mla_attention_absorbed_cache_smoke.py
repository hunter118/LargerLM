#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path

from mla_attention_batch_smoke import bf16, f32, reference


def write_fixture(root: Path) -> tuple[list[float], list[float], list[list[float]], list[list[float]]]:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    kv_b = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [1.0, -1.0],
    ]
    embed_q = [1.0, 0.0, 0.5, 0.5]
    unembed_out = [0.0, 1.0, 1.0, -1.0]
    embed_payload = bf16(embed_q)
    unembed_payload = bf16(unembed_out)
    (resident / "resident.bin").write_bytes(embed_payload + unembed_payload)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(embed_payload) + len(unembed_payload),
                "tensors": [
                    {
                        "name": "model.layers.1.self_attn.embed_q.weight",
                        "offset": 0,
                        "size": len(embed_payload),
                        "dtype": "BF16",
                        "shape": [2, 2, 1],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.unembed_out.weight",
                        "offset": len(embed_payload),
                        "size": len(unembed_payload),
                        "dtype": "BF16",
                        "shape": [2, 1, 2],
                        "category": "attention",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    q_nope = [0.2, -0.3, 0.4, 0.1]
    q_rope = [0.1, 0.2, -0.2, 0.3, 0.4, -0.1, -0.5, 0.2]
    cache = [
        [0.5, 1.0, 0.1, 0.2],
        [1.5, -0.5, 0.0, 0.3],
        [0.25, 0.75, -0.1, 0.4],
    ]
    (root / "q_nope.f32").write_bytes(f32(q_nope))
    (root / "q_rope.f32").write_bytes(f32(q_rope))
    cache_payload = bf16([value for row in cache for value in row])
    (root / "cache.bin").write_bytes(cache_payload)
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
            }
        ),
        encoding="utf-8",
    )
    return q_nope, q_rope, cache, kv_b


def run_case(
    root: Path,
    *,
    runner: str,
    cache_dir: Path,
    output: Path,
    expected: list[float],
) -> str:
    cmd = [
        runner,
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(root / "cache.bin"),
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
        "--kv-lora-dim",
        "2",
        "--cache-position-offset",
        "0",
        "--rope-theta",
        "10000",
        "--mla-kv-b-cache-dir",
        str(cache_dir),
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
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    got = struct.unpack("<4f", output.read_bytes())
    if any(abs(a - b) > 3e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"output {got} != expected {expected}")
    return completed.stdout


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-mla-absorbed-cache-", dir="/private/tmp"))
    q_nope, q_rope, cache, kv_b = write_fixture(root)
    expected = reference(
        q_nope,
        q_rope,
        cache,
        kv_b,
        start_position=1,
        batch_tokens=2,
        num_heads=2,
        kv_lora_dim=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        position_offset=0,
        theta=10000.0,
        interleave=False,
    )
    runner = "metal/largerlm-runner"
    cache_dir = root / "mla_kv_b_cache"
    first_stdout = run_case(
        root,
        runner=runner,
        cache_dir=cache_dir,
        output=root / "first.f32",
        expected=expected,
    )
    cache_files = sorted(cache_dir.glob("*.bin"))
    if len(cache_files) != 1:
        raise SystemExit(f"expected one cache file, found {cache_files}")
    expected_cache = f32([value for row in kv_b for value in row])
    if cache_files[0].read_bytes() != expected_cache:
        raise SystemExit("materialized cache file does not match expected f32 kv_b")
    if "kv_b cache dir:" not in first_stdout:
        raise SystemExit("runner did not report MLA kv_b cache dir")

    bad_alias_payload = bf16([0.0] * 8)
    (root / "resident" / "resident.bin").write_bytes(bad_alias_payload)
    run_case(
        root,
        runner=runner,
        cache_dir=cache_dir,
        output=root / "second.f32",
        expected=expected,
    )
    print(f"fixture: {root}")
    print("  mla attention absorbed cache: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
