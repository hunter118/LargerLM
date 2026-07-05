#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from decoder_layer_smoke import (
    attention_reference,
    f32,
    mlp_reference,
    read_f32,
    rotate,
    write_cache,
    write_experts,
    write_resident,
)


def run(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def replace_kv_b_with_absorbed_aliases(root: Path, kv_b: list[list[float]]) -> None:
    resident = root / "resident"
    layout_path = resident / "layout.json"
    bin_path = resident / "resident.bin"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    tensors = layout["tensors"]
    assert isinstance(tensors, list)
    tensors[:] = [
        tensor
        for tensor in tensors
        if not tensor["name"].endswith(".self_attn.kv_b_proj.weight")
    ]

    payload = bytearray(bin_path.read_bytes())

    def add(name: str, shape: list[int], values: list[float]) -> None:
        data = f32(values)
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

    embed_q = [
        kv_b[0][0],
        kv_b[0][1],
        kv_b[2][0],
        kv_b[2][1],
    ]
    unembed_out = [
        kv_b[1][0],
        kv_b[1][1],
        kv_b[3][0],
        kv_b[3][1],
    ]
    add("model.layers.1.self_attn.embed_q.weight", [2, 2, 1], embed_q)
    add("model.layers.1.self_attn.unembed_out.weight", [2, 1, 2], unembed_out)
    layout["total_bytes"] = len(payload)
    bin_path.write_bytes(bytes(payload))
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-decoder-absorbed-", dir="/private/tmp"))
    write_experts(root)
    kv_b, o_proj = write_resident(root)
    replace_kv_b_with_absorbed_aliases(root, kv_b)
    cache = write_cache(root)
    residual = [1.0] * 8
    (root / "input.f32").write_bytes(f32(residual))
    runner = str(Path(__file__).with_name("largerlm-runner"))
    cache_dir = root / "mla-kv-b-cache"

    stdout = run(
        [
            runner,
            "--layout",
            str(root / "experts" / "layout.json"),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--cache-layout",
            str(root / "cache_layout.json"),
            "--cache-file",
            str(root / "decode_cache.bin"),
            "--layer",
            "1",
            "--run-decoder-layer",
            "--input-f32",
            str(root / "input.f32"),
            "--position",
            "1",
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
            "--top-k",
            "2",
            "--max-k",
            "2",
            "--router-score",
            "raw",
            "--routed-scaling-factor",
            "1",
            "--include-shared-expert",
            "--rms-norm-eps",
            "0",
            "--mla-kv-b-cache-dir",
            str(cache_dir),
            "--output-f32",
            str(root / "decoder_out.f32"),
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
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
    if "value source:       absorbed-alias" not in stdout:
        raise SystemExit("decoder command did not use absorbed alias attention")
    if "kv_b cache dir:" not in stdout:
        raise SystemExit("decoder command did not report MLA kv_b cache dir")
    cache_files = sorted(cache_dir.glob("*.bin"))
    if len(cache_files) != 1:
        raise SystemExit(f"expected one materialized MLA kv_b cache file, got {cache_files}")
    expected_cache = f32([value for row in kv_b for value in row])
    if cache_files[0].read_bytes() != expected_cache:
        raise SystemExit("materialized decoder MLA kv_b cache file is incorrect")

    q_nope = [1.0, -1.0]
    q_rope = rotate([-1.0, 2.0], 1) + rotate([1.0, -1.0], 1)
    attn_out = attention_reference(q_nope, q_rope, residual, cache, kv_b, o_proj)
    expected = mlp_reference(attn_out, eps=0.0)
    got = read_f32(root / "decoder_out.f32", 8)
    if any(abs(a - b) > 5e-3 for a, b in zip(got, expected)):
        raise SystemExit(f"absorbed decoder command output {got} != expected {expected}")
    print(f"fixture: {root}")
    print("  absorbed decoder command: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
