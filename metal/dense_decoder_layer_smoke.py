#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from decoder_layer_smoke import (
    attention_reference,
    f32,
    read_f32,
    rmsnorm,
    rotate,
    swiglu,
    write_cache,
    write_resident,
)


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def append_dense_mlp(root: Path) -> None:
    resident_dir = root / "resident"
    layout_path = resident_dir / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    payload = bytearray((resident_dir / "resident.bin").read_bytes())
    dense = f32([1.0] * 64)
    for component in ("gate_proj", "up_proj", "down_proj"):
        offset = len(payload)
        payload.extend(dense)
        layout["tensors"].append(
            {
                "name": f"model.layers.1.mlp.{component}.weight",
                "offset": offset,
                "size": len(dense),
                "dtype": "F32",
                "shape": [8, 8],
                "category": "dense_mlp",
            }
        )
    layout["total_bytes"] = len(payload)
    (resident_dir / "resident.bin").write_bytes(payload)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def dense_mlp_reference(attn_out: list[float], eps: float) -> list[float]:
    normed = rmsnorm(attn_out, eps)
    gate = sum(normed)
    up = sum(normed)
    value = 8.0 * swiglu(gate) * up
    return [x + value for x in attn_out]


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-dense-decoder-layer-", dir="/private/tmp"))
    kv_b, o_proj = write_resident(root)
    append_dense_mlp(root)
    cache = write_cache(root)
    residual = [1.0] * 8
    (root / "input.f32").write_bytes(f32(residual))
    runner = str(Path(__file__).with_name("largerlm-runner"))

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
            "--run-dense-decoder-layer",
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
            "--rms-norm-eps",
            "0",
            "--output-f32",
            str(root / "decoder_out.f32"),
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

    q_nope = [1.0, -1.0]
    q_rope = rotate([-1.0, 2.0], 1) + rotate([1.0, -1.0], 1)
    attn_out = attention_reference(q_nope, q_rope, residual, cache, kv_b, o_proj)
    expected = dense_mlp_reference(attn_out, eps=0.0)
    got = read_f32(root / "decoder_out.f32", 8)
    if any(math.fabs(a - b) > 5e-3 for a, b in zip(got, expected)):
        raise SystemExit(f"dense decoder output {got} != expected {expected}")
    print(f"fixture: {root}")
    print(f"  expected output[0]: {expected[0]:.6f}")
    print("  dense decoder:      ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
