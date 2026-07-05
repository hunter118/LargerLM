#!/usr/bin/env python3
from __future__ import annotations

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


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-decoder-command-", dir="/private/tmp"))
    write_experts(root)
    kv_b, o_proj = write_resident(root)
    cache = write_cache(root)
    residual = [1.0] * 8
    (root / "input.f32").write_bytes(f32(residual))
    runner = str(Path(__file__).with_name("largerlm-runner"))

    run(
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

    q_nope = [1.0, -1.0]
    q_rope = rotate([-1.0, 2.0], 1) + rotate([1.0, -1.0], 1)
    attn_out = attention_reference(q_nope, q_rope, residual, cache, kv_b, o_proj)
    expected = mlp_reference(attn_out, eps=0.0)
    got = read_f32(root / "decoder_out.f32", 8)
    if any(abs(a - b) > 5e-3 for a, b in zip(got, expected)):
        raise SystemExit(f"decoder command output {got} != expected {expected}")
    print(f"fixture: {root}")
    print("  decoder command:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
