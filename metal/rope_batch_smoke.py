#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
import subprocess
import tempfile
from pathlib import Path


def write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


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


def expected_q(q: list[float], *, num_heads: int, rope_dim: int, start: int, interleave: bool) -> list[float]:
    out: list[float] = []
    per_token = num_heads * rope_dim
    for token in range(len(q) // per_token):
        position = start + token
        token_values = q[token * per_token : (token + 1) * per_token]
        for head in range(num_heads):
            base = head * rope_dim
            out.extend(rotate(token_values[base : base + rope_dim], position, 10000.0, interleave))
    return out


def expected_k(k: list[float], *, rope_dim: int, start: int, interleave: bool) -> list[float]:
    out: list[float] = []
    for token in range(len(k) // rope_dim):
        position = start + token
        base = token * rope_dim
        out.extend(rotate(k[base : base + rope_dim], position, 10000.0, interleave))
    return out


def run_case(root: Path, interleave: bool) -> None:
    num_heads = 2
    rope_dim = 4
    start = 3
    q = [
        1.0,
        2.0,
        3.0,
        4.0,
        0.5,
        1.0,
        1.5,
        2.0,
        -1.0,
        -2.0,
        0.5,
        1.5,
        2.0,
        -1.0,
        3.0,
        -2.0,
    ]
    k = [1.0, -1.0, 2.0, -2.0, 0.25, 0.5, -0.75, 1.0]
    suffix = "interleave" if interleave else "default"
    q_path = root / f"q_batch_{suffix}.f32"
    k_path = root / f"k_batch_{suffix}.f32"
    out_q = root / f"out_q_batch_{suffix}.f32"
    out_k = root / f"out_k_batch_{suffix}.f32"
    write_f32(q_path, q)
    write_f32(k_path, k)
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--run-rope-batch",
        "--q-f32",
        str(q_path),
        "--k-f32",
        str(k_path),
        "--output-q-f32",
        str(out_q),
        "--output-k-f32",
        str(out_k),
        "--num-heads",
        str(num_heads),
        "--rope-dim",
        str(rope_dim),
        "--start-position",
        str(start),
        "--batch-tokens",
        "2",
        "--rope-theta",
        "10000",
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
    got_q = read_f32(out_q, len(q))
    got_k = read_f32(out_k, len(k))
    exp_q = expected_q(q, num_heads=num_heads, rope_dim=rope_dim, start=start, interleave=interleave)
    exp_k = expected_k(k, rope_dim=rope_dim, start=start, interleave=interleave)
    if any(abs(a - b) > 1e-5 for a, b in zip(got_q, exp_q)):
        raise SystemExit(f"batch q interleave={interleave} {got_q} != {exp_q}")
    if any(abs(a - b) > 1e-5 for a, b in zip(got_k, exp_k)):
        raise SystemExit(f"batch k interleave={interleave} {got_k} != {exp_k}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-rope-batch-", dir="/private/tmp"))
    run_case(root, interleave=False)
    run_case(root, interleave=True)
    print(f"fixture: {root}")
    print("  rope batch smoke:   ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
