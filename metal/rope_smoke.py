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


def run_case(root: Path, interleave: bool) -> None:
    q = [1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0]
    k = [1.0, -1.0, 2.0, -2.0]
    position = 3
    theta = 10000.0
    q_path = root / ("q_interleave.f32" if interleave else "q_default.f32")
    k_path = root / ("k_interleave.f32" if interleave else "k_default.f32")
    out_q = root / ("out_q_interleave.f32" if interleave else "out_q_default.f32")
    out_k = root / ("out_k_interleave.f32" if interleave else "out_k_default.f32")
    write_f32(q_path, q)
    write_f32(k_path, k)
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--run-rope",
        "--q-f32",
        str(q_path),
        "--k-f32",
        str(k_path),
        "--output-q-f32",
        str(out_q),
        "--output-k-f32",
        str(out_k),
        "--num-heads",
        "2",
        "--rope-dim",
        "4",
        "--position",
        str(position),
        "--rope-theta",
        str(theta),
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

    expected_q = rotate(q[:4], position, theta, interleave) + rotate(q[4:], position, theta, interleave)
    expected_k = rotate(k, position, theta, interleave)
    got_q = read_f32(out_q, 8)
    got_k = read_f32(out_k, 4)
    if any(abs(a - b) > 1e-5 for a, b in zip(got_q, expected_q)):
        raise SystemExit(f"q interleave={interleave} {got_q} != {expected_q}")
    if any(abs(a - b) > 1e-5 for a, b in zip(got_k, expected_k)):
        raise SystemExit(f"k interleave={interleave} {got_k} != {expected_k}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-rope-", dir="/private/tmp"))
    run_case(root, interleave=False)
    run_case(root, interleave=True)
    print(f"fixture: {root}")
    print("  rope smoke:         ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
