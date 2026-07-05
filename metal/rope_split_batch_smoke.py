#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def split_q_b(
    values: tuple[float, ...],
    *,
    batch: int,
    heads: int,
    qk_nope_dim: int,
    rope_dim: int,
) -> tuple[list[float], list[float]]:
    head_dim = qk_nope_dim + rope_dim
    q_nope: list[float] = []
    q_rope: list[float] = []
    for token in range(batch):
        row_base = token * heads * head_dim
        for head in range(heads):
            base = row_base + head * head_dim
            q_nope.extend(values[base : base + qk_nope_dim])
            q_rope.extend(values[base + qk_nope_dim : base + head_dim])
    return q_nope, q_rope


def run_checked(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def run_server_checked(cmd: list[str], request: dict[str, object]) -> str:
    payload = (
        json.dumps(request, separators=(",", ":"))
        + "\n"
        + json.dumps({"command": "quit"}, separators=(",", ":"))
        + "\n"
    )
    completed = subprocess.run(cmd, input=payload, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def assert_close(lhs: tuple[float, ...], rhs: tuple[float, ...], *, label: str) -> None:
    if len(lhs) != len(rhs):
        raise SystemExit(f"{label} length mismatch: {len(lhs)} != {len(rhs)}")
    max_diff = max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)
    if max_diff > 1e-6:
        raise SystemExit(f"{label} max diff {max_diff} exceeds tolerance")


def run_smoke(runner: Path, root: Path) -> None:
    batch = 3
    heads = 2
    qk_nope_dim = 2
    rope_dim = 4
    q_b_values = [math.sin(i * 0.17) for i in range(batch * heads * (qk_nope_dim + rope_dim))]
    k_values = [math.cos(i * 0.13) for i in range(batch * rope_dim)]
    q_b = root / "q_b.f32"
    k_rope = root / "k_rope.f32"
    q_rope_old = root / "q_rope_old.f32"
    q_rope_rot_old = root / "q_rope_rot_old.f32"
    k_rope_rot_old = root / "k_rope_rot_old.f32"
    q_nope_new = root / "q_nope_new.f32"
    q_rope_new = root / "q_rope_new.f32"
    q_rope_rot_new = root / "q_rope_rot_new.f32"
    k_rope_rot_new = root / "k_rope_rot_new.f32"
    q_nope_server = root / "q_nope_server.f32"
    q_rope_server = root / "q_rope_server.f32"
    q_rope_rot_server = root / "q_rope_rot_server.f32"
    k_rope_rot_server = root / "k_rope_rot_server.f32"
    q_b.write_bytes(f32(q_b_values))
    k_rope.write_bytes(f32(k_values))
    q_nope_expected, q_rope_expected = split_q_b(
        tuple(q_b_values),
        batch=batch,
        heads=heads,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
    )
    q_rope_old.write_bytes(f32(q_rope_expected))

    common = [
        "--num-heads",
        str(heads),
        "--rope-dim",
        str(rope_dim),
        "--start-position",
        "5",
        "--batch-tokens",
        str(batch),
        "--rope-theta",
        "10000",
        "--rope-interleave",
        "--max-runner-scratch-mib",
        "1",
    ]
    run_checked(
        [
            str(runner),
            "--run-rope-batch",
            "--q-f32",
            str(q_rope_old),
            "--k-f32",
            str(k_rope),
            "--output-q-f32",
            str(q_rope_rot_old),
            "--output-k-f32",
            str(k_rope_rot_old),
            *common,
        ]
    )
    run_checked(
        [
            str(runner),
            "--run-rope-split-batch",
            "--q-b-f32",
            str(q_b),
            "--k-f32",
            str(k_rope),
            "--output-q-nope-f32",
            str(q_nope_new),
            "--output-q-rope-f32",
            str(q_rope_new),
            "--output-q-f32",
            str(q_rope_rot_new),
            "--output-k-f32",
            str(k_rope_rot_new),
            "--qk-nope-dim",
            str(qk_nope_dim),
            *common,
        ]
    )
    assert_close(read_f32(q_nope_new), tuple(q_nope_expected), label="q_nope")
    assert_close(read_f32(q_rope_new), tuple(q_rope_expected), label="q_rope")
    assert_close(read_f32(q_rope_rot_new), read_f32(q_rope_rot_old), label="q_rotated")
    assert_close(read_f32(k_rope_rot_new), read_f32(k_rope_rot_old), label="k_rotated")
    run_server_checked(
        [
            str(runner),
            "--run-rope-split-batch-server-jsonl",
        ],
        {
            "q_b_f32": str(q_b),
            "k_f32": str(k_rope),
            "output_q_nope_f32": str(q_nope_server),
            "output_q_rope_f32": str(q_rope_server),
            "output_q_f32": str(q_rope_rot_server),
            "output_k_f32": str(k_rope_rot_server),
            "num_heads": heads,
            "qk_nope_dim": qk_nope_dim,
            "rope_dim": rope_dim,
            "start_position": 5,
            "batch_tokens": batch,
            "rope_theta": 10000.0,
            "rope_interleave": True,
            "max_runner_scratch_mib": 1,
        },
    )
    assert_close(read_f32(q_nope_server), tuple(q_nope_expected), label="q_nope_server")
    assert_close(read_f32(q_rope_server), tuple(q_rope_expected), label="q_rope_server")
    assert_close(
        read_f32(q_rope_rot_server),
        read_f32(q_rope_rot_old),
        label="q_rotated_server",
    )
    assert_close(
        read_f32(k_rope_rot_server),
        read_f32(k_rope_rot_old),
        label="k_rotated_server",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoPE split batch smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-rope-split-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    run_smoke(args.runner, root)
    print(f"fixture: {root}")
    print("  rope split batch: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
