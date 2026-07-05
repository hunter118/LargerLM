#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


TOKENS = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
NORM_WEIGHTS = (1.0, 2.0, 0.5)
Q_A = (
    (1.0, 1.0, 1.0),
    (2.0, 0.0, 1.0),
)
KV_A = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)


def pack_matrix(rows: tuple[tuple[float, ...], ...]) -> bytes:
    values = tuple(value for row in rows for value in row)
    return struct.pack(f"<{len(values)}f", *values)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    q_a = pack_matrix(Q_A)
    kv_a = pack_matrix(KV_A)
    norm = struct.pack("<3f", *NORM_WEIGHTS)
    payload = q_a + kv_a + norm
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
                        "name": "model.layers.1.self_attn.q_a_proj.weight",
                        "offset": 0,
                        "size": len(q_a),
                        "dtype": "F32",
                        "shape": [len(Q_A), len(Q_A[0])],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                        "offset": len(q_a),
                        "size": len(kv_a),
                        "dtype": "F32",
                        "shape": [len(KV_A), len(KV_A[0])],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.input_layernorm.weight",
                        "offset": len(q_a) + len(kv_a),
                        "size": len(norm),
                        "dtype": "F32",
                        "shape": [len(NORM_WEIGHTS)],
                        "category": "attention",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    values = tuple(value for token in TOKENS for value in token)
    (root / "input.f32").write_bytes(struct.pack(f"<{len(values)}f", *values))


def matmul_rows(rows: tuple[tuple[float, ...], ...], token: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sum(a * b for a, b in zip(row, token)) for row in rows)


def reference() -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    norm_values: list[float] = []
    q_a_values: list[float] = []
    kv_a_values: list[float] = []
    for token in TOKENS:
        inv = 1.0 / math.sqrt(sum(v * v for v in token) / len(token))
        normed = tuple(v * inv * w for v, w in zip(token, NORM_WEIGHTS))
        norm_values.extend(normed)
        q_a_values.extend(matmul_rows(Q_A, normed))
        kv_a_values.extend(matmul_rows(KV_A, normed))
    return tuple(norm_values), tuple(q_a_values), tuple(kv_a_values)


def check_f32(path: Path, expected: tuple[float, ...], label: str) -> None:
    got = struct.unpack(f"<{len(expected)}f", path.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"{label} output {got} != expected {expected}")


def run_cli_case(root: Path) -> None:
    output_dir = root / "prefix"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-attention-prefix",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "2",
        "--rms-norm-eps",
        "0",
        "--output-dir",
        str(output_dir),
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--quiet-runner",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    if (
        payload["batch_tokens"] != 2
        or payload["hidden_dim"] != 3
        or payload["q_a_output_bytes"] != 16
        or payload["kv_a_output_bytes"] != 32
    ):
        raise SystemExit(f"unexpected CLI payload: {payload}")
    norm, q_a, kv_a = reference()
    check_f32(output_dir / "input_layernorm.f32", norm, "input_layernorm")
    check_f32(output_dir / "q_a_proj.f32", q_a, "q_a_proj")
    check_f32(output_dir / "kv_a_proj_with_mqa.f32", kv_a, "kv_a_proj_with_mqa")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-attention-prefix-", dir="/private/tmp"))
    write_fixture(root)
    run_cli_case(root)
    print(f"fixture: {root}")
    print("  prefill attention prefix: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
