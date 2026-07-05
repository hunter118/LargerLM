#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 32
GROUP_SIZE = 32
SCALE_E8M0_ONE = 127


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def mxfp4_matrix(out_dim: int, in_dim: int, code: int = 2) -> tuple[bytes, bytes]:
    weights = bytearray()
    packed = pack8(code)
    for _row in range(out_dim):
        for _col in range(in_dim // 8):
            weights.extend(struct.pack("<I", packed))
    scales = bytes([SCALE_E8M0_ONE]) * (out_dim * (in_dim // GROUP_SIZE))
    return bytes(weights), scales


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, dtype: str, shape: list[int], data: bytes, category: str) -> None:
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": category,
            }
        )
        payload.extend(data)

    for suffix in (
        "input_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
    ):
        add(f"model.layers.1.{suffix}", "F32", [HIDDEN_DIM], f32([1.0] * HIDDEN_DIM), "norms")

    for suffix in (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    ):
        weights, scales = mxfp4_matrix(HIDDEN_DIM, HIDDEN_DIM)
        stem = f"model.layers.1.self_attn.{suffix}"
        add(
            f"{stem}.weight",
            "U32",
            [HIDDEN_DIM, HIDDEN_DIM // 8],
            weights,
            "attention",
        )
        add(
            f"{stem}.scales",
            "U8",
            [HIDDEN_DIM, HIDDEN_DIM // GROUP_SIZE],
            scales,
            "attention",
        )

    (resident / "resident.bin").write_bytes(bytes(payload))
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(f32([1.0 / HIDDEN_DIM] * HIDDEN_DIM))
    (root / "attn_input.f32").write_bytes(f32([1.0 / HIDDEN_DIM] * HIDDEN_DIM))
    (root / "residual.f32").write_bytes(f32([0.25] * HIDDEN_DIM))


def run_checked(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def run_smoke(runner: Path, root: Path) -> None:
    layout = root / "resident" / "layout.json"
    out_dir = root / "attn"
    stdout = run_checked(
        [
            str(runner),
            "--resident-layout",
            str(layout),
            "--layer",
            "1",
            "--run-attn-projections",
            "--input-f32",
            str(root / "input.f32"),
            "--output-dir",
            str(out_dir),
            "--rms-norm-eps",
            "0",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    if "q lora dim:         32" not in stdout:
        raise SystemExit("attention projections did not use logical MXFP4 dims")
    checks = {
        "attn_q_a.f32": 32.0,
        "attn_q_b.f32": 32.0,
        "attn_kv_a.f32": 32.0,
        "attn_kv_b.f32": 32.0,
    }
    for name, expected in checks.items():
        value = read_f32(out_dir / name, HIDDEN_DIM)[0]
        if abs(value - expected) > 2e-4:
            raise SystemExit(f"{name} value {value:.6f} != expected {expected:.6f}")

    out_path = root / "attn_out.f32"
    stdout = run_checked(
        [
            str(runner),
            "--resident-layout",
            str(layout),
            "--layer",
            "1",
            "--run-attn-output",
            "--input-f32",
            str(root / "attn_input.f32"),
            "--residual-f32",
            str(root / "residual.f32"),
            "--output-f32",
            str(out_path),
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )
    if "dtype:              mlx-mxfp4" not in stdout:
        raise SystemExit("attention output did not report MXFP4 dtype")
    value = read_f32(out_path, HIDDEN_DIM)[0]
    if abs(value - 1.25) > 2e-4:
        raise SystemExit(f"attention output {value:.6f} != expected 1.25")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resident MXFP4 attention smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-mxfp4-attn-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    run_smoke(args.runner, root)
    print(f"fixture: {root}")
    print("  resident MXFP4 attention: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
