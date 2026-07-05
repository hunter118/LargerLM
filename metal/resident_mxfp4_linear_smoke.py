#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 32
OUT_DIM = 32
GROUP_SIZE = 32
SCALE_E8M0_ONE = 127


def pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    packed_one = pack8(2)
    weights = bytearray()
    for _row in range(OUT_DIM):
        for _col in range(HIDDEN_DIM // 8):
            weights.extend(struct.pack("<I", packed_one))
    scales = bytes([SCALE_E8M0_ONE]) * (OUT_DIM * (HIDDEN_DIM // GROUP_SIZE))
    payload = bytes(weights) + scales
    scale_offset = len(weights)
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
                        "name": "model.layers.1.self_attn.q_mxfp4_proj.weight",
                        "offset": 0,
                        "size": len(weights),
                        "dtype": "U32",
                        "shape": [OUT_DIM, HIDDEN_DIM // 8],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_mxfp4_proj.scales",
                        "offset": scale_offset,
                        "size": len(scales),
                        "dtype": "U8",
                        "shape": [OUT_DIM, HIDDEN_DIM // GROUP_SIZE],
                        "category": "attention",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row0 = [1.0 / HIDDEN_DIM] * HIDDEN_DIM
    row1 = [0.5 / HIDDEN_DIM] * HIDDEN_DIM
    (root / "input.f32").write_bytes(struct.pack("<32f", *row0))
    (root / "batch_input.f32").write_bytes(struct.pack("<64f", *(row0 + row1)))


def run_single(runner: Path, root: Path) -> None:
    output = root / "single_out.f32"
    cmd = [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-resident-linear",
        "--tensor-suffix",
        ".self_attn.q_mxfp4_proj.weight",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(output),
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
    if "dtype:              mlx-mxfp4" not in completed.stdout:
        raise SystemExit("runner did not report resident mlx-mxfp4 dtype")
    if "group size:         32" not in completed.stdout:
        raise SystemExit("runner did not report resident MXFP4 group size")
    values = struct.unpack("<32f", output.read_bytes())
    if abs(values[0] - 1.0) > 2e-4:
        raise SystemExit(f"single output[0] {values[0]:.6f} != expected 1.0")


def run_batch(runner: Path, root: Path) -> None:
    output = root / "batch_out.f32"
    cmd = [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-resident-linear-batch",
        "--tensor-suffix",
        ".self_attn.q_mxfp4_proj.weight",
        "--input-f32",
        str(root / "batch_input.f32"),
        "--batch-tokens",
        "2",
        "--output-f32",
        str(output),
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
    if "dtype:              mlx-mxfp4" not in completed.stdout:
        raise SystemExit("runner did not report resident batch mlx-mxfp4 dtype")
    values = struct.unpack("<64f", output.read_bytes())
    if abs(values[0] - 1.0) > 2e-4 or abs(values[OUT_DIM] - 0.5) > 2e-4:
        raise SystemExit(
            f"batch outputs {values[0]:.6f}, {values[OUT_DIM]:.6f} "
            "!= expected 1.0, 0.5"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resident MXFP4 linear smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-resident-mxfp4-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    run_single(args.runner, root)
    run_batch(args.runner, root)
    print(f"fixture: {root}")
    print("  resident MXFP4 linear: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
