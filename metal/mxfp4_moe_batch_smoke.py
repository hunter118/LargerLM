#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

from mxfp4_layer_moe_smoke import HIDDEN_DIM, fp4_code_to_value, write_fixture


def expert_output_for_code(code: int, input_sum: float) -> float:
    value = fp4_code_to_value(code)
    gate = value * input_sum
    up = value * input_sum
    act = (gate / (1.0 + math.exp(-gate))) * up
    return HIDDEN_DIM * value * act


def expected_row(input_sum: float) -> float:
    expert0 = expert_output_for_code(1, input_sum)
    expert1 = expert_output_for_code(2, input_sum)
    return (2.0 / 3.0) * expert1 + (1.0 / 3.0) * expert0


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-mxfp4-moe-batch-", dir="/private/tmp"))
    write_fixture(root)
    routes_json = root / "routes.json"
    input_f32 = root / "batch_input.f32"
    output_f32 = root / "batch_output.f32"
    routes_json.write_text(
        json.dumps(
            {
                "batch_tokens": 2,
                "routes": [
                    {"experts": [1, 0], "weights": [2.0 / 3.0, 1.0 / 3.0]},
                    {"experts": [1, 0], "weights": [2.0 / 3.0, 1.0 / 3.0]},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row0 = [1.0 / HIDDEN_DIM] * HIDDEN_DIM
    row1 = [0.5 / HIDDEN_DIM] * HIDDEN_DIM
    input_f32.write_bytes(struct.pack("<64f", *(row0 + row1)))
    runner = Path(__file__).with_name("largerlm-runner")
    completed = subprocess.run(
        [
            str(runner),
            "--layout",
            str(root / "experts" / "layout.json"),
            "--layer",
            "1",
            "--run-moe-batch",
            "--routes-json",
            str(routes_json),
            "--input-f32",
            str(input_f32),
            "--batch-tokens",
            "2",
            "--output-f32",
            str(output_f32),
            "--max-k",
            "2",
            "--max-slot-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--expert-read-advise-merge-gap-kib",
            "0",
            "--expert-read-advise-align-kib",
            "4",
        ],
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        return completed.returncode
    if "quantization:       mlx-mxfp4" not in completed.stdout:
        raise SystemExit("runner did not report mlx-mxfp4 batch execution")
    if "used compact experts: 2" not in completed.stdout:
        raise SystemExit("runner did not report two used compact experts")
    if "route assignments:  4" not in completed.stdout:
        raise SystemExit("runner did not report four route assignments")
    values = struct.unpack("<64f", output_f32.read_bytes())
    expected = (expected_row(1.0), expected_row(0.5))
    if abs(values[0] - expected[0]) > 5e-3 or abs(values[HIDDEN_DIM] - expected[1]) > 5e-3:
        raise SystemExit(
            f"outputs {values[0]:.6f}, {values[HIDDEN_DIM]:.6f} != expected "
            f"{expected[0]:.6f}, {expected[1]:.6f}"
        )
    print(f"fixture: {root}")
    print("  MXFP4 MoE batch smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
