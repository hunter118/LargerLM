#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from mlp_block_smoke import expert_output, write_experts, write_resident


def expected_row(input_value: float, eps: float) -> float:
    normed = input_value / math.sqrt(input_value * input_value + eps)
    routed = (2.0 / 3.0) * expert_output(2.0, normed) + (
        1.0 / 3.0
    ) * expert_output(1.0, normed)
    return input_value + routed + expert_output(1.0, normed)


def main() -> int:
    eps = 1e-5
    root = Path(tempfile.mkdtemp(prefix="largerlm-prefill-staged-mlp-", dir="/private/tmp"))
    write_experts(root)
    write_resident(root)
    values = [1.0] * 8 + [0.5] * 8
    (root / "input.f32").write_bytes(struct.pack("<16f", *values))
    output = root / "staged_routed_mlp.f32"
    output_dir = root / "staged_routed_mlp"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-staged-routed-mlp-block-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "experts" / "layout.json"),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--input-f32",
        str(root / "input.f32"),
        "--output-dir",
        str(output_dir),
        "--output-f32",
        str(output),
        "--batch-tokens",
        "2",
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
        str(eps),
        "--max-resident-matrix-mib",
        "1",
        "--max-slot-mib",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--max-stage-mib",
        "1",
        "--max-compact-stage-mib",
        "1",
        "--copy-chunk-mib",
        "0.0001",
        "--prefill-linear-backend",
        "mpsgraph-f32",
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
        or payload["hidden_dim"] != 8
        or payload["output_bytes"] != 64
        or payload["router_command_count"] != 1
        or payload["routed_command_count"] != 1
        or "--run-resident-linear-batch" not in payload["first_router_command"]
        or ".mlp.gate.weight" not in payload["first_router_command"]
        or not payload["include_shared_expert"]
        or payload["staged_bytes"] != 384
        or payload["compact_stage_bytes"] != 384
        or payload["shared_output_bytes"] != 64
        or payload["shared_gate_proj"]["backend"] != "mpsgraph-f32"
        or payload["shared_up_proj"]["backend"] != "mpsgraph-f32"
        or payload["shared_down_proj"]["backend"] != "mpsgraph-f32"
    ):
        raise SystemExit(f"unexpected staged prefill MLP payload: {payload}")
    got = struct.unpack("<16f", output.read_bytes())
    expected = (expected_row(1.0, eps), expected_row(0.5, eps))
    if abs(got[0] - expected[0]) > 3e-3 or abs(got[8] - expected[1]) > 3e-3:
        raise SystemExit(
            f"outputs {got[0]:.6f}, {got[8]:.6f} != expected "
            f"{expected[0]:.6f}, {expected[1]:.6f}"
        )
    print(f"fixture: {root}")
    print("  prefill staged routed MLP batch: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
