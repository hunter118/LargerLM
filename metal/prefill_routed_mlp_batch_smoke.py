#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from mlp_block_smoke import write_experts, write_resident


def expert_output(value: float, normed: float) -> float:
    gate = 8.0 * value * normed
    up = 8.0 * value * normed
    act = (gate / (1.0 + math.exp(-gate))) * up
    return 8.0 * value * act


def expected_row(input_value: float, eps: float) -> float:
    normed = input_value / math.sqrt(input_value * input_value + eps)
    routed = (2.0 / 3.0) * expert_output(2.0, normed) + (
        1.0 / 3.0
    ) * expert_output(1.0, normed)
    return input_value + routed + expert_output(1.0, normed)


def main() -> int:
    eps = 1e-5
    root = Path(tempfile.mkdtemp(prefix="largerlm-prefill-routed-mlp-", dir="/private/tmp"))
    write_experts(root)
    write_resident(root)
    values = [1.0] * 8 + [0.5] * 8
    (root / "input.f32").write_bytes(struct.pack("<16f", *values))
    output = root / "routed_batch.f32"
    output_dir = root / "routed_mlp"
    router_json_dir = root / "router_json"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-routed-mlp-block-batch",
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
        "--max-slot-mib",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--expert-read-advise-align-kib",
        "4",
        "--router-json-dir",
        str(router_json_dir),
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
        or payload["command_count"] != 2
        or payload["read_bytes"] != 2 * 2 * 192
    ):
        raise SystemExit(f"unexpected routed MLP payload: {payload}")
    got = struct.unpack("<16f", output.read_bytes())
    expected = (expected_row(1.0, eps), expected_row(0.5, eps))
    if abs(got[0] - expected[0]) > 3e-3 or abs(got[8] - expected[1]) > 3e-3:
        raise SystemExit(
            f"outputs {got[0]:.6f}, {got[8]:.6f} != expected "
            f"{expected[0]:.6f}, {expected[1]:.6f}"
        )
    router_files = sorted(path.name for path in router_json_dir.glob("*.json"))
    if router_files != ["token_000000.router.json", "token_000001.router.json"]:
        raise SystemExit(f"unexpected router JSON files: {router_files}")
    print(f"fixture: {root}")
    print("  prefill routed MLP batch: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
