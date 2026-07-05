#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    norm = struct.pack("<3f", 1.0, 2.0, 0.5)
    (resident / "resident.bin").write_bytes(norm)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(norm),
                "tensors": [
                    {
                        "name": "model.layers.1.input_layernorm.weight",
                        "offset": 0,
                        "size": len(norm),
                        "dtype": "F32",
                        "shape": [3],
                        "category": "attention",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))


def reference() -> tuple[float, ...]:
    weights = (1.0, 2.0, 0.5)
    values: list[float] = []
    for token in ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)):
        inv = 1.0 / math.sqrt(sum(v * v for v in token) / len(token))
        values.extend(v * inv * w for v, w in zip(token, weights))
    return tuple(values)


def check_output(path: Path, expected: tuple[float, ...]) -> None:
    got = struct.unpack("<6f", path.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"RMSNorm output {got} != expected {expected}")


def run_runner_case(root: Path, expected: tuple[float, ...]) -> None:
    output = root / "runner_norm.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-rmsnorm-batch",
        "--norm-suffix",
        ".input_layernorm.weight",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "2",
        "--rms-norm-eps",
        "0",
        "--output-f32",
        str(output),
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
    check_output(output, expected)


def run_server_case(root: Path, expected: tuple[float, ...]) -> None:
    output = root / "server_norm.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--run-rmsnorm-batch-server-jsonl",
    ]
    request = {
        "resident_layout": str(root / "resident" / "layout.json"),
        "layer": 1,
        "norm_suffix": ".input_layernorm.weight",
        "input_f32": str(root / "input.f32"),
        "batch_tokens": 2,
        "rms_norm_eps": 0,
        "output_f32": str(output),
        "max_runner_scratch_mib": 64,
    }
    stdin = (
        json.dumps(request, separators=(",", ":"))
        + "\n"
        + json.dumps({"command": "quit"}, separators=(",", ":"))
        + "\n"
    )
    completed = subprocess.run(cmd, input=stdin, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if "server request:      ok" not in completed.stdout:
        raise SystemExit("RMSNorm server did not acknowledge request")
    check_output(output, expected)


def run_cli_case(root: Path, expected: tuple[float, ...]) -> None:
    output = root / "cli_norm.f32"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-rmsnorm-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--norm-suffix",
        ".input_layernorm.weight",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "2",
        "--rms-norm-eps",
        "0",
        "--output-f32",
        str(output),
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
    if payload["batch_tokens"] != 2 or payload["output_bytes"] != 24:
        raise SystemExit(f"unexpected CLI payload: {payload}")
    check_output(output, expected)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-rmsnorm-batch-", dir="/private/tmp"))
    write_fixture(root)
    expected = reference()
    run_runner_case(root, expected)
    run_server_case(root, expected)
    run_cli_case(root, expected)
    print(f"fixture: {root}")
    print("  resident batch RMSNorm: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
