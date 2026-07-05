#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def pack8(value: int) -> int:
    out = 0
    for i in range(8):
        out |= (value & 0xF) << (i * 4)
    return out


def write_resident(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []
    for component, nibble in (
        ("gate_proj", 1),
        ("up_proj", 2),
        ("down_proj", 1),
    ):
        stem = f"model.layers.1.mlp.shared_experts.{component}"
        weight = struct.pack("<8I", *([pack8(nibble)] * 8))
        scales = bytes([127]) * 8
        for suffix, data, dtype, shape in (
            ("weight", weight, "U32", [8, 1]),
            ("scales", scales, "U8", [8, 1]),
        ):
            tensors.append(
                {
                    "name": f"{stem}.{suffix}",
                    "offset": len(payload),
                    "size": len(data),
                    "dtype": dtype,
                    "shape": shape,
                    "category": "shared_experts",
                }
            )
            payload.extend(data)
    (resident / "resident.bin").write_bytes(bytes(payload))
    layout = resident / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return layout


def run(cmd: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        input=stdin,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout[-4000:], end="")
        if completed.stderr:
            print(completed.stderr[-4000:], end="")
        raise SystemExit(completed.returncode)
    return completed


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack("<" + "f" * (len(raw) // 4), raw)


def max_abs_diff(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-shared-expert-batch-", dir="/private/tmp"))
    runner = Path(__file__).with_name("largerlm-runner")
    layout = write_resident(root)
    input_path = root / "input.f32"
    values = [math.sin(i + 1) * 0.25 for i in range(16)]
    input_path.write_bytes(struct.pack("<16f", *values))
    one_shot = root / "one_shot.f32"
    server_out = root / "server.f32"
    base = [
        str(runner),
        "--resident-layout",
        str(layout),
        "--layer",
        "1",
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        "2",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "8",
    ]
    run(
        base
        + [
            "--run-shared-expert-batch",
            "--output-f32",
            str(one_shot),
        ]
    )
    request = {
        "resident_layout": str(layout),
        "layer": 1,
        "input_f32": str(input_path),
        "output_f32": str(server_out),
        "batch_tokens": 2,
        "max_resident_matrix_mib": 1,
        "max_runner_scratch_mib": 8,
    }
    run(
        [str(runner), "--run-shared-expert-batch-server-jsonl"],
        stdin=json.dumps(request, separators=(",", ":"))
        + "\n"
        + json.dumps({"command": "quit"}, separators=(",", ":"))
        + "\n",
    )
    diff = max_abs_diff(read_f32(one_shot), read_f32(server_out))
    if diff > 1e-6:
        raise SystemExit(f"shared expert server diff too large: {diff}")
    print(f"fixture: {root}")
    print(f"  max abs diff:        {diff:.9g}")
    print("  shared expert batch: ok")
    print("  shared expert server: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
