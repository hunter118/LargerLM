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
NORM = (1.0, 2.0, 0.5)
GATE = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
UP = (
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)
DOWN = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
)


def pack(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def pack_matrix(rows: tuple[tuple[float, ...], ...]) -> bytes:
    return pack(tuple(value for row in rows for value in row))


def add_blob(
    tensors: list[dict[str, object]],
    payload: bytearray,
    name: str,
    shape: list[int],
    data: bytes,
) -> None:
    tensors.append(
        {
            "name": name,
            "offset": len(payload),
            "size": len(data),
            "dtype": "F32",
            "shape": shape,
            "category": "dense_mlp" if len(shape) == 2 else "norms",
        }
    )
    payload.extend(data)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []
    add_blob(
        tensors,
        payload,
        "model.layers.0.post_attention_layernorm.weight",
        [len(NORM)],
        pack(NORM),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.0.mlp.switch_mlp.gate_proj.weight",
        [len(GATE), len(GATE[0])],
        pack_matrix(GATE),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.0.mlp.switch_mlp.up_proj.weight",
        [len(UP), len(UP[0])],
        pack_matrix(UP),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.0.mlp.switch_mlp.down_proj.weight",
        [len(DOWN), len(DOWN[0])],
        pack_matrix(DOWN),
    )
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
                "tensors": tensors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    values = tuple(value for token in TOKENS for value in token)
    (root / "input.f32").write_bytes(pack(values))


def rmsnorm(token: tuple[float, ...], weights: tuple[float, ...]) -> tuple[float, ...]:
    inv = 1.0 / math.sqrt(sum(value * value for value in token) / len(token))
    return tuple(value * inv * weight for value, weight in zip(token, weights))


def matmul(rows: tuple[tuple[float, ...], ...], token: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sum(left * right for left, right in zip(row, token)) for row in rows)


def silu(value: float) -> float:
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def reference() -> tuple[float, ...]:
    out: list[float] = []
    for token in TOKENS:
        normed = rmsnorm(token, NORM)
        gate = matmul(GATE, normed)
        up = matmul(UP, normed)
        activated = tuple(silu(gate_value) * up_value for gate_value, up_value in zip(gate, up))
        down = matmul(DOWN, activated)
        out.extend(left + right for left, right in zip(down, token))
    return tuple(out)


def check_f32(path: Path, expected: tuple[float, ...]) -> None:
    got = struct.unpack(f"<{len(expected)}f", path.read_bytes())
    if any(abs(left - right) > 1e-4 for left, right in zip(got, expected)):
        raise SystemExit(f"{path.name} output {got} != expected {expected}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-prefill-dense-mlp-", dir="/private/tmp"))
    write_fixture(root)
    output = root / "dense_out.f32"
    output_dir = root / "dense_mlp"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-dense-mlp-block-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "0",
        "--input-f32",
        str(root / "input.f32"),
        "--output-dir",
        str(output_dir),
        "--output-f32",
        str(output),
        "--batch-tokens",
        "2",
        "--rms-norm-eps",
        "0",
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
        payload["hidden_dim"] != 3
        or payload["intermediate_dim"] != 2
        or payload["swiglu_output_bytes"] != 16
        or payload["output_bytes"] != 24
        or payload["gate_proj"]["tensor_suffix"] != ".mlp.switch_mlp.gate_proj.weight"
    ):
        raise SystemExit(f"unexpected dense MLP payload: {payload}")
    check_f32(output, reference())
    print(f"fixture: {root}")
    print("  prefill dense MLP batch: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
