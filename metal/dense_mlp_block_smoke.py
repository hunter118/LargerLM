#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def pack8(value: int) -> int:
    out = 0
    for i in range(8):
        out |= (value & 0xF) << (i * 4)
    return out


def bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


def write_fixture(root: Path, *, affine_dense: bool = False) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    norm = f32([1.0] * 8)
    dense = f32([1.0] * 64)
    payload = bytearray()
    tensors: list[dict] = []

    def add(
        name: str,
        data: bytes,
        shape: list[int],
        category: str,
        *,
        dtype: str = "F32",
    ) -> None:
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

    add("model.layers.0.post_attention_layernorm.weight", norm, [8], "norms")
    for component in ("gate_proj", "up_proj", "down_proj"):
        stem = f"model.layers.0.mlp.switch_mlp.{component}"
        if affine_dense:
            add(
                f"{stem}.weight",
                struct.pack("<8I", *([pack8(1)] * 8)),
                [8, 1],
                "dense_mlp",
                dtype="U32",
            )
            add(
                f"{stem}.scales",
                bf16(1.0) * 8,
                [8, 1],
                "dense_mlp",
                dtype="BF16",
            )
            add(
                f"{stem}.biases",
                bf16(0.0) * 8,
                [8, 1],
                "dense_mlp",
                dtype="BF16",
            )
        else:
            add(
                f"{stem}.weight",
                dense,
                [8, 8],
                "dense_mlp",
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
    (root / "input.f32").write_bytes(f32([1.0] * 8))


def dense_output(normed: float) -> float:
    gate = 8.0 * normed
    up = 8.0 * normed
    act = (gate / (1.0 + math.exp(-gate))) * up
    return 8.0 * act


def run_case(root: Path, *, affine_dense: bool) -> None:
    eps = 1e-5
    write_fixture(root, affine_dense=affine_dense)
    output = root / "output.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "0",
        "--run-dense-mlp-block",
        "--input-f32",
        str(root / "input.f32"),
        "--rms-norm-eps",
        str(eps),
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

    got = struct.unpack("<8f", output.read_bytes())[0]
    normed = 1.0 / math.sqrt(1.0 + eps)
    expected = 1.0 + dense_output(normed)
    if abs(got - expected) > 3e-3:
        raise SystemExit(f"output[0] {got:.6f} != expected {expected:.6f}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-dense-mlp-block-", dir="/private/tmp"))
    run_case(root / "f32", affine_dense=False)
    run_case(root / "affine", affine_dense=True)
    eps = 1e-5
    normed = 1.0 / math.sqrt(1.0 + eps)
    expected = 1.0 + dense_output(normed)
    print(f"fixture: {root}")
    print(f"  expected output[0]: {expected:.6f}")
    print("  affine dense mlp:   ok")
    print("  dense mlp smoke:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
