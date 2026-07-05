#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def pack_int4(values: tuple[int, ...]) -> bytes:
    if len(values) != 8:
        raise AssertionError(values)
    packed = 0
    for index, value in enumerate(values):
        if value < 0 or value > 15:
            raise AssertionError(values)
        packed |= value << (index * 4)
    return struct.pack("<I", packed)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    q_a = struct.pack("<6f", 1.0, 1.0, 1.0, 2.0, 0.0, 1.0)
    q_b_values = [1.0, 2.0, 3.0, 0.5, 0.5, 0.5]
    q_b = b"".join(f32_to_bf16(v) for v in q_b_values)
    q_c_w = pack_int4((1, 2, 3, 4, 0, 0, 0, 0)) + pack_int4((0, 1, 0, 1, 0, 1, 0, 1))
    q_c_s = f32_to_bf16(1.0) + f32_to_bf16(1.0)
    q_c_b = f32_to_bf16(0.0) + f32_to_bf16(0.0)
    payload = q_a + q_b + q_c_w + q_c_s + q_c_b
    q_c_offset = len(q_a) + len(q_b)
    q_c_s_offset = q_c_offset + len(q_c_w)
    q_c_b_offset = q_c_s_offset + len(q_c_s)
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
                        "shape": [2, 3],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_b_proj.weight",
                        "offset": len(q_a),
                        "size": len(q_b),
                        "dtype": "BF16",
                        "shape": [2, 3],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_c_proj.weight",
                        "offset": q_c_offset,
                        "size": len(q_c_w),
                        "dtype": "U32",
                        "shape": [2, 1],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_c_proj.scales",
                        "offset": q_c_s_offset,
                        "size": len(q_c_s),
                        "dtype": "BF16",
                        "shape": [2, 1],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_c_proj.biases",
                        "offset": q_c_b_offset,
                        "size": len(q_c_b),
                        "dtype": "BF16",
                        "shape": [2, 1],
                        "category": "attention",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(struct.pack("<3f", 1.0, 2.0, 3.0))
    (root / "input8.f32").write_bytes(
        struct.pack("<8f", 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0)
    )


def run_case(
    root: Path,
    suffix: str,
    expected: tuple[float, float],
    *,
    input_name: str = "input.f32",
) -> None:
    output = root / f"{suffix.split('.')[-3]}_out.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-resident-linear",
        "--tensor-suffix",
        suffix,
        "--input-f32",
        str(root / input_name),
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
    got = struct.unpack("<2f", output.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"{suffix} output {got} != expected {expected}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-resident-linear-", dir="/private/tmp"))
    write_fixture(root)
    run_case(root, ".self_attn.q_a_proj.weight", (6.0, 5.0))
    run_case(root, ".self_attn.q_b_proj.weight", (14.0, 3.0))
    run_case(
        root,
        ".self_attn.q_c_proj.weight",
        (10.0, 6.0),
        input_name="input8.f32",
    )
    print(f"fixture: {root}")
    print("  resident linear:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
