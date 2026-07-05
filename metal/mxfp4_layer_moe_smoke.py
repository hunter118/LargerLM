#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 32
INTERMEDIATE_DIM = 32
GROUP_SIZE = 32
SCALE_E8M0_ONE = 127


COMPONENTS = [
    ("gate_proj.weight", 0, 512, "U32", [INTERMEDIATE_DIM, HIDDEN_DIM // 8]),
    ("gate_proj.scales", 512, 32, "U8", [INTERMEDIATE_DIM, HIDDEN_DIM // GROUP_SIZE]),
    ("up_proj.weight", 544, 512, "U32", [INTERMEDIATE_DIM, HIDDEN_DIM // 8]),
    ("up_proj.scales", 1056, 32, "U8", [INTERMEDIATE_DIM, HIDDEN_DIM // GROUP_SIZE]),
    ("down_proj.weight", 1088, 512, "U32", [HIDDEN_DIM, INTERMEDIATE_DIM // 8]),
    ("down_proj.scales", 1600, 32, "U8", [HIDDEN_DIM, INTERMEDIATE_DIM // GROUP_SIZE]),
]


def pack8(code: int) -> int:
    out = 0
    for i in range(8):
        out |= (code & 0xF) << (i * 4)
    return out


def fp4_code_to_value(code: int) -> float:
    mag = code & 0x7
    if mag == 0:
        return 0.0
    sign = -1.0 if code & 0x8 else 1.0
    if mag == 1:
        return sign * 0.5
    exp_bits = mag >> 1
    mant = mag & 0x1
    return sign * (1.0 + 0.5 * mant) * (2.0 ** (exp_bits - 1))


def write_fixture(root: Path) -> None:
    experts = root / "experts"
    resident = root / "resident"
    experts.mkdir(parents=True, exist_ok=True)
    resident.mkdir(parents=True, exist_ok=True)

    slot_bytes = 1632
    layer = bytearray()
    for fp4_code in (1, 2):
        slot = bytearray(slot_bytes)
        packed = pack8(fp4_code)
        for weight_offset, _scale_offset in ((0, 512), (544, 1056), (1088, 1600)):
            for row in range(32):
                for packed_col in range(4):
                    struct.pack_into(
                        "<I",
                        slot,
                        weight_offset + (row * 4 + packed_col) * 4,
                        packed,
                    )
        for scale_offset in (512, 1056, 1600):
            slot[scale_offset : scale_offset + 32] = bytes([SCALE_E8M0_ONE]) * 32
        layer.extend(slot)
    (experts / "layer_001.bin").write_bytes(layer)
    (experts / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-mxfp4",
                "group_size": GROUP_SIZE,
                "num_layers": 2,
                "num_experts": 2,
                "component_order": [name for name, *_ in COMPONENTS],
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 2,
                        "expert_slot_bytes": slot_bytes,
                        "layer_file": "layer_001.bin",
                        "components": [
                            {
                                "name": name,
                                "offset": offset,
                                "size": size,
                                "dtype": dtype,
                                "shape": shape,
                            }
                            for name, offset, size, dtype, shape in COMPONENTS
                        ],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    router = struct.pack("<64f", *([1.0] * HIDDEN_DIM + [2.0] * HIDDEN_DIM))
    (resident / "resident.bin").write_bytes(router)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(router),
                "tensors": [
                    {
                        "name": "model.layers.1.mlp.gate.weight",
                        "offset": 0,
                        "size": len(router),
                        "dtype": "F32",
                        "shape": [2, HIDDEN_DIM],
                        "category": "routers",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    input_values = [1.0 / HIDDEN_DIM] * HIDDEN_DIM
    (root / "input.f32").write_bytes(struct.pack("<32f", *input_values))


def expert_output_for_code(code: int) -> float:
    value = fp4_code_to_value(code)
    gate = value
    up = value
    act = (gate / (1.0 + math.exp(-gate))) * up
    return HIDDEN_DIM * value * act


def expected_output0() -> float:
    expert0 = expert_output_for_code(1)
    expert1 = expert_output_for_code(2)
    return (2.0 / 3.0) * expert1 + (1.0 / 3.0) * expert0


def run_smoke(runner: Path, root: Path) -> None:
    output = root / "output.f32"
    router_json = root / "router.json"
    cmd = [
        str(runner),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-layer-moe",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "2",
        "--max-k",
        "2",
        "--router-score",
        "raw",
        "--routed-scaling-factor",
        "1",
        "--output-f32",
        str(output),
        "--output-router-json",
        str(router_json),
        "--max-slot-mib",
        "1",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--expert-read-advise-merge-gap-kib",
        "0",
        "--expert-read-advise-align-kib",
        "4",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if "quantization:       mlx-mxfp4" not in completed.stdout:
        raise SystemExit("runner did not report mlx-mxfp4 execution")
    if "read advise:" not in completed.stdout:
        raise SystemExit("runner did not report expert read advice")

    values = struct.unpack("<32f", output.read_bytes())
    expected = expected_output0()
    if abs(values[0] - expected) > 5e-3:
        raise SystemExit(f"output[0] {values[0]:.6f} != expected {expected:.6f}")
    route = json.loads(router_json.read_text(encoding="utf-8"))
    if route["experts"] != [1, 0]:
        raise SystemExit(f"unexpected router experts: {route['experts']}")
    print(f"  expected output[0]: {expected:.6f}")
    print("  smoke result:       ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and run a tiny MXFP4 layer-MoE smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
        help="Path to the compiled Metal runner.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Fixture directory. Defaults to a new /private/tmp directory.",
    )
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-mxfp4-layer-moe-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    print(f"fixture: {root}")
    run_smoke(args.runner, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
