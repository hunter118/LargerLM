#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


COMPONENTS = [
    ("gate_proj.weight", 0, 32, "U32", [8, 1]),
    ("gate_proj.scales", 32, 16, "BF16", [8, 1]),
    ("gate_proj.biases", 48, 16, "BF16", [8, 1]),
    ("up_proj.weight", 64, 32, "U32", [8, 1]),
    ("up_proj.scales", 96, 16, "BF16", [8, 1]),
    ("up_proj.biases", 112, 16, "BF16", [8, 1]),
    ("down_proj.weight", 128, 32, "U32", [8, 1]),
    ("down_proj.scales", 160, 16, "BF16", [8, 1]),
    ("down_proj.biases", 176, 16, "BF16", [8, 1]),
]


def pack8(value: int) -> int:
    out = 0
    for i in range(8):
        out |= (value & 0xF) << (i * 4)
    return out


def write_fixture(root: Path) -> None:
    experts = root / "experts"
    resident = root / "resident"
    experts.mkdir(parents=True, exist_ok=True)
    resident.mkdir(parents=True, exist_ok=True)

    slot_bytes = 192
    layer = bytearray()
    for expert_value in (1, 2):
        slot = bytearray(slot_bytes)
        for base in (0, 64, 128):
            packed = pack8(expert_value)
            for row in range(8):
                struct.pack_into("<I", slot, base + row * 4, packed)
        for base in (32, 96, 160):
            slot[base : base + 16] = b"\x80\x3f" * 8
        for base in (48, 112, 176):
            slot[base : base + 16] = b"\x00\x00" * 8
        layer.extend(slot)
    (experts / "layer_001.bin").write_bytes(layer)
    (experts / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-affine-int4",
                "group_size": 8,
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

    router = struct.pack("<16f", *([1.0] * 8 + [2.0] * 8))
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
                        "shape": [2, 8],
                        "category": "routers",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(struct.pack("<8f", *([1.0] * 8)))


def expected_output0() -> float:
    expert0 = 512.0 / (1.0 + math.exp(-8.0))
    expert1 = 4096.0 / (1.0 + math.exp(-16.0))
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
    if "read advise:" not in completed.stdout:
        raise SystemExit("runner did not report expert read advice")

    values = struct.unpack("<8f", output.read_bytes())
    expected = expected_output0()
    if abs(values[0] - expected) > 2e-3:
        raise SystemExit(f"output[0] {values[0]:.6f} != expected {expected:.6f}")
    route = json.loads(router_json.read_text(encoding="utf-8"))
    if route["experts"] != [1, 0]:
        raise SystemExit(f"unexpected router experts: {route['experts']}")
    print(f"  expected output[0]: {expected:.6f}")
    print("  smoke result:       ok")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and run a tiny layer-MoE runner smoke.")
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

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-layer-moe-smoke-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    print(f"fixture: {root}")
    run_smoke(args.runner, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
