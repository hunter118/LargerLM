#!/usr/bin/env python3
from __future__ import annotations

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


def write_experts(root: Path) -> None:
    experts = root / "experts"
    experts.mkdir(parents=True, exist_ok=True)
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


def write_resident(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    router = struct.pack("<16f", *([1.0] * 8 + [2.0] * 8))
    norm = struct.pack("<8f", *([1.0] * 8))
    shared = struct.pack("<64f", *([1.0] * 64))
    payload = router + norm + shared + shared + shared
    tensors = [
        {
            "name": "model.layers.1.mlp.gate.weight",
            "offset": 0,
            "size": len(router),
            "dtype": "F32",
            "shape": [2, 8],
            "category": "routers",
        },
        {
            "name": "model.layers.1.post_attention_layernorm.weight",
            "offset": len(router),
            "size": len(norm),
            "dtype": "F32",
            "shape": [8],
            "category": "norms",
        },
    ]
    offset = len(router) + len(norm)
    for component in ("gate_proj", "up_proj", "down_proj"):
        tensors.append(
            {
                "name": f"model.layers.1.mlp.shared_experts.{component}.weight",
                "offset": offset,
                "size": len(shared),
                "dtype": "F32",
                "shape": [8, 8],
                "category": "shared_experts",
            }
        )
        offset += len(shared)
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


def expert_output(value: float, normed: float) -> float:
    gate = 8.0 * value * normed
    up = 8.0 * value * normed
    act = (gate / (1.0 + math.exp(-gate))) * up
    down = 8.0 * value * act
    return down


def main() -> int:
    eps = 1e-5
    root = Path(tempfile.mkdtemp(prefix="largerlm-mlp-block-", dir="/private/tmp"))
    write_experts(root)
    write_resident(root)
    (root / "input.f32").write_bytes(struct.pack("<8f", *([1.0] * 8)))
    output = root / "output.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--layout",
        str(root / "experts" / "layout.json"),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-mlp-block",
        "--include-shared-expert",
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
        "--rms-norm-eps",
        str(eps),
        "--output-f32",
        str(output),
        "--max-slot-mib",
        "1",
        "--max-router-mib",
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
    routed = (2.0 / 3.0) * expert_output(2.0, normed) + (1.0 / 3.0) * expert_output(1.0, normed)
    expected = 1.0 + routed + expert_output(1.0, normed)
    if abs(got - expected) > 3e-3:
        raise SystemExit(f"output[0] {got:.6f} != expected {expected:.6f}")
    print(f"fixture: {root}")
    print(f"  expected output[0]: {expected:.6f}")
    print("  mlp block smoke:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
