#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.packer import pack_experts


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16_to_f32(raw: bytes) -> float:
    bits = int.from_bytes(raw, "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def f32_to_f16(value: float) -> bytes:
    return struct.pack("<e", value)


def write_raw_checkpoint(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 2,
                "moe_layer_freq": 1,
                "first_k_dense_replace": 1,
            }
        ),
        encoding="utf-8",
    )

    tensors: dict[str, bytes] = {}
    for expert, value in ((0, 1.0), (1, 2.0)):
        raw = f32_to_bf16(value) * 64
        for component in ("gate_proj", "up_proj", "down_proj"):
            tensors[
                f"model.layers.1.mlp.experts.{expert}.{component}.weight"
            ] = raw

    shard = root / "model-00001-of-00001.safetensors"
    header = {}
    payload = bytearray()
    for name, data in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "BF16",
            "shape": [8, 8],
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard.name for name in tensors},
            }
        ),
        encoding="utf-8",
    )


def write_resident(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
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


def convert_expert_metadata_to_f16(root: Path) -> None:
    layout_path = root / "experts" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    for layer in layout["layers"]:
        layer_file = root / "experts" / layer["layer_file"]
        data = bytearray(layer_file.read_bytes())
        slot_bytes = int(layer["expert_slot_bytes"])
        num_experts = int(layer["num_experts"])
        for component in layer["components"]:
            if not component["name"].endswith((".scales", ".biases")):
                continue
            offset = int(component["offset"])
            size = int(component["size"])
            for expert in range(num_experts):
                start = expert * slot_bytes + offset
                end = start + size
                for cursor in range(start, end, 2):
                    value = bf16_to_f32(bytes(data[cursor : cursor + 2]))
                    data[cursor : cursor + 2] = f32_to_f16(value)
            component["dtype"] = "F16"
        layer_file.write_bytes(data)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def rewrite_expert_layout_to_common_aliases(root: Path) -> None:
    layout_path = root / "experts" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    for layer in layout["layers"]:
        for component in layer["components"]:
            if component["name"].endswith(".weight"):
                component["dtype"] = "uint32"
            elif component["name"].endswith((".scales", ".biases")):
                component["dtype"] = "BFLOAT16"
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def run_smoke(runner: Path, root: Path, *, label: str) -> None:
    output = root / f"output-{label.lower()}.f32"
    router_json = root / f"router-{label.lower()}.json"
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
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    got = struct.unpack("<8f", output.read_bytes())[0]
    expected = expected_output0()
    if abs(got - expected) > 2e-3:
        raise SystemExit(f"output[0] {got:.6f} != expected {expected:.6f}")
    route = json.loads(router_json.read_text(encoding="utf-8"))
    if route["experts"] != [1, 0]:
        raise SystemExit(f"unexpected router experts: {route['experts']}")
    print(f"  metadata dtype:      {label}")
    print(f"  expected output[0]: {expected:.6f}")
    print("  quantized smoke:    ok")


def main() -> int:
    runner = Path(__file__).with_name("largerlm-runner")
    root = Path(tempfile.mkdtemp(prefix="largerlm-quantized-layer-moe-", dir="/private/tmp"))
    checkpoint = root / "checkpoint"
    write_raw_checkpoint(checkpoint)
    pack_experts(
        checkpoint,
        root / "experts",
        dry_run=False,
        chunk_size=17,
        disk_safety_margin_bytes=0,
        quantize_raw_to_int4=True,
        group_size=8,
    )
    write_resident(root)
    print(f"fixture: {root}")
    run_smoke(runner, root, label="BF16")
    rewrite_expert_layout_to_common_aliases(root)
    run_smoke(runner, root, label="BFLOAT16-alias")
    convert_expert_metadata_to_f16(root)
    run_smoke(runner, root, label="F16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
