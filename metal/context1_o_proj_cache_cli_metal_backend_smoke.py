#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


SCALE_E8M0_ONE = 127


def pack8(code: int) -> bytes:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def bf16_to_f32(raw: bytes) -> float:
    bits = int.from_bytes(raw, "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def write_prepared(root: Path) -> Path:
    prepared = root / "prepared"
    resident = prepared / "resident"
    resident.mkdir(parents=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, dtype: str, shape: list[int], data: bytes) -> None:
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": "attention",
            }
        )
        payload.extend(data)

    prefix = "model.layers.0.self_attn"
    add(
        f"{prefix}.o_proj.weight",
        "U32",
        [4, 1],
        pack8(2) + pack8(1) + pack8(0) + pack8(0),
    )
    add(
        f"{prefix}.o_proj.scales",
        "U8",
        [4, 1],
        bytes([SCALE_E8M0_ONE]) * 4,
    )
    add(f"{prefix}.embed_q.weight", "U32", [1, 8, 1], pack8(2) * 8)
    add(
        f"{prefix}.embed_q.scales",
        "U8",
        [1, 8, 1],
        bytes([SCALE_E8M0_ONE]) * 8,
    )
    add(f"{prefix}.unembed_out.weight", "U32", [1, 8, 1], pack8(2) * 8)
    add(
        f"{prefix}.unembed_out.scales",
        "U8",
        [1, 8, 1],
        bytes([SCALE_E8M0_ONE]) * 8,
    )
    (resident / "resident.bin").write_bytes(bytes(payload))
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
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "resident_layout": "resident/layout.json",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a tiny context=1 o_proj*B_v cache through the Python CLI Metal backend."
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("glm_moe_infer"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-context1-cli-metal-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    prepared = write_prepared(root)
    completed = subprocess.run(
        [
            "python3",
            "-m",
            "largerlm",
            "context1-o-proj-cache",
            str(prepared),
            "--output-dir",
            str(root / "cache"),
            "--execute",
            "--backend",
            "metal",
            "--metal-binary",
            str(args.runner),
            "--layer",
            "0",
            "--max-cache-gib",
            "1",
            "--max-build-fma",
            "1024",
            "--json",
        ],
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        return completed.returncode
    payload = json.loads(completed.stdout)
    if payload.get("executed") is not True or payload.get("backend") != "metal":
        raise SystemExit("CLI Metal backend did not execute")
    if payload.get("max_metal_builder_live_bytes") != 512 * 1024 * 1024:
        raise SystemExit(
            "CLI Metal backend did not report the default builder live cap"
        )
    selected = payload.get("selected_build")
    if not isinstance(selected, dict):
        raise SystemExit("CLI Metal backend did not report selected_build")
    if selected.get("max_estimated_metal_builder_live_bytes") != 124:
        raise SystemExit(
            "CLI Metal backend reported unexpected builder live estimate: "
            f"{selected.get('max_estimated_metal_builder_live_bytes')}"
        )
    summary = payload.get("layer_result_summary")
    if not isinstance(summary, dict):
        raise SystemExit("CLI Metal backend did not report layer_result_summary")
    if summary.get("measured_layer_count") != 1:
        raise SystemExit(
            "CLI Metal backend reported unexpected measured layer count: "
            f"{summary.get('measured_layer_count')}"
        )
    if summary.get("measured_gfma_per_second") is None:
        raise SystemExit("CLI Metal backend did not report measured GFMA/s")
    raw = (root / "cache" / "context1_o_proj_bv.bin").read_bytes()
    values = [bf16_to_f32(raw[index : index + 2]) for index in range(0, len(raw), 2)]
    if (
        values[:8] != [8.0] * 8
        or values[8:16] != [4.0] * 8
        or values[16:] != [0.0] * 16
    ):
        raise SystemExit(f"built cache values are wrong: {values}")
    print(f"fixture: {root}")
    print("  context1 o_proj cache CLI Metal backend: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
