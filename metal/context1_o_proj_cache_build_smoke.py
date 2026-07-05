#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


SCALE_E8M0_ONE = 127


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def pack8(code: int) -> bytes:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def bf16_to_f32(raw: bytes) -> float:
    bits = int.from_bytes(raw, "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    cache = root / "cache"
    resident.mkdir(parents=True)
    cache.mkdir(parents=True)
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
    (cache / "context1_o_proj_bv.bin").write_bytes(b"\0" * 64)
    (cache / "layout.json").write_text(
        json.dumps(
            {
                "schema": "largerlm.context1_o_proj_bv_cache.v1",
                "version": 1,
                "config_sha256": None,
                "source_resident_layout": str(resident / "layout.json"),
                "context_limit": "decode/context_length_1_only",
                "dtype": "BF16",
                "dtype_bytes": 2,
                "weight_file": "context1_o_proj_bv.bin",
                "total_bytes": 64,
                "dims": {
                    "hidden_dim": 4,
                    "attention_value_dim": 8,
                    "num_heads": 1,
                    "v_head_dim": 8,
                    "qk_nope_dim": 8,
                    "kv_lora_dim": 8,
                },
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.context1_o_proj_bv.weight",
                        "layer": 0,
                        "offset": 0,
                        "size": 64,
                        "dtype": "BF16",
                        "shape": [4, 8],
                        "category": "context1_attention_output",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "latent.f32").write_bytes(f32([1.0] * 8))
    (root / "residual.f32").write_bytes(f32([0.5, -1.0, 0.0, 0.0]))


def run_json(cmd: list[str], *, expect_success: bool = True) -> dict[str, object]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if expect_success and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not expect_success and completed.returncode == 0:
        raise SystemExit("command unexpectedly succeeded")
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and consume a tiny context=1 o_proj*B_v cache with Metal."
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("glm_moe_infer"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-context1-build-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    cache_layout = root / "cache" / "layout.json"
    build = run_json(
        [
            str(args.runner),
            "--build-context1-o-proj-cache-layer",
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--probe-layer",
            "0",
            "--context1-o-proj-cache-layout",
            str(cache_layout),
            "--max-cache-read-mib",
            "1",
            "--max-context1-o-proj-build-gfma",
            "1",
            "--max-live-working-set-mib",
            "1",
            "--json",
        ]
    )
    if build.get("ok") is not True:
        raise SystemExit("context1 cache builder did not report ok")
    if build.get("estimated_live_working_set_bytes") != 124:
        raise SystemExit(
            "context1 cache builder reported unexpected live bytes: "
            f"{build.get('estimated_live_working_set_bytes')}"
        )
    if build.get("max_live_working_set_mib") != 1:
        raise SystemExit(
            "context1 cache builder did not echo max live cap: "
            f"{build.get('max_live_working_set_mib')}"
        )
    if build.get("live_working_set_ok") not in (True, 1):
        raise SystemExit("context1 cache builder did not report live cap ok")

    rejected = run_json(
        [
            str(args.runner),
            "--build-context1-o-proj-cache-layer",
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--probe-layer",
            "0",
            "--context1-o-proj-cache-layout",
            str(cache_layout),
            "--max-cache-read-mib",
            "1",
            "--max-context1-o-proj-build-gfma",
            "1",
            "--max-live-working-set-mib",
            "0.0001",
            "--json",
        ],
        expect_success=False,
    )
    if rejected.get("ok") is not False:
        raise SystemExit("context1 cache builder low live cap was not rejected")
    if rejected.get("live_working_set_ok") not in (False, 0):
        raise SystemExit("context1 cache builder rejection did not report live cap failure")
    if rejected.get("estimated_live_working_set_bytes") != 124:
        raise SystemExit(
            "context1 cache builder rejection reported unexpected live bytes: "
            f"{rejected.get('estimated_live_working_set_bytes')}"
        )
    raw = (root / "cache" / "context1_o_proj_bv.bin").read_bytes()
    values = [bf16_to_f32(raw[index : index + 2]) for index in range(0, len(raw), 2)]
    if (
        values[:8] != [8.0] * 8
        or values[8:16] != [4.0] * 8
        or values[16:] != [0.0] * 16
    ):
        raise SystemExit(f"built cache values are wrong: {values}")

    output = root / "output.f32"
    consume = run_json(
        [
            str(args.runner),
            "--probe-context1-o-proj-cache-output",
            "--probe-layer",
            "0",
            "--context1-o-proj-cache-layout",
            str(cache_layout),
            "--input-f32",
            str(root / "latent.f32"),
            "--residual-f32",
            str(root / "residual.f32"),
            "--output-f32",
            str(output),
            "--max-cache-read-mib",
            "1",
            "--json",
        ]
    )
    if consume.get("ok") is not True:
        raise SystemExit("context1 cache consumer did not report ok")
    got = struct.unpack("<4f", output.read_bytes())
    expected = (64.5, 31.0, 0.0, 0.0)
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"context1 cache output {got} != expected {expected}")
    print(f"fixture: {root}")
    print("  context1 o_proj cache build:  ok")
    print("  context1 o_proj cache live cap: ok")
    print("  context1 o_proj cache output: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
