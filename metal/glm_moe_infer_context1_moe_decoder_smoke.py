#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


SCALE_E8M0_ONE = 127


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", float(value)), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(value) for value in values)


def pack8(codes: list[int]) -> bytes:
    if len(codes) != 8:
        raise ValueError("MXFP4 pack8 expects exactly 8 values")
    packed = 0
    for index, code in enumerate(codes):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def packed_row(logical_cols: int, one_at: int | None = None) -> bytes:
    if logical_cols % 8 != 0:
        raise ValueError("logical_cols must be divisible by 8")
    data = bytearray()
    for base in range(0, logical_cols, 8):
        codes = [0] * 8
        if one_at is not None and base <= one_at < base + 8:
            codes[one_at - base] = 2
        data.extend(pack8(codes))
    return bytes(data)


def identity_matrix_rows(rows: int, cols: int) -> bytes:
    out = bytearray()
    for row in range(rows):
        out.extend(packed_row(cols, row if row < cols else None))
    return bytes(out)


def zero_matrix_rows(rows: int, cols: int) -> bytes:
    return packed_row(cols, None) * rows


def add_tensor(
    tensors: list[dict[str, object]],
    payload: bytearray,
    name: str,
    dtype: str,
    shape: list[int],
    category: str,
    data: bytes,
) -> None:
    if len(payload) % 4 != 0:
        payload.extend(b"\0" * (4 - len(payload) % 4))
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


def expert_slot_zero() -> bytes:
    slot = bytearray()
    for _name in ("gate_proj", "up_proj", "down_proj"):
        slot.extend(zero_matrix_rows(32, 32))
        slot.extend(bytes([SCALE_E8M0_ONE]) * 32)
    return bytes(slot)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    experts = root / "experts"
    context1 = root / "context1"
    resident.mkdir(parents=True)
    experts.mkdir(parents=True)
    context1.mkdir(parents=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []
    prefix = "model.layers.0"

    for suffix in (
        "input_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "post_attention_layernorm.weight",
    ):
        dim = 8 if suffix == "self_attn.q_a_layernorm.weight" else 32
        add_tensor(tensors, payload, f"{prefix}.{suffix}", "F32", [dim], "norms", f32([1.0] * dim))

    matrix_specs = (
        ("self_attn.q_a_proj", 8, 32, zero_matrix_rows(8, 32)),
        ("self_attn.q_b_proj", 16, 8, zero_matrix_rows(16, 8)),
        ("self_attn.kv_a_proj_with_mqa", 40, 32, identity_matrix_rows(40, 32)),
        ("self_attn.o_proj", 32, 32, identity_matrix_rows(32, 32)),
    )
    for suffix, rows, cols, data in matrix_specs:
        add_tensor(
            tensors,
            payload,
            f"{prefix}.{suffix}.weight",
            "U32",
            [rows, cols // 8],
            "attention",
            data,
        )
        add_tensor(
            tensors,
            payload,
            f"{prefix}.{suffix}.scales",
            "U8",
            [rows, max(1, cols // 32)],
            "attention",
            bytes([SCALE_E8M0_ONE]) * rows * max(1, cols // 32),
        )

    add_tensor(
        tensors,
        payload,
        f"{prefix}.self_attn.embed_q.weight",
        "U32",
        [1, 32, 1],
        "attention",
        zero_matrix_rows(32, 8),
    )
    add_tensor(
        tensors,
        payload,
        f"{prefix}.self_attn.embed_q.scales",
        "U8",
        [1, 32, 1],
        "attention",
        bytes([SCALE_E8M0_ONE]) * 32,
    )
    add_tensor(
        tensors,
        payload,
        f"{prefix}.self_attn.unembed_out.weight",
        "U32",
        [1, 32, 4],
        "attention",
        identity_matrix_rows(32, 32),
    )
    add_tensor(
        tensors,
        payload,
        f"{prefix}.self_attn.unembed_out.scales",
        "U8",
        [1, 32, 1],
        "attention",
        bytes([SCALE_E8M0_ONE]) * 32,
    )
    add_tensor(
        tensors,
        payload,
        f"{prefix}.mlp.gate.weight",
        "BF16",
        [1, 32],
        "router",
        bf16([0.0] * 32),
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

    slot = expert_slot_zero()
    (experts / "layer_000.bin").write_bytes(slot)
    components: list[dict[str, object]] = []
    cursor = 0
    for name in ("gate_proj", "up_proj", "down_proj"):
        components.append(
            {
                "name": f"{name}.weight",
                "offset": cursor,
                "size": 512,
                "dtype": "U32",
                "shape": [32, 4],
            }
        )
        cursor += 512
        components.append(
            {
                "name": f"{name}.scales",
                "offset": cursor,
                "size": 32,
                "dtype": "U8",
                "shape": [32, 1],
            }
        )
        cursor += 32
    (experts / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-mxfp4",
                "group_size": 32,
                "num_layers": 1,
                "num_experts": 1,
                "layers": [
                    {
                        "layer": 0,
                        "num_experts": 1,
                        "expert_slot_bytes": len(slot),
                        "layer_file": "layer_000.bin",
                        "components": components,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    identity_cache: list[float] = []
    for row in range(32):
        for col in range(32):
            identity_cache.append(1.0 if row == col else 0.0)
    (context1 / "context1_o_proj_bv.bin").write_bytes(bf16(identity_cache))
    (context1 / "layout.json").write_text(
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
                "total_bytes": len(identity_cache) * 2,
                "dims": {
                    "hidden_dim": 32,
                    "attention_value_dim": 32,
                    "num_heads": 1,
                    "v_head_dim": 32,
                    "qk_nope_dim": 8,
                    "kv_lora_dim": 32,
                },
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.context1_o_proj_bv.weight",
                        "layer": 0,
                        "offset": 0,
                        "size": len(identity_cache) * 2,
                        "dtype": "BF16",
                        "shape": [32, 32],
                        "category": "context1_attention_output",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    cache_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "max_context_tokens": 1,
        "dtype": "F32",
        "dtype_bytes": 4,
        "alignment": 64,
        "total_bytes": 160,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": 0,
                "offset": 0,
                "width": 40,
                "dtype": "F32",
                "dtype_bytes": 4,
                "token_stride_bytes": 160,
                "max_context_tokens": 1,
                "total_bytes": 160,
            }
        ],
    }
    (root / "cache_layout.json").write_text(
        json.dumps(cache_layout, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "baseline_cache.bin").write_bytes(b"\0" * 160)
    (root / "context_cache.bin").write_bytes(b"\0" * 160)
    values = [1.0 + 0.05 * i for i in range(32)]
    (root / "input.f32").write_bytes(f32(values))


def run_json(cmd: list[str]) -> dict[str, object]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return json.loads(completed.stdout)


def decoder_cmd(root: Path, binary: Path, cache_file: Path, output: Path, work: Path) -> list[str]:
    return [
        str(binary),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--expert-layout",
        str(root / "experts" / "layout.json"),
        "--probe-decoder-layer",
        "--probe-layer",
        "0",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(output),
        "--output-dir",
        str(work),
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(cache_file),
        "--position",
        "0",
        "--context-length",
        "1",
        "--num-heads",
        "1",
        "--kv-lora-dim",
        "32",
        "--qk-nope-dim",
        "8",
        "--rope-dim",
        "8",
        "--v-head-dim",
        "32",
        "--cache-position-offset",
        "0",
        "--top-k",
        "1",
        "--router-score",
        "raw",
        "--routed-scaling-factor",
        "1",
        "--ignore-router-bias",
        "--rms-norm-eps",
        "0",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-live-working-set-mib",
        "128",
        "--json",
    ]


def read_f32(path: Path) -> tuple[float, ...]:
    return struct.unpack("<32f", path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify glm_moe_infer MoE decoder can consume a context=1 o_proj*B_v cache."
    )
    parser.add_argument("--binary", type=Path, default=Path(__file__).with_name("glm_moe_infer"))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-context1-moe-decoder-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    baseline = run_json(
        decoder_cmd(
            root,
            args.binary,
            root / "baseline_cache.bin",
            root / "baseline_out.f32",
            root / "baseline_work",
        )
    )
    context_cmd = decoder_cmd(
        root,
        args.binary,
        root / "context_cache.bin",
        root / "context_out.f32",
        root / "context_work",
    )
    context_cmd.extend(
        [
            "--context1-o-proj-cache-layout",
            str(root / "context1" / "layout.json"),
        ]
    )
    context = run_json(context_cmd)
    baseline_decoder = baseline.get("probe_decoder_layer") or {}
    context_decoder = context.get("probe_decoder_layer") or {}
    if bool(baseline_decoder.get("attn_output_context1_o_proj_cache")):
        raise SystemExit("baseline unexpectedly used context1 o_proj cache")
    if not bool(context_decoder.get("attn_output_context1_o_proj_cache")):
        raise SystemExit(f"context run did not use context1 o_proj cache: {context}")
    got_baseline = read_f32(root / "baseline_out.f32")
    got_context = read_f32(root / "context_out.f32")
    max_diff = max(math.fabs(a - b) for a, b in zip(got_baseline, got_context))
    if max_diff > 2e-3:
        raise SystemExit(
            f"context1 MoE decoder output differs from baseline by {max_diff}: "
            f"{got_baseline} vs {got_context}"
        )
    if (root / "baseline_cache.bin").read_bytes() != (root / "context_cache.bin").read_bytes():
        raise SystemExit("context1 MoE decoder cache write differs from baseline")
    print(f"fixture: {root}")
    print(f"  max output diff: {max_diff:.6g}")
    print("  context1 MoE decoder: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
