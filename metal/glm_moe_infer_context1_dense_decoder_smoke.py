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


def pack_row(codes: list[int]) -> bytes:
    if len(codes) != 8:
        raise ValueError("MXFP4 tiny rows must have exactly 8 logical columns")
    packed = 0
    for index, code in enumerate(codes):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def identity_rows(rows: int) -> bytes:
    data = bytearray()
    for row in range(rows):
        codes = [0] * 8
        if row < 8:
            codes[row] = 2
        data.extend(pack_row(codes))
    return bytes(data)


def zero_rows(rows: int) -> bytes:
    return pack_row([0] * 8) * rows


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

    for name in (
        "input_layernorm.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_layernorm.weight",
        "post_attention_layernorm.weight",
    ):
        dim = 8
        add_tensor(
            tensors,
            payload,
            f"{prefix}.{name}",
            "F32",
            [dim],
            "norms",
            f32([1.0] * dim),
        )

    for suffix, rows, data in (
        ("self_attn.q_a_proj", 8, identity_rows(8)),
        ("self_attn.q_b_proj", 12, zero_rows(12)),
        ("self_attn.kv_a_proj_with_mqa", 12, identity_rows(12)),
        ("self_attn.o_proj", 8, identity_rows(8)),
        ("mlp.gate_proj", 8, zero_rows(8)),
        ("mlp.up_proj", 8, zero_rows(8)),
        ("mlp.down_proj", 8, zero_rows(8)),
    ):
        add_tensor(
            tensors,
            payload,
            f"{prefix}.{suffix}.weight",
            "U32",
            [rows, 1],
            "weights",
            data,
        )
        add_tensor(
            tensors,
            payload,
            f"{prefix}.{suffix}.scales",
            "U8",
            [rows, 1],
            "weights",
            bytes([SCALE_E8M0_ONE]) * rows,
        )

    for suffix in ("embed_q", "unembed_out"):
        add_tensor(
            tensors,
            payload,
            f"{prefix}.self_attn.{suffix}.weight",
            "U32",
            [1, 8, 1],
            "attention",
            identity_rows(8),
        )
        add_tensor(
            tensors,
            payload,
            f"{prefix}.self_attn.{suffix}.scales",
            "U8",
            [1, 8, 1],
            "attention",
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
    (experts / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-mxfp4",
                "group_size": 32,
                "num_layers": 1,
                "num_experts": 0,
                "layers": [
                    {
                        "layer": 99,
                        "num_experts": 0,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_099.bin",
                        "components": [],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (experts / "layer_099.bin").write_bytes(b"")
    identity_cache: list[float] = []
    for row in range(8):
        for col in range(8):
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
                "total_bytes": 128,
                "dims": {
                    "hidden_dim": 8,
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
                        "size": 128,
                        "dtype": "BF16",
                        "shape": [8, 8],
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
        "total_bytes": 48,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": 0,
                "offset": 0,
                "width": 12,
                "dtype": "F32",
                "dtype_bytes": 4,
                "token_stride_bytes": 48,
                "max_context_tokens": 1,
                "total_bytes": 48,
            }
        ],
    }
    (root / "cache_layout.json").write_text(
        json.dumps(cache_layout, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "baseline_cache.bin").write_bytes(b"\0" * 48)
    (root / "mmap_cache.bin").write_bytes(b"\0" * 48)
    (root / "context_cache.bin").write_bytes(b"\0" * 48)
    (root / "decode_baseline_cache.bin").write_bytes(b"\0" * 48)
    (root / "decode_mmap_cache.bin").write_bytes(b"\0" * 48)
    (root / "input.f32").write_bytes(f32([1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 1.5, -0.5]))


def run_json(cmd: list[str]) -> dict[str, object]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return json.loads(completed.stdout)


def read_f32(path: Path) -> tuple[float, ...]:
    return struct.unpack("<8f", path.read_bytes())


def decoder_cmd(root: Path, binary: Path, cache_file: Path, output: Path, work: Path) -> list[str]:
    return [
        str(binary),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--expert-layout",
        str(root / "experts" / "layout.json"),
        "--no-open-experts",
        "--probe-dense-decoder-layer",
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
        "8",
        "--qk-nope-dim",
        "8",
        "--rope-dim",
        "4",
        "--v-head-dim",
        "8",
        "--cache-position-offset",
        "0",
        "--rms-norm-eps",
        "0",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-live-working-set-mib",
        "64",
        "--json",
    ]


def decode_layers_cmd(root: Path, binary: Path, cache_file: Path, output: Path, work: Path) -> list[str]:
    cmd = decoder_cmd(root, binary, cache_file, output, work)
    dense_index = cmd.index("--probe-dense-decoder-layer")
    cmd[dense_index:dense_index + 1] = [
        "--probe-decode-layers",
        "--decode-layers",
        "0",
    ]
    cmd.extend(
        [
            "--top-k",
            "1",
            "--skip-debug-intermediates",
        ]
    )
    return cmd


def first_decode_layer(payload: dict[str, object]) -> dict[str, object]:
    decode = payload.get("probe_decode_layers")
    if not isinstance(decode, dict):
        raise SystemExit(f"missing probe_decode_layers payload: {payload}")
    layers = decode.get("layers")
    if not isinstance(layers, list) or not layers:
        raise SystemExit(f"missing decode layer items: {decode}")
    item = layers[0]
    if not isinstance(item, dict):
        raise SystemExit(f"invalid decode layer item: {item}")
    return item


def aggregate_mmap_count(payload: dict[str, object]) -> int:
    decode = payload.get("probe_decode_layers")
    if not isinstance(decode, dict):
        return -1
    value = decode.get("attn_output_resident_mmap_backed_count")
    return int(value) if isinstance(value, int) else -1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify glm_moe_infer dense decoder can consume a context=1 o_proj*B_v cache."
    )
    parser.add_argument("--binary", type=Path, default=Path(__file__).with_name("glm_moe_infer"))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-context1-dense-decoder-", dir="/private/tmp"))
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
    baseline_decoder = baseline.get("probe_dense_decoder_layer") or {}
    mmap_cmd = decoder_cmd(
        root,
        args.binary,
        root / "mmap_cache.bin",
        root / "mmap_out.f32",
        root / "mmap_work",
    )
    mmap_cmd.extend(["--mmap-resident", "--wrap-resident-metal"])
    mmap_payload = run_json(mmap_cmd)
    mmap_decoder = mmap_payload.get("probe_dense_decoder_layer") or {}
    context_decoder = context.get("probe_dense_decoder_layer") or {}
    if bool(baseline_decoder.get("attn_output_context1_o_proj_cache")):
        raise SystemExit("baseline unexpectedly used context1 o_proj cache")
    if bool(baseline_decoder.get("attn_output_resident_mmap_backed")):
        raise SystemExit("baseline unexpectedly used resident mmap-backed attention output")
    if not bool(mmap_payload.get("resident_metal_wrapped")):
        raise SystemExit(f"mmap dense decoder did not wrap resident Metal buffer: {mmap_payload}")
    if not bool(mmap_decoder.get("attn_output_resident_mmap_backed")):
        raise SystemExit(f"mmap dense decoder did not use resident-backed o_proj: {mmap_payload}")
    if int(mmap_decoder.get("attn_output_bytes_read", -1)) != 0:
        raise SystemExit(f"mmap dense decoder staged o_proj bytes: {mmap_decoder}")
    if not bool(context_decoder.get("attn_output_context1_o_proj_cache")):
        raise SystemExit(f"context run did not use context1 o_proj cache: {context}")
    got_baseline = read_f32(root / "baseline_out.f32")
    got_mmap = read_f32(root / "mmap_out.f32")
    got_context = read_f32(root / "context_out.f32")
    max_diff = max(math.fabs(a - b) for a, b in zip(got_baseline, got_context))
    max_mmap_diff = max(math.fabs(a - b) for a, b in zip(got_baseline, got_mmap))
    if max_mmap_diff > 2e-3:
        raise SystemExit(
            f"mmap dense decoder output differs from baseline by {max_mmap_diff}: "
            f"{got_baseline} vs {got_mmap}"
        )
    if max_diff > 2e-3:
        raise SystemExit(
            f"context1 dense decoder output differs from baseline by {max_diff}: "
            f"{got_baseline} vs {got_context}"
        )
    if (root / "baseline_cache.bin").read_bytes() != (root / "mmap_cache.bin").read_bytes():
        raise SystemExit("mmap dense decoder cache write differs from baseline")
    if (root / "baseline_cache.bin").read_bytes() != (root / "context_cache.bin").read_bytes():
        raise SystemExit("context1 dense decoder cache write differs from baseline")

    decode_baseline = run_json(
        decode_layers_cmd(
            root,
            args.binary,
            root / "decode_baseline_cache.bin",
            root / "decode_baseline_out.f32",
            root / "decode_baseline_work",
        )
    )
    decode_mmap_cmd = decode_layers_cmd(
        root,
        args.binary,
        root / "decode_mmap_cache.bin",
        root / "decode_mmap_out.f32",
        root / "decode_mmap_work",
    )
    decode_mmap_cmd.extend(["--mmap-resident", "--wrap-resident-metal"])
    decode_mmap = run_json(decode_mmap_cmd)
    decode_baseline_layer = first_decode_layer(decode_baseline)
    decode_mmap_layer = first_decode_layer(decode_mmap)
    if bool(decode_baseline_layer.get("attn_output_resident_mmap_backed")):
        raise SystemExit("decode-layers baseline unexpectedly used resident mmap-backed o_proj")
    if not bool(decode_mmap_layer.get("attn_output_resident_mmap_backed")):
        raise SystemExit(f"decode-layers mmap path did not use resident-backed o_proj: {decode_mmap}")
    if int(decode_mmap_layer.get("attn_output_bytes_read", -1)) != 0:
        raise SystemExit(f"decode-layers mmap path staged o_proj bytes: {decode_mmap_layer}")
    if not bool(decode_mmap_layer.get("attn_output_fused_matvec_add")):
        raise SystemExit(f"decode-layers mmap path did not use fused o_proj add: {decode_mmap_layer}")
    if not bool(decode_mmap_layer.get("rope_mla_fused")):
        raise SystemExit(f"decode-layers mmap path did not fuse RoPE/MLA: {decode_mmap_layer}")
    if aggregate_mmap_count(decode_mmap) != 1:
        raise SystemExit(f"decode-layers mmap aggregate count is wrong: {decode_mmap}")
    got_decode_baseline = read_f32(root / "decode_baseline_out.f32")
    got_decode_mmap = read_f32(root / "decode_mmap_out.f32")
    max_decode_mmap_diff = max(
        math.fabs(a - b) for a, b in zip(got_decode_baseline, got_decode_mmap)
    )
    if max_decode_mmap_diff > 2e-3:
        raise SystemExit(f"decode-layers mmap output differs by {max_decode_mmap_diff}")
    if (
        root / "decode_baseline_cache.bin"
    ).read_bytes() != (root / "decode_mmap_cache.bin").read_bytes():
        raise SystemExit("decode-layers mmap cache write differs from baseline")
    print(f"fixture: {root}")
    print(f"  mmap max diff: {max_mmap_diff:.6g}")
    print(f"  decode-layers mmap max diff: {max_decode_mmap_diff:.6g}")
    print(f"  max output diff: {max_diff:.6g}")
    print("  context1 dense decoder: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
