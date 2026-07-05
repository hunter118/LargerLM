#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from decoder_layer_smoke import (
    attention_reference,
    f32,
    read_f32,
    rmsnorm,
    rotate,
    write_experts,
    write_resident,
)
from dense_decoder_layer_smoke import dense_mlp_reference
from decode_layers_smoke import bf16, layer_reference


RMS_NORM_EPS = 1e-6


def add_dense_layer0(root: Path) -> None:
    resident = root / "resident"
    layout_path = resident / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    payload = bytearray((resident / "resident.bin").read_bytes())
    original = list(layout["tensors"])
    for tensor in original:
        name = tensor["name"]
        if ".layers.1." not in name:
            continue
        if ".self_attn." not in name and "layernorm.weight" not in name:
            continue
        data = payload[tensor["offset"] : tensor["offset"] + tensor["size"]]
        copy = dict(tensor)
        copy["name"] = name.replace(".layers.1.", ".layers.0.")
        copy["offset"] = len(payload)
        payload.extend(data)
        layout["tensors"].append(copy)

    dense = f32([1.0] * 64)
    for component in ("gate_proj", "up_proj", "down_proj"):
        layout["tensors"].append(
            {
                "name": f"model.layers.0.mlp.{component}.weight",
                "offset": len(payload),
                "size": len(dense),
                "dtype": "F32",
                "shape": [8, 8],
                "category": "dense_mlp",
            }
        )
        payload.extend(dense)

    layout["total_bytes"] = len(payload)
    (resident / "resident.bin").write_bytes(payload)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def write_mixed_cache(root: Path) -> list[float]:
    previous = [0.0, 1.0, -0.5, 0.25]
    cache = bytearray(80)
    cache[0:16] = bf16(previous + [0.0, 0.0, 0.0, 0.0])
    cache[64:80] = bf16(previous + [0.0, 0.0, 0.0, 0.0])
    (root / "decode_cache.bin").write_bytes(cache)
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 2,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 80,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 0,
                        "offset": 0,
                        "width": 4,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": 8,
                        "max_context_tokens": 2,
                        "total_bytes": 16,
                    },
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 64,
                        "width": 4,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": 8,
                        "max_context_tokens": 2,
                        "total_bytes": 16,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return previous


def run(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def attention_block_reference(
    hidden: list[float],
    previous_cache: list[float],
    kv_b: list[list[float]],
    o_proj: list[list[float]],
) -> list[float]:
    normed = rmsnorm(hidden, RMS_NORM_EPS)
    q_a_norm = rmsnorm([normed[0], normed[1]], RMS_NORM_EPS)
    q_b_values = [
        q_a_norm[0],
        -q_a_norm[1],
        q_a_norm[0] + q_a_norm[1],
        -q_a_norm[0],
        0.5 * q_a_norm[0] + 0.5 * q_a_norm[1],
        -q_a_norm[1],
    ]
    q_nope = [q_b_values[0], q_b_values[3]]
    q_rope = rotate([q_b_values[1], q_b_values[2]], 1) + rotate(
        [q_b_values[4], q_b_values[5]], 1
    )
    current_cache = [normed[0], 2.0 * normed[1], 0.5 * normed[0], normed[1]]
    return attention_reference(
        q_nope,
        q_rope,
        hidden,
        [previous_cache, current_cache],
        kv_b,
        o_proj,
    )


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-mixed-decode-layers-", dir="/private/tmp"))
    write_experts(root)
    kv_b, o_proj = write_resident(root)
    add_dense_layer0(root)
    previous_cache = write_mixed_cache(root)
    hidden = [1.0] * 8
    (root / "input.f32").write_bytes(f32(hidden))
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "num_attention_heads": 2,
                "kv_lora_rank": 2,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
                "scoring_func": "raw",
                "rms_norm_eps": RMS_NORM_EPS,
                "rope_parameters": {"rope_theta": 10000.0},
                "mlp_layer_types": ["dense", "sparse"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stdout = run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "decode-layers",
            str(root / "experts" / "layout.json"),
            str(root / "resident" / "layout.json"),
            str(root / "cache_layout.json"),
            str(root / "decode_cache.bin"),
            "--model-config",
            str(root / "config.json"),
            "--runner",
            str(Path(__file__).with_name("largerlm-runner")),
            "--input-f32",
            str(root / "input.f32"),
            "--output-f32",
            str(root / "output.f32"),
            "--position",
            "1",
            "--context-length",
            "2",
            "--max-k",
            "2",
            "--routed-scaling-factor",
            "1",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
        ]
    )

    if "dense layers:          0" not in stdout:
        raise SystemExit(f"dense layer summary missing from output:\n{stdout}")
    dense_attn = attention_block_reference(hidden, previous_cache, kv_b, o_proj)
    dense_out = dense_mlp_reference(dense_attn, eps=RMS_NORM_EPS)
    expected = layer_reference(dense_out, previous_cache, kv_b, o_proj)
    got = read_f32(root / "output.f32", 8)
    if any(abs(a - b) > 5e-3 for a, b in zip(got, expected)):
        raise SystemExit(f"mixed decode output {got} != expected {expected}")
    print(f"fixture: {root}")
    print("  mixed decode layers: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
