#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from decoder_layer_smoke import (
    attention_reference,
    f32,
    mlp_reference,
    read_f32,
    rmsnorm,
    rotate,
    write_experts,
    write_resident,
)


RMS_NORM_EPS = 1e-6


def bf16(values: list[float]) -> bytes:
    from decoder_layer_smoke import bf16 as encode

    return encode(values)


def duplicate_expert_layer(root: Path) -> None:
    experts = root / "experts"
    shutil.copyfile(experts / "layer_001.bin", experts / "layer_002.bin")
    layout_path = experts / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layer2 = dict(layout["layers"][0])
    layer2["layer"] = 2
    layer2["layer_file"] = "layer_002.bin"
    layout["layers"].append(layer2)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def duplicate_resident_layer(root: Path) -> None:
    resident = root / "resident"
    layout_path = resident / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    payload = bytearray((resident / "resident.bin").read_bytes())
    original = list(layout["tensors"])
    for tensor in original:
        data = payload[tensor["offset"] : tensor["offset"] + tensor["size"]]
        copy = dict(tensor)
        copy["name"] = copy["name"].replace(".layers.1.", ".layers.2.")
        copy["offset"] = len(payload)
        payload.extend(data)
        layout["tensors"].append(copy)
    layout["total_bytes"] = len(payload)
    (resident / "resident.bin").write_bytes(payload)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def add_final_logits_tensors(root: Path) -> None:
    resident = root / "resident"
    layout_path = resident / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    payload = bytearray((resident / "resident.bin").read_bytes())

    def add(name: str, dtype: str, shape: list[int], data: bytes, category: str) -> None:
        layout["tensors"].append(
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

    add(
        "model.embed_tokens.weight",
        "F32",
        [4, 8],
        f32(
            [1.0] * 8
            + [2.0] * 8
            + [3.0] * 8
            + [4.0] * 8
        ),
        "embeddings",
    )
    add("model.norm.weight", "F32", [8], f32([1.0] * 8), "norms")
    rows = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    add(
        "lm_head.weight",
        "F32",
        [4, 8],
        f32([value for row in rows for value in row]),
        "lm_head",
    )
    layout["total_bytes"] = len(payload)
    (resident / "resident.bin").write_bytes(payload)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def write_two_layer_cache(root: Path) -> list[float]:
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
                        "layer": 1,
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
                        "layer": 2,
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


def layer_reference(
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
    attn_out = attention_reference(
        q_nope,
        q_rope,
        hidden,
        [previous_cache, current_cache],
        kv_b,
        o_proj,
    )
    return mlp_reference(attn_out, eps=RMS_NORM_EPS)


def run(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-decode-layers-", dir="/private/tmp"))
    write_experts(root)
    duplicate_expert_layer(root)
    kv_b, o_proj = write_resident(root)
    duplicate_resident_layer(root)
    add_final_logits_tensors(root)
    previous_cache = write_two_layer_cache(root)
    hidden = [1.0] * 8
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "num_hidden_layers": 3,
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    runner = str(Path(__file__).with_name("largerlm-runner"))
    mla_kv_b_cache_dir = root / "mla-kv-b-cache"

    run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "embed-token",
            str(root / "resident" / "layout.json"),
            "--token-id",
            "0",
            "--output-f32",
            str(root / "input.f32"),
            "--max-row-mib",
            "1",
        ]
    )
    run(
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
            runner,
            "--layers",
            "1-2",
            "--input-f32",
            str(root / "input.f32"),
            "--output-f32",
            str(root / "output.f32"),
            "--position",
            "1",
            "--context-length",
            "2",
            "--mla-kv-b-cache-dir",
            str(mla_kv_b_cache_dir),
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
    run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "final-logits",
            str(root / "resident" / "layout.json"),
            "--input-f32",
            str(root / "output.f32"),
            "--output-topk-json",
            str(root / "topk.json"),
            "--top-k",
            "2",
            "--rms-norm-eps",
            "0",
            "--max-chunk-mib",
            "1",
        ]
    )

    expected1 = layer_reference(hidden, previous_cache, kv_b, o_proj)
    expected2 = layer_reference(expected1, previous_cache, kv_b, o_proj)
    got = read_f32(root / "output.f32", 8)
    if any(abs(a - b) > 5e-3 for a, b in zip(got, expected2)):
        raise SystemExit(f"decode layers output {got} != expected {expected2}")
    topk = json.loads((root / "topk.json").read_text(encoding="utf-8"))
    if [item["token_id"] for item in topk["topk"]] != [2, 0]:
        raise SystemExit(f"final top-k mismatch: {topk}")
    generated_json = run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "generate-token-ids",
            str(root / "experts" / "layout.json"),
            str(root / "resident" / "layout.json"),
            str(root / "cache_layout.json"),
            str(root / "decode_cache.bin"),
            "--model-config",
            str(root / "config.json"),
            "--runner",
            runner,
            "--layers",
            "1-2",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--prefill-mla-kv-b-cache-dir",
            str(mla_kv_b_cache_dir),
            "--max-k",
            "2",
            "--routed-scaling-factor",
            "1",
            "--logits-top-k",
            "2",
            "--metal-final-logits",
            "--max-embedding-row-mib",
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
            "--json",
        ]
    )
    generated = json.loads(generated_json)
    if generated["generated_token_ids"] != [2]:
        raise SystemExit(f"generated token mismatch: {generated}")
    print(f"fixture: {root}")
    print("  embedding:          ok")
    print("  decode layers:      ok")
    print("  final logits:       ok")
    print("  greedy generate:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
