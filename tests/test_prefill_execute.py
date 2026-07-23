from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import pytest

from largerlm.cli import main as cli_main
from largerlm.prefill_execute import (
    AttentionOutputBatchServerSession,
    AttentionProjectionsServerSession,
    MLAAttentionBatchServerSession,
    PrefillExecuteError,
    ResidentBatchLinearResult,
    ResidentBatchLinearServerSession,
    ResidentBatchRMSNormServerSession,
    ResidentSharedExpertBatchServerSession,
    ResidentLinearCalibrationCase,
    ResidentLinearCalibrationResult,
    RopeSplitBatchServerSession,
    _resolve_prefill_linear_backend,
    run_prefill_attention_block_batch,
    run_prefill_attention_projection_batch,
    run_prefill_attention_output_batch,
    run_prefill_attention_prefix_batch,
    run_prefill_dense_mlp_block_batch,
    run_prefill_mla_attention_batch,
    run_prefill_routed_mlp_block_batch,
    run_prefill_rope_batch,
    run_resident_batch_linear,
    run_resident_linear_calibration,
    run_resident_batch_rmsnorm,
    run_resident_shared_expert_batch,
)


def _write_resident(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir()
    matrix = struct.pack("<6f", 1.0, 1.0, 1.0, 2.0, 0.0, 1.0)
    kv_matrix = struct.pack(
        "<12f",
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    norm = struct.pack("<3f", 1.0, 2.0, 0.5)
    q_norm = struct.pack("<2f", 1.0, 1.0)
    q_b = struct.pack("<8f", 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0)
    kv_norm = struct.pack("<2f", 1.0, 1.0)
    kv_b = struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 2.0, 0.0)
    o_proj = struct.pack("<6f", 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    post_norm = struct.pack("<3f", 1.0, 1.0, 1.0)
    dense_gate = struct.pack("<6f", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    dense_up = struct.pack("<6f", 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    dense_down = struct.pack("<6f", 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    dsa_wk = struct.pack("<6f", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    dsa_k_norm_weight = struct.pack("<2f", 1.0, 1.0)
    dsa_k_norm_bias = struct.pack("<2f", 0.0, 0.0)
    dsa_wq_b = struct.pack("<4f", 1.0, 0.0, 0.0, 1.0)
    dsa_weights_proj = struct.pack("<3f", 1.0, 0.0, 0.0)
    dsa_wk_offset = (
        len(matrix)
        + len(kv_matrix)
        + len(norm)
        + len(q_norm)
        + len(q_b)
        + len(kv_norm)
        + len(kv_b)
        + len(o_proj)
        + len(post_norm)
        + len(dense_gate)
        + len(dense_up)
        + len(dense_down)
    )
    dsa_k_norm_weight_offset = dsa_wk_offset + len(dsa_wk)
    dsa_k_norm_bias_offset = dsa_k_norm_weight_offset + len(dsa_k_norm_weight)
    dsa_wq_b_offset = dsa_k_norm_bias_offset + len(dsa_k_norm_bias)
    dsa_weights_proj_offset = dsa_wq_b_offset + len(dsa_wq_b)
    payload = (
        matrix
        + kv_matrix
        + norm
        + q_norm
        + q_b
        + kv_norm
        + kv_b
        + o_proj
        + post_norm
        + dense_gate
        + dense_up
        + dense_down
        + dsa_wk
        + dsa_k_norm_weight
        + dsa_k_norm_bias
        + dsa_wq_b
        + dsa_weights_proj
    )
    (resident / "resident.bin").write_bytes(payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": [
            {
                "name": "model.layers.1.self_attn.q_a_proj.weight",
                "offset": 0,
                "size": len(matrix),
                "dtype": "F32",
                "shape": [2, 3],
                "category": "attention",
            },
            {
                "name": "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                "offset": len(matrix),
                "size": len(kv_matrix),
                "dtype": "F32",
                "shape": [4, 3],
                "category": "attention",
            },
            {
                "name": "model.layers.1.input_layernorm.weight",
                "offset": len(matrix) + len(kv_matrix),
                "size": len(norm),
                "dtype": "F32",
                "shape": [3],
                "category": "attention",
            },
            {
                "name": "model.layers.1.self_attn.q_a_layernorm.weight",
                "offset": len(matrix) + len(kv_matrix) + len(norm),
                "size": len(q_norm),
                "dtype": "F32",
                "shape": [2],
                "category": "norms",
            },
            {
                "name": "model.layers.1.self_attn.q_b_proj.weight",
                "offset": len(matrix) + len(kv_matrix) + len(norm) + len(q_norm),
                "size": len(q_b),
                "dtype": "F32",
                "shape": [4, 2],
                "category": "attention",
            },
            {
                "name": "model.layers.1.self_attn.kv_a_layernorm.weight",
                "offset": (
                    len(matrix) + len(kv_matrix) + len(norm) + len(q_norm) + len(q_b)
                ),
                "size": len(kv_norm),
                "dtype": "F32",
                "shape": [2],
                "category": "norms",
            },
            {
                "name": "model.layers.1.self_attn.kv_b_proj.weight",
                "offset": (
                    len(matrix)
                    + len(kv_matrix)
                    + len(norm)
                    + len(q_norm)
                    + len(q_b)
                    + len(kv_norm)
                ),
                "size": len(kv_b),
                "dtype": "F32",
                "shape": [4, 2],
                "category": "attention",
            },
            {
                "name": "model.layers.1.self_attn.o_proj.weight",
                "offset": (
                    len(matrix)
                    + len(kv_matrix)
                    + len(norm)
                    + len(q_norm)
                    + len(q_b)
                    + len(kv_norm)
                    + len(kv_b)
                ),
                "size": len(o_proj),
                "dtype": "F32",
                "shape": [3, 2],
                "category": "attention",
            },
            {
                "name": "model.layers.1.post_attention_layernorm.weight",
                "offset": (
                    len(matrix)
                    + len(kv_matrix)
                    + len(norm)
                    + len(q_norm)
                    + len(q_b)
                    + len(kv_norm)
                    + len(kv_b)
                    + len(o_proj)
                ),
                "size": len(post_norm),
                "dtype": "F32",
                "shape": [3],
                "category": "norms",
            },
            {
                "name": "model.layers.1.mlp.gate_proj.weight",
                "offset": (
                    len(matrix)
                    + len(kv_matrix)
                    + len(norm)
                    + len(q_norm)
                    + len(q_b)
                    + len(kv_norm)
                    + len(kv_b)
                    + len(o_proj)
                    + len(post_norm)
                ),
                "size": len(dense_gate),
                "dtype": "F32",
                "shape": [2, 3],
                "category": "dense_mlp",
            },
            {
                "name": "model.layers.1.mlp.up_proj.weight",
                "offset": (
                    len(matrix)
                    + len(kv_matrix)
                    + len(norm)
                    + len(q_norm)
                    + len(q_b)
                    + len(kv_norm)
                    + len(kv_b)
                    + len(o_proj)
                    + len(post_norm)
                    + len(dense_gate)
                ),
                "size": len(dense_up),
                "dtype": "F32",
                "shape": [2, 3],
                "category": "dense_mlp",
            },
            {
                "name": "model.layers.1.mlp.down_proj.weight",
                "offset": (
                    len(matrix)
                    + len(kv_matrix)
                    + len(norm)
                    + len(q_norm)
                    + len(q_b)
                    + len(kv_norm)
                    + len(kv_b)
                    + len(o_proj)
                    + len(post_norm)
                    + len(dense_gate)
                    + len(dense_up)
                ),
                "size": len(dense_down),
                "dtype": "F32",
                "shape": [3, 2],
                "category": "dense_mlp",
            },
            {
                "name": "model.layers.1.self_attn.indexer.wk.weight",
                "offset": dsa_wk_offset,
                "size": len(dsa_wk),
                "dtype": "F32",
                "shape": [2, 3],
                "category": "dsa_indexer",
            },
            {
                "name": "model.layers.1.self_attn.indexer.k_norm.weight",
                "offset": dsa_k_norm_weight_offset,
                "size": len(dsa_k_norm_weight),
                "dtype": "F32",
                "shape": [2],
                "category": "dsa_indexer",
            },
            {
                "name": "model.layers.1.self_attn.indexer.k_norm.bias",
                "offset": dsa_k_norm_bias_offset,
                "size": len(dsa_k_norm_bias),
                "dtype": "F32",
                "shape": [2],
                "category": "dsa_indexer",
            },
            {
                "name": "model.layers.1.self_attn.indexer.wq_b.weight",
                "offset": dsa_wq_b_offset,
                "size": len(dsa_wq_b),
                "dtype": "F32",
                "shape": [2, 2],
                "category": "dsa_indexer",
            },
            {
                "name": "model.layers.1.self_attn.indexer.weights_proj.weight",
                "offset": dsa_weights_proj_offset,
                "size": len(dsa_weights_proj),
                "dtype": "F32",
                "shape": [1, 3],
                "category": "dsa_indexer",
            },
        ],
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def _write_shared_mxfp4_resident(root: Path) -> Path:
    resident = root / "shared_mxfp4_resident"
    resident.mkdir()
    tensors: list[dict[str, object]] = []
    payload = bytearray()
    for component in ("gate_proj", "up_proj", "down_proj"):
        stem = f"model.layers.1.mlp.shared_experts.{component}"
        weight = bytes([len(payload) % 251]) * (8 * 4)
        scales = bytes([7]) * 8
        weight_offset = len(payload)
        payload.extend(weight)
        scales_offset = len(payload)
        payload.extend(scales)
        tensors.append(
            {
                "name": f"{stem}.weight",
                "offset": weight_offset,
                "size": len(weight),
                "dtype": "U32",
                "shape": [8, 1],
                "category": "shared_experts",
            }
        )
        tensors.append(
            {
                "name": f"{stem}.scales",
                "offset": scales_offset,
                "size": len(scales),
                "dtype": "U8",
                "shape": [8, 1],
                "category": "shared_experts",
            }
        )
    (resident / "resident.bin").write_bytes(bytes(payload))
    layout = resident / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    return layout


def _write_input(root: Path, values: tuple[float, ...] = (1, 2, 3, 4, 5, 6)) -> Path:
    path = root / "input.f32"
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))
    return path


def _mutate_layout(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _replace_kv_b_with_absorbed_aliases(layout: Path) -> None:
    payload = json.loads(layout.read_text(encoding="utf-8"))
    tensors = payload["tensors"]
    assert isinstance(tensors, list)
    tensors[:] = [
        tensor
        for tensor in tensors
        if not tensor["name"].endswith(".self_attn.kv_b_proj.weight")
    ]
    resident_bin = layout.parent / str(payload["weight_file"])
    resident_payload = bytearray(resident_bin.read_bytes())
    for name, shape in (
        ("model.layers.1.self_attn.embed_q.weight", [1, 2, 2]),
        ("model.layers.1.self_attn.unembed_out.weight", [1, 2, 2]),
    ):
        data = struct.pack("<4f", *([1.0] * 4))
        tensors.append(
            {
                "name": name,
                "offset": len(resident_payload),
                "size": len(data),
                "dtype": "F32",
                "shape": shape,
                "category": "attention",
            }
        )
        resident_payload.extend(data)
    payload["total_bytes"] = len(resident_payload)
    resident_bin.write_bytes(bytes(resident_payload))
    layout.write_text(json.dumps(payload), encoding="utf-8")


def _write_large_linear_fixture(root: Path, *, dtype: str = "F32") -> tuple[Path, Path]:
    resident = root / "large_resident"
    resident.mkdir()
    matrix_count = 32 * 32
    if dtype == "F32":
        matrix_values = [0.0] * matrix_count
        matrix = struct.pack(f"<{len(matrix_values)}f", *matrix_values)
    elif dtype == "BF16":
        matrix = struct.pack(f"<{matrix_count}H", *([0] * matrix_count))
    else:
        raise AssertionError(f"unsupported test dtype {dtype}")
    (resident / "resident.bin").write_bytes(matrix)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(matrix),
        "tensors": [
            {
                "name": "model.layers.1.self_attn.q_a_proj.weight",
                "offset": 0,
                "size": len(matrix),
                "dtype": dtype,
                "shape": [32, 32],
                "category": "attention",
            }
        ],
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    input_path = root / "large_input.f32"
    input_values = [1.0] * (128 * 32)
    input_path.write_bytes(struct.pack(f"<{len(input_values)}f", *input_values))
    return layout_path, input_path


def _write_cache(root: Path) -> tuple[Path, Path]:
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "max_context_tokens": 4,
        "dtype": "BF16",
        "dtype_bytes": 2,
        "alignment": 64,
        "total_bytes": 48,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": 1,
                "offset": 0,
                "width": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "token_stride_bytes": 8,
                "max_context_tokens": 4,
                "total_bytes": 32,
            },
            {
                "kind": "dsa_index",
                "layer": 1,
                "offset": 32,
                "width": 2,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "token_stride_bytes": 4,
                "max_context_tokens": 4,
                "total_bytes": 16,
            }
        ],
    }
    layout_path = root / "cache_layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    cache_path = root / "decode_cache.bin"
    cache_path.write_bytes(b"\0" * 48)
    return layout_path, cache_path


def _write_moe_fixture(root: Path) -> tuple[Path, Path]:
    experts = root / "experts"
    experts.mkdir()
    components = [
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
    (experts / "layer_001.bin").write_bytes(b"\0" * (2 * 192))
    expert_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "quantization": "mlx-affine-int4",
        "group_size": 8,
        "layers": [
            {
                "layer": 1,
                "num_experts": 2,
                "expert_slot_bytes": 192,
                "layer_file": "layer_001.bin",
                "components": [
                    {
                        "name": name,
                        "offset": offset,
                        "size": size,
                        "dtype": dtype,
                        "shape": shape,
                    }
                    for name, offset, size, dtype, shape in components
                ],
            }
        ],
    }
    expert_layout_path = experts / "layout.json"
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")

    resident = root / "moe_resident"
    resident.mkdir()
    router = struct.pack("<16f", *([1.0] * 8 + [2.0] * 8))
    norm = struct.pack("<8f", *([1.0] * 8))
    payload = router + norm
    resident_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": [
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
        ],
    }
    (resident / "resident.bin").write_bytes(payload)
    resident_layout_path = resident / "layout.json"
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")
    return expert_layout_path, resident_layout_path


def _write_fake_runner(root: Path) -> Path:
    runner = root / "fake-runner"
    runner.write_text(
        """#!/usr/bin/env python3
import json
import os
import struct
import sys
from pathlib import Path

args = sys.argv[1:]
if "--run-attn-projections-server-jsonl" in args:
    print("LargerLM attention projections server")
    print("  protocol:           jsonl")
    sys.stdout.flush()
    requests = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("command") == "quit":
            break
        requests += 1
        out_dir = Path(request.get("output_dir") or request.get("output_dir_path"))
        batch = int(request["batch_tokens"])
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "attn_input_norm.f32": (1.0, 2.0, 3.0),
            "attn_q_a.f32": (6.0, 5.0),
            "attn_q_a_norm.f32": (0.76822, 0.64018),
            "attn_q_b.f32": (0.76822, 0.64018, 1.40840, 3.45700),
            "attn_kv_a.f32": (1.0, 2.0, 3.0, 6.0),
            "attn_kv_a_norm.f32": (0.44721, 0.89443),
            "attn_kv_b.f32": (2.23607, 4.91935, 0.67082, 0.89443),
        }
        for name, values in outputs.items():
            path = out_dir / name
            expanded = values * batch
            path.write_bytes(struct.pack(f"<{len(expanded)}f", *expanded))
        argv = [
            "--run-attn-projections-server-jsonl",
            "--resident-layout",
            str(Path(request.get("resident_layout") or request.get("resident_layout_path"))),
            "--layer",
            str(request["layer"]),
            "--input-f32",
            str(Path(request.get("input_f32") or request.get("input_f32_path"))),
            "--batch-tokens",
            str(batch),
            "--output-dir",
            str(out_dir),
        ]
        (out_dir / "attn_q_b.f32.argv.json").write_text(json.dumps(argv))
        print("LargerLM attention projections server request")
        print(f"  request:             {requests}")
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM attention projections server done")
    print(f"  requests:            {requests}")
    sys.exit(0)
if "--run-resident-linear-batch-plan-server-jsonl" in args:
    print("LargerLM resident batch linear server")
    print("  protocol:           jsonl")
    sys.stdout.flush()
    requests = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("command") == "quit":
            break
        requests += 1
        layout_path = Path(request.get("resident_layout") or request.get("resident_layout_path"))
        layout = json.loads(layout_path.read_text())
        suffix = request["tensor_suffix"]
        tensor = next(item for item in layout["tensors"] if item["name"].endswith(suffix))
        out_dim = int(tensor["shape"][0])
        batch = int(request["batch_tokens"])
        out = Path(request.get("output_f32") or request.get("output_f32_path"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(struct.pack(f"<{batch * out_dim}f", *([0.0] * (batch * out_dim))))
        argv = [
            "--resident-layout",
            str(layout_path),
            "--layer",
            str(request["layer"]),
            "--run-resident-linear-batch",
            "--tensor-suffix",
            suffix,
            "--input-f32",
            str(Path(request.get("input_f32") or request.get("input_f32_path"))),
            "--batch-tokens",
            str(batch),
            "--output-f32",
            str(out),
            "--max-resident-matrix-mib",
            str(request.get("max_resident_matrix_mib", 512)),
            "--max-runner-scratch-mib",
            str(request.get("max_runner_scratch_mib", 4096)),
        ]
        backend = request.get("prefill_linear_backend", "custom-metal")
        if backend != "custom-metal":
            argv.extend(["--prefill-linear-backend", str(backend)])
        out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(argv))
        print("LargerLM resident batch linear server request")
        print(f"  request:             {requests}")
        print("  timing backend:     0.123000")
        print("  timing matrix f32:  0.045000")
        print("  timing accelerator: 0.067000")
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM resident batch linear server done")
    print(f"  requests:            {requests}")
    sys.exit(0)
if "--run-rmsnorm-batch-server-jsonl" in args:
    print("LargerLM resident RMSNorm batch server")
    print("  protocol:           jsonl")
    sys.stdout.flush()
    requests = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("command") == "quit":
            break
        requests += 1
        suffix = request["norm_suffix"]
        out = Path(request.get("output_f32") or request.get("output_f32_path"))
        out.parent.mkdir(parents=True, exist_ok=True)
        if suffix.endswith("input_layernorm.weight"):
            payload = struct.pack("<6f", 0.46291, 1.85164, 0.69436, 0.91168, 2.27921, 0.68376)
        elif suffix.endswith("q_a_layernorm.weight"):
            payload = struct.pack("<4f", 0.76822, 0.64018, 0.73106, 0.68232)
        elif suffix.endswith("post_attention_layernorm.weight"):
            payload = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        else:
            payload = struct.pack("<4f", 0.44721, 0.89443, 0.62470, 0.78087)
        out.write_bytes(payload)
        argv = [
            "--run-rmsnorm-batch-server-jsonl",
            "--resident-layout",
            str(Path(request.get("resident_layout") or request.get("resident_layout_path"))),
            "--layer",
            str(request["layer"]),
            "--norm-suffix",
            suffix,
            "--input-f32",
            str(Path(request.get("input_f32") or request.get("input_f32_path"))),
            "--batch-tokens",
            str(request["batch_tokens"]),
            "--rms-norm-eps",
            str(request.get("rms_norm_eps", 1e-5)),
            "--output-f32",
            str(out),
            "--max-runner-scratch-mib",
            str(request.get("max_runner_scratch_mib", 4096)),
        ]
        out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(argv))
        print("LargerLM resident RMSNorm batch server request")
        print(f"  request:             {requests}")
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM resident RMSNorm batch server done")
    print(f"  requests:            {requests}")
    sys.exit(0)
if "--run-rope-split-batch-server-jsonl" in args:
    print("LargerLM RoPE split batch server")
    print("  protocol:           jsonl")
    sys.stdout.flush()
    requests = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("command") == "quit":
            break
        requests += 1
        q_b = Path(request.get("q_b_f32") or request.get("q_b_f32_path"))
        out_nope = Path(
            request.get("output_q_nope_f32") or request.get("output_q_nope_f32_path")
        )
        out_rope = Path(
            request.get("output_q_rope_f32") or request.get("output_q_rope_f32_path")
        )
        out_q = Path(request.get("output_q_f32") or request.get("output_q_f32_path"))
        out_k = Path(request.get("output_k_f32") or request.get("output_k_f32_path"))
        batch = int(request["batch_tokens"])
        heads = int(request["num_heads"])
        nope = int(request["qk_nope_dim"])
        rope = int(request["rope_dim"])
        raw = q_b.read_bytes()
        values = struct.unpack(f"<{len(raw) // 4}f", raw)
        q_nope = []
        q_rope = []
        head_dim = nope + rope
        for token in range(batch):
            row_base = token * heads * head_dim
            for head in range(heads):
                base = row_base + head * head_dim
                q_nope.extend(values[base : base + nope])
                q_rope.extend(values[base + nope : base + head_dim])
        for out in (out_nope, out_rope, out_q, out_k):
            out.parent.mkdir(parents=True, exist_ok=True)
        out_nope.write_bytes(struct.pack(f"<{len(q_nope)}f", *q_nope))
        out_rope.write_bytes(struct.pack(f"<{len(q_rope)}f", *q_rope))
        out_q.write_bytes(
            struct.pack(f"<{batch * heads * rope}f", *([0.25] * (batch * heads * rope)))
        )
        out_k.write_bytes(struct.pack(f"<{batch * rope}f", *([0.5] * (batch * rope))))
        argv = [
            "--run-rope-split-batch-server-jsonl",
            "--q-b-f32",
            str(q_b),
            "--k-f32",
            str(Path(request.get("k_f32") or request.get("k_f32_path"))),
            "--output-q-nope-f32",
            str(out_nope),
            "--output-q-rope-f32",
            str(out_rope),
            "--output-q-f32",
            str(out_q),
            "--output-k-f32",
            str(out_k),
            "--num-heads",
            str(heads),
            "--qk-nope-dim",
            str(nope),
            "--rope-dim",
            str(rope),
            "--start-position",
            str(request["start_position"]),
            "--batch-tokens",
            str(batch),
            "--rope-theta",
            str(request.get("rope_theta", 10000)),
            "--max-runner-scratch-mib",
            str(request.get("max_runner_scratch_mib", 4096)),
        ]
        if request.get("rope_interleave"):
            argv.append("--rope-interleave")
        out_q.with_suffix(out_q.suffix + ".argv.json").write_text(json.dumps(argv))
        print("LargerLM RoPE split batch server request")
        print(f"  request:             {requests}")
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM RoPE split batch server done")
    print(f"  requests:            {requests}")
    sys.exit(0)
if "--run-rope-split-batch" in args:
    q_b = Path(args[args.index("--q-b-f32") + 1])
    out_nope = Path(args[args.index("--output-q-nope-f32") + 1])
    out_rope = Path(args[args.index("--output-q-rope-f32") + 1])
    out_q = Path(args[args.index("--output-q-f32") + 1])
    out_k = Path(args[args.index("--output-k-f32") + 1])
    batch = int(args[args.index("--batch-tokens") + 1])
    heads = int(args[args.index("--num-heads") + 1])
    nope = int(args[args.index("--qk-nope-dim") + 1])
    rope = int(args[args.index("--rope-dim") + 1])
    raw = q_b.read_bytes()
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    q_nope = []
    q_rope = []
    head_dim = nope + rope
    for token in range(batch):
        row_base = token * heads * head_dim
        for head in range(heads):
            base = row_base + head * head_dim
            q_nope.extend(values[base : base + nope])
            q_rope.extend(values[base + nope : base + head_dim])
    out_nope.write_bytes(struct.pack(f"<{len(q_nope)}f", *q_nope))
    out_rope.write_bytes(struct.pack(f"<{len(q_rope)}f", *q_rope))
    out_q.write_bytes(struct.pack(f"<{batch * heads * rope}f", *([0.25] * (batch * heads * rope))))
    out_k.write_bytes(struct.pack(f"<{batch * rope}f", *([0.5] * (batch * rope))))
    out_q.with_suffix(out_q.suffix + ".argv.json").write_text(json.dumps(args))
    sys.exit(0)
if "--run-rope-batch" in args:
    out_q = Path(args[args.index("--output-q-f32") + 1])
    out_k = Path(args[args.index("--output-k-f32") + 1])
    batch = int(args[args.index("--batch-tokens") + 1])
    heads = int(args[args.index("--num-heads") + 1])
    rope = int(args[args.index("--rope-dim") + 1])
    out_q.write_bytes(struct.pack(f"<{batch * heads * rope}f", *([0.25] * (batch * heads * rope))))
    out_k.write_bytes(struct.pack(f"<{batch * rope}f", *([0.5] * (batch * rope))))
    out_q.with_suffix(out_q.suffix + ".argv.json").write_text(json.dumps(args))
    sys.exit(0)
if "--run-attn-projections" in args:
    out_dir = Path(args[args.index("--output-dir") + 1])
    batch = int(args[args.index("--batch-tokens") + 1]) if "--batch-tokens" in args else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "attn_input_norm.f32": (1.0, 2.0, 3.0),
        "attn_q_a.f32": (6.0, 5.0),
        "attn_q_a_norm.f32": (0.76822, 0.64018),
        "attn_q_b.f32": (0.76822, 0.64018, 1.40840, 3.45700),
        "attn_kv_a.f32": (1.0, 2.0, 3.0, 6.0),
        "attn_kv_a_norm.f32": (0.44721, 0.89443),
        "attn_kv_b.f32": (2.23607, 4.91935, 0.67082, 0.89443),
    }
    for name, values in outputs.items():
        path = out_dir / name
        expanded = values * batch
        path.write_bytes(struct.pack(f"<{len(expanded)}f", *expanded))
    (out_dir / "attn_q_b.f32.argv.json").write_text(json.dumps(args))
    sys.exit(0)
if "--run-mla-attention-batch-server-jsonl" in args:
    print("LargerLM MLA attention batch server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM MLA attention batch server request")
        batch = int(request["batch_tokens"])
        heads = int(request["num_heads"])
        v_head = int(request["v_head_dim"])
        out = Path(request.get("output_f32") or request.get("output_f32_path"))
        out.parent.mkdir(parents=True, exist_ok=True)
        value = 0.625 if request.get("indices_u32") or request.get("indices_u32_path") else 0.75
        out.write_bytes(struct.pack(f"<{batch * heads * v_head}f", *([value] * (batch * heads * v_head))))
        out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(["--run-mla-attention-batch-server-jsonl"]))
        out.with_suffix(out.suffix + ".env.json").write_text(json.dumps({
            "LARGERLM_MLA_KEY_CACHE": os.environ.get("LARGERLM_MLA_KEY_CACHE"),
            "LARGERLM_MLA_DISABLE_VALUE_CACHE": os.environ.get("LARGERLM_MLA_DISABLE_VALUE_CACHE"),
        }))
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM MLA attention batch server done")
    sys.exit(0)
if "--run-attn-output-batch-server-jsonl" in args:
    print("LargerLM attention output batch server")
    print("  protocol:           jsonl")
    sys.stdout.flush()
    requests = 0
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        requests += 1
        out = Path(request.get("output_f32") or request.get("output_f32_path"))
        residual_path = Path(
            request.get("residual_f32") or request.get("residual_f32_path")
        )
        residual_raw = residual_path.read_bytes()
        count = len(residual_raw) // 4
        residual = struct.unpack(f"<{count}f", residual_raw)
        projection = tuple(float(i + 1) for i in range(count))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(
            struct.pack(f"<{count}f", *(a + b for a, b in zip(residual, projection)))
        )
        projection_path_raw = request.get("projection_f32") or request.get(
            "projection_f32_path"
        )
        if projection_path_raw:
            projection_path = Path(projection_path_raw)
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            projection_path.write_bytes(struct.pack(f"<{count}f", *projection))
            projection_path.with_suffix(
                projection_path.suffix + ".argv.json"
            ).write_text(json.dumps(["--run-attn-output-batch-server-jsonl"]))
        out.with_suffix(out.suffix + ".argv.json").write_text(
            json.dumps(["--run-attn-output-batch-server-jsonl"])
        )
        print("LargerLM attention output batch server request")
        print(f"  request:             {requests}")
        print("  timing backend:     0.031000")
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM attention output batch server done")
    print(f"  requests:            {requests}")
    sys.exit(0)
if "--run-shared-expert-batch-server-jsonl" in args:
    print("LargerLM resident shared expert batch server")
    print("  protocol:           jsonl")
    sys.stdout.flush()
    requests = 0
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        requests += 1
        input_path = Path(request.get("input_f32") or request.get("input_f32_path"))
        out = Path(request.get("output_f32") or request.get("output_f32_path"))
        raw = input_path.read_bytes()
        count = len(raw) // 4
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(struct.pack(f"<{count}f", *([2.5] * count)))
        out.with_suffix(out.suffix + ".argv.json").write_text(
            json.dumps(["--run-shared-expert-batch-server-jsonl"])
        )
        print("LargerLM resident shared expert batch server request")
        print(f"  request:             {requests}")
        print("  timing backend:     0.044000")
        print("  server request:      ok")
        sys.stdout.flush()
    print("LargerLM resident shared expert batch server done")
    print(f"  requests:            {requests}")
    sys.exit(0)
out = Path(args[args.index("--output-f32") + 1])
if "--run-attn-output" in args or "--run-attn-output-batch" in args:
    residual_path = Path(args[args.index("--residual-f32") + 1])
    residual_raw = residual_path.read_bytes()
    count = len(residual_raw) // 4
    residual = struct.unpack(f"<{count}f", residual_raw)
    projection = tuple(float(i + 1) for i in range(count))
    out.write_bytes(struct.pack(f"<{count}f", *(a + b for a, b in zip(residual, projection))))
    if "--projection-f32" in args:
        projection_path = Path(args[args.index("--projection-f32") + 1])
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        projection_path.write_bytes(struct.pack(f"<{count}f", *projection))
        projection_path.with_suffix(projection_path.suffix + ".argv.json").write_text(json.dumps(args))
    out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(args))
    print("  timing backend:     0.031000")
    sys.exit(0)
if "--run-mlp-block" in args:
    input_path = Path(args[args.index("--input-f32") + 1])
    raw = input_path.read_bytes()
    count = len(raw) // 4
    values = struct.unpack(f"<{count}f", raw)
    out.write_bytes(struct.pack(f"<{count}f", *(value + 10.0 for value in values)))
    if "--output-router-json" in args:
        router_json = Path(args[args.index("--output-router-json") + 1])
        router_json.parent.mkdir(parents=True, exist_ok=True)
        router_json.write_text(json.dumps({"experts": [1, 0], "weights": [0.75, 0.25]}))
    out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(args))
    sys.exit(0)
if "--run-mla-attention-batch" in args:
    batch = int(args[args.index("--batch-tokens") + 1])
    heads = int(args[args.index("--num-heads") + 1])
    v_head = int(args[args.index("--v-head-dim") + 1])
    out.write_bytes(struct.pack(f"<{batch * heads * v_head}f", *([0.75] * (batch * heads * v_head))))
    out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(args))
    out.with_suffix(out.suffix + ".env.json").write_text(json.dumps({
        "LARGERLM_MLA_KEY_CACHE": os.environ.get("LARGERLM_MLA_KEY_CACHE"),
        "LARGERLM_MLA_DISABLE_VALUE_CACHE": os.environ.get("LARGERLM_MLA_DISABLE_VALUE_CACHE"),
    }))
    sys.exit(0)
if "--run-mla-attention-indexed-batch" in args:
    batch = int(args[args.index("--batch-tokens") + 1])
    heads = int(args[args.index("--num-heads") + 1])
    v_head = int(args[args.index("--v-head-dim") + 1])
    out.write_bytes(struct.pack(f"<{batch * heads * v_head}f", *([0.625] * (batch * heads * v_head))))
    out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(args))
    sys.exit(0)
if "--run-rmsnorm-batch" in args:
    suffix = args[args.index("--norm-suffix") + 1]
    if suffix.endswith("input_layernorm.weight"):
        payload = struct.pack("<6f", 0.46291, 1.85164, 0.69436, 0.91168, 2.27921, 0.68376)
    elif suffix.endswith("q_a_layernorm.weight"):
        payload = struct.pack("<4f", 0.76822, 0.64018, 0.73106, 0.68232)
    elif suffix.endswith("post_attention_layernorm.weight"):
        payload = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    else:
        payload = struct.pack("<4f", 0.44721, 0.89443, 0.62470, 0.78087)
else:
    suffix = args[args.index("--tensor-suffix") + 1]
    if suffix.endswith("q_a_proj.weight"):
        payload = struct.pack("<4f", 6.0, 5.0, 15.0, 14.0)
    elif suffix.endswith("kv_a_proj_with_mqa.weight"):
        payload = struct.pack("<8f", 1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0)
    elif suffix.endswith("q_b_proj.weight"):
        payload = struct.pack("<8f", 0.76822, 0.64018, 1.40840, 3.45700, 0.73106, 0.68232, 1.41338, 3.50908)
    elif suffix.endswith("o_proj.weight"):
        batch = int(args[args.index("--batch-tokens") + 1])
        values = (1.0, 2.0, 3.0, 4.0, 5.0, 9.0)
        payload = struct.pack(f"<{batch * 3}f", *values[: batch * 3])
    elif suffix.endswith("mlp.gate_proj.weight"):
        payload = struct.pack("<4f", 0.0, 1.0, 2.0, -1.0)
    elif suffix.endswith("mlp.up_proj.weight"):
        payload = struct.pack("<4f", 10.0, 20.0, 30.0, 40.0)
    elif suffix.endswith("mlp.down_proj.weight"):
        payload = struct.pack("<6f", 0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
    else:
        payload = struct.pack("<8f", 2.23607, 4.91935, 0.67082, 0.89443, 2.96731, 4.99672, 0.70279, 1.24939)
out.write_bytes(payload)
out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(args))
if "--run-resident-linear-batch" in args:
    print("  timing backend:     0.123000")
    print("  timing matrix f32:  0.045000")
    print("  timing accelerator: 0.067000")
""",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    return runner


def _write_shape_runner(root: Path) -> Path:
    runner = root / "shape-runner"
    runner.write_text(
        """#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path

args = sys.argv[1:]
layout = json.loads(Path(args[args.index("--resident-layout") + 1]).read_text())
suffix = args[args.index("--tensor-suffix") + 1]
batch = int(args[args.index("--batch-tokens") + 1])
out = Path(args[args.index("--output-f32") + 1])
tensor = next(item for item in layout["tensors"] if item["name"].endswith(suffix))
out_dim = int(tensor["shape"][0])
out.write_bytes(struct.pack(f"<{batch * out_dim}f", *([0.0] * (batch * out_dim))))
out.with_suffix(out.suffix + ".argv.json").write_text(json.dumps(args))
""",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    return runner


def test_run_resident_batch_linear_invokes_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        echo_runner_output=False,
    )

    assert result.tensor == "model.layers.1.self_attn.q_a_proj.weight"
    assert result.in_dim == 3
    assert result.out_dim == 2
    assert result.input_bytes == 24
    assert result.output_bytes == 16
    assert result.matrix_scratch_bytes == 2 * 1024 * 1024
    assert result.matrix_f32_bytes == 0
    assert result.matrix_raw_conversion_bytes == 0
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 24 + 16
    assert result.elapsed_seconds >= 0.0
    assert result.runner_backend_elapsed_seconds == pytest.approx(0.123)
    assert result.runner_matrix_f32_elapsed_seconds == pytest.approx(0.045)
    assert result.runner_accelerator_elapsed_seconds == pytest.approx(0.067)
    assert result.backend == "custom-metal"
    assert struct.unpack("<4f", output.read_bytes()) == (6.0, 5.0, 15.0, 14.0)
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert "--run-resident-linear-batch" in argv
    assert argv[argv.index("--batch-tokens") + 1] == "2"
    assert "--prefill-linear-backend" not in argv


def test_run_resident_batch_linear_can_use_server_session(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    first_output = tmp_path / "server-out-1.f32"
    second_output = tmp_path / "server-out-2.f32"

    with ResidentBatchLinearServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        first = run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=first_output,
            batch_tokens=2,
            echo_runner_output=False,
            resident_linear_server_session=session,
        )
        second = run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=second_output,
            batch_tokens=2,
            prefill_linear_backend="mpsgraph-f32",
            echo_runner_output=False,
            resident_linear_server_session=session,
        )
        assert session.running

    assert first.command == (
        str(runner),
        "--run-resident-linear-batch-plan-server-jsonl",
    )
    assert second.command == first.command
    assert first.runner_backend_elapsed_seconds == pytest.approx(0.123)
    assert second.runner_matrix_f32_elapsed_seconds == pytest.approx(0.045)
    assert first_output.stat().st_size == first.output_bytes
    assert second_output.stat().st_size == second.output_bytes
    first_argv = json.loads(
        first_output.with_suffix(first_output.suffix + ".argv.json").read_text()
    )
    second_argv = json.loads(
        second_output.with_suffix(second_output.suffix + ".argv.json").read_text()
    )
    assert "--run-resident-linear-batch" in first_argv
    assert "--prefill-linear-backend" not in first_argv
    assert second_argv[second_argv.index("--prefill-linear-backend") + 1] == "mpsgraph-f32"


def test_run_resident_batch_linear_can_select_mpsgraph_backend(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        prefill_linear_backend="mpsgraph-f32",
        echo_runner_output=False,
    )

    assert result.backend == "mpsgraph-f32"
    assert result.matrix_scratch_bytes == 2 * 1024 * 1024
    assert result.matrix_f32_bytes == 2 * 3 * 4
    assert result.matrix_raw_conversion_bytes == 0
    assert result.runner_backend_elapsed_seconds == pytest.approx(0.123)
    assert result.runner_matrix_f32_elapsed_seconds == pytest.approx(0.045)
    assert result.runner_accelerator_elapsed_seconds == pytest.approx(0.067)
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--prefill-linear-backend") + 1] == "mpsgraph-f32"


def test_run_resident_batch_linear_can_select_mpp_backend(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        prefill_linear_backend="mpp-f32",
        echo_runner_output=False,
    )

    assert result.backend == "mpp-f32"
    assert result.matrix_scratch_bytes == 2 * 1024 * 1024
    assert result.matrix_f32_bytes == 2 * 3 * 4
    assert result.matrix_raw_conversion_bytes == 0
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--prefill-linear-backend") + 1] == "mpp-f32"


def test_run_resident_batch_linear_can_select_mps_matrix_backend(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        prefill_linear_backend="mps-matrix-f32",
        echo_runner_output=False,
    )

    assert result.backend == "mps-matrix-f32"
    assert result.matrix_scratch_bytes == 2 * 1024 * 1024
    assert result.matrix_f32_bytes == 2 * 3 * 4
    assert result.matrix_raw_conversion_bytes == 0
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--prefill-linear-backend") + 1] == "mps-matrix-f32"


def test_run_resident_batch_linear_auto_keeps_custom_for_tiny_f32(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        prefill_linear_backend="auto",
        echo_runner_output=False,
    )

    assert result.backend == "custom-metal"
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert "--prefill-linear-backend" not in argv


def test_run_resident_batch_linear_auto_thresholds_are_configurable(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=2,
        prefill_mpsgraph_min_matrix_dim=2,
        echo_runner_output=False,
    )

    assert result.backend == "mpsgraph-f32"
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--prefill-linear-backend") + 1] == "mpsgraph-f32"


def test_run_resident_linear_calibration_builds_bounded_cases(tmp_path: Path) -> None:
    runner = _write_shape_runner(tmp_path)

    result = run_resident_linear_calibration(
        runner_path=runner,
        batch_token_values=(2,),
        matrix_dim_values=(2,),
        max_calibration_case_mib=1,
        max_runner_scratch_mib=8,
        work_dir=tmp_path / "calibration",
    )

    assert result.kept_work_dir is True
    assert result.batch_token_values == (2,)
    assert result.matrix_dim_values == (2,)
    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.batch_tokens == 2
    assert case.matrix_dim == 2
    assert case.in_dim == 2
    assert case.out_dim == 2
    assert case.min_matrix_dim == 2
    assert case.matrix_bytes == 16
    assert case.input_bytes == 16
    assert case.output_bytes == 16
    assert case.estimated_flops == 16
    assert case.custom_elapsed_seconds >= 0.0
    assert case.mpsgraph_elapsed_seconds >= 0.0
    assert case.mps_matrix_elapsed_seconds >= 0.0
    assert case.backend_elapsed_seconds is not None
    assert set(case.backend_elapsed_seconds) == {
        "custom-metal",
        "mpsgraph-f32",
        "mps-matrix-f32",
    }
    assert result.work_dir_budget is not None
    assert result.work_dir_budget.case_count == 1
    assert result.work_dir_budget.backend_count == 3
    assert result.work_dir_budget.calibrated_backends == (
        "custom-metal",
        "mpsgraph-f32",
        "mps-matrix-f32",
    )
    assert result.work_dir_budget.backend_output_file_count == 3
    assert result.work_dir_budget.matrix_bytes == 16
    assert result.work_dir_budget.input_bytes == 16
    assert result.work_dir_budget.single_backend_output_bytes == 16
    assert result.work_dir_budget.total_backend_output_bytes == 48
    assert result.work_dir_budget.single_case_bytes == 48
    assert result.work_dir_budget.estimated_work_dir_bytes == 80
    assert result.work_dir_budget.max_calibration_work_dir_mib == 8192
    assert result.work_dir_budget.disk_usage_path == tmp_path
    assert result.work_dir_budget.disk_safety_margin_bytes == 512 * 1024 * 1024
    assert result.work_dir_budget.disk_required_bytes == 80 + 512 * 1024 * 1024
    assert result.work_dir_budget.disk_available_bytes is not None
    assert result.backend_comparison is not None
    assert result.backend_comparison["case_count"] == 1
    assert result.backend_comparison["calibrated_backends"] == (
        "custom-metal",
        "mpsgraph-f32",
        "mps-matrix-f32",
    )
    assert set(result.backend_comparison["backend_total_elapsed_seconds"]) == {
        "custom-metal",
        "mpsgraph-f32",
        "mps-matrix-f32",
    }
    assert result.backend_comparison["winner_counts"][case.winner] == 1


def test_run_resident_linear_calibration_accepts_rectangular_shapes(
    tmp_path: Path,
) -> None:
    runner = _write_shape_runner(tmp_path)

    result = run_resident_linear_calibration(
        runner_path=runner,
        batch_token_values=(2,),
        matrix_shapes=((3, 2),),
        max_calibration_case_mib=1,
        max_runner_scratch_mib=8,
        work_dir=tmp_path / "calibration",
    )

    assert result.matrix_dim_values == (2,)
    assert result.matrix_shapes == ((3, 2),)
    case = result.cases[0]
    assert case.matrix_dim == 2
    assert case.in_dim == 3
    assert case.out_dim == 2
    assert case.min_matrix_dim == 2
    assert case.matrix_bytes == 24
    assert case.input_bytes == 24
    assert case.output_bytes == 16
    assert case.estimated_flops == 24


def test_run_resident_linear_calibration_accepts_bf16_matrix_dtype(
    tmp_path: Path,
) -> None:
    runner = _write_shape_runner(tmp_path)

    result = run_resident_linear_calibration(
        runner_path=runner,
        batch_token_values=(2,),
        matrix_shapes=((3, 2),),
        matrix_dtype="BF16",
        max_calibration_case_mib=1,
        max_runner_scratch_mib=8,
        work_dir=tmp_path / "calibration",
    )

    assert result.matrix_dtype == "BF16"
    assert result.work_dir_budget is not None
    assert result.work_dir_budget.matrix_bytes == 3 * 2 * 2
    case = result.cases[0]
    assert case.matrix_dtype == "BF16"
    assert case.matrix_bytes == 3 * 2 * 2
    assert case.input_bytes == 2 * 3 * 4
    assert case.output_bytes == 2 * 2 * 4


def test_run_resident_linear_calibration_recommends_safe_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    seen_backends: list[str] = []

    def fake_run_resident_batch_linear(**kwargs):
        layout = json.loads(Path(kwargs["resident_layout_path"]).read_text())
        tensor = layout["tensors"][0]
        dim = int(tensor["shape"][0])
        batch = int(kwargs["batch_tokens"])
        backend = str(kwargs["prefill_linear_backend"])
        seen_backends.append(backend)
        output = Path(kwargs["output_f32_path"])
        output.write_bytes(b"\0" * (batch * dim * 4))
        elapsed = 0.010
        if backend == "mpsgraph-f32":
            elapsed = 0.005 if batch >= 64 and dim >= 32 else 0.020
        return ResidentBatchLinearResult(
            runner_path=Path(kwargs["runner_path"]),
            resident_layout_path=Path(kwargs["resident_layout_path"]),
            input_path=Path(kwargs["input_f32_path"]),
            output_path=output,
            layer=1,
            tensor=str(tensor["name"]),
            tensor_suffix=str(kwargs["tensor_suffix"]),
            dtype="F32",
            backend=backend,
            batch_tokens=batch,
            in_dim=dim,
            out_dim=dim,
            matrix_bytes=int(tensor["size"]),
            matrix_scratch_bytes=2 * 1024 * 1024,
            matrix_f32_bytes=(
                int(tensor["size"]) if backend != "custom-metal" else 0
            ),
            matrix_raw_conversion_bytes=0,
            input_bytes=batch * dim * 4,
            output_bytes=batch * dim * 4,
            estimated_peak_bytes=2 * 1024 * 1024 + 2 * batch * dim * 4,
            elapsed_seconds=elapsed,
            command=(),
        )

    monkeypatch.setattr(
        "largerlm.prefill_execute.run_resident_batch_linear",
        fake_run_resident_batch_linear,
    )

    result = run_resident_linear_calibration(
        runner_path=runner,
        batch_token_values=(32, 64, 128),
        matrix_dim_values=(16, 32),
        work_dir=tmp_path / "calibration",
    )

    assert result.recommended_prefill_mpsgraph_min_batch_tokens == 64
    assert result.recommended_prefill_mpsgraph_min_matrix_dim == 32
    assert set(seen_backends) == {
        "custom-metal",
        "mpsgraph-f32",
        "mps-matrix-f32",
    }
    assert all(case.mps_matrix_elapsed_seconds == 0.010 for case in result.cases)
    assert result.suggested_prefill_runtime_policy_flags == {
        "source": "prefill_linear_calibration",
        "prefill_mpsgraph_min_batch_tokens": 64,
        "prefill_mpsgraph_min_matrix_dim": 32,
        "argv": (
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
        ),
    }
    assert result.suggested_launch_profile is not None
    assert result.suggested_launch_profile["argv_safe_to_replay"] is True
    assert result.suggested_launch_profile["sections"][
        "prefill_runtime_policy_flags"
    ] == result.suggested_prefill_runtime_policy_flags
    assert result.backend_comparison is not None
    assert result.backend_comparison["recommended_explicit_backend"] == "custom-metal"


def test_run_resident_linear_calibration_recommends_explicit_custom_when_fallbacks_lose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run_resident_batch_linear(**kwargs):
        layout = json.loads(Path(kwargs["resident_layout_path"]).read_text())
        tensor = layout["tensors"][0]
        dim = int(tensor["shape"][0])
        batch = int(kwargs["batch_tokens"])
        backend = str(kwargs["prefill_linear_backend"])
        output = Path(kwargs["output_f32_path"])
        output.write_bytes(b"\0" * (batch * dim * 4))
        elapsed = {
            "custom-metal": 0.010,
            "mpsgraph-f32": 0.025,
            "mps-matrix-f32": 0.020,
        }[backend]
        return ResidentBatchLinearResult(
            runner_path=Path(kwargs["runner_path"]),
            resident_layout_path=Path(kwargs["resident_layout_path"]),
            input_path=Path(kwargs["input_f32_path"]),
            output_path=output,
            layer=1,
            tensor=str(tensor["name"]),
            tensor_suffix=str(kwargs["tensor_suffix"]),
            dtype="F32",
            backend=backend,
            batch_tokens=batch,
            in_dim=dim,
            out_dim=dim,
            matrix_bytes=int(tensor["size"]),
            matrix_scratch_bytes=2 * 1024 * 1024,
            matrix_f32_bytes=(
                int(tensor["size"]) if backend != "custom-metal" else 0
            ),
            matrix_raw_conversion_bytes=0,
            input_bytes=batch * dim * 4,
            output_bytes=batch * dim * 4,
            estimated_peak_bytes=2 * 1024 * 1024 + 2 * batch * dim * 4,
            elapsed_seconds=elapsed,
            command=(),
        )

    monkeypatch.setattr(
        "largerlm.prefill_execute.run_resident_batch_linear",
        fake_run_resident_batch_linear,
    )

    result = run_resident_linear_calibration(
        runner_path=runner,
        batch_token_values=(128,),
        matrix_dim_values=(64,),
        min_mpsgraph_speedup=1.1,
        work_dir=tmp_path / "calibration",
    )

    assert result.recommended_prefill_mpsgraph_min_batch_tokens is None
    assert result.recommended_prefill_mpsgraph_min_matrix_dim is None
    assert result.backend_comparison is not None
    assert result.backend_comparison["recommended_explicit_backend"] == "custom-metal"
    assert result.backend_comparison["recommended_explicit_backend_policy_flags"] == {
        "source": "prefill_linear_calibration",
        "prefill_linear_backend": "custom-metal",
        "argv": ("--prefill-linear-backend", "custom-metal"),
    }
    assert result.suggested_prefill_runtime_policy_flags == {
        "source": "prefill_linear_calibration",
        "prefill_linear_backend": "custom-metal",
        "argv": ("--prefill-linear-backend", "custom-metal"),
    }
    assert result.suggested_launch_profile is not None
    assert result.suggested_launch_profile["argv"] == (
        "--prefill-linear-backend",
        "custom-metal",
    )


def test_run_resident_linear_calibration_recommends_explicit_mps_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_run_resident_batch_linear(**kwargs):
        layout = json.loads(Path(kwargs["resident_layout_path"]).read_text())
        tensor = layout["tensors"][0]
        dim = int(tensor["shape"][0])
        batch = int(kwargs["batch_tokens"])
        backend = str(kwargs["prefill_linear_backend"])
        output = Path(kwargs["output_f32_path"])
        output.write_bytes(b"\0" * (batch * dim * 4))
        elapsed = {
            "custom-metal": 0.010,
            "mpsgraph-f32": 0.012,
            "mps-matrix-f32": 0.004,
        }[backend]
        return ResidentBatchLinearResult(
            runner_path=Path(kwargs["runner_path"]),
            resident_layout_path=Path(kwargs["resident_layout_path"]),
            input_path=Path(kwargs["input_f32_path"]),
            output_path=output,
            layer=1,
            tensor=str(tensor["name"]),
            tensor_suffix=str(kwargs["tensor_suffix"]),
            dtype="F32",
            backend=backend,
            batch_tokens=batch,
            in_dim=dim,
            out_dim=dim,
            matrix_bytes=int(tensor["size"]),
            matrix_scratch_bytes=2 * 1024 * 1024,
            matrix_f32_bytes=(
                int(tensor["size"]) if backend != "custom-metal" else 0
            ),
            matrix_raw_conversion_bytes=0,
            input_bytes=batch * dim * 4,
            output_bytes=batch * dim * 4,
            estimated_peak_bytes=2 * 1024 * 1024 + 2 * batch * dim * 4,
            elapsed_seconds=elapsed,
            command=(),
        )

    monkeypatch.setattr(
        "largerlm.prefill_execute.run_resident_batch_linear",
        fake_run_resident_batch_linear,
    )

    result = run_resident_linear_calibration(
        runner_path=runner,
        batch_token_values=(64,),
        matrix_dim_values=(32,),
        min_mpsgraph_speedup=1.25,
        work_dir=tmp_path / "calibration",
    )

    assert result.backend_comparison is not None
    assert result.backend_comparison["recommended_explicit_backend"] == (
        "mps-matrix-f32"
    )
    assert result.backend_comparison["recommended_explicit_backend_policy_flags"] == {
        "source": "prefill_linear_calibration",
        "prefill_linear_backend": "mps-matrix-f32",
        "argv": ("--prefill-linear-backend", "mps-matrix-f32"),
    }
    assert result.backend_comparison["backend_speedup_vs_custom"][
        "mps-matrix-f32"
    ] == pytest.approx(2.5)
    assert result.backend_comparison["backend_all_cases_meet_min_speedup"][
        "mps-matrix-f32"
    ] is True
    assert result.cases[0].winner == "mps-matrix-f32"
    assert result.suggested_prefill_runtime_policy_flags == {
        "source": "prefill_linear_calibration",
        "prefill_linear_backend": "mps-matrix-f32",
        "argv": ("--prefill-linear-backend", "mps-matrix-f32"),
    }
    assert result.suggested_launch_profile is not None
    assert result.suggested_launch_profile["sections"][
        "prefill_runtime_policy_flags"
    ] == result.suggested_prefill_runtime_policy_flags
    assert result.suggested_launch_profile["argv"] == (
        "--prefill-linear-backend",
        "mps-matrix-f32",
    )


def test_run_resident_linear_calibration_rejects_oversized_case(
    tmp_path: Path,
) -> None:
    runner = _write_shape_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match="calibration case"):
        run_resident_linear_calibration(
            runner_path=runner,
            batch_token_values=(1024,),
            matrix_dim_values=(1024,),
            max_calibration_case_mib=1,
            work_dir=tmp_path / "calibration",
        )


def test_run_resident_linear_calibration_rejects_work_dir_budget_before_work_dir(
    tmp_path: Path,
) -> None:
    runner = _write_shape_runner(tmp_path)
    work_dir = tmp_path / "calibration"

    with pytest.raises(PrefillExecuteError, match="calibration work dir"):
        run_resident_linear_calibration(
            runner_path=runner,
            batch_token_values=(128,),
            matrix_shapes=(
                (256, 128),
                (128, 256),
                (64, 128),
                (128, 32),
                (32, 128),
                (32, 64),
            ),
            max_calibration_case_mib=1,
            max_calibration_work_dir_mib=1,
            work_dir=work_dir,
        )

    assert not work_dir.exists()


def test_run_resident_linear_calibration_rejects_low_disk_before_work_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _write_shape_runner(tmp_path)
    work_dir = tmp_path / "calibration"

    class FakeDiskUsage:
        free = 1024

    monkeypatch.setattr(
        "largerlm.prefill_execute.shutil.disk_usage",
        lambda path: FakeDiskUsage(),
    )

    with pytest.raises(PrefillExecuteError, match="only 1024 bytes available"):
        run_resident_linear_calibration(
            runner_path=runner,
            batch_token_values=(2,),
            matrix_dim_values=(2,),
            max_calibration_case_mib=1,
            calibration_work_dir_free_margin_mib=1,
            work_dir=work_dir,
        )

    assert not work_dir.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"prefill_mpsgraph_min_batch_tokens": 0},
            "prefill_mpsgraph_min_batch_tokens",
        ),
        (
            {"prefill_mpsgraph_min_matrix_dim": False},
            "prefill_mpsgraph_min_matrix_dim",
        ),
    ),
)
def test_run_resident_batch_linear_rejects_invalid_auto_thresholds(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match=message):
        run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            prefill_linear_backend="auto",
            echo_runner_output=False,
            **kwargs,
        )


def test_run_resident_batch_linear_auto_selects_mpsgraph_for_large_f32(
    tmp_path: Path,
) -> None:
    layout, input_path = _write_large_linear_fixture(tmp_path)
    runner = _write_shape_runner(tmp_path)
    output = tmp_path / "large_out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=128,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
        echo_runner_output=False,
    )

    assert result.backend == "mpsgraph-f32"
    assert result.matrix_scratch_bytes == 2 * 1024 * 1024
    assert result.matrix_f32_bytes == 32 * 32 * 4
    assert result.matrix_raw_conversion_bytes == 0
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--prefill-linear-backend") + 1] == "mpsgraph-f32"


def test_run_resident_batch_linear_auto_selects_mpsgraph_for_large_bf16(
    tmp_path: Path,
) -> None:
    layout, input_path = _write_large_linear_fixture(tmp_path, dtype="BF16")
    runner = _write_shape_runner(tmp_path)
    output = tmp_path / "large_bf16_out.f32"

    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        tensor_suffix=".self_attn.q_a_proj.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=128,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
        echo_runner_output=False,
    )

    assert result.backend == "mpsgraph-f32"
    assert result.matrix_bytes == 32 * 32 * 2
    assert result.matrix_scratch_bytes == 2 * 1024 * 1024 + result.matrix_bytes
    assert result.matrix_f32_bytes == 32 * 32 * 4
    assert result.matrix_raw_conversion_bytes == result.matrix_bytes
    assert result.estimated_peak_bytes == (
        2 * 1024 * 1024 + result.matrix_bytes + result.input_bytes + result.output_bytes
    )
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--prefill-linear-backend") + 1] == "mpsgraph-f32"


@pytest.mark.parametrize(
    ("dtype", "expected"),
    (
        ("BF16", "mpp-f32"),
        ("F16", "mpp-f32"),
        ("F32", "mpp-f32"),
        ("mlx-mxfp4", "custom-metal"),
        ("affine-int4", "custom-metal"),
    ),
)
def test_auto_mpp_selects_mpp_only_for_supported_resident_matrices(
    dtype: str,
    expected: str,
) -> None:
    assert (
        _resolve_prefill_linear_backend(
            "auto-mpp",
            dtype,
            batch_tokens=128,
            in_dim=32,
            out_dim=32,
            mpsgraph_min_batch_tokens=128,
            mpsgraph_min_matrix_dim=32,
        )
        == expected
    )


def test_run_resident_batch_linear_rejects_wrong_input_size(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path, (1, 2, 3))
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match="input bytes"):
        run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            echo_runner_output=False,
        )


def test_run_resident_batch_linear_rejects_boolean_matrix_shape(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["shape"][0] = True

    _mutate_layout(layout, mutate)

    with pytest.raises(PrefillExecuteError, match="must have shape \\[rows, cols\\]"):
        run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            echo_runner_output=False,
        )


def test_run_resident_batch_linear_rejects_boolean_matrix_size(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["size"] = False

    _mutate_layout(layout, mutate)

    with pytest.raises(PrefillExecuteError, match="size must be an integer"):
        run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            echo_runner_output=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layer": True}, "layer must be an integer"),
        ({"batch_tokens": False}, "batch_tokens must be an integer"),
        ({"max_resident_matrix_mib": True}, "max_resident_matrix_mib"),
        ({"max_runner_scratch_mib": False}, "max_runner_scratch_mib"),
    ],
)
def test_run_resident_batch_linear_rejects_boolean_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    args: dict[str, object] = {
        "runner_path": tmp_path / "missing-runner",
        "resident_layout_path": tmp_path / "missing-layout.json",
        "layer": 1,
        "tensor_suffix": ".self_attn.q_a_proj.weight",
        "input_f32_path": tmp_path / "input.f32",
        "output_f32_path": tmp_path / "out.f32",
        "batch_tokens": 2,
    }
    args.update(kwargs)

    with pytest.raises(PrefillExecuteError, match=message):
        run_resident_batch_linear(**args)


def test_run_resident_batch_linear_rejects_truncated_resident_file(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    (layout.parent / "resident.bin").write_bytes(b"\0" * 23)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match="resident weight file"):
        run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            echo_runner_output=False,
        )


def test_run_resident_batch_linear_rejects_tiny_scratch_limit(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match="scratch limit"):
        run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            tensor_suffix=".self_attn.q_a_proj.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            max_runner_scratch_mib=1,
            echo_runner_output=False,
        )


def test_run_resident_batch_rmsnorm_invokes_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "norm.f32"

    result = run_resident_batch_rmsnorm(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        norm_suffix=".input_layernorm.weight",
        input_f32_path=input_path,
        output_f32_path=output,
        batch_tokens=2,
        rms_norm_eps=0.0,
        echo_runner_output=False,
    )

    assert result.tensor == "model.layers.1.input_layernorm.weight"
    assert result.hidden_dim == 3
    assert result.vector_bytes == 12
    assert result.input_bytes == 24
    assert result.output_bytes == 24
    assert result.estimated_peak_bytes == 72
    assert len(struct.unpack("<6f", output.read_bytes())) == 6
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert "--run-rmsnorm-batch" in argv
    assert argv[argv.index("--rms-norm-eps") + 1] == "0"


def test_run_resident_batch_rmsnorm_can_use_server_session(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output = tmp_path / "norm_server.f32"

    with ResidentBatchRMSNormServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_resident_batch_rmsnorm(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            norm_suffix=".input_layernorm.weight",
            input_f32_path=input_path,
            output_f32_path=output,
            batch_tokens=2,
            rms_norm_eps=0.0,
            echo_runner_output=False,
            rmsnorm_server_session=session,
        )

    server_flag = "--run-rmsnorm-batch-server-jsonl"
    assert result.command == (str(runner), server_flag)
    assert len(struct.unpack("<6f", output.read_bytes())) == 6
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert server_flag in argv
    assert argv[argv.index("--rms-norm-eps") + 1] == "0.0"


def test_run_resident_batch_rmsnorm_rejects_truncated_resident_file(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    resident_bin = layout.parent / "resident.bin"
    resident_bin.write_bytes(resident_bin.read_bytes()[:83])
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match="resident weight file"):
        run_resident_batch_rmsnorm(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            norm_suffix=".input_layernorm.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "norm.f32",
            batch_tokens=2,
            rms_norm_eps=0.0,
            echo_runner_output=False,
        )


def test_run_resident_batch_rmsnorm_rejects_negative_eps(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PrefillExecuteError, match="rms_norm_eps"):
        run_resident_batch_rmsnorm(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            norm_suffix=".input_layernorm.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "norm.f32",
            batch_tokens=2,
            rms_norm_eps=-1.0,
            echo_runner_output=False,
        )


def test_run_resident_batch_rmsnorm_rejects_boolean_vector_shape(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[2]["shape"][0] = True

    _mutate_layout(layout, mutate)

    with pytest.raises(PrefillExecuteError, match="must have shape \\[dim\\]"):
        run_resident_batch_rmsnorm(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            norm_suffix=".input_layernorm.weight",
            input_f32_path=input_path,
            output_f32_path=tmp_path / "norm.f32",
            batch_tokens=2,
            echo_runner_output=False,
        )


def test_run_prefill_attention_prefix_batch_invokes_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "prefix"

    result = run_prefill_attention_prefix_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        batch_tokens=2,
        rms_norm_eps=0.0,
        prefill_linear_backend="auto",
        echo_runner_output=False,
    )

    assert result.hidden_dim == 3
    assert result.q_a_dim == 2
    assert result.kv_a_dim == 4
    assert result.input_bytes == 24
    assert result.norm_output_bytes == 24
    assert result.q_a_output_bytes == 16
    assert result.kv_a_output_bytes == 32
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 24 + 32
    assert len(struct.unpack("<6f", (output_dir / "input_layernorm.f32").read_bytes())) == 6
    assert struct.unpack("<4f", (output_dir / "q_a_proj.f32").read_bytes()) == (
        6.0,
        5.0,
        15.0,
        14.0,
    )
    assert len(struct.unpack("<8f", (output_dir / "kv_a_proj_with_mqa.f32").read_bytes())) == 8
    kv_argv = json.loads(
        (output_dir / "kv_a_proj_with_mqa.f32.argv.json").read_text()
    )
    assert kv_argv[kv_argv.index("--tensor-suffix") + 1] == (
        ".self_attn.kv_a_proj_with_mqa.weight"
    )
    assert "--prefill-linear-backend" not in kv_argv
    assert result.q_a_proj.backend == "custom-metal"
    assert result.kv_a_proj_with_mqa.backend == "custom-metal"


def test_run_prefill_attention_projection_batch_invokes_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "projections"

    result = run_prefill_attention_projection_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        batch_tokens=2,
        rms_norm_eps=0.0,
        echo_runner_output=False,
    )

    assert result.hidden_dim == 3
    assert result.q_a_dim == 2
    assert result.q_b_dim == 4
    assert result.kv_a_dim == 4
    assert result.kv_lora_dim == 2
    assert result.kv_rope_dim == 2
    assert result.kv_b_dim == 4
    assert result.attention_value_source == "kv_b_proj"
    assert result.q_b_output_bytes == 32
    assert result.kv_a_lora_bytes == 16
    assert result.kv_a_rope_bytes == 16
    assert result.kv_b_output_bytes == 32
    assert result.split_peak_bytes == 16
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 24 + 32
    assert struct.unpack("<4f", (output_dir / "kv_a_lora.f32").read_bytes()) == (
        1.0,
        2.0,
        4.0,
        5.0,
    )
    assert struct.unpack("<4f", (output_dir / "kv_a_rope.f32").read_bytes()) == (
        3.0,
        6.0,
        6.0,
        15.0,
    )
    assert len(struct.unpack("<8f", (output_dir / "q_b_proj.f32").read_bytes())) == 8
    assert len(struct.unpack("<8f", (output_dir / "kv_b_proj.f32").read_bytes())) == 8
    q_b_argv = json.loads((output_dir / "q_b_proj.f32.argv.json").read_text())
    assert q_b_argv[q_b_argv.index("--tensor-suffix") + 1] == ".self_attn.q_b_proj.weight"
    kv_norm_argv = json.loads((output_dir / "kv_a_layernorm.f32.argv.json").read_text())
    assert kv_norm_argv[kv_norm_argv.index("--input-f32") + 1].endswith("kv_a_lora.f32")


def test_run_prefill_attention_projection_batch_uses_fused_single_token_runner(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = tmp_path / "single_input.f32"
    input_path.write_bytes(struct.pack("<3f", 1.0, 2.0, 3.0))
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output_dir = tmp_path / "projections_fused"

    result = run_prefill_attention_projection_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        batch_tokens=1,
        rms_norm_eps=0.0,
        echo_runner_output=False,
    )

    assert result.batch_tokens == 1
    assert result.prefix.input_layernorm.output_path == output_dir / "attn_input_norm.f32"
    assert result.prefix.q_a_proj.output_path == output_dir / "attn_q_a.f32"
    assert result.prefix.kv_a_proj_with_mqa.output_path == output_dir / "attn_kv_a.f32"
    assert result.q_a_layernorm.output_path == output_dir / "attn_q_a_norm.f32"
    assert result.q_b_proj.output_path == output_dir / "attn_q_b.f32"
    assert result.kv_a_layernorm.output_path == output_dir / "attn_kv_a_norm.f32"
    assert result.kv_b_proj is not None
    assert result.kv_b_proj.output_path == output_dir / "attn_kv_b.f32"
    assert result.q_b_proj.backend == "fused-metal"
    assert "--run-attn-projections" in result.q_b_proj.command
    assert struct.unpack("<2f", (output_dir / "kv_a_lora.f32").read_bytes()) == (
        1.0,
        2.0,
    )
    assert struct.unpack("<2f", (output_dir / "kv_a_rope.f32").read_bytes()) == (
        3.0,
        6.0,
    )


def test_run_prefill_attention_projection_batch_uses_fused_batch_runner(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output_dir = tmp_path / "projections_fused_batch"

    result = run_prefill_attention_projection_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        batch_tokens=2,
        rms_norm_eps=0.0,
        echo_runner_output=False,
    )

    assert result.batch_tokens == 2
    assert result.q_b_output_bytes == 32
    assert result.kv_a_lora_bytes == 16
    assert result.kv_a_rope_bytes == 16
    assert result.kv_b_output_bytes == 32
    assert result.prefix.input_layernorm.batch_tokens == 2
    assert result.q_a_layernorm.batch_tokens == 2
    assert result.q_b_proj.backend == "fused-metal"
    assert "--run-attn-projections" in result.q_b_proj.command
    assert result.q_b_proj.command[result.q_b_proj.command.index("--batch-tokens") + 1] == "2"
    assert struct.unpack("<4f", (output_dir / "kv_a_lora.f32").read_bytes()) == (
        1.0,
        2.0,
        1.0,
        2.0,
    )
    assert struct.unpack("<4f", (output_dir / "kv_a_rope.f32").read_bytes()) == (
        3.0,
        6.0,
        3.0,
        6.0,
    )
    assert len(struct.unpack("<8f", (output_dir / "attn_q_b.f32").read_bytes())) == 8
    assert len(struct.unpack("<8f", (output_dir / "attn_kv_b.f32").read_bytes())) == 8


def test_run_prefill_attention_projection_batch_can_use_server_session(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output_dir = tmp_path / "projections_server"

    with AttentionProjectionsServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_prefill_attention_projection_batch(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            input_f32_path=input_path,
            output_dir=output_dir,
            batch_tokens=2,
            rms_norm_eps=0.0,
            echo_runner_output=False,
            attention_projection_server_session=session,
        )

    assert result.batch_tokens == 2
    assert result.q_b_proj.backend == "fused-metal"
    assert "--run-attn-projections-server-jsonl" in result.q_b_proj.command
    assert "--run-attn-projections-server-jsonl" in result.prefix.q_a_proj.command
    assert struct.unpack("<4f", (output_dir / "kv_a_lora.f32").read_bytes()) == (
        1.0,
        2.0,
        1.0,
        2.0,
    )
    assert len(struct.unpack("<8f", (output_dir / "attn_q_b.f32").read_bytes())) == 8


def test_run_prefill_attention_projection_batch_can_disable_fused_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_PROJECTIONS", "1")
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output_dir = tmp_path / "projections_default_batch"

    result = run_prefill_attention_projection_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        batch_tokens=2,
        rms_norm_eps=0.0,
        echo_runner_output=False,
    )

    assert result.batch_tokens == 2
    assert result.q_b_proj.output_path == output_dir / "q_b_proj.f32"
    assert "--run-attn-projections" not in result.q_b_proj.command
    assert "--run-resident-linear-batch" in result.q_b_proj.command
    q_b_argv = json.loads((output_dir / "q_b_proj.f32.argv.json").read_text())
    assert "--run-resident-linear-batch" in q_b_argv
    assert q_b_argv[q_b_argv.index("--batch-tokens") + 1] == "2"


def test_run_prefill_attention_projection_batch_accepts_absorbed_aliases(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    _replace_kv_b_with_absorbed_aliases(layout)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "projections_absorbed"

    result = run_prefill_attention_projection_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        batch_tokens=2,
        rms_norm_eps=0.0,
        echo_runner_output=False,
    )

    assert result.attention_value_source == "absorbed-alias"
    assert result.kv_b_dim == 4
    assert result.kv_b_output_bytes == 0
    assert result.kv_b_proj is None
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 24 + 32
    assert len(struct.unpack("<8f", (output_dir / "q_b_proj.f32").read_bytes())) == 8
    assert len(struct.unpack("<4f", (output_dir / "kv_a_layernorm.f32").read_bytes())) == 4
    assert not (output_dir / "kv_b_proj.f32").exists()


def test_run_prefill_rope_batch_splits_q_b_and_invokes_runner(tmp_path: Path) -> None:
    runner = _write_fake_runner(tmp_path)
    q_b = tmp_path / "q_b.f32"
    q_b.write_bytes(struct.pack("<12f", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    k_rope = tmp_path / "k_rope.f32"
    k_rope.write_bytes(struct.pack("<4f", 100, 101, 102, 103))
    output_dir = tmp_path / "rope"

    result = run_prefill_rope_batch(
        runner_path=runner,
        q_b_f32_path=q_b,
        k_rope_f32_path=k_rope,
        output_dir=output_dir,
        batch_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        start_position=4,
        rope_interleave=True,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.q_b_input_bytes == 48
    assert result.k_rope_input_bytes == 16
    assert result.q_nope_bytes == 16
    assert result.q_rope_bytes == 32
    assert result.k_rope_bytes == 16
    assert result.estimated_peak_bytes == 96
    assert struct.unpack("<4f", (output_dir / "q_nope.f32").read_bytes()) == (
        1.0,
        4.0,
        7.0,
        10.0,
    )
    assert struct.unpack("<8f", (output_dir / "q_rope.f32").read_bytes()) == (
        2.0,
        3.0,
        5.0,
        6.0,
        8.0,
        9.0,
        11.0,
        12.0,
    )
    assert len(struct.unpack("<8f", (output_dir / "q_rope_rotated.f32").read_bytes())) == 8
    assert len(struct.unpack("<4f", (output_dir / "k_rope_rotated.f32").read_bytes())) == 4
    argv = json.loads((output_dir / "q_rope_rotated.f32.argv.json").read_text())
    assert "--run-rope-batch" in argv
    assert "--rope-interleave" in argv
    assert argv[argv.index("--start-position") + 1] == "4"


def test_run_prefill_rope_batch_largerlm_runner_uses_fused_split(
    tmp_path: Path,
) -> None:
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    q_b = tmp_path / "q_b.f32"
    q_b.write_bytes(struct.pack("<12f", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    k_rope = tmp_path / "k_rope.f32"
    k_rope.write_bytes(struct.pack("<4f", 100, 101, 102, 103))
    output_dir = tmp_path / "rope_fused"

    result = run_prefill_rope_batch(
        runner_path=runner,
        q_b_f32_path=q_b,
        k_rope_f32_path=k_rope,
        output_dir=output_dir,
        batch_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        start_position=4,
        rope_interleave=True,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.estimated_peak_bytes == 160
    assert struct.unpack("<4f", (output_dir / "q_nope.f32").read_bytes()) == (
        1.0,
        4.0,
        7.0,
        10.0,
    )
    assert struct.unpack("<8f", (output_dir / "q_rope.f32").read_bytes()) == (
        2.0,
        3.0,
        5.0,
        6.0,
        8.0,
        9.0,
        11.0,
        12.0,
    )
    argv = json.loads((output_dir / "q_rope_rotated.f32.argv.json").read_text())
    assert "--run-rope-split-batch" in argv
    assert "--run-rope-batch" not in argv
    assert argv[argv.index("--q-b-f32") + 1] == str(q_b)
    assert argv[argv.index("--output-q-nope-f32") + 1] == str(output_dir / "q_nope.f32")


def test_run_prefill_rope_batch_can_use_server_session(tmp_path: Path) -> None:
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    q_b = tmp_path / "q_b.f32"
    q_b.write_bytes(struct.pack("<12f", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    k_rope = tmp_path / "k_rope.f32"
    k_rope.write_bytes(struct.pack("<4f", 100, 101, 102, 103))
    output_dir = tmp_path / "rope_server"

    with RopeSplitBatchServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_prefill_rope_batch(
            runner_path=runner,
            q_b_f32_path=q_b,
            k_rope_f32_path=k_rope,
            output_dir=output_dir,
            batch_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            start_position=4,
            rope_interleave=True,
            max_runner_scratch_mib=1,
            echo_runner_output=False,
            rope_split_server_session=session,
        )

    assert result.command == (str(runner), "--run-rope-split-batch-server-jsonl")
    assert result.estimated_peak_bytes == 160
    assert struct.unpack("<4f", (output_dir / "q_nope.f32").read_bytes()) == (
        1.0,
        4.0,
        7.0,
        10.0,
    )
    assert struct.unpack("<8f", (output_dir / "q_rope.f32").read_bytes()) == (
        2.0,
        3.0,
        5.0,
        6.0,
        8.0,
        9.0,
        11.0,
        12.0,
    )
    argv = json.loads((output_dir / "q_rope_rotated.f32.argv.json").read_text())
    assert "--run-rope-split-batch-server-jsonl" in argv
    assert "--rope-interleave" in argv
    assert argv[argv.index("--batch-tokens") + 1] == "2"


def test_run_prefill_rope_batch_singleton_largerlm_runner_uses_python_fast_path(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "largerlm-runner"
    runner.write_text(
        "#!/usr/bin/env python3\nraise SystemExit(99)\n",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    q_b = tmp_path / "q_b.f32"
    q_b.write_bytes(struct.pack("<6f", 10, 1, 2, 20, 3, 4))
    k_rope = tmp_path / "k_rope.f32"
    k_rope.write_bytes(struct.pack("<2f", 5, 6))
    output_dir = tmp_path / "rope_python"

    result = run_prefill_rope_batch(
        runner_path=runner,
        q_b_f32_path=q_b,
        k_rope_f32_path=k_rope,
        output_dir=output_dir,
        batch_tokens=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        start_position=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    c = math.cos(1.0)
    s = math.sin(1.0)
    assert result.command[0] == "python-rope-singleton"
    assert struct.unpack("<2f", (output_dir / "q_nope.f32").read_bytes()) == (
        10.0,
        20.0,
    )
    assert struct.unpack("<4f", (output_dir / "q_rope.f32").read_bytes()) == (
        1.0,
        2.0,
        3.0,
        4.0,
    )
    assert struct.unpack("<4f", (output_dir / "q_rope_rotated.f32").read_bytes()) == pytest.approx(
        (1 * c - 2 * s, 2 * c + 1 * s, 3 * c - 4 * s, 4 * c + 3 * s),
        abs=1e-6,
    )
    assert struct.unpack("<2f", (output_dir / "k_rope_rotated.f32").read_bytes()) == pytest.approx(
        (5 * c - 6 * s, 6 * c + 5 * s),
        abs=1e-6,
    )
    assert not (output_dir / "q_rope_rotated.f32.argv.json").exists()


def test_run_prefill_mla_attention_batch_invokes_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn.f32"

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.context_length == 3
    assert result.start_position == 1
    assert result.batch_tokens == 2
    assert result.kv_lora_dim == 2
    assert result.q_nope_bytes == 16
    assert result.q_rope_bytes == 16
    assert result.attention_value_source == "kv_b_proj"
    assert result.output_bytes == 16
    assert result.cache_read_bytes == 24
    assert result.cache_f32_bytes == 48
    assert result.kv_b_matrix_bytes == 32
    assert result.kv_b_f32_bytes == 32
    assert result.mla_key_cache is False
    assert result.mla_key_cache_bytes == 0
    assert result.mla_value_cache is True
    assert result.mla_value_cache_bytes == 24
    assert result.estimated_peak_bytes == 208
    assert struct.unpack("<4f", output.read_bytes()) == (0.75, 0.75, 0.75, 0.75)
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert "--run-mla-attention-batch" in argv
    assert argv[argv.index("--context-length") + 1] == "3"
    assert argv[argv.index("--start-position") + 1] == "1"
    assert argv[argv.index("--kv-lora-dim") + 1] == "2"
    env = json.loads(output.with_suffix(output.suffix + ".env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] is None
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_run_prefill_mla_attention_batch_can_use_server_session(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn_server.f32"

    with MLAAttentionBatchServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_prefill_mla_attention_batch(
            runner_path=runner,
            resident_layout_path=layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=1,
            q_nope_f32_path=q_nope,
            q_rope_f32_path=q_rope,
            output_f32_path=output,
            context_length=3,
            start_position=1,
            batch_tokens=2,
            num_heads=1,
            qk_nope_dim=2,
            rope_dim=2,
            v_head_dim=2,
            kv_lora_dim=2,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=1,
            echo_runner_output=False,
            mla_attention_server_session=session,
        )

    assert result.command == (str(runner), "--run-mla-attention-batch-server-jsonl")
    assert struct.unpack("<4f", output.read_bytes()) == (0.75, 0.75, 0.75, 0.75)
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv == ["--run-mla-attention-batch-server-jsonl"]
    env = json.loads(output.with_suffix(output.suffix + ".env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] is None
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_run_prefill_mla_attention_batch_single_token_skips_value_cache_by_default(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<2f", 1, 2))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<2f", 5, 6))
    output = tmp_path / "attn.f32"

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=2,
        batch_tokens=1,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.batch_tokens == 1
    assert result.q_nope_bytes == 8
    assert result.q_rope_bytes == 8
    assert result.output_bytes == 8
    assert result.mla_key_cache is False
    assert result.mla_key_cache_bytes == 0
    assert result.mla_value_cache is False
    assert result.mla_value_cache_bytes == 0
    assert result.estimated_peak_bytes == 160
    assert struct.unpack("<2f", output.read_bytes()) == (0.75, 0.75)
    env = json.loads(output.with_suffix(output.suffix + ".env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] is None
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_run_prefill_mla_attention_batch_single_token_value_cache_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<2f", 1, 2))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<2f", 5, 6))
    output = tmp_path / "attn.f32"
    monkeypatch.setenv("LARGERLM_MLA_VALUE_CACHE_SINGLETON", "1")

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=2,
        batch_tokens=1,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.mla_value_cache is True
    assert result.mla_value_cache_bytes == 24
    assert result.estimated_peak_bytes == 184


def test_run_prefill_mla_attention_batch_env_can_disable_value_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn.f32"
    monkeypatch.setenv("LARGERLM_MLA_DISABLE_VALUE_CACHE", "1")

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.mla_value_cache is False
    assert result.mla_value_cache_bytes == 0
    assert result.estimated_peak_bytes == 184
    env = json.loads(output.with_suffix(output.suffix + ".env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] is None
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] == "1"


def test_run_prefill_mla_attention_batch_can_request_key_cache(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn.f32"

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
        mla_key_cache=True,
    )

    assert result.mla_key_cache is True
    assert result.mla_key_cache_bytes == 24
    assert result.mla_value_cache is True
    assert result.mla_value_cache_bytes == 24
    assert result.estimated_peak_bytes == 232
    env = json.loads(output.with_suffix(output.suffix + ".env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] == "1"
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_run_prefill_mla_attention_batch_rejects_non_bool_key_cache(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))

    with pytest.raises(PrefillExecuteError, match="mla_key_cache must be a boolean"):
        run_prefill_mla_attention_batch(
            runner_path=runner,
            resident_layout_path=layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=1,
            q_nope_f32_path=q_nope,
            q_rope_f32_path=q_rope,
            output_f32_path=tmp_path / "attn.f32",
            context_length=3,
            start_position=1,
            batch_tokens=2,
            num_heads=1,
            qk_nope_dim=2,
            rope_dim=2,
            v_head_dim=2,
            kv_lora_dim=2,
            mla_key_cache=1,  # type: ignore[arg-type]
        )


def test_run_prefill_mla_attention_batch_accepts_absorbed_aliases(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    _replace_kv_b_with_absorbed_aliases(layout)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn_absorbed.f32"

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.kv_lora_dim == 2
    assert result.attention_value_source == "absorbed-alias"
    assert result.kv_b_matrix_bytes == 32
    assert result.kv_b_f32_bytes == 32
    assert result.mla_value_cache is True
    assert result.mla_value_cache_bytes == 24
    assert result.estimated_peak_bytes == 240
    assert struct.unpack("<4f", output.read_bytes()) == (0.75, 0.75, 0.75, 0.75)
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert "--run-mla-attention-batch" in argv
    assert argv[argv.index("--kv-lora-dim") + 1] == "2"


def test_run_prefill_mla_attention_batch_uses_absorbed_alias_cache_dir(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    _replace_kv_b_with_absorbed_aliases(layout)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn_absorbed_cache.f32"
    cache_dir = tmp_path / "mla_kv_b_cache"

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        output_f32_path=output,
        context_length=3,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        mla_kv_b_cache_dir=cache_dir,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.attention_value_source == "absorbed-alias-cache"
    assert result.mla_kv_b_cache_dir == cache_dir
    assert cache_dir.is_dir()
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv[argv.index("--mla-kv-b-cache-dir") + 1] == str(cache_dir)


def test_run_prefill_mla_attention_batch_rejects_truncated_kv_b(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    payload = json.loads(layout.read_text(encoding="utf-8"))
    kv_b = next(
        tensor
        for tensor in payload["tensors"]
        if tensor["name"].endswith(".self_attn.kv_b_proj.weight")
    )
    resident_bin = layout.parent / "resident.bin"
    resident_bin.write_bytes(
        resident_bin.read_bytes()[: kv_b["offset"] + kv_b["size"] - 1]
    )
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))

    with pytest.raises(PrefillExecuteError, match="resident weight file"):
        run_prefill_mla_attention_batch(
            runner_path=runner,
            resident_layout_path=layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=1,
            q_nope_f32_path=q_nope,
            q_rope_f32_path=q_rope,
            output_f32_path=tmp_path / "attn.f32",
            context_length=3,
            start_position=1,
            batch_tokens=2,
            num_heads=1,
            qk_nope_dim=2,
            rope_dim=2,
            v_head_dim=2,
            kv_lora_dim=2,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=1,
            echo_runner_output=False,
        )


def test_run_prefill_mla_attention_batch_rejects_boolean_kv_b_size(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        for tensor in tensors:
            if tensor["name"].endswith(".self_attn.kv_b_proj.weight"):
                tensor["size"] = True
                return
        raise AssertionError("kv_b fixture tensor not found")

    _mutate_layout(layout, mutate)

    with pytest.raises(PrefillExecuteError, match="size must be an integer"):
        run_prefill_mla_attention_batch(
            runner_path=runner,
            resident_layout_path=layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            layer=1,
            q_nope_f32_path=q_nope,
            q_rope_f32_path=q_rope,
            output_f32_path=tmp_path / "attn.f32",
            context_length=3,
            start_position=1,
            batch_tokens=2,
            num_heads=1,
            qk_nope_dim=2,
            rope_dim=2,
            v_head_dim=2,
            kv_lora_dim=2,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=1,
            echo_runner_output=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layer": True}, "layer must be an integer"),
        ({"context_length": False}, "context_length must be an integer"),
        ({"batch_tokens": True}, "batch_tokens must be an integer"),
        ({"num_heads": False}, "num_heads must be an integer"),
        ({"kv_lora_dim": True}, "kv_lora_dim must be an integer"),
        ({"index_topk": False}, "index_topk must be an integer"),
        ({"cache_position_offset": True}, "cache_position_offset must be an integer"),
        ({"attention_scale": False}, "attention_scale must be a finite number"),
        ({"max_cache_read_mib": True}, "max_cache_read_mib must be a finite number"),
    ],
)
def test_run_prefill_mla_attention_batch_rejects_boolean_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    args: dict[str, object] = {
        "runner_path": tmp_path / "missing-runner",
        "resident_layout_path": tmp_path / "missing-layout.json",
        "cache_layout_path": tmp_path / "missing-cache-layout.json",
        "cache_file_path": tmp_path / "missing-cache.bin",
        "layer": 1,
        "q_nope_f32_path": tmp_path / "q_nope.f32",
        "q_rope_f32_path": tmp_path / "q_rope.f32",
        "output_f32_path": tmp_path / "attn.f32",
        "context_length": 3,
        "start_position": 1,
        "batch_tokens": 2,
        "num_heads": 1,
        "qk_nope_dim": 2,
        "rope_dim": 2,
        "v_head_dim": 2,
        "kv_lora_dim": 2,
    }
    if "index_topk" in kwargs:
        args["indices_u32_path"] = tmp_path / "indices.u32"
    args.update(kwargs)

    with pytest.raises(PrefillExecuteError, match=message):
        run_prefill_mla_attention_batch(**args)


def test_run_prefill_mla_attention_batch_with_indices_invokes_indexed_runner(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    indices = tmp_path / "indices.u32"
    indices.write_bytes(struct.pack("<6I", 1, 0, 0, 2, 1, 0))
    output = tmp_path / "attn_indexed.f32"

    result = run_prefill_mla_attention_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        q_nope_f32_path=q_nope,
        q_rope_f32_path=q_rope,
        indices_u32_path=indices,
        output_f32_path=output,
        context_length=3,
        start_position=1,
        batch_tokens=2,
        index_topk=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        kv_lora_dim=2,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=1,
        echo_runner_output=False,
    )

    assert result.indexed is True
    assert result.index_topk == 2
    assert result.indices_u32_bytes == 24
    assert result.cache_read_bytes == 24
    assert result.cache_f32_bytes == 64
    assert result.mla_value_cache is False
    assert result.mla_value_cache_bytes == 0
    assert result.estimated_peak_bytes == 224
    assert struct.unpack("<4f", output.read_bytes()) == (0.625, 0.625, 0.625, 0.625)
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert "--run-mla-attention-indexed-batch" in argv
    assert "--run-mla-attention-batch" not in argv
    assert argv[argv.index("--indices-u32") + 1] == str(indices)
    assert argv[argv.index("--index-topk") + 1] == "2"


def test_run_prefill_attention_output_batch_invokes_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    runner = _write_fake_runner(tmp_path)
    attn_value = tmp_path / "attn_value.f32"
    attn_value.write_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
    residual = tmp_path / "residual.f32"
    residual.write_bytes(struct.pack("<6f", 10, 20, 30, 40, 50, 60))
    output = tmp_path / "attn_hidden.f32"

    result = run_prefill_attention_output_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        attn_value_f32_path=attn_value,
        residual_f32_path=residual,
        output_f32_path=output,
        batch_tokens=2,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    projection = output.with_name("attn_hidden.o_proj.f32")
    assert result.attn_value_dim == 2
    assert result.hidden_dim == 3
    assert result.attn_value_bytes == 16
    assert result.residual_bytes == 24
    assert result.projection_bytes == 24
    assert result.output_bytes == 24
    assert result.residual_add_peak_bytes == 36
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 16 + 24
    assert result.projection_path == projection
    assert result.o_proj.tensor == "model.layers.1.self_attn.o_proj.weight"
    assert struct.unpack("<6f", projection.read_bytes()) == (1, 2, 3, 4, 5, 9)
    assert struct.unpack("<6f", output.read_bytes()) == (11, 22, 33, 44, 55, 69)
    argv = json.loads(projection.with_suffix(projection.suffix + ".argv.json").read_text())
    assert "--run-resident-linear-batch" in argv
    assert argv[argv.index("--tensor-suffix") + 1] == ".self_attn.o_proj.weight"


def test_run_prefill_attention_output_batch_uses_fused_runner(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    fake = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake.rename(runner)
    attn_value = tmp_path / "attn_value.f32"
    attn_value.write_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
    residual = tmp_path / "residual.f32"
    residual.write_bytes(struct.pack("<6f", 10, 20, 30, 40, 50, 60))
    output = tmp_path / "attn_hidden.f32"

    result = run_prefill_attention_output_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        attn_value_f32_path=attn_value,
        residual_f32_path=residual,
        output_f32_path=output,
        batch_tokens=2,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    projection = output.with_name("attn_hidden.o_proj.f32")
    assert result.o_proj.backend == "fused-metal"
    assert result.o_proj.runner_backend_elapsed_seconds == pytest.approx(0.031)
    assert result.attn_value_bytes == 16
    assert result.residual_bytes == 24
    assert result.projection_bytes == 24
    assert result.output_bytes == 24
    assert result.residual_add_peak_bytes == 48
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 16 + 2 * 24
    assert struct.unpack("<6f", projection.read_bytes()) == (1, 2, 3, 4, 5, 6)
    assert struct.unpack("<6f", output.read_bytes()) == (11, 22, 33, 44, 55, 66)
    argv = json.loads(projection.with_suffix(projection.suffix + ".argv.json").read_text())
    assert "--run-attn-output-batch" in argv
    assert "--run-resident-linear-batch" not in argv
    assert argv[argv.index("--batch-tokens") + 1] == "2"


def test_run_prefill_attention_output_batch_can_use_server_session(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    fake = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake.rename(runner)
    attn_value = tmp_path / "attn_value.f32"
    attn_value.write_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
    residual = tmp_path / "residual.f32"
    residual.write_bytes(struct.pack("<6f", 10, 20, 30, 40, 50, 60))
    output = tmp_path / "attn_hidden.f32"

    with AttentionOutputBatchServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_prefill_attention_output_batch(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            attn_value_f32_path=attn_value,
            residual_f32_path=residual,
            output_f32_path=output,
            batch_tokens=2,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=3,
            attention_output_server_session=session,
            echo_runner_output=False,
        )

    projection = output.with_name("attn_hidden.o_proj.f32")
    assert result.o_proj.backend == "fused-metal"
    assert result.o_proj.runner_backend_elapsed_seconds == pytest.approx(0.031)
    assert result.o_proj.command == (
        str(runner),
        "--run-attn-output-batch-server-jsonl",
    )
    assert struct.unpack("<6f", projection.read_bytes()) == (1, 2, 3, 4, 5, 6)
    assert struct.unpack("<6f", output.read_bytes()) == (11, 22, 33, 44, 55, 66)
    argv = json.loads(projection.with_suffix(projection.suffix + ".argv.json").read_text())
    assert argv == ["--run-attn-output-batch-server-jsonl"]


def test_run_resident_shared_expert_batch_can_use_server_session(
    tmp_path: Path,
) -> None:
    layout = _write_shared_mxfp4_resident(tmp_path)
    fake = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake.rename(runner)
    input_path = tmp_path / "shared_input.f32"
    input_path.write_bytes(struct.pack("<16f", *[1.0] * 16))
    output = tmp_path / "shared_output.f32"

    with ResidentSharedExpertBatchServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_resident_shared_expert_batch(
            runner_path=runner,
            resident_layout_path=layout,
            layer=1,
            input_f32_path=input_path,
            output_f32_path=output,
            batch_tokens=2,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=8,
            shared_expert_server_session=session,
            echo_runner_output=False,
        )

    assert result.command == (
        str(runner),
        "--run-shared-expert-batch-server-jsonl",
    )
    assert result.runner_backend_elapsed_seconds == pytest.approx(0.044)
    assert result.hidden_dim == 8
    assert result.intermediate_dim == 8
    assert struct.unpack("<16f", output.read_bytes()) == (2.5,) * 16
    argv = json.loads(output.with_suffix(output.suffix + ".argv.json").read_text())
    assert argv == ["--run-shared-expert-batch-server-jsonl"]


def test_run_prefill_attention_output_batch_disable_fused_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT", "1")
    layout = _write_resident(tmp_path)
    fake = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake.rename(runner)
    attn_value = tmp_path / "attn_value.f32"
    attn_value.write_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
    residual = tmp_path / "residual.f32"
    residual.write_bytes(struct.pack("<6f", 10, 20, 30, 40, 50, 60))
    output = tmp_path / "attn_hidden.f32"

    result = run_prefill_attention_output_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        attn_value_f32_path=attn_value,
        residual_f32_path=residual,
        output_f32_path=output,
        batch_tokens=2,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.o_proj.backend == "custom-metal"
    projection = output.with_name("attn_hidden.o_proj.f32")
    argv = json.loads(projection.with_suffix(projection.suffix + ".argv.json").read_text())
    assert "--run-resident-linear-batch" in argv
    assert "--run-attn-output-batch" not in argv


def test_run_prefill_attention_output_batch_singleton_uses_fused_runner(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    runner = tmp_path / "largerlm-runner"
    runner.write_text(
        """#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path

args = sys.argv[1:]
if "--run-attn-output" not in args:
    raise SystemExit(97)
projection = Path(args[args.index("--projection-f32") + 1])
output = Path(args[args.index("--output-f32") + 1])
projection.parent.mkdir(parents=True, exist_ok=True)
output.parent.mkdir(parents=True, exist_ok=True)
projection.write_bytes(struct.pack("<3f", 1.0, 2.0, 9.0))
output.write_bytes(struct.pack("<3f", 11.0, 22.0, 69.0))
projection.with_suffix(projection.suffix + ".argv.json").write_text(json.dumps(args))
""",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    attn_value = tmp_path / "attn_value.f32"
    attn_value.write_bytes(struct.pack("<2f", 0.1, 0.2))
    residual = tmp_path / "residual.f32"
    residual.write_bytes(struct.pack("<3f", 10, 20, 60))
    output = tmp_path / "attn_hidden.f32"

    result = run_prefill_attention_output_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        attn_value_f32_path=attn_value,
        residual_f32_path=residual,
        output_f32_path=output,
        batch_tokens=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.o_proj.backend == "fused-metal"
    assert result.projection_path == output.with_name("attn_hidden.o_proj.f32")
    assert struct.unpack("<3f", result.projection_path.read_bytes()) == (1.0, 2.0, 9.0)
    assert struct.unpack("<3f", output.read_bytes()) == (11.0, 22.0, 69.0)
    argv = json.loads(
        result.projection_path.with_suffix(".f32.argv.json").read_text()
    )
    assert "--run-attn-output" in argv
    assert "--projection-f32" in argv


def test_run_prefill_attention_block_batch_invokes_substeps(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "attn_block"
    output = tmp_path / "attn_block_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.context_length == 3
    assert result.hidden_dim == 3
    assert result.kv_lora_dim == 2
    assert result.input_bytes == 24
    assert result.output_bytes == 24
    assert result.cache_write.encoded_bytes == 16
    assert result.cache_write_peak_bytes == 48
    assert result.projections.attention_value_source == "kv_b_proj"
    assert result.mla_attention.attention_value_source == "kv_b_proj"
    assert result.mla_key_cache is False
    assert result.mla_key_cache_bytes == 0
    assert result.mla_attention.mla_key_cache is False
    assert result.mla_attention.mla_key_cache_bytes == 0
    assert result.mla_value_cache is True
    assert result.mla_value_cache_bytes == 24
    assert result.mla_attention.mla_value_cache is True
    assert result.mla_attention.mla_value_cache_bytes == 24
    assert result.rope.q_nope_bytes == 16
    assert result.mla_attention.output_bytes == 16
    assert result.attention_output.projection_path == output_dir / "attn_output.o_proj.f32"
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 24 + 32
    assert result.projections_elapsed_seconds >= 0.0
    assert result.cache_write_elapsed_seconds >= 0.0
    assert result.rope_elapsed_seconds >= 0.0
    assert result.dsa_indexer_elapsed_seconds == 0.0
    assert result.mla_attention_elapsed_seconds >= 0.0
    assert result.attention_output_elapsed_seconds >= 0.0
    assert struct.unpack("<6f", output.read_bytes()) == (2, 4, 6, 8, 10, 15)
    env = json.loads((output_dir / "attn_value.f32.env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] is None
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_run_prefill_attention_block_batch_singleton_uses_kv_b_value_fast_path(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path, values=(1, 2, 3))
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.rename(runner)
    output_dir = tmp_path / "attn_block_singleton"
    output = tmp_path / "attn_block_singleton_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=0,
        batch_tokens=1,
        context_length=1,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.mla_attention.command[0] == "python-singleton-mla-value"
    assert result.mla_attention.attention_value_source == "kv_b_proj-singleton"
    assert result.mla_attention.cache_read_bytes == 0
    assert struct.unpack("<2f", result.mla_attention.output_path.read_bytes()) == pytest.approx(
        (0.67082, 0.89443),
        abs=1e-6,
    )
    assert not result.mla_attention.output_path.with_suffix(".f32.argv.json").exists()
    assert struct.unpack("<3f", output.read_bytes()) == (2.0, 4.0, 6.0)


def test_run_prefill_attention_block_batch_accepts_absorbed_aliases(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    _replace_kv_b_with_absorbed_aliases(layout)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "attn_block_absorbed"
    output = tmp_path / "attn_block_absorbed_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=1,
        batch_tokens=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.projections.attention_value_source == "absorbed-alias"
    assert result.projections.kv_b_proj is None
    assert result.projections.kv_b_output_bytes == 0
    assert result.mla_attention.attention_value_source == "absorbed-alias"
    assert result.mla_attention.kv_b_matrix_bytes == 32
    assert result.mla_attention.kv_b_f32_bytes == 32
    assert result.cache_write.encoded_bytes == 16
    assert result.mla_attention.output_bytes == 16
    assert not (output_dir / "projections" / "kv_b_proj.f32").exists()
    assert struct.unpack("<6f", output.read_bytes()) == (2, 4, 6, 8, 10, 15)


def test_run_prefill_attention_block_batch_full_dsa_indexes_mla(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "attn_block_full_dsa"
    output = tmp_path / "attn_block_full_dsa_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=1,
        batch_tokens=2,
        context_length=3,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        dsa_indexer_mode="full",
        dsa_index_topk=2,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        dsa_rope_interleave=True,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.dsa_indexer is not None
    assert result.dsa_indexer_elapsed_seconds >= 0.0
    assert result.dsa_rope_interleave is True
    assert result.dsa_indexer.cache_write.rope_interleave is True
    assert result.dsa_indexer.topk.rope_interleave is True
    assert result.dsa_indices_u32_path == output_dir / "dsa_topk.u32"
    assert result.dsa_indices_u32_path.stat().st_size == 24
    assert result.dsa_indexer.topk.output_indices_u32_bytes == 24
    assert result.dsa_indexer.topk.topk_indices_collected is False
    assert result.mla_attention.indexed is True
    assert result.mla_attention.indices_u32_path == result.dsa_indices_u32_path
    assert result.mla_attention.index_topk == 2
    assert result.mla_attention.cache_f32_bytes == 64
    assert "--run-mla-attention-indexed-batch" in result.mla_attention.command
    assert result.mla_attention.command[
        result.mla_attention.command.index("--indices-u32") + 1
    ] == str(result.dsa_indices_u32_path)


def test_run_prefill_attention_block_batch_can_skip_final_singleton_dsa_cache(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path, (1.0, 2.0, 3.0))
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output_dir = tmp_path / "attn_block_final_singleton"
    output = tmp_path / "attn_block_final_singleton_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=0,
        batch_tokens=1,
        context_length=1,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        dsa_indexer_mode="full",
        dsa_index_topk=2,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        write_dsa_future_cache=False,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.dsa_indexer is None
    assert result.dsa_indices_u32_path == output_dir / "dsa_topk.u32"
    assert struct.unpack("<3I", result.dsa_indices_u32_path.read_bytes()) == (1, 0, 0)
    assert result.mla_attention.indexed is False
    assert result.mla_attention.indices_u32_path is None
    assert result.mla_attention.index_topk is None
    assert result.dsa_indexer_elapsed_seconds >= 0.0
    assert struct.unpack("<3f", output.read_bytes()) == pytest.approx((2.0, 4.0, 6.0))


def test_run_prefill_attention_block_batch_can_skip_final_short_context_dsa_cache(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path, (1.0, 2.0, 3.0))
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    output_dir = tmp_path / "attn_block_final_short_context"
    output = tmp_path / "attn_block_final_short_context_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=1,
        batch_tokens=1,
        context_length=2,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        dsa_indexer_mode="full",
        dsa_index_topk=4,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        write_dsa_future_cache=False,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.dsa_indexer is None
    assert result.dsa_indices_u32_path == output_dir / "dsa_topk.u32"
    assert struct.unpack("<5I", result.dsa_indices_u32_path.read_bytes()) == (
        2,
        0,
        1,
        0,
        0,
    )
    assert result.mla_attention.indexed is False
    assert result.mla_attention.indices_u32_path is None
    assert result.mla_attention.index_topk is None
    assert "--run-mla-attention-batch" in result.mla_attention.command
    assert "--run-mla-attention-indexed-batch" not in result.mla_attention.command
    assert result.dsa_indexer_elapsed_seconds >= 0.0
    assert struct.unpack("<3f", output.read_bytes()) == pytest.approx((2.0, 4.0, 6.0))


def test_run_prefill_attention_block_batch_skips_indexed_attention_when_topk_covers_context(
    tmp_path: Path,
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "attn_block_full_dsa_visible_context"
    output = tmp_path / "attn_block_full_dsa_visible_context_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=1,
        batch_tokens=2,
        context_length=3,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        dsa_indexer_mode="full",
        dsa_index_topk=4,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.dsa_indexer is None
    assert result.dsa_indices_u32_path is None
    assert not (output_dir / "dsa_topk.u32").exists()
    assert result.mla_attention.indexed is False
    assert result.mla_attention.indices_u32_path is None
    assert result.mla_attention.index_topk is None
    assert "--run-mla-attention-batch" in result.mla_attention.command
    assert "--run-mla-attention-indexed-batch" not in result.mla_attention.command


def test_run_prefill_attention_block_batch_shared_dsa_reuses_indices(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    indices = tmp_path / "previous_dsa.u32"
    indices.write_bytes(struct.pack("<6I", 1, 1, 0, 2, 1, 2))
    output_dir = tmp_path / "attn_block_shared_dsa"
    output = tmp_path / "attn_block_shared_dsa_out.f32"

    result = run_prefill_attention_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        start_position=1,
        batch_tokens=2,
        context_length=3,
        num_heads=1,
        qk_nope_dim=2,
        rope_dim=2,
        v_head_dim=2,
        dsa_indexer_mode="shared",
        dsa_prev_indices_u32_path=indices,
        dsa_index_topk=2,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    assert result.dsa_indexer is None
    assert result.dsa_indices_u32_path == indices
    assert result.mla_attention.indexed is True
    assert result.mla_attention.indices_u32_path == indices
    assert result.mla_attention.cache_read_bytes == 24
    assert "--run-mla-attention-indexed-batch" in result.mla_attention.command
    assert result.mla_attention.command[
        result.mla_attention.command.index("--indices-u32") + 1
    ] == str(indices)


def test_run_prefill_dense_mlp_block_batch_invokes_substeps(tmp_path: Path) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "dense_mlp"
    output = tmp_path / "dense_mlp_out.f32"

    result = run_prefill_dense_mlp_block_batch(
        runner_path=runner,
        resident_layout_path=layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        batch_tokens=2,
        rms_norm_eps=0.0,
        max_resident_matrix_mib=1,
        max_runner_scratch_mib=3,
        echo_runner_output=False,
    )

    expected_swiglu = (
        0.0,
        (1.0 / (1.0 + math.exp(-1.0))) * 20.0,
        (2.0 / (1.0 + math.exp(-2.0))) * 30.0,
        (-math.exp(-1.0) / (1.0 + math.exp(-1.0))) * 40.0,
    )
    swiglu = struct.unpack("<4f", (output_dir / "swiglu.f32").read_bytes())
    assert swiglu == pytest.approx(expected_swiglu, rel=1e-6, abs=1e-6)
    assert result.hidden_dim == 3
    assert result.intermediate_dim == 2
    assert result.input_bytes == 24
    assert result.norm_output_bytes == 24
    assert result.gate_output_bytes == 16
    assert result.up_output_bytes == 16
    assert result.swiglu_output_bytes == 16
    assert result.down_output_bytes == 24
    assert result.output_bytes == 24
    assert result.swiglu_peak_bytes == 24
    assert result.residual_add_peak_bytes == 36
    assert result.estimated_peak_bytes == 2 * 1024 * 1024 + 24 + 16
    assert struct.unpack("<6f", output.read_bytes()) == (1.5, 3.5, 5.5, 7.5, 9.5, 11.5)
    gate_argv = json.loads((output_dir / "gate_proj.f32.argv.json").read_text())
    assert gate_argv[gate_argv.index("--tensor-suffix") + 1] == ".mlp.gate_proj.weight"


def test_run_prefill_routed_mlp_block_batch_invokes_runner_per_token(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_moe_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_path = tmp_path / "moe_input.f32"
    values = tuple(float(value) for value in range(16))
    input_path.write_bytes(struct.pack("<16f", *values))
    output_dir = tmp_path / "routed_mlp"
    output = tmp_path / "routed_mlp_out.f32"
    router_json_dir = tmp_path / "router_json"

    result = run_prefill_routed_mlp_block_batch(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        layer=1,
        input_f32_path=input_path,
        output_dir=output_dir,
        output_f32_path=output,
        batch_tokens=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        no_norm_topk_prob=True,
        max_slot_mib=1,
        max_router_mib=1,
        max_runner_scratch_mib=64,
        expert_read_advise_merge_gap_kib=4,
        expert_read_advise_align_kib=4,
        router_json_dir=router_json_dir,
        echo_runner_output=False,
    )

    assert result.hidden_dim == 8
    assert result.input_bytes == 64
    assert result.output_bytes == 64
    assert result.token_input_bytes == 32
    assert result.command_count == 2
    assert result.read_bytes == 2 * 2 * 192
    assert "--run-mlp-block" in result.first_command
    assert "--no-norm-topk-prob" in result.first_command
    assert "--expert-read-advise-align-kib" in result.first_command
    assert struct.unpack("<16f", output.read_bytes()) == tuple(value + 10.0 for value in values)
    assert sorted(path.name for path in router_json_dir.glob("*.json")) == [
        "token_000000.router.json",
        "token_000001.router.json",
    ]
    assert not any((output_dir / "tokens").glob("*.f32"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"layer": True}, "layer must be an integer"),
        ({"batch_tokens": False}, "batch_tokens must be an integer"),
        ({"top_k": 1.5}, "top_k must be an integer"),
        ({"max_k": True}, "max_k must be an integer"),
        ({"routed_scaling_factor": False}, "routed_scaling_factor must be a finite number"),
        ({"router_n_group": 1.5}, "router_n_group must be an integer"),
        ({"router_topk_group": True}, "router_topk_group must be an integer"),
        ({"rms_norm_eps": False}, "rms_norm_eps must be a finite number"),
        ({"max_slot_mib": 1.5}, "max_slot_mib must be an integer MiB value"),
        (
            {"expert_read_advise_merge_gap_kib": 1.5},
            "expert_read_advise_merge_gap_kib must be an integer",
        ),
        (
            {"expert_read_advise_align_kib": True},
            "expert_read_advise_align_kib must be an integer",
        ),
        (
            {"expert_read_advise_align_kib": -1},
            "expert_read_advise_align_kib must be non-negative",
        ),
    ),
)
def test_run_prefill_routed_mlp_block_batch_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    args: dict[str, object] = {
        "runner_path": tmp_path / "missing-runner",
        "expert_layout_path": tmp_path / "missing-experts.json",
        "resident_layout_path": tmp_path / "missing-resident.json",
        "layer": 1,
        "input_f32_path": tmp_path / "input.f32",
        "output_dir": tmp_path / "routed_mlp",
        "output_f32_path": tmp_path / "out.f32",
        "batch_tokens": 2,
    }
    args.update(kwargs)

    with pytest.raises(PrefillExecuteError, match=message):
        run_prefill_routed_mlp_block_batch(**args)


def test_prefill_linear_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    status = cli_main(
        [
            "prefill-linear-batch",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--tensor-suffix",
            ".self_attn.q_a_proj.weight",
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(output),
            "--batch-tokens",
            "2",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["backend"] == "custom-metal"
    assert payload["matrix_scratch_bytes"] == 2 * 1024 * 1024
    assert payload["matrix_f32_bytes"] == 0
    assert payload["runner_backend_elapsed_seconds"] == pytest.approx(0.123)
    assert payload["runner_matrix_f32_elapsed_seconds"] == pytest.approx(0.045)
    assert payload["runner_accelerator_elapsed_seconds"] == pytest.approx(0.067)
    assert payload["elapsed_seconds"] >= 0.0


def test_prefill_linear_batch_cli_accepts_auto_policy_thresholds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "out.f32"

    status = cli_main(
        [
            "prefill-linear-batch",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--tensor-suffix",
            ".self_attn.q_a_proj.weight",
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(output),
            "--batch-tokens",
            "2",
            "--prefill-linear-backend",
            "auto",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "2",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "mpsgraph-f32"
    assert payload["matrix_f32_bytes"] == 2 * 3 * 4
    assert payload["matrix_raw_conversion_bytes"] == 0
    assert payload["runner_backend_elapsed_seconds"] == pytest.approx(0.123)
    assert payload["runner_matrix_f32_elapsed_seconds"] == pytest.approx(0.045)
    assert payload["runner_accelerator_elapsed_seconds"] == pytest.approx(0.067)
    assert payload["output_bytes"] == 16
    assert payload["tensor"] == "model.layers.1.self_attn.q_a_proj.weight"


def test_prefill_linear_calibrate_cli_json_writes_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile = {
        "source": "prefill_linear_calibration",
        "argv_safe_to_replay": True,
        "sections": {
            "prefill_runtime_policy_flags": {
                "source": "prefill_linear_calibration",
                "prefill_mpsgraph_min_batch_tokens": 64,
                "prefill_mpsgraph_min_matrix_dim": 32,
                "argv": (
                    "--prefill-mpsgraph-min-batch-tokens",
                    "64",
                    "--prefill-mpsgraph-min-matrix-dim",
                    "32",
                ),
            }
        },
        "argv": (
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
        ),
    }

    def fake_calibration(**kwargs):
        assert kwargs["batch_token_values"] == (32, 64)
        assert kwargs["matrix_dim_values"] == (16, 32)
        assert kwargs["matrix_shapes"] is None
        assert kwargs["matrix_dtype"] == "BF16"
        assert kwargs["repeats"] == 2
        assert kwargs["max_calibration_case_mib"] == 4
        assert kwargs["max_calibration_work_dir_mib"] == 9
        assert kwargs["calibration_work_dir_free_margin_mib"] == 10
        return ResidentLinearCalibrationResult(
            runner_path=Path(kwargs["runner_path"]),
            work_dir=tmp_path / "calibration",
            kept_work_dir=False,
            batch_token_values=(32, 64),
            matrix_dim_values=(16, 32),
            repeats=2,
            min_mpsgraph_speedup=1.0,
            max_calibration_case_bytes=4 * 1024 * 1024,
            max_resident_matrix_mib=4,
            max_runner_scratch_mib=8,
            recommended_prefill_mpsgraph_min_batch_tokens=64,
            recommended_prefill_mpsgraph_min_matrix_dim=32,
            suggested_prefill_runtime_policy_flags=profile["sections"][
                "prefill_runtime_policy_flags"
            ],
            suggested_launch_profile=profile,
            cases=(
                ResidentLinearCalibrationCase(
                    batch_tokens=64,
                    matrix_dim=32,
                    matrix_bytes=4096,
                    input_bytes=8192,
                    output_bytes=8192,
                    estimated_flops=131072,
                    custom_elapsed_seconds=0.010,
                    mpsgraph_elapsed_seconds=0.005,
                    mpsgraph_speedup=2.0,
                    mpsgraph_meets_threshold=True,
                    winner="mpsgraph-f32",
                    custom_estimated_peak_bytes=2 * 1024 * 1024,
                    mpsgraph_estimated_peak_bytes=2 * 1024 * 1024,
                    in_dim=32,
                    out_dim=32,
                    min_matrix_dim=32,
                ),
            ),
            matrix_shapes=((16, 16), (32, 32)),
            matrix_dtype="BF16",
        )

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fake_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    profile_path = tmp_path / "profile.json"

    status = cli_main(
        [
            "prefill-linear-calibrate",
            str(runner),
            "--batch-tokens",
            "32,64",
            "--matrix-dims",
            "16,32",
            "--matrix-dtype",
            "BF16",
            "--repeats",
            "2",
            "--max-calibration-case-mib",
            "4",
            "--max-resident-matrix-mib",
            "4",
            "--max-runner-scratch-mib",
            "8",
            "--max-calibration-work-dir-mib",
            "9",
            "--calibration-work-dir-free-margin-mib",
            "10",
            "--write-launch-profile",
            str(profile_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_prefill_mpsgraph_min_batch_tokens"] == 64
    assert payload["recommended_prefill_mpsgraph_min_matrix_dim"] == 32
    assert payload["matrix_dtype"] == "BF16"
    written = json.loads(profile_path.read_text())
    assert written["argv"] == [
        "--prefill-mpsgraph-min-batch-tokens",
        "64",
        "--prefill-mpsgraph-min-matrix-dim",
        "32",
    ]


def test_prefill_linear_calibrate_cli_accepts_matrix_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_calibration(**kwargs):
        assert kwargs["matrix_shapes"] == ((3, 2), (4, 8))
        assert kwargs["matrix_dtype"] == "F32"
        return ResidentLinearCalibrationResult(
            runner_path=Path(kwargs["runner_path"]),
            work_dir=tmp_path / "calibration",
            kept_work_dir=False,
            batch_token_values=(2,),
            matrix_dim_values=(2, 4),
            repeats=1,
            min_mpsgraph_speedup=1.0,
            max_calibration_case_bytes=1024 * 1024,
            max_resident_matrix_mib=1,
            max_runner_scratch_mib=8,
            recommended_prefill_mpsgraph_min_batch_tokens=None,
            recommended_prefill_mpsgraph_min_matrix_dim=None,
            suggested_prefill_runtime_policy_flags=None,
            suggested_launch_profile=None,
            cases=(),
            matrix_shapes=((3, 2), (4, 8)),
        )

    monkeypatch.setattr(
        "largerlm.cli.run_resident_linear_calibration",
        fake_calibration,
    )
    runner = tmp_path / "runner"
    runner.write_text("#!/bin/sh\n", encoding="utf-8")

    status = cli_main(
        [
            "prefill-linear-calibrate",
            str(runner),
            "--batch-tokens",
            "2",
            "--matrix-shapes",
            "3x2,4x8",
            "--max-calibration-case-mib",
            "1",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matrix_shapes"] == [[3, 2], [4, 8]]


def test_prefill_rmsnorm_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output = tmp_path / "norm.f32"

    status = cli_main(
        [
            "prefill-rmsnorm-batch",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--norm-suffix",
            ".input_layernorm.weight",
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(output),
            "--batch-tokens",
            "2",
            "--rms-norm-eps",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["hidden_dim"] == 3
    assert payload["output_bytes"] == 24
    assert payload["tensor"] == "model.layers.1.input_layernorm.weight"


def test_prefill_attention_prefix_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "prefix"

    status = cli_main(
        [
            "prefill-attention-prefix",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--batch-tokens",
            "2",
            "--rms-norm-eps",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["hidden_dim"] == 3
    assert payload["q_a_dim"] == 2
    assert payload["kv_a_dim"] == 4
    assert payload["kv_a_output_bytes"] == 32
    assert payload["q_a_proj"]["tensor"] == "model.layers.1.self_attn.q_a_proj.weight"
    assert payload["kv_a_proj_with_mqa"]["tensor"] == (
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight"
    )


def test_prefill_attention_projections_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "projections"

    status = cli_main(
        [
            "prefill-attention-projections",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--batch-tokens",
            "2",
            "--rms-norm-eps",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["q_b_dim"] == 4
    assert payload["kv_lora_dim"] == 2
    assert payload["kv_rope_dim"] == 2
    assert payload["kv_b_dim"] == 4
    assert payload["attention_value_source"] == "kv_b_proj"
    assert payload["q_b_output_bytes"] == 32
    assert payload["kv_b_output_bytes"] == 32
    assert payload["prefix"]["kv_a_output_bytes"] == 32
    assert payload["q_b_proj"]["tensor"] == "model.layers.1.self_attn.q_b_proj.weight"
    assert payload["kv_b_proj"]["tensor"] == "model.layers.1.self_attn.kv_b_proj.weight"


def test_prefill_attention_projections_cli_json_accepts_absorbed_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    _replace_kv_b_with_absorbed_aliases(layout)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "projections_absorbed"

    status = cli_main(
        [
            "prefill-attention-projections",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--batch-tokens",
            "2",
            "--rms-norm-eps",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["attention_value_source"] == "absorbed-alias"
    assert payload["kv_b_dim"] == 4
    assert payload["kv_b_output_bytes"] == 0
    assert payload["kv_b_proj"] is None
    assert not (output_dir / "kv_b_proj.f32").exists()


def test_prefill_rope_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _write_fake_runner(tmp_path)
    q_b = tmp_path / "q_b.f32"
    q_b.write_bytes(struct.pack("<12f", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    k_rope = tmp_path / "k_rope.f32"
    k_rope.write_bytes(struct.pack("<4f", 100, 101, 102, 103))
    output_dir = tmp_path / "rope"

    status = cli_main(
        [
            "prefill-rope-batch",
            str(runner),
            "--q-b-f32",
            str(q_b),
            "--k-rope-f32",
            str(k_rope),
            "--output-dir",
            str(output_dir),
            "--batch-tokens",
            "2",
            "--num-heads",
            "2",
            "--qk-nope-dim",
            "1",
            "--rope-dim",
            "2",
            "--start-position",
            "4",
            "--max-runner-scratch-mib",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["num_heads"] == 2
    assert payload["q_nope_bytes"] == 16
    assert payload["q_rope_bytes"] == 32
    assert payload["k_rope_bytes"] == 16
    assert payload["command"][1] == "--run-rope-batch"


def test_prefill_mla_attention_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    output = tmp_path / "attn.f32"

    status = cli_main(
        [
            "prefill-mla-attention-batch",
            str(runner),
            str(layout),
            str(cache_layout),
            str(cache_file),
            "--layer",
            "1",
            "--q-nope-f32",
            str(q_nope),
            "--q-rope-f32",
            str(q_rope),
            "--output-f32",
            str(output),
            "--context-length",
            "3",
            "--start-position",
            "1",
            "--batch-tokens",
            "2",
            "--num-heads",
            "1",
            "--qk-nope-dim",
            "2",
            "--rope-dim",
            "2",
            "--v-head-dim",
            "2",
            "--kv-lora-dim",
            "2",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "1",
            "--mla-key-cache",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["output_bytes"] == 16
    assert payload["cache_read_bytes"] == 24
    assert payload["attention_value_source"] == "kv_b_proj"
    assert payload["mla_key_cache"] is True
    assert payload["mla_key_cache_bytes"] == 24
    assert payload["mla_value_cache"] is True
    assert payload["mla_value_cache_bytes"] == 24
    assert payload["estimated_peak_bytes"] == 232
    assert "--run-mla-attention-batch" in payload["command"]
    env = json.loads(output.with_suffix(output.suffix + ".env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] == "1"
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_prefill_mla_attention_batch_cli_json_with_indices(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    runner = _write_fake_runner(tmp_path)
    q_nope = tmp_path / "q_nope.f32"
    q_nope.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    q_rope = tmp_path / "q_rope.f32"
    q_rope.write_bytes(struct.pack("<4f", 5, 6, 7, 8))
    indices = tmp_path / "indices.u32"
    indices.write_bytes(struct.pack("<6I", 1, 0, 0, 2, 1, 0))
    output = tmp_path / "attn.f32"

    status = cli_main(
        [
            "prefill-mla-attention-batch",
            str(runner),
            str(layout),
            str(cache_layout),
            str(cache_file),
            "--layer",
            "1",
            "--q-nope-f32",
            str(q_nope),
            "--q-rope-f32",
            str(q_rope),
            "--indices-u32",
            str(indices),
            "--index-topk",
            "2",
            "--output-f32",
            str(output),
            "--context-length",
            "3",
            "--start-position",
            "1",
            "--batch-tokens",
            "2",
            "--num-heads",
            "1",
            "--qk-nope-dim",
            "2",
            "--rope-dim",
            "2",
            "--v-head-dim",
            "2",
            "--kv-lora-dim",
            "2",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed"] is True
    assert payload["index_topk"] == 2
    assert payload["attention_value_source"] == "kv_b_proj"
    assert payload["indices_u32_bytes"] == 24
    assert payload["cache_f32_bytes"] == 64
    assert "--run-mla-attention-indexed-batch" in payload["command"]


def test_prefill_attention_output_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    runner = _write_fake_runner(tmp_path)
    attn_value = tmp_path / "attn_value.f32"
    attn_value.write_bytes(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))
    residual = tmp_path / "residual.f32"
    residual.write_bytes(struct.pack("<6f", 10, 20, 30, 40, 50, 60))
    output = tmp_path / "attn_hidden.f32"

    status = cli_main(
        [
            "prefill-attention-output-batch",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--attn-value-f32",
            str(attn_value),
            "--residual-f32",
            str(residual),
            "--output-f32",
            str(output),
            "--batch-tokens",
            "2",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "3",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["hidden_dim"] == 3
    assert payload["output_bytes"] == 24
    assert payload["projection_bytes"] == 24
    assert payload["residual_add_peak_bytes"] == 36
    assert payload["o_proj"]["tensor"] == "model.layers.1.self_attn.o_proj.weight"


def test_prefill_attention_block_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "attn_block"
    output = tmp_path / "attn_block_out.f32"

    status = cli_main(
        [
            "prefill-attention-block-batch",
            str(runner),
            str(layout),
            str(cache_layout),
            str(cache_file),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-f32",
            str(output),
            "--start-position",
            "1",
            "--batch-tokens",
            "2",
            "--num-heads",
            "1",
            "--qk-nope-dim",
            "2",
            "--rope-dim",
            "2",
            "--v-head-dim",
            "2",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "3",
            "--mla-key-cache",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_length"] == 3
    assert payload["hidden_dim"] == 3
    assert payload["output_bytes"] == 24
    assert payload["cache_write"]["encoded_bytes"] == 16
    assert payload["projections"]["attention_value_source"] == "kv_b_proj"
    assert payload["mla_attention"]["attention_value_source"] == "kv_b_proj"
    assert payload["mla_key_cache"] is True
    assert payload["mla_key_cache_bytes"] == 24
    assert payload["mla_attention"]["mla_key_cache"] is True
    assert payload["mla_attention"]["mla_key_cache_bytes"] == 24
    assert payload["mla_value_cache"] is True
    assert payload["mla_value_cache_bytes"] == 24
    assert payload["mla_attention"]["mla_value_cache"] is True
    assert payload["mla_attention"]["mla_value_cache_bytes"] == 24
    assert payload["mla_attention"]["output_bytes"] == 16
    assert payload["attention_output"]["o_proj"]["tensor"] == (
        "model.layers.1.self_attn.o_proj.weight"
    )
    env = json.loads((output_dir / "attn_value.f32.env.json").read_text())
    assert env["LARGERLM_MLA_KEY_CACHE"] == "1"
    assert env["LARGERLM_MLA_DISABLE_VALUE_CACHE"] is None


def test_prefill_attention_block_batch_cli_json_accepts_absorbed_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    _replace_kv_b_with_absorbed_aliases(layout)
    cache_layout, cache_file = _write_cache(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "attn_block_absorbed"
    output = tmp_path / "attn_block_absorbed_out.f32"

    status = cli_main(
        [
            "prefill-attention-block-batch",
            str(runner),
            str(layout),
            str(cache_layout),
            str(cache_file),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-f32",
            str(output),
            "--start-position",
            "1",
            "--batch-tokens",
            "2",
            "--num-heads",
            "1",
            "--qk-nope-dim",
            "2",
            "--rope-dim",
            "2",
            "--v-head-dim",
            "2",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "3",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["projections"]["attention_value_source"] == "absorbed-alias"
    assert payload["projections"]["kv_b_output_bytes"] == 0
    assert payload["projections"]["kv_b_proj"] is None
    assert payload["mla_attention"]["attention_value_source"] == "absorbed-alias"
    assert payload["mla_attention"]["output_bytes"] == 16
    assert not (output_dir / "projections" / "kv_b_proj.f32").exists()


def test_prefill_dense_mlp_block_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_resident(tmp_path)
    input_path = _write_input(tmp_path)
    runner = _write_fake_runner(tmp_path)
    output_dir = tmp_path / "dense_mlp"
    output = tmp_path / "dense_mlp_out.f32"

    status = cli_main(
        [
            "prefill-dense-mlp-block-batch",
            str(runner),
            str(layout),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-f32",
            str(output),
            "--batch-tokens",
            "2",
            "--rms-norm-eps",
            "0",
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "3",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["hidden_dim"] == 3
    assert payload["intermediate_dim"] == 2
    assert payload["swiglu_output_bytes"] == 16
    assert payload["output_bytes"] == 24
    assert payload["gate_proj"]["tensor"] == "model.layers.1.mlp.gate_proj.weight"
    assert payload["down_proj"]["tensor"] == "model.layers.1.mlp.down_proj.weight"


def test_prefill_routed_mlp_block_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert_layout, resident_layout = _write_moe_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_path = tmp_path / "moe_input.f32"
    input_path.write_bytes(struct.pack("<16f", *(float(value) for value in range(16))))
    output_dir = tmp_path / "routed_mlp"
    output = tmp_path / "routed_mlp_out.f32"

    status = cli_main(
        [
            "prefill-routed-mlp-block-batch",
            str(runner),
            str(expert_layout),
            str(resident_layout),
            "--layer",
            "1",
            "--input-f32",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-f32",
            str(output),
            "--batch-tokens",
            "2",
            "--top-k",
            "2",
            "--max-k",
            "2",
            "--router-score",
            "raw",
            "--routed-scaling-factor",
            "1",
            "--no-norm-topk-prob",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["hidden_dim"] == 8
    assert payload["output_bytes"] == 64
    assert payload["top_k"] == 2
    assert payload["router_score"] == "raw"
    assert payload["command_count"] == 2
    assert payload["read_bytes"] == 2 * 2 * 192
