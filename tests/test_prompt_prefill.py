from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import largerlm.expert_io as expert_io_module
from largerlm.cli import main as cli_main
from largerlm.generation_guard import GenerationGuardError, LiveMemoryBudget
from largerlm.prompt_prefill import (
    PromptPrefillError,
    _prefill_acceleration_coverage_from_counts,
    _prefill_linear_summary_from_layout,
    prompt_prefill_acceleration_failure_reason,
    run_prompt_prefill,
)
from largerlm.safety import DiskBudget
from largerlm.token_generator import TokenGeneratorError, generate_token_ids


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


def test_prefill_acceleration_coverage_honors_min_flop_fraction() -> None:
    coverage = _prefill_acceleration_coverage_from_counts(
        {"mpsgraph-f32": 1, "custom-metal": 1},
        flops_by_backend={"mpsgraph-f32": 25, "custom-metal": 75},
        required=True,
        min_accelerated_flop_fraction=0.5,
    )

    assert coverage["ok"] is False
    assert coverage["min_accelerated_flop_fraction"] == 0.5
    assert coverage["accelerated_flop_fraction"] == 0.25
    assert "below required 0.5" in coverage["reason"]
    assert coverage["router_gate_matrix_count"] == 0
    assert coverage["router_gate_accelerated_matrix_count"] == 0
    assert coverage["accelerated_router_gate_only"] is False
    assert coverage["non_router_matrix_count"] == 2
    assert coverage["non_router_estimated_flops"] == 100
    assert coverage["non_router_unaccelerated_matrix_count"] == 1
    assert coverage["non_router_unaccelerated_estimated_flops"] == 75
    assert coverage["non_router_unaccelerated_flop_fraction"] == 0.75


def test_prefill_acceleration_coverage_reports_router_gate_only() -> None:
    coverage = _prefill_acceleration_coverage_from_counts(
        {"mpsgraph-f32": 1},
        flops_by_backend={"mpsgraph-f32": 25},
        router_gate_matrix_count=1,
        router_gate_estimated_flops=25,
        router_gate_accelerated_matrix_count=1,
        router_gate_accelerated_estimated_flops=25,
    )

    assert coverage["router_gate_matrix_count"] == 1
    assert coverage["router_gate_estimated_flops"] == 25
    assert coverage["router_gate_accelerated_matrix_count"] == 1
    assert coverage["router_gate_accelerated_estimated_flops"] == 25
    assert coverage["non_router_accelerated_matrix_count"] == 0
    assert coverage["non_router_accelerated_estimated_flops"] == 0
    assert coverage["non_router_matrix_count"] == 0
    assert coverage["non_router_estimated_flops"] == 0
    assert coverage["non_router_unaccelerated_matrix_count"] == 0
    assert coverage["non_router_unaccelerated_estimated_flops"] == 0
    assert coverage["accelerated_router_gate_flop_share"] == 1.0
    assert coverage["accelerated_router_gate_only"] is True


def test_prefill_acceleration_failure_rejects_router_gate_only_by_default() -> None:
    result = type(
        "PromptPrefillStub",
        (),
        {
            "prefill_acceleration_coverage": _prefill_acceleration_coverage_from_counts(
                {"mpsgraph-f32": 1},
                flops_by_backend={"mpsgraph-f32": 25},
                router_gate_matrix_count=1,
                router_gate_estimated_flops=25,
                router_gate_accelerated_matrix_count=1,
                router_gate_accelerated_estimated_flops=25,
            ),
            "prefill_acceleration_frontier": None,
        },
    )()

    reason = prompt_prefill_acceleration_failure_reason(result)

    assert reason is not None
    assert "only from MoE router gates" in reason
    assert (
        prompt_prefill_acceleration_failure_reason(
            result,
            allow_router_gate_only_acceleration=True,
        )
        is None
    )


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _add_tensor(
    tensors: list[dict],
    payload: bytearray,
    name: str,
    dtype: str,
    shape: list[int],
    data: bytes,
    category: str,
) -> None:
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


def _write_fixture(root: Path, *, include_dsa: bool = False) -> tuple[Path, Path, Path, Path, Path]:
    experts = root / "experts"
    resident = root / "resident"
    experts.mkdir()
    resident.mkdir()

    expert_layout = experts / "layout.json"
    slot_bytes = 192
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
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
            }
        ),
        encoding="utf-8",
    )
    (experts / "layer_001.bin").write_bytes(
        b"".join(bytes([expert]) * slot_bytes for expert in range(2))
    )

    tensors: list[dict] = []
    payload = bytearray()
    embedding_values: list[float] = []
    for token in range(5):
        embedding_values.extend(float(token * 10 + dim) for dim in range(8))
    _add_tensor(
        tensors,
        payload,
        "model.embed_tokens.weight",
        "F32",
        [5, 8],
        f32(embedding_values),
        "embeddings",
    )
    _add_tensor(
        tensors,
        payload,
        "model.norm.weight",
        "F32",
        [8],
        f32([1.0] * 8),
        "norms",
    )
    for layer in (0, 1):
        _add_tensor(
            tensors,
            payload,
            f"model.layers.{layer}.input_layernorm.weight",
            "F32",
            [8],
            f32([1.0] * 8),
            "norms",
        )
        _add_tensor(
            tensors,
            payload,
            f"model.layers.{layer}.self_attn.q_a_layernorm.weight",
            "F32",
            [2],
            f32([1.0] * 2),
            "norms",
        )
        _add_tensor(
            tensors,
            payload,
            f"model.layers.{layer}.self_attn.kv_a_layernorm.weight",
            "F32",
            [2],
            f32([1.0] * 2),
            "norms",
        )
        _add_tensor(
            tensors,
            payload,
            f"model.layers.{layer}.post_attention_layernorm.weight",
            "F32",
            [8],
            f32([1.0] * 8),
            "norms",
        )
        matrices = [
            ("self_attn.q_a_proj.weight", [2, 8], "attention"),
            ("self_attn.q_b_proj.weight", [6, 2], "attention"),
            ("self_attn.kv_a_proj_with_mqa.weight", [4, 8], "attention"),
            ("self_attn.kv_b_proj.weight", [4, 2], "attention"),
            ("self_attn.o_proj.weight", [8, 2], "attention"),
        ]
        for suffix, shape, category in matrices:
            rows, cols = shape
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.{suffix}",
                "F32",
                shape,
                f32([0.0] * (rows * cols)),
                category,
            )
        if include_dsa:
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.self_attn.indexer.wk.weight",
                "F32",
                [2, 8],
                f32(
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                "dsa_indexer",
            )
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.self_attn.indexer.k_norm.weight",
                "F32",
                [2],
                f32([1.0, 1.0]),
                "dsa_indexer",
            )
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.self_attn.indexer.k_norm.bias",
                "F32",
                [2],
                f32([0.0, 0.0]),
                "dsa_indexer",
            )
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.self_attn.indexer.wq_b.weight",
                "F32",
                [2, 2],
                f32([1.0, 0.0, 0.0, 1.0]),
                "dsa_indexer",
            )
            _add_tensor(
                tensors,
                payload,
                f"model.layers.{layer}.self_attn.indexer.weights_proj.weight",
                "F32",
                [1, 8],
                f32([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                "dsa_indexer",
            )
    for component in ("gate_proj", "up_proj", "down_proj"):
        _add_tensor(
            tensors,
            payload,
            f"model.layers.0.mlp.{component}.weight",
            "F32",
            [8, 8],
            f32([0.0] * 64),
            "dense_mlp",
        )
    _add_tensor(
        tensors,
        payload,
        "model.layers.1.mlp.gate.weight",
        "F32",
        [2, 8],
        f32([1.0] * 16),
        "routers",
    )
    resident_layout = resident / "layout.json"
    resident_layout.write_text(
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
    (resident / "resident.bin").write_bytes(payload)

    cache_total_bytes = 96
    cache_segments = [
        {
            "kind": "mla_kv",
            "layer": 0,
            "offset": 0,
            "width": 4,
            "dtype": "BF16",
            "dtype_bytes": 2,
            "token_stride_bytes": 8,
            "max_context_tokens": 5,
            "total_bytes": 40,
        },
        {
            "kind": "mla_kv",
            "layer": 1,
            "offset": 56,
            "width": 4,
            "dtype": "BF16",
            "dtype_bytes": 2,
            "token_stride_bytes": 8,
            "max_context_tokens": 5,
            "total_bytes": 40,
        },
    ]
    if include_dsa:
        cache_total_bytes = 136
        cache_segments.extend(
            [
                {
                    "kind": "dsa_index",
                    "layer": 0,
                    "offset": 96,
                    "width": 2,
                    "dtype": "BF16",
                    "dtype_bytes": 2,
                    "token_stride_bytes": 4,
                    "max_context_tokens": 5,
                    "total_bytes": 20,
                },
                {
                    "kind": "dsa_index",
                    "layer": 1,
                    "offset": 116,
                    "width": 2,
                    "dtype": "BF16",
                    "dtype_bytes": 2,
                    "token_stride_bytes": 4,
                    "max_context_tokens": 5,
                    "total_bytes": 20,
                },
            ]
        )

    cache_layout = root / "cache_layout.json"
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 5,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": cache_total_bytes,
                "segments": cache_segments,
            }
        ),
        encoding="utf-8",
    )
    cache_file = root / "decode_cache.bin"
    cache_file.write_bytes(b"\0" * cache_total_bytes)

    config_payload = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 8,
        "moe_intermediate_size": 8,
        "n_routed_experts": 2,
        "n_shared_experts": 0,
        "num_experts_per_tok": 2,
        "num_attention_heads": 2,
        "q_lora_rank": 2,
        "kv_lora_rank": 2,
        "qk_nope_head_dim": 1,
        "qk_rope_head_dim": 2,
        "v_head_dim": 1,
        "scoring_func": "raw",
        "routed_scaling_factor": 1.0,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "mlp_layer_types": ["dense", "sparse"],
    }
    if include_dsa:
        config_payload.update(
            {
                "indexer_types": ["full", "shared"],
                "index_topk": 2,
                "index_n_heads": 1,
                "index_head_dim": 2,
            }
        )
    config = root / "config.json"
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    return expert_layout, resident_layout, cache_layout, cache_file, config


def _write_fake_runner(root: Path) -> Path:
    runner = root / "fake_runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)


def value(flag: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1]


def zeros(path: str, count: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(struct.pack("<" + "f" * count, *([0.0] * count)))


def copy_shifted(input_path: str, output_path: str, delta: float) -> None:
    raw = Path(input_path).read_bytes()
    count = len(raw) // 4
    values = struct.unpack("<" + "f" * count, raw)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(
        struct.pack("<" + "f" * count, *(item + delta for item in values))
    )


def run_moe_batch(input_path: str, output_path: str, batch_tokens: int, requested: str) -> None:
    raw = Path(input_path).read_bytes()
    count = len(raw) // 4
    values = struct.unpack("<" + "f" * count, raw)
    Path(output_path).write_bytes(
        struct.pack("<" + "f" * count, *(item + 1.0 for item in values))
    )
    effective = batch_tokens if requested == "auto" else min(int(requested), batch_tokens)
    print("LargerLM MoE batch")
    print(f"  token block mode:   {'auto' if requested == 'auto' else 'fixed'}")
    print(f"  max expert tokens:  {batch_tokens}")
    print(f"  token block:        {effective}")
    accumulator = os.environ.get("LARGERLM_MOE_BATCH_ACCUMULATOR", "file").strip().lower()
    if accumulator in ("1", "memory", "mem", "in-memory", "ram", "true", "yes", "on"):
        accumulator = "memory"
    else:
        accumulator = "file"
    print(f"  output accumulator: {accumulator}")
    print(f"  output accum bytes: {count * 4 if accumulator == 'memory' else 0}")
    print(f"  batch buffer bytes: {effective * 100}")
    print(f"  estimated peak:     {effective * 1000}")


def run_moe_plan(plan_path: str) -> None:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    for index, job in enumerate(plan["jobs"], start=1):
        print("LargerLM MoE batch plan job")
        print(f"  job:                 {index}/{len(plan['jobs'])}")
        run_moe_batch(
            job["input_f32"],
            job["output_f32"],
            int(job["batch_tokens"]),
            str(job.get("moe_token_block", "auto")),
        )
    print("LargerLM MoE batch plan")
    print(f"  jobs:                {len(plan['jobs'])}")
    print("  plan timing total:   0.123000")


def resident_linear_out_dim(suffix: str) -> int:
    if suffix.endswith("q_a_proj.weight"):
        return 2
    if suffix.endswith("q_b_proj.weight"):
        return 6
    if suffix.endswith("kv_a_proj_with_mqa.weight"):
        return 4
    if suffix.endswith("kv_b_proj.weight"):
        return 4
    if suffix.endswith("o_proj.weight"):
        return 8
    if suffix.endswith("mlp.gate.weight"):
        return 2
    if suffix.endswith("mlp.gate_proj.weight"):
        return 8
    if suffix.endswith("mlp.up_proj.weight"):
        return 8
    if suffix.endswith("mlp.down_proj.weight"):
        return 8
    raise SystemExit(f"unknown tensor suffix {suffix}")


def run_resident_linear_batch(input_path: str, output_path: str, suffix: str, batch: int) -> None:
    _ = input_path
    zeros(output_path, batch * resident_linear_out_dim(suffix))
    print("LargerLM resident batch linear")
    print("  timing backend:     0.123000")
    print("  timing matrix f32:  0.000000")
    print("  timing accelerator: 0.000000")


def run_attention_projection_outputs(output_dir: str, batch: int) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "attn_input_norm.f32": batch * 8,
        "attn_q_a.f32": batch * 2,
        "attn_q_a_norm.f32": batch * 2,
        "attn_q_b.f32": batch * 6,
        "attn_kv_a.f32": batch * 4,
        "attn_kv_a_norm.f32": batch * 2,
        "attn_kv_b.f32": batch * 4,
    }
    for name, count in outputs.items():
        zeros(str(out_dir / name), count)


def run_attention_output_batch(request: dict[str, object], command: str) -> None:
    residual_path = Path(
        str(request.get("residual_f32") or request.get("residual_f32_path"))
    )
    output_path = Path(str(request.get("output_f32") or request.get("output_f32_path")))
    projection_path_raw = request.get("projection_f32") or request.get(
        "projection_f32_path"
    )
    raw = residual_path.read_bytes()
    count = len(raw) // 4
    if projection_path_raw is not None:
        projection_path = Path(str(projection_path_raw))
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        zeros(str(projection_path), count)
        projection_path.with_suffix(projection_path.suffix + ".argv.json").write_text(
            json.dumps([command])
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    output_path.with_suffix(output_path.suffix + ".argv.json").write_text(
        json.dumps([command])
    )
    print("  timing backend:     0.031000")


def run_rope_split_batch(request: dict[str, object]) -> None:
    q_b = Path(str(request.get("q_b_f32") or request.get("q_b_f32_path")))
    out_nope = Path(
        str(request.get("output_q_nope_f32") or request.get("output_q_nope_f32_path"))
    )
    out_rope = Path(
        str(request.get("output_q_rope_f32") or request.get("output_q_rope_f32_path"))
    )
    out_q = Path(str(request.get("output_q_f32") or request.get("output_q_f32_path")))
    out_k = Path(str(request.get("output_k_f32") or request.get("output_k_f32_path")))
    batch = int(request["batch_tokens"])
    heads = int(request["num_heads"])
    nope = int(request["qk_nope_dim"])
    rope = int(request["rope_dim"])
    raw = q_b.read_bytes()
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    q_nope = []
    q_rope = []
    head_dim = nope + rope
    for token in range(batch):
        row_base = token * heads * head_dim
        for head in range(heads):
            base = row_base + head * head_dim
            q_nope.extend(values[base : base + nope])
            q_rope.extend(values[base + nope : base + head_dim])
    out_nope.parent.mkdir(parents=True, exist_ok=True)
    out_rope.parent.mkdir(parents=True, exist_ok=True)
    out_nope.write_bytes(struct.pack("<" + "f" * len(q_nope), *q_nope))
    out_rope.write_bytes(struct.pack("<" + "f" * len(q_rope), *q_rope))
    zeros(str(out_q), batch * heads * rope)
    zeros(str(out_k), batch * rope)
    out_q.with_suffix(out_q.suffix + ".argv.json").write_text(
        json.dumps(["--run-rope-split-batch-server-jsonl"])
    )


def run_rmsnorm_batch(request: dict[str, object]) -> None:
    input_path = Path(str(request.get("input_f32") or request.get("input_f32_path")))
    output_path = Path(str(request.get("output_f32") or request.get("output_f32_path")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(input_path.read_bytes())
    output_path.with_suffix(output_path.suffix + ".argv.json").write_text(
        json.dumps(["--run-rmsnorm-batch-server-jsonl"])
    )


def run_mla_attention_batch(request: dict[str, object]) -> None:
    output_path = Path(str(request.get("output_f32") or request.get("output_f32_path")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch = int(request["batch_tokens"])
    heads = int(request["num_heads"])
    v_head = int(request["v_head_dim"])
    zeros(str(output_path), batch * heads * v_head)
    output_path.with_suffix(output_path.suffix + ".argv.json").write_text(
        json.dumps(["--run-mla-attention-batch-server-jsonl"])
    )


if "--run-rmsnorm-batch" in sys.argv:
    Path(value("--output-f32")).write_bytes(Path(value("--input-f32")).read_bytes())
elif "--run-resident-linear-batch" in sys.argv:
    run_resident_linear_batch(
        value("--input-f32"),
        value("--output-f32"),
        value("--tensor-suffix"),
        int(value("--batch-tokens")),
    )
elif "--run-rope-split-batch" in sys.argv:
    run_rope_split_batch(
        {
            "q_b_f32": value("--q-b-f32"),
            "k_f32": value("--k-f32"),
            "output_q_nope_f32": value("--output-q-nope-f32"),
            "output_q_rope_f32": value("--output-q-rope-f32"),
            "output_q_f32": value("--output-q-f32"),
            "output_k_f32": value("--output-k-f32"),
            "num_heads": value("--num-heads"),
            "qk_nope_dim": value("--qk-nope-dim"),
            "rope_dim": value("--rope-dim"),
            "start_position": value("--start-position"),
            "batch_tokens": value("--batch-tokens"),
        }
    )
elif "--run-rope-batch" in sys.argv:
    batch = int(value("--batch-tokens"))
    heads = int(value("--num-heads"))
    rope = int(value("--rope-dim"))
    zeros(value("--output-q-f32"), batch * heads * rope)
    zeros(value("--output-k-f32"), batch * rope)
elif "--run-mla-attention-batch" in sys.argv:
    batch = int(value("--batch-tokens"))
    heads = int(value("--num-heads"))
    v_head = int(value("--v-head-dim"))
    zeros(value("--output-f32"), batch * heads * v_head)
elif "--run-mla-attention-indexed-batch" in sys.argv:
    batch = int(value("--batch-tokens"))
    heads = int(value("--num-heads"))
    v_head = int(value("--v-head-dim"))
    zeros(value("--output-f32"), batch * heads * v_head)
elif "--run-mla-attention-batch-server-jsonl" in sys.argv:
    print("LargerLM MLA attention batch server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM MLA attention batch server request")
        run_mla_attention_batch(request)
        print("  server request:      ok")
    print("LargerLM MLA attention batch server done")
elif "--run-attn-output-batch-server-jsonl" in sys.argv:
    print("LargerLM attention output batch server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM attention output batch server request")
        run_attention_output_batch(request, "--run-attn-output-batch-server-jsonl")
        print("  server request:      ok")
    print("LargerLM attention output batch server done")
elif "--run-shared-expert-batch-server-jsonl" in sys.argv:
    print("LargerLM resident shared expert batch server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        output_path = Path(
            str(request.get("output_f32") or request.get("output_f32_path"))
        )
        input_path = Path(
            str(request.get("input_f32") or request.get("input_f32_path"))
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(input_path.read_bytes())
        output_path.with_suffix(output_path.suffix + ".argv.json").write_text(
            json.dumps(["--run-shared-expert-batch-server-jsonl"])
        )
        print("LargerLM resident shared expert batch server request")
        print("  timing backend:     0.044000")
        print("  server request:      ok")
    print("LargerLM resident shared expert batch server done")
elif "--run-router-batch" in sys.argv:
    out_dir = Path(value("--output-router-json-dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = int(value("--batch-tokens"))
    for token in range(batch):
        (out_dir / f"token_{token:06d}.router.json").write_text(
            json.dumps({"experts": [1, 0], "weights": [0.5, 0.5]}),
            encoding="utf-8",
        )
elif "--run-moe-batch-plan-server-jsonl" in sys.argv:
    print("LargerLM MoE batch plan server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        plan_path = request.get("batch_plan_json") or request.get("plan_path")
        print("LargerLM MoE batch plan server request")
        print(f"  plan:                {plan_path}")
        run_moe_plan(plan_path)
        print("  server request:      ok")
    print("LargerLM MoE batch plan server done")
elif "--run-resident-linear-batch-plan-server-jsonl" in sys.argv:
    print("LargerLM resident batch linear server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM resident batch linear server request")
        run_resident_linear_batch(
            request.get("input_f32") or request.get("input_f32_path"),
            request.get("output_f32") or request.get("output_f32_path"),
            request["tensor_suffix"],
            int(request["batch_tokens"]),
        )
        print("  server request:      ok")
    print("LargerLM resident batch linear server done")
elif "--run-rope-split-batch-server-jsonl" in sys.argv:
    print("LargerLM RoPE split batch server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM RoPE split batch server request")
        run_rope_split_batch(request)
        print("  server request:      ok")
    print("LargerLM RoPE split batch server done")
elif "--run-rmsnorm-batch-server-jsonl" in sys.argv:
    print("LargerLM resident RMSNorm batch server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM resident RMSNorm batch server request")
        run_rmsnorm_batch(request)
        print("  server request:      ok")
    print("LargerLM resident RMSNorm batch server done")
elif "--run-attn-projections-server-jsonl" in sys.argv:
    print("LargerLM attention projections server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        print("LargerLM attention projections server request")
        out_dir = request.get("output_dir") or request.get("output_dir_path")
        batch = int(request["batch_tokens"])
        run_attention_projection_outputs(out_dir, batch)
        argv = [
            "--run-attn-projections-server-jsonl",
            "--resident-layout",
            str(request.get("resident_layout") or request.get("resident_layout_path")),
            "--layer",
            str(request["layer"]),
            "--input-f32",
            str(request.get("input_f32") or request.get("input_f32_path")),
            "--batch-tokens",
            str(batch),
            "--output-dir",
            str(out_dir),
        ]
        (Path(out_dir) / "attn_q_b.f32.argv.json").write_text(json.dumps(argv))
        print("  server request:      ok")
    print("LargerLM attention projections server done")
elif "--run-attn-projections" in sys.argv:
    run_attention_projection_outputs(
        value("--output-dir"),
        int(value("--batch-tokens")),
    )
elif "--run-attn-output" in sys.argv:
    run_attention_output_batch(
        {
            "residual_f32": value("--residual-f32"),
            "output_f32": value("--output-f32"),
            "projection_f32": value("--projection-f32"),
        },
        "--run-attn-output",
    )
elif "--run-attn-output-batch" in sys.argv:
    run_attention_output_batch(
        {
            "residual_f32": value("--residual-f32"),
            "output_f32": value("--output-f32"),
            "projection_f32": value("--projection-f32"),
        },
        "--run-attn-output-batch",
    )
elif "--run-moe-batch-plan" in sys.argv:
    run_moe_plan(value("--batch-plan-json"))
elif "--run-moe-batch" in sys.argv:
    run_moe_batch(
        value("--input-f32"),
        value("--output-f32"),
        int(value("--batch-tokens")),
        value("--moe-token-block") if "--moe-token-block" in sys.argv else "auto",
    )
elif "--run-dense-decoder-layer" in sys.argv:
    copy_shifted(value("--input-f32"), value("--output-f32"), 0.0)
elif "--run-decoder-layer" in sys.argv:
    copy_shifted(value("--input-f32"), value("--output-f32"), 1.0)
else:
    raise SystemExit(f"unsupported fake runner command: {' '.join(sys.argv[1:])}")
""",
        encoding="utf-8",
    )
    os.chmod(runner, 0o755)
    return runner


def _read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack("<" + "f" * (len(raw) // 4), raw)


def test_prefill_linear_summary_reports_mpp_tensor_ops_candidates(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "resident-layout.json"
    layout.write_text(
        json.dumps(
            {
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "dtype": "BF16",
                        "shape": [32, 32],
                        "size": 32 * 32 * 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = _prefill_linear_summary_from_layout(
        resident_layout_path=layout,
        prefill_linear_backend="auto",
        prompt_chunk_tokens=128,
        mpsgraph_min_batch_tokens=256,
        mpsgraph_min_matrix_dim=32,
    )

    expected_flops = 2 * 128 * 32 * 32
    assert summary["matrix_count"] == 1
    assert summary["custom_metal_matrix_count"] == 1
    assert summary["mpp_candidate_policy"] == {
        "candidate_backend": "mpp_tensor_ops_prefill",
        "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
        "mpp_tensor_ops_min_batch_tokens": 128,
        "mpp_tensor_ops_min_matrix_dim": 32,
        "selectable_prefill_backend": False,
    }
    assert summary["mpp_tensor_ops_candidate_matrix_count"] == 1
    assert summary["mpp_tensor_ops_candidate_estimated_flops"] == expected_flops
    assert summary["mpp_tensor_ops_candidate_flop_fraction"] == 1.0
    assert summary["mpp_tensor_ops_candidate_backend_counts"] == {"custom-metal": 1}
    assert summary["mpp_tensor_ops_candidate_backend_flops"] == {
        "custom-metal": expected_flops
    }
    assert summary["non_router_unaccelerated_matrix_count"] == 1
    assert summary["non_router_unaccelerated_estimated_flops"] == expected_flops
    assert summary[
        "non_router_unaccelerated_streamed_routed_expert_matrix_count"
    ] == 0
    assert summary[
        "non_router_unaccelerated_streamed_routed_expert_estimated_flops"
    ] == 0
    assert summary["non_router_unaccelerated_non_streamed_matrix_count"] == 1
    assert (
        summary["non_router_unaccelerated_non_streamed_estimated_flops"]
        == expected_flops
    )
    assert summary["unaccelerated_backend_matrix_counts"] == {"custom-metal": 1}
    assert summary["unaccelerated_backend_estimated_flops"] == {
        "custom-metal": expected_flops
    }


def test_prefill_linear_summary_excludes_router_gate_backend_candidates(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "resident-layout.json"
    layout.write_text(
        json.dumps(
            {
                "tensors": [
                    {
                        "name": "model.layers.0.mlp.gate.weight",
                        "dtype": "BF16",
                        "shape": [256, 6144],
                        "size": 256 * 6144 * 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = _prefill_linear_summary_from_layout(
        resident_layout_path=layout,
        prefill_linear_backend="auto",
        prompt_chunk_tokens=16,
        mpsgraph_min_batch_tokens=32,
        mpsgraph_min_matrix_dim=32,
    )

    assert summary["matrix_count"] == 0
    assert summary["mpsgraph_matrix_count"] == 0
    assert summary["router_gate_matrix_count"] == 0
    assert summary["router_gate_accelerated_matrix_count"] == 0
    assert summary["any_resident_matrix_accelerated"] is False

    accelerated = _prefill_linear_summary_from_layout(
        resident_layout_path=layout,
        prefill_linear_backend="auto",
        prompt_chunk_tokens=16,
        mpsgraph_min_batch_tokens=16,
        mpsgraph_min_matrix_dim=32,
    )

    assert accelerated["matrix_count"] == 1
    assert accelerated["mpsgraph_matrix_count"] == 1
    assert accelerated["router_gate_matrix_count"] == 1
    assert accelerated["router_gate_accelerated_matrix_count"] == 1
    assert accelerated["non_router_accelerated_matrix_count"] == 0
    assert accelerated["non_router_unaccelerated_matrix_count"] == 0
    assert accelerated["unaccelerated_backend_matrix_counts"] == {}
    assert accelerated["unaccelerated_backend_estimated_flops"] == {}
    assert accelerated["accelerated_router_gate_flop_share"] == 1.0
    assert accelerated["accelerated_router_gate_only"] is True
    assert accelerated["any_resident_matrix_accelerated"] is True


def test_prefill_linear_summary_counts_mps_matrix_as_accelerated(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "resident-layout.json"
    layout.write_text(
        json.dumps(
            {
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "dtype": "BF16",
                        "shape": [32, 32],
                        "size": 32 * 32 * 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = _prefill_linear_summary_from_layout(
        resident_layout_path=layout,
        prefill_linear_backend="mps-matrix-f32",
        prompt_chunk_tokens=128,
        mpsgraph_min_batch_tokens=256,
        mpsgraph_min_matrix_dim=32,
    )

    expected_flops = 2 * 128 * 32 * 32
    assert summary["matrix_count"] == 1
    assert summary["accelerated_matrix_count"] == 1
    assert summary["mpsgraph_matrix_count"] == 0
    assert summary["custom_metal_matrix_count"] == 0
    assert summary["accelerated_estimated_flops"] == expected_flops
    assert summary["accelerated_backends"] == ("mps-matrix-f32",)
    assert summary["any_resident_matrix_accelerated"] is True
    assert summary["all_resident_matrices_accelerated"] is True
    assert summary["total_matrix_raw_conversion_bytes"] == 32 * 32 * 2


def test_prefill_linear_summary_folds_affine_int4_triplets(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "resident-layout.json"
    layout.write_text(
        json.dumps(
            {
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.weight",
                        "dtype": "U32",
                        "shape": [32, 4],
                        "size": 32 * 4 * 4,
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.scales",
                        "dtype": "BF16",
                        "shape": [32, 1],
                        "size": 32 * 1 * 2,
                    },
                    {
                        "name": "model.layers.0.self_attn.q_a_proj.biases",
                        "dtype": "BF16",
                        "shape": [32, 1],
                        "size": 32 * 1 * 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = _prefill_linear_summary_from_layout(
        resident_layout_path=layout,
        prefill_linear_backend="auto",
        prompt_chunk_tokens=128,
        mpsgraph_min_batch_tokens=256,
        mpsgraph_min_matrix_dim=32,
    )

    expected_flops = 2 * 128 * 32 * 32
    assert summary["matrix_count"] == 1
    assert summary["custom_metal_matrix_count"] == 1
    assert summary["mpsgraph_matrix_count"] == 0
    assert summary["total_estimated_flops"] == expected_flops
    assert summary["mpp_tensor_ops_candidate_matrix_count"] == 1
    assert summary["mpp_tensor_ops_candidate_estimated_flops"] == expected_flops
    assert summary["total_matrix_raw_conversion_bytes"] == 0
    assert summary["max_matrix_scratch_bytes"] == 2 * 1024 * 1024


def test_run_prompt_prefill_chunks_prompt_and_extracts_last_hidden(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "last_hidden.f32"
    final_chunk = tmp_path / "final_chunk.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        output_final_chunk_f32_path=final_chunk,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        mla_key_cache=True,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        expert_stage_max_raw_ranges=8,
        expert_stage_max_coalesced_ranges=2,
        static_capacity_per_expert="auto",
        prefill_ssd_read_gib_per_second=1.0,
        prefill_max_routed_read_seconds=1.0,
        echo_runner_output=False,
    )

    assert result.chunk_count == 2
    assert result.layers == (0, 1)
    assert result.dense_layers == (0,)
    assert [chunk.start_position for chunk in result.chunks] == [0, 2]
    assert [layer.kind for layer in result.chunks[0].layers] == ["dense", "moe"]
    assert result.chunks[0].layers[0].attention.mla_key_cache is True
    assert result.chunks[0].layers[0].attention.mla_key_cache_bytes == 16
    assert result.chunks[0].layers[1].attention.mla_key_cache is True
    assert result.chunks[0].layers[1].attention.mla_key_cache_bytes == 16
    assert result.chunks[1].layers[0].attention.mla_key_cache is True
    assert result.chunks[1].layers[0].attention.mla_key_cache_bytes == 24
    streamed_routed_flops = 6 * 6 * 8 * 8
    assert result.linear_backend_counts == {"custom-metal": 32}
    assert set(result.linear_backend_elapsed_seconds or {}) == {"custom-metal"}
    custom_elapsed = (result.linear_backend_elapsed_seconds or {})["custom-metal"]
    assert custom_elapsed > 0.0
    assert set(result.linear_backend_estimated_tflops or {}) == {"custom-metal"}
    assert (result.linear_backend_estimated_tflops or {})[
        "custom-metal"
    ] == pytest.approx(result.linear_backend_flops["custom-metal"] / custom_elapsed / 1e12)
    component_stats = result.linear_backend_component_stats
    assert component_stats is not None
    assert component_stats["attention.o_proj"]["linear_backend_counts"][
        "custom-metal"
    ] > 0
    coverage = result.prefill_acceleration_coverage
    assert coverage is not None
    assert coverage["ok"] is True
    assert coverage["matrix_count"] == 32
    assert coverage["accelerated_matrix_count"] == 0
    assert coverage["custom_metal_matrix_count"] == 32
    assert coverage["non_router_unaccelerated_matrix_count"] == 32
    assert coverage["non_router_unaccelerated_estimated_flops"] == (
        result.linear_backend_flops["custom-metal"]
    )
    assert coverage["streamed_routed_expert_layer_count"] == 2
    assert coverage["streamed_routed_expert_matrix_count"] == 6
    assert coverage["streamed_routed_expert_assignments"] == 6
    assert coverage["streamed_routed_expert_estimated_flops"] == streamed_routed_flops
    assert coverage[
        "non_router_unaccelerated_streamed_routed_expert_matrix_count"
    ] == 6
    assert coverage[
        "non_router_unaccelerated_streamed_routed_expert_estimated_flops"
    ] == streamed_routed_flops
    assert coverage["non_router_unaccelerated_non_streamed_matrix_count"] == 26
    assert (
        coverage["non_router_unaccelerated_non_streamed_estimated_flops"]
        == result.linear_backend_flops["custom-metal"] - streamed_routed_flops
    )
    assert coverage["unaccelerated_backend_matrix_counts"] == {"custom-metal": 32}
    assert coverage["unaccelerated_backend_estimated_flops"] == {
        "custom-metal": result.linear_backend_flops["custom-metal"]
    }
    streamed_component = component_stats["moe.routed_experts_streamed"]
    assert streamed_component["linear_backend_counts"]["custom-metal"] == coverage[
        "streamed_routed_expert_matrix_count"
    ]
    assert streamed_component["linear_backend_flops"]["custom-metal"] == coverage[
        "streamed_routed_expert_estimated_flops"
    ]
    assert coverage["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert coverage["mpp_tensor_ops_candidate_estimated_flops"] == 0
    assert coverage["mpp_tensor_ops_candidate_flop_fraction"] == 0.0
    assert coverage["mpp_tensor_ops_candidate_backend_counts"] == {}
    assert coverage["mpp_tensor_ops_candidate_backend_flops"] == {}
    assert coverage["any_resident_matrix_accelerated"] is False
    assert coverage["all_resident_matrices_accelerated"] is False
    accel_frontier = result.prefill_acceleration_frontier
    assert accel_frontier is not None
    assert accel_frontier["source"] == "prompt_prefill_actual"
    assert accel_frontier["prompt_token_count"] == 3
    assert accel_frontier["resolved_prompt_chunk_tokens"] == 2
    actual_candidate = next(
        item for item in accel_frontier["candidates"] if item["is_resolved"]
    )
    assert actual_candidate["prompt_chunk_tokens"] == 2
    assert actual_candidate["matrix_count"] == 32
    assert actual_candidate["custom_metal_matrix_count"] == 32
    assert actual_candidate["streamed_routed_expert_estimated_flops"] == (
        streamed_routed_flops
    )
    assert actual_candidate[
        "non_router_unaccelerated_streamed_routed_expert_estimated_flops"
    ] == streamed_routed_flops
    assert (
        actual_candidate["non_router_unaccelerated_non_streamed_estimated_flops"]
        == result.linear_backend_flops["custom-metal"] - streamed_routed_flops
    )
    assert actual_candidate["unaccelerated_backend_matrix_counts"] == {
        "custom-metal": 32
    }
    assert actual_candidate["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert actual_candidate["mpp_tensor_ops_candidate_backend_counts"] == {}
    assert result.total_linear_matrix_scratch_bytes == 26 * 2 * 1024 * 1024
    assert result.max_linear_matrix_scratch_bytes == 2 * 1024 * 1024
    assert result.total_linear_matrix_f32_bytes == 0
    assert result.total_linear_matrix_raw_conversion_bytes == 0
    assert result.total_expert_stage_serial_read_bytes == 1152
    assert result.total_expert_stage_unique_requested_bytes == 768
    assert result.total_expert_stage_planned_read_bytes == 768
    assert result.total_expert_stage_planned_read_seconds == pytest.approx(
        768 / 1024**3
    )
    assert result.prefill_ssd_read_gib_per_second == 1.0
    assert result.prefill_max_routed_read_seconds == 1.0
    assert result.total_expert_stage_read_seconds_ok is True
    assert result.total_expert_stage_copy_seconds_ok is True
    assert result.total_expert_stage_waste_bytes == 0
    assert result.total_expert_stage_coalesced_savings_bytes == 384
    assert result.total_expert_stage_assignment_read_amplification == pytest.approx(
        2 / 3
    )
    assert result.total_expert_stage_unique_read_amplification == pytest.approx(1.0)
    assert result.total_expert_stage_read_advice_attempted_ranges > 0
    assert result.total_expert_stage_read_advice_calls >= 0
    assert result.total_expert_stage_read_advice_bytes >= 0
    assert result.total_expert_stage_read_advice_failures >= 0
    assert result.total_expert_stage_copy_read_calls > 0
    assert result.total_expert_stage_copy_write_calls > 0
    assert result.total_expert_stage_copy_average_read_bytes is not None
    assert result.total_expert_stage_copy_average_read_bytes > 0
    assert result.total_expert_stage_copy_average_write_bytes is not None
    assert result.total_expert_stage_copy_average_write_bytes > 0
    assert result.total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib
    assert result.total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib[
        "8"
    ] > 0
    assert result.prefill_max_stage_raw_ranges == 8
    assert result.prefill_max_stage_coalesced_ranges == 2
    assert result.total_expert_stage_raw_ranges > 0
    assert result.total_expert_stage_coalesced_ranges > 0
    assert result.max_expert_stage_raw_ranges > 0
    assert result.max_expert_stage_coalesced_ranges > 0
    assert result.max_expert_stage_raw_ranges <= 8
    assert result.max_expert_stage_coalesced_ranges <= 2
    assert result.total_expert_stage_raw_ranges_ok is True
    assert result.total_expert_stage_coalesced_ranges_ok is True
    assert result.max_expert_stage_unique_read_amplification == pytest.approx(1.0)
    assert result.max_expert_stage_stage_budget_utilization == pytest.approx(384 / 1024**2)
    assert result.total_routed_expert_assignments == 6
    assert result.total_routed_unique_expert_slots == 4
    assert result.max_routed_unique_experts_per_call == 2
    assert result.max_routed_tokens_per_expert == 2
    assert result.moe_token_block == "auto"
    assert result.moe_token_block_mode_counts == {"auto": 2}
    assert result.max_effective_moe_token_block == 2
    assert result.max_moe_max_expert_tokens == 2
    assert result.max_moe_batch_buffer_bytes == 200
    assert result.max_moe_estimated_peak_bytes == 2000
    assert result.static_capacity_per_expert == "auto"
    assert result.max_static_capacity_per_expert == 2
    assert result.total_static_capacity_used_slots == 6
    assert result.total_static_capacity_slots == 6
    assert result.total_static_capacity_overflow_assignments == 0
    assert result.total_static_capacity_binary_bytes > 0
    assert (
        result.live_memory_budget.max_live_working_set_bytes
        == 8192 * 1024**2
    )
    assert result.live_memory_budget.estimated_live_working_set_bytes > 0
    first_moe = result.chunks[0].layers[1].staged_mlp
    assert first_moe is not None
    assert first_moe.stage_result.io_summary.serial_read_bytes == 768
    assert first_moe.stage_result.io_summary.unique_requested_bytes == 384
    assert first_moe.stage_result.io_summary.planned_read_bytes == 384
    assert first_moe.stage_result.io_summary.raw_range_count > 0
    assert first_moe.stage_result.io_summary.coalesced_range_count > 0
    assert result.total_expert_stage_raw_ranges >= (
        first_moe.stage_result.io_summary.raw_range_count
    )
    assert result.total_expert_stage_coalesced_ranges >= (
        first_moe.stage_result.io_summary.coalesced_range_count
    )
    assert result.expert_stage_io_stage_count >= 1
    assert 1 <= len(result.expert_stage_copy_hotspots) <= 5
    assert 1 <= len(result.expert_stage_range_hotspots) <= 5
    copy_hotspot = result.expert_stage_copy_hotspots[0]
    assert copy_hotspot.layer in result.layers
    assert copy_hotspot.chunk_index >= 0
    assert copy_hotspot.tile_index >= 0
    assert len(copy_hotspot.selected_experts) == copy_hotspot.selected_expert_count
    assert copy_hotspot.coalesced_range_count > 0
    assert copy_hotspot.copy_read_calls >= 0
    range_hotspot = result.expert_stage_range_hotspots[0]
    assert range_hotspot.layer in result.layers
    assert range_hotspot.raw_range_count >= range_hotspot.coalesced_range_count
    assert first_moe.stage_result.read_advice.attempted_ranges > 0
    assert result.total_expert_stage_read_advice_attempted_ranges >= (
        first_moe.stage_result.read_advice.attempted_ranges
    )
    assert result.total_expert_stage_read_advice_calls >= (
        first_moe.stage_result.read_advice.calls
    )
    assert result.total_expert_stage_read_advice_bytes >= (
        first_moe.stage_result.read_advice.advised_bytes
    )
    assert first_moe.stage_plus_compact_bytes == (
        first_moe.staged_bytes + first_moe.compact_stage_bytes
    )
    assert first_moe.compact_stage_materialized_bytes == 0
    assert first_moe.stage_plus_compact_materialized_bytes == first_moe.staged_bytes
    assert first_moe.static_capacity_per_expert == 2
    assert first_moe.static_capacity_path is None
    assert first_moe.staged_moe.static_capacity_path is None
    assert first_moe.static_capacity_binary_path is not None
    assert "--routes-bin" in first_moe.staged_moe.first_command
    assert result.total_staged_bytes > 0
    assert result.total_compact_stage_bytes > 0
    assert result.total_compact_stage_materialized_bytes == 0
    assert result.max_staged_bytes >= first_moe.staged_bytes
    assert result.max_compact_stage_bytes >= first_moe.compact_stage_bytes
    assert result.max_compact_stage_materialized_bytes == 0
    assert result.total_stage_plus_compact_bytes == (
        result.total_staged_bytes + result.total_compact_stage_bytes
    )
    assert result.total_stage_plus_compact_materialized_bytes == (
        result.total_staged_bytes
    )
    assert result.max_stage_plus_compact_bytes >= first_moe.stage_plus_compact_bytes
    assert result.max_stage_plus_compact_materialized_bytes >= first_moe.staged_bytes
    assert final_chunk.stat().st_size == 8 * 4
    assert _read_f32(last_hidden) == (61.0, 63.0, 65.0, 67.0, 69.0, 71.0, 73.0, 75.0)


def test_run_prompt_prefill_can_use_persistent_moe_plan_server(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "last_hidden_persistent_moe.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_moe_plan_server=True,
        moe_output_accumulator="memory",
        echo_runner_output=False,
    )

    moe_records = [
        layer.staged_mlp
        for chunk in result.chunks
        for layer in chunk.layers
        if layer.staged_mlp is not None
    ]

    assert result.persistent_moe_plan_server is True
    assert result.moe_output_accumulator == "memory"
    assert result.moe_plan_server_plan_count == 2
    assert result.moe_plan_server_job_count == 2
    assert result.routed_moe_runner_command_count == 0
    assert len(moe_records) == 2
    assert all(record.routed_command_count == 0 for record in moe_records)
    assert all(record.staged_moe.command_count == 0 for record in moe_records)
    assert all(
        record.staged_moe.moe_output_accumulator == "memory" for record in moe_records
    )
    assert sorted(
        record.staged_moe.moe_output_accumulator_bytes for record in moe_records
    ) == [32, 64]
    assert all(
        "--run-moe-batch-plan-server-jsonl" in record.staged_moe.first_command
        for record in moe_records
    )
    assert result.moe_token_block_mode_counts == {"auto": 2}
    assert result.max_effective_moe_token_block == 2
    assert _read_f32(last_hidden) == (61.0, 63.0, 65.0, 67.0, 69.0, 71.0, 73.0, 75.0)


def test_run_prompt_prefill_can_use_persistent_resident_linear_server(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "last_hidden_persistent_linear.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_resident_linear_server=True,
        echo_runner_output=False,
    )

    server_flag = "--run-resident-linear-batch-plan-server-jsonl"
    first_layer = result.chunks[0].layers[0]
    assert result.persistent_resident_linear_server is True
    assert server_flag in first_layer.attention.projections.prefix.q_a_proj.command
    assert server_flag in first_layer.attention.projections.prefix.kv_a_proj_with_mqa.command
    assert first_layer.dense_mlp is not None
    assert server_flag in first_layer.dense_mlp.gate_proj.command
    assert server_flag in first_layer.dense_mlp.up_proj.command
    assert server_flag in first_layer.dense_mlp.down_proj.command
    assert _read_f32(last_hidden) == (
        61.0,
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    )


def test_run_prompt_prefill_can_use_persistent_attention_projection_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH", "1")
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT", "1")
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    last_hidden = tmp_path / "last_hidden_persistent_attention_projection.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_attention_projection_server=True,
        echo_runner_output=False,
    )

    server_flag = "--run-attn-projections-server-jsonl"
    first_layer = result.chunks[0].layers[0]
    projections = first_layer.attention.projections
    assert result.persistent_attention_projection_server is True
    assert server_flag in projections.prefix.q_a_proj.command
    assert server_flag in projections.prefix.kv_a_proj_with_mqa.command
    assert server_flag in projections.q_a_layernorm.command
    assert server_flag in projections.q_b_proj.command
    assert server_flag in projections.kv_a_layernorm.command
    assert server_flag in projections.kv_b_proj.command
    assert _read_f32(last_hidden) == (
        61.0,
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    )


def test_run_prompt_prefill_can_use_persistent_attention_output_server(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    last_hidden = tmp_path / "last_hidden_persistent_attention_output.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_attention_output_server=True,
        echo_runner_output=False,
    )

    server_flag = "--run-attn-output-batch-server-jsonl"
    first_layer = result.chunks[0].layers[0]
    assert result.persistent_attention_output_server is True
    assert first_layer.attention.attention_output.o_proj.command == (
        str(runner),
        server_flag,
    )
    assert _read_f32(last_hidden) == (
        61.0,
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    )


def test_run_prompt_prefill_can_use_persistent_rope_split_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT", "1")
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    last_hidden = tmp_path / "last_hidden_persistent_rope_split.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_rope_split_server=True,
        echo_runner_output=False,
    )

    server_flag = "--run-rope-split-batch-server-jsonl"
    first_layer = result.chunks[0].layers[0]
    assert result.persistent_rope_split_server is True
    assert first_layer.attention.rope.command == (str(runner), server_flag)
    assert _read_f32(last_hidden) == (
        61.0,
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    )


def test_run_prompt_prefill_can_use_persistent_rmsnorm_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_PROJECTIONS", "1")
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT", "1")
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    last_hidden = tmp_path / "last_hidden_persistent_rmsnorm.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_rmsnorm_server=True,
        echo_runner_output=False,
    )

    server_flag = "--run-rmsnorm-batch-server-jsonl"
    first_layer = result.chunks[0].layers[0]
    assert result.persistent_rmsnorm_server is True
    assert first_layer.attention.projections.prefix.input_layernorm.command == (
        str(runner),
        server_flag,
    )
    assert first_layer.dense_mlp.post_attention_layernorm.command == (
        str(runner),
        server_flag,
    )
    assert _read_f32(last_hidden) == (
        61.0,
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    )


def test_run_prompt_prefill_can_use_persistent_mla_attention_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_PROJECTIONS", "1")
    monkeypatch.setenv("LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT", "1")
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    fake_runner = _write_fake_runner(tmp_path)
    runner = tmp_path / "largerlm-runner"
    fake_runner.replace(runner)
    last_hidden = tmp_path / "last_hidden_persistent_mla.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        static_capacity_per_expert="auto",
        persistent_mla_attention_server=True,
        echo_runner_output=False,
    )

    server_flag = "--run-mla-attention-batch-server-jsonl"
    first_layer = result.chunks[0].layers[0]
    assert result.persistent_mla_attention_server is True
    assert first_layer.attention.mla_attention.command == (str(runner), server_flag)
    assert _read_f32(last_hidden) == (
        61.0,
        63.0,
        65.0,
        67.0,
        69.0,
        71.0,
        73.0,
        75.0,
    )


def test_run_prompt_prefill_can_use_persistent_moe_plan_server_with_tiling(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=tmp_path / "last_hidden_tiled_session.f32",
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=0.0002,
        max_compact_stage_mib=0.0002,
        expert_stage_tiling=True,
        static_capacity_per_expert="auto",
        persistent_moe_plan_server=True,
        echo_runner_output=False,
    )

    moe_records = [
        layer.staged_mlp
        for chunk in result.chunks
        for layer in chunk.layers
        if layer.staged_mlp is not None
    ]

    assert result.persistent_moe_plan_server is True
    assert result.moe_plan_server_plan_count == 4
    assert result.moe_plan_server_job_count == 4
    assert result.routed_moe_runner_command_count == 0
    assert len(moe_records) == 2
    assert all(record.tiled_staged_moe is not None for record in moe_records)
    assert all(record.routed_command_count == 0 for record in moe_records)
    assert all(
        tile.command_count == 0
        for record in moe_records
        for tile in record.tiled_staged_moe.tile_results
    )


def test_run_prompt_prefill_can_tile_expert_stage(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "last_hidden_tiled.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=0.0002,
        max_compact_stage_mib=0.0002,
        expert_stage_tiling=True,
        static_capacity_per_expert="auto",
        prefill_ssd_read_gib_per_second=1.0,
        prefill_max_routed_read_seconds=1.0,
        echo_runner_output=False,
    )

    moe_records = [
        layer.staged_mlp
        for chunk in result.chunks
        for layer in chunk.layers
        if layer.staged_mlp is not None
    ]

    assert len(moe_records) == 2
    assert all(record.tiled_staged_moe is not None for record in moe_records)
    assert all(record.tiled_staged_moe.tile_count == 2 for record in moe_records)
    assert all(record.compact_stage_storage == "tiled" for record in moe_records)
    assert result.total_routed_expert_assignments == 6
    assert result.total_routed_unique_expert_slots == 4
    assert result.max_routed_unique_experts_per_call == 1
    assert result.max_routed_tokens_per_expert == 2
    assert result.moe_token_block_mode_counts == {"auto": 4}
    assert result.total_static_capacity_used_slots == 6
    assert result.total_static_capacity_slots == 6
    assert result.total_static_capacity_binary_bytes > 0
    assert result.total_staged_bytes == 768
    assert result.total_compact_stage_bytes == 768
    assert result.total_expert_stage_planned_read_bytes == 768
    assert last_hidden.stat().st_size == 8 * 4


def test_run_prompt_prefill_enforces_cumulative_stage_read_seconds_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "prompt_read_seconds_work"

    monkeypatch.setattr(
        expert_io_module.time,
        "perf_counter",
        lambda: 0.0,
    )

    with pytest.raises(PromptPrefillError, match="planned stage read time"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 2, 3],
            output_last_hidden_f32_path=tmp_path / "last_hidden_slow.f32",
            work_dir=work_dir,
            keep_work_dir=True,
            dense_layers={0},
            prompt_chunk_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            rms_norm_eps=0.0,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            prefill_ssd_read_gib_per_second=1.0,
            prefill_max_routed_read_seconds=5e-7,
            echo_runner_output=False,
        )

    first_stage = (
        work_dir
        / "chunk_0000"
        / "layer_0001"
        / "staged_mlp"
        / "experts.stage.bin"
    )
    second_stage = (
        work_dir
        / "chunk_0001"
        / "layer_0001"
        / "staged_mlp"
        / "experts.stage.bin"
    )
    assert first_stage.exists()
    assert not second_stage.exists()


def test_run_prompt_prefill_uses_full_and_shared_dsa_indices(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(
        tmp_path,
        include_dsa=True,
    )
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "last_hidden_dsa.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        work_dir=tmp_path / "prefill_dsa_work",
        keep_work_dir=True,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        dsa_indexer_types=["full", "shared"],
        dsa_index_topk=2,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        dsa_rope_interleave=True,
        echo_runner_output=False,
    )

    assert result.chunk_count == 2
    for chunk in result.chunks:
        full_layer, shared_layer = chunk.layers
        assert full_layer.dsa_indexer_mode == "full"
        assert full_layer.dsa_rope_interleave is True
        assert shared_layer.dsa_indexer_mode == "shared"
        assert shared_layer.dsa_rope_interleave is True
        context_fits_topk = chunk.start_position + chunk.batch_tokens <= 2
        if context_fits_topk:
            assert full_layer.attention.dsa_indexer is None
            assert full_layer.dsa_indices_u32_path is None
            assert shared_layer.attention.dsa_indexer is None
            assert shared_layer.dsa_indices_u32_path is None
        else:
            assert full_layer.attention.dsa_indexer is not None
            assert full_layer.dsa_indices_u32_path is not None
            assert full_layer.dsa_indices_u32_path.stat().st_size == chunk.batch_tokens * 12
            assert shared_layer.attention.dsa_indexer is None
            assert shared_layer.dsa_indices_u32_path == full_layer.dsa_indices_u32_path
        assert full_layer.attention.mla_attention.indexed is not context_fits_topk
        assert shared_layer.attention.mla_attention.indexed is not context_fits_topk


def test_run_prompt_prefill_skips_dsa_future_cache_when_request_fits_topk(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(
        tmp_path,
        include_dsa=True,
    )
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "last_hidden_dsa_visible.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        work_dir=tmp_path / "prefill_dsa_visible_work",
        keep_work_dir=True,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        dsa_indexer_types=["full", "shared"],
        dsa_index_topk=8,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        write_dsa_future_cache=False,
        echo_runner_output=False,
    )

    assert result.chunk_count == 2
    for chunk in result.chunks:
        full_layer, shared_layer = chunk.layers
        assert full_layer.dsa_indexer_mode == "full"
        assert shared_layer.dsa_indexer_mode == "shared"
        assert full_layer.attention.dsa_indexer is None
        assert full_layer.dsa_indices_u32_path is not None
        assert full_layer.dsa_indices_u32_path.stat().st_size == chunk.batch_tokens * 36
        assert full_layer.attention.mla_attention.indexed is False
        assert shared_layer.attention.dsa_indexer is None
        assert shared_layer.dsa_indices_u32_path == full_layer.dsa_indices_u32_path
        assert shared_layer.attention.mla_attention.indexed is False


def test_run_prompt_prefill_rejects_selected_shared_dsa_without_selected_full(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(
        tmp_path,
        include_dsa=True,
    )
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PromptPrefillError, match="selected DSA layer 1 is shared"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            layers={1},
            prompt_chunk_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            dsa_indexer_types=["full", "shared"],
            dsa_index_topk=2,
            dsa_index_n_heads=1,
            dsa_qk_rope_dim=2,
            echo_runner_output=False,
        )


def test_run_prompt_prefill_cleans_chunk_work_when_not_kept(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "prompt_work"
    last_hidden = tmp_path / "last_hidden_clean.f32"

    result = run_prompt_prefill(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        output_last_hidden_f32_path=last_hidden,
        work_dir=work_dir,
        dense_layers={0},
        prompt_chunk_tokens=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        echo_runner_output=False,
    )

    assert result.chunk_count == 2
    assert last_hidden.exists()
    assert not (work_dir / "chunk_0000").exists()
    assert not (work_dir / "chunk_0001").exists()


def test_generate_token_ids_can_batch_prefill_prompt(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        max_new_tokens=1,
        dense_layers={0},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=2,
        prefill_max_cache_write_mib=1,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (4,)
    assert result.prompt_prefill is not None
    assert result.prompt_prefill.chunk_count == 2
    assert result.auto_prefill_prompt_chunk_plan is None
    assert result.max_safe_prefill_prompt_chunk_plan is not None
    assert result.max_safe_prefill_prompt_chunk_plan.chunk_tokens >= 2
    assert result.steps[0].position == 2
    assert result.steps[0].input_token_id == 3
    assert result.steps[0].embedding_read_bytes == 3 * 8 * 4
    assert result.steps[0].expert_read_bytes > 0


def test_generate_token_ids_reports_prefill_and_decode_read_time(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        max_new_tokens=2,
        dense_layers={0},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=2,
        prefill_max_cache_write_mib=1,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        prefill_ssd_read_gib_per_second=16.0,
        prefill_max_routed_read_seconds=1.0,
        decode_max_routed_read_seconds_per_token=1.0,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (4, 4)
    assert len(result.steps) == 2
    assert result.steps[0].position == 2
    assert not result.steps[0].decode_layers
    assert result.steps[0].expert_read_bytes > 0
    assert result.steps[1].position == 3
    assert result.steps[1].decode_layers
    assert tuple(layer.kind for layer in result.steps[1].decode_layers) == (
        "dense",
        "moe",
    )
    assert result.steps[1].expert_read_bytes == 2 * 192
    assert result.prompt_prefill is not None
    assert result.prompt_prefill.chunk_count == 2
    assert result.prefill_actual_read_time is not None
    assert result.prefill_actual_read_time["source"] == "generation_actual_prefill"
    assert result.prefill_actual_read_time[
        "total_expert_stage_planned_read_bytes"
    ] == result.prompt_prefill.total_expert_stage_planned_read_bytes
    assert result.prefill_actual_read_time[
        "total_expert_stage_planned_read_seconds"
    ] == pytest.approx(
        result.prompt_prefill.total_expert_stage_planned_read_bytes
        / (16 * 1024**3)
    )
    assert result.prefill_actual_read_time["prefill_max_routed_read_seconds"] == 1.0
    assert result.prefill_actual_read_time[
        "total_expert_stage_read_seconds_ok"
    ] is True
    assert result.prefill_actual_read_time[
        "total_expert_stage_copy_seconds_ok"
    ] is True
    assert result.decode_actual_read_time is not None
    assert result.decode_actual_read_time["source"] == "generation_actual_decode"
    assert result.decode_actual_read_time["decode_step_count"] == 1
    assert result.decode_actual_read_time["decode_read_bytes_per_token"] == 384
    assert result.decode_actual_read_time[
        "actual_decode_routed_read_bytes"
    ] == result.steps[1].expert_read_bytes
    assert result.decode_actual_read_time[
        "actual_decode_routed_read_seconds"
    ] == pytest.approx(result.steps[1].expert_read_bytes / (16 * 1024**3))
    assert result.decode_actual_read_time[
        "total_decode_routed_read_seconds_ok"
    ] is True


def test_generate_token_ids_auto_sizes_batch_prefill_prompt(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        max_new_tokens=1,
        dense_layers={0},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=0,
        prefill_max_prompt_batch_mib=64 / (1024 * 1024),
        prefill_max_cache_write_mib=1,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (4,)
    assert result.prompt_prefill is not None
    assert result.prompt_prefill.chunk_tokens == 2
    assert result.prompt_prefill.chunk_count == 2
    assert result.auto_prefill_prompt_chunk_plan is not None
    assert result.auto_prefill_prompt_chunk_plan.chunk_tokens == 2
    assert result.max_safe_prefill_prompt_chunk_plan is not None
    assert result.max_safe_prefill_prompt_chunk_plan.chunk_tokens == 2


def test_generate_token_ids_rejects_explicit_prefill_chunk_over_safety_cap(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "too_large_prefill_work"

    with pytest.raises(TokenGeneratorError) as exc_info:
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 2, 3],
            max_new_tokens=1,
            dense_layers={0},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            batch_prefill_prompt=True,
            prefill_prompt_chunk_tokens=3,
            prefill_max_prompt_batch_mib=64 / (1024 * 1024),
            prefill_max_cache_write_mib=1,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert "prefill_prompt_chunk_tokens 3 exceeds safety-capped maximum 2" in str(
        exc_info.value
    )
    assert not work_dir.exists()


def test_generate_token_ids_rejects_prefill_routed_read_amplification_over_cap(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "too_much_routed_read_work"

    with pytest.raises(TokenGeneratorError) as exc_info:
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 2, 3],
            max_new_tokens=1,
            dense_layers={0},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            batch_prefill_prompt=True,
            prefill_prompt_chunk_tokens=2,
            prefill_max_routed_read_amplification=1.5,
            prefill_max_cache_write_mib=1,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    message = str(exc_info.value)
    assert "prefill routed expert read amplification 2 exceeds cap 1.5" in message
    assert "planned_read_bytes=768" in message
    assert "baseline_read_bytes=384" in message
    assert "use prefill_prompt_chunk_tokens>=3" in message
    assert not work_dir.exists()


def test_generate_token_ids_rejects_prefill_routed_read_bytes_over_cap(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "too_many_routed_read_bytes_work"

    with pytest.raises(TokenGeneratorError) as exc_info:
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 2, 3],
            max_new_tokens=1,
            dense_layers={0},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            batch_prefill_prompt=True,
            prefill_prompt_chunk_tokens=2,
            prefill_max_routed_read_amplification=3.0,
            prefill_max_routed_read_gib=1e-7,
            prefill_max_cache_write_mib=1,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    message = str(exc_info.value)
    assert "prefill routed expert planned read 768 bytes exceeds cap 107 bytes" in message
    assert "read_amplification=2" in message
    assert "baseline_read_bytes=384" in message
    assert "no prompt chunk size can satisfy these routed-read caps" in message
    assert not work_dir.exists()


def test_generate_token_ids_rejects_prefill_routed_read_seconds_over_cap(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "too_slow_routed_read_work"

    with pytest.raises(TokenGeneratorError) as exc_info:
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 2, 3],
            max_new_tokens=1,
            dense_layers={0},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            batch_prefill_prompt=True,
            prefill_prompt_chunk_tokens=2,
            prefill_ssd_read_gib_per_second=1.0,
            prefill_max_routed_read_seconds=1e-7,
            prefill_max_cache_write_mib=1,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    message = str(exc_info.value)
    assert "prefill routed expert planned read time" in message
    assert "exceeds cap 1e-07s" in message
    assert "planned_read_bytes=768" in message
    assert "ssd_read_gib_per_second=1" in message
    assert "no prompt chunk size can satisfy these routed-read caps" in message
    assert not work_dir.exists()


def test_generate_token_ids_rejects_strict_static_capacity_below_chunk(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "too_small_static_capacity_work"

    with pytest.raises(TokenGeneratorError) as exc_info:
        generate_token_ids(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 2, 3],
            max_new_tokens=1,
            dense_layers={0},
            work_dir=work_dir,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            rms_norm_eps=0.0,
            logits_top_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            batch_prefill_prompt=True,
            prefill_prompt_chunk_tokens=2,
            prefill_static_capacity_per_expert=1,
            prefill_max_cache_write_mib=1,
            prefill_max_stage_mib=1,
            prefill_max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert (
        "prefill_static_capacity_per_expert 1 cannot guarantee overflow-free "
        "strict routing for a prompt chunk of 2 tokens"
        in str(exc_info.value)
    )
    assert not work_dir.exists()


def test_generate_token_ids_auto_prefill_respects_cache_read_cap(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        max_new_tokens=1,
        dense_layers={0},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=32 / (1024 * 1024),
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=0,
        prefill_max_cache_write_mib=1,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        echo_runner_output=False,
    )

    assert result.generated_token_ids == (4,)
    assert result.prompt_prefill is not None
    assert result.prompt_prefill.chunk_tokens == 1
    assert result.prompt_prefill.chunk_count == 3


def test_generate_token_ids_auto_prefill_respects_work_disk_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    def fake_disk_budget(
        output_dir: str | Path,
        required_bytes: int,
        *,
        safety_margin_bytes: int,
    ) -> DiskBudget:
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=20_000,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr("largerlm.token_generator.disk_budget", fake_disk_budget)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        max_new_tokens=1,
        dense_layers={0},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=0,
        prefill_max_cache_write_mib=1,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        echo_runner_output=False,
    )

    assert result.prompt_prefill is not None
    assert result.prompt_prefill.chunk_tokens == 2
    assert result.prompt_prefill.chunk_count == 2


def test_generate_token_ids_can_batch_prefill_prompt_with_dsa(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(
        tmp_path,
        include_dsa=True,
    )
    runner = _write_fake_runner(tmp_path)

    result = generate_token_ids(
        runner_path=runner,
        expert_layout_path=expert,
        resident_layout_path=resident,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        prompt_token_ids=[0, 2, 3],
        max_new_tokens=1,
        dense_layers={0},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        kv_lora_dim=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        rms_norm_eps=0.0,
        logits_top_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        batch_prefill_prompt=True,
        prefill_prompt_chunk_tokens=2,
        prefill_max_cache_write_mib=1,
        prefill_max_stage_mib=1,
        prefill_max_compact_stage_mib=1,
        dsa_indexer_types=["full", "shared"],
        dsa_index_topk=2,
        dsa_index_n_heads=1,
        dsa_index_head_dim=2,
        dsa_qk_rope_dim=2,
        dsa_rope_interleave=True,
        echo_runner_output=False,
    )

    assert result.runtime_guard is not None
    assert result.runtime_guard.dsa_indexer_runtime is True
    assert result.prompt_prefill is not None
    assert result.prompt_prefill.chunks[0].layers[0].dsa_indexer_mode == "full"
    assert result.prompt_prefill.chunks[0].layers[0].dsa_rope_interleave is True
    assert result.prompt_prefill.chunks[0].layers[1].dsa_indexer_mode == "shared"


def test_run_prompt_prefill_rejects_prompt_batch_limit(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(PromptPrefillError, match="embedding batch output"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            dense_layers={0},
            prompt_chunk_tokens=2,
            max_prompt_batch_mib=0.00001,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"prompt_token_ids": [True]}, "prompt_token_ids must be an integer"),
        ({"prompt_token_ids": [1.5]}, "prompt_token_ids must be an integer"),
        ({"start_position": True}, "start_position must be an integer"),
        ({"start_position": 1.5}, "start_position must be an integer"),
        ({"prompt_chunk_tokens": True}, "prompt_chunk_tokens must be an integer"),
        ({"prompt_chunk_tokens": 1.5}, "prompt_chunk_tokens must be an integer"),
        ({"prompt_chunk_tokens": 0}, "prompt_chunk_tokens must be positive"),
        ({"num_heads": False}, "num_heads must be an integer"),
        ({"qk_nope_dim": 1.5}, "qk_nope_dim must be an integer"),
        ({"rope_dim": True}, "rope_dim must be an integer"),
        ({"v_head_dim": 1.5}, "v_head_dim must be an integer"),
        ({"kv_lora_dim": True}, "kv_lora_dim must be an integer"),
        ({"cache_position_offset": 1.5}, "cache_position_offset must be an integer"),
        ({"top_k": 1.5}, "top_k must be an integer"),
        ({"max_k": True}, "max_k must be an integer"),
        ({"router_n_group": 1.5}, "router_n_group must be an integer"),
        ({"router_topk_group": True}, "router_topk_group must be an integer"),
        ({"dsa_index_topk": False}, "dsa_index_topk must be an integer"),
        ({"dsa_index_n_heads": 1.5}, "dsa_index_n_heads must be an integer"),
        ({"dsa_qk_rope_dim": True}, "dsa_qk_rope_dim must be an integer"),
        ({"expected_vocab_size": False}, "expected_vocab_size must be an integer"),
        ({"expected_hidden_size": True}, "expected_hidden_size must be an integer"),
        ({"layers": {True}}, "layers must be an integer"),
        ({"dense_layers": {False}}, "layers must be an integer"),
        (
            {"static_capacity_per_expert": True},
            "static_capacity_per_expert must be an integer",
        ),
        (
            {"static_capacity_per_expert": 1.5},
            "static_capacity_per_expert must be an integer",
        ),
        ({"max_cache_file_mib": 0.0}, "max_cache_file_mib"),
        ({"max_cache_write_mib": 0.0}, "max_cache_write_mib"),
        ({"max_runner_scratch_mib": float("nan")}, "max_runner_scratch_mib"),
        ({"expert_stage_merge_gap_kib": -1.0}, "expert_stage_merge_gap_kib"),
        ({"expert_stage_align_kib": 0.0}, "expert_stage_align_kib"),
        (
            {"expert_stage_max_raw_ranges": True},
            "expert_stage_max_raw_ranges must be an integer",
        ),
        (
            {"expert_stage_max_raw_ranges": -1},
            "expert_stage_max_raw_ranges must be non-negative",
        ),
        (
            {"expert_stage_max_coalesced_ranges": True},
            "expert_stage_max_coalesced_ranges must be an integer",
        ),
        (
            {"expert_stage_max_coalesced_ranges": -1},
            "expert_stage_max_coalesced_ranges must be non-negative",
        ),
        ({"expert_stage_tiling": 1}, "expert_stage_tiling must be a boolean"),
        (
            {"persistent_moe_plan_server": 1},
            "persistent_moe_plan_server must be a boolean",
        ),
        (
            {"persistent_resident_linear_server": 1},
            "persistent_resident_linear_server must be a boolean",
        ),
        (
            {"persistent_attention_projection_server": 1},
            "persistent_attention_projection_server must be a boolean",
        ),
        (
            {"persistent_attention_output_server": 1},
            "persistent_attention_output_server must be a boolean",
        ),
        (
            {"persistent_shared_expert_server": 1},
            "persistent_shared_expert_server must be a boolean",
        ),
        (
            {"persistent_rope_split_server": 1},
            "persistent_rope_split_server must be a boolean",
        ),
        (
            {"persistent_mla_attention_server": 1},
            "persistent_mla_attention_server must be a boolean",
        ),
        (
            {"persistent_rmsnorm_server": 1},
            "persistent_rmsnorm_server must be a boolean",
        ),
        (
            {"moe_output_accumulator": "bad"},
            "moe_output_accumulator must be env, file, or memory",
        ),
        ({"max_stage_mib": 0.0}, "max_stage_mib"),
        (
            {"stage_disk_safety_margin_bytes": True},
            "stage_disk_safety_margin_bytes must be an integer",
        ),
        (
            {"stage_disk_safety_margin_bytes": 1.5},
            "stage_disk_safety_margin_bytes must be an integer",
        ),
        ({"stage_disk_safety_margin_bytes": -1}, "stage_disk_safety_margin_bytes"),
        ({"max_live_working_set_mib": -1.0}, "max_live_working_set_mib"),
        ({"min_free_unified_memory_gib": -1.0}, "min_free_unified_memory_gib"),
    ),
)
def test_run_prompt_prefill_rejects_invalid_memory_caps_before_work_dir(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "invalid_prefill_work"
    kwargs = {
        "runner_path": runner,
        "expert_layout_path": expert,
        "resident_layout_path": resident,
        "cache_layout_path": cache_layout,
        "cache_file_path": cache_file,
        "prompt_token_ids": [0, 1],
        "output_last_hidden_f32_path": tmp_path / "last_hidden.f32",
        "dense_layers": {0},
        "prompt_chunk_tokens": 2,
        "num_heads": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "kv_lora_dim": 2,
        "top_k": 2,
        "max_k": 2,
        "router_score": "raw",
        "max_slot_mib": 1,
        "max_router_mib": 1,
        "max_resident_matrix_mib": 1,
        "max_cache_file_mib": 1,
        "max_cache_write_mib": 1,
        "max_cache_read_mib": 1,
        "max_runner_scratch_mib": 64,
        "work_dir": work_dir,
        "echo_runner_output": False,
    }
    kwargs.update(overrides)

    with pytest.raises(PromptPrefillError, match=message):
        run_prompt_prefill(**kwargs)
    assert not work_dir.exists()


def test_run_prompt_prefill_rejects_expected_hidden_size_mismatch(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    with pytest.raises(
        PromptPrefillError,
        match="embedding hidden dim 8 does not match expected_hidden_size 7",
    ):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            dense_layers={0},
            prompt_chunk_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            expected_hidden_size=7,
            echo_runner_output=False,
        )


def test_run_prompt_prefill_rejects_context_overflow_before_work_dir(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "context_overflow_prefill_work"

    with pytest.raises(PromptPrefillError, match="exceed decode cache context"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            work_dir=work_dir,
            start_position=4,
            dense_layers={0},
            prompt_chunk_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not work_dir.exists()


def test_run_prompt_prefill_rejects_live_working_set_over_cap(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "live_guard_work"

    with pytest.raises(PromptPrefillError, match="live working set"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            work_dir=work_dir,
            dense_layers={0},
            prompt_chunk_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            max_live_working_set_mib=0.001,
            echo_runner_output=False,
        )

    assert not work_dir.exists()


def test_run_prompt_prefill_cleans_auto_work_dir_on_embedding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    auto_root = tmp_path / "auto_prefill_failure"

    def fake_mkdtemp(*args, **kwargs) -> str:
        del args, kwargs
        auto_root.mkdir()
        return str(auto_root)

    monkeypatch.setattr("largerlm.prompt_prefill.tempfile.mkdtemp", fake_mkdtemp)

    with pytest.raises(PromptPrefillError, match="embedding row"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            dense_layers={0},
            prompt_chunk_tokens=1,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_embedding_row_mib=1 / (1024 * 1024),
            echo_runner_output=False,
        )

    assert not auto_root.exists()


def test_run_prompt_prefill_rechecks_live_memory_between_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "live_recheck_work"
    calls = 0

    def fake_live_memory_check(**kwargs) -> LiveMemoryBudget:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise GenerationGuardError("available unified memory fell below reserve")
        return LiveMemoryBudget(
            estimated_live_working_set_bytes=kwargs["estimated_live_working_set_bytes"],
            max_live_working_set_bytes=kwargs["max_live_working_set_bytes"],
            min_available_memory_bytes=kwargs["min_available_memory_bytes"],
            system_available_bytes=128 * 1024**3,
            system_total_bytes=128 * 1024**3,
            system_source="fake",
        )

    monkeypatch.setattr(
        "largerlm.prompt_prefill.check_live_memory_budget",
        fake_live_memory_check,
    )

    with pytest.raises(PromptPrefillError, match="live memory guard failed"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            work_dir=work_dir,
            dense_layers={0},
            prompt_chunk_tokens=1,
            min_free_unified_memory_gib=1,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert calls == 5
    assert not (work_dir / "chunk_0000").exists()
    assert not (work_dir / "chunk_0001").exists()


def test_run_prompt_prefill_rechecks_live_memory_between_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "live_layer_recheck_work"
    calls = 0

    def fake_live_memory_check(**kwargs) -> LiveMemoryBudget:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise GenerationGuardError("available unified memory fell below reserve")
        return LiveMemoryBudget(
            estimated_live_working_set_bytes=kwargs["estimated_live_working_set_bytes"],
            max_live_working_set_bytes=kwargs["max_live_working_set_bytes"],
            min_available_memory_bytes=kwargs["min_available_memory_bytes"],
            system_available_bytes=128 * 1024**3,
            system_total_bytes=128 * 1024**3,
            system_source="fake",
        )

    monkeypatch.setattr(
        "largerlm.prompt_prefill.check_live_memory_budget",
        fake_live_memory_check,
    )

    with pytest.raises(PromptPrefillError, match="live memory guard failed"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            work_dir=work_dir,
            dense_layers={0},
            prompt_chunk_tokens=1,
            min_free_unified_memory_gib=1,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert calls == 3
    assert not (work_dir / "chunk_0000").exists()


def test_run_prompt_prefill_rejects_strict_static_capacity_below_chunk(
    tmp_path: Path,
) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "strict_static_capacity_work"

    with pytest.raises(PromptPrefillError) as exc_info:
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            work_dir=work_dir,
            dense_layers={0},
            prompt_chunk_tokens=2,
            static_capacity_per_expert=1,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert (
        "static_capacity_per_expert 1 cannot guarantee overflow-free strict "
        "routing for a prompt chunk of 2 tokens"
        in str(exc_info.value)
    )
    assert not work_dir.exists()


def test_run_prompt_prefill_rejects_work_disk_budget(tmp_path: Path) -> None:
    expert, resident, cache_layout, cache_file, _config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    work_dir = tmp_path / "prefill_work"

    with pytest.raises(PromptPrefillError, match="prompt prefill work files"):
        run_prompt_prefill(
            runner_path=runner,
            expert_layout_path=expert,
            resident_layout_path=resident,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            prompt_token_ids=[0, 1],
            output_last_hidden_f32_path=tmp_path / "last_hidden.f32",
            work_dir=work_dir,
            dense_layers={0},
            prompt_chunk_tokens=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            kv_lora_dim=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            stage_disk_safety_margin_bytes=10**30,
            echo_runner_output=False,
        )

    assert not (work_dir / "chunk_0000" / "embedding.f32").exists()


def test_prompt_prefill_cli_derives_config_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file, config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "cli_last_hidden.f32"

    status = cli_main(
        [
            "prefill-prompt",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--prompt-token-ids",
            "0,2,3",
            "--output-last-hidden-f32",
            str(last_hidden),
            "--prompt-chunk-tokens",
            "auto",
            "--max-prompt-batch-mib",
            "0.00006103515625",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--moe-token-block",
            "1",
            "--static-capacity-per-expert",
            "auto",
            "--max-stage-mib",
            "1",
            "--max-compact-stage-mib",
            "1",
            "--stage-disk-margin-mib",
            "0",
            "--prefill-mla-key-cache",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunk_tokens"] == 2
    assert payload["chunk_count"] == 2
    auto_plan = payload["auto_prompt_chunk_plan"]
    assert auto_plan["chunk_tokens"] == 2
    assert auto_plan["raw_tokens"] == 2
    assert {cap["name"] for cap in auto_plan["limiting_caps"]} == {
        "prompt_batch_bytes"
    }
    assert auto_plan["max_matrix_scratch_bytes"] > 0
    assert auto_plan["next_token_matrix_scratch_bytes"] is not None
    assert payload["dense_layers"] == [0]
    assert payload["layers"] == [0, 1]
    assert payload["chunks"][1]["start_position"] == 2
    assert payload["chunks"][0]["layers"][0]["attention"]["mla_key_cache"] is True
    assert payload["chunks"][0]["layers"][0]["attention"]["mla_key_cache_bytes"] == 16
    assert payload["chunks"][1]["layers"][0]["attention"]["mla_key_cache"] is True
    assert payload["chunks"][1]["layers"][0]["attention"]["mla_key_cache_bytes"] == 24
    assert payload["moe_token_block"] == 1
    assert payload["moe_token_block_mode_counts"] == {"fixed": 2}
    assert payload["max_effective_moe_token_block"] == 1
    coverage = payload["prefill_acceleration_coverage"]
    assert coverage["matrix_count"] == 32
    assert coverage["custom_metal_matrix_count"] == 32
    assert coverage["streamed_routed_expert_layer_count"] == 2
    assert coverage["streamed_routed_expert_matrix_count"] == 6
    assert coverage["streamed_routed_expert_assignments"] == 6
    assert coverage["streamed_routed_expert_estimated_flops"] == 6 * 6 * 8 * 8
    assert coverage["mpp_tensor_ops_candidate_backend_counts"] == {}
    assert coverage["any_resident_matrix_accelerated"] is False
    assert payload["prefill_acceleration_frontier"]["source"] == (
        "prompt_prefill_actual"
    )
    assert payload["max_moe_batch_buffer_bytes"] == 100
    assert payload["total_expert_stage_serial_read_bytes"] == 1152
    assert payload["total_expert_stage_unique_requested_bytes"] == 768
    assert payload["total_expert_stage_planned_read_bytes"] == 768
    assert payload["total_expert_stage_coalesced_savings_bytes"] == 384
    assert payload["total_expert_stage_unique_read_amplification"] == pytest.approx(
        1.0
    )
    assert payload["total_expert_stage_read_advice_attempted_ranges"] > 0
    assert payload["total_expert_stage_read_advice_calls"] >= 0
    assert payload["total_expert_stage_read_advice_bytes"] >= 0
    assert payload["total_expert_stage_read_advice_failures"] >= 0
    assert payload["static_capacity_per_expert"] == "auto"
    assert payload["max_static_capacity_per_expert"] == 2
    assert payload["total_static_capacity_used_slots"] == 6
    assert payload["total_static_capacity_slots"] == 6
    assert payload["total_static_capacity_binary_bytes"] > 0
    assert last_hidden.stat().st_size == 8 * 4


def test_prompt_prefill_cli_passes_auto_mpsgraph_thresholds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file, config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "cli_last_hidden_mpsgraph.f32"

    status = cli_main(
        [
            "prefill-prompt",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--prompt-token-ids",
            "0,2",
            "--output-last-hidden-f32",
            str(last_hidden),
            "--prompt-chunk-tokens",
            "2",
            "--prefill-linear-backend",
            "auto",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--max-stage-mib",
            "1",
            "--max-compact-stage-mib",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    coverage = payload["prefill_acceleration_coverage"]
    assert coverage["any_resident_matrix_accelerated"] is True
    assert coverage["mpsgraph_matrix_count"] > 0
    assert payload["linear_backend_counts"]["mpsgraph-f32"] > 0
    assert last_hidden.stat().st_size == 8 * 4


def test_prompt_prefill_cli_derives_dsa_config_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file, config = _write_fixture(
        tmp_path,
        include_dsa=True,
    )
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "cli_last_hidden_dsa.f32"

    status = cli_main(
        [
            "prefill-prompt",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--prompt-token-ids",
            "0,2,3",
            "--output-last-hidden-f32",
            str(last_hidden),
            "--work-dir",
            str(tmp_path / "cli_dsa_work"),
            "--keep-work-dir",
            "--prompt-chunk-tokens",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--max-stage-mib",
            "1",
            "--max-compact-stage-mib",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks"][0]["layers"][0]["dsa_indexer_mode"] == "full"
    assert payload["chunks"][0]["layers"][1]["dsa_indexer_mode"] == "shared"
    assert payload["chunks"][0]["layers"][0]["attention"]["mla_attention"]["indexed"] is False
    assert payload["chunks"][0]["layers"][1]["attention"]["mla_attention"]["indexed"] is False
    assert payload["chunks"][1]["layers"][0]["attention"]["mla_attention"]["indexed"] is True
    assert payload["chunks"][1]["layers"][1]["attention"]["mla_attention"]["indexed"] is True


def test_prompt_prefill_cli_auto_chunk_accounts_for_start_position_dsa_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file, config = _write_fixture(
        tmp_path,
        include_dsa=True,
    )
    runner = _write_fake_runner(tmp_path)
    last_hidden = tmp_path / "cli_last_hidden_dsa_start.f32"

    status = cli_main(
        [
            "prefill-prompt",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--prompt-token-ids",
            "0,2,3",
            "--output-last-hidden-f32",
            str(last_hidden),
            "--start-position",
            "2",
            "--prompt-chunk-tokens",
            "auto",
            "--max-prompt-batch-mib",
            "1",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-write-mib",
            "1",
            "--max-cache-read-mib",
            str(24 / (1024 * 1024)),
            "--max-runner-scratch-mib",
            "64",
            "--max-stage-mib",
            "1",
            "--max-compact-stage-mib",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunk_tokens"] == 1
    assert payload["chunk_count"] == 3
    assert [chunk["start_position"] for chunk in payload["chunks"]] == [2, 3, 4]


def test_generate_token_ids_cli_can_batch_prefill_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert, resident, cache_layout, cache_file, config = _write_fixture(tmp_path)
    runner = _write_fake_runner(tmp_path)

    status = cli_main(
        [
            "generate-token-ids",
            str(expert),
            str(resident),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--prompt-token-ids",
            "0,2,3",
            "--max-new-tokens",
            "1",
            "--logits-top-k",
            "2",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--batch-prefill-prompt",
            "--prefill-prompt-chunk-tokens",
            "auto",
            "--prefill-max-prompt-batch-mib",
            "0.00006103515625",
            "--prefill-max-cache-write-mib",
            "1",
            "--prefill-max-stage-mib",
            "1",
            "--prefill-max-compact-stage-mib",
            "1",
            "--prefill-moe-token-block",
            "1",
            "--prefill-static-capacity-per-expert",
            "auto",
            "--prefill-stage-disk-margin-mib",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["generated_token_ids"] == [4]
    auto_plan = payload["auto_prefill_prompt_chunk_plan"]
    assert auto_plan["chunk_tokens"] == 2
    assert auto_plan["raw_tokens"] == 2
    assert {cap["name"] for cap in auto_plan["limiting_caps"]} == {
        "prompt_batch_bytes"
    }
    assert auto_plan["max_matrix_scratch_bytes"] == 2 * 1024 * 1024
    assert auto_plan["next_token_matrix_scratch_bytes"] == 2 * 1024 * 1024
    max_safe_plan = payload["max_safe_prefill_prompt_chunk_plan"]
    assert max_safe_plan["chunk_tokens"] == 2
    assert max_safe_plan["raw_tokens"] == 2
    assert {cap["name"] for cap in max_safe_plan["limiting_caps"]} == {
        "prompt_batch_bytes"
    }
    assert payload["prompt_prefill"]["chunk_count"] == 2
    assert payload["prompt_prefill"]["linear_backend_counts"] == {"custom-metal": 32}
    assert set(payload["prompt_prefill"]["linear_backend_elapsed_seconds"]) == {
        "custom-metal"
    }
    payload_custom_elapsed = payload["prompt_prefill"][
        "linear_backend_elapsed_seconds"
    ]["custom-metal"]
    assert payload_custom_elapsed > 0.0
    assert set(payload["prompt_prefill"]["linear_backend_estimated_tflops"]) == {
        "custom-metal"
    }
    assert payload["prompt_prefill"]["linear_backend_estimated_tflops"][
        "custom-metal"
    ] == pytest.approx(
        payload["prompt_prefill"]["linear_backend_flops"]["custom-metal"]
        / payload_custom_elapsed
        / 1e12
    )
    payload_components = payload["prompt_prefill"]["linear_backend_component_stats"]
    assert payload_components["attention.o_proj"]["linear_backend_counts"][
        "custom-metal"
    ] > 0
    assert payload_components["moe.routed_experts_streamed"][
        "linear_backend_counts"
    ]["custom-metal"] == 6
    coverage = payload["prompt_prefill"]["prefill_acceleration_coverage"]
    assert coverage["matrix_count"] == 32
    assert coverage["accelerated_matrix_count"] == 0
    assert coverage["custom_metal_matrix_count"] == 32
    assert coverage["streamed_routed_expert_layer_count"] == 2
    assert coverage["streamed_routed_expert_matrix_count"] == 6
    assert coverage["streamed_routed_expert_assignments"] == 6
    assert coverage["streamed_routed_expert_estimated_flops"] == 6 * 6 * 8 * 8
    assert coverage["any_resident_matrix_accelerated"] is False
    accel_frontier = payload["prompt_prefill"]["prefill_acceleration_frontier"]
    assert accel_frontier["source"] == "prompt_prefill_actual"
    assert accel_frontier["resolved_prompt_chunk_tokens"] == 2
    assert any(item["is_resolved"] for item in accel_frontier["candidates"])
    assert (
        payload["prompt_prefill"]["total_linear_matrix_scratch_bytes"]
        == 26 * 2 * 1024 * 1024
    )
    assert payload["prompt_prefill"]["max_linear_matrix_scratch_bytes"] == 2 * 1024 * 1024
    assert payload["prompt_prefill"]["total_expert_stage_planned_read_bytes"] == 768
    assert (
        payload["prompt_prefill"]["total_expert_stage_unique_requested_bytes"] == 768
    )
    assert payload["prompt_prefill"]["total_expert_stage_waste_bytes"] == 0
    assert (
        payload["prompt_prefill"]["total_expert_stage_unique_read_amplification"]
        == 1.0
    )
    assert (
        payload["prompt_prefill"]["total_expert_stage_read_advice_attempted_ranges"]
        > 0
    )
    assert payload["prompt_prefill"]["total_expert_stage_read_advice_calls"] >= 0
    assert payload["prompt_prefill"]["total_expert_stage_read_advice_bytes"] >= 0
    assert payload["prompt_prefill"]["total_expert_stage_read_advice_failures"] >= 0
    assert payload["prompt_prefill"]["total_routed_expert_assignments"] == 6
    assert payload["prompt_prefill"]["max_routed_tokens_per_expert"] == 2
    assert payload["prompt_prefill"]["moe_token_block"] == 1
    assert payload["prompt_prefill"]["moe_token_block_mode_counts"] == {"fixed": 2}
    assert payload["prompt_prefill"]["max_effective_moe_token_block"] == 1
    assert payload["prompt_prefill"]["static_capacity_per_expert"] == "auto"
    assert payload["prompt_prefill"]["max_static_capacity_per_expert"] == 2
    assert payload["prompt_prefill"]["total_static_capacity_binary_bytes"] > 0
