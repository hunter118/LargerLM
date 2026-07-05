from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import largerlm.decode_driver as decode_driver_module
from largerlm.cli import main as cli_main
from largerlm.decode_driver import DecodeDriverError, layers_from_expert_layout, run_decode_layers


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


def _write_layouts(root: Path, *, include_dsa: bool = False) -> tuple[Path, Path, Path, Path]:
    experts = root / "experts"
    resident = root / "resident"
    experts.mkdir()
    resident.mkdir()
    layers = []
    for layer in (1, 2):
        layers.append(
            {
                "layer": layer,
                "num_experts": 2,
                "expert_slot_bytes": 192,
                "layer_file": f"layer_{layer:03d}.bin",
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
        )
    expert_layout = experts / "layout.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "quantization": "mlx-affine-int4",
                "group_size": 8,
                "num_layers": 3,
                "num_experts": 2,
                "component_order": [name for name, *_ in COMPONENTS],
                "layers": layers,
            }
        ),
        encoding="utf-8",
    )

    tensors = []
    offset = 0
    attention_specs = [
        ("input_layernorm.weight", "F32", [8], "norms", 32),
        ("self_attn.q_a_layernorm.weight", "F32", [2], "norms", 8),
        ("self_attn.kv_a_layernorm.weight", "F32", [2], "norms", 8),
        ("post_attention_layernorm.weight", "F32", [8], "norms", 32),
        ("self_attn.q_a_proj.weight", "F32", [2, 8], "attention", 64),
        ("self_attn.q_b_proj.weight", "F32", [6, 2], "attention", 48),
        ("self_attn.kv_a_proj_with_mqa.weight", "F32", [4, 8], "attention", 128),
        ("self_attn.kv_b_proj.weight", "F32", [4, 2], "attention", 32),
        ("self_attn.o_proj.weight", "F32", [8, 2], "attention", 64),
    ]
    for layer in (0, 1, 2):
        for suffix, dtype, shape, category, size in attention_specs:
            tensors.append(
                {
                    "name": f"model.layers.{layer}.{suffix}",
                    "offset": offset,
                    "size": size,
                    "dtype": dtype,
                    "shape": shape,
                    "category": category,
                }
            )
            offset += size
        if include_dsa and layer in {1, 2}:
            for suffix, shape, size in (
                ("self_attn.indexer.wk.weight", [2, 8], 64),
                ("self_attn.indexer.k_norm.weight", [2], 8),
                ("self_attn.indexer.k_norm.bias", [2], 8),
                ("self_attn.indexer.wq_b.weight", [2, 2], 16),
                ("self_attn.indexer.weights_proj.weight", [1, 8], 32),
            ):
                tensors.append(
                    {
                        "name": f"model.layers.{layer}.{suffix}",
                        "offset": offset,
                        "size": size,
                        "dtype": "F32",
                        "shape": shape,
                        "category": "dsa_indexer",
                    }
                )
                offset += size
        if layer == 0:
            for component in ("gate_proj", "up_proj", "down_proj"):
                tensors.append(
                    {
                        "name": f"model.layers.0.mlp.{component}.weight",
                        "offset": offset,
                        "size": 256,
                        "dtype": "F32",
                        "shape": [8, 8],
                        "category": "dense_mlp",
                    }
                )
                offset += 256
        else:
            tensors.append(
                {
                    "name": f"model.layers.{layer}.mlp.gate.weight",
                    "offset": offset,
                    "size": 64,
                    "dtype": "F32",
                    "shape": [2, 8],
                    "category": "routers",
                }
            )
            offset += 64
    resident_layout = resident / "layout.json"
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": offset,
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    (resident / "resident.bin").write_bytes(b"\0" * offset)

    cache_segments = [
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
        {
            "kind": "mla_kv",
            "layer": 2,
            "offset": 128,
            "width": 4,
            "dtype": "BF16",
            "dtype_bytes": 2,
            "token_stride_bytes": 8,
            "max_context_tokens": 2,
            "total_bytes": 16,
        },
    ]
    cache_total_bytes = 144
    if include_dsa:
        cache_segments.extend(
            [
                {
                    "kind": "dsa_index",
                    "layer": 1,
                    "offset": 144,
                    "width": 2,
                    "dtype": "BF16",
                    "dtype_bytes": 2,
                    "token_stride_bytes": 4,
                    "max_context_tokens": 2,
                    "total_bytes": 8,
                },
                {
                    "kind": "dsa_index",
                    "layer": 2,
                    "offset": 152,
                    "width": 2,
                    "dtype": "BF16",
                    "dtype_bytes": 2,
                    "token_stride_bytes": 4,
                    "max_context_tokens": 2,
                    "total_bytes": 8,
                },
            ]
        )
        cache_total_bytes = 160

    cache_layout = root / "cache_layout.json"
    cache_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 2,
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
    return expert_layout, resident_layout, cache_layout, cache_file


def test_layers_from_expert_layout_rejects_boolean_layer_id(tmp_path: Path) -> None:
    expert_layout, _resident_layout, _cache_layout, _cache_file = _write_layouts(tmp_path)
    payload = json.loads(expert_layout.read_text(encoding="utf-8"))
    payload["layers"] = [{"layer": True}]
    expert_layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DecodeDriverError, match="no runnable layers"):
        layers_from_expert_layout(expert_layout)


def _write_fake_runner(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import struct
import sys
from pathlib import Path

def arg(name):
    i = sys.argv.index(name)
    return sys.argv[i + 1]

def write(path, values):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(struct.pack(f"<{len(values)}f", *values))
    out_path.with_suffix(out_path.suffix + ".argv.json").write_text(json.dumps(sys.argv[1:]))

if "--run-decoder-layers" in sys.argv:
    src = Path(arg("--input-f32")).read_bytes()
    values = list(struct.unpack(f"<{len(src) // 4}f", src))
    layers = [int(part) for part in arg("--layers").split(",") if part]
    output_f32 = Path(arg("--output-f32"))
    fused_logits = "--output-topk-json" in sys.argv
    work_dir = Path(arg("--work-dir")) if "--work-dir" in sys.argv else output_f32.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    records = []
    emit_mla_diagnostics = os.environ.get("LARGERLM_FAKE_RUNNER_MLA_DIAGNOSTICS") == "1"
    current_input = Path(arg("--input-f32"))
    for index, layer_id in enumerate(layers):
        out_path = output_f32 if index + 1 == len(layers) else work_dir / f"layer_{layer_id:04d}.f32"
        values = [value + float(layer_id) for value in values]
        if not (fused_logits and index + 1 == len(layers)):
            write(out_path, values)
        record = {
            "layer": layer_id,
            "kind": "dense" if str(layer_id) in (arg("--dense-layers").split(",") if "--dense-layers" in sys.argv else []) else "moe",
            "input_path": str(current_input),
            "output_path": str(out_path),
            "input_in_memory": index > 0,
            "output_in_memory": index + 1 < len(layers) or fused_logits,
            "elapsed_seconds": 0.01,
        }
        if emit_mla_diagnostics:
            key_cache_enabled = os.environ.get("LARGERLM_MLA_KEY_CACHE") == "1"
            key_cache_bytes = 8 if key_cache_enabled else 0
            record.update({
                "attention_projections_elapsed_seconds": 0.001,
                "mla_attention_elapsed_seconds": 0.007,
                "attention_output_elapsed_seconds": 0.002,
                "mlp_elapsed_seconds": 0.003,
                "mlp_timing_rmsnorm_elapsed_seconds": 0.0007,
                "mlp_timing_router_elapsed_seconds": 0.0008,
                "mlp_timing_moe_elapsed_seconds": 0.0019,
                "mlp_timing_moe_setup_elapsed_seconds": 0.0001,
                "mlp_timing_moe_clear_elapsed_seconds": 0.0002,
                "mlp_timing_expert_read_elapsed_seconds": 0.0003,
                "mlp_timing_expert_kernel_elapsed_seconds": 0.0009,
                "mlp_timing_shared_elapsed_seconds": 0.0004,
                "mlp_timing_residual_elapsed_seconds": 0.0002,
                "mlp_timing_output_elapsed_seconds": 0.0001,
                "mlp_timing_total_elapsed_seconds": 0.003,
                "mlp_preload_selected_enabled": True,
                "mlp_preload_selected_bytes": 8192,
                "mlp_mxfp4_fused_decode_enabled": True,
                "mla_timing_input_elapsed_seconds": 0.0001,
                "mla_timing_cache_read_elapsed_seconds": 0.0002,
                "mla_timing_value_read_elapsed_seconds": 0.0003,
                "mla_timing_metal_setup_elapsed_seconds": 0.0004,
                "mla_timing_kernel_elapsed_seconds": 0.005,
                "mla_timing_kernel_weights_elapsed_seconds": 0.003,
                "mla_timing_kernel_values_elapsed_seconds": 0.002,
                "mla_timing_write_elapsed_seconds": 0.0006,
                "mla_timing_total_elapsed_seconds": 0.0066,
                "mla_weights_bytes": 64,
                "mla_key_cache_bytes": key_cache_bytes,
                "mla_rope_cache_bytes": 0,
                "mla_value_cache_bytes": 0,
                "mla_estimated_peak_bytes": 4096,
                "mla_singleton_kernel": False,
                "mla_split_kernel_timing_enabled": True,
                "mla_key_cache_enabled": key_cache_enabled,
                "mla_rope_cache_enabled": False,
                "mla_value_cache_enabled": False,
            })
        records.append(record)
        current_input = out_path
    if fused_logits:
        Path(arg("--output-topk-json")).write_text(json.dumps({
            "topk": [{"token_id": 2, "logit": 3.0}],
            "elapsed_seconds": 0.02,
        }))
    if "--output-report-json" in sys.argv:
        Path(arg("--output-report-json")).write_text(json.dumps({"elapsed_seconds": 0.01 * len(layers), "records": records}))
    raise SystemExit(0)

layer = int(arg("--layer")) if "--layer" in sys.argv else 0
if "--run-rmsnorm-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    suffix = arg("--norm-suffix")
    dim = 2 if "q_a_layernorm" in suffix or "kv_a_layernorm" in suffix else 8
    write(arg("--output-f32"), [0.0] * (batch * dim))
elif "--run-resident-linear-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    suffix = arg("--tensor-suffix")
    if suffix.endswith("q_a_proj.weight"):
        out_dim = 2
    elif suffix.endswith("q_b_proj.weight"):
        out_dim = 6
    elif suffix.endswith("kv_a_proj_with_mqa.weight"):
        out_dim = 4
    elif suffix.endswith("kv_b_proj.weight"):
        out_dim = 4
    elif suffix.endswith("o_proj.weight"):
        out_dim = 8
    elif suffix.endswith("mlp.gate_proj.weight"):
        out_dim = 8
    elif suffix.endswith("mlp.up_proj.weight"):
        out_dim = 8
    elif suffix.endswith("mlp.down_proj.weight"):
        out_dim = 8
    else:
        raise SystemExit(f"unknown tensor suffix {suffix}")
    write(arg("--output-f32"), [0.0] * (batch * out_dim))
elif "--run-rope-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    heads = int(arg("--num-heads"))
    rope = int(arg("--rope-dim"))
    write(arg("--output-q-f32"), [0.0] * (batch * heads * rope))
    write(arg("--output-k-f32"), [0.0] * (batch * rope))
elif "--run-mla-attention-indexed-batch" in sys.argv or "--run-mla-attention-batch" in sys.argv:
    batch = int(arg("--batch-tokens"))
    heads = int(arg("--num-heads"))
    v_head = int(arg("--v-head-dim"))
    write(arg("--output-f32"), [0.0] * (batch * heads * v_head))
else:
    src = Path(arg("--input-f32")).read_bytes()
    values = struct.unpack(f"<{len(src) // 4}f", src)
    out = [value + float(layer) for value in values]
    write(arg("--output-f32"), out)
""",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


def _write_failing_runner(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stderr.write('runner exploded\\n')\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


def _write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def _read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def test_decode_layers_runs_selected_layers_in_order(tmp_path: Path) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    mla_cache_dir = tmp_path / "mla-kv-b-cache"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={1, 2},
        position=0,
        context_length=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        mla_kv_b_cache_dir=mla_cache_dir,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        expert_read_advise_merge_gap_kib=128,
        expert_read_advise_align_kib=4,
        echo_runner_output=False,
    )

    assert result.layers == (1, 2)
    assert [record.layer for record in result.records] == [1, 2]
    assert _read_f32(output_f32, 8) == (4.0,) * 8
    assert result.records[1].input_path == result.records[0].output_path
    assert "--run-decoder-layer" in result.records[0].command
    assert result.records[0].expert_read_bytes == 2 * 192
    assert result.records[0].attention_read_bytes > 0
    assert result.records[0].cache_read_bytes > 0
    assert result.records[0].estimated_peak_bytes > 0
    assert result.records[0].moe_stage_peak_bytes > 0
    assert result.records[0].elapsed_seconds >= 0.0
    assert result.records[0].attention_elapsed_seconds == 0.0
    assert result.records[0].mlp_elapsed_seconds == 0.0
    assert result.records[0].attention_stage_elapsed_seconds == {}
    command = result.records[0].command
    merge_index = command.index("--expert-read-advise-merge-gap-kib")
    align_index = command.index("--expert-read-advise-align-kib")
    assert command[merge_index + 1] == "128"
    assert command[align_index + 1] == "4"
    assert command[command.index("--mla-kv-b-cache-dir") + 1] == str(mla_cache_dir)


def test_decode_layers_parses_mla_attention_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "largerlm-runner"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)
    monkeypatch.setenv("LARGERLM_FAKE_RUNNER_MLA_DIAGNOSTICS", "1")

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={1, 2},
        position=0,
        context_length=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        write_dsa_future_cache=False,
        echo_runner_output=False,
    )

    record = result.records[0]
    assert record.attention_stage_elapsed_seconds == {
        "projections": 0.001,
        "split_q": 0.0,
        "dummy_k": 0.0,
        "rope": 0.0,
        "mla_attention": 0.007,
        "attention_output": 0.002,
    }
    assert record.mlp_stage_elapsed_seconds == {
        "rmsnorm": 0.0007,
        "router": 0.0008,
        "moe": 0.0019,
        "moe_setup": 0.0001,
        "moe_clear": 0.0002,
        "expert_read": 0.0003,
        "expert_kernel": 0.0009,
        "shared": 0.0004,
        "residual": 0.0002,
        "output": 0.0001,
        "total": 0.003,
    }
    assert record.mlp_diagnostics == {
        "preload_selected_enabled": True,
        "preload_selected_bytes": 8192,
        "mxfp4_fused_decode_enabled": True,
    }
    assert record.attention_elapsed_seconds == pytest.approx(0.010)
    assert record.mla_attention_timing_elapsed_seconds == {
        "input": 0.0001,
        "cache_read": 0.0002,
        "value_read": 0.0003,
        "metal_setup": 0.0004,
        "kernel": 0.005,
        "kernel_weights": 0.003,
        "kernel_values": 0.002,
        "write": 0.0006,
        "total": 0.0066,
    }
    assert record.mla_attention_diagnostics == {
        "weights_bytes": 64,
        "key_cache_bytes": 0,
        "rope_cache_bytes": 0,
        "value_cache_bytes": 0,
        "estimated_peak_bytes": 4096,
        "singleton_kernel": False,
        "split_kernel_timing_enabled": True,
        "key_cache_enabled": False,
        "rope_cache_enabled": False,
        "value_cache_enabled": False,
    }


def test_decode_layers_passes_mla_key_cache_to_attention_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "largerlm-runner"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)
    monkeypatch.delenv("LARGERLM_MLA_KEY_CACHE", raising=False)
    monkeypatch.setenv("LARGERLM_FAKE_RUNNER_MLA_DIAGNOSTICS", "1")

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={1},
        position=0,
        context_length=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        mla_key_cache=True,
        write_dsa_future_cache=False,
        echo_runner_output=False,
    )

    diagnostics = result.records[0].mla_attention_diagnostics
    assert diagnostics["key_cache_enabled"] is True
    assert diagnostics["key_cache_bytes"] == 8


def test_decode_layers_uses_full_and_shared_dsa_indices(tmp_path: Path) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    mla_cache_dir = tmp_path / "mla-kv-b-cache"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={1, 2},
        work_dir=tmp_path / "decode_dsa_work",
        keep_work_dir=True,
        position=1,
        context_length=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        mla_kv_b_cache_dir=mla_cache_dir,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["none", "full", "shared"],
        dsa_index_topk=1,
        dsa_index_n_heads=1,
        dsa_qk_rope_dim=2,
        dsa_rope_interleave=True,
        echo_runner_output=False,
    )

    assert _read_f32(output_f32, 8) == (4.0,) * 8
    assert result.budgets[0].dsa_index_head_dim == 2
    assert [record.composed for record in result.records] == [True, True]
    assert [record.dsa_indexer_mode for record in result.records] == ["full", "shared"]
    assert all(record.attention_elapsed_seconds >= 0.0 for record in result.records)
    assert all(record.mlp_elapsed_seconds >= 0.0 for record in result.records)
    assert set(result.records[0].attention_stage_elapsed_seconds) == {
        "projections",
        "cache_write",
        "rope",
        "dsa_indexer",
        "mla_attention",
        "attention_output",
    }
    assert result.records[0].attention_stage_elapsed_seconds["dsa_indexer"] >= 0.0
    assert result.records[1].attention_stage_elapsed_seconds["dsa_indexer"] == 0.0
    assert [record.dsa_rope_interleave for record in result.records] == [True, True]
    assert result.records[0].dsa_indices_u32_path is not None
    assert result.records[0].dsa_indices_u32_path.stat().st_size == 8
    assert result.records[1].dsa_indices_u32_path == result.records[0].dsa_indices_u32_path
    assert "--run-mla-attention-indexed-batch" in result.records[0].command
    assert "--run-mla-attention-indexed-batch" in result.records[1].command


def test_decode_layers_rejects_selected_shared_dsa_without_selected_full(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)

    with pytest.raises(DecodeDriverError, match="selected DSA layer 2 is shared"):
        run_decode_layers(
            runner_path=runner,
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            input_path=input_f32,
            output_path=output_f32,
            layers={2},
            position=1,
            context_length=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_write_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            dsa_indexer_types=["none", "full", "shared"],
            dsa_index_topk=2,
            dsa_index_n_heads=1,
            dsa_qk_rope_dim=2,
            echo_runner_output=False,
        )


def test_decode_layers_skips_dsa_composition_for_final_singleton_context(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={2},
        position=0,
        context_length=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["none", "full", "shared"],
        write_dsa_future_cache=False,
        echo_runner_output=False,
    )

    assert _read_f32(output_f32, 8) == (3.0,) * 8
    assert [record.composed for record in result.records] == [False]
    assert [record.dsa_indexer_mode for record in result.records] == ["none"]
    assert result.records[0].dsa_indices_u32_path is None
    assert result.budgets[0].dsa_indexer_mode == "none"
    assert "--run-decoder-layer" in result.records[0].command
    assert "--run-mla-attention-indexed-batch" not in result.records[0].command


def test_decode_layers_batches_final_singleton_largerlm_runner(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "largerlm-runner"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    mla_cache_dir = tmp_path / "mla-kv-b-cache"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={0, 1, 2},
        dense_layers={0},
        position=0,
        context_length=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        mla_kv_b_cache_dir=mla_cache_dir,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["full", "full", "shared"],
        write_dsa_future_cache=False,
        echo_runner_output=False,
    )

    assert _read_f32(output_f32, 8) == (4.0,) * 8
    assert result.layers == (0, 1, 2)
    assert [record.kind for record in result.records] == ["dense", "moe", "moe"]
    assert [record.composed for record in result.records] == [False, False, False]
    assert [record.input_in_memory for record in result.records] == [False, True, True]
    assert [record.output_in_memory for record in result.records] == [True, True, False]
    assert [record.dsa_indexer_mode for record in result.records] == [
        "none",
        "none",
        "none",
    ]
    assert [record.dsa_index_cache_read_bytes for record in result.records] == [0, 0, 0]
    assert all("--run-decoder-layers" in record.command for record in result.records)
    assert "--dense-layers" in result.records[0].command
    assert "--quiet-inner-layers" in result.records[0].command
    assert result.records[0].command[
        result.records[0].command.index("--mla-kv-b-cache-dir") + 1
    ] == str(mla_cache_dir)
    assert "--run-decoder-layer" not in result.records[0].command


def test_decode_layers_batches_final_logits_without_hidden_file(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "largerlm-runner"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    topk_json = tmp_path / "topk.json"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={0, 1, 2},
        dense_layers={0},
        position=0,
        context_length=1,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["full", "full", "shared"],
        write_dsa_future_cache=False,
        final_logits_topk_path=topk_json,
        final_logits_top_k=1,
        final_logits_chunk_rows=2,
        final_logits_max_chunk_mib=1,
        final_logits_rms_norm_eps=1e-6,
        echo_runner_output=False,
    )

    assert not output_f32.exists()
    assert json.loads(topk_json.read_text(encoding="utf-8"))["topk"] == [
        {"token_id": 2, "logit": 3.0}
    ]
    assert [record.output_in_memory for record in result.records] == [True, True, True]
    assert result.records[-1].input_in_memory is True
    command = result.records[0].command
    assert "--output-topk-json" in command
    assert command[command.index("--final-logits-top-k") + 1] == "1"
    assert command[command.index("--final-logits-chunk-rows") + 1] == "2"
    assert command[command.index("--final-logits-rms-norm-eps") + 1] == "1e-06"


def test_decode_layers_batches_short_context_that_fits_dsa_topk(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "largerlm-runner"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        layers={0, 1, 2},
        dense_layers={0},
        position=1,
        context_length=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_write_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        dsa_indexer_types=["full", "full", "shared"],
        dsa_index_topk=4,
        dsa_index_n_heads=1,
        write_dsa_future_cache=False,
        echo_runner_output=False,
    )

    assert _read_f32(output_f32, 8) == (4.0,) * 8
    assert [record.composed for record in result.records] == [False, False, False]
    assert [record.input_in_memory for record in result.records] == [False, True, True]
    assert [record.output_in_memory for record in result.records] == [True, True, False]
    assert [record.dsa_indexer_mode for record in result.records] == [
        "none",
        "none",
        "none",
    ]
    assert [record.dsa_index_cache_read_bytes for record in result.records] == [0, 0, 0]
    assert [budget.dsa_indexer_mode for budget in result.budgets] == [
        "none",
        "none",
        "none",
    ]
    assert all("--run-decoder-layers" in record.command for record in result.records)
    assert all("--run-mla-attention-indexed-batch" not in record.command for record in result.records)


def test_decode_layers_runs_dense_and_moe_layers_in_order(tmp_path: Path) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)

    result = run_decode_layers(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout,
        cache_file_path=cache_file,
        input_path=input_f32,
        output_path=output_f32,
        dense_layers={0},
        position=1,
        context_length=2,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=2,
        max_k=2,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.layers == (0, 1, 2)
    assert result.dense_layers == (0,)
    assert [record.kind for record in result.records] == ["dense", "moe", "moe"]
    assert "--run-dense-decoder-layer" in result.records[0].command
    assert "--run-decoder-layer" in result.records[1].command
    assert _read_f32(output_f32, 8) == (4.0,) * 8


def test_decode_layers_rejects_position_outside_context(tmp_path: Path) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    _write_f32(input_f32, [1.0] * 8)

    with pytest.raises(DecodeDriverError, match="position"):
        run_decode_layers(
            runner_path=runner,
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            input_path=input_f32,
            output_path=tmp_path / "output.f32",
            layers={1},
            position=2,
            context_length=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            echo_runner_output=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"position": True}, "position must be an integer"),
        ({"position": 1.5}, "position must be an integer"),
        ({"position": -1}, "position must be non-negative"),
        ({"context_length": False}, "context_length must be an integer"),
        ({"context_length": 1.5}, "context_length must be an integer"),
        ({"context_length": 0}, "context_length must be positive"),
        ({"num_heads": True}, "num_heads must be an integer"),
        ({"qk_nope_dim": 1.5}, "qk_nope_dim must be an integer"),
        ({"rope_dim": False}, "rope_dim must be an integer"),
        ({"v_head_dim": 1.5}, "v_head_dim must be an integer"),
        ({"kv_lora_dim": True}, "kv_lora_dim must be an integer"),
        ({"mla_kv_b_cache_dir": True}, "mla_kv_b_cache_dir must be a path"),
        ({"cache_position_offset": 1.5}, "cache_position_offset must be an integer"),
        ({"top_k": False}, "top_k must be an integer"),
        ({"max_k": 1.5}, "max_k must be an integer"),
        ({"router_n_group": True}, "router_n_group must be an integer"),
        ({"router_topk_group": 1.5}, "router_topk_group must be an integer"),
        ({"cache_dtype_bytes": False}, "cache_dtype_bytes must be an integer"),
        ({"dsa_index_topk": True}, "dsa_index_topk must be an integer"),
        ({"dsa_index_n_heads": 1.5}, "dsa_index_n_heads must be an integer"),
        ({"dsa_index_head_dim": False}, "dsa_index_head_dim must be an integer"),
        ({"dsa_qk_rope_dim": 1.5}, "dsa_qk_rope_dim must be an integer"),
        ({"layers": {True}}, "layers must be an integer"),
        ({"dense_layers": {False}}, "layers must be an integer"),
        ({"attention_scale": False}, "attention_scale must be numeric"),
        ({"rope_theta": False}, "rope_theta must be numeric"),
        ({"rms_norm_eps": False}, "rms_norm_eps must be numeric"),
        ({"dsa_layer_norm_eps": False}, "dsa_layer_norm_eps must be numeric"),
        ({"routed_scaling_factor": False}, "routed_scaling_factor must be numeric"),
        ({"max_slot_mib": 0.0}, "max_slot_mib"),
        ({"max_router_mib": 0.0}, "max_router_mib"),
        ({"max_resident_matrix_mib": 0.0}, "max_resident_matrix_mib"),
        ({"max_cache_file_mib": 0.0}, "max_cache_file_mib"),
        ({"max_cache_write_mib": 0.0}, "max_cache_write_mib"),
        ({"max_cache_read_mib": 0.0}, "max_cache_read_mib"),
        ({"max_runner_scratch_mib": float("nan")}, "max_runner_scratch_mib"),
        (
            {"expert_read_advise_merge_gap_kib": 1.5},
            "expert_read_advise_merge_gap_kib must be an integer",
        ),
        (
            {"expert_read_advise_align_kib": True},
            "expert_read_advise_align_kib must be an integer",
        ),
        ({"expert_read_advise_merge_gap_kib": -1}, "expert_read_advise_merge_gap_kib"),
        ({"expert_read_advise_align_kib": -1}, "expert_read_advise_align_kib"),
    ),
)
def test_decode_layers_rejects_invalid_memory_caps_before_work_dir(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    work_dir = tmp_path / "invalid_decode_work"
    _write_f32(input_f32, [1.0] * 8)
    kwargs = {
        "runner_path": runner,
        "expert_layout_path": expert_layout,
        "resident_layout_path": resident_layout,
        "cache_layout_path": cache_layout,
        "cache_file_path": cache_file,
        "input_path": input_f32,
        "output_path": tmp_path / "output.f32",
        "layers": {1},
        "work_dir": work_dir,
        "position": 1,
        "context_length": 2,
        "num_heads": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "top_k": 2,
        "max_k": 2,
        "max_slot_mib": 1,
        "max_router_mib": 1,
        "max_resident_matrix_mib": 1,
        "max_cache_file_mib": 1,
        "max_cache_write_mib": 1,
        "max_cache_read_mib": 1,
        "max_runner_scratch_mib": 64,
        "echo_runner_output": False,
    }
    kwargs.update(overrides)

    with pytest.raises(DecodeDriverError, match=message):
        run_decode_layers(**kwargs)
    assert not work_dir.exists()


def test_decode_layers_cleans_auto_work_dir_on_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "failing_runner.py"
    _write_failing_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    auto_root = tmp_path / "auto_decode_failure"
    _write_f32(input_f32, [1.0] * 8)

    def fake_mkdtemp(*args, **kwargs) -> str:
        del args, kwargs
        auto_root.mkdir()
        return str(auto_root)

    monkeypatch.setattr(decode_driver_module.tempfile, "mkdtemp", fake_mkdtemp)

    with pytest.raises(DecodeDriverError, match="decoder layer command failed"):
        run_decode_layers(
            runner_path=runner,
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            input_path=input_f32,
            output_path=output_f32,
            layers={1, 2},
            position=1,
            context_length=2,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            top_k=2,
            max_k=2,
            max_slot_mib=1,
            max_router_mib=1,
            max_resident_matrix_mib=1,
            max_cache_file_mib=1,
            max_cache_read_mib=1,
            max_runner_scratch_mib=64,
            echo_runner_output=False,
        )

    assert not auto_root.exists()


def test_decode_layers_cli_derives_dimensions_from_model_config(tmp_path: Path) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "num_hidden_layers": 3,
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
                "rms_norm_eps": 1e-5,
                "scoring_func": "raw",
                "rope_parameters": {"rope_theta": 10000.0},
                "rope_interleave": True,
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "decode-layers",
            str(expert_layout),
            str(resident_layout),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--layers",
            "1-2",
            "--input-f32",
            str(input_f32),
            "--output-f32",
            str(output_f32),
            "--position",
            "1",
            "--context-length",
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
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert _read_f32(output_f32, 8) == (4.0,) * 8
    argv = json.loads(output_f32.with_suffix(output_f32.suffix + ".argv.json").read_text())
    assert "--rope-interleave" in argv


def test_decode_layers_cli_derives_dsa_config_defaults(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(
        tmp_path,
        include_dsa=True,
    )
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "num_hidden_layers": 3,
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
                "rms_norm_eps": 1e-5,
                "scoring_func": "raw",
                "rope_parameters": {"rope_theta": 10000.0},
                "indexer_types": ["none", "full", "shared"],
                "index_topk": 1,
                "index_n_heads": 1,
                "index_head_dim": 2,
                "indexer_rope_interleave": True,
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "decode-layers",
            str(expert_layout),
            str(resident_layout),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--layers",
            "1-2",
            "--input-f32",
            str(input_f32),
            "--output-f32",
            str(output_f32),
            "--position",
            "1",
            "--context-length",
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
            "--work-dir",
            str(tmp_path / "cli_decode_dsa_work"),
            "--keep-work-dir",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert [record["dsa_indexer_mode"] for record in payload["records"]] == [
        "full",
        "shared",
    ]
    assert [record["dsa_rope_interleave"] for record in payload["records"]] == [
        True,
        True,
    ]
    assert all(record["composed"] for record in payload["records"])
    assert "--run-mla-attention-indexed-batch" in payload["records"][0]["command"]
    assert "--run-mla-attention-indexed-batch" in payload["records"][1]["command"]


def test_decode_layers_cli_derives_dense_layers_from_model_config(tmp_path: Path) -> None:
    expert_layout, resident_layout, cache_layout, cache_file = _write_layouts(tmp_path)
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    input_f32 = tmp_path / "input.f32"
    output_f32 = tmp_path / "output.f32"
    _write_f32(input_f32, [1.0] * 8)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "num_hidden_layers": 3,
                "intermediate_size": 8,
                "moe_intermediate_size": 8,
                "n_routed_experts": 2,
                "n_shared_experts": 0,
                "num_experts_per_tok": 2,
                "num_attention_heads": 2,
                "kv_lora_rank": 2,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
                "rms_norm_eps": 1e-5,
                "scoring_func": "raw",
                "rope_parameters": {"rope_theta": 10000.0},
                "mlp_layer_types": ["dense", "sparse", "sparse"],
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "decode-layers",
            str(expert_layout),
            str(resident_layout),
            str(cache_layout),
            str(cache_file),
            "--model-config",
            str(config),
            "--runner",
            str(runner),
            "--input-f32",
            str(input_f32),
            "--output-f32",
            str(output_f32),
            "--position",
            "1",
            "--context-length",
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
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert _read_f32(output_f32, 8) == (4.0,) * 8
