from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

import largerlm.prefill_execute as prefill_execute
from largerlm.cli import main as cli_main
from largerlm.prefill_execute import (
    PrefillExecuteError,
    run_prefill_staged_routed_mlp_block_batch,
)


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


def _write_expert_layout(root: Path) -> Path:
    experts = root / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
    slot_bytes = 192
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-affine-int4",
                "group_size": 8,
                "num_layers": 1,
                "num_experts": 4,
                "component_order": [name for name, *_ in COMPONENTS],
                "layers": [
                    {
                        "layer": 3,
                        "num_experts": 4,
                        "expert_slot_bytes": slot_bytes,
                        "layer_file": "layer_003.bin",
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
    (experts / "layer_003.bin").write_bytes(
        b"".join(bytes([expert]) * slot_bytes for expert in range(4))
    )
    return layout


def _write_resident_layout(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir()
    router = struct.pack("<32f", *([1.0] * 32))
    norm = struct.pack("<8f", *([1.0] * 8))
    shared = struct.pack("<64f", *([1.0] * 64))
    payload = router + norm + shared + shared + shared
    (resident / "resident.bin").write_bytes(payload)
    tensors = [
        {
            "name": "model.layers.3.mlp.gate.weight",
            "offset": 0,
            "size": len(router),
            "dtype": "F32",
            "shape": [4, 8],
            "category": "routers",
        },
        {
            "name": "model.layers.3.post_attention_layernorm.weight",
            "offset": len(router),
            "size": len(norm),
            "dtype": "F32",
            "shape": [8],
            "category": "norms",
        },
    ]
    offset = len(router) + len(norm)
    for component in ("gate_proj", "up_proj", "down_proj"):
        tensors.append(
            {
                "name": f"model.layers.3.mlp.shared_experts.{component}.weight",
                "offset": offset,
                "size": len(shared),
                "dtype": "F32",
                "shape": [8, 8],
                "category": "shared_experts",
            }
        )
        offset += len(shared)
    layout = resident / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    return layout


def _write_fake_runner(root: Path) -> Path:
    runner = root / "fake_runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys


def value(flag: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1]


if "--run-rmsnorm-batch" in sys.argv:
    open(value("--output-f32"), "wb").write(open(value("--input-f32"), "rb").read())
elif "--run-resident-linear-batch" in sys.argv:
    if value("--tensor-suffix") == ".mlp.gate.weight":
        batch = int(value("--batch-tokens"))
        row = [0.0, 0.0, 4.0, 0.0]
        open(value("--output-f32"), "wb").write(struct.pack("<" + "f" * (batch * 4), *(row * batch)))
    else:
        open(value("--output-f32"), "wb").write(open(value("--input-f32"), "rb").read())
elif "--run-router-batch" in sys.argv:
    from pathlib import Path
    out_dir = Path(value("--output-router-json-dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = int(value("--batch-tokens"))
    for token in range(batch):
        (out_dir / f"token_{token:06d}.router.json").write_text(
            json.dumps({"experts": [2, 0], "weights": [0.75, 0.25]}),
            encoding="utf-8",
        )
elif "--run-router" in sys.argv:
    open(value("--output-router-json"), "w", encoding="utf-8").write(
        json.dumps({"experts": [2, 0], "weights": [0.75, 0.25]})
    )
elif "--run-moe-batch" in sys.argv:
    if "--routes-bin" in sys.argv:
        raw_routes = open(value("--routes-bin"), "rb").read()
        header = struct.unpack_from("<8sIIIIIIII", raw_routes, 0)
        magic, version, batch_tokens, expert_count, capacity, _total, _used, overflow_count, _reserved = header
        if magic != b"LLMSCAP1" or version != 1:
            raise SystemExit("bad static capacity binary")
        offset = struct.calcsize("<8sIIIIIIII")
        experts = struct.unpack_from("<" + "I" * expert_count, raw_routes, offset)
        offset += 4 * expert_count
        routes = [{"experts": [], "weights": []} for _ in range(batch_tokens)]
        for expert in experts:
            for _slot in range(capacity):
                token, weight, active = struct.unpack_from("<IfI", raw_routes, offset)
                offset += struct.calcsize("<IfI")
                if active:
                    routes[token]["experts"].append(expert)
                    routes[token]["weights"].append(weight)
        for _ in range(overflow_count):
            expert, _overflow_index, token, weight = struct.unpack_from("<IIIf", raw_routes, offset)
            offset += struct.calcsize("<IIIf")
            routes[token]["experts"].append(expert)
            routes[token]["weights"].append(weight)
    else:
        routes = json.load(open(value("--routes-json"), "r", encoding="utf-8"))["routes"]
    raw = open(value("--input-f32"), "rb").read()
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    batch = int(value("--batch-tokens"))
    hidden = len(values) // batch
    out = []
    for token, route in enumerate(routes):
        offset = sum(float(item) for item in route["weights"])
        row = values[token * hidden : (token + 1) * hidden]
        out.extend(item + offset for item in row)
    open(value("--output-f32"), "wb").write(struct.pack("<" + "f" * len(out), *out))
    print("  timing sort:        0.001")
    print("  timing setup:       0.002")
    print("  timing expert read: 0.003")
    print("  timing input read:  0.004")
    print("  timing output read: 0.005")
    print("  timing kernel:      0.006")
    print("  timing output write:0.007")
    print("  timing final read:  0.008")
    print("  timing total:       0.036")
elif "--run-moe" in sys.argv:
    weights = [float(item) for item in value("--weights").split(",") if item]
    raw = open(value("--input-f32"), "rb").read()
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    offset = sum(weights)
    open(value("--output-f32"), "wb").write(
        struct.pack("<" + "f" * len(values), *(item + offset for item in values))
    )
else:
    raise SystemExit("unsupported fake runner command")
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def _read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack("<" + "f" * (len(raw) // 4), raw)


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def _install_fake_resident_linear_for_router_hybrid(
    monkeypatch: pytest.MonkeyPatch,
    *,
    custom_row: tuple[float, ...],
    fallback_row: tuple[float, ...],
) -> list[str]:
    calls: list[str] = []

    def fake_run_resident_batch_linear(**kwargs):
        backend = str(kwargs["prefill_linear_backend"])
        row = custom_row if backend == "custom-metal" else fallback_row
        batch_tokens = int(kwargs["batch_tokens"])
        output_path = Path(kwargs["output_f32_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            struct.pack("<" + "f" * (batch_tokens * len(row)), *(row * batch_tokens))
        )
        calls.append(backend)
        input_path = Path(kwargs["input_f32_path"])
        input_bytes = input_path.stat().st_size
        output_bytes = batch_tokens * len(row) * 4
        return prefill_execute.ResidentBatchLinearResult(
            runner_path=Path(kwargs["runner_path"]),
            resident_layout_path=Path(kwargs["resident_layout_path"]),
            input_path=input_path,
            output_path=output_path,
            layer=int(kwargs["layer"]),
            tensor="model.layers.3.mlp.gate.weight",
            tensor_suffix=str(kwargs["tensor_suffix"]),
            dtype="F32",
            backend=backend,
            batch_tokens=batch_tokens,
            in_dim=8,
            out_dim=len(row),
            matrix_bytes=128,
            matrix_scratch_bytes=128,
            matrix_f32_bytes=0,
            matrix_raw_conversion_bytes=0,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            estimated_peak_bytes=128 + input_bytes + output_bytes,
            elapsed_seconds=0.01 if backend == "custom-metal" else 0.02,
            command=("fake-runner", backend),
            runner_backend_elapsed_seconds=None,
            runner_matrix_f32_elapsed_seconds=None,
            runner_accelerator_elapsed_seconds=None,
        )

    monkeypatch.setattr(
        prefill_execute,
        "run_resident_batch_linear",
        fake_run_resident_batch_linear,
    )
    return calls


def _run_router_hybrid_for_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    custom_row: tuple[float, ...],
    fallback_row: tuple[float, ...] = (4.0, 3.0, 0.0, 0.0),
    threshold: float = 0.1,
):
    resident_layout = _write_resident_layout(tmp_path)
    input_f32 = tmp_path / "router_input.f32"
    input_f32.write_bytes(struct.pack("<16f", *tuple(float(item) for item in range(16))))
    calls = _install_fake_resident_linear_for_router_hybrid(
        monkeypatch,
        custom_row=custom_row,
        fallback_row=fallback_row,
    )
    output_dir = tmp_path / "router_hybrid"
    result = prefill_execute._run_router_json_batch(
        runner_path=tmp_path / "fake_runner.py",
        resident_layout_path=resident_layout,
        layer=3,
        input_f32_path=input_f32,
        output_dir=output_dir,
        router_json_dir=output_dir / "router_json",
        batch_tokens=2,
        hidden_dim=8,
        top_k=2,
        router_score="raw",
        routed_scaling_factor=None,
        norm_topk_prob=False,
        no_norm_topk_prob=False,
        router_n_group=None,
        router_topk_group=None,
        ignore_router_bias=False,
        max_resident_matrix_mib=1,
        max_router_mib=1,
        max_runner_scratch_mib=64,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=2,
        prefill_mpsgraph_min_matrix_dim=4,
        router_hybrid_margin_threshold=threshold,
        keep_token_files=True,
        echo_runner_output=False,
    )
    return result, calls, output_dir


def test_prefill_staged_routed_mlp_block_batch_runs_full_flow(tmp_path: Path) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_values = tuple(float(item) for item in range(1, 17))
    input_f32.write_bytes(struct.pack("<16f", *input_values))
    output = tmp_path / "out.f32"

    result = run_prefill_staged_routed_mlp_block_batch(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        layer=3,
        input_f32_path=input_f32,
        output_dir=tmp_path / "staged_prefill",
        output_f32_path=output,
        batch_tokens=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_runner_scratch_mib=64,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        prefill_ssd_read_gib_per_second=10.0,
        prefill_max_routed_read_seconds=1.0,
        static_capacity_per_expert=2,
        prefill_linear_backend="auto",
        echo_runner_output=False,
    )

    assert result.batch_tokens == 2
    assert result.hidden_dim == 8
    assert result.router_command_count == 1
    assert result.routed_command_count == 1
    assert result.staged_bytes == 2 * 192
    assert result.compact_stage_bytes == 2 * 192
    assert result.compact_stage_materialized_bytes == 0
    assert result.compact_stage_storage == "hardlink"
    assert result.stage_plus_compact_materialized_bytes == result.staged_bytes
    assert result.static_capacity_path == tmp_path / "staged_prefill" / "staged_moe" / "static_capacity.json"
    assert result.static_capacity_binary_path == tmp_path / "staged_prefill" / "staged_moe" / "static_capacity.bin"
    assert result.static_capacity_per_expert == 2
    assert result.static_capacity_used_slots == 4
    assert result.static_capacity_total_slots == 4
    assert result.static_capacity_overflow_assignments == 0
    assert result.static_capacity_binary_bytes == result.static_capacity_binary_path.stat().st_size
    assert result.stage_result.selected_experts == (0, 2)
    assert result.stage_result.io_summary.planned_read_seconds == pytest.approx(
        result.stage_result.planned_read_bytes / (10.0 * 1024**3)
    )
    assert result.stage_result.io_summary.max_read_seconds == 1.0
    assert result.stage_result.io_summary.read_seconds_ok is True
    assert result.stage_result.io_summary.max_raw_ranges == 0
    assert result.stage_result.io_summary.raw_range_count_ok is None
    assert result.stage_result.io_summary.max_coalesced_ranges == 0
    assert result.stage_result.io_summary.coalesced_range_count_ok is None
    assert result.stage_result.io_summary.copy_elapsed_seconds is not None
    assert result.stage_result.io_summary.copy_elapsed_seconds >= 0.0
    assert result.stage_result.io_summary.copy_throughput_gib_per_second is not None
    assert result.stage_result.io_summary.copy_throughput_gib_per_second >= 0.0
    assert result.stage_copy_elapsed_seconds == pytest.approx(
        result.stage_result.io_summary.copy_elapsed_seconds
    )
    assert result.stage_copy_throughput_gib_per_second == pytest.approx(
        result.stage_result.io_summary.copy_throughput_gib_per_second
    )
    assert result.staged_moe.selected_experts == (0, 2)
    assert result.staged_moe.stage_copy_elapsed_seconds == pytest.approx(
        result.stage_result.io_summary.copy_elapsed_seconds
    )
    assert result.staged_moe.stage_copy_throughput_gib_per_second == pytest.approx(
        result.stage_result.io_summary.copy_throughput_gib_per_second
    )
    assert result.staged_moe.static_capacity_path == result.static_capacity_path
    assert result.staged_moe.static_capacity_binary_path == result.static_capacity_binary_path
    assert result.staged_moe.moe_token_block == "auto"
    assert result.staged_moe.moe_timing_sort_seconds == pytest.approx(0.001)
    assert result.staged_moe.moe_timing_setup_seconds == pytest.approx(0.002)
    assert result.staged_moe.moe_timing_expert_read_seconds == pytest.approx(0.003)
    assert result.staged_moe.moe_timing_input_read_seconds == pytest.approx(0.004)
    assert result.staged_moe.moe_timing_output_read_seconds == pytest.approx(0.005)
    assert result.staged_moe.moe_timing_kernel_seconds == pytest.approx(0.006)
    assert result.staged_moe.moe_timing_output_write_seconds == pytest.approx(0.007)
    assert result.staged_moe.moe_timing_final_read_seconds == pytest.approx(0.008)
    assert result.staged_moe.moe_timing_total_seconds == pytest.approx(0.036)
    assert "--routes-bin" in result.staged_moe.first_command
    assert "--moe-token-block" in result.staged_moe.first_command
    assert "auto" in result.staged_moe.first_command
    static_payload = json.loads(result.static_capacity_path.read_text(encoding="utf-8"))
    assert static_payload["capacity_per_expert"] == 2
    assert static_payload["selected_experts"] == [0, 1]
    assert _read_f32(output) == pytest.approx(tuple((2.0 * value) + 1.0 for value in input_values))
    assert sorted(path.name for path in result.router_json_dir.glob("*.json")) == [
        "token_000000.router.json",
        "token_000001.router.json",
    ]


def test_prefill_staged_routed_mlp_accelerates_router_gate(
    tmp_path: Path,
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_values = tuple(float(item) for item in range(1, 17))
    input_f32.write_bytes(struct.pack("<16f", *input_values))
    output = tmp_path / "out.f32"

    result = run_prefill_staged_routed_mlp_block_batch(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        layer=3,
        input_f32_path=input_f32,
        output_dir=tmp_path / "staged_prefill",
        output_f32_path=output,
        batch_tokens=2,
        top_k=2,
        max_k=2,
        include_shared_expert=True,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        prefill_ssd_read_gib_per_second=10.0,
        prefill_max_routed_read_seconds=1.0,
        static_capacity_per_expert=2,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=2,
        prefill_mpsgraph_min_matrix_dim=4,
        echo_runner_output=False,
    )

    assert result.router_gate_proj is not None
    assert result.router_gate_proj.backend == "mpsgraph-f32"
    assert "--run-resident-linear-batch" in result.first_router_command
    assert "--run-router-batch" not in result.first_router_command
    payload = json.loads(
        (result.router_json_dir / "token_000000.router.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experts"] == [2, 0]
    assert payload["used_correction_bias"] is False
    assert payload["router_margin"]["topk_score_margin"] == pytest.approx(0.0)
    assert payload["router_margin"]["effective_score_margin"] == pytest.approx(0.0)
    assert result.router_margin_summary is not None
    assert result.router_margin_summary["min_effective_score_margin"] == pytest.approx(
        0.0
    )
    assert result.router_margin_summary["effective_near_tie_counts"]["le_1e-06"] == 2
    assert result.stage_result.selected_experts == (0, 2)


def test_router_hybrid_accepts_custom_when_margin_is_above_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (command_count, first_command, router_gate, summary), calls, output_dir = (
        _run_router_hybrid_for_test(
            tmp_path,
            monkeypatch,
            custom_row=(4.0, 3.0, 0.0, 0.0),
        )
    )

    assert calls == ["custom-metal"]
    assert command_count == 1
    assert first_command == ("fake-runner", "custom-metal")
    assert router_gate is not None
    assert router_gate.backend == "custom-metal"
    assert summary is not None
    assert summary["min_effective_score_margin"] == pytest.approx(3.0)
    assert summary["router_gate_policy"]["decision"] == "custom-metal"
    payload = json.loads(
        (output_dir / "router_json" / "token_000000.router.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experts"] == [0, 1]


def test_router_hybrid_falls_back_to_mpsgraph_when_margin_is_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (command_count, first_command, router_gate, summary), calls, output_dir = (
        _run_router_hybrid_for_test(
            tmp_path,
            monkeypatch,
            custom_row=(4.0, 3.0, 3.0, 0.0),
        )
    )

    assert calls == ["custom-metal", "mpsgraph-f32"]
    assert command_count == 2
    assert first_command == ("fake-runner", "custom-metal")
    assert router_gate is not None
    assert router_gate.backend == "mpsgraph-f32"
    assert summary is not None
    policy = summary["router_gate_policy"]
    assert policy["decision"] == "mpsgraph-f32-fallback"
    assert policy["custom_min_effective_score_margin"] == pytest.approx(0.0)
    assert policy["fallback_min_effective_score_margin"] == pytest.approx(3.0)
    payload = json.loads(
        (output_dir / "router_json" / "token_000000.router.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experts"] == [0, 1]


def test_router_hybrid_keeps_env_threshold_fallback_for_legacy_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        prefill_execute.PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD_ENV,
        "0.1",
    )
    (command_count, _, router_gate, summary), calls, _ = _run_router_hybrid_for_test(
        tmp_path,
        monkeypatch,
        custom_row=(4.0, 3.0, 3.0, 0.0),
        threshold=0.0,
    )

    assert calls == ["custom-metal", "mpsgraph-f32"]
    assert command_count == 2
    assert router_gate is not None
    assert router_gate.backend == "mpsgraph-f32"
    assert summary is not None
    policy = summary["router_gate_policy"]
    assert policy["decision"] == "mpsgraph-f32-fallback"
    assert policy["margin_threshold"] == pytest.approx(0.1)


def test_prefill_staged_routed_mlp_rejects_stage_read_seconds_before_copy(
    tmp_path: Path,
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<16f", *tuple(float(item) for item in range(1, 17)))
    )
    output_dir = tmp_path / "staged_prefill"

    with pytest.raises(PrefillExecuteError, match="planned stage read time"):
        run_prefill_staged_routed_mlp_block_batch(
            runner_path=runner,
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            layer=3,
            input_f32_path=input_f32,
            output_dir=output_dir,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            max_slot_mib=1,
            max_router_mib=1,
            max_runner_scratch_mib=64,
            expert_stage_align_kib=0.0009765625,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            copy_chunk_mib=0.0001,
            prefill_ssd_read_gib_per_second=1.0,
            prefill_max_routed_read_seconds=1e-12,
            echo_runner_output=False,
        )

    assert not (output_dir / "experts.stage.bin").exists()


def test_prefill_staged_routed_mlp_rejects_stage_range_cap_before_copy(
    tmp_path: Path,
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<16f", *tuple(float(item) for item in range(1, 17)))
    )
    output_dir = tmp_path / "staged_prefill"

    with pytest.raises(PrefillExecuteError, match="coalesced range count"):
        run_prefill_staged_routed_mlp_block_batch(
            runner_path=runner,
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            layer=3,
            input_f32_path=input_f32,
            output_dir=output_dir,
            output_f32_path=tmp_path / "out.f32",
            batch_tokens=2,
            top_k=2,
            max_k=2,
            router_score="raw",
            routed_scaling_factor=1.0,
            max_slot_mib=1,
            max_router_mib=1,
            max_runner_scratch_mib=64,
            expert_stage_align_kib=0.0009765625,
            max_stage_mib=1,
            max_compact_stage_mib=1,
            copy_chunk_mib=0.0001,
            expert_stage_max_coalesced_ranges=1,
            echo_runner_output=False,
        )

    assert not (output_dir / "experts.stage.bin").exists()
    assert not (output_dir / "experts.stage.manifest.json").exists()


def test_prefill_staged_routed_mlp_block_batch_can_skip_static_capacity_json(
    tmp_path: Path,
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_values = tuple(float(item) for item in range(1, 17))
    input_f32.write_bytes(struct.pack("<16f", *input_values))
    output = tmp_path / "out.f32"

    result = run_prefill_staged_routed_mlp_block_batch(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        layer=3,
        input_f32_path=input_f32,
        output_dir=tmp_path / "staged_prefill",
        output_f32_path=output,
        batch_tokens=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_runner_scratch_mib=64,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        static_capacity_per_expert=2,
        write_static_capacity_json=False,
        prefill_linear_backend="auto",
        echo_runner_output=False,
    )

    assert result.static_capacity_path is None
    assert result.staged_moe.static_capacity_path is None
    assert result.static_capacity_binary_path == (
        tmp_path / "staged_prefill" / "staged_moe" / "static_capacity.bin"
    )
    assert result.static_capacity_binary_path.exists()
    assert not (
        tmp_path / "staged_prefill" / "staged_moe" / "static_capacity.json"
    ).exists()
    assert "--routes-bin" in result.staged_moe.first_command
    assert _read_f32(output) == pytest.approx(tuple((2.0 * value) + 1.0 for value in input_values))


def test_prefill_staged_routed_mlp_block_batch_can_tile_expert_stage(
    tmp_path: Path,
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<16f", *([0.0] * 16)))
    output = tmp_path / "out.f32"

    result = run_prefill_staged_routed_mlp_block_batch(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        layer=3,
        input_f32_path=input_f32,
        output_dir=tmp_path / "staged_prefill",
        output_f32_path=output,
        batch_tokens=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        max_slot_mib=1,
        max_router_mib=1,
        max_runner_scratch_mib=64,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=0.0002,
        max_compact_stage_mib=0.0002,
        copy_chunk_mib=0.0001,
        expert_stage_tiling=True,
        static_capacity_per_expert=2,
        prefill_linear_backend="auto",
        echo_runner_output=False,
    )

    assert result.tiled_staged_moe is not None
    assert result.tiled_staged_moe.tile_count == 2
    assert result.routed_command_count == 2
    assert result.staged_bytes == 384
    assert result.compact_stage_bytes == 384
    assert result.compact_stage_storage == "tiled"
    assert result.static_capacity_path is None
    assert result.static_capacity_binary_path is None
    assert result.static_capacity_per_expert == 2
    assert result.static_capacity_used_slots == 4
    assert result.static_capacity_total_slots == 4
    assert result.static_capacity_overflow_assignments == 0
    assert result.static_capacity_binary_bytes > 0
    assert sorted(
        tuple(stage.selected_experts)
        for stage in result.tiled_staged_moe.tile_stage_results
    ) == [(0,), (2,)]
    assert _read_f32(output) == pytest.approx(tuple([1.0] * 16))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"layer": True}, "layer must be an integer"),
        ({"batch_tokens": False}, "batch_tokens must be an integer"),
        ({"batch_tokens": 1.5}, "batch_tokens must be an integer"),
        ({"top_k": 1.5}, "top_k must be an integer"),
        ({"max_k": True}, "max_k must be an integer"),
        ({"routed_scaling_factor": False}, "routed_scaling_factor must be a finite number"),
        ({"router_n_group": 1.5}, "router_n_group must be an integer"),
        ({"router_topk_group": True}, "router_topk_group must be an integer"),
        ({"rms_norm_eps": False}, "rms_norm_eps must be a finite number"),
        ({"max_slot_mib": 1.5}, "max_slot_mib must be an integer MiB value"),
        ({"expert_stage_merge_gap_kib": True}, "expert_stage_merge_gap_kib must be a finite number"),
        ({"expert_stage_align_kib": float("nan")}, "expert_stage_align_kib must be a finite number"),
        ({"max_stage_mib": 0}, "max_stage_mib must be positive"),
        ({"max_compact_stage_mib": float("nan")}, "max_compact_stage_mib must be a finite number"),
        ({"copy_chunk_mib": False}, "copy_chunk_mib must be a finite number"),
        ({"stage_disk_safety_margin_bytes": True}, "stage_disk_safety_margin_bytes must be an integer"),
        ({"stage_disk_safety_margin_bytes": 1.5}, "stage_disk_safety_margin_bytes must be an integer"),
        ({"stage_disk_safety_margin_bytes": -1}, "stage_disk_safety_margin_bytes must be non-negative"),
        ({"prefill_ssd_read_gib_per_second": False}, "prefill_ssd_read_gib_per_second must be a finite number"),
        ({"prefill_max_routed_read_seconds": -1.0}, "prefill_max_routed_read_seconds must be non-negative"),
        ({"prefill_max_routed_read_seconds": 1.0}, "prefill_ssd_read_gib_per_second must be positive"),
        ({"expert_stage_max_raw_ranges": True}, "expert_stage_max_raw_ranges must be an integer"),
        ({"expert_stage_max_raw_ranges": -1}, "expert_stage_max_raw_ranges must be non-negative"),
        ({"expert_stage_max_coalesced_ranges": False}, "expert_stage_max_coalesced_ranges must be an integer"),
        ({"expert_stage_max_coalesced_ranges": -1}, "expert_stage_max_coalesced_ranges must be non-negative"),
        ({"expert_stage_tiling": 1}, "expert_stage_tiling must be a boolean"),
        ({"static_capacity_per_expert": True}, "static_capacity_per_expert must be an integer"),
        ({"static_capacity_per_expert": 1.5}, "static_capacity_per_expert must be an integer"),
        ({"static_capacity_per_expert": 0}, "static_capacity_per_expert must be positive"),
    ),
)
def test_prefill_staged_routed_mlp_block_batch_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    args: dict[str, object] = {
        "runner_path": tmp_path / "missing-runner",
        "expert_layout_path": tmp_path / "missing-experts.json",
        "resident_layout_path": tmp_path / "missing-resident.json",
        "layer": 3,
        "input_f32_path": tmp_path / "input.f32",
        "output_dir": tmp_path / "staged_prefill",
        "output_f32_path": tmp_path / "out.f32",
        "batch_tokens": 2,
    }
    args.update(kwargs)

    with pytest.raises(PrefillExecuteError, match=message):
        run_prefill_staged_routed_mlp_block_batch(**args)


def test_prefill_staged_routed_mlp_block_batch_can_include_shared_expert(
    tmp_path: Path,
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_values = tuple(float(item) for item in range(1, 17))
    input_f32.write_bytes(struct.pack("<16f", *input_values))
    output = tmp_path / "out.f32"

    result = run_prefill_staged_routed_mlp_block_batch(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        layer=3,
        input_f32_path=input_f32,
        output_dir=tmp_path / "staged_prefill",
        output_f32_path=output,
        batch_tokens=2,
        top_k=2,
        max_k=2,
        router_score="raw",
        routed_scaling_factor=1.0,
        include_shared_expert=True,
        max_resident_matrix_mib=1,
        max_slot_mib=1,
        max_router_mib=1,
        max_runner_scratch_mib=64,
        expert_stage_align_kib=0.0009765625,
        max_stage_mib=1,
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        prefill_linear_backend="auto",
        echo_runner_output=False,
    )

    expected = tuple((2.0 * value) + 1.0 + (_silu(value) * value) for value in input_values)
    assert result.include_shared_expert is True
    assert result.shared_output_bytes == 64
    assert result.shared_gate_proj is not None
    assert result.shared_up_proj is not None
    assert result.shared_down_proj is not None
    assert result.shared_gate_proj.backend == "custom-metal"
    assert result.shared_up_proj.backend == "custom-metal"
    assert result.shared_down_proj.backend == "custom-metal"
    assert _read_f32(output) == pytest.approx(expected)


def test_prefill_staged_routed_mlp_block_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expert_layout = _write_expert_layout(tmp_path)
    resident_layout = _write_resident_layout(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<16f", *([1.0] * 16)))
    output = tmp_path / "out.f32"

    status = cli_main(
        [
            "prefill-staged-routed-mlp-block-batch",
            str(runner),
            str(expert_layout),
            str(resident_layout),
            "--layer",
            "3",
            "--input-f32",
            str(input_f32),
            "--output-dir",
            str(tmp_path / "staged_prefill"),
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
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--expert-stage-align-kib",
            "0.0009765625",
            "--max-stage-mib",
            "1",
            "--max-compact-stage-mib",
            "1",
            "--copy-chunk-mib",
            "0.0001",
            "--expert-stage-max-raw-ranges",
            "2",
            "--expert-stage-max-coalesced-ranges",
            "2",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["staged_bytes"] == 384
    assert payload["compact_stage_bytes"] == 384
    assert payload["compact_stage_materialized_bytes"] == 0
    assert payload["compact_stage_storage"] == "hardlink"
    assert payload["stage_plus_compact_materialized_bytes"] == 384
    assert payload["stage_result"]["io_summary"]["max_raw_ranges"] == 2
    assert payload["stage_result"]["io_summary"]["raw_range_count_ok"] is True
    assert payload["stage_result"]["io_summary"]["max_coalesced_ranges"] == 2
    assert payload["stage_result"]["io_summary"]["coalesced_range_count_ok"] is True
    assert payload["output_path"] == str(output)
