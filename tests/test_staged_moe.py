from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import largerlm.staged_moe as staged_moe_module
from largerlm.cli import main as cli_main
from largerlm.expert_io import stage_batch_experts
from largerlm.safety import DiskBudget
from largerlm.staged_moe import (
    StagedMoEError,
    StagedRoutedMoEBatchPlanServerSession,
    StagedRoutedMoEBatchPlanJob,
    run_staged_routed_moe_batch,
    run_staged_routed_moe_batch_plan,
    run_staged_routed_moe_batch_plan_server,
    run_tiled_staged_routed_moe_batch,
    write_staged_routed_moe_batch_plan,
)


def _write_expert_layout(root: Path, *, slot_bytes: int = 32) -> Path:
    experts = root / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
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
                "component_order": ["down_proj.weight"],
                "layers": [
                    {
                        "layer": 3,
                        "num_experts": 4,
                        "expert_slot_bytes": slot_bytes,
                        "layer_file": "layer_003.bin",
                        "components": [
                            {
                                "name": "down_proj.weight",
                                "offset": 0,
                                "size": 16,
                                "dtype": "U32",
                                "shape": [4, 1],
                            }
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


def _write_router_jsons(root: Path) -> Path:
    router_dir = root / "router"
    router_dir.mkdir()
    (router_dir / "token_000000.router.json").write_text(
        json.dumps({"experts": [2, 0], "weights": [0.75, 0.25]}),
        encoding="utf-8",
    )
    (router_dir / "token_000001.router.json").write_text(
        json.dumps({"experts": [2], "weights": [0.125]}),
        encoding="utf-8",
    )
    return router_dir


def _write_fake_runner(root: Path) -> Path:
    runner = root / "fake_runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import struct
import sys


sys.stdout.reconfigure(line_buffering=True)


parser = argparse.ArgumentParser()
parser.add_argument("--layout")
parser.add_argument("--layer")
parser.add_argument("--run-moe", action="store_true")
parser.add_argument("--run-moe-batch", action="store_true")
parser.add_argument("--run-moe-batch-plan", action="store_true")
parser.add_argument("--run-moe-batch-plan-server-jsonl", action="store_true")
parser.add_argument("--experts")
parser.add_argument("--weights")
parser.add_argument("--routes-json")
parser.add_argument("--routes-bin")
parser.add_argument("--batch-plan-json")
parser.add_argument("--input-f32")
parser.add_argument("--output-f32")
parser.add_argument("--batch-tokens")
parser.add_argument("--max-k")
parser.add_argument("--max-slot-mib")
parser.add_argument("--max-runner-scratch-mib")
parser.add_argument("--moe-token-block")
args = parser.parse_args()


def load_routes(routes_json, routes_bin):
    if routes_bin:
        raw_routes = open(routes_bin, "rb").read()
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
        routes = json.load(open(routes_json, "r", encoding="utf-8"))["routes"]
    return routes


def run_batch(*, routes_json, routes_bin, input_f32, output_f32, batch_tokens, moe_token_block):
    raw = open(input_f32, "rb").read()
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    routes = load_routes(routes_json, routes_bin)
    batch = int(batch_tokens)
    hidden = len(values) // batch
    out = []
    for token, route in enumerate(routes):
        offset = sum(float(item) for item in route["weights"])
        row = values[token * hidden : (token + 1) * hidden]
        out.extend(value + offset for value in row)
    open(output_f32, "wb").write(struct.pack("<" + "f" * len(out), *out))
    expert_counts = {}
    for route in routes:
        for expert in route["experts"]:
            expert_counts[expert] = expert_counts.get(expert, 0) + 1
    max_expert_tokens = max(expert_counts.values())
    requested = str(moe_token_block or "auto")
    effective = max_expert_tokens if requested == "auto" else min(int(requested), max_expert_tokens)
    print("LargerLM MoE batch")
    print(f"  token block mode:   {'auto' if requested == 'auto' else 'fixed'}")
    print(f"  max expert tokens:  {max_expert_tokens}")
    print(f"  token block:        {effective}")
    accumulator = os.environ.get("LARGERLM_MOE_BATCH_ACCUMULATOR", "file").strip().lower()
    if accumulator in ("1", "memory", "mem", "in-memory", "ram", "true", "yes", "on"):
        accumulator = "memory"
    else:
        accumulator = "file"
    accum_bytes = batch * hidden * 4 if accumulator == "memory" else 0
    print(f"  output accumulator: {accumulator}")
    print(f"  output accum bytes: {accum_bytes}")
    print(f"  batch buffer bytes: {effective * 100}")
    print(f"  estimated peak:     {effective * 1000}")


def run_plan(plan_path):
    plan = json.load(open(plan_path, "r", encoding="utf-8"))
    for index, job in enumerate(plan["jobs"], start=1):
        print("LargerLM MoE batch plan job")
        print(f"  job:                 {index}/{len(plan['jobs'])}")
        run_batch(
            routes_json=job.get("routes_json"),
            routes_bin=job.get("routes_bin"),
            input_f32=job["input_f32"],
            output_f32=job["output_f32"],
            batch_tokens=job["batch_tokens"],
            moe_token_block=job.get("moe_token_block", "auto"),
        )
    print("LargerLM MoE batch plan")
    print(f"  jobs:                {len(plan['jobs'])}")
    print("  plan timing total:   0.123000")


if args.run_moe_batch_plan_server_jsonl:
    print("LargerLM MoE batch plan server")
    for raw_line in sys.stdin:
        request = json.loads(raw_line)
        if request.get("command") == "quit":
            break
        plan_path = request.get("batch_plan_json") or request.get("plan_path")
        print("LargerLM MoE batch plan server request")
        print(f"  plan:                {plan_path}")
        run_plan(plan_path)
        print("  server request:      ok")
    print("LargerLM MoE batch plan server done")
elif args.run_moe_batch_plan:
    run_plan(args.batch_plan_json)
elif args.run_moe_batch:
    run_batch(
        routes_json=args.routes_json,
        routes_bin=args.routes_bin,
        input_f32=args.input_f32,
        output_f32=args.output_f32,
        batch_tokens=args.batch_tokens,
        moe_token_block=args.moe_token_block or "auto",
    )
else:
    raw = open(args.input_f32, "rb").read()
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    weights = [float(item) for item in args.weights.split(",") if item]
    offset = sum(weights)
    open(args.output_f32, "wb").write(
        struct.pack("<" + "f" * len(values), *(value + offset for value in values))
    )
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def _stage_manifest(root: Path) -> tuple[Path, Path, Path]:
    layout = _write_expert_layout(root)
    router_dir = _write_router_jsons(root)
    stage_file = root / "stage.bin"
    manifest = root / "stage_manifest.json"
    stage_batch_experts(
        layout,
        layer=3,
        router_json_dir=router_dir,
        stage_file_path=stage_file,
        manifest_path=manifest,
        align_bytes=1,
        max_stage_mib=1,
    )
    return layout, stage_file, manifest


def _mutate_layout(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mutate_manifest(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    return struct.unpack("<" + "f" * (len(raw) // 4), raw)


def test_gather_and_scatter_add_f32_rows_for_tiled_moe(tmp_path: Path) -> None:
    source = tmp_path / "source.f32"
    gathered = tmp_path / "gathered.f32"
    partial_a = tmp_path / "partial_a.f32"
    partial_b = tmp_path / "partial_b.f32"
    accumulator = tmp_path / "accumulator.f32"
    source.write_bytes(
        struct.pack(
            "<12f",
            1.0,
            2.0,
            3.0,
            4.0,
            10.0,
            20.0,
            30.0,
            40.0,
            100.0,
            200.0,
            300.0,
            400.0,
        )
    )

    staged_moe_module._gather_f32_rows(
        input_path=source,
        output_path=gathered,
        token_indices=[2, 0],
        batch_tokens=3,
        hidden_dim=4,
    )

    assert _read_f32(gathered) == (
        100.0,
        200.0,
        300.0,
        400.0,
        1.0,
        2.0,
        3.0,
        4.0,
    )
    partial_a.write_bytes(struct.pack("<8f", *([1.0] * 8)))
    staged_moe_module._scatter_add_f32_rows(
        accumulator_path=accumulator,
        partial_path=partial_a,
        token_indices=[2, 0],
        batch_tokens=3,
        hidden_dim=4,
        initialize=True,
    )
    partial_b.write_bytes(struct.pack("<4f", 2.0, 3.0, 4.0, 5.0))
    staged_moe_module._scatter_add_f32_rows(
        accumulator_path=accumulator,
        partial_path=partial_b,
        token_indices=[2],
        batch_tokens=3,
        hidden_dim=4,
    )

    assert _read_f32(accumulator) == (
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        4.0,
        5.0,
        6.0,
    )


def test_gather_f32_rows_rejects_duplicate_tokens(tmp_path: Path) -> None:
    source = tmp_path / "source.f32"
    source.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="token_indices must be unique"):
        staged_moe_module._gather_f32_rows(
            input_path=source,
            output_path=tmp_path / "out.f32",
            token_indices=[0, 0],
            batch_tokens=2,
            hidden_dim=4,
        )


def test_contiguous_row_runs_group_adjacent_indices() -> None:
    assert staged_moe_module._contiguous_row_runs((0, 1, 2, 5, 7, 8)) == (
        (0, 0, 3),
        (3, 5, 1),
        (4, 7, 2),
    )


def test_gather_and_scatter_add_f32_rows_use_contiguous_runs(tmp_path: Path) -> None:
    source = tmp_path / "source.f32"
    gathered = tmp_path / "gathered.f32"
    partial = tmp_path / "partial.f32"
    accumulator = tmp_path / "accumulator.f32"
    source.write_bytes(struct.pack("<24f", *[float(i) for i in range(24)]))

    staged_moe_module._gather_f32_rows(
        input_path=source,
        output_path=gathered,
        token_indices=[1, 2, 3, 5],
        batch_tokens=6,
        hidden_dim=4,
    )

    assert _read_f32(gathered) == tuple(float(i) for i in range(4, 16)) + tuple(
        float(i) for i in range(20, 24)
    )
    partial.write_bytes(struct.pack("<16f", *([1.0] * 16)))
    staged_moe_module._scatter_add_f32_rows(
        accumulator_path=accumulator,
        partial_path=partial,
        token_indices=[1, 2, 3, 5],
        batch_tokens=6,
        hidden_dim=4,
        initialize=True,
    )

    assert _read_f32(accumulator) == (
        *(0.0 for _ in range(4)),
        *(1.0 for _ in range(12)),
        *(0.0 for _ in range(4)),
        *(1.0 for _ in range(4)),
    )


def test_tile_token_routes_for_experts_builds_active_subbatch() -> None:
    original_indices, routes = staged_moe_module._tile_token_routes_for_experts(
        [
            {"token_index": 0, "experts": [0, 2], "weights": [0.25, 0.75]},
            {"token_index": 1, "experts": [3], "weights": [1.0]},
            {"token_index": 2, "experts": [2, 3], "weights": [0.4, 0.6]},
        ],
        selected_experts=(2,),
    )

    assert original_indices == (0, 2)
    assert routes == [
        {"token_index": 0, "experts": [2], "weights": [0.75]},
        {"token_index": 1, "experts": [2], "weights": [0.4]},
    ]


def test_write_tile_router_jsons_materializes_active_subbatch(tmp_path: Path) -> None:
    original_indices, router_dir = staged_moe_module._write_tile_router_jsons(
        output_dir=tmp_path / "tile_router_json",
        token_routes=[
            {"token_index": 0, "experts": [0, 2], "weights": [0.25, 0.75]},
            {"token_index": 1, "experts": [3], "weights": [1.0]},
            {"token_index": 2, "experts": [2, 3], "weights": [0.4, 0.6]},
        ],
        selected_experts=(2,),
    )

    assert original_indices == (0, 2)
    paths = sorted(router_dir.glob("*.router.json"))
    assert [path.name for path in paths] == [
        "token_000000.router.json",
        "token_000001.router.json",
    ]
    assert [json.loads(path.read_text(encoding="utf-8")) for path in paths] == [
        {"token_index": 0, "experts": [2], "weights": [0.75]},
        {"token_index": 1, "experts": [2], "weights": [0.4]},
    ]


def test_run_staged_routed_moe_batch_compacts_and_runs_tokens(tmp_path: Path) -> None:
    _layout, stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0))
    output = tmp_path / "out.f32"

    result = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=output,
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.batch_tokens == 2
    assert result.hidden_dim == 4
    assert result.selected_experts == (0, 2)
    assert result.compact_stage_bytes == 64
    assert result.compact_stage_materialized_bytes == 0
    assert result.compact_stage_storage == "hardlink"
    assert result.stage_io_summary_available is True
    assert result.stage_serial_read_bytes == 96
    assert result.stage_unique_requested_bytes == 64
    assert result.stage_planned_read_bytes == 64
    assert result.stage_staged_bytes == 64
    assert result.stage_waste_bytes == 0
    assert result.stage_coalesced_savings_bytes == 32
    assert result.stage_raw_range_count == 2
    assert result.stage_coalesced_range_count == 2
    assert result.stage_assignment_read_amplification == pytest.approx(64 / 96)
    assert result.stage_unique_read_amplification == pytest.approx(1.0)
    assert result.stage_staged_unique_read_amplification == pytest.approx(1.0)
    assert result.stage_budget_utilization == pytest.approx(64 / 1024**2)
    assert result.stage_copy_elapsed_seconds is not None
    assert result.stage_copy_elapsed_seconds >= 0.0
    assert result.stage_copy_throughput_gib_per_second is not None
    assert result.stage_copy_throughput_gib_per_second >= 0.0
    assert result.stage_copy_seconds_ok is None
    assert result.stage_read_advice_available is True
    assert result.stage_read_advice_attempted_ranges == 2
    assert result.stage_read_advice_calls >= 0
    assert result.stage_read_advice_bytes >= 0
    assert result.compact_layer_path.samefile(stage_file)
    assert result.command_count == 1
    assert result.moe_token_block == "auto"
    assert result.moe_token_block_mode == "auto"
    assert result.effective_moe_token_block == 2
    assert result.moe_max_expert_tokens == 2
    assert result.moe_batch_buffer_bytes == 200
    assert result.moe_estimated_peak_bytes == 2000
    assert result.moe_output_accumulator == "file"
    assert result.moe_output_accumulator_bytes == 0
    assert "--run-moe-batch" in result.first_command
    assert "--moe-token-block" in result.first_command
    assert "auto" in result.first_command
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_summary = manifest_payload["io_summary"]
    assert manifest_summary["copy_elapsed_seconds"] == pytest.approx(
        result.stage_copy_elapsed_seconds
    )
    assert manifest_summary["copy_throughput_gib_per_second"] == pytest.approx(
        result.stage_copy_throughput_gib_per_second
    )
    assert manifest_summary.get("copy_seconds_ok") is None
    routes = json.loads(result.compact_routes_path.read_text(encoding="utf-8"))
    assert routes["routes"][0]["experts"] == [1, 0]
    assert _read_f32(output) == pytest.approx(
        (2.0, 3.0, 4.0, 5.0, 10.125, 20.125, 30.125, 40.125)
    )
    compact_stage = result.compact_layer_path.read_bytes()
    assert compact_stage[:1] == b"\x00"
    assert compact_stage[32:33] == b"\x02"
    compact_layout = json.loads(result.compact_layout_path.read_text(encoding="utf-8"))
    assert compact_layout["num_experts"] == 2
    assert compact_layout["original_experts"] == [0, 2]
    assert not any((result.output_dir / "tokens").glob("*.f32"))


def test_run_staged_routed_moe_batch_can_pin_memory_output_accumulator(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0)
    )

    result = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=tmp_path / "out.f32",
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        moe_output_accumulator="memory",
        echo_runner_output=False,
    )

    assert result.moe_output_accumulator == "memory"
    assert result.moe_output_accumulator_bytes == 32


def test_run_staged_routed_moe_batch_plan_runs_materialized_job(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0)
    )
    staged_output = tmp_path / "staged-out.f32"
    staged = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=staged_output,
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )
    plan_output = tmp_path / "plan-out.f32"
    plan_path = write_staged_routed_moe_batch_plan(
        tmp_path / "moe-plan.json",
        [
            StagedRoutedMoEBatchPlanJob(
                layout_path=staged.compact_layout_path,
                layer=staged.layer,
                routes_json_path=staged.compact_routes_path,
                routes_bin_path=None,
                input_path=input_f32,
                output_path=plan_output,
                batch_tokens=staged.batch_tokens,
                max_k=2,
                max_slot_mib=1,
                max_runner_scratch_mib=64,
                moe_token_block=2,
            )
        ],
    )

    result = run_staged_routed_moe_batch_plan(
        runner_path=runner,
        plan_path=plan_path,
        echo_runner_output=False,
    )

    assert result.job_count == 1
    assert result.command_count == 1
    assert result.runner_reported_elapsed_seconds == pytest.approx(0.123)
    assert "--run-moe-batch-plan" in result.first_command
    assert result.output_paths == (plan_output,)
    assert _read_f32(plan_output) == pytest.approx(_read_f32(staged_output))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "largerlm.staged_routed_moe_batch_plan.v1"
    assert payload["jobs"][0]["moe_token_block"] == 2
    assert payload["jobs"][0]["routes_json"] == str(staged.compact_routes_path)


def test_run_staged_routed_moe_batch_plan_server_runs_multiple_plans(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0)
    )
    staged = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=tmp_path / "staged-out.f32",
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )
    plans: list[Path] = []
    outputs: list[Path] = []
    for index in range(2):
        output = tmp_path / f"server-out-{index}.f32"
        outputs.append(output)
        plans.append(
            write_staged_routed_moe_batch_plan(
                tmp_path / f"moe-plan-{index}.json",
                [
                    StagedRoutedMoEBatchPlanJob(
                        layout_path=staged.compact_layout_path,
                        layer=staged.layer,
                        routes_json_path=staged.compact_routes_path,
                        routes_bin_path=None,
                        input_path=input_f32,
                        output_path=output,
                        batch_tokens=staged.batch_tokens,
                        max_k=2,
                        max_slot_mib=1,
                        max_runner_scratch_mib=64,
                    )
                ],
            )
        )

    result = run_staged_routed_moe_batch_plan_server(
        runner_path=runner,
        plan_paths=plans,
        echo_runner_output=False,
    )

    assert result.plan_count == 2
    assert result.job_count == 2
    assert result.command_count == 1
    assert "--run-moe-batch-plan-server-jsonl" in result.first_command
    assert result.plan_paths == tuple(plans)
    assert result.output_paths == tuple(outputs)
    for output in outputs:
        assert _read_f32(output) == pytest.approx(_read_f32(staged.output_path))


def test_staged_routed_moe_batch_plan_server_session_submits_plans(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0)
    )
    staged = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=tmp_path / "staged-out.f32",
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )
    plans: list[Path] = []
    outputs: list[Path] = []
    for index in range(2):
        output = tmp_path / f"session-out-{index}.f32"
        outputs.append(output)
        plans.append(
            write_staged_routed_moe_batch_plan(
                tmp_path / f"session-plan-{index}.json",
                [
                    StagedRoutedMoEBatchPlanJob(
                        layout_path=staged.compact_layout_path,
                        layer=staged.layer,
                        routes_json_path=staged.compact_routes_path,
                        routes_bin_path=None,
                        input_path=input_f32,
                        output_path=output,
                        batch_tokens=staged.batch_tokens,
                        max_k=2,
                        max_slot_mib=1,
                        max_runner_scratch_mib=64,
                    )
                ],
            )
        )

    with StagedRoutedMoEBatchPlanServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        first = session.submit_plan(plans[0])
        second = session.submit_plan(plans[1])
        assert session.running

    assert first.command_count == 0
    assert second.command_count == 0
    assert first.output_paths == (outputs[0],)
    assert second.output_paths == (outputs[1],)
    for output in outputs:
        assert _read_f32(output) == pytest.approx(_read_f32(staged.output_path))


def test_run_staged_routed_moe_batch_can_use_plan_server_session(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(
        struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0)
    )
    output = tmp_path / "session-staged-out.f32"

    with StagedRoutedMoEBatchPlanServerSession(
        runner_path=runner,
        moe_output_accumulator="memory",
        echo_runner_output=False,
    ) as session:
        result = run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=output,
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            copy_chunk_mib=0.0001,
            max_slot_mib=1,
            max_runner_scratch_mib=64,
            echo_runner_output=False,
            moe_plan_server_session=session,
            moe_output_accumulator="file",
        )

    assert result.command_count == 0
    assert "--run-moe-batch-plan-server-jsonl" in result.first_command
    assert result.effective_moe_token_block == 2
    assert result.moe_estimated_peak_bytes == 2000
    assert result.moe_output_accumulator == "memory"
    assert result.moe_output_accumulator_bytes == 32
    assert (result.output_dir / "moe_batch_plan.json").exists()
    assert _read_f32(output) == pytest.approx(
        (2.0, 3.0, 4.0, 5.0, 10.125, 20.125, 30.125, 40.125)
    )


def test_run_tiled_staged_routed_moe_batch_scatters_tile_outputs(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output = tmp_path / "out.f32"

    result = run_tiled_staged_routed_moe_batch(
        runner_path=runner,
        expert_layout_path=layout,
        layer=3,
        router_json_dir=router_dir,
        input_f32_path=input_f32,
        output_f32_path=output,
        output_dir=tmp_path / "tiled_moe",
        merge_gap_bytes=0,
        align_bytes=1,
        max_stage_mib=32 / 1024**2,
        max_compact_stage_mib=32 / 1024**2,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.tile_count == 2
    assert result.tile_original_token_indices == ((0,), (0, 1))
    assert result.total_stage_planned_read_bytes == 64
    assert result.max_tile_stage_planned_read_bytes == 32
    assert result.total_compact_stage_bytes == 64
    assert result.max_tile_compact_stage_bytes == 32
    assert result.output_bytes == 32
    assert _read_f32(output) == pytest.approx(
        (1.0, 1.0, 1.0, 1.0, 0.125, 0.125, 0.125, 0.125)
    )


def test_run_tiled_staged_routed_moe_batch_can_use_plan_server_session(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output = tmp_path / "out-session.f32"

    with StagedRoutedMoEBatchPlanServerSession(
        runner_path=runner,
        echo_runner_output=False,
    ) as session:
        result = run_tiled_staged_routed_moe_batch(
            runner_path=runner,
            expert_layout_path=layout,
            layer=3,
            router_json_dir=router_dir,
            input_f32_path=input_f32,
            output_f32_path=output,
            output_dir=tmp_path / "tiled_moe_session",
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=32 / 1024**2,
            max_compact_stage_mib=32 / 1024**2,
            copy_chunk_mib=0.0001,
            max_slot_mib=1,
            max_runner_scratch_mib=64,
            echo_runner_output=False,
            moe_plan_server_session=session,
        )

    assert result.tile_count == 2
    assert all(tile.command_count == 0 for tile in result.tile_results)
    assert all(
        "--run-moe-batch-plan-server-jsonl" in tile.first_command
        for tile in result.tile_results
    )
    assert _read_f32(output) == pytest.approx(
        (1.0, 1.0, 1.0, 1.0, 0.125, 0.125, 0.125, 0.125)
    )


def test_run_tiled_staged_routed_moe_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output = tmp_path / "out.f32"

    status = cli_main(
        [
            "run-tiled-staged-routed-moe-batch",
            str(runner),
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--input-f32",
            str(input_f32),
            "--output-dir",
            str(tmp_path / "tiled_moe_cli"),
            "--output-f32",
            str(output),
            "--align-kib",
            "0.0009765625",
            "--max-stage-mib",
            str(32 / 1024**2),
            "--max-compact-stage-mib",
            str(32 / 1024**2),
            "--copy-chunk-mib",
            "0.0001",
            "--max-slot-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tile_count"] == 2
    assert payload["tile_original_token_indices"] == [[0], [0, 1]]
    assert payload["max_tile_stage_planned_read_bytes"] == 32
    assert _read_f32(output) == pytest.approx(
        (1.0, 1.0, 1.0, 1.0, 0.125, 0.125, 0.125, 0.125)
    )


def test_run_staged_routed_moe_batch_falls_back_to_compact_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([1.0] * 8)))
    output = tmp_path / "out.f32"

    def fail_link(source: object, destination: object) -> None:
        del source, destination
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(staged_moe_module.os, "link", fail_link)

    result = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=output,
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        echo_runner_output=False,
    )

    assert result.compact_stage_storage == "copy"
    assert result.compact_stage_materialized_bytes == result.compact_stage_bytes
    assert not result.compact_layer_path.samefile(stage_file)
    assert result.compact_layer_path.read_bytes() == stage_file.read_bytes()


def test_run_staged_routed_moe_batch_rejects_boolean_component_shape(
    tmp_path: Path,
) -> None:
    layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0))

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        components = layers[0]["components"]
        components[0]["shape"][0] = True

    _mutate_layout(layout, mutate)

    with pytest.raises(StagedMoEError, match="component down_proj.weight must have shape"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            copy_chunk_mib=0.0001,
            max_slot_mib=1,
            max_runner_scratch_mib=64,
            echo_runner_output=False,
        )


def test_run_staged_routed_moe_batch_rejects_invalid_output_accumulator(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(
        StagedMoEError,
        match="moe_output_accumulator must be env, file, or memory",
    ):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            moe_output_accumulator="bad",
            echo_runner_output=False,
        )


def test_run_staged_routed_moe_batch_writes_static_capacity_artifact(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([1.0] * 8)))
    output = tmp_path / "out.f32"

    result = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=output,
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        static_capacity_per_expert=2,
        echo_runner_output=False,
    )

    assert result.static_capacity_path == result.output_dir / "static_capacity.json"
    assert result.static_capacity_binary_path == result.output_dir / "static_capacity.bin"
    assert result.static_capacity_per_expert == 2
    assert result.static_capacity_used_slots == 3
    assert result.static_capacity_total_slots == 4
    assert result.static_capacity_overflow_assignments == 0
    assert result.static_capacity_binary_bytes == result.static_capacity_binary_path.stat().st_size
    assert "--routes-bin" in result.first_command
    assert "--moe-token-block" in result.first_command
    assert str(result.static_capacity_binary_path) in result.first_command
    payload = json.loads(result.static_capacity_path.read_text(encoding="utf-8"))
    assert payload["selected_experts"] == [0, 1]
    assert payload["experts"][1]["slots"] == [
        {"active": True, "slot": 0, "token_index": 0, "weight": 0.75},
        {"active": True, "slot": 1, "token_index": 1, "weight": 0.125},
    ]
    assert payload["experts"][0]["slots"][1]["active"] is False
    assert (
        struct.unpack_from("<8sIIIIIIII", result.static_capacity_binary_path.read_bytes(), 0)
        == (b"LLMSCAP1", 1, 2, 2, 2, 3, 3, 0, 0)
    )


def test_run_staged_routed_moe_batch_can_skip_static_capacity_json(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([1.0] * 8)))
    output = tmp_path / "out.f32"

    result = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=output,
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        copy_chunk_mib=0.0001,
        max_slot_mib=1,
        max_runner_scratch_mib=64,
        static_capacity_per_expert=2,
        write_static_capacity_json=False,
        echo_runner_output=False,
    )

    assert result.static_capacity_path is None
    assert result.static_capacity_binary_path == result.output_dir / "static_capacity.bin"
    assert result.static_capacity_binary_path.exists()
    assert not (result.output_dir / "static_capacity.json").exists()
    assert "--routes-bin" in result.first_command
    assert str(result.static_capacity_binary_path) in result.first_command
    assert _read_f32(output) == pytest.approx(
        (2.0, 2.0, 2.0, 2.0, 1.125, 1.125, 1.125, 1.125)
    )


def test_run_staged_routed_moe_batch_rejects_static_capacity_overflow(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    with pytest.raises(StagedMoEError, match="overflow assignments"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            static_capacity_per_expert=1,
            echo_runner_output=False,
        )

    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_static_capacity_disk_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"
    calls = 0

    def fail_link(source: object, destination: object) -> None:
        del source, destination
        raise OSError("hardlink unavailable")

    def fake_disk_budget(
        output_dir: str | Path,
        required_bytes: int,
        *,
        safety_margin_bytes: int = 0,
    ) -> DiskBudget:
        nonlocal calls
        calls += 1
        available = required_bytes + safety_margin_bytes if calls == 1 else 0
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=available,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr(staged_moe_module.os, "link", fail_link)
    monkeypatch.setattr(staged_moe_module, "disk_budget", fake_disk_budget)

    with pytest.raises(StagedMoEError, match="static capacity route artifacts"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            static_capacity_per_expert=2,
            echo_runner_output=False,
        )

    assert calls >= 2
    assert not (output_dir / "static_capacity.json").exists()
    assert not (output_dir / "static_capacity.bin").exists()
    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_validates_static_capacity_binary_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    def fail_validate(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise staged_moe_module.ExpertIOPlanError("route table corrupted")

    monkeypatch.setattr(
        staged_moe_module,
        "validate_static_expert_capacity_binary",
        fail_validate,
    )

    with pytest.raises(StagedMoEError, match="route table corrupted"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            static_capacity_per_expert=2,
            echo_runner_output=False,
        )

    assert not (output_dir / "static_capacity.bin").exists()
    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_compact_stage_limit(tmp_path: Path) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="exceed limit"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=0.00001,
            echo_runner_output=False,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"max_compact_stage_mib": float("nan")}, "max_compact_stage_mib must be finite"),
        ({"copy_chunk_mib": 0.0}, "copy_chunk_mib must be positive"),
        ({"max_slot_mib": 1.5}, "max_slot_mib must be an integer"),
        ({"max_runner_scratch_mib": True}, "max_runner_scratch_mib must be an integer"),
        (
            {"static_capacity_per_expert": True},
            "static_capacity_per_expert must be an integer",
        ),
        (
            {"static_capacity_per_expert": 1.5},
            "static_capacity_per_expert must be an integer",
        ),
        ({"disk_safety_margin_bytes": -1}, "disk_safety_margin_bytes must be non-negative"),
    ),
)
def test_run_staged_routed_moe_batch_rejects_invalid_caps_before_output(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    kwargs = {
        "runner_path": runner,
        "stage_manifest_path": manifest,
        "input_f32_path": input_f32,
        "output_f32_path": tmp_path / "out.f32",
        "output_dir": output_dir,
        "echo_runner_output": False,
    }
    kwargs.update(overrides)

    with pytest.raises(StagedMoEError, match=message):
        run_staged_routed_moe_batch(**kwargs)

    assert not output_dir.exists()
    assert not (tmp_path / "out.f32").exists()


@pytest.mark.parametrize(
    ("capacity_per_expert", "message"),
    (
        (True, "static_capacity_per_expert must be an integer"),
        (1.5, "static_capacity_per_expert must be an integer"),
        (0, "static_capacity_per_expert must be positive"),
    ),
)
def test_static_capacity_plan_from_compact_routes_rejects_invalid_capacity(
    capacity_per_expert: object,
    message: str,
) -> None:
    with pytest.raises(StagedMoEError, match=message):
        staged_moe_module._static_capacity_plan_from_compact_routes(
            batch_tokens=1,
            selected_compact_experts=(0,),
            routes=[{"token_index": 0, "experts": [0], "weights": [1.0]}],
            capacity_per_expert=capacity_per_expert,
        )


def test_run_staged_routed_moe_batch_rejects_duplicate_selected_experts(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    _mutate_manifest(manifest, lambda payload: payload.update({"selected_experts": [0, 0]}))
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    with pytest.raises(StagedMoEError, match="selected_experts must be unique"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (output_dir / "compact_stage.bin").exists()
    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_unsorted_selected_experts(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    _mutate_manifest(
        manifest,
        lambda payload: payload.update({"selected_experts": [2, 0]}),
    )
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="selected_experts must be sorted"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_stage_manifest_size_mismatch(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    _mutate_manifest(
        manifest,
        lambda payload: payload.update({"staged_bytes": payload["staged_bytes"] + 1}),
    )
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="staged_bytes .* stage file size"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_stage_range_gap(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        ranges = payload["ranges"]
        assert isinstance(ranges, list)
        ranges[0]["stage_offset"] = 1

    _mutate_manifest(manifest, mutate)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="ranges must cover the stage file"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_slot_source_offset_mismatch(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        slots = payload["slots"]
        assert isinstance(slots, list)
        slots[1]["source_offset"] = 0

    _mutate_manifest(manifest, mutate)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="source_offset .* expected"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_accepts_expert_order_source_offsets(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["expert_order"] = [2, 0, 1, 3]

    _mutate_layout(layout, mutate)
    (layout.parent / "layer_003.bin").write_bytes(
        b"".join(bytes([expert]) * 32 for expert in (2, 0, 1, 3))
    )
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    manifest = tmp_path / "stage_manifest.json"
    stage_batch_experts(
        layout,
        layer=3,
        router_json_dir=router_dir,
        stage_file_path=stage_file,
        manifest_path=manifest,
        align_bytes=1,
        max_stage_mib=1,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["slots"][0]["expert"] == 0
    assert payload["slots"][0]["source_offset"] == 32
    assert payload["slots"][1]["expert"] == 2
    assert payload["slots"][1]["source_offset"] == 0

    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    result = run_staged_routed_moe_batch(
        runner_path=runner,
        stage_manifest_path=manifest,
        input_f32_path=input_f32,
        output_f32_path=tmp_path / "out.f32",
        output_dir=tmp_path / "staged_moe",
        max_compact_stage_mib=1,
        echo_runner_output=False,
    )

    assert result.selected_experts == (0, 2)
    assert (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_slot_stage_offset_mismatch(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        slots = payload["slots"]
        assert isinstance(slots, list)
        slots[1]["stage_offset"] = 0

    _mutate_manifest(manifest, mutate)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="stage_offset .* expected"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (tmp_path / "out.f32").exists()


@pytest.mark.parametrize(
    ("route_update", "message"),
    (
        ({"token_index": 1.5}, "token route token_index must be an integer"),
        (
            {"experts": [True], "weights": [1.0]},
            "token route expert must be an integer",
        ),
        (
            {"experts": [2], "weights": [float("nan")]},
            "token route weight must be finite",
        ),
    ),
)
def test_run_staged_routed_moe_batch_rejects_invalid_token_routes_and_cleans_compact_files(
    tmp_path: Path,
    route_update: dict[str, object],
    message: str,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["batch_plan"]["token_routes"][0].update(route_update)

    _mutate_manifest(manifest, mutate)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    with pytest.raises(StagedMoEError, match=message):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (output_dir / "compact_stage.bin").exists()
    assert not (output_dir / "compact_layout.json").exists()
    assert not (output_dir / "compact_routes.json").exists()
    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_rejects_duplicate_token_index(
    tmp_path: Path,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        payload["batch_plan"]["token_routes"][1]["token_index"] = 0

    _mutate_manifest(manifest, mutate)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))

    with pytest.raises(StagedMoEError, match="token_index values must be unique"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=tmp_path / "staged_moe",
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )


def test_run_staged_routed_moe_batch_rejects_compact_disk_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    def fail_link(source: object, destination: object) -> None:
        del source, destination
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(staged_moe_module.os, "link", fail_link)

    with pytest.raises(StagedMoEError, match="not enough free disk"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            disk_safety_margin_bytes=10**30,
            echo_runner_output=False,
        )

    assert not (output_dir / "compact_stage.bin").exists()


def test_run_staged_routed_moe_batch_removes_partial_compact_stage_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    def fail_link(source: object, destination: object) -> None:
        del source, destination
        raise OSError("hardlink unavailable")

    def fail_copy(**kwargs) -> None:
        kwargs["destination"].write(b"partial")
        raise StagedMoEError("copy exploded")

    monkeypatch.setattr(staged_moe_module.os, "link", fail_link)
    monkeypatch.setattr(staged_moe_module, "_copy_exact_range", fail_copy)

    with pytest.raises(StagedMoEError, match="failed to build compact stage file"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (output_dir / "compact_stage.bin").exists()
    assert not (tmp_path / "out.f32").exists()


def test_staged_moe_copy_exact_range_rejects_short_write(tmp_path: Path) -> None:
    source_path = tmp_path / "stage.bin"
    source_path.write_bytes(b"abcdefgh")

    class ShortWriter:
        def write(self, chunk: bytes) -> int:
            del chunk
            return 3

    with source_path.open("rb") as source:
        with pytest.raises(
            StagedMoEError,
            match="failed to write 4 compact stage bytes at source offset 0",
        ):
            staged_moe_module._copy_exact_range(
                source=source,
                destination=ShortWriter(),
                source_offset=0,
                length=4,
                copy_chunk_bytes=4,
            )


def test_staged_moe_write_json_atomic_removes_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "compact_layout.json"

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace exploded")

    monkeypatch.setattr(type(path), "replace", fail_replace)

    with pytest.raises(OSError, match="replace exploded"):
        staged_moe_module._write_json_atomic(path, {"ok": True})

    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_run_staged_routed_moe_batch_removes_compact_stage_on_layout_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"

    def fail_layout(path: Path, payload: object) -> None:
        if path.name == "compact_layout.json":
            raise OSError("layout exploded")
        staged_moe_module._write_json_atomic(path, payload)

    monkeypatch.setattr(staged_moe_module, "_write_json_atomic", fail_layout)

    with pytest.raises(StagedMoEError, match="failed to write compact layout"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (output_dir / "compact_stage.bin").exists()
    assert not (output_dir / "compact_layout.json").exists()
    assert stage_file.exists()
    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_removes_compact_files_on_routes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([0.0] * 8)))
    output_dir = tmp_path / "staged_moe"
    original_write_json_atomic = staged_moe_module._write_json_atomic

    def fail_routes(path: Path, payload: object) -> None:
        if path.name == "compact_routes.json":
            raise OSError("routes exploded")
        original_write_json_atomic(path, payload)

    monkeypatch.setattr(staged_moe_module, "_write_json_atomic", fail_routes)

    with pytest.raises(StagedMoEError, match="failed to write compact routes"):
        run_staged_routed_moe_batch(
            runner_path=runner,
            stage_manifest_path=manifest,
            input_f32_path=input_f32,
            output_f32_path=tmp_path / "out.f32",
            output_dir=output_dir,
            max_compact_stage_mib=1,
            echo_runner_output=False,
        )

    assert not (output_dir / "compact_stage.bin").exists()
    assert not (output_dir / "compact_layout.json").exists()
    assert not (output_dir / "compact_routes.json").exists()
    assert stage_file.exists()
    assert not (tmp_path / "out.f32").exists()


def test_run_staged_routed_moe_batch_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([1.0] * 8)))
    output = tmp_path / "out.f32"

    status = cli_main(
        [
            "run-staged-routed-moe-batch",
            str(runner),
            "--stage-manifest",
            str(manifest),
            "--input-f32",
            str(input_f32),
            "--output-dir",
            str(tmp_path / "staged_moe"),
            "--output-f32",
            str(output),
            "--max-compact-stage-mib",
            "1",
            "--copy-chunk-mib",
            "0.0001",
            "--compact-stage-disk-margin-mib",
            "0",
            "--max-slot-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--static-capacity-per-expert",
            "2",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 2
    assert payload["compact_stage_bytes"] == 64
    assert payload["compact_stage_materialized_bytes"] == 0
    assert payload["compact_stage_storage"] == "hardlink"
    assert payload["stage_io_summary_available"] is True
    assert payload["stage_serial_read_bytes"] == 96
    assert payload["stage_unique_requested_bytes"] == 64
    assert payload["stage_planned_read_bytes"] == 64
    assert payload["stage_staged_bytes"] == 64
    assert payload["stage_waste_bytes"] == 0
    assert payload["stage_coalesced_savings_bytes"] == 32
    assert payload["stage_raw_range_count"] == 2
    assert payload["stage_coalesced_range_count"] == 2
    assert payload["stage_assignment_read_amplification"] == pytest.approx(64 / 96)
    assert payload["stage_unique_read_amplification"] == pytest.approx(1.0)
    assert payload["stage_budget_utilization"] == pytest.approx(64 / 1024**2)
    assert payload["stage_read_advice_available"] is True
    assert payload["stage_read_advice_attempted_ranges"] == 2
    assert payload["stage_read_advice_calls"] >= 0
    assert payload["stage_read_advice_bytes"] >= 0
    assert payload["stage_copy_seconds_ok"] is None
    assert "compact_routes_path" in payload
    assert payload["static_capacity_per_expert"] == 2
    assert payload["static_capacity_overflow_assignments"] == 0
    assert Path(payload["static_capacity_path"]).exists()
    assert Path(payload["static_capacity_binary_path"]).exists()
    assert payload["static_capacity_binary_bytes"] > 0
    assert payload["moe_token_block"] == "auto"
    assert payload["moe_token_block_mode"] == "auto"
    assert payload["effective_moe_token_block"] == 2
    assert payload["moe_max_expert_tokens"] == 2
    assert payload["moe_output_accumulator"] == "file"
    assert payload["moe_output_accumulator_bytes"] == 0
    assert "--routes-bin" in payload["first_command"]
    assert "--moe-token-block" in payload["first_command"]
    assert "auto" in payload["first_command"]
    assert payload["output_path"] == str(output)
    assert output.exists()


def test_run_staged_routed_moe_batch_cli_can_skip_static_capacity_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _layout, _stage_file, manifest = _stage_manifest(tmp_path)
    runner = _write_fake_runner(tmp_path)
    input_f32 = tmp_path / "input.f32"
    input_f32.write_bytes(struct.pack("<8f", *([1.0] * 8)))
    output = tmp_path / "out.f32"
    output_dir = tmp_path / "staged_moe"

    status = cli_main(
        [
            "run-staged-routed-moe-batch",
            str(runner),
            "--stage-manifest",
            str(manifest),
            "--input-f32",
            str(input_f32),
            "--output-dir",
            str(output_dir),
            "--output-f32",
            str(output),
            "--max-compact-stage-mib",
            "1",
            "--copy-chunk-mib",
            "0.0001",
            "--max-slot-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--static-capacity-per-expert",
            "2",
            "--no-static-capacity-json",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["static_capacity_path"] is None
    assert Path(payload["static_capacity_binary_path"]).exists()
    assert not (output_dir / "static_capacity.json").exists()
    assert "--routes-bin" in payload["first_command"]
