#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from layer_moe_smoke import write_fixture


def expert_output(value: float, input_value: float) -> float:
    gate = 8.0 * value * input_value
    up = 8.0 * value * input_value
    act = (gate / (1.0 + math.exp(-gate))) * up
    return 8.0 * value * act


def expected_row(input_value: float) -> float:
    return (2.0 / 3.0) * expert_output(2.0, input_value) + (
        1.0 / 3.0
    ) * expert_output(1.0, input_value)


def _run(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-staged-moe-", dir="/private/tmp"))
    write_fixture(root)
    router_dir = root / "router_json"
    router_dir.mkdir()
    for token in range(2):
        (router_dir / f"token_{token:06d}.router.json").write_text(
            json.dumps({"experts": [1, 0], "weights": [2.0 / 3.0, 1.0 / 3.0]}),
            encoding="utf-8",
        )
    (root / "batch_input.f32").write_bytes(
        struct.pack("<16f", *([1.0] * 8 + [0.5] * 8))
    )
    runner = Path(__file__).with_name("largerlm-runner")
    stage_file = root / "stage.bin"
    manifest = root / "stage_manifest.json"
    output_dir = root / "staged_moe"
    output = root / "staged_moe_output.f32"

    _run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "stage-batch-experts",
            str(root / "experts" / "layout.json"),
            "--layer",
            "1",
            "--router-json-dir",
            str(router_dir),
            "--stage-file",
            str(stage_file),
            "--manifest",
            str(manifest),
            "--max-stage-mib",
            "1",
            "--copy-chunk-mib",
            "0.0001",
            "--json",
        ]
    )
    stdout = _run(
        [
            sys.executable,
            "-m",
            "largerlm",
            "run-staged-routed-moe-batch",
            str(runner),
            "--stage-manifest",
            str(manifest),
            "--input-f32",
            str(root / "batch_input.f32"),
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
            "--quiet-runner",
            "--json",
        ]
    )
    payload = json.loads(stdout)
    if (
        payload["batch_tokens"] != 2
        or payload["hidden_dim"] != 8
        or payload["output_bytes"] != 64
        or payload["compact_stage_bytes"] != 384
        or payload["command_count"] != 1
        or "--run-moe-batch" not in payload["first_command"]
        or "--routes-bin" not in payload["first_command"]
        or not Path(payload["static_capacity_binary_path"]).exists()
    ):
        raise SystemExit(f"unexpected staged MoE payload: {payload}")
    Path(payload["compact_routes_path"]).unlink()
    direct_stdout = _run(list(payload["first_command"]))
    if "assignment order:   pre-sorted" not in direct_stdout:
        raise SystemExit(f"static routes did not bypass assignment sort:\n{direct_stdout}")
    if "token block mode:   auto" not in direct_stdout:
        raise SystemExit(f"static routes did not use auto token block:\n{direct_stdout}")
    if "token block:        2" not in direct_stdout:
        raise SystemExit(f"unexpected MoE token block:\n{direct_stdout}")
    if "output zero rows:   2" not in direct_stdout or "output read rows:   2" not in direct_stdout:
        raise SystemExit(f"unexpected output accumulator read pattern:\n{direct_stdout}")
    values = struct.unpack("<16f", output.read_bytes())
    expected = (expected_row(1.0), expected_row(0.5))
    if abs(values[0] - expected[0]) > 3e-3 or abs(values[8] - expected[1]) > 3e-3:
        raise SystemExit(
            f"outputs {values[0]:.6f}, {values[8]:.6f} != expected "
            f"{expected[0]:.6f}, {expected[1]:.6f}"
        )

    plan_output = root / "staged_moe_plan_output.f32"
    plan_json = root / "moe_batch_plan.json"
    plan_json.write_text(
        json.dumps(
            {
                "schema": "largerlm.staged_routed_moe_batch_plan.v1",
                "version": 1,
                "job_count": 1,
                "jobs": [
                    {
                        "layout": payload["compact_layout_path"],
                        "layer": payload["layer"],
                        "routes_bin": payload["static_capacity_binary_path"],
                        "input_f32": str(root / "batch_input.f32"),
                        "output_f32": str(plan_output),
                        "batch_tokens": payload["batch_tokens"],
                        "max_k": 2,
                        "max_slot_mib": 1,
                        "max_runner_scratch_mib": 64,
                        "moe_token_block": "auto",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plan_stdout = _run(
        [
            str(runner),
            "--run-moe-batch-plan",
            "--batch-plan-json",
            str(plan_json),
        ]
    )
    if "LargerLM MoE batch plan" not in plan_stdout:
        raise SystemExit(f"plan runner did not report plan summary:\n{plan_stdout}")
    if "plan timing total:" not in plan_stdout:
        raise SystemExit(f"plan runner did not report plan timing:\n{plan_stdout}")
    plan_values = struct.unpack("<16f", plan_output.read_bytes())
    if plan_values != values:
        raise SystemExit("plan output does not match staged MoE output")

    server_output = root / "staged_moe_server_output.f32"
    server_plan_json = root / "moe_batch_server_plan.json"
    server_payload = json.loads(plan_json.read_text(encoding="utf-8"))
    server_payload["jobs"][0]["output_f32"] = str(server_output)
    server_plan_json.write_text(json.dumps(server_payload, indent=2), encoding="utf-8")
    server_request = (
        json.dumps({"batch_plan_json": str(server_plan_json)})
        + "\n"
        + json.dumps({"command": "quit"})
        + "\n"
    )
    server_completed = subprocess.run(
        [str(runner), "--run-moe-batch-plan-server-jsonl"],
        input=server_request,
        text=True,
        capture_output=True,
    )
    if server_completed.stdout:
        print(server_completed.stdout, end="")
    if server_completed.stderr:
        print(server_completed.stderr, end="")
    if server_completed.returncode != 0:
        raise SystemExit(server_completed.returncode)
    if "LargerLM MoE batch plan server" not in server_completed.stdout:
        raise SystemExit("server runner did not report JSONL server header")
    if "server request:      ok" not in server_completed.stdout:
        raise SystemExit("server runner did not complete request")
    server_values = struct.unpack("<16f", server_output.read_bytes())
    if server_values != values:
        raise SystemExit("server output does not match staged MoE output")
    print(f"fixture: {root}")
    print("  staged routed MoE batch: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
