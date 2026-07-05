#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path


DEFAULT_EXPERTS = "34,36,89,149,152,183,201,206"
DEFAULT_WEIGHTS = "0.18,0.15,0.13,0.12,0.11,0.1,0.11,0.1"
HIDDEN_DIM = 6144


def csv_value_count(csv: str) -> int:
    return sum(1 for item in csv.split(",") if item.strip())


def assert_layer_moe_read_telemetry(
    probe: dict[str, object],
    *,
    route_count: int,
    active_expert_buffers: int,
) -> None:
    if route_count <= 0:
        raise SystemExit("route_count must be positive for MoE read telemetry")
    read_batch = max(1, min(active_expert_buffers, route_count))
    expected_dispatches = 0
    expected_pool_dispatches = 0
    expected_serial_dispatches = 0
    remaining = route_count
    while remaining > 0:
        batch = min(read_batch, remaining)
        expected_dispatches += 1
        if batch > 1:
            expected_pool_dispatches += 1
        else:
            expected_serial_dispatches += 1
        remaining -= batch
    if probe.get("expert_read_dispatch_count") != expected_dispatches:
        raise SystemExit(f"unexpected expert read dispatch count: {probe}")
    if probe.get("expert_read_task_count") != route_count:
        raise SystemExit(f"unexpected expert read task count: {probe}")
    if probe.get("expert_read_max_task_count") != read_batch:
        raise SystemExit(f"unexpected expert read max task count: {probe}")
    if read_batch > 1 and int(probe.get("expert_read_max_worker_count") or 0) < read_batch:
        raise SystemExit(f"expected at least {read_batch} pread workers: {probe}")
    if probe.get("expert_read_pool_dispatch_count") != expected_pool_dispatches:
        raise SystemExit(f"unexpected pooled expert read dispatch count: {probe}")
    if probe.get("expert_read_serial_dispatch_count") != expected_serial_dispatches:
        raise SystemExit(f"unexpected serial expert read dispatch count: {probe}")


def write_input(path: Path) -> None:
    values = [
        (math.sin(i * 0.013) + math.cos(i * 0.007)) / float(HIDDEN_DIM)
        for i in range(HIDDEN_DIM)
    ]
    path.write_bytes(struct.pack(f"<{HIDDEN_DIM}f", *values))


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def compare_outputs(old_path: Path, new_path: Path) -> dict[str, float | int]:
    old = struct.unpack(f"<{HIDDEN_DIM}f", old_path.read_bytes())
    new = struct.unpack(f"<{HIDDEN_DIM}f", new_path.read_bytes())
    diffs = [abs(a - b) for a, b in zip(old, new)]
    max_index = max(range(len(diffs)), key=diffs.__getitem__)
    return {
        "count": len(diffs),
        "max_abs_diff": diffs[max_index],
        "max_index": max_index,
        "old_at_max": old[max_index],
        "new_at_max": new[max_index],
        "mean_abs_diff": sum(diffs) / len(diffs),
        "old0": old[0],
        "new0": new[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer against largerlm-runner on one real GLM MXFP4 MoE layer."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--experts", default=DEFAULT_EXPERTS)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--expert-buffer-count", type=int, default=8)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-infer-real-layer-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.f32"
    old_output = root / "old_runner_output.f32"
    new_output = root / "glm_moe_infer_output.f32"
    write_input(input_path)
    print(f"fixture: {root}")

    runner_env = os.environ.copy()
    runner_env["LARGERLM_MOE_DECODE_MXFP4_FUSED"] = "1"
    run_command(
        [
            str(args.runner),
            "--layout",
            str(args.prepared / "experts" / "layout.json"),
            "--layer",
            str(args.layer),
            "--run-moe",
            "--experts",
            args.experts,
            "--weights",
            args.weights,
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(old_output),
            "--max-slot-mib",
            "64",
            "--max-runner-scratch-mib",
            "512",
            "--expert-read-advise-merge-gap-kib",
            "0",
            "--expert-read-advise-align-kib",
            "4",
        ],
        env=runner_env,
    )

    completed = run_command(
        [
            str(args.binary),
            "--prepared",
            str(args.prepared),
            "--probe-layer-moe",
            "--probe-layer",
            str(args.layer),
            "--probe-experts",
            args.experts,
            "--probe-weights",
            args.weights,
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(new_output),
            "--expert-buffer-count",
            str(args.expert_buffer_count),
            "--max-live-working-set-mib",
            "256",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    probe = payload.get("probe_layer_moe") or {}
    if not payload.get("ok") or not probe.get("ok"):
        raise SystemExit("glm_moe_infer real layer probe did not report ok")
    route_count = csv_value_count(args.experts)
    expected_buffers = min(args.expert_buffer_count, route_count)
    if payload.get("expert_buffer_count") != expected_buffers:
        raise SystemExit(
            f"expected {expected_buffers} active expert buffers, got {payload.get('expert_buffer_count')}"
        )
    assert_layer_moe_read_telemetry(
        probe,
        route_count=route_count,
        active_expert_buffers=expected_buffers,
    )

    comparison = compare_outputs(old_output, new_output)
    print(
        json.dumps(
            {
                "comparison": comparison,
                "read_dispatch": {
                    "dispatches": probe["expert_read_dispatch_count"],
                    "tasks": probe["expert_read_task_count"],
                    "max_task_count": probe["expert_read_max_task_count"],
                    "max_worker_count": probe["expert_read_max_worker_count"],
                    "pool_dispatches": probe["expert_read_pool_dispatch_count"],
                    "serial_dispatches": probe["expert_read_serial_dispatch_count"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if comparison["max_abs_diff"] > args.max_diff:
        raise SystemExit(
            f"max_abs_diff {comparison['max_abs_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    print("  smoke result:       ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
