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

from glm_moe_infer_real_layer_moe_smoke import assert_layer_moe_read_telemetry


HIDDEN_DIM = 6144


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


def max_abs_diff(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise SystemExit(f"length mismatch: {len(left)} != {len(right)}")
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def compare_router(old_path: Path, new_path: Path) -> dict[str, object]:
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    for key in (
        "router_score",
        "norm_topk_prob",
        "routed_scaling_factor",
        "n_group",
        "topk_group",
        "used_correction_bias",
    ):
        if old.get(key) != new.get(key):
            raise SystemExit(f"{key} mismatch: old={old.get(key)} new={new.get(key)}")
    if old.get("experts") != new.get("experts"):
        raise SystemExit(f"expert mismatch: old={old.get('experts')} new={new.get('experts')}")
    return {
        "experts": old["experts"],
        "max_weight_diff": max_abs_diff(old.get("weights", []), new.get("weights", [])),
        "max_logit_diff": max_abs_diff(old.get("logits", []), new.get("logits", [])),
        "weight_sum": sum(old.get("weights", [])),
    }


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
        description="Compare glm_moe_infer MLP block against largerlm-runner on one real GLM layer."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expert-buffer-count", type=int, default=None)
    parser.add_argument("--max-live-working-set-mib", type=float, default=None)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--include-shared-expert", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()
    expected_route_slots = args.top_k + (1 if args.include_shared_expert else 0)
    expert_buffer_count = args.expert_buffer_count or expected_route_slots
    max_live_working_set_mib = (
        args.max_live_working_set_mib
        if args.max_live_working_set_mib is not None
        else (768.0 if args.include_shared_expert else 256.0)
    )

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-infer-mlp-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.f32"
    old_router = root / "old_router.json"
    new_router = root / "new_router.json"
    old_output = root / "old_runner_output.f32"
    new_output = root / "glm_moe_infer_output.f32"
    write_input(input_path)
    print(f"fixture: {root}")

    runner_env = os.environ.copy()
    runner_env["LARGERLM_MOE_DECODE_MXFP4_FUSED"] = "1"
    old_cmd = [
            str(args.runner),
            "--layout",
            str(args.prepared / "experts" / "layout.json"),
            "--resident-layout",
            str(args.prepared / "resident" / "layout.json"),
            "--layer",
            str(args.layer),
            "--run-mlp-block",
            "--input-f32",
            str(input_path),
            "--top-k",
            str(args.top_k),
            "--max-k",
            str(args.top_k),
            "--rms-norm-eps",
            f"{args.rms_norm_eps:.9g}",
            "--output-f32",
            str(old_output),
            "--output-router-json",
            str(old_router),
            "--max-slot-mib",
            "64",
            "--max-router-mib",
            "8",
            "--max-runner-scratch-mib",
            "512",
            "--expert-read-advise-merge-gap-kib",
            "0",
            "--expert-read-advise-align-kib",
            "4",
    ]
    if args.include_shared_expert:
        old_cmd.append("--include-shared-expert")
    run_command(old_cmd, env=runner_env)

    new_cmd = [
            str(args.binary),
            "--prepared",
            str(args.prepared),
            "--probe-mlp-block",
            "--probe-layer",
            str(args.layer),
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(new_output),
            "--top-k",
            str(args.top_k),
            "--expert-buffer-count",
            str(expert_buffer_count),
            "--rms-norm-eps",
            f"{args.rms_norm_eps:.9g}",
            "--output-router-json",
            str(new_router),
            "--max-live-working-set-mib",
            f"{max_live_working_set_mib:.9g}",
            "--json",
    ]
    if args.include_shared_expert:
        new_cmd.append("--include-shared-expert")
    completed = run_command(new_cmd)
    payload = json.loads(completed.stdout)
    if not payload.get("ok"):
        raise SystemExit("glm_moe_infer MLP block probe did not report ok")
    expected_buffers = min(expert_buffer_count, expected_route_slots)
    if payload.get("expert_buffer_count") != expected_buffers:
        raise SystemExit(f"MLP block probe activated unexpected expert buffer count: {payload}")
    mlp = payload.get("probe_mlp_block") or {}
    if not mlp.get("ok") or bool(mlp.get("include_shared_expert")) != args.include_shared_expert:
        raise SystemExit("glm_moe_infer MLP block probe metadata is unexpected")
    probe = payload.get("probe_layer_moe") or {}
    if not probe.get("residual_add"):
        raise SystemExit("glm_moe_infer MLP block should report residual_add")
    assert_layer_moe_read_telemetry(
        probe,
        route_count=args.top_k,
        active_expert_buffers=expected_buffers,
    )

    router_comparison = compare_router(old_router, new_router)
    output_comparison = compare_outputs(old_output, new_output)
    print(
        json.dumps(
            {
                "router": router_comparison,
                "output": output_comparison,
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
    if router_comparison["max_weight_diff"] > args.max_diff:
        raise SystemExit(
            f"max_weight_diff {router_comparison['max_weight_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    if router_comparison["max_logit_diff"] > args.max_diff:
        raise SystemExit(
            f"max_logit_diff {router_comparison['max_logit_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    if output_comparison["max_abs_diff"] > args.max_diff:
        raise SystemExit(
            f"max_abs_diff {output_comparison['max_abs_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    print("  smoke result:       ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
