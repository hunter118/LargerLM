#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 6144


def write_input(path: Path) -> None:
    values = [
        (math.sin(i * 0.013) + math.cos(i * 0.007)) / float(HIDDEN_DIM)
        for i in range(HIDDEN_DIM)
    ]
    path.write_bytes(struct.pack(f"<{HIDDEN_DIM}f", *values))


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
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
    old_experts = old.get("experts")
    new_experts = new.get("experts")
    if old_experts != new_experts:
        raise SystemExit(f"expert mismatch: old={old_experts} new={new_experts}")
    weight_diff = max_abs_diff(old.get("weights", []), new.get("weights", []))
    logit_diff = max_abs_diff(old.get("logits", []), new.get("logits", []))
    return {
        "experts": old_experts,
        "router_score": old.get("router_score"),
        "routed_scaling_factor": old.get("routed_scaling_factor"),
        "used_correction_bias": old.get("used_correction_bias"),
        "max_weight_diff": weight_diff,
        "max_logit_diff": logit_diff,
        "old_weight_sum": sum(old.get("weights", [])),
        "new_weight_sum": sum(new.get("weights", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer against largerlm-runner on one real GLM production router."
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
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-infer-router-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.f32"
    old_router = root / "old_router.json"
    new_router = root / "new_router.json"
    write_input(input_path)
    print(f"fixture: {root}")

    run_command(
        [
            str(args.runner),
            "--resident-layout",
            str(args.prepared / "resident" / "layout.json"),
            "--layer",
            str(args.layer),
            "--run-router",
            "--input-f32",
            str(input_path),
            "--top-k",
            str(args.top_k),
            "--output-router-json",
            str(old_router),
            "--max-router-mib",
            "8",
            "--max-runner-scratch-mib",
            "64",
        ]
    )

    completed = run_command(
        [
            str(args.binary),
            "--prepared",
            str(args.prepared),
            "--probe-router",
            "--probe-layer",
            str(args.layer),
            "--input-f32",
            str(input_path),
            "--top-k",
            str(args.top_k),
            "--output-router-json",
            str(new_router),
            "--max-live-working-set-mib",
            "64",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    if not payload.get("ok") or not (payload.get("probe_router") or {}).get("ok"):
        raise SystemExit("glm_moe_infer router probe did not report ok")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("router-only probe should not allocate expert buffers")

    comparison = compare_router(old_router, new_router)
    print(json.dumps({"comparison": comparison}, indent=2, sort_keys=True))
    if comparison["max_weight_diff"] > args.max_diff:
        raise SystemExit(
            f"max_weight_diff {comparison['max_weight_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    if comparison["max_logit_diff"] > args.max_diff:
        raise SystemExit(
            f"max_logit_diff {comparison['max_logit_diff']:.9g} exceeds {args.max_diff:.9g}"
        )
    print("  smoke result:       ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
