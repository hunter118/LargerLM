#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

from mxfp4_layer_moe_smoke import HIDDEN_DIM, expected_output0, write_fixture


def run_expert_read_probe(binary: Path, root: Path) -> None:
    cmd = [
        str(binary),
        "--prepared",
        str(root),
        "--probe-expert-read",
        "--probe-layer",
        "1",
        "--probe-experts",
        "1,0",
        "--expert-buffer-count",
        "2",
        "--max-live-working-set-mib",
        "8",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    probe = payload.get("probe_expert_read") or {}
    if not payload.get("ok") or not probe.get("ok"):
        raise SystemExit("glm_moe_infer expert-read probe did not report ok")
    if probe.get("expert_read_dispatch_count") != 1:
        raise SystemExit(f"expected one expert read dispatch, got {probe}")
    if probe.get("expert_read_task_count") != 2:
        raise SystemExit(f"expected two expert read tasks, got {probe}")
    if probe.get("expert_read_max_task_count") != 2:
        raise SystemExit(f"expected max read batch of two tasks, got {probe}")
    if probe.get("expert_read_max_worker_count", 0) < 2:
        raise SystemExit(f"expected at least two persistent pread workers, got {probe}")
    if probe.get("expert_read_pool_dispatch_count") != 1:
        raise SystemExit(f"expected one pooled expert read dispatch, got {probe}")
    if probe.get("expert_read_serial_dispatch_count") != 0:
        raise SystemExit(f"expected no serial expert read dispatch, got {probe}")
    if len(probe.get("results") or []) != 2:
        raise SystemExit(f"expected two read checksums, got {probe}")
    print(
        "  direct expert read: "
        f"{probe['expert_read_task_count']} tasks via "
        f"{probe['expert_read_max_worker_count']} workers"
    )


def run_expert_read_rejects_oob(binary: Path, root: Path) -> None:
    cmd = [
        str(binary),
        "--prepared",
        str(root),
        "--probe-expert-read",
        "--probe-layer",
        "1",
        "--probe-experts",
        "2",
        "--expert-buffer-count",
        "1",
        "--max-live-working-set-mib",
        "8",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise SystemExit("glm_moe_infer accepted an out-of-range expert id")
    if "outside layer 1 expert count" not in combined:
        raise SystemExit(f"expected out-of-range expert error, got:\n{combined}")
    print("  direct expert read guard: rejected out-of-range expert")


def run_smoke(binary: Path, root: Path) -> None:
    output = root / "glm_moe_infer_output.f32"
    expected = expected_output0()
    cmd = [
        str(binary),
        "--prepared",
        str(root),
        "--probe-layer-moe",
        "--probe-layer",
        "1",
        "--probe-experts",
        "1,0",
        "--probe-weights",
        "0.6666666667,0.3333333333",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(output),
        "--expect-output0",
        f"{expected:.9f}",
        "--expert-buffer-count",
        "2",
        "--max-live-working-set-mib",
        "8",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    probe = payload.get("probe_layer_moe") or {}
    if not payload.get("ok") or not probe.get("ok"):
        raise SystemExit("glm_moe_infer layer MoE probe did not report ok")
    if probe.get("expert_read_dispatch_count") != 1:
        raise SystemExit(f"expected one expert read dispatch, got {probe}")
    if probe.get("expert_read_task_count") != 2:
        raise SystemExit(f"expected two expert read tasks, got {probe}")
    if probe.get("expert_read_max_task_count") != 2:
        raise SystemExit(f"expected max read batch of two tasks, got {probe}")
    if probe.get("expert_read_max_worker_count", 0) < 2:
        raise SystemExit(f"expected at least two persistent pread workers, got {probe}")
    if probe.get("expert_read_pool_dispatch_count") != 1:
        raise SystemExit(f"expected one pooled expert read dispatch, got {probe}")
    if probe.get("expert_read_serial_dispatch_count") != 0:
        raise SystemExit(f"expected no serial expert read dispatch, got {probe}")
    values = struct.unpack(f"<{HIDDEN_DIM}f", output.read_bytes())
    diffs = [abs(value - expected) for value in values]
    max_abs_diff = max(diffs)
    if max_abs_diff > 5e-3:
        raise SystemExit(
            f"max output diff {max_abs_diff:.6g} exceeds tolerance; expected {expected:.6f}"
        )
    print(f"  expected output[0]: {expected:.6f}")
    print(f"  max abs diff:       {max_abs_diff:.6g}")
    print(
        "  read dispatch:      "
        f"{probe['expert_read_task_count']} tasks via "
        f"{probe['expert_read_max_worker_count']} workers"
    )
    print("  smoke result:       ok")


def run_resident_cache_smoke(binary: Path, root: Path) -> None:
    layout = json.loads((root / "experts" / "layout.json").read_text())
    layer = layout["layers"][0]
    slot_bytes = int(layer["expert_slot_bytes"])
    plan = {
        "schema": "largerlm.expert_pin_plan.v1",
        "quality_preserving": True,
        "changes_routing": False,
        "max_pin_bytes": slot_bytes,
        "selected_bytes": slot_bytes,
        "selected_experts": [
            {
                "layer": 1,
                "expert": 1,
                "expert_slot_bytes": slot_bytes,
            }
        ],
        "memory_envelope": {
            "adaptive_evictable_expert_cache_gib": 0.01,
            "minimum_free_unified_memory_gib": 0,
        },
    }
    plan_path = root / "expert-pin-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    output = root / "glm_moe_infer_cached_output.f32"
    expected = expected_output0()
    cmd = [
        str(binary),
        "--prepared",
        str(root),
        "--expert-pin-plan",
        str(plan_path),
        "--probe-layer-moe",
        "--probe-layer",
        "1",
        "--probe-experts",
        "1,0,0",
        "--probe-weights",
        "0.6666666667,0.16666666665,0.16666666665",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(output),
        "--expect-output0",
        f"{expected:.9f}",
        "--expert-buffer-count",
        "1",
        "--max-live-working-set-mib",
        "8",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    probe = payload.get("probe_layer_moe") or {}
    cache = payload.get("expert_resident_cache") or {}
    if not payload.get("ok") or not probe.get("ok") or not cache.get("enabled"):
        raise SystemExit("expert resident cache smoke did not report ok")
    expected_cache = {
        "preload_count": 1,
        "entry_count": 2,
        "hit_count": 2,
        "miss_count": 1,
        "eviction_count": 0,
        "pressure_reject_count": 0,
    }
    for field, value in expected_cache.items():
        if cache.get(field) != value:
            raise SystemExit(f"unexpected cache {field}: {cache}")
    if probe.get("expert_read_task_count") != 1:
        raise SystemExit(f"expected only one SSD expert read after cache hits: {probe}")
    values = struct.unpack(f"<{HIDDEN_DIM}f", output.read_bytes())
    max_abs_diff = max(abs(value - expected) for value in values)
    if max_abs_diff > 5e-3:
        raise SystemExit(
            f"cached max output diff {max_abs_diff:.6g} exceeds tolerance"
        )
    print(
        "  expert cache:       "
        f"{cache['hit_count']} hits, {cache['miss_count']} miss, "
        f"{probe['expert_read_task_count']} SSD task"
    )
    print("  resident cache:     ok")


def run_adaptive_lru_smoke(binary: Path, root: Path) -> None:
    plan = {
        "schema": "largerlm.expert_pin_plan.v1",
        "quality_preserving": True,
        "changes_routing": False,
        "max_pin_bytes": 1,
        "selected_bytes": 0,
        "selected_experts": [],
        "memory_envelope": {
            "adaptive_evictable_expert_cache_gib": 2 / 1024,
            "minimum_free_unified_memory_gib": 0,
        },
    }
    plan_path = root / "expert-lru-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    output = root / "glm_moe_infer_lru_output.f32"
    expected = expected_output0()
    cmd = [
        str(binary),
        "--prepared",
        str(root),
        "--expert-pin-plan",
        str(plan_path),
        "--probe-layer-moe",
        "--probe-layer",
        "1",
        "--probe-experts",
        "0,1,1",
        "--probe-weights",
        "0.3333333333,0.33333333335,0.33333333335",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(output),
        "--expect-output0",
        f"{expected:.9f}",
        "--expert-buffer-count",
        "1",
        "--max-live-working-set-mib",
        "8",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="")
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    cache = payload.get("expert_resident_cache") or {}
    expected_cache = {
        "preload_count": 0,
        "entry_count": 1,
        "hit_count": 1,
        "miss_count": 2,
        "eviction_count": 1,
        "pressure_reject_count": 0,
    }
    for field, value in expected_cache.items():
        if cache.get(field) != value:
            raise SystemExit(f"unexpected adaptive LRU {field}: {cache}")
    values = struct.unpack(f"<{HIDDEN_DIM}f", output.read_bytes())
    if max(abs(value - expected) for value in values) > 5e-3:
        raise SystemExit("adaptive LRU changed MoE output")
    print(
        "  adaptive LRU:       "
        f"{cache['hit_count']} hit, {cache['miss_count']} misses, "
        f"{cache['eviction_count']} eviction"
    )


def run_memory_guard_smoke(binary: Path, root: Path) -> None:
    layout = json.loads((root / "experts" / "layout.json").read_text())
    slot_bytes = int(layout["layers"][0]["expert_slot_bytes"])
    plan = {
        "schema": "largerlm.expert_pin_plan.v1",
        "quality_preserving": True,
        "changes_routing": False,
        "max_pin_bytes": slot_bytes,
        "selected_bytes": slot_bytes,
        "selected_experts": [
            {"layer": 1, "expert": 1, "expert_slot_bytes": slot_bytes}
        ],
        "memory_envelope": {
            "adaptive_evictable_expert_cache_gib": 0,
            "minimum_free_unified_memory_gib": 1024,
        },
    }
    plan_path = root / "expert-impossible-memory-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    cmd = [
        str(binary),
        "--prepared",
        str(root),
        "--expert-pin-plan",
        str(plan_path),
        "--probe-layer-moe",
        "--probe-layer",
        "1",
        "--probe-experts",
        "1",
        "--probe-weights",
        "1",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(root / "must-not-run.f32"),
        "--expert-buffer-count",
        "1",
        "--max-live-working-set-mib",
        "8",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode == 0:
        raise SystemExit("expert cache accepted an impossible minimum-free guard")
    payload = json.loads(completed.stdout)
    if payload.get("admission_ok") or payload.get("available_unified_memory_ok"):
        raise SystemExit(f"memory guard did not reject expert preload: {payload}")
    if (root / "must-not-run.f32").exists():
        raise SystemExit("memory-guarded MoE unexpectedly wrote output")
    print("  memory guard:       rejected impossible 1024 GiB reserve")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run glm_moe_infer's tiny MXFP4 layer-MoE probe."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).with_name("glm_moe_infer"),
        help="Path to the compiled glm_moe_infer binary.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Fixture directory. Defaults to a new /private/tmp directory.",
    )
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-infer-moe-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    print(f"fixture: {root}")
    run_expert_read_probe(args.binary, root)
    run_expert_read_rejects_oob(args.binary, root)
    run_smoke(args.binary, root)
    run_resident_cache_smoke(args.binary, root)
    run_adaptive_lru_smoke(args.binary, root)
    run_memory_guard_smoke(args.binary, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
