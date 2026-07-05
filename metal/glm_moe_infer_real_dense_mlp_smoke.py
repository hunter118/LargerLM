#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "artifacts" / "glm-5.2-mxfp4" / "largerlm-prepared"
RESIDENT_LAYOUT = PREPARED / "resident" / "layout.json"
EXPERT_LAYOUT = PREPARED / "experts" / "layout.json"
RUNNER = ROOT / "metal" / "largerlm-runner"
INFER = ROOT / "metal" / "glm_moe_infer"
HIDDEN = 6144


def write_input(path: Path) -> None:
    values = [
        math.sin(i * 0.013) * 0.25 + math.cos(i * 0.007) * 0.125
        for i in range(HIDDEN)
    ]
    path.write_bytes(struct.pack(f"<{HIDDEN}f", *values))


def run_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) != HIDDEN * 4:
        raise SystemExit(f"{path} bytes {len(data)} != {HIDDEN * 4}")
    return list(struct.unpack(f"<{HIDDEN}f", data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--max-diff", type=float, default=1e-5)
    parser.add_argument("--max-live-working-set-mib", type=float, default=512.0)
    args = parser.parse_args()

    if not RESIDENT_LAYOUT.exists() or not EXPERT_LAYOUT.exists():
        raise SystemExit(f"prepared GLM layout not found under {PREPARED}")
    root = Path(tempfile.mkdtemp(prefix="largerlm-real-dense-mlp-", dir="/private/tmp"))
    input_path = root / "input.f32"
    old_output = root / "old_runner_output.f32"
    new_output = root / "glm_moe_infer_output.f32"
    write_input(input_path)

    old_cmd = [
        str(RUNNER),
        "--resident-layout",
        str(RESIDENT_LAYOUT),
        "--layer",
        str(args.layer),
        "--run-dense-mlp-block",
        "--input-f32",
        str(input_path),
        "--rms-norm-eps",
        "1e-5",
        "--output-f32",
        str(old_output),
        "--max-resident-matrix-mib",
        "128",
        "--max-runner-scratch-mib",
        "512",
    ]
    new_cmd = [
        str(INFER),
        "--resident-layout",
        str(RESIDENT_LAYOUT),
        "--expert-layout",
        str(EXPERT_LAYOUT),
        "--probe-dense-mlp-block",
        "--probe-layer",
        str(args.layer),
        "--input-f32",
        str(input_path),
        "--output-f32",
        str(new_output),
        "--rms-norm-eps",
        "1e-5",
        "--max-live-working-set-mib",
        str(args.max_live_working_set_mib),
        "--json",
    ]

    print("old runner:")
    run_command(old_cmd)
    print("glm_moe_infer:")
    completed = run_command(new_cmd)
    payload = json.loads(completed.stdout)
    dense = payload.get("probe_dense_mlp_block") or {}
    if not payload.get("ok") or not dense.get("ok"):
        raise SystemExit("glm_moe_infer dense MLP JSON reports failure")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit(
            f"dense probe allocated expert buffers: {payload.get('expert_buffer_count')}"
        )

    old = read_f32(old_output)
    new = read_f32(new_output)
    diffs = [abs(a - b) for a, b in zip(old, new)]
    max_diff = max(diffs)
    max_index = diffs.index(max_diff)
    if max_diff > args.max_diff:
        raise SystemExit(
            f"max diff {max_diff:.9g} at {max_index} exceeds {args.max_diff:.9g}: "
            f"old={old[max_index]:.9g} new={new[max_index]:.9g}"
        )

    print(f"fixture: {root}")
    print(f"  layer:                 {args.layer}")
    print(f"  max diff:              {max_diff:.9g} at {max_index}")
    print(f"  expert buffers:        {payload['expert_buffer_count']}")
    print(f"  estimated live MiB:    {payload['estimated_live_working_set_bytes'] / 1048576.0:.3f}")
    print(f"  bytes read MiB:        {dense['bytes_read'] / 1048576.0:.3f}")
    print(f"  elapsed:               {dense['elapsed_seconds']:.6f} s")
    print(f"  dense MLP smoke:       ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
