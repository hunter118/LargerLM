#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from glm_moe_infer_real_attn_projections_smoke import (
    mxfp4_dims,
    tensor_map,
)
from glm_moe_infer_real_mla_attention_smoke import compare_output, read_f32, run_command


def run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def compare_named(name: str, old_path: Path, new_path: Path) -> dict[str, float | int | str]:
    result = compare_output(old_path, new_path)
    result["file"] = name
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer attention output against largerlm-runner on a real GLM layer."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--position", type=int, default=19)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--rope-interleave", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    resident_layout = args.prepared / "resident" / "layout.json"
    if not resident_layout.exists():
        print(f"prepared resident layout missing, skipping: {resident_layout}")
        return 0

    tensors = tensor_map(resident_layout)
    prefix = f"model.layers.{args.layer}"
    out_dim, value_dim, group_size = mxfp4_dims(tensors, f"{prefix}.self_attn.o_proj.weight")

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-attn-out-real-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"fixture: {root}")

    mla_cmd = [
        sys.executable,
        str(Path(__file__).with_name("glm_moe_infer_real_mla_attention_smoke.py")),
        "--prepared",
        str(args.prepared),
        "--runner",
        str(args.runner),
        "--binary",
        str(args.binary),
        "--layer",
        str(args.layer),
        "--position",
        str(args.position),
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--root",
        str(root),
        "--max-diff",
        f"{args.max_diff:.9g}",
    ]
    if args.rope_interleave:
        mla_cmd.append("--rope-interleave")
    run_checked(mla_cmd)

    mla_value = root / "new_mla.f32"
    residual = root / "input.f32"
    if len(read_f32(mla_value)) != value_dim:
        raise SystemExit("MLA value length does not match o_proj input dim")
    if len(read_f32(residual)) != out_dim:
        raise SystemExit("residual length does not match o_proj output dim")

    old_out = root / "old_attn_output.f32"
    old_projection = root / "old_attn_projection.f32"
    new_out = root / "new_attn_output.f32"
    new_projection = root / "new_attn_projection.f32"

    run_command(
        [
            str(args.runner),
            "--resident-layout",
            str(resident_layout),
            "--layer",
            str(args.layer),
            "--run-attn-output",
            "--input-f32",
            str(mla_value),
            "--residual-f32",
            str(residual),
            "--projection-f32",
            str(old_projection),
            "--output-f32",
            str(old_out),
            "--max-resident-matrix-mib",
            "128",
            "--max-runner-scratch-mib",
            "256",
        ]
    )

    completed = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(resident_layout),
            "--expert-layout",
            str(root / "experts" / "layout.json"),
            "--no-open-experts",
            "--probe-attn-output",
            "--probe-layer",
            str(args.layer),
            "--input-f32",
            str(mla_value),
            "--residual-f32",
            str(residual),
            "--projection-f32",
            str(new_projection),
            "--output-f32",
            str(new_out),
            "--max-live-working-set-mib",
            "192",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    attn_out = payload.get("probe_attn_output") or {}
    if not payload.get("ok") or not attn_out.get("ok"):
        raise SystemExit(f"glm_moe_infer attention-output probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("attention-output probe should not allocate expert buffers")
    if attn_out.get("out_dim") != out_dim or attn_out.get("in_dim") != value_dim:
        raise SystemExit(f"unexpected attention-output dims: {attn_out}")

    comparisons = [
        compare_named("projection", old_projection, new_projection),
        compare_named("output", old_out, new_out),
    ]
    print(
        json.dumps(
            {
                "comparison": comparisons,
                "dims": {
                    "out_dim": out_dim,
                    "value_dim": value_dim,
                    "group_size": group_size,
                    "estimated_live_working_set_bytes": payload["estimated_live_working_set_bytes"],
                    "scratch_bytes": attn_out["scratch_bytes"],
                    "bytes_read": attn_out["bytes_read"],
                    "projection_kernel_seconds": attn_out["projection_kernel_seconds"],
                    "residual_add_seconds": attn_out["residual_add_seconds"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    worst = max(comparisons, key=lambda item: float(item["max_abs_diff"]))
    if float(worst["max_abs_diff"]) > args.max_diff:
        raise SystemExit(
            f"{worst['file']} max_abs_diff {worst['max_abs_diff']:.9g} "
            f"exceeds {args.max_diff:.9g}"
        )
    print("  real attention output smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
