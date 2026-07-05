#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from glm_moe_infer_real_attn_projections_smoke import write_dummy_expert_layout
from resident_mxfp4_attention_smoke import HIDDEN_DIM, read_f32, write_fixture


def run_command(cmd: list[str]) -> str:
    import subprocess

    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Run glm_moe_infer attention-output tiny smoke.")
    parser.add_argument("--binary", type=Path, default=Path(__file__).with_name("glm_moe_infer"))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-attn-output-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    expert_layout = write_dummy_expert_layout(root)
    output = root / "output.f32"
    projection = root / "projection.f32"
    default_stdout = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--expert-layout",
            str(expert_layout),
            "--no-open-experts",
            "--probe-attn-output",
            "--probe-layer",
            "1",
            "--input-f32",
            str(root / "attn_input.f32"),
            "--residual-f32",
            str(root / "residual.f32"),
            "--projection-f32",
            str(projection),
            "--output-f32",
            str(output),
            "--max-live-working-set-mib",
            "8",
            "--json",
        ]
    )
    payload = json.loads(default_stdout)
    attn = payload.get("probe_attn_output") or {}
    if not payload.get("ok") or not attn.get("ok"):
        raise SystemExit(f"attention-output probe did not report ok: {payload}")
    if attn.get("resident_mmap_backed"):
        raise SystemExit(f"default attention-output unexpectedly used mmap backing: {attn}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("attention-output probe should not allocate expert buffers")
    if attn.get("out_dim") != HIDDEN_DIM or attn.get("in_dim") != HIDDEN_DIM:
        raise SystemExit(f"unexpected attention-output dims: {attn}")

    projected = read_f32(projection, HIDDEN_DIM)
    result = read_f32(output, HIDDEN_DIM)
    if abs(projected[0] - 1.0) > 2e-4:
        raise SystemExit(f"projection[0] {projected[0]:.6f} != expected 1.0")
    if abs(result[0] - 1.25) > 2e-4:
        raise SystemExit(f"output[0] {result[0]:.6f} != expected 1.25")

    fused_output = root / "fused_output.f32"
    fused_stdout = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--expert-layout",
            str(expert_layout),
            "--no-open-experts",
            "--probe-attn-output",
            "--probe-layer",
            "1",
            "--input-f32",
            str(root / "attn_input.f32"),
            "--residual-f32",
            str(root / "residual.f32"),
            "--output-f32",
            str(fused_output),
            "--max-live-working-set-mib",
            "8",
            "--json",
        ]
    )
    fused_payload = json.loads(fused_stdout)
    fused = fused_payload.get("probe_attn_output") or {}
    if not fused_payload.get("ok") or not fused.get("ok"):
        raise SystemExit(f"fused attention-output probe did not report ok: {fused_payload}")
    if fused.get("resident_mmap_backed"):
        raise SystemExit(f"default fused attention-output unexpectedly used mmap backing: {fused}")
    if fused.get("bytes_read") != 544:
        raise SystemExit(f"unexpected fused attention-output bytes_read: {fused}")

    mmap_output = root / "mmap_output.f32"
    mmap_stdout = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--expert-layout",
            str(expert_layout),
            "--no-open-experts",
            "--mmap-resident",
            "--wrap-resident-metal",
            "--probe-attn-output",
            "--probe-layer",
            "1",
            "--input-f32",
            str(root / "attn_input.f32"),
            "--residual-f32",
            str(root / "residual.f32"),
            "--output-f32",
            str(mmap_output),
            "--max-live-working-set-mib",
            "8",
            "--json",
        ]
    )
    mmap_payload = json.loads(mmap_stdout)
    mmap_attn = mmap_payload.get("probe_attn_output") or {}
    if not mmap_payload.get("ok") or not mmap_attn.get("ok"):
        raise SystemExit(f"mmap-backed attention-output probe did not report ok: {mmap_payload}")
    if not mmap_payload.get("resident_metal_wrapped") or not mmap_attn.get("resident_mmap_backed"):
        raise SystemExit(f"attention-output mmap-backed path was not used: {mmap_payload}")
    if mmap_attn.get("bytes_read") != 0:
        raise SystemExit(f"mmap-backed attention-output should not pread stage bytes: {mmap_attn}")
    if mmap_attn.get("scratch_bytes", 0) >= fused.get("scratch_bytes", 0):
        raise SystemExit(
            f"mmap-backed attention-output did not reduce scratch bytes: "
            f"default={fused.get('scratch_bytes')} mmap={mmap_attn.get('scratch_bytes')}"
        )

    fused_result = read_f32(fused_output, HIDDEN_DIM)
    mmap_result = read_f32(mmap_output, HIDDEN_DIM)
    for name, values in (("fused", fused_result), ("mmap", mmap_result)):
        if abs(values[0] - 1.25) > 2e-4:
            raise SystemExit(f"{name} output[0] {values[0]:.6f} != expected 1.25")
    max_mmap_diff = max(abs(a - b) for a, b in zip(fused_result, mmap_result))
    if max_mmap_diff > 2e-4:
        raise SystemExit(f"mmap attention-output max diff {max_mmap_diff:.9g} exceeds tolerance")
    print("  glm_moe_infer attention output smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
