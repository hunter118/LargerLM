#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path

from resident_mxfp4_linear_smoke import OUT_DIM, write_fixture


TENSOR_NAME = "model.layers.1.self_attn.q_mxfp4_proj.weight"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def write_dummy_expert_layout(root: Path) -> Path:
    experts = root / "experts"
    experts.mkdir(parents=True, exist_ok=True)
    (experts / "layer_001.bin").write_bytes(b"\0")
    layout = experts / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "quantization": "mlx-mxfp4",
                "group_size": 32,
                "num_layers": 2,
                "num_experts": 1,
                "component_order": [],
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 1,
                        "expert_slot_bytes": 1,
                        "layer_file": "layer_001.bin",
                        "components": [],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return layout


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-resident-linear-", dir="/private/tmp"))
    write_fixture(root)
    expert_layout = write_dummy_expert_layout(root)
    runner = Path(__file__).with_name("largerlm-runner")
    infer = Path(__file__).with_name("glm_moe_infer")
    old_out = root / "old_out.f32"
    new_out = root / "new_out.f32"
    mmap_out = root / "mmap_out.f32"

    run(
        [
            str(runner),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--layer",
            "1",
            "--run-resident-linear",
            "--tensor-suffix",
            ".self_attn.q_mxfp4_proj.weight",
            "--input-f32",
            str(root / "input.f32"),
            "--output-f32",
            str(old_out),
            "--max-resident-matrix-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
        ]
    )

    completed = run(
        [
            str(infer),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--expert-layout",
            str(expert_layout),
            "--no-open-experts",
            "--probe-resident-linear",
            "--resident-tensor-name",
            TENSOR_NAME,
            "--input-f32",
            str(root / "input.f32"),
            "--output-f32",
            str(new_out),
            "--max-live-working-set-mib",
            "8",
            "--expect-output0",
            "1",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    linear = payload["probe_resident_linear"]
    if not payload["ok"] or not linear["ok"]:
        raise SystemExit(f"glm_moe_infer resident linear failed: {payload}")
    if linear.get("resident_mmap_backed"):
        raise SystemExit(f"default resident linear unexpectedly used mmap backing: {linear}")
    if payload["expert_buffer_count"] != 0:
        raise SystemExit(f"resident-only probe allocated expert buffers: {payload}")
    if payload["estimated_live_working_set_bytes"] > 8 * 1024 * 1024:
        raise SystemExit(f"resident linear exceeded live cap: {payload}")
    if linear["bytes_read"] != 544:
        raise SystemExit(f"unexpected resident bytes read: {linear}")
    if linear["out_dim"] != OUT_DIM or linear["in_dim"] != 32 or linear["group_size"] != 32:
        raise SystemExit(f"unexpected resident linear dims: {linear}")

    completed_mmap = run(
        [
            str(infer),
            "--resident-layout",
            str(root / "resident" / "layout.json"),
            "--expert-layout",
            str(expert_layout),
            "--no-open-experts",
            "--mmap-resident",
            "--wrap-resident-metal",
            "--probe-resident-linear",
            "--resident-tensor-name",
            TENSOR_NAME,
            "--input-f32",
            str(root / "input.f32"),
            "--output-f32",
            str(mmap_out),
            "--max-live-working-set-mib",
            "8",
            "--expect-output0",
            "1",
            "--json",
        ]
    )
    payload_mmap = json.loads(completed_mmap.stdout)
    linear_mmap = payload_mmap["probe_resident_linear"]
    if not payload_mmap["ok"] or not linear_mmap["ok"]:
        raise SystemExit(f"glm_moe_infer mmap resident linear failed: {payload_mmap}")
    if not payload_mmap["resident_metal_wrapped"] or not linear_mmap["resident_mmap_backed"]:
        raise SystemExit(f"resident mmap-backed path was not used: {payload_mmap}")
    if linear_mmap["bytes_read"] != 0:
        raise SystemExit(f"mmap-backed resident linear should not pread stage bytes: {linear_mmap}")
    if linear_mmap["scratch_bytes"] >= linear["scratch_bytes"]:
        raise SystemExit(
            f"mmap-backed resident linear did not reduce scratch bytes: "
            f"default={linear['scratch_bytes']} mmap={linear_mmap['scratch_bytes']}"
        )

    old_values = read_f32(old_out, OUT_DIM)
    new_values = read_f32(new_out, OUT_DIM)
    mmap_values = read_f32(mmap_out, OUT_DIM)
    diffs = [abs(a - b) for a, b in zip(old_values, new_values)]
    mmap_diffs = [abs(a - b) for a, b in zip(old_values, mmap_values)]
    max_diff = max(diffs)
    max_mmap_diff = max(mmap_diffs)
    if max(max_diff, max_mmap_diff) > 2e-4:
        index = diffs.index(max_diff) if max_diff >= max_mmap_diff else mmap_diffs.index(max_mmap_diff)
        raise SystemExit(
            f"resident linear mismatch at {index}: old={old_values[index]} "
            f"new={new_values[index]} mmap={mmap_values[index]} "
            f"diff={max_diff} mmap_diff={max_mmap_diff}"
        )
    print(f"fixture: {root}")
    print(f"  glm_moe_infer resident MXFP4 linear max_diff={max_diff:.9g}")
    print(f"  glm_moe_infer resident mmap MXFP4 linear max_diff={max_mmap_diff:.9g}")
    print("  glm_moe_infer resident MXFP4 linear: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
