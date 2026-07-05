#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def find_mxfp4_dims(resident_layout: Path, tensor_name: str) -> tuple[int, int, int, int]:
    layout = json.loads(resident_layout.read_text())
    tensors = {tensor["name"]: tensor for tensor in layout["tensors"]}
    weight = tensors[tensor_name]
    scale_name = tensor_name.removesuffix(".weight") + ".scales"
    scales = tensors[scale_name]
    out_dim = int(weight["shape"][0])
    in_dim = int(weight["shape"][1]) * 8
    groups = int(scales["shape"][1])
    group_size = in_dim // groups
    total_bytes = int(weight["size"]) + int(scales["size"])
    return out_dim, in_dim, group_size, total_bytes


def write_input(path: Path, dim: int) -> None:
    values = [
        (math.sin(i * 0.013) + math.cos(i * 0.007)) / float(dim)
        for i in range(dim)
    ]
    path.write_bytes(struct.pack(f"<{dim}f", *values))


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
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer against largerlm-runner on one real GLM resident MXFP4 projection."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--suffix", default=".self_attn.q_a_proj.weight")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    resident_layout = args.prepared / "resident" / "layout.json"
    if not resident_layout.exists():
        print(f"prepared resident layout missing, skipping: {resident_layout}")
        return 0

    tensor_name = f"model.layers.{args.layer}{args.suffix}"
    out_dim, in_dim, group_size, total_bytes = find_mxfp4_dims(resident_layout, tensor_name)
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-real-resident-linear-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    dummy_expert_layout = write_dummy_expert_layout(root)
    input_path = root / "input.f32"
    old_output = root / "old_runner_output.f32"
    new_output = root / "glm_moe_infer_output.f32"
    write_input(input_path, in_dim)
    print(f"fixture: {root}")

    run_command(
        [
            str(args.runner),
            "--resident-layout",
            str(resident_layout),
            "--layer",
            str(args.layer),
            "--run-resident-linear",
            "--tensor-suffix",
            args.suffix,
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(old_output),
            "--max-resident-matrix-mib",
            "16",
            "--max-runner-scratch-mib",
            "64",
        ]
    )

    completed = run_command(
        [
            str(args.binary),
            "--resident-layout",
            str(resident_layout),
            "--expert-layout",
            str(dummy_expert_layout),
            "--no-open-experts",
            "--probe-resident-linear",
            "--resident-tensor-name",
            tensor_name,
            "--input-f32",
            str(input_path),
            "--output-f32",
            str(new_output),
            "--max-live-working-set-mib",
            "16",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    linear = payload.get("probe_resident_linear") or {}
    if not payload.get("ok") or not linear.get("ok"):
        raise SystemExit("glm_moe_infer real resident-linear probe did not report ok")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("resident-only real probe should not allocate expert buffers")
    if linear.get("out_dim") != out_dim or linear.get("in_dim") != in_dim:
        raise SystemExit(f"unexpected output/input dims: {linear}")
    if linear.get("group_size") != group_size or linear.get("bytes_read") != total_bytes:
        raise SystemExit(f"unexpected group/bytes metadata: {linear}")

    old_values = read_f32(old_output, out_dim)
    new_values = read_f32(new_output, out_dim)
    diffs = [abs(a - b) for a, b in zip(old_values, new_values)]
    max_diff = max(diffs)
    max_index = diffs.index(max_diff)
    comparison = {
        "count": out_dim,
        "tensor": tensor_name,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "group_size": group_size,
        "bytes_read": total_bytes,
        "estimated_live_working_set_bytes": payload["estimated_live_working_set_bytes"],
        "max_abs_diff": max_diff,
        "max_index": max_index,
        "old_at_max": old_values[max_index],
        "new_at_max": new_values[max_index],
        "old0": old_values[0],
        "new0": new_values[0],
    }
    print(json.dumps({"comparison": comparison}, indent=2, sort_keys=True))
    if max_diff > args.max_diff:
        raise SystemExit(f"max_abs_diff {max_diff:.9g} exceeds {args.max_diff:.9g}")
    print("  real resident MXFP4 linear smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
