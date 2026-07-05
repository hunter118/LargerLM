#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


FILES = (
    "attn_input_norm.f32",
    "attn_q_a.f32",
    "attn_q_a_norm.f32",
    "attn_q_b.f32",
    "attn_kv_a.f32",
    "attn_kv_a_norm.f32",
)


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def tensor_map(resident_layout: Path) -> dict[str, dict[str, object]]:
    layout = json.loads(resident_layout.read_text())
    return {tensor["name"]: tensor for tensor in layout["tensors"]}


def mxfp4_dims(tensors: dict[str, dict[str, object]], tensor_name: str) -> tuple[int, int, int]:
    weight = tensors[tensor_name]
    scale_name = tensor_name.removesuffix(".weight") + ".scales"
    scales = tensors[scale_name]
    out_dim = int(weight["shape"][0])
    in_dim = int(weight["shape"][1]) * 8
    group_size = in_dim // int(scales["shape"][1])
    return out_dim, in_dim, group_size


def vector_dim(tensors: dict[str, dict[str, object]], tensor_name: str) -> int:
    return int(tensors[tensor_name]["shape"][0])


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


def read_f32(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    if len(raw) % 4 != 0:
        raise SystemExit(f"{path} is not f32-aligned")
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def compare_file(old_path: Path, new_path: Path) -> dict[str, float | int | str]:
    old = read_f32(old_path)
    new = read_f32(new_path)
    if len(old) != len(new):
        raise SystemExit(f"{old_path.name} length mismatch: {len(old)} != {len(new)}")
    diffs = [abs(a - b) for a, b in zip(old, new)]
    max_index = max(range(len(diffs)), key=diffs.__getitem__) if diffs else 0
    return {
        "file": old_path.name,
        "count": len(diffs),
        "max_abs_diff": diffs[max_index] if diffs else 0.0,
        "max_index": max_index,
        "old_at_max": old[max_index] if diffs else 0.0,
        "new_at_max": new[max_index] if diffs else 0.0,
        "old0": old[0] if old else 0.0,
        "new0": new[0] if new else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare glm_moe_infer attention projections against largerlm-runner on one real GLM layer."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layer", type=int, default=67)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--max-diff", type=float, default=1e-6)
    args = parser.parse_args()

    resident_layout = args.prepared / "resident" / "layout.json"
    if not resident_layout.exists():
        print(f"prepared resident layout missing, skipping: {resident_layout}")
        return 0

    tensors = tensor_map(resident_layout)
    prefix = f"model.layers.{args.layer}"
    q_a_out, hidden_dim, q_group = mxfp4_dims(tensors, f"{prefix}.self_attn.q_a_proj.weight")
    q_b_out, q_b_in, q_b_group = mxfp4_dims(tensors, f"{prefix}.self_attn.q_b_proj.weight")
    kv_a_out, kv_a_in, kv_a_group = mxfp4_dims(
        tensors,
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
    )
    kv_lora_dim = vector_dim(tensors, f"{prefix}.self_attn.kv_a_layernorm.weight")
    if q_b_in != q_a_out or kv_a_in != hidden_dim:
        raise SystemExit("real GLM attention projection dims are inconsistent")

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-attn-proj-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    dummy_expert_layout = write_dummy_expert_layout(root)
    input_path = root / "input.f32"
    old_dir = root / "old"
    new_dir = root / "new"
    write_input(input_path, hidden_dim)
    print(f"fixture: {root}")

    run_command(
        [
            str(args.runner),
            "--resident-layout",
            str(resident_layout),
            "--layer",
            str(args.layer),
            "--run-attn-projections",
            "--input-f32",
            str(input_path),
            "--rms-norm-eps",
            f"{args.rms_norm_eps:.9g}",
            "--output-dir",
            str(old_dir),
            "--max-resident-matrix-mib",
            "64",
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
            str(dummy_expert_layout),
            "--no-open-experts",
            "--probe-attn-projections",
            "--probe-layer",
            str(args.layer),
            "--input-f32",
            str(input_path),
            "--rms-norm-eps",
            f"{args.rms_norm_eps:.9g}",
            "--output-dir",
            str(new_dir),
            "--max-live-working-set-mib",
            "64",
            "--json",
        ]
    )
    payload = json.loads(completed.stdout)
    attn = payload.get("probe_attn_projections") or {}
    if not payload.get("ok") or not attn.get("ok"):
        raise SystemExit(f"glm_moe_infer attention projection probe did not report ok: {payload}")
    if payload.get("expert_buffer_count") != 0:
        raise SystemExit("attention-only probe should not allocate expert buffers")
    if attn.get("hidden_dim") != hidden_dim or attn.get("q_lora_dim") != q_a_out:
        raise SystemExit(f"unexpected q dims: {attn}")
    if attn.get("q_out_dim") != q_b_out or attn.get("kv_a_out_dim") != kv_a_out:
        raise SystemExit(f"unexpected output dims: {attn}")
    if attn.get("kv_lora_dim") != kv_lora_dim:
        raise SystemExit(f"unexpected kv lora dim: {attn}")

    files = list(FILES)
    if (old_dir / "attn_kv_b.f32").exists() or (new_dir / "attn_kv_b.f32").exists():
        files.append("attn_kv_b.f32")
    comparisons = [compare_file(old_dir / name, new_dir / name) for name in files]
    print(
        json.dumps(
            {
                "comparison": comparisons,
                "dims": {
                    "hidden_dim": hidden_dim,
                    "q_a_out": q_a_out,
                    "q_b_out": q_b_out,
                    "kv_a_out": kv_a_out,
                    "kv_lora_dim": kv_lora_dim,
                    "groups": {
                        "q_a": q_group,
                        "q_b": q_b_group,
                        "kv_a": kv_a_group,
                    },
                    "estimated_live_working_set_bytes": payload["estimated_live_working_set_bytes"],
                    "bytes_read": attn["bytes_read"],
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
    print("  real attention projections smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
