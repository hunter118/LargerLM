#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_tensor(layout: dict[str, Any], *, layer: int, suffix: str) -> dict[str, Any]:
    target = f"model.layers.{layer}{suffix}"
    for tensor in layout.get("tensors", []):
        if isinstance(tensor, dict) and tensor.get("name") == target:
            return tensor
    raise SystemExit(f"tensor {target} not found")


def _parse_group32_modes(value: str) -> tuple[str, ...]:
    modes: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"auto", "default"}:
            modes.append("auto")
        elif item in {"0", "off", "false", "no", "scalar"}:
            modes.append("off")
        elif item in {"1", "on", "true", "yes", "group32"}:
            modes.append("on")
        else:
            raise argparse.ArgumentTypeError(
                "group32 modes must be auto, off, or on"
            )
    if not modes:
        raise argparse.ArgumentTypeError("group32 mode list must not be empty")
    return tuple(modes)


def _write_f32(path: Path, *, count: int, seed: int, scale: float) -> None:
    rng = random.Random(seed)
    values = array.array("f", (rng.uniform(-scale, scale) for _ in range(count)))
    with path.open("wb") as handle:
        values.tofile(handle)


def _read_f32(path: Path, count: int) -> array.array[float]:
    values = array.array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, count)
    if len(values) != count:
        raise SystemExit(f"{path} had {len(values)} f32 values, expected {count}")
    return values


def _max_abs_diff(lhs: array.array[float], rhs: array.array[float]) -> float:
    return max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)


def _series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _parse_runner_stdout(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("timing backend:"):
            parsed["timing_backend_seconds"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("MXFP4 group32 path:"):
            parsed["mxfp4_group32_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("estimated peak:"):
            parsed["estimated_peak_bytes"] = int(line.split(":", 1)[1].strip())
    return parsed


def _run_one(
    *,
    runner: Path,
    resident_layout: Path,
    layer: int,
    input_f32: Path,
    output_f32: Path,
    batch_tokens: int,
    group32_mode: str,
    max_resident_matrix_mib: int,
    max_runner_scratch_mib: int,
) -> dict[str, Any]:
    env = dict(os.environ)
    if group32_mode == "auto":
        env["LARGERLM_SHARED_MXFP4_GROUP32_SPECIALIZED"] = "auto"
    elif group32_mode == "off":
        env["LARGERLM_SHARED_MXFP4_GROUP32_SPECIALIZED"] = "0"
    else:
        env["LARGERLM_SHARED_MXFP4_GROUP32_SPECIALIZED"] = "1"
    command = [
        str(runner),
        "--resident-layout",
        str(resident_layout),
        "--layer",
        str(layer),
        "--run-shared-expert-batch",
        "--input-f32",
        str(input_f32),
        "--batch-tokens",
        str(batch_tokens),
        "--output-f32",
        str(output_f32),
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(
            f"runner failed for group32={group32_mode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    parsed = _parse_runner_stdout(result.stdout)
    parsed.update({"elapsed_wall_seconds": elapsed, "stderr": result.stderr})
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare resident MXFP4 group32 kernels on GLM shared experts.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument(
        "--group32-modes",
        type=_parse_group32_modes,
        default="off,auto",
        help="Comma-separated modes: off, auto, on.",
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--order", choices=("interleave", "grouped"), default="interleave")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--max-resident-matrix-mib", type=int, default=16)
    parser.add_argument("--max-runner-scratch-mib", type=int, default=256)
    parser.add_argument("--write-result", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    if args.batch_tokens <= 0:
        raise SystemExit("--batch-tokens must be positive")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.max_resident_matrix_mib <= 0 or args.max_runner_scratch_mib <= 0:
        raise SystemExit("memory limits must be positive")

    resident_layout = args.prepared_dir / "resident" / "layout.json"
    layout = _load_json(resident_layout)
    gate = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".mlp.shared_experts.gate_proj.weight",
    )
    up = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".mlp.shared_experts.up_proj.weight",
    )
    down = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".mlp.shared_experts.down_proj.weight",
    )
    gate_scales = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".mlp.shared_experts.gate_proj.scales",
    )
    up_scales = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".mlp.shared_experts.up_proj.scales",
    )
    down_scales = _find_tensor(
        layout,
        layer=args.layer,
        suffix=".mlp.shared_experts.down_proj.scales",
    )
    tensors = (gate, up, down)
    scales = (gate_scales, up_scales, down_scales)
    if any(tensor.get("dtype") != "U32" for tensor in tensors) or any(
        tensor.get("dtype") != "U8" for tensor in scales
    ):
        raise SystemExit("this sweep expects MLX MXFP4 shared expert tensors")
    gate_shape = gate.get("shape")
    up_shape = up.get("shape")
    down_shape = down.get("shape")
    if not (
        isinstance(gate_shape, list)
        and isinstance(up_shape, list)
        and isinstance(down_shape, list)
        and len(gate_shape) == len(up_shape) == len(down_shape) == 2
    ):
        raise SystemExit("unexpected shared expert shapes")
    hidden_dim = int(gate_shape[1]) * 8
    intermediate_dim = int(gate_shape[0])
    if int(up_shape[0]) != intermediate_dim or int(up_shape[1]) * 8 != hidden_dim:
        raise SystemExit("shared gate/up shapes differ")
    if int(down_shape[0]) != hidden_dim or int(down_shape[1]) * 8 != intermediate_dim:
        raise SystemExit("shared down shape is inconsistent")

    group_sizes: dict[str, int] = {}
    for name, weight, scale in (
        ("gate", gate, gate_scales),
        ("up", up, up_scales),
        ("down", down, down_scales),
    ):
        scale_shape = scale.get("shape")
        weight_shape = weight.get("shape")
        if not (
            isinstance(scale_shape, list)
            and len(scale_shape) == 2
            and isinstance(weight_shape, list)
            and len(weight_shape) == 2
        ):
            raise SystemExit(f"unexpected {name} scale shape")
        logical_in = int(weight_shape[1]) * 8
        groups = int(scale_shape[1])
        if groups <= 0 or logical_in % groups != 0:
            raise SystemExit(f"could not infer {name} group size")
        group_sizes[name] = logical_in // groups

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-shared-mxfp4-group32-"))
    cleaned = False
    try:
        input_f32 = work_dir / "hidden.f32"
        _write_f32(
            input_f32,
            count=args.batch_tokens * hidden_dim,
            seed=args.seed,
            scale=args.input_scale,
        )
        schedule: list[tuple[str, int]] = []
        if args.order == "grouped":
            for mode in args.group32_modes:
                for index in range(args.repeat):
                    schedule.append((mode, index))
        else:
            for index in range(args.repeat):
                for mode in args.group32_modes:
                    schedule.append((mode, index))

        output_count = args.batch_tokens * hidden_dim
        baseline_output: array.array[float] | None = None
        records: list[dict[str, Any]] = []
        for group32_mode, repeat_index in schedule:
            output_f32 = work_dir / f"output_{group32_mode}_rep{repeat_index:02d}.f32"
            parsed = _run_one(
                runner=args.runner,
                resident_layout=resident_layout,
                layer=args.layer,
                input_f32=input_f32,
                output_f32=output_f32,
                batch_tokens=args.batch_tokens,
                group32_mode=group32_mode,
                max_resident_matrix_mib=args.max_resident_matrix_mib,
                max_runner_scratch_mib=args.max_runner_scratch_mib,
            )
            output = _read_f32(output_f32, output_count)
            if baseline_output is None:
                baseline_output = output
                output_diff = 0.0
            else:
                output_diff = _max_abs_diff(baseline_output, output)
            records.append(
                {
                    "group32_mode": group32_mode,
                    "repeat_index": repeat_index,
                    "mxfp4_group32_path": parsed.get("mxfp4_group32_path"),
                    "timing_backend_seconds": parsed.get("timing_backend_seconds"),
                    "elapsed_wall_seconds": parsed["elapsed_wall_seconds"],
                    "estimated_peak_bytes": parsed.get("estimated_peak_bytes"),
                    "max_abs_diff_output_vs_first": output_diff,
                }
            )

        by_mode: dict[str, dict[str, Any]] = {}
        for mode in args.group32_modes:
            mode_records = [record for record in records if record["group32_mode"] == mode]
            by_mode[mode] = {
                "timing_backend_seconds": _series(
                    [
                        float(record["timing_backend_seconds"])
                        for record in mode_records
                        if record["timing_backend_seconds"] is not None
                    ]
                ),
                "elapsed_wall_seconds": _series(
                    [float(record["elapsed_wall_seconds"]) for record in mode_records]
                ),
                "max_abs_diff_output_vs_first": max(
                    (float(record["max_abs_diff_output_vs_first"]) for record in mode_records),
                    default=0.0,
                ),
                "observed_group32_paths": sorted(
                    {
                        str(record["mxfp4_group32_path"])
                        for record in mode_records
                        if record["mxfp4_group32_path"] is not None
                    }
                ),
            }

        payload = {
            "schema": "largerlm.glm_shared_mxfp4_group32_sweep.v1",
            "prepared_dir": args.prepared_dir,
            "runner": args.runner,
            "resident_layout": resident_layout,
            "work_dir": work_dir,
            "layer": args.layer,
            "batch_tokens": args.batch_tokens,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "group_sizes": group_sizes,
            "group32_modes": args.group32_modes,
            "repeat": args.repeat,
            "records": records,
            "by_mode": by_mode,
            "work_dir_cleaned": False,
        }
        if not args.keep_work_dir:
            shutil.rmtree(work_dir)
            cleaned = True
            payload["work_dir_cleaned"] = True
        if args.write_result:
            args.write_result.parent.mkdir(parents=True, exist_ok=True)
            args.write_result.write_text(
                json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
        return 0
    finally:
        if not args.keep_work_dir and not cleaned and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
