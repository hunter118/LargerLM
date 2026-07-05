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
from typing import Any


DEFAULT_PREPARED = Path("artifacts/glm-5.2-mxfp4/largerlm-prepared")


def _write_input(path: Path, hidden_dim: int, mode: str) -> None:
    if mode == "zeros":
        values = [0.0] * hidden_dim
    elif mode == "ones":
        values = [1.0 / float(hidden_dim)] * hidden_dim
    else:
        values = [
            math.sin((i + 1) * 0.013) * 0.01
            for i in range(hidden_dim)
        ]
    path.write_bytes(struct.pack(f"<{hidden_dim}f", *values))


def _parse_stdout(stdout: str) -> dict[str, Any]:
    result: dict[str, Any] = {"timing": {}}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("MXFP4 fused decode:"):
            result["mxfp4_fused_decode"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("top-k:"):
            raw = stripped.split(":", 1)[1].strip()
            result["top_k"] = [int(item) for item in raw.split(",") if item]
        elif stripped.startswith("weights:"):
            raw = stripped.split(":", 1)[1].strip()
            result["weights"] = [float(item) for item in raw.split(",") if item]
        elif stripped.startswith("output[0]:"):
            result["output0"] = float(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("timing "):
            key, raw = stripped.removeprefix("timing ").split(":", 1)
            result["timing"][key.strip().replace(" ", "_")] = float(raw.strip())
    return result


def _read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path} byte size is not divisible by 4")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def _max_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _run_case(
    *,
    name: str,
    runner: Path,
    prepared: Path,
    layer: int,
    top_k: int,
    input_path: Path,
    output_path: Path,
    router_path: Path,
    max_runner_scratch_mib: int,
    include_shared: bool,
    fused: bool,
) -> dict[str, Any]:
    cmd = [
        str(runner),
        "--layout",
        str(prepared / "experts" / "layout.json"),
        "--resident-layout",
        str(prepared / "resident" / "layout.json"),
        "--layer",
        str(layer),
        "--run-layer-moe",
        "--input-f32",
        str(input_path),
        "--top-k",
        str(top_k),
        "--max-k",
        str(top_k),
        "--router-score",
        "raw",
        "--routed-scaling-factor",
        "1",
        "--output-f32",
        str(output_path),
        "--output-router-json",
        str(router_path),
        "--max-slot-mib",
        "32",
        "--max-router-mib",
        "8",
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
        "--expert-read-advise-merge-gap-kib",
        "0",
        "--expert-read-advise-align-kib",
        "4",
    ]
    if include_shared:
        cmd.append("--include-shared-expert")
    env = os.environ.copy()
    if fused:
        env["LARGERLM_MOE_DECODE_MXFP4_FUSED"] = "1"
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    payload = {
        "name": name,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed": _parse_stdout(completed.stdout),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit {completed.returncode}:\n"
            f"{completed.stderr or completed.stdout}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a safe single-layer GLM MXFP4 decode fused-path A/B."
    )
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=6144)
    parser.add_argument(
        "--input-mode",
        choices=("ones", "zeros", "sin"),
        default="ones",
    )
    parser.add_argument("--max-runner-scratch-mib", type=int, default=2048)
    parser.add_argument("--no-shared", action="store_true")
    parser.add_argument(
        "--write-json",
        type=Path,
        default=DEFAULT_PREPARED / "glm-layer19-decode-mxfp4-fused-ab-latest.json",
    )
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="largerlm-glm-mxfp4-fused-ab-", dir="/private/tmp"))
    input_path = work / "input.f32"
    _write_input(input_path, args.hidden_dim, args.input_mode)

    baseline = _run_case(
        name="baseline",
        runner=args.runner,
        prepared=args.prepared,
        layer=args.layer,
        top_k=args.top_k,
        input_path=input_path,
        output_path=work / "baseline.f32",
        router_path=work / "baseline-router.json",
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        include_shared=not args.no_shared,
        fused=False,
    )
    fused = _run_case(
        name="fused",
        runner=args.runner,
        prepared=args.prepared,
        layer=args.layer,
        top_k=args.top_k,
        input_path=input_path,
        output_path=work / "fused.f32",
        router_path=work / "fused-router.json",
        max_runner_scratch_mib=args.max_runner_scratch_mib,
        include_shared=not args.no_shared,
        fused=True,
    )
    baseline_output = _read_f32(work / "baseline.f32")
    fused_output = _read_f32(work / "fused.f32")
    max_abs_diff = _max_abs_diff(baseline_output, fused_output)
    router_equal = (work / "baseline-router.json").read_bytes() == (
        work / "fused-router.json"
    ).read_bytes()

    comparison: dict[str, Any] = {
        "schema": "largerlm.glm_decode_mxfp4_fused_ab.v1",
        "work_dir": str(work),
        "prepared": str(args.prepared),
        "runner": str(args.runner),
        "layer": args.layer,
        "top_k": args.top_k,
        "input_mode": args.input_mode,
        "include_shared": not args.no_shared,
        "max_abs_diff": max_abs_diff,
        "router_json_equal": router_equal,
        "baseline": baseline,
        "fused": fused,
    }
    args.write_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    base_timing = baseline["parsed"].get("timing", {})
    fused_timing = fused["parsed"].get("timing", {})
    print(f"wrote: {args.write_json}")
    print(f"work:  {work}")
    print(f"router_json_equal: {router_equal}")
    print(f"max_abs_diff: {max_abs_diff:.9g}")
    for key in ("moe", "expert_kernel", "expert_read", "shared"):
        b = base_timing.get(key)
        f = fused_timing.get(key)
        if b is not None and f is not None:
            print(f"{key}: baseline={b:.6f}s fused={f:.6f}s delta={f - b:+.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
