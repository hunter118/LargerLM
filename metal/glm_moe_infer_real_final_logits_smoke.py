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
RUNNER = ROOT / "metal" / "largerlm-runner"
INFER = ROOT / "metal" / "glm_moe_infer"
HIDDEN = 6144


def write_input(path: Path) -> None:
    values = [
        math.sin(i * 0.011) * 0.03125 + math.cos(i * 0.017) * 0.015625
        for i in range(HIDDEN)
    ]
    path.write_bytes(struct.pack(f"<{HIDDEN}f", *values))


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed


def compare_topk(old_path: Path, new_path: Path, *, max_diff: float) -> dict[str, object]:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old_topk = old["topk"]
    new_topk = new["topk"]
    old_ids = [int(item["token_id"]) for item in old_topk]
    new_ids = [int(item["token_id"]) for item in new_topk]
    if old_ids != new_ids:
        raise SystemExit(f"top-k token ids differ: old={old_ids} new={new_ids}")
    diffs = [
        abs(float(a["logit"]) - float(b["logit"]))
        for a, b in zip(old_topk, new_topk)
    ]
    max_abs = max(diffs, default=0.0)
    max_index = diffs.index(max_abs) if diffs else 0
    if max_abs > max_diff:
        raise SystemExit(
            f"top-k logit diff {max_abs:.9g} at {max_index} exceeds {max_diff:.9g}: "
            f"old={old_topk[max_index]} new={new_topk[max_index]}"
        )
    return {
        "token_ids": old_ids,
        "max_abs_diff": max_abs,
        "max_index": max_index,
        "old_logit_at_max": float(old_topk[max_index]["logit"]) if diffs else 0.0,
        "new_logit_at_max": float(new_topk[max_index]["logit"]) if diffs else 0.0,
        "old_read_bytes": old.get("read_bytes"),
        "new_read_bytes": new.get("read_bytes"),
        "old_chunks": old.get("chunks"),
        "new_chunks": new.get("chunks"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-diff", type=float, default=1e-4)
    parser.add_argument("--max-chunk-mib", type=float, default=16.0)
    parser.add_argument("--max-live-working-set-mib", type=float, default=128.0)
    args = parser.parse_args()

    if not RESIDENT_LAYOUT.exists():
        raise SystemExit(f"resident layout not found under {PREPARED}")
    root = Path(tempfile.mkdtemp(prefix="largerlm-real-final-logits-", dir="/private/tmp"))
    input_path = root / "hidden.f32"
    old_topk = root / "old_topk.json"
    new_topk = root / "new_topk.json"
    write_input(input_path)

    old_cmd = [
        str(RUNNER),
        "--resident-layout",
        str(RESIDENT_LAYOUT),
        "--run-final-logits",
        "--input-f32",
        str(input_path),
        "--output-topk-json",
        str(old_topk),
        "--top-k",
        str(args.top_k),
        "--rms-norm-eps",
        "1e-5",
        "--max-chunk-mib",
        str(args.max_chunk_mib),
        "--max-runner-scratch-mib",
        "256",
    ]
    new_cmd = [
        str(INFER),
        "--prepared",
        str(PREPARED),
        "--probe-final-logits",
        "--input-f32",
        str(input_path),
        "--output-topk-json",
        str(new_topk),
        "--top-k",
        str(args.top_k),
        "--rms-norm-eps",
        "1e-5",
        "--max-chunk-mib",
        str(args.max_chunk_mib),
        "--max-live-working-set-mib",
        str(args.max_live_working_set_mib),
        "--json",
    ]

    print("old runner:")
    run_command(old_cmd)
    print("glm_moe_infer:")
    completed = run_command(new_cmd)
    payload = json.loads(completed.stdout)
    logits = payload.get("probe_final_logits") or {}
    if not payload.get("ok") or not logits.get("ok"):
        raise SystemExit("glm_moe_infer final logits JSON reports failure")
    if payload.get("expert_buffer_count") != 0 or payload.get("expert_files_opened") != 0:
        raise SystemExit("final logits probe should not open expert files or allocate expert buffers")
    comparison = compare_topk(old_topk, new_topk, max_diff=args.max_diff)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"fixture: {root}")
    print(f"  estimated live MiB:    {payload['estimated_live_working_set_bytes'] / 1048576.0:.3f}")
    print(f"  read MiB:              {logits['bytes_read'] / 1048576.0:.3f}")
    print(f"  elapsed:               {logits['elapsed_seconds']:.6f} s")
    print("  final logits smoke:    ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
