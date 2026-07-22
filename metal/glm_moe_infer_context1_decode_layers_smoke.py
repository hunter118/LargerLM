#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

from glm_moe_infer_context1_moe_decoder_smoke import write_fixture


def run_json(cmd: list[str]) -> dict[str, object]:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return json.loads(completed.stdout)


def read_f32(path: Path) -> tuple[float, ...]:
    return struct.unpack("<32f", path.read_bytes())


def decode_layers_cmd(
    root: Path,
    binary: Path,
    cache_file: Path,
    output: Path,
    work: Path,
) -> list[str]:
    return [
        str(binary),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--expert-layout",
        str(root / "experts" / "layout.json"),
        "--probe-decode-layers",
        "--decode-layers",
        "0",
        "--input-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(output),
        "--output-dir",
        str(work),
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(cache_file),
        "--position",
        "0",
        "--context-length",
        "1",
        "--num-heads",
        "1",
        "--kv-lora-dim",
        "32",
        "--qk-nope-dim",
        "8",
        "--rope-dim",
        "8",
        "--v-head-dim",
        "32",
        "--cache-position-offset",
        "0",
        "--top-k",
        "1",
        "--router-score",
        "raw",
        "--routed-scaling-factor",
        "1",
        "--ignore-router-bias",
        "--rms-norm-eps",
        "0",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-live-working-set-mib",
        "128",
        "--skip-debug-intermediates",
        "--json",
    ]


def first_layer(payload: dict[str, object]) -> dict[str, object]:
    decode = payload.get("probe_decode_layers")
    if not isinstance(decode, dict):
        raise SystemExit(f"missing probe_decode_layers payload: {payload}")
    layers = decode.get("layers")
    if not isinstance(layers, list) or not layers:
        raise SystemExit(f"missing decode layer items: {decode}")
    item = layers[0]
    if not isinstance(item, dict):
        raise SystemExit(f"invalid decode layer item: {item}")
    return item


def aggregate_count(payload: dict[str, object]) -> int:
    decode = payload.get("probe_decode_layers")
    if not isinstance(decode, dict):
        return -1
    value = decode.get("attn_output_context1_o_proj_cache_count")
    return int(value) if isinstance(value, int) else -1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify probe_decode_layers can consume a context=1 o_proj*B_v cache."
    )
    parser.add_argument("--binary", type=Path, default=Path(__file__).with_name("glm_moe_infer"))
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-context1-decode-layers-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    baseline = run_json(
        decode_layers_cmd(
            root,
            args.binary,
            root / "baseline_cache.bin",
            root / "baseline_out.f32",
            root / "baseline_work",
        )
    )
    context_cmd = decode_layers_cmd(
        root,
        args.binary,
        root / "context_cache.bin",
        root / "context_out.f32",
        root / "context_work",
    )
    context_cmd.extend(
        [
            "--context1-o-proj-cache-layout",
            str(root / "context1" / "layout.json"),
        ]
    )
    context = run_json(context_cmd)
    if first_layer(baseline).get("selected_experts") != [0]:
        raise SystemExit(f"decode-layers did not expose the routed expert: {baseline}")
    if bool(first_layer(baseline).get("attn_output_context1_o_proj_cache")):
        raise SystemExit("baseline decode-layers unexpectedly used context1 cache")
    if not bool(first_layer(context).get("attn_output_context1_o_proj_cache")):
        raise SystemExit(f"context decode-layers did not use context1 cache: {context}")
    if aggregate_count(context) != 1:
        raise SystemExit(f"context aggregate cache count is wrong: {context}")
    got_baseline = read_f32(root / "baseline_out.f32")
    got_context = read_f32(root / "context_out.f32")
    max_diff = max(math.fabs(a - b) for a, b in zip(got_baseline, got_context))
    if max_diff > 2e-3:
        raise SystemExit(f"context decode-layers output differs by {max_diff}")
    if (root / "baseline_cache.bin").read_bytes() != (root / "context_cache.bin").read_bytes():
        raise SystemExit("context decode-layers cache write differs from baseline")
    print(f"fixture: {root}")
    print(f"  max output diff: {max_diff:.6g}")
    print("  context1 decode-layers: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
