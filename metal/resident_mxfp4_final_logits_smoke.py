#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 32
VOCAB_SIZE = 4
GROUP_SIZE = 32
SCALE_E8M0_ONE = 127


def pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    weights = bytearray()
    for code in (2, 1, 0, 10):
        for _col in range(HIDDEN_DIM // 8):
            weights.extend(struct.pack("<I", pack8(code)))
    scales = bytes([SCALE_E8M0_ONE]) * (VOCAB_SIZE * (HIDDEN_DIM // GROUP_SIZE))
    payload = bytes(weights) + scales
    scale_offset = len(weights)
    (resident / "resident.bin").write_bytes(payload)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": [
                    {
                        "name": "lm_head.weight",
                        "offset": 0,
                        "size": len(weights),
                        "dtype": "U32",
                        "shape": [VOCAB_SIZE, HIDDEN_DIM // 8],
                        "category": "lm_head",
                    },
                    {
                        "name": "lm_head.scales",
                        "offset": scale_offset,
                        "size": len(scales),
                        "dtype": "U8",
                        "shape": [VOCAB_SIZE, HIDDEN_DIM // GROUP_SIZE],
                        "category": "lm_head",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "hidden.f32").write_bytes(f32([1.0 / HIDDEN_DIM] * HIDDEN_DIM))


def run_smoke(runner: Path, root: Path) -> None:
    topk_path = root / "topk.json"
    cmd = [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--run-final-logits",
        "--input-f32",
        str(root / "hidden.f32"),
        "--output-topk-json",
        str(topk_path),
        "--top-k",
        "3",
        "--chunk-rows",
        "2",
        "--skip-final-norm",
        "--max-chunk-mib",
        "0.001",
        "--max-runner-scratch-mib",
        "64",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if "dtype:              mlx-mxfp4" not in completed.stdout:
        raise SystemExit("runner did not report MXFP4 final logits dtype")
    if "group size:         32" not in completed.stdout:
        raise SystemExit("runner did not report MXFP4 final logits group size")
    payload = json.loads(topk_path.read_text(encoding="utf-8"))
    top_ids = [item["token_id"] for item in payload["topk"]]
    if top_ids != [0, 1, 2]:
        raise SystemExit(f"unexpected MXFP4 final logits top-k: {top_ids}")
    top_logits = [float(item["logit"]) for item in payload["topk"]]
    expected = [1.0, 0.5, 0.0]
    if any(abs(got - want) > 2e-4 for got, want in zip(top_logits, expected)):
        raise SystemExit(f"unexpected MXFP4 final logits values: {top_logits}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resident MXFP4 final logits smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-mxfp4-final-logits-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    run_smoke(args.runner, root)
    print(f"fixture: {root}")
    print("  resident MXFP4 final logits: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
