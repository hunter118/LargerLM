#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


HIDDEN_DIM = 32
NUM_EXPERTS = 4
GROUP_SIZE = 32
SCALE_E8M0_ONE = 127


def pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    weights = bytearray()
    for code in (2, 1, 0, 0xA):
        packed = pack8(code)
        for _packed_col in range(HIDDEN_DIM // 8):
            weights.extend(struct.pack("<I", packed))
    scales = bytes([SCALE_E8M0_ONE]) * (NUM_EXPERTS * (HIDDEN_DIM // GROUP_SIZE))
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
                        "name": "model.layers.1.mlp.gate.weight",
                        "offset": 0,
                        "size": len(weights),
                        "dtype": "U32",
                        "shape": [NUM_EXPERTS, HIDDEN_DIM // 8],
                        "category": "routers",
                    },
                    {
                        "name": "model.layers.1.mlp.gate.scales",
                        "offset": scale_offset,
                        "size": len(scales),
                        "dtype": "U8",
                        "shape": [NUM_EXPERTS, HIDDEN_DIM // GROUP_SIZE],
                        "category": "routers",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row0 = [1.0 / HIDDEN_DIM] * HIDDEN_DIM
    row1 = [0.5 / HIDDEN_DIM] * HIDDEN_DIM
    (root / "input.f32").write_bytes(struct.pack("<32f", *row0))
    (root / "batch_input.f32").write_bytes(struct.pack("<64f", *(row0 + row1)))


def check_payload(payload: dict[str, object], *, expected_logits: list[float]) -> None:
    if payload["experts"] != [0, 1]:
        raise SystemExit(f"unexpected MXFP4 router experts: {payload['experts']}")
    expected_weights = [2.0 / 3.0, 1.0 / 3.0]
    got_weights = payload["weights"]
    if any(abs(a - b) > 2e-5 for a, b in zip(got_weights, expected_weights)):
        raise SystemExit(f"MXFP4 router weights {got_weights} != {expected_weights}")
    got_logits = payload["logits"]
    if any(abs(a - b) > 2e-4 for a, b in zip(got_logits, expected_logits)):
        raise SystemExit(f"MXFP4 router logits {got_logits} != {expected_logits}")
    if payload["used_correction_bias"]:
        raise SystemExit("MXFP4 router unexpectedly reported correction bias")


def run_single(runner: Path, root: Path) -> None:
    out = root / "router.json"
    cmd = [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "2",
        "--router-score",
        "raw",
        "--output-router-json",
        str(out),
        "--max-router-mib",
        "1",
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
        raise SystemExit("runner did not report MXFP4 router dtype")
    check_payload(
        json.loads(out.read_text(encoding="utf-8")),
        expected_logits=[1.0, 0.5, 0.0, -1.0],
    )


def run_batch(runner: Path, root: Path) -> None:
    out_dir = root / "router_batch"
    cmd = [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router-batch",
        "--input-f32",
        str(root / "batch_input.f32"),
        "--batch-tokens",
        "2",
        "--top-k",
        "2",
        "--router-score",
        "raw",
        "--output-router-json-dir",
        str(out_dir),
        "--max-router-mib",
        "1",
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
        raise SystemExit("runner did not report MXFP4 router batch dtype")
    check_payload(
        json.loads((out_dir / "token_000000.router.json").read_text(encoding="utf-8")),
        expected_logits=[1.0, 0.5, 0.0, -1.0],
    )
    check_payload(
        json.loads((out_dir / "token_000001.router.json").read_text(encoding="utf-8")),
        expected_logits=[0.5, 0.25, 0.0, -0.5],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resident MXFP4 router smoke.")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-mxfp4-router-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    run_single(args.runner, root)
    run_batch(args.runner, root)
    print(f"fixture: {root}")
    print("  resident MXFP4 router: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
