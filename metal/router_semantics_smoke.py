#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    router = struct.pack("<4f", 0.0, 0.1, 0.2, 0.3)
    bias = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    (resident / "resident.bin").write_bytes(router + bias)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(router) + len(bias),
                "router": {
                    "scoring_func": "sigmoid",
                    "norm_topk_prob": True,
                    "routed_scaling_factor": 2.5,
                    "n_group": 1,
                    "topk_group": 1,
                    "topk_method": "noaux_tc",
                    "num_experts_per_tok": 2,
                },
                "tensors": [
                    {
                        "name": "model.layers.1.mlp.gate.e_score_correction_bias",
                        "offset": len(router),
                        "size": len(bias),
                        "dtype": "F32",
                        "shape": [4],
                        "category": "routers",
                    },
                    {
                        "name": "model.layers.1.mlp.gate.weight",
                        "offset": 0,
                        "size": len(router),
                        "dtype": "F32",
                        "shape": [4, 1],
                        "category": "routers",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(struct.pack("<f", 1.0))
    (root / "batch_input.f32").write_bytes(struct.pack("<2f", 1.0, 1.0))


def bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return bits >> 16


def pack8(values: list[int]) -> int:
    out = 0
    for idx, value in enumerate(values):
        out |= (value & 0xF) << (idx * 4)
    return out


def write_affine_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    packed_rows = [pack8([value] * 8) for value in (0, 1, 2, 3)]
    weights = struct.pack("<4I", *packed_rows)
    scales = struct.pack("<4H", *([bf16(1.0)] * 4))
    biases = struct.pack("<4H", *([bf16(0.0)] * 4))
    (resident / "resident.bin").write_bytes(weights + scales + biases)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(weights) + len(scales) + len(biases),
                "router": {
                    "scoring_func": "raw",
                    "norm_topk_prob": True,
                    "routed_scaling_factor": 1.0,
                    "n_group": 1,
                    "topk_group": 1,
                    "num_experts_per_tok": 2,
                },
                "tensors": [
                    {
                        "name": "model.layers.1.mlp.gate.weight",
                        "offset": 0,
                        "size": len(weights),
                        "dtype": "U32",
                        "shape": [4, 1],
                        "category": "routers",
                    },
                    {
                        "name": "model.layers.1.mlp.gate.scales",
                        "offset": len(weights),
                        "size": len(scales),
                        "dtype": "BF16",
                        "shape": [4, 1],
                        "category": "routers",
                    },
                    {
                        "name": "model.layers.1.mlp.gate.biases",
                        "offset": len(weights) + len(scales),
                        "size": len(biases),
                        "dtype": "BF16",
                        "shape": [4, 1],
                        "category": "routers",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row = [1.0] * 8
    (root / "input.f32").write_bytes(struct.pack("<8f", *row))
    (root / "batch_input.f32").write_bytes(struct.pack("<16f", *(row + row)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def assert_affine_payload(payload: dict[str, object]) -> None:
    if payload["experts"] != [3, 2]:
        raise SystemExit(f"unexpected affine experts: {payload['experts']}")
    expected_weights = [24.0 / 40.0, 16.0 / 40.0]
    got_weights = payload["weights"]
    if any(abs(a - b) > 2e-6 for a, b in zip(got_weights, expected_weights)):
        raise SystemExit(f"affine weights {got_weights} != {expected_weights}")
    expected_logits = [0.0, 8.0, 16.0, 24.0]
    got_logits = payload["logits"]
    if any(abs(a - b) > 2e-5 for a, b in zip(got_logits, expected_logits)):
        raise SystemExit(f"affine logits {got_logits} != {expected_logits}")
    if payload["used_correction_bias"]:
        raise SystemExit("affine router unexpectedly reported correction bias")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-router-semantics-", dir="/private/tmp"))
    write_fixture(root)
    out = root / "router.json"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router",
        "--input-f32",
        str(root / "input.f32"),
        "--top-k",
        "2",
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

    payload = json.loads(out.read_text(encoding="utf-8"))
    if payload["experts"] != [0, 3]:
        raise SystemExit(f"unexpected experts: {payload['experts']}")
    s0 = sigmoid(0.0)
    s3 = sigmoid(0.3)
    expected = [2.5 * s0 / (s0 + s3), 2.5 * s3 / (s0 + s3)]
    got = payload["weights"]
    if any(abs(a - b) > 2e-6 for a, b in zip(got, expected)):
        raise SystemExit(f"weights {got} != expected {expected}")
    if not payload["used_correction_bias"]:
        raise SystemExit("runner did not report correction bias usage")

    batch_dir = root / "router_batch"
    batch_cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
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
        "--output-router-json-dir",
        str(batch_dir),
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    completed = subprocess.run(batch_cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    for token in range(2):
        batch_payload = json.loads(
            (batch_dir / f"token_{token:06d}.router.json").read_text(encoding="utf-8")
        )
        if batch_payload["experts"] != payload["experts"]:
            raise SystemExit(f"batch token {token} experts mismatch: {batch_payload}")
        if any(abs(a - b) > 2e-6 for a, b in zip(batch_payload["weights"], expected)):
            raise SystemExit(f"batch token {token} weights mismatch: {batch_payload}")
        if not batch_payload["used_correction_bias"]:
            raise SystemExit(f"batch token {token} did not report correction bias")

    affine_root = root / "affine"
    write_affine_fixture(affine_root)
    affine_out = affine_root / "router.json"
    affine_cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(affine_root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router",
        "--input-f32",
        str(affine_root / "input.f32"),
        "--top-k",
        "2",
        "--router-score",
        "raw",
        "--output-router-json",
        str(affine_out),
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    completed = subprocess.run(affine_cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    assert_affine_payload(json.loads(affine_out.read_text(encoding="utf-8")))

    affine_batch_dir = affine_root / "router_batch"
    affine_batch_cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(affine_root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-router-batch",
        "--input-f32",
        str(affine_root / "batch_input.f32"),
        "--batch-tokens",
        "2",
        "--top-k",
        "2",
        "--router-score",
        "raw",
        "--output-router-json-dir",
        str(affine_batch_dir),
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    completed = subprocess.run(affine_batch_cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    for token in range(2):
        assert_affine_payload(
            json.loads(
                (affine_batch_dir / f"token_{token:06d}.router.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    print(f"fixture: {root}")
    print("  expected experts:   0,3")
    print("  affine experts:     3,2")
    print("  router semantics:   ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
