from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(value) for value in values)


def pack8(values: list[int]) -> int:
    out = 0
    for index, value in enumerate(values):
        out |= (value & 0xF) << (index * 4)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "metal" / "largerlm-runner"
    with tempfile.TemporaryDirectory(prefix="largerlm-final-logits-smoke-") as tmp_s:
        tmp = Path(tmp_s)
        resident = tmp / "resident"
        resident.mkdir()
        payload = bytearray()
        tensors = []

        def add(name: str, dtype: str, shape: list[int], data: bytes, category: str) -> None:
            tensors.append(
                {
                    "name": name,
                    "offset": len(payload),
                    "size": len(data),
                    "dtype": dtype,
                    "shape": shape,
                    "category": category,
                }
            )
            payload.extend(data)

        add("model.norm.weight", "F32", [4], f32([1.0, 1.0, 1.0, 1.0]), "norms")
        add(
            "lm_head.weight",
            "F32",
            [4, 4],
            f32(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ]
            ),
            "lm_head",
        )
        (resident / "resident.bin").write_bytes(payload)
        layout = resident / "layout.json"
        layout.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_type": "glm_moe_dsa",
                    "alignment": 64,
                    "weight_file": "resident.bin",
                    "total_bytes": len(payload),
                    "tensors": tensors,
                }
            ),
            encoding="utf-8",
        )
        hidden = tmp / "hidden.f32"
        hidden.write_bytes(f32([1.0, 1.0, 1.0, 1.0]))
        topk = tmp / "topk.json"
        subprocess.run(
            [
                str(runner),
                "--resident-layout",
                str(layout),
                "--run-final-logits",
                "--input-f32",
                str(hidden),
                "--output-topk-json",
                str(topk),
                "--top-k",
                "2",
                "--rms-norm-eps",
                "0",
                "--max-chunk-mib",
                "1",
                "--max-runner-scratch-mib",
                "64",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        data = json.loads(topk.read_text(encoding="utf-8"))
        assert data["topk"][0]["token_id"] == 2
        assert data["topk"][1]["token_id"] == 0
        assert abs(data["topk"][0]["logit"] - 8.0) < 1e-5

        cli_topk = tmp / "topk_cli.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "largerlm",
                "final-logits",
                str(layout),
                "--runner",
                str(runner),
                "--input-f32",
                str(hidden),
                "--output-topk-json",
                str(cli_topk),
                "--top-k",
                "2",
                "--rms-norm-eps",
                "0",
                "--max-chunk-mib",
                "1",
                "--max-runner-scratch-mib",
                "64",
                "--quiet-runner",
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        cli_data = json.loads(cli_topk.read_text(encoding="utf-8"))
        assert cli_data == data

        affine_resident = tmp / "affine_resident"
        affine_resident.mkdir()
        affine_payload = bytearray()
        affine_tensors = []

        def add_affine(name: str, dtype: str, shape: list[int], data: bytes, category: str) -> None:
            affine_tensors.append(
                {
                    "name": name,
                    "offset": len(affine_payload),
                    "size": len(data),
                    "dtype": dtype,
                    "shape": shape,
                    "category": category,
                }
            )
            affine_payload.extend(data)

        add_affine("model.norm.weight", "F32", [8], f32([1.0] * 8), "norms")
        add_affine(
            "lm_head.weight",
            "U32",
            [4, 1],
            struct.pack("<4I", *(pack8([value] * 8) for value in (0, 1, 2, 3))),
            "lm_head",
        )
        add_affine("lm_head.scales", "BF16", [4, 1], bf16([1.0] * 4), "lm_head")
        add_affine("lm_head.biases", "BF16", [4, 1], bf16([0.0] * 4), "lm_head")
        (affine_resident / "resident.bin").write_bytes(affine_payload)
        affine_layout = affine_resident / "layout.json"
        affine_layout.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_type": "glm_moe_dsa",
                    "alignment": 64,
                    "weight_file": "resident.bin",
                    "total_bytes": len(affine_payload),
                    "tensors": affine_tensors,
                }
            ),
            encoding="utf-8",
        )
        affine_hidden = tmp / "affine_hidden.f32"
        affine_hidden.write_bytes(f32([1.0] * 8))
        affine_topk = tmp / "affine_topk.json"
        subprocess.run(
            [
                str(runner),
                "--resident-layout",
                str(affine_layout),
                "--run-final-logits",
                "--input-f32",
                str(affine_hidden),
                "--output-topk-json",
                str(affine_topk),
                "--top-k",
                "2",
                "--rms-norm-eps",
                "0",
                "--chunk-rows",
                "2",
                "--max-chunk-mib",
                "1",
                "--max-runner-scratch-mib",
                "64",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        affine_data = json.loads(affine_topk.read_text(encoding="utf-8"))
        assert affine_data["topk"][0]["token_id"] == 3
        assert affine_data["topk"][1]["token_id"] == 2
        assert abs(affine_data["topk"][0]["logit"] - 24.0) < 1e-5

        affine_cli_topk = tmp / "affine_topk_cli.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "largerlm",
                "final-logits",
                str(affine_layout),
                "--runner",
                str(runner),
                "--input-f32",
                str(affine_hidden),
                "--output-topk-json",
                str(affine_cli_topk),
                "--top-k",
                "2",
                "--rms-norm-eps",
                "0",
                "--chunk-rows",
                "2",
                "--max-chunk-mib",
                "1",
                "--max-runner-scratch-mib",
                "64",
                "--quiet-runner",
            ],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
        assert json.loads(affine_cli_topk.read_text(encoding="utf-8")) == affine_data


if __name__ == "__main__":
    main()
