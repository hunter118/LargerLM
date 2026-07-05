#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def pack_int4(values: tuple[int, ...]) -> bytes:
    if len(values) != 8:
        raise AssertionError(values)
    packed = 0
    for index, value in enumerate(values):
        if value < 0 or value > 15:
            raise AssertionError(values)
        packed |= value << (index * 4)
    return struct.pack("<I", packed)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    q_a = struct.pack("<6f", 1.0, 1.0, 1.0, 2.0, 0.0, 1.0)
    q_b_values = [1.0, 2.0, 3.0, 0.5, 0.5, 0.5]
    q_b = b"".join(f32_to_bf16(v) for v in q_b_values)
    large_f32 = struct.pack(
        "<1024f",
        *(
            1.0 if row == col else 0.0
            for row in range(32)
            for col in range(32)
        ),
    )
    large_bf16 = b"".join(
        f32_to_bf16(1.0 if row == col else 0.0)
        for row in range(32)
        for col in range(32)
    )
    q_c_w = pack_int4((1, 2, 3, 4, 0, 0, 0, 0)) + pack_int4((0, 1, 0, 1, 0, 1, 0, 1))
    q_c_s = f32_to_bf16(1.0) + f32_to_bf16(1.0)
    q_c_b = f32_to_bf16(0.0) + f32_to_bf16(0.0)
    payload = q_a + q_b + large_f32 + large_bf16 + q_c_w + q_c_s + q_c_b
    q_c_offset = len(q_a) + len(q_b) + len(large_f32) + len(large_bf16)
    q_c_s_offset = q_c_offset + len(q_c_w)
    q_c_b_offset = q_c_s_offset + len(q_c_s)
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
                        "name": "model.layers.1.self_attn.q_a_proj.weight",
                        "offset": 0,
                        "size": len(q_a),
                        "dtype": "F32",
                        "shape": [2, 3],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_b_proj.weight",
                        "offset": len(q_a),
                        "size": len(q_b),
                        "dtype": "BF16",
                        "shape": [2, 3],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.large_f32_proj.weight",
                        "offset": len(q_a) + len(q_b),
                        "size": len(large_f32),
                        "dtype": "F32",
                        "shape": [32, 32],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.large_bf16_proj.weight",
                        "offset": len(q_a) + len(q_b) + len(large_f32),
                        "size": len(large_bf16),
                        "dtype": "BF16",
                        "shape": [32, 32],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_c_proj.weight",
                        "offset": q_c_offset,
                        "size": len(q_c_w),
                        "dtype": "U32",
                        "shape": [2, 1],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_c_proj.scales",
                        "offset": q_c_s_offset,
                        "size": len(q_c_s),
                        "dtype": "BF16",
                        "shape": [2, 1],
                        "category": "attention",
                    },
                    {
                        "name": "model.layers.1.self_attn.q_c_proj.biases",
                        "offset": q_c_b_offset,
                        "size": len(q_c_b),
                        "dtype": "BF16",
                        "shape": [2, 1],
                        "category": "attention",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    tokens = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    (root / "input.f32").write_bytes(struct.pack("<6f", *tokens))
    tokens8 = (
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    )
    (root / "input8.f32").write_bytes(struct.pack("<16f", *tokens8))
    large_values = [float((token * 32 + dim) % 17 - 8) for token in range(128) for dim in range(32)]
    (root / "large_input.f32").write_bytes(struct.pack("<4096f", *large_values))


def run_case(
    root: Path,
    suffix: str,
    expected: tuple[float, ...],
    *,
    backend: str | None = None,
    expected_raw_conversion_bytes: int = 0,
    input_name: str = "input.f32",
) -> None:
    backend_tag = backend or "custom-metal"
    output = root / f"{suffix.split('.')[-3]}_{backend_tag}_batch_out.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-resident-linear-batch",
        "--tensor-suffix",
        suffix,
        "--input-f32",
        str(root / input_name),
        "--batch-tokens",
        "2",
        "--output-f32",
        str(output),
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    if backend is not None:
        cmd.extend(["--prefill-linear-backend", backend])
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if f"backend:            {backend_tag}" not in completed.stdout:
        raise SystemExit(f"runner did not report backend {backend_tag}")
    if "matrix scratch:" not in completed.stdout:
        raise SystemExit("runner did not report matrix scratch bytes")
    if "estimated peak:" not in completed.stdout:
        raise SystemExit("runner did not report estimated peak bytes")
    if "timing backend:" not in completed.stdout:
        raise SystemExit("runner did not report backend timing")
    if "timing matrix f32:" not in completed.stdout:
        raise SystemExit("runner did not report matrix-f32 timing")
    if "timing accelerator:" not in completed.stdout:
        raise SystemExit("runner did not report accelerator timing")
    raw_line = f"matrix raw conv:    {expected_raw_conversion_bytes}"
    if raw_line not in completed.stdout:
        raise SystemExit(
            f"runner did not report expected raw conversion bytes: {raw_line}"
        )
    got = struct.unpack("<4f", output.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"{suffix} {backend_tag} output {got} != expected {expected}")


def run_cli_case(
    root: Path,
    expected: tuple[float, ...],
    *,
    suffix: str = ".self_attn.q_a_proj.weight",
    input_name: str = "input.f32",
) -> None:
    output = root / "cli_batch_out.f32"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-linear-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--tensor-suffix",
        suffix,
        "--input-f32",
        str(root / input_name),
        "--batch-tokens",
        "2",
        "--output-f32",
        str(output),
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--quiet-runner",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    if payload["batch_tokens"] != 2 or payload["output_bytes"] != 16:
        raise SystemExit(f"unexpected CLI payload: {payload}")
    got = struct.unpack("<4f", output.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"CLI output {got} != expected {expected}")


def run_server_case(root: Path) -> None:
    q_a_output = root / "server_q_a_batch_out.f32"
    q_b_output = root / "server_q_b_batch_out.f32"
    requests = [
        {
            "resident_layout": str(root / "resident" / "layout.json"),
            "layer": 1,
            "tensor_suffix": ".self_attn.q_a_proj.weight",
            "input_f32": str(root / "input.f32"),
            "batch_tokens": 2,
            "output_f32": str(q_a_output),
            "max_resident_matrix_mib": 1,
            "max_runner_scratch_mib": 64,
            "prefill_linear_backend": "custom-metal",
        },
        {
            "resident_layout": str(root / "resident" / "layout.json"),
            "layer": 1,
            "tensor_suffix": ".self_attn.q_b_proj.weight",
            "input_f32": str(root / "input.f32"),
            "batch_tokens": 2,
            "output_f32": str(q_b_output),
            "max_resident_matrix_mib": 1,
            "max_runner_scratch_mib": 64,
            "prefill_linear_backend": "mpsgraph-f32",
        },
        {"command": "quit"},
    ]
    payload = "\n".join(json.dumps(request) for request in requests) + "\n"
    completed = subprocess.run(
        [str(Path(__file__).with_name("largerlm-runner")),
         "--run-resident-linear-batch-plan-server-jsonl"],
        input=payload,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if completed.stdout.count("  server request:      ok") != 2:
        raise SystemExit(f"server did not accept two requests:\n{completed.stdout}")
    if "  requests:            2" not in completed.stdout:
        raise SystemExit(f"server did not report two requests:\n{completed.stdout}")
    q_a = struct.unpack("<4f", q_a_output.read_bytes())
    q_b = struct.unpack("<4f", q_b_output.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(q_a, (6.0, 5.0, 15.0, 14.0))):
        raise SystemExit(f"server q_a output {q_a}")
    if any(abs(a - b) > 1e-4 for a, b in zip(q_b, (14.0, 3.0, 32.0, 7.5))):
        raise SystemExit(f"server q_b output {q_b}")


def run_auto_cli_case(root: Path, suffix: str) -> None:
    output = root / f"{suffix.split('.')[-3]}_auto_cli_batch_out.f32"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-linear-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--tensor-suffix",
        suffix,
        "--input-f32",
        str(root / "large_input.f32"),
        "--batch-tokens",
        "128",
        "--output-f32",
        str(output),
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--prefill-linear-backend",
        "auto",
        "--prefill-mpsgraph-min-batch-tokens",
        "64",
        "--prefill-mpsgraph-min-matrix-dim",
        "16",
        "--quiet-runner",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    payload = json.loads(completed.stdout)
    if payload["backend"] != "mpsgraph-f32" or payload["batch_tokens"] != 128:
        raise SystemExit(f"unexpected auto CLI payload: {payload}")
    got = struct.unpack("<4096f", output.read_bytes())
    expected = struct.unpack("<4096f", (root / "large_input.f32").read_bytes())
    max_err = max(abs(a - b) for a, b in zip(got, expected))
    if max_err > 1e-4:
        raise SystemExit(f"{suffix} auto output max error {max_err}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-resident-linear-batch-", dir="/private/tmp"))
    write_fixture(root)
    run_case(root, ".self_attn.q_a_proj.weight", (6.0, 5.0, 15.0, 14.0))
    run_case(
        root,
        ".self_attn.q_a_proj.weight",
        (6.0, 5.0, 15.0, 14.0),
        backend="mpsgraph-f32",
    )
    run_case(
        root,
        ".self_attn.q_a_proj.weight",
        (6.0, 5.0, 15.0, 14.0),
        backend="mps-matrix-f32",
    )
    run_case(root, ".self_attn.q_b_proj.weight", (14.0, 3.0, 32.0, 7.5))
    run_case(
        root,
        ".self_attn.q_c_proj.weight",
        (10.0, 6.0, 5.0, 0.0),
        input_name="input8.f32",
    )
    run_case(
        root,
        ".self_attn.q_b_proj.weight",
        (14.0, 3.0, 32.0, 7.5),
        backend="mpsgraph-f32",
        expected_raw_conversion_bytes=12,
    )
    run_case(
        root,
        ".self_attn.q_b_proj.weight",
        (14.0, 3.0, 32.0, 7.5),
        backend="mps-matrix-f32",
        expected_raw_conversion_bytes=12,
    )
    run_cli_case(root, (6.0, 5.0, 15.0, 14.0))
    run_cli_case(
        root,
        (10.0, 6.0, 5.0, 0.0),
        suffix=".self_attn.q_c_proj.weight",
        input_name="input8.f32",
    )
    run_server_case(root)
    run_auto_cli_case(root, ".self_attn.large_f32_proj.weight")
    run_auto_cli_case(root, ".self_attn.large_bf16_proj.weight")
    print(f"fixture: {root}")
    print("  resident batch linear: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
