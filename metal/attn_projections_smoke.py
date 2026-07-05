#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(v) for v in values)


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    blobs = [
        (
            "model.layers.1.input_layernorm.weight",
            "F32",
            [3],
            "norms",
            f32([1.0, 1.0, 1.0]),
        ),
        (
            "model.layers.1.self_attn.q_a_layernorm.weight",
            "F32",
            [2],
            "norms",
            f32([1.0, 1.0]),
        ),
        (
            "model.layers.1.self_attn.kv_a_layernorm.weight",
            "F32",
            [2],
            "norms",
            f32([1.0, 1.0]),
        ),
        (
            "model.layers.1.self_attn.q_a_proj.weight",
            "F32",
            [2, 3],
            "attention",
            f32([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ),
        (
            "model.layers.1.self_attn.q_b_proj.weight",
            "BF16",
            [4, 2],
            "attention",
            bf16([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 3.0]),
        ),
        (
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
            "F32",
            [3, 3],
            "attention",
            f32([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0]),
        ),
        (
            "model.layers.1.self_attn.kv_b_proj.weight",
            "F32",
            [4, 2],
            "attention",
            f32([1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 2.0, 0.0]),
        ),
    ]

    payload = bytearray()
    tensors = []
    for name, dtype, shape, category, data in blobs:
        offset = len(payload)
        payload.extend(data)
        tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": category,
            }
        )

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
                "tensors": tensors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "input.f32").write_bytes(f32([1.0, 1.0, 1.0]))
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 64,
                "total_bytes": 24,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 0,
                        "width": 3,
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": 6,
                        "max_context_tokens": 4,
                        "total_bytes": 24,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "decode_cache.bin").write_bytes(b"\0" * 24)


def expect_file(path: Path, expected: tuple[float, ...]) -> None:
    got = struct.unpack(f"<{len(expected)}f", path.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"{path.name} {got} != expected {expected}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-attn-proj-", dir="/private/tmp"))
    write_fixture(root)
    output = root / "out"
    cache_file = root / "decode_cache.bin"
    runner = Path(__file__).with_name("largerlm-runner")
    cmd = [
        str(runner),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-attn-projections",
        "--input-f32",
        str(root / "input.f32"),
        "--rms-norm-eps",
        "0",
        "--output-dir",
        str(output),
        "--cache-layout",
        str(root / "cache_layout.json"),
        "--cache-file",
        str(cache_file),
        "--position",
        "2",
        "--max-cache-file-mib",
        "1",
        "--max-resident-matrix-mib",
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

    expect_file(output / "attn_input_norm.f32", (1.0, 1.0, 1.0))
    expect_file(output / "attn_q_a.f32", (1.0, 1.0))
    expect_file(output / "attn_q_a_norm.f32", (1.0, 1.0))
    expect_file(output / "attn_q_b.f32", (1.0, 1.0, 2.0, 5.0))
    expect_file(output / "attn_kv_a.f32", (1.0, 1.0, 4.0))
    expect_file(output / "attn_kv_a_norm.f32", (1.0, 1.0))
    expect_file(output / "attn_kv_b.f32", (3.0, 7.0, 1.0, 2.0))
    cache = cache_file.read_bytes()
    expected_cache = bf16([1.0, 1.0, 4.0])
    if cache[12:18] != expected_cache:
        raise SystemExit(f"cache token bytes {cache[12:18]!r} != {expected_cache!r}")

    server_output = root / "out_server"
    server_request = {
        "resident_layout": str(root / "resident" / "layout.json"),
        "layer": 1,
        "input_f32": str(root / "input.f32"),
        "output_dir": str(server_output),
        "batch_tokens": 1,
        "rms_norm_eps": 0.0,
        "max_resident_matrix_mib": 1,
        "max_runner_scratch_mib": 64,
    }
    server_completed = subprocess.run(
        [str(runner), "--run-attn-projections-server-jsonl"],
        input=json.dumps(server_request) + "\n" + json.dumps({"command": "quit"}) + "\n",
        text=True,
        capture_output=True,
    )
    if server_completed.stdout:
        print(server_completed.stdout, end="")
    if server_completed.stderr:
        print(server_completed.stderr, end="")
    if server_completed.returncode != 0:
        raise SystemExit(server_completed.returncode)
    if "server request:      ok" not in server_completed.stdout:
        raise SystemExit("attention projections server did not acknowledge request")
    expect_file(server_output / "attn_input_norm.f32", (1.0, 1.0, 1.0))
    expect_file(server_output / "attn_q_a.f32", (1.0, 1.0))
    expect_file(server_output / "attn_q_a_norm.f32", (1.0, 1.0))
    expect_file(server_output / "attn_q_b.f32", (1.0, 1.0, 2.0, 5.0))
    expect_file(server_output / "attn_kv_a.f32", (1.0, 1.0, 4.0))
    expect_file(server_output / "attn_kv_a_norm.f32", (1.0, 1.0))
    expect_file(server_output / "attn_kv_b.f32", (3.0, 7.0, 1.0, 2.0))
    print(f"fixture: {root}")
    print("  attention projections: ok")
    print("  attention projections server: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
