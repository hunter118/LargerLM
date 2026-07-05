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
    o_proj = bf16([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    (resident / "resident.bin").write_bytes(o_proj)
    (resident / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "config_sha256": None,
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(o_proj),
                "tensors": [
                    {
                        "name": "model.layers.1.self_attn.o_proj.weight",
                        "offset": 0,
                        "size": len(o_proj),
                        "dtype": "BF16",
                        "shape": [3, 2],
                        "category": "attention",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "attn_value.f32").write_bytes(f32([1.0, 2.0]))
    (root / "residual.f32").write_bytes(f32([0.5, -1.0, 3.0]))
    (root / "attn_value_batch.f32").write_bytes(f32([1.0, 2.0, 3.0, 4.0]))
    (root / "residual_batch.f32").write_bytes(f32([0.5, -1.0, 3.0, 1.0, 2.0, 3.0]))


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-attn-out-", dir="/private/tmp"))
    write_fixture(root)
    output = root / "output.f32"
    cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-attn-output",
        "--input-f32",
        str(root / "attn_value.f32"),
        "--residual-f32",
        str(root / "residual.f32"),
        "--output-f32",
        str(output),
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
    got = struct.unpack("<3f", output.read_bytes())
    expected = (1.5, 1.0, 6.0)
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"attention output {got} != expected {expected}")
    batch_output = root / "output_batch.f32"
    batch_projection = root / "projection_batch.f32"
    batch_cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--resident-layout",
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--run-attn-output-batch",
        "--input-f32",
        str(root / "attn_value_batch.f32"),
        "--residual-f32",
        str(root / "residual_batch.f32"),
        "--batch-tokens",
        "2",
        "--output-f32",
        str(batch_output),
        "--projection-f32",
        str(batch_projection),
        "--max-resident-matrix-mib",
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
    got_projection = struct.unpack("<6f", batch_projection.read_bytes())
    expected_projection = (1.0, 2.0, 3.0, 3.0, 4.0, 7.0)
    if any(abs(a - b) > 1e-4 for a, b in zip(got_projection, expected_projection)):
        raise SystemExit(
            f"attention output batch projection {got_projection} != expected {expected_projection}"
        )
    got_batch = struct.unpack("<6f", batch_output.read_bytes())
    expected_batch = (1.5, 1.0, 6.0, 4.0, 6.0, 10.0)
    if any(abs(a - b) > 1e-4 for a, b in zip(got_batch, expected_batch)):
        raise SystemExit(f"attention output batch {got_batch} != expected {expected_batch}")
    server_output = root / "output_batch_server.f32"
    server_projection = root / "projection_batch_server.f32"
    server_cmd = [
        str(Path(__file__).with_name("largerlm-runner")),
        "--run-attn-output-batch-server-jsonl",
    ]
    request = {
        "resident_layout": str(root / "resident" / "layout.json"),
        "layer": 1,
        "input_f32": str(root / "attn_value_batch.f32"),
        "residual_f32": str(root / "residual_batch.f32"),
        "output_f32": str(server_output),
        "projection_f32": str(server_projection),
        "batch_tokens": 2,
        "max_resident_matrix_mib": 1,
        "max_runner_scratch_mib": 64,
    }
    server = subprocess.Popen(
        server_cmd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert server.stdin is not None
    assert server.stdout is not None
    server.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
    server.stdin.write(json.dumps({"command": "quit"}, separators=(",", ":")) + "\n")
    server.stdin.close()
    server_stdout = server.stdout.read()
    returncode = server.wait()
    if server_stdout:
        print(server_stdout, end="")
    if returncode != 0:
        raise SystemExit(returncode)
    if "  server request:      ok" not in server_stdout:
        raise SystemExit("attention output batch server did not report success")
    got_server_projection = struct.unpack("<6f", server_projection.read_bytes())
    if any(
        abs(a - b) > 1e-4
        for a, b in zip(got_server_projection, expected_projection)
    ):
        raise SystemExit(
            "attention output batch server projection "
            f"{got_server_projection} != expected {expected_projection}"
        )
    got_server = struct.unpack("<6f", server_output.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got_server, expected_batch)):
        raise SystemExit(
            f"attention output batch server {got_server} != expected {expected_batch}"
        )
    print(f"fixture: {root}")
    print("  attention output:   ok")
    print("  attention batch:    ok")
    print("  attention server:   ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
