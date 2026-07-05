#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    return b"".join(f32_to_bf16(value) for value in values)


def write_fixture(root: Path) -> None:
    cache = root / "cache"
    cache.mkdir(parents=True)
    (cache / "context1_o_proj_bv.bin").write_bytes(bf16([8.0] * 8 + [4.0] * 8))
    (cache / "layout.json").write_text(
        json.dumps(
            {
                "schema": "largerlm.context1_o_proj_bv_cache.v1",
                "version": 1,
                "config_sha256": None,
                "source_prepared_manifest": str(root / "prepared" / "manifest.json"),
                "source_resident_layout": str(root / "prepared" / "resident" / "layout.json"),
                "source_resident_weight": str(root / "prepared" / "resident" / "resident.bin"),
                "context_limit": "decode/context_length_1_only",
                "dtype": "BF16",
                "dtype_bytes": 2,
                "weight_file": "context1_o_proj_bv.bin",
                "total_bytes": 32,
                "dims": {
                    "hidden_dim": 2,
                    "attention_value_dim": 8,
                    "num_heads": 1,
                    "v_head_dim": 8,
                    "qk_nope_dim": 8,
                    "kv_lora_dim": 8,
                },
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.context1_o_proj_bv.weight",
                        "layer": 0,
                        "offset": 0,
                        "size": 32,
                        "dtype": "BF16",
                        "shape": [2, 8],
                        "category": "context1_attention_output",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "latent.f32").write_bytes(f32([1.0] * 8))
    (root / "residual.f32").write_bytes(f32([0.5, -1.0]))


def run_probe(
    *,
    runner: Path,
    root: Path,
    output: Path,
    cache_file_override: Path | None = None,
) -> dict[str, object]:
    cmd = [
        str(runner),
        "--probe-context1-o-proj-cache-output",
        "--probe-layer",
        "0",
        "--context1-o-proj-cache-layout",
        str(root / "cache" / "layout.json"),
        "--input-f32",
        str(root / "latent.f32"),
        "--residual-f32",
        str(root / "residual.f32"),
        "--output-f32",
        str(output),
        "--max-cache-read-mib",
        "1",
        "--json",
    ]
    if cache_file_override is not None:
        cmd.extend(["--context1-o-proj-cache-file", str(cache_file_override)])
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return json.loads(completed.stdout)


def assert_probe_output(payload: dict[str, object], output: Path) -> None:
    if payload.get("ok") not in (True, 1):
        raise SystemExit("context1 cache probe did not report ok")
    got = struct.unpack("<2f", output.read_bytes())
    expected = (64.5, 31.0)
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"context1 cache output {got} != expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a tiny context=1 o_proj*B_v cache Metal output smoke."
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("glm_moe_infer"),
    )
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    root = args.root or Path(tempfile.mkdtemp(prefix="largerlm-context1-cache-", dir="/private/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    write_fixture(root)
    output = root / "output.f32"
    payload = run_probe(runner=args.runner, root=root, output=output)
    assert_probe_output(payload, output)

    default_cache = root / "cache" / "context1_o_proj_bv.bin"
    override_cache = root / "override_context1_o_proj_bv.bin"
    override_cache.write_bytes(default_cache.read_bytes())
    default_cache.unlink()
    override_output = root / "override_output.f32"
    override_payload = run_probe(
        runner=args.runner,
        root=root,
        output=override_output,
        cache_file_override=override_cache,
    )
    assert_probe_output(override_payload, override_output)
    print(f"fixture: {root}")
    print("  context1 o_proj cache output: ok")
    print("  context1 o_proj cache override: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
