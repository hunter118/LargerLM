#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


TOKENS = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
INPUT_NORM = (1.0, 2.0, 0.5)
Q_A_NORM = (1.0, 1.0)
KV_A_NORM = (1.0, 1.0)
Q_A = (
    (1.0, 1.0, 1.0),
    (2.0, 0.0, 1.0),
)
Q_B = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (2.0, 3.0),
)
KV_A = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)
KV_B = (
    (1.0, 2.0),
    (3.0, 4.0),
    (0.5, 0.5),
    (2.0, 0.0),
)
O_PROJ = (
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
)


def pack_matrix(rows: tuple[tuple[float, ...], ...]) -> bytes:
    values = tuple(value for row in rows for value in row)
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16_to_f32(raw: bytes) -> float:
    bits = int.from_bytes(raw, "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def bf16_round(value: float) -> float:
    return bf16_to_f32(f32_to_bf16(value))


def bf16(values: tuple[float, ...]) -> bytes:
    return b"".join(f32_to_bf16(value) for value in values)


def add_blob(
    tensors: list[dict[str, object]],
    payload: bytearray,
    name: str,
    dtype: str,
    shape: list[int],
    category: str,
    data: bytes,
) -> None:
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


def write_fixture(root: Path) -> None:
    resident = root / "resident"
    resident.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []
    add_blob(
        tensors,
        payload,
        "model.layers.1.input_layernorm.weight",
        "F32",
        [len(INPUT_NORM)],
        "norms",
        struct.pack("<3f", *INPUT_NORM),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.q_a_layernorm.weight",
        "F32",
        [len(Q_A_NORM)],
        "norms",
        struct.pack("<2f", *Q_A_NORM),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.kv_a_layernorm.weight",
        "F32",
        [len(KV_A_NORM)],
        "norms",
        struct.pack("<2f", *KV_A_NORM),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.q_a_proj.weight",
        "F32",
        [len(Q_A), len(Q_A[0])],
        "attention",
        pack_matrix(Q_A),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.q_b_proj.weight",
        "F32",
        [len(Q_B), len(Q_B[0])],
        "attention",
        pack_matrix(Q_B),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "F32",
        [len(KV_A), len(KV_A[0])],
        "attention",
        pack_matrix(KV_A),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.kv_b_proj.weight",
        "F32",
        [len(KV_B), len(KV_B[0])],
        "attention",
        pack_matrix(KV_B),
    )
    add_blob(
        tensors,
        payload,
        "model.layers.1.self_attn.o_proj.weight",
        "F32",
        [len(O_PROJ), len(O_PROJ[0])],
        "attention",
        pack_matrix(O_PROJ),
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
    values = tuple(value for token in TOKENS for value in token)
    (root / "input.f32").write_bytes(struct.pack(f"<{len(values)}f", *values))
    (root / "cache_layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "max_context_tokens": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "alignment": 8,
                "total_bytes": 40,
                "segments": [
                    {
                        "kind": "mla_kv",
                        "layer": 1,
                        "offset": 8,
                        "width": len(KV_A),
                        "dtype": "BF16",
                        "dtype_bytes": 2,
                        "token_stride_bytes": len(KV_A) * 2,
                        "max_context_tokens": 4,
                        "total_bytes": len(KV_A) * 2 * 4,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "decode_cache.bin").write_bytes(b"\0" * 40)


def rmsnorm(token: tuple[float, ...], weights: tuple[float, ...]) -> tuple[float, ...]:
    inv = 1.0 / math.sqrt(sum(v * v for v in token) / len(token))
    return tuple(v * inv * w for v, w in zip(token, weights))


def rotate(values: tuple[float, ...], position: int, theta: float, interleave: bool) -> tuple[float, ...]:
    half = len(values) // 2
    out: list[float] = []
    for idx, x in enumerate(values):
        if interleave:
            pair = idx ^ 1
            rot = values[pair] if idx & 1 else -values[pair]
            freq_idx = idx // 2
        else:
            if idx < half:
                rot = -values[idx + half]
                freq_idx = idx
            else:
                rot = values[idx - half]
                freq_idx = idx - half
        angle = position / (theta ** (2.0 * freq_idx / len(values)))
        out.append(x * math.cos(angle) + rot * math.sin(angle))
    return tuple(out)


def matmul_rows(rows: tuple[tuple[float, ...], ...], token: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sum(a * b for a, b in zip(row, token)) for row in rows)


def attention_reference(
    q_nope: tuple[float, ...],
    q_rope: tuple[float, ...],
    cache: tuple[tuple[float, ...], ...],
    *,
    start_position: int,
) -> tuple[float, ...]:
    qk_nope_dim = 2
    rope_dim = 2
    v_head_dim = 2
    kv_lora_dim = 2
    scale = 1.0 / math.sqrt(qk_nope_dim + rope_dim)
    out: list[float] = []
    for token in range(2):
        context = start_position + token + 1
        scores: list[float] = []
        values: list[tuple[float, ...]] = []
        for t in range(context):
            row = cache[t]
            latent = row[:kv_lora_dim]
            k_rope = rotate(row[kv_lora_dim:], t, 10000.0, False)
            score = 0.0
            for d in range(qk_nope_dim):
                k_nope = sum(KV_B[d][r] * latent[r] for r in range(kv_lora_dim))
                score += q_nope[token * qk_nope_dim + d] * k_nope
            score += sum(
                q_rope[token * rope_dim + d] * k_rope[d] for d in range(rope_dim)
            )
            scores.append(score * scale)
            values.append(
                tuple(
                    sum(KV_B[qk_nope_dim + vd][r] * latent[r] for r in range(kv_lora_dim))
                    for vd in range(v_head_dim)
                )
            )
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        denom = sum(weights)
        for vd in range(v_head_dim):
            out.append(sum(w * value[vd] for w, value in zip(weights, values)) / denom)
    return tuple(out)


def attention_output_reference(attn_value: tuple[float, ...]) -> tuple[float, ...]:
    v_head_dim = len(O_PROJ[0])
    out: list[float] = []
    for token, residual in enumerate(TOKENS):
        values = attn_value[token * v_head_dim : (token + 1) * v_head_dim]
        projected = tuple(
            sum(weight * value for weight, value in zip(row, values)) for row in O_PROJ
        )
        out.extend(left + right for left, right in zip(projected, residual))
    return tuple(out)


def reference() -> dict[str, tuple[float, ...]]:
    out: dict[str, list[float]] = {
        "input_layernorm.f32": [],
        "q_a_proj.f32": [],
        "q_a_layernorm.f32": [],
        "q_b_proj.f32": [],
        "kv_a_proj_with_mqa.f32": [],
        "kv_a_lora.f32": [],
        "kv_a_rope.f32": [],
        "kv_a_layernorm.f32": [],
        "kv_b_proj.f32": [],
    }
    for token in TOKENS:
        input_norm = rmsnorm(token, INPUT_NORM)
        q_a = matmul_rows(Q_A, input_norm)
        q_a_norm = rmsnorm(q_a, Q_A_NORM)
        q_b = matmul_rows(Q_B, q_a_norm)
        kv_a = matmul_rows(KV_A, input_norm)
        kv_a_lora = kv_a[: len(KV_A_NORM)]
        kv_a_rope = kv_a[len(KV_A_NORM) :]
        kv_a_norm = rmsnorm(kv_a_lora, KV_A_NORM)
        kv_b = matmul_rows(KV_B, kv_a_norm)
        out["input_layernorm.f32"].extend(input_norm)
        out["q_a_proj.f32"].extend(q_a)
        out["q_a_layernorm.f32"].extend(q_a_norm)
        out["q_b_proj.f32"].extend(q_b)
        out["kv_a_proj_with_mqa.f32"].extend(kv_a)
        out["kv_a_lora.f32"].extend(kv_a_lora)
        out["kv_a_rope.f32"].extend(kv_a_rope)
        out["kv_a_layernorm.f32"].extend(kv_a_norm)
        out["kv_b_proj.f32"].extend(kv_b)
    return {name: tuple(values) for name, values in out.items()}


def check_f32(path: Path, expected: tuple[float, ...]) -> None:
    got = struct.unpack(f"<{len(expected)}f", path.read_bytes())
    if any(abs(a - b) > 1e-4 for a, b in zip(got, expected)):
        raise SystemExit(f"{path.name} output {got} != expected {expected}")


def run_cli_case(root: Path) -> None:
    output_dir = root / "projections"
    cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-attention-projections",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--input-f32",
        str(root / "input.f32"),
        "--batch-tokens",
        "2",
        "--rms-norm-eps",
        "0",
        "--output-dir",
        str(output_dir),
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
    if (
        payload["batch_tokens"] != 2
        or payload["q_b_output_bytes"] != 32
        or payload["kv_a_lora_bytes"] != 16
        or payload["kv_a_rope_bytes"] != 16
        or payload["kv_b_output_bytes"] != 32
    ):
        raise SystemExit(f"unexpected CLI payload: {payload}")
    expected = reference()
    for filename, values in expected.items():
        check_f32(output_dir / filename, values)

    cache_cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-cache-write",
        str(root / "cache_layout.json"),
        str(root / "decode_cache.bin"),
        "--layer",
        "1",
        "--input-f32",
        str(output_dir / "kv_a_proj_with_mqa.f32"),
        "--start-position",
        "1",
        "--batch-tokens",
        "2",
        "--max-cache-file-mib",
        "1",
        "--max-cache-write-mib",
        "1",
        "--json",
    ]
    cache_completed = subprocess.run(cache_cmd, text=True, capture_output=True)
    if cache_completed.stdout:
        print(cache_completed.stdout, end="")
    if cache_completed.stderr:
        print(cache_completed.stderr, end="")
    if cache_completed.returncode != 0:
        raise SystemExit(cache_completed.returncode)
    cache_payload = json.loads(cache_completed.stdout)
    if cache_payload["encoded_bytes"] != 16 or cache_payload["first_write_offset"] != 16:
        raise SystemExit(f"unexpected cache payload: {cache_payload}")
    cache = (root / "decode_cache.bin").read_bytes()
    kv_a = expected["kv_a_proj_with_mqa.f32"]
    if cache[16:24] != bf16(kv_a[:4]) or cache[24:32] != bf16(kv_a[4:]):
        raise SystemExit("prefill cache write bytes did not match expected KV-A rows")

    rope_dir = root / "rope"
    rope_cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-rope-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        "--q-b-f32",
        str(output_dir / "q_b_proj.f32"),
        "--k-rope-f32",
        str(output_dir / "kv_a_rope.f32"),
        "--output-dir",
        str(rope_dir),
        "--batch-tokens",
        "2",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "2",
        "--rope-dim",
        "2",
        "--start-position",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--quiet-runner",
        "--json",
    ]
    rope_completed = subprocess.run(rope_cmd, text=True, capture_output=True)
    if rope_completed.stdout:
        print(rope_completed.stdout, end="")
    if rope_completed.stderr:
        print(rope_completed.stderr, end="")
    if rope_completed.returncode != 0:
        raise SystemExit(rope_completed.returncode)
    rope_payload = json.loads(rope_completed.stdout)
    if rope_payload["q_nope_bytes"] != 16 or rope_payload["q_rope_bytes"] != 16:
        raise SystemExit(f"unexpected rope payload: {rope_payload}")
    q_b = expected["q_b_proj.f32"]
    q_nope = q_b[:2] + q_b[4:6]
    q_rope = q_b[2:4] + q_b[6:8]
    q_rope_rot = rotate(q_b[2:4], 1, 10000.0, False) + rotate(q_b[6:8], 2, 10000.0, False)
    k_rope_rot = rotate(kv_a[2:4], 1, 10000.0, False) + rotate(kv_a[6:8], 2, 10000.0, False)
    check_f32(rope_dir / "q_nope.f32", q_nope)
    check_f32(rope_dir / "q_rope.f32", q_rope)
    check_f32(rope_dir / "q_rope_rotated.f32", q_rope_rot)
    check_f32(rope_dir / "k_rope_rotated.f32", k_rope_rot)

    attn_out = root / "attn_value.f32"
    attn_cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-mla-attention-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        str(root / "cache_layout.json"),
        str(root / "decode_cache.bin"),
        "--layer",
        "1",
        "--q-nope-f32",
        str(rope_dir / "q_nope.f32"),
        "--q-rope-f32",
        str(rope_dir / "q_rope_rotated.f32"),
        "--output-f32",
        str(attn_out),
        "--context-length",
        "3",
        "--start-position",
        "1",
        "--batch-tokens",
        "2",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "2",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "2",
        "--kv-lora-dim",
        "2",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--quiet-runner",
        "--json",
    ]
    attn_completed = subprocess.run(attn_cmd, text=True, capture_output=True)
    if attn_completed.stdout:
        print(attn_completed.stdout, end="")
    if attn_completed.stderr:
        print(attn_completed.stderr, end="")
    if attn_completed.returncode != 0:
        raise SystemExit(attn_completed.returncode)
    attn_payload = json.loads(attn_completed.stdout)
    if attn_payload["output_bytes"] != 16 or attn_payload["cache_read_bytes"] != 24:
        raise SystemExit(f"unexpected attention payload: {attn_payload}")
    cache_rows = (
        (0.0, 0.0, 0.0, 0.0),
        tuple(bf16_round(value) for value in kv_a[:4]),
        tuple(bf16_round(value) for value in kv_a[4:]),
    )
    attn_value_expected = attention_reference(
        q_nope, q_rope_rot, cache_rows, start_position=1
    )
    check_f32(attn_out, attn_value_expected)

    attn_hidden = root / "attn_hidden.f32"
    attn_output_cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-attention-output-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        "--layer",
        "1",
        "--attn-value-f32",
        str(attn_out),
        "--residual-f32",
        str(root / "input.f32"),
        "--output-f32",
        str(attn_hidden),
        "--batch-tokens",
        "2",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--quiet-runner",
        "--json",
    ]
    attn_output_completed = subprocess.run(
        attn_output_cmd, text=True, capture_output=True
    )
    if attn_output_completed.stdout:
        print(attn_output_completed.stdout, end="")
    if attn_output_completed.stderr:
        print(attn_output_completed.stderr, end="")
    if attn_output_completed.returncode != 0:
        raise SystemExit(attn_output_completed.returncode)
    attn_output_payload = json.loads(attn_output_completed.stdout)
    if (
        attn_output_payload["hidden_dim"] != 3
        or attn_output_payload["projection_bytes"] != 24
        or attn_output_payload["output_bytes"] != 24
    ):
        raise SystemExit(f"unexpected attention output payload: {attn_output_payload}")
    attn_hidden_expected = attention_output_reference(attn_value_expected)
    check_f32(attn_hidden, attn_hidden_expected)

    block_hidden = root / "attn_block_hidden.f32"
    block_cmd = [
        sys.executable,
        "-m",
        "largerlm",
        "prefill-attention-block-batch",
        str(Path(__file__).with_name("largerlm-runner")),
        str(root / "resident" / "layout.json"),
        str(root / "cache_layout.json"),
        str(root / "decode_cache.bin"),
        "--layer",
        "1",
        "--input-f32",
        str(root / "input.f32"),
        "--output-dir",
        str(root / "attention_block"),
        "--output-f32",
        str(block_hidden),
        "--start-position",
        "1",
        "--batch-tokens",
        "2",
        "--num-heads",
        "1",
        "--qk-nope-dim",
        "2",
        "--rope-dim",
        "2",
        "--v-head-dim",
        "2",
        "--rms-norm-eps",
        "0",
        "--max-cache-file-mib",
        "1",
        "--max-cache-write-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-resident-matrix-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
        "--quiet-runner",
        "--json",
    ]
    block_completed = subprocess.run(block_cmd, text=True, capture_output=True)
    if block_completed.stdout:
        print(block_completed.stdout, end="")
    if block_completed.stderr:
        print(block_completed.stderr, end="")
    if block_completed.returncode != 0:
        raise SystemExit(block_completed.returncode)
    block_payload = json.loads(block_completed.stdout)
    if (
        block_payload["context_length"] != 3
        or block_payload["output_bytes"] != 24
        or block_payload["cache_write"]["encoded_bytes"] != 16
    ):
        raise SystemExit(f"unexpected attention block payload: {block_payload}")
    check_f32(block_hidden, attn_hidden_expected)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="largerlm-attention-projections-", dir="/private/tmp"))
    write_fixture(root)
    run_cli_case(root)
    print(f"fixture: {root}")
    print("  prefill attention projections: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
