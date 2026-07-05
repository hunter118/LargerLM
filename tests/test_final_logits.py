from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from largerlm import final_logits as final_logits_module
from largerlm.final_logits import FinalLogitsError, compute_final_logits


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(v) for v in values)


def pack8(values: list[int]) -> int:
    out = 0
    for index, value in enumerate(values):
        out |= (value & 0xF) << (index * 4)
    return out


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def write_resident(
    root: Path,
    *,
    use_lm_head: bool = True,
    head_dtype: str = "F32",
) -> Path:
    resident = root / "resident"
    resident.mkdir()
    tensors = []
    payload = bytearray()

    def add(name: str, dtype: str, shape: list[int], data: bytes) -> None:
        nonlocal payload
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": "lm_head" if "lm_head" in name else "norms",
            }
        )
        payload.extend(data)

    add("model.norm.weight", "F32", [2], f32([1.0, 1.0]))
    head_values = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0]
    head_name = "lm_head.weight" if use_lm_head else "model.embed_tokens.weight"
    head_data = f32(head_values) if head_dtype == "F32" else bf16(head_values)
    add(head_name, head_dtype, [4, 2], head_data)

    (resident / "resident.bin").write_bytes(payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": tensors,
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def write_affine_resident(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir()
    tensors = []
    payload = bytearray()

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

    add("model.norm.weight", "F32", [8], f32([1.0] * 8), "norms")
    add(
        "lm_head.weight",
        "U32",
        [4, 1],
        struct.pack("<4I", *(pack8([value] * 8) for value in (0, 1, 2, 3))),
        "lm_head",
    )
    add(
        "lm_head.scales",
        "BF16",
        [4, 1],
        bf16([1.0] * 4),
        "lm_head",
    )
    add(
        "lm_head.biases",
        "BF16",
        [4, 1],
        bf16([0.0] * 4),
        "lm_head",
    )
    (resident / "resident.bin").write_bytes(payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": tensors,
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def write_mxfp4_resident(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir()
    tensors = []
    payload = bytearray()

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

    hidden_dim = 32
    out_dim = 4
    group_size = 32
    add("model.norm.weight", "F32", [hidden_dim], f32([1.0] * hidden_dim), "norms")
    add(
        "lm_head.weight",
        "U32",
        [out_dim, hidden_dim // 8],
        struct.pack(
            "<16I",
            *(pack8([value] * 8) for value in (2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 10, 10, 10, 10)),
        ),
        "lm_head",
    )
    add(
        "lm_head.scales",
        "U8",
        [out_dim, hidden_dim // group_size],
        bytes([127] * out_dim),
        "lm_head",
    )
    (resident / "resident.bin").write_bytes(payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(payload),
        "tensors": tensors,
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def mutate_layout(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compute_final_logits_streams_topk_and_logits(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    logits = tmp_path / "logits.f32"
    topk = tmp_path / "topk.json"
    hidden.write_bytes(f32([1.0, 2.0]))

    result = compute_final_logits(
        layout,
        hidden,
        output_logits_f32_path=logits,
        output_topk_json_path=topk,
        top_k=2,
        rms_norm_eps=0.0,
        chunk_rows=2,
        max_chunk_bytes=16,
    )

    inv = 1.0 / math.sqrt(2.5)
    expected = (inv, 2.0 * inv, 3.0 * inv, -inv)
    assert result.chunks == 2
    assert result.topk[0].token_id == 2
    assert result.topk[1].token_id == 1
    assert read_f32(logits, 4) == pytest.approx(expected)
    payload = json.loads(topk.read_text(encoding="utf-8"))
    assert [item["token_id"] for item in payload["topk"]] == [2, 1]


def test_compute_final_logits_topk_json_replace_failure_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    topk = tmp_path / "topk.json"
    hidden.write_bytes(f32([1.0, 2.0]))
    topk.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(final_logits_module.Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        compute_final_logits(
            layout,
            hidden,
            output_topk_json_path=topk,
            top_k=2,
            rms_norm_eps=0.0,
            max_chunk_bytes=16,
        )

    assert json.loads(topk.read_text(encoding="utf-8")) == {"old": True}
    assert not topk.with_name(topk.name + ".tmp").exists()


def test_compute_final_logits_output_logits_replace_failure_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    logits = tmp_path / "logits.f32"
    hidden.write_bytes(f32([1.0, 2.0]))
    logits.write_bytes(b"old")

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(final_logits_module.Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        compute_final_logits(
            layout,
            hidden,
            output_logits_f32_path=logits,
            top_k=2,
            rms_norm_eps=0.0,
            max_chunk_bytes=16,
        )

    assert logits.read_bytes() == b"old"
    assert not logits.with_name(logits.name + ".tmp").exists()


def test_compute_final_logits_uses_tied_embedding_fallback(tmp_path: Path) -> None:
    layout = write_resident(tmp_path, use_lm_head=False, head_dtype="BF16")
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    result = compute_final_logits(
        layout,
        hidden,
        top_k=1,
        rms_norm_eps=0.0,
        max_chunk_bytes=16,
    )

    assert result.head_tensor == "model.embed_tokens.weight"
    assert result.topk[0].token_id == 2


def test_compute_final_logits_streams_affine_int4_lm_head(
    tmp_path: Path,
) -> None:
    layout = write_affine_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    logits = tmp_path / "logits.f32"
    topk = tmp_path / "topk.json"
    hidden.write_bytes(f32([1.0] * 8))

    result = compute_final_logits(
        layout,
        hidden,
        output_logits_f32_path=logits,
        output_topk_json_path=topk,
        top_k=2,
        rms_norm_eps=0.0,
        chunk_rows=2,
        max_chunk_bytes=16,
    )

    assert result.dtype == "affine-int4"
    assert result.hidden_dim == 8
    assert result.vocab_size == 4
    assert result.chunk_rows == 2
    assert result.chunks == 2
    assert result.read_bytes == 32
    assert [record.token_id for record in result.topk] == [3, 2]
    assert read_f32(logits, 4) == pytest.approx((0.0, 8.0, 16.0, 24.0))
    payload = json.loads(topk.read_text(encoding="utf-8"))
    assert [item["token_id"] for item in payload["topk"]] == [3, 2]


def test_compute_final_logits_streams_mxfp4_lm_head(
    tmp_path: Path,
) -> None:
    layout = write_mxfp4_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    logits = tmp_path / "logits.f32"
    topk = tmp_path / "topk.json"
    hidden.write_bytes(f32([1.0 / 32.0] * 32))

    result = compute_final_logits(
        layout,
        hidden,
        output_logits_f32_path=logits,
        output_topk_json_path=topk,
        top_k=2,
        chunk_rows=2,
        max_chunk_bytes=34,
        skip_final_norm=True,
    )

    assert result.dtype == "mlx-mxfp4"
    assert result.hidden_dim == 32
    assert result.vocab_size == 4
    assert result.chunk_rows == 2
    assert result.chunks == 2
    assert result.read_bytes == 68
    assert [record.token_id for record in result.topk] == [0, 1]
    assert read_f32(logits, 4) == pytest.approx((1.0, 0.5, 0.0, -1.0))
    payload = json.loads(topk.read_text(encoding="utf-8"))
    assert [item["token_id"] for item in payload["topk"]] == [0, 1]


def test_compute_final_logits_rejects_expected_vocab_size_mismatch(
    tmp_path: Path,
) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    with pytest.raises(
        FinalLogitsError,
        match="lm_head/embedding vocab size 4 does not match expected_vocab_size 5",
    ):
        compute_final_logits(
            layout,
            hidden,
            top_k=1,
            expected_vocab_size=5,
            max_chunk_bytes=16,
        )


def test_compute_final_logits_rejects_expected_hidden_size_mismatch(
    tmp_path: Path,
) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    with pytest.raises(
        FinalLogitsError,
        match="lm_head/embedding hidden dim 2 does not match expected_hidden_size 3",
    ):
        compute_final_logits(
            layout,
            hidden,
            top_k=1,
            expected_hidden_size=3,
            max_chunk_bytes=16,
        )


def test_compute_final_logits_rejects_row_over_chunk_cap(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    with pytest.raises(FinalLogitsError, match="row"):
        compute_final_logits(
            layout,
            hidden,
            top_k=1,
            max_chunk_bytes=7,
        )


def test_compute_final_logits_rejects_boolean_head_shape(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[1]["shape"][1] = True

    mutate_layout(layout, mutate)

    with pytest.raises(FinalLogitsError, match="lm_head/embedding must have shape"):
        compute_final_logits(layout, hidden, top_k=1, max_chunk_bytes=16)


def test_compute_final_logits_rejects_boolean_norm_shape(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["shape"][0] = False

    mutate_layout(layout, mutate)

    with pytest.raises(FinalLogitsError, match="final norm must have shape"):
        compute_final_logits(layout, hidden, top_k=1, max_chunk_bytes=16)


def test_compute_final_logits_rejects_boolean_tensor_size(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[1]["size"] = True

    mutate_layout(layout, mutate)

    with pytest.raises(
        FinalLogitsError,
        match="lm_head/embedding size must be an integer",
    ):
        compute_final_logits(layout, hidden, top_k=1, max_chunk_bytes=16)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": True}, "top_k must be an integer"),
        ({"chunk_rows": False}, "chunk_rows must be an integer"),
        ({"rms_norm_eps": False}, "rms_norm_eps must be a finite number"),
        ({"rms_norm_eps": float("nan")}, "rms_norm_eps must be a finite number"),
        ({"max_chunk_bytes": True}, "max_chunk_bytes must be an integer"),
        (
            {"max_output_logits_bytes": True},
            "max_output_logits_bytes must be an integer",
        ),
        (
            {"expected_hidden_size": False},
            "expected_hidden_size must be an integer",
        ),
    ],
)
def test_compute_final_logits_rejects_boolean_integer_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))
    args: dict[str, object] = {"top_k": 1, "max_chunk_bytes": 16}
    args.update(kwargs)

    with pytest.raises(FinalLogitsError, match=message):
        compute_final_logits(layout, hidden, **args)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"top_k": True}, "top_k must be an integer"),
        ({"rms_norm_eps": False}, "rms_norm_eps must be a finite number"),
        (
            {"expected_hidden_size": False},
            "expected_hidden_size must be an integer",
        ),
    ),
)
def test_compute_final_logits_metal_rejects_invalid_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(FinalLogitsError, match=message):
        final_logits_module.compute_final_logits_metal(
            runner_path=tmp_path / "missing-runner",
            resident_layout_path=tmp_path / "missing-layout.json",
            input_f32_path=tmp_path / "missing-hidden.f32",
            **kwargs,
        )


def test_compute_final_logits_metal_rejects_boolean_topk_token_id(
    tmp_path: Path,
) -> None:
    layout = write_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))
    runner = tmp_path / "fake_final_logits.py"
    runner.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

out = sys.argv[sys.argv.index("--output-topk-json") + 1]
open(out, "w", encoding="utf-8").write(
    json.dumps({"topk": [{"token_id": True, "logit": 1.0}]})
)
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    with pytest.raises(FinalLogitsError, match=r"topk\[0\].token_id"):
        final_logits_module.compute_final_logits_metal(
            runner_path=runner,
            resident_layout_path=layout,
            input_f32_path=hidden,
            top_k=1,
            max_chunk_bytes=16,
            echo_runner_output=False,
        )


def test_compute_final_logits_metal_accepts_affine_int4_lm_head_metadata(
    tmp_path: Path,
) -> None:
    layout = write_affine_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0] * 8))
    runner = tmp_path / "fake_final_logits.py"
    runner.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

out = sys.argv[sys.argv.index("--output-topk-json") + 1]
open(out, "w", encoding="utf-8").write(
    json.dumps({"topk": [{"token_id": 3, "logit": 24.0}, {"token_id": 2, "logit": 16.0}]})
)
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    result = final_logits_module.compute_final_logits_metal(
        runner_path=runner,
        resident_layout_path=layout,
        input_f32_path=hidden,
        top_k=2,
        chunk_rows=2,
        max_chunk_bytes=16,
        echo_runner_output=False,
    )

    assert result.dtype == "affine-int4"
    assert result.hidden_dim == 8
    assert result.vocab_size == 4
    assert result.chunk_rows == 2
    assert result.chunks == 2
    assert result.read_bytes == 32
    assert [record.token_id for record in result.topk] == [3, 2]


def test_compute_final_logits_metal_accepts_mxfp4_lm_head_metadata(
    tmp_path: Path,
) -> None:
    layout = write_mxfp4_resident(tmp_path)
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0 / 32.0] * 32))
    runner = tmp_path / "fake_final_logits.py"
    runner.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

out = sys.argv[sys.argv.index("--output-topk-json") + 1]
open(out, "w", encoding="utf-8").write(
    json.dumps({"topk": [{"token_id": 0, "logit": 1.0}, {"token_id": 1, "logit": 0.5}]})
)
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    result = final_logits_module.compute_final_logits_metal(
        runner_path=runner,
        resident_layout_path=layout,
        input_f32_path=hidden,
        top_k=2,
        chunk_rows=2,
        max_chunk_bytes=34,
        skip_final_norm=True,
        echo_runner_output=False,
    )

    assert result.dtype == "mlx-mxfp4"
    assert result.hidden_dim == 32
    assert result.vocab_size == 4
    assert result.chunk_rows == 2
    assert result.chunks == 2
    assert result.read_bytes == 68
    assert [record.token_id for record in result.topk] == [0, 1]


def test_compute_final_logits_rejects_truncated_head_before_outputs(
    tmp_path: Path,
) -> None:
    layout = write_resident(tmp_path)
    resident_bin = layout.parent / "resident.bin"
    resident_bin.write_bytes(resident_bin.read_bytes()[:-1])
    hidden = tmp_path / "hidden.f32"
    hidden.write_bytes(f32([1.0, 2.0]))
    logits = tmp_path / "logits.f32"
    topk = tmp_path / "topk.json"

    with pytest.raises(FinalLogitsError, match="resident weight file"):
        compute_final_logits(
            layout,
            hidden,
            output_logits_f32_path=logits,
            output_topk_json_path=topk,
            top_k=1,
            max_chunk_bytes=16,
        )

    assert not logits.exists()
    assert not logits.with_name(logits.name + ".tmp").exists()
    assert not topk.exists()
