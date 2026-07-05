from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from largerlm.cli import main as cli_main
from largerlm.embedding import EmbeddingError, embed_token, embed_tokens_batch


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", value), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def bf16(values: list[float]) -> bytes:
    return b"".join(f32_to_bf16(value) for value in values)


def pack8(code: int) -> int:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return packed


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    return struct.unpack(f"<{count}f", path.read_bytes())


def write_resident(root: Path, *, dtype: str = "F32") -> Path:
    resident = root / "resident"
    resident.mkdir()
    values = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0]
    data = f32(values) if dtype == "F32" else bf16(values)
    (resident / "resident.bin").write_bytes(data)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(data),
        "tensors": [
            {
                "name": "model.embed_tokens.weight",
                "offset": 0,
                "size": len(data),
                "dtype": dtype,
                "shape": [3, 2],
                "category": "embeddings",
            }
        ],
    }
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def write_mxfp4_resident(root: Path) -> Path:
    resident = root / "resident"
    resident.mkdir()
    hidden = 32
    vocab = 3
    weights = bytearray()
    for code in (2, 1, 10):
        for _ in range(hidden // 8):
            weights.extend(struct.pack("<I", pack8(code)))
    scales = bytes([127] * vocab)
    data = bytes(weights) + scales
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": len(data),
        "tensors": [
            {
                "name": "model.embed_tokens.weight",
                "offset": 0,
                "size": len(weights),
                "dtype": "U32",
                "shape": [vocab, hidden // 8],
                "category": "embeddings",
            },
            {
                "name": "model.embed_tokens.scales",
                "offset": len(weights),
                "size": len(scales),
                "dtype": "U8",
                "shape": [vocab, 1],
                "category": "embeddings",
            },
        ],
    }
    (resident / "resident.bin").write_bytes(data)
    layout_path = resident / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    return layout_path


def mutate_layout(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_embed_token_streams_one_f32_row(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    out = tmp_path / "hidden.f32"

    result = embed_token(layout, token_id=1, output_f32_path=out)

    assert result.tensor == "model.embed_tokens.weight"
    assert result.read_bytes == 8
    assert read_f32(out, 2) == (3.0, 4.0)


def test_embed_token_supports_bf16(tmp_path: Path) -> None:
    layout = write_resident(tmp_path, dtype="BF16")
    out = tmp_path / "hidden.f32"

    embed_token(layout, token_id=2, output_f32_path=out)

    assert read_f32(out, 2) == (-1.0, -2.0)


def test_embed_token_supports_mxfp4(tmp_path: Path) -> None:
    layout = write_mxfp4_resident(tmp_path)
    out = tmp_path / "hidden.f32"

    result = embed_token(layout, token_id=1, output_f32_path=out)

    assert result.dtype == "mlx-mxfp4"
    assert result.hidden_dim == 32
    assert result.read_bytes == 17
    assert result.output_bytes == 128
    assert read_f32(out, 32) == pytest.approx((0.5,) * 32)


def test_embed_tokens_batch_streams_f32_rows(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    out = tmp_path / "prompt.f32"

    result = embed_tokens_batch(layout, token_ids=(2, 0, 1), output_f32_path=out)

    assert result.tensor == "model.embed_tokens.weight"
    assert result.token_count == 3
    assert result.first_token_id == 2
    assert result.last_token_id == 1
    assert result.read_bytes == 24
    assert result.output_bytes == 24
    assert read_f32(out, 6) == (-1.0, -2.0, 1.0, 2.0, 3.0, 4.0)


def test_embed_tokens_batch_supports_bf16(tmp_path: Path) -> None:
    layout = write_resident(tmp_path, dtype="BF16")
    out = tmp_path / "prompt.f32"

    result = embed_tokens_batch(layout, token_ids=(0, 2), output_f32_path=out)

    assert result.read_bytes == 8
    assert result.output_bytes == 16
    assert read_f32(out, 4) == (1.0, 2.0, -1.0, -2.0)


def test_embed_tokens_batch_supports_mxfp4(tmp_path: Path) -> None:
    layout = write_mxfp4_resident(tmp_path)
    out = tmp_path / "prompt.f32"

    result = embed_tokens_batch(layout, token_ids=(2, 0), output_f32_path=out)

    assert result.dtype == "mlx-mxfp4"
    assert result.read_bytes == 34
    assert result.output_bytes == 256
    assert read_f32(out, 64) == pytest.approx((-1.0,) * 32 + (1.0,) * 32)


def test_embed_tokens_batch_rejects_output_limit(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)

    with pytest.raises(EmbeddingError, match="batch output"):
        embed_tokens_batch(
            layout,
            token_ids=(0, 1),
            output_f32_path=tmp_path / "prompt.f32",
            max_output_bytes=8,
        )


def test_embed_token_rejects_out_of_range_token(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)

    with pytest.raises(EmbeddingError, match="token_id"):
        embed_token(layout, token_id=3, output_f32_path=tmp_path / "hidden.f32")


def test_embed_token_rejects_expected_vocab_size_mismatch(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)

    with pytest.raises(
        EmbeddingError,
        match="embedding vocab size 3 does not match expected_vocab_size 4",
    ):
        embed_token(
            layout,
            token_id=0,
            output_f32_path=tmp_path / "hidden.f32",
            expected_vocab_size=4,
        )


def test_embed_tokens_batch_rejects_expected_hidden_size_mismatch(
    tmp_path: Path,
) -> None:
    layout = write_resident(tmp_path)

    with pytest.raises(
        EmbeddingError,
        match="embedding hidden dim 2 does not match expected_hidden_size 3",
    ):
        embed_tokens_batch(
            layout,
            token_ids=(0, 1),
            output_f32_path=tmp_path / "prompt.f32",
            expected_hidden_size=3,
        )


def test_embed_token_rejects_boolean_shape(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["shape"][1] = True

    mutate_layout(layout, mutate)

    with pytest.raises(EmbeddingError, match="shape \\[vocab, hidden\\]"):
        embed_token(layout, token_id=0, output_f32_path=tmp_path / "hidden.f32")


def test_embed_token_rejects_boolean_size(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["size"] = False

    mutate_layout(layout, mutate)

    with pytest.raises(EmbeddingError, match="embedding size must be an integer"):
        embed_token(layout, token_id=0, output_f32_path=tmp_path / "hidden.f32")


def test_embed_token_rejects_boolean_integer_arguments(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)

    with pytest.raises(EmbeddingError, match="token_id must be an integer"):
        embed_token(layout, token_id=True, output_f32_path=tmp_path / "hidden.f32")

    with pytest.raises(EmbeddingError, match="max_row_bytes must be an integer"):
        embed_token(
            layout,
            token_id=0,
            output_f32_path=tmp_path / "hidden.f32",
            max_row_bytes=True,
        )

    with pytest.raises(EmbeddingError, match="expected_vocab_size must be an integer"):
        embed_token(
            layout,
            token_id=0,
            output_f32_path=tmp_path / "hidden.f32",
            expected_vocab_size=False,
        )

    batch_output = tmp_path / "prompt.f32"
    with pytest.raises(EmbeddingError, match="token_id at position 1 must be an integer"):
        embed_tokens_batch(
            layout,
            token_ids=(0, False),
            output_f32_path=batch_output,
        )
    assert not batch_output.exists()

    with pytest.raises(EmbeddingError, match="token_id at position 1 must be an integer"):
        embed_tokens_batch(
            layout,
            token_ids=(0, 1.5),
            output_f32_path=batch_output,
        )
    assert not batch_output.exists()

    with pytest.raises(EmbeddingError, match="expected_hidden_size must be an integer"):
        embed_tokens_batch(
            layout,
            token_ids=(0, 1),
            output_f32_path=batch_output,
            expected_hidden_size=True,
        )
    assert not batch_output.exists()


def test_embed_token_rejects_truncated_resident_before_output(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    (layout.parent / "resident.bin").write_bytes(b"\0" * 23)
    out = tmp_path / "hidden.f32"

    with pytest.raises(EmbeddingError, match="resident weight file"):
        embed_token(layout, token_id=0, output_f32_path=out)

    assert not out.exists()


def test_embed_token_cli_writes_output(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    out = tmp_path / "hidden.f32"

    status = cli_main(
        [
            "embed-token",
            str(layout),
            "--token-id",
            "0",
            "--output-f32",
            str(out),
        ]
    )

    assert status == 0
    assert read_f32(out, 2) == (1.0, 2.0)


def test_embed_tokens_batch_cli_writes_output(tmp_path: Path) -> None:
    layout = write_resident(tmp_path)
    out = tmp_path / "prompt.f32"

    status = cli_main(
        [
            "embed-tokens-batch",
            str(layout),
            "--token-ids",
            "2,0,1",
            "--output-f32",
            str(out),
            "--json",
        ]
    )

    assert status == 0
    assert read_f32(out, 6) == (-1.0, -2.0, 1.0, 2.0, 3.0, 4.0)
