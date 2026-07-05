from __future__ import annotations

import json
import struct
from pathlib import Path

from largerlm.baseline import load_manifest, validate_baseline
from largerlm.mlx_baseline import export_mlx_baseline, preflight_mlx_baseline


def _write_tiny_checkpoint(root: Path, payload_size: int = 8) -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 1,
                "n_routed_experts": 1,
                "num_experts_per_tok": 1,
            }
        ),
        encoding="utf-8",
    )
    shard = root / "model-00001-of-00001.safetensors"
    name = "model.embed_tokens.weight"
    header = {
        name: {
            "dtype": "U8",
            "shape": [payload_size],
            "data_offsets": [0, payload_size],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"x" * payload_size)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard.name}}),
        encoding="utf-8",
    )


class FakeTensor:
    shape = (1, 4)
    dtype = "float32"

    def __getitem__(self, _key):
        return self

    def tobytes(self) -> bytes:
        return b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"


class FakeToken:
    def item(self) -> int:
        return 3


class FakeMx:
    @staticmethod
    def array(value):
        return value

    @staticmethod
    def eval(*_args):
        return None

    @staticmethod
    def argmax(_value, axis=-1):
        return FakeToken()


class FakeTokenizer:
    def encode(self, prompt: str):
        return [ord(ch) % 13 for ch in prompt]


class FakeModel:
    def make_cache(self):
        return []

    def __call__(self, _input_ids, cache=None):
        return FakeTensor()


def test_mlx_preflight_rejects_large_checkpoint(tmp_path: Path) -> None:
    _write_tiny_checkpoint(tmp_path, payload_size=128)

    report = preflight_mlx_baseline(
        tmp_path,
        tmp_path / "baseline",
        max_model_load_bytes=64,
    )

    assert report.safe_to_load is False
    assert "exceed" in report.reason


def test_export_mlx_baseline_with_fake_stack(tmp_path: Path) -> None:
    _write_tiny_checkpoint(tmp_path)
    output = tmp_path / "baseline"

    result = export_mlx_baseline(
        tmp_path,
        output,
        prompt="hi",
        max_tokens=1,
        max_model_load_bytes=1024,
        mx_module=FakeMx(),
        load_fn=lambda _path: (FakeModel(), FakeTokenizer()),
    )

    assert result.generated_tokens == (3,)
    assert result.tensor_count == 1
    validation = validate_baseline(output)
    assert validation.ok is True
    manifest = load_manifest(output)
    assert manifest["records"][0]["metadata"]["next_token"] == 3
    assert manifest["records"][0]["tensors"][0]["shape"] == [1, 4]
