from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from largerlm.baseline import BaselineError, BaselineWriter, validate_baseline


def test_baseline_writer_records_raw_tensors(tmp_path: Path) -> None:
    with BaselineWriter(
        tmp_path,
        model_type="glm_moe_dsa",
        prompt_tokens=[1, 2, 3],
        metadata={"source": "unit"},
    ) as writer:
        tensor = writer.write_tensor(
            name="layer.1.router_logits",
            dtype="float32",
            shape=(1, 2),
            data=b"\x00\x00\x80?\x00\x00\x00@",
        )
        writer.add_record(
            kind="decode_layer",
            token_index=0,
            layer=1,
            tensors=[tensor],
            metadata={"topk": [0]},
        )
        writer.add_generated_token(42)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["model_type"] == "glm_moe_dsa"
    assert manifest["generated_tokens"] == [42]
    recorded = manifest["records"][0]["tensors"][0]
    assert recorded["path"] == "tensors/000001_layer.1.router_logits.bin"
    assert recorded["sha256"] == hashlib.sha256(
        b"\x00\x00\x80?\x00\x00\x00@"
    ).hexdigest()
    assert (tmp_path / recorded["path"]).read_bytes() == b"\x00\x00\x80?\x00\x00\x00@"

    validation = validate_baseline(tmp_path)
    assert validation.ok is True
    assert validation.tensor_count == 1
    assert validation.total_bytes == 8


def test_baseline_writer_cleans_tensor_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = BaselineWriter(
        tmp_path,
        model_type="glm_moe_dsa",
        prompt_tokens=[1],
    )
    writer.__enter__()

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace exploded")

    monkeypatch.setattr(type(tmp_path), "replace", fail_replace)

    with pytest.raises(OSError, match="replace exploded"):
        writer.write_tensor(
            name="layer.1.hidden",
            dtype="float32",
            shape=(1, 1),
            data=b"\x00\x00\x80?",
        )

    assert not any(tmp_path.rglob("*.tmp"))
    assert not list((tmp_path / "tensors").glob("*.bin"))


def test_baseline_writer_cleans_manifest_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = BaselineWriter(
        tmp_path,
        model_type="glm_moe_dsa",
        prompt_tokens=[1],
    )
    writer.__enter__()

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace exploded")

    monkeypatch.setattr(type(tmp_path), "replace", fail_replace)

    with pytest.raises(OSError, match="replace exploded"):
        writer.write_manifest()

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "manifest.json.tmp").exists()


def _write_valid_baseline(root: Path) -> dict:
    with BaselineWriter(
        root,
        model_type="glm_moe_dsa",
        prompt_tokens=[1],
    ) as writer:
        tensor = writer.write_tensor(
            name="layer.1.hidden",
            dtype="float32",
            shape=(1, 1),
            data=b"\x00\x00\x80?",
        )
        writer.add_record(
            kind="decode_layer",
            token_index=0,
            layer=1,
            tensors=[tensor],
        )
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_validate_baseline_rejects_tensor_path_traversal(tmp_path: Path) -> None:
    manifest = _write_valid_baseline(tmp_path)
    manifest["records"][0]["tensors"][0]["path"] = "../outside.bin"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BaselineError, match="invalid tensor path"):
        validate_baseline(tmp_path)


def test_validate_baseline_rejects_bad_tensor_shape(tmp_path: Path) -> None:
    manifest = _write_valid_baseline(tmp_path)
    manifest["records"][0]["tensors"][0]["shape"] = [1, 0]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BaselineError, match="shape\\[1\\]"):
        validate_baseline(tmp_path)


def test_validate_baseline_rejects_bad_sha256_metadata(tmp_path: Path) -> None:
    manifest = _write_valid_baseline(tmp_path)
    manifest["records"][0]["tensors"][0]["sha256"] = "not-a-hash"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BaselineError, match="sha256"):
        validate_baseline(tmp_path)
