from __future__ import annotations

import json
from pathlib import Path

from scripts.glm_prefill_cache_write_sweep import _config_comparison, main


def _write_prepared_cache_layout(root: Path) -> Path:
    prepared = root / "prepared"
    prepared.mkdir()
    payload = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "max_context_tokens": 4096,
        "dtype": "BF16",
        "dtype_bytes": 2,
        "alignment": 64,
        "total_bytes": 64,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": 7,
                "offset": 0,
                "width": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "token_stride_bytes": 8,
                "max_context_tokens": 8,
                "total_bytes": 64,
            }
        ],
    }
    (prepared / "decode_cache_layout.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return prepared


def test_config_comparison_keeps_default_baseline_when_fastest() -> None:
    comparison = _config_comparison(
        {
            "default": {
                "elapsed_seconds": {"count": 4, "mean": 0.10},
                "max_diff_bytes_vs_baseline": 0,
            },
            "one-row": {
                "elapsed_seconds": {"count": 4, "mean": 0.13},
                "max_diff_bytes_vs_baseline": 0,
            },
        }
    )

    assert comparison["baseline_mode"] == "default"
    assert comparison["fastest_wall_mode"] == "default"
    assert comparison["candidate_mode"] is None
    assert comparison["candidate_for_full_replay"] is False


def test_config_comparison_can_promote_non_default_chunk() -> None:
    comparison = _config_comparison(
        {
            "default": {
                "elapsed_seconds": {"count": 4, "mean": 0.10},
                "max_diff_bytes_vs_baseline": 0,
            },
            "256KiB": {
                "elapsed_seconds": {"count": 4, "mean": 0.09},
                "max_diff_bytes_vs_baseline": 0,
            },
        }
    )

    assert comparison["candidate_mode"] == "256KiB"
    assert comparison["candidate_for_full_replay"] is True
    assert comparison["requires_full_replay_bakeoff"] is True


def test_config_comparison_rejects_byte_drift() -> None:
    comparison = _config_comparison(
        {
            "default": {
                "elapsed_seconds": {"count": 4, "mean": 0.10},
                "max_diff_bytes_vs_baseline": 0,
            },
            "256KiB": {
                "elapsed_seconds": {"count": 4, "mean": 0.09},
                "max_diff_bytes_vs_baseline": 1,
            },
        }
    )

    assert comparison["candidate_for_full_replay"] is False
    row = next(row for row in comparison["rows"] if row["mode"] == "256KiB")
    assert row["byte_identical_within_promotion_policy"] is False


def test_main_writes_bounded_cache_write_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_prepared_cache_layout(tmp_path)
    result_path = tmp_path / "cache-write-sweep.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "glm_prefill_cache_write_sweep.py",
            "--prepared-dir",
            str(prepared),
            "--layer",
            "7",
            "--batch-tokens",
            "3",
            "--repeat",
            "2",
            "--chunk-modes",
            "default,one-row",
            "--write-result",
            str(result_path),
        ],
    )

    assert main() == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "largerlm.glm_prefill_cache_write_sweep.v1"
    assert payload["layer"] == 7
    assert payload["batch_tokens"] == 3
    assert payload["encoded_bytes"] == 24
    assert payload["work_dir_cleaned"] is True
    assert set(payload["by_mode"]) == {"default", "one-row"}
    assert payload["config_comparison"]["baseline_mode"] == "default"
