from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "flash_moe_efficiency_envelope.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "flash_moe_efficiency_envelope_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
envelope = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(envelope)


def _write_tiny_prepared(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    model = tmp_path / "model"
    experts = prepared / "experts"
    prepared.mkdir()
    model.mkdir()
    experts.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm-test",
                "hidden_size": 16,
                "num_hidden_layers": 3,
                "num_experts_per_tok": 3,
                "n_routed_experts": 8,
                "mlp_layer_types": ["dense", "sparse", "sparse"],
            }
        ),
        encoding="utf-8",
    )
    (experts / "layout.json").write_text(
        json.dumps(
            {
                "group_size": 32,
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 8,
                        "expert_slot_bytes": 100,
                        "layer_file": "layer_001.bin",
                    },
                    {
                        "layer": 2,
                        "num_experts": 8,
                        "expert_slot_bytes": 200,
                        "layer_file": "layer_002.bin",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "model_dir": str(model),
                "experts_layout": "experts/layout.json",
                "prepare_cold_read_gib_per_second": 10.0,
                "expert_quantization": "mlx-mxfp4",
                "expert_group_size": 32,
            }
        ),
        encoding="utf-8",
    )
    return prepared


def test_build_efficiency_envelope_uses_layout_config_and_manifest_io(
    tmp_path: Path,
) -> None:
    prepared = _write_tiny_prepared(tmp_path)

    payload = envelope.build_efficiency_envelope(prepared)

    assert payload["schema"] == "largerlm.flash_moe_efficiency_envelope.v1"
    assert payload["model"]["top_k"] == 3
    assert payload["model"]["moe_layer_count_from_config"] == 2
    assert payload["expert_layout"]["moe_layer_count"] == 2
    assert payload["expert_layout"]["total_one_expert_per_layer_bytes"] == 300
    assert payload["routed_expert_io"]["bytes_per_token"] == 900
    assert payload["routed_expert_io"]["routed_expert_reads_per_token"] == 6
    assert payload["routed_expert_io"]["io_gib_per_second"] == 10.0
    assert payload["routed_expert_io"]["io_gib_per_second_source"] == (
        "manifest.prepare_cold_read_gib_per_second"
    )
    expected_seconds = (900 / envelope.GIB) / 10.0
    assert payload["routed_expert_io"]["io_floor_seconds_per_token"] == expected_seconds


def test_build_efficiency_envelope_can_override_top_k_and_io(
    tmp_path: Path,
) -> None:
    prepared = _write_tiny_prepared(tmp_path)

    payload = envelope.build_efficiency_envelope(
        prepared,
        top_k=2,
        io_gib_per_second=5.0,
    )

    assert payload["model"]["top_k"] == 2
    assert payload["routed_expert_io"]["bytes_per_token"] == 600
    assert payload["routed_expert_io"]["io_gib_per_second"] == 5.0
    assert payload["routed_expert_io"]["io_gib_per_second_source"] == "argument"
    assert payload["flash_moe_reference"]["bytes_per_token"] == (
        60 * 4 * 7_077_888
    )
    assert (
        payload["comparison"]["glm_to_flash_moe_bytes_per_token_ratio"]
        == 600 / (60 * 4 * 7_077_888)
    )


def test_main_prints_json(tmp_path: Path, capsys) -> None:
    prepared = _write_tiny_prepared(tmp_path)

    status = envelope.main([str(prepared), "--json"])

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prepared_dir"] == str(prepared)
    assert output["routed_expert_io"]["bytes_per_token"] == 900
