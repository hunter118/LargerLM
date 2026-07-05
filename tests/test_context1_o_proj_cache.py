from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from largerlm import cli
from largerlm.context1_o_proj_cache import (
    CACHE_SCHEMA,
    Context1OProjCacheError,
    build_context1_o_proj_cache,
    load_context1_o_proj_cache_layout,
    load_context1_o_proj_cache_progress,
    plan_context1_o_proj_cache,
)


SCALE_E8M0_ONE = 127


def _pack8(code: int) -> bytes:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def _bf16_to_f32(raw: bytes) -> float:
    bits = int.from_bytes(raw, "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def _write_prepared(tmp_path: Path, *, layers: int = 1) -> Path:
    prepared = tmp_path / "prepared"
    resident = prepared / "resident"
    resident.mkdir(parents=True)
    payload = bytearray()
    tensors: list[dict[str, object]] = []

    def add(name: str, dtype: str, shape: list[int], data: bytes) -> None:
        tensors.append(
            {
                "name": name,
                "offset": len(payload),
                "size": len(data),
                "dtype": dtype,
                "shape": shape,
                "category": "attention",
            }
        )
        payload.extend(data)

    for layer in range(layers):
        prefix = f"model.layers.{layer}.self_attn"
        add(
            f"{prefix}.o_proj.weight",
            "U32",
            [2, 1],
            _pack8(2) + _pack8(1),
        )
        add(
            f"{prefix}.o_proj.scales",
            "U8",
            [2, 1],
            bytes([SCALE_E8M0_ONE, SCALE_E8M0_ONE]),
        )
        add(
            f"{prefix}.embed_q.weight",
            "U32",
            [1, 8, 1],
            _pack8(2) * 8,
        )
        add(
            f"{prefix}.embed_q.scales",
            "U8",
            [1, 8, 1],
            bytes([SCALE_E8M0_ONE]) * 8,
        )
        add(
            f"{prefix}.unembed_out.weight",
            "U32",
            [1, 8, 1],
            _pack8(2) * 8,
        )
        add(
            f"{prefix}.unembed_out.scales",
            "U8",
            [1, 8, 1],
            bytes([SCALE_E8M0_ONE]) * 8,
        )

    (resident / "resident.bin").write_bytes(bytes(payload))
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
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "resident_layout": "resident/layout.json",
            }
        ),
        encoding="utf-8",
    )
    return prepared


def _write_fake_metal_builder(tmp_path: Path) -> Path:
    runner = tmp_path / "fake_metal_builder.py"
    runner.write_text(
        """#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path


def f32_to_bf16_bytes(value):
    bits = int.from_bytes(struct.pack("<f", float(value)), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


args = sys.argv[1:]
layout = Path(args[args.index("--context1-o-proj-cache-layout") + 1])
layer = int(args[args.index("--probe-layer") + 1])
max_live = (
    float(args[args.index("--max-live-working-set-mib") + 1])
    if "--max-live-working-set-mib" in args
    else 0.0
)
payload = json.loads(layout.read_text(encoding="utf-8"))
tensor = next(item for item in payload["tensors"] if item["layer"] == layer)
cache = layout.parent / payload["weight_file"]
values = [8.0] * 8 + [4.0] * 8
raw = b"".join(f32_to_bf16_bytes(value) for value in values)
with cache.open("r+b") as out:
    out.seek(tensor["offset"])
    out.write(raw)
print(json.dumps({
    "ok": True,
    "layer": layer,
    "device_name": "fake-metal",
    "source_bytes_read": 50,
    "cache_bytes_written": len(raw),
    "fma_count": 128,
    "estimated_live_working_set_bytes": 82,
    "max_live_working_set_mib": max_live,
    "live_working_set_ok": 1,
    "read_seconds": 0.001,
    "kernel_seconds": 0.002,
    "write_seconds": 0.003,
    "elapsed_seconds": 0.006,
    "output0": 8.0,
}))
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    return runner


def test_plan_context1_o_proj_cache_estimates_tiny_fixture(tmp_path: Path) -> None:
    prepared = _write_prepared(tmp_path, layers=12)

    plan = plan_context1_o_proj_cache(prepared, dtype="BF16")

    assert [layer.layer for layer in plan.layers] == list(range(12))
    assert plan.hidden_dim == 2
    assert plan.attention_value_dim == 8
    assert plan.num_heads == 1
    assert plan.v_head_dim == 8
    assert plan.kv_lora_dim == 8
    assert plan.total_bytes == 12 * 2 * 8 * 2
    assert plan.fma_total == 12 * 2 * 8 * 8
    assert plan.current_o_proj_storage_per_token == 12 * (8 + 2)
    report = plan.to_report()
    assert report["build_work"]["max_layer_fma"] == 128
    assert report["build_work"]["per_layer_fma"] == [128] * 12


def test_build_context1_o_proj_cache_reference_writes_layout_and_bf16(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"

    result = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        dtype="BF16",
        execute=True,
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
        row_tile=1,
    )

    assert result.executed is True
    assert result.completed_layers == (0,)
    assert len(result.layer_results) == 1
    reference_result = result.layer_results[0]
    assert reference_result["backend"] == "reference"
    assert reference_result["layer"] == 0
    assert reference_result["source_bytes_read"] == 50
    assert reference_result["cache_bytes_written"] == 32
    assert reference_result["fma_count"] == 128
    assert reference_result["estimated_metal_builder_live_bytes"] == 82
    assert reference_result["elapsed_seconds"] >= 0.0
    reference_summary = result.layer_result_summary()
    assert reference_summary["measured_layer_count"] == 1
    assert reference_summary["measured_layers"] == [0]
    assert reference_summary["measured_fma_total"] == 128
    assert reference_summary["completed_fma_total"] == 128
    assert reference_summary["remaining_fma_total"] == 0
    assert reference_summary["completed_fma_fraction"] == 1.0
    layout = json.loads((output_dir / "layout.json").read_text(encoding="utf-8"))
    assert layout["schema"] == CACHE_SCHEMA
    assert layout["dtype"] == "BF16"
    assert layout["total_bytes"] == 32
    assert layout["tensors"] == [
        {
            "category": "context1_attention_output",
            "dtype": "BF16",
            "layer": 0,
            "name": "model.layers.0.self_attn.context1_o_proj_bv.weight",
            "offset": 0,
            "shape": [2, 8],
            "size": 32,
        }
    ]
    raw = (output_dir / "context1_o_proj_bv.bin").read_bytes()
    values = [_bf16_to_f32(raw[index : index + 2]) for index in range(0, len(raw), 2)]
    assert values[:8] == [8.0] * 8
    assert values[8:] == [4.0] * 8


def test_build_context1_o_proj_cache_can_execute_full_layout_in_layer_chunks(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=2)
    output_dir = tmp_path / "cache"

    first = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        dtype="BF16",
        execute=True,
        max_build_fma=128,
        disk_safety_margin_bytes=0,
        row_tile=1,
        build_layers=(1,),
    )

    assert first.plan.layers[0].layer == 0
    assert first.plan.layers[1].layer == 1
    assert first.requested_build_layers == (1,)
    assert first.completed_layers == (1,)
    selected = first.to_json()["selected_build"]
    assert selected["full_plan"] is False
    assert selected["layers"] == [1]
    assert selected["cache_bytes"] == 32
    assert selected["source_bytes"] == 50
    assert selected["max_estimated_metal_builder_live_bytes"] == 82
    assert selected["per_layer"][0]["estimated_metal_builder_live_bytes"] == 82
    assert selected["fma_total"] == 128
    assert selected["max_layer_fma"] == 128
    assert selected["min_max_build_fma"] == 128
    first_summary = first.layer_result_summary()
    assert first_summary["measured_layer_count"] == 1
    assert first_summary["completed_fma_total"] == 128
    assert first_summary["remaining_fma_total"] == 128
    assert first_summary["completed_fma_fraction"] == 0.5
    layout = json.loads((output_dir / "layout.json").read_text(encoding="utf-8"))
    assert [tensor["layer"] for tensor in layout["tensors"]] == [0, 1]
    assert layout["total_bytes"] == 64
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_layers"] == [1]
    raw = (output_dir / "context1_o_proj_bv.bin").read_bytes()
    layer0 = [_bf16_to_f32(raw[index : index + 2]) for index in range(0, 32, 2)]
    layer1 = [_bf16_to_f32(raw[index : index + 2]) for index in range(32, 64, 2)]
    assert layer0 == [0.0] * 16
    assert layer1[:8] == [8.0] * 8
    assert layer1[8:] == [4.0] * 8

    preview = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        dtype="BF16",
        disk_safety_margin_bytes=0,
        row_tile=1,
        build_next_layers=1,
    )

    assert preview.executed is False
    assert preview.requested_build_next_layers == 1
    assert preview.requested_build_layers == (0,)
    assert preview.completed_layers == (1,)
    assert [item["layer"] for item in preview.layer_results] == [1]
    assert preview.to_json()["requested_build_next_layers"] == 1
    assert preview.to_json()["selected_build"]["layers"] == [0]

    second = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        dtype="BF16",
        execute=True,
        max_build_fma=128,
        disk_safety_margin_bytes=0,
        row_tile=1,
        build_next_layers=1,
    )

    assert second.requested_build_next_layers == 1
    assert second.requested_build_layers == (0,)
    assert second.completed_layers == (0, 1)
    assert [item["layer"] for item in second.layer_results] == [1, 0]
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_layers"] == [0, 1]
    raw = (output_dir / "context1_o_proj_bv.bin").read_bytes()
    values = [_bf16_to_f32(raw[index : index + 2]) for index in range(0, len(raw), 2)]
    assert values[:8] == [8.0] * 8
    assert values[8:16] == [4.0] * 8
    assert values[16:24] == [8.0] * 8
    assert values[24:] == [4.0] * 8


def test_build_context1_o_proj_cache_metal_backend_invokes_runner(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"
    runner = _write_fake_metal_builder(tmp_path)

    result = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        dtype="BF16",
        execute=True,
        backend="metal",
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
        metal_binary=runner,
    )

    assert result.executed is True
    assert result.backend == "metal"
    assert result.completed_layers == (0,)
    assert len(result.layer_results) == 1
    layer_result = result.layer_results[0]
    assert layer_result["backend"] == "metal"
    assert layer_result["layer"] == 0
    assert layer_result["device_name"] == "fake-metal"
    assert layer_result["source_bytes_read"] == 50
    assert layer_result["cache_bytes_written"] == 32
    assert layer_result["fma_count"] == 128
    assert layer_result["estimated_metal_builder_live_bytes"] == 82
    assert layer_result["estimated_live_working_set_bytes"] == 82
    assert layer_result["max_live_working_set_mib"] == 512.0
    assert layer_result["live_working_set_ok"] is True
    assert layer_result["kernel_seconds"] == 0.002
    assert result.to_json()["layer_results"][0]["backend"] == "metal"
    summary = result.layer_result_summary()
    assert summary["measured_layer_count"] == 1
    assert summary["measured_layers"] == [0]
    assert summary["measured_fma_total"] == 128
    assert summary["measured_source_bytes_read"] == 50
    assert summary["measured_cache_bytes_written"] == 32
    assert summary["measured_elapsed_seconds"] == pytest.approx(0.006)
    assert summary["measured_read_seconds"] == pytest.approx(0.001)
    assert summary["measured_kernel_seconds"] == pytest.approx(0.002)
    assert summary["measured_write_seconds"] == pytest.approx(0.003)
    assert summary["measured_gfma_per_second"] == pytest.approx(128 / 0.006 / 1.0e9)
    assert summary["estimated_full_build_seconds_from_measured"] == pytest.approx(0.006)
    assert summary["estimated_remaining_build_seconds_from_measured"] == pytest.approx(0.0)
    assert summary["max_estimated_live_working_set_bytes"] == 82
    assert result.to_json()["layer_result_summary"]["measured_layer_count"] == 1
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["backend"] == "metal"
    assert progress["layer_results"][0]["backend"] == "metal"
    assert progress["layer_results"][0]["layer"] == 0
    raw = (output_dir / "context1_o_proj_bv.bin").read_bytes()
    values = [_bf16_to_f32(raw[index : index + 2]) for index in range(0, len(raw), 2)]
    assert values[:8] == [8.0] * 8
    assert values[8:] == [4.0] * 8


def test_build_context1_o_proj_cache_metal_backend_honors_build_layers(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=2)
    output_dir = tmp_path / "cache"
    runner = _write_fake_metal_builder(tmp_path)

    result = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        dtype="BF16",
        execute=True,
        backend="metal",
        max_build_fma=128,
        disk_safety_margin_bytes=0,
        build_layers=(1,),
        metal_binary=runner,
    )

    assert result.executed is True
    assert result.backend == "metal"
    assert result.requested_build_layers == (1,)
    assert result.completed_layers == (1,)
    assert [item["layer"] for item in result.layer_results] == [1]
    layout = json.loads((output_dir / "layout.json").read_text(encoding="utf-8"))
    assert [tensor["layer"] for tensor in layout["tensors"]] == [0, 1]
    raw = (output_dir / "context1_o_proj_bv.bin").read_bytes()
    layer0 = [_bf16_to_f32(raw[index : index + 2]) for index in range(0, 32, 2)]
    layer1 = [_bf16_to_f32(raw[index : index + 2]) for index in range(32, 64, 2)]
    assert layer0 == [0.0] * 16
    assert layer1[:8] == [8.0] * 8
    assert layer1[8:] == [4.0] * 8


def test_build_context1_o_proj_cache_refuses_reference_over_fma_cap(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)

    with pytest.raises(Context1OProjCacheError, match="exceeds max_build_fma"):
        build_context1_o_proj_cache(
            prepared,
            execute=True,
            max_build_fma=127,
            disk_safety_margin_bytes=0,
        )


def test_build_context1_o_proj_cache_refuses_progress_without_cache_file(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
    )
    (output_dir / "context1_o_proj_bv.bin").unlink()

    with pytest.raises(Context1OProjCacheError, match="progress exists"):
        build_context1_o_proj_cache(
            prepared,
            output_dir=output_dir,
            execute=True,
            max_build_fma=1024,
            disk_safety_margin_bytes=0,
        )


def test_build_context1_o_proj_cache_dry_run_writes_nothing(tmp_path: Path) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"

    result = build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        disk_safety_margin_bytes=0,
    )

    assert result.executed is False
    assert result.plan.total_bytes == 32
    assert result.disk_budget is not None
    assert result.disk_budget.required_bytes == 32
    assert result.disk_budget.safety_margin_bytes == 0
    assert result.to_json()["disk_budget"]["ok"] is True
    assert not output_dir.exists()


def test_build_context1_o_proj_cache_refuses_disk_budget_shortfall(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)

    with pytest.raises(Context1OProjCacheError, match="not enough free disk"):
        build_context1_o_proj_cache(
            prepared,
            execute=True,
            max_build_fma=1024,
            disk_safety_margin_bytes=10**30,
        )

    assert not (prepared / "context1-o-proj-bv-cache").exists()


def test_build_context1_o_proj_cache_refuses_metal_builder_live_cap(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    runner = _write_fake_metal_builder(tmp_path)

    with pytest.raises(Context1OProjCacheError, match="Metal builder estimated live"):
        build_context1_o_proj_cache(
            prepared,
            execute=True,
            backend="metal",
            max_build_fma=1024,
            disk_safety_margin_bytes=0,
            max_metal_builder_live_bytes=81,
            metal_binary=runner,
        )

    assert not (prepared / "context1-o-proj-bv-cache").exists()


def test_build_context1_o_proj_cache_rejects_build_layers_with_next(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=2)

    with pytest.raises(Context1OProjCacheError, match="mutually exclusive"):
        build_context1_o_proj_cache(
            prepared,
            build_layers=(0,),
            build_next_layers=1,
        )


def test_context1_o_proj_cache_cli_dry_run_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    report = tmp_path / "report.json"

    rc = cli.main(
        [
            "context1-o-proj-cache",
            str(prepared),
            "--disk-margin-gib",
            "0",
            "--write-report",
            str(report),
        ]
    )

    assert rc == 0
    assert "dry-run ok" in capsys.readouterr().out
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["executed"] is False
    assert payload["bytes"]["total_bytes"] == 32
    assert payload["disk_budget"]["required_bytes"] == 32
    assert payload["disk_budget"]["safety_margin_bytes"] == 0
    assert payload["disk_budget"]["ok"] is True
    assert payload["max_metal_builder_live_bytes"] == 512 * 1024**2
    assert payload["selected_build"]["max_estimated_metal_builder_live_bytes"] == 82


def test_context1_o_proj_cache_cli_accepts_build_layers_spec(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=3)
    report = tmp_path / "report.json"

    rc = cli.main(
        [
            "context1-o-proj-cache",
            str(prepared),
            "--build-layers",
            "1-2",
            "--disk-margin-gib",
            "0",
            "--write-report",
            str(report),
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "build layers:          2" in stdout
    assert "selected build FMA:    256" in stdout
    assert "min --max-build-fma:   256" in stdout
    assert "min --max-build-gfma:  2.56e-07" in stdout
    assert "disk budget ok:        True" in stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["executed"] is False
    assert payload["layers"] == [0, 1, 2]
    assert payload["requested_build_layers"] == [1, 2]
    assert payload["build_work"]["fma_total"] == 384
    assert payload["build_work"]["max_layer_fma"] == 128
    assert payload["selected_build"]["full_plan"] is False
    assert payload["selected_build"]["layers"] == [1, 2]
    assert payload["selected_build"]["cache_bytes"] == 64
    assert payload["selected_build"]["source_bytes"] == 100
    assert payload["selected_build"]["max_estimated_metal_builder_live_bytes"] == 82
    assert payload["selected_build"]["fma_total"] == 256
    assert payload["selected_build"]["max_layer_fma"] == 128
    assert payload["selected_build"]["min_max_build_fma"] == 256


def test_context1_o_proj_cache_cli_accepts_build_next_layers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=3)
    report = tmp_path / "report.json"

    rc = cli.main(
        [
            "context1-o-proj-cache",
            str(prepared),
            "--build-next-layers",
            "2",
            "--disk-margin-gib",
            "0",
            "--write-report",
            str(report),
        ]
    )

    assert rc == 0
    stdout = capsys.readouterr().out
    assert "build next layers:     2" in stdout
    assert "build layers:          2" in stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["executed"] is False
    assert payload["requested_build_next_layers"] == 2
    assert payload["requested_build_layers"] == [0, 1]
    assert payload["selected_build"]["layers"] == [0, 1]
    assert payload["selected_build"]["fma_total"] == 256


def test_context1_o_proj_cache_cli_accepts_max_build_gfma(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"

    rc = cli.main(
        [
            "context1-o-proj-cache",
            str(prepared),
            "--output-dir",
            str(output_dir),
            "--execute",
            "--max-build-gfma",
            "0.000000129",
            "--disk-margin-gib",
            "0",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is True
    assert payload["completed_layers"] == [0]


def test_context1_o_proj_cache_cli_rejects_conflicting_build_caps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "context1-o-proj-cache",
                str(prepared),
                "--max-build-fma",
                "128",
                "--max-build-gfma",
                "0.000000129",
            ]
        )

    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_load_context1_o_proj_cache_layout_validates_backing_and_prepared(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
    )

    layout = load_context1_o_proj_cache_layout(
        output_dir / "layout.json",
        prepared_dir=prepared,
    )

    assert layout.layers == (0,)
    assert layout.total_bytes == 32
    assert layout.cache_file_path == output_dir / "context1_o_proj_bv.bin"


def test_load_context1_o_proj_cache_progress_rejects_incomplete_by_default(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=2)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=128,
        disk_safety_margin_bytes=0,
        build_layers=(1,),
    )
    layout = load_context1_o_proj_cache_layout(output_dir / "layout.json")

    with pytest.raises(Context1OProjCacheError, match="progress is incomplete"):
        load_context1_o_proj_cache_progress(layout)

    progress = load_context1_o_proj_cache_progress(
        layout,
        require_complete=False,
    )
    assert progress.exists is True
    assert progress.complete is False
    assert progress.completed_layers == (1,)
    assert progress.missing_layers == (0,)
    assert progress.total_layers == 2
    assert progress.missing_layer_count == 1
    assert progress.next_missing_layer == 0
    assert [item["layer"] for item in progress.layer_results] == [1]
    assert progress.layer_results[0]["backend"] == "reference"
    assert progress.layer_result_summary is not None
    assert progress.layer_result_summary["measured_layer_count"] == 1
    assert progress.layer_result_summary["completed_fma_total"] == 128
    assert progress.layer_result_summary["remaining_fma_total"] == 128
    assert progress.layer_result_summary["completed_fma_fraction"] == 0.5
    suggestion = progress.suggested_resume_build
    assert suggestion is not None
    assert suggestion["available"] is True
    assert suggestion["backend"] == "reference"
    assert suggestion["next_layer"] == 0
    assert suggestion["next_layers"] == [0]
    assert suggestion["recommended_next_layers"] == 1
    assert suggestion["remaining_layer_count"] == 1
    assert suggestion["min_max_build_fma"] == 128
    assert suggestion["min_max_build_gfma_arg"] == "1.28e-07"
    assert suggestion["prepared_dir"] == str(prepared)
    assert suggestion["output_dir"] == str(output_dir)
    assert "--execute" not in suggestion["dry_run_argv"]
    assert suggestion["execute_argv"][-1] == "--execute"


def test_load_context1_o_proj_cache_layout_rejects_missing_backing(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
    )
    (output_dir / "context1_o_proj_bv.bin").unlink()

    with pytest.raises(Context1OProjCacheError, match="failed to stat cache file"):
        load_context1_o_proj_cache_layout(output_dir / "layout.json")

    layout = load_context1_o_proj_cache_layout(
        output_dir / "layout.json",
        require_cache_file=False,
    )
    assert layout.total_bytes == 32


def test_load_context1_o_proj_cache_layout_rejects_unsorted_layers(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path, layers=2)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
    )
    layout_path = output_dir / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["tensors"] = list(reversed(payload["tensors"]))
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Context1OProjCacheError, match="sorted by numeric layer"):
        load_context1_o_proj_cache_layout(layout_path)


def test_validate_context1_o_proj_cache_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=1)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=1024,
        disk_safety_margin_bytes=0,
    )

    rc = cli.main(
        [
            "validate-context1-o-proj-cache",
            str(output_dir / "layout.json"),
            "--prepared-dir",
            str(prepared),
        ]
    )

    assert rc == 0
    assert "result:                ok" in capsys.readouterr().out


def test_validate_context1_o_proj_cache_cli_rejects_incomplete_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path, layers=2)
    output_dir = tmp_path / "cache"
    build_context1_o_proj_cache(
        prepared,
        output_dir=output_dir,
        execute=True,
        max_build_fma=128,
        disk_safety_margin_bytes=0,
        build_layers=(1,),
    )

    rc = cli.main(
        [
            "validate-context1-o-proj-cache",
            str(output_dir / "layout.json"),
            "--prepared-dir",
            str(prepared),
        ]
    )

    assert rc == 1
    assert "progress is incomplete" in capsys.readouterr().err

    rc = cli.main(
        [
            "validate-context1-o-proj-cache",
            str(output_dir / "layout.json"),
            "--prepared-dir",
            str(prepared),
            "--allow-incomplete-progress",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["progress"]["exists"] is True
    assert payload["progress"]["complete"] is False
    assert payload["progress"]["completed_layers"] == [1]
    assert payload["progress"]["missing_layers"] == [0]
    assert payload["progress"]["missing_layer_count"] == 1
    assert payload["progress"]["next_missing_layer"] == 0
    assert [item["layer"] for item in payload["progress"]["layer_results"]] == [1]
    assert payload["progress"]["layer_result_summary"]["remaining_fma_total"] == 128
    suggestion = payload["progress"]["suggested_resume_build"]
    assert suggestion["available"] is True
    assert suggestion["backend"] == "reference"
    assert suggestion["next_layer"] == 0
    assert suggestion["min_max_build_fma"] == 128
    assert suggestion["dry_run_argv"][:5] == [
        "context1-o-proj-cache",
        str(prepared),
        "--output-dir",
        str(output_dir),
        "--backend",
    ]
    assert "--execute" not in suggestion["dry_run_argv"]
    assert suggestion["execute_argv"][-1] == "--execute"
