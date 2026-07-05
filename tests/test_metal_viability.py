from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

from largerlm.metal_viability import (
    MetalViabilityError,
    build_metal_viability_report,
)


SCALE_E8M0_ONE = 127


def _run_viability_script(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> int:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "glm_metal_viability_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "glm_metal_viability_report_script",
        script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script)] + argv)
    return int(module.main())


def _pack8(code: int) -> bytes:
    packed = 0
    for index in range(8):
        packed |= (code & 0xF) << (index * 4)
    return struct.pack("<I", packed)


def _write_prepared(tmp_path: Path) -> Path:
    prepared = tmp_path / "prepared"
    expert_dir = prepared / "experts"
    expert_dir.mkdir(parents=True)
    (expert_dir / "layout.json").write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 3,
                        "num_experts": 4,
                        "expert_slot_bytes": 10,
                    },
                    {
                        "layer": 4,
                        "num_experts": 4,
                        "expert_slot_bytes": 20,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    resident_dir = prepared / "resident"
    resident_dir.mkdir(parents=True)
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

    for layer in (3, 4):
        prefix = f"model.layers.{layer}.self_attn"
        add(f"{prefix}.o_proj.weight", "U32", [2, 1], _pack8(2) * 2)
        add(
            f"{prefix}.o_proj.scales",
            "U8",
            [2, 1],
            bytes([SCALE_E8M0_ONE]) * 2,
        )
        add(f"{prefix}.embed_q.weight", "U32", [1, 8, 1], _pack8(2) * 8)
        add(
            f"{prefix}.embed_q.scales",
            "U8",
            [1, 8, 1],
            bytes([SCALE_E8M0_ONE]) * 8,
        )
        add(f"{prefix}.unembed_out.weight", "U32", [1, 8, 1], _pack8(2) * 8)
        add(
            f"{prefix}.unembed_out.scales",
            "U8",
            [1, 8, 1],
            bytes([SCALE_E8M0_ONE]) * 8,
        )
    (resident_dir / "resident.bin").write_bytes(bytes(payload))
    (resident_dir / "layout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": len(payload),
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps({"version": 1, "resident_layout": "resident/layout.json"}),
        encoding="utf-8",
    )
    return prepared


def test_build_metal_viability_report_reads_layout_and_smoke(tmp_path: Path) -> None:
    prepared = _write_prepared(tmp_path)
    smoke = prepared / "smoke-decode-1tok-metal-logits-result.json"
    smoke.write_text(
        json.dumps(
            {
                "token_result": {
                    "elapsed_seconds": 6.0,
                    "generated_token_ids": [15],
                    "estimated_read_bytes": 100,
                    "estimated_logits_read_bytes": 30,
                    "estimated_expert_read_bytes": 70,
                    "decode_actual_read_time": {
                        "prefill_ssd_read_gib_per_second": 10.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_metal_viability_report(
        prepared,
        top_k=2,
        flash_moe_reference_layers=1,
        flash_moe_reference_top_k=2,
        flash_moe_reference_expert_slot_bytes=10,
    )

    assert report.expert_layer_count == 2
    assert report.routed_expert_read_bytes_per_token == 60
    assert report.qwen_flash_moe_reference_read_bytes_per_token == 20
    assert report.glm_to_qwen_flash_moe_read_ratio == 3.0
    assert report.ssd_read_gib_per_second == 10.0
    assert report.routed_read_lower_bound_seconds_per_token == pytest.approx(
        60 / 1024**3 / 10.0
    )
    assert report.observed_decode_seconds_per_token == 6.0
    assert report.observed_tokens_per_second == pytest.approx(1 / 6)
    assert report.observed_generated_token_ids == (15,)
    assert report.observed_estimated_read_bytes == 100
    assert report.observed_estimated_logits_read_bytes == 30
    assert report.observed_estimated_expert_read_bytes == 70
    assert report.decision.startswith("runnable_but_not_usable_speed")


def test_build_metal_viability_report_can_override_ssd_speed(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path)

    report = build_metal_viability_report(
        prepared,
        top_k=2,
        ssd_read_gib_per_second=5.0,
    )

    assert report.ssd_read_gib_per_second == 5.0
    assert report.observed_decode_seconds_per_token is None
    assert report.decision.startswith("estimate_only")


def test_build_metal_viability_report_projects_context1_collapse(
    tmp_path: Path,
) -> None:
    prepared = _write_prepared(tmp_path)
    plan = prepared / "context1-plan.json"
    plan.write_text(
        json.dumps(
            {
                "layers_supported": 2,
                "bytes": {
                    "chosen_cache_total": 12,
                    "current_o_proj_storage_per_token": 120,
                },
                "build_work": {
                    "fma_per_layer": 128,
                    "fma_total": 256,
                },
            }
        ),
        encoding="utf-8",
    )
    telemetry = prepared / "decode.json"
    telemetry.write_text(
        json.dumps(
            {
                "generated_token_ids": [1, 2, 3],
                "decode_elapsed_seconds": [10.0, 10.0, 10.0],
                "final_logits_elapsed_seconds": [1.0, 1.0, 1.0],
                "decode_attn_output_bytes_read": [120, 120, 120],
                "decode_attn_output_read_seconds": [2.0, 2.0, 2.0],
                "decode_attn_output_projection_kernel_seconds": [3.0, 3.0, 3.0],
            }
        ),
        encoding="utf-8",
    )

    report = build_metal_viability_report(
        prepared,
        top_k=2,
        context1_collapse_plan_path=plan,
        decode_telemetry_path=telemetry,
        target_tokens_per_second=5.0,
    )

    assert report.context1_collapse_plan_path == plan
    assert report.context1_layers_supported == 2
    assert report.context1_cache_bytes == 12
    assert report.context1_current_o_proj_bytes_per_token == 120
    assert report.context1_cache_read_ratio == pytest.approx(0.1)
    assert report.context1_build_fma_per_layer == 128
    assert report.context1_build_fma_total == 256
    assert report.observed_decode_token_count == 3
    assert report.observed_attn_output_bytes_per_token == pytest.approx(120.0)
    assert report.observed_attn_output_read_seconds_per_token == pytest.approx(2.0)
    assert report.observed_attn_output_projection_seconds_per_token == pytest.approx(
        3.0
    )
    assert report.projected_context1_only_total_with_logits_seconds == pytest.approx(
        19.5
    )
    assert report.projected_context1_only_tokens_per_second == pytest.approx(3 / 19.5)
    assert report.projected_context1_only_speedup == pytest.approx(
        (3 / 19.5) / (3 / 33)
    )
    assert report.exact_all_context_per_head_cache_bytes == 64
    assert report.exact_all_context_per_head_cache_read_ratio == pytest.approx(
        64 / 20
    )
    assert report.exact_all_context_per_head_int4_floor_bytes == 16
    assert report.exact_all_context_per_head_int4_floor_read_ratio == pytest.approx(
        16 / 20
    )
    assert report.exact_all_context_per_head_runtime_fma_ratio == pytest.approx(1.0)
    assert report.exact_all_context_break_even_shared_head_groups == 0
    assert report.exact_all_context_independent_attention_head_groups == 1
    assert (
        report.projected_exact_all_context_per_head_total_with_logits_seconds
        == pytest.approx(46.2)
    )
    assert report.projected_exact_all_context_per_head_tokens_per_second == (
        pytest.approx(3 / 46.2)
    )
    assert report.projected_exact_all_context_per_head_speedup == pytest.approx(
        (3 / 46.2) / (3 / 33)
    )
    assert "exact_all_context_per_head_cache_not_viable" in (
        report.efficiency_rewrite_decision or ""
    )
    assert report.target_tokens_per_second == 5.0
    assert report.target_reference_tokens_per_second == pytest.approx(3 / 19.5)
    assert report.target_reference_source == "projected_context1_only_with_logits"
    assert report.target_met is False
    assert report.target_gap_multiplier == pytest.approx(5.0 / (3 / 19.5))
    assert (report.target_decision or "").startswith("stop_at_minimal_usable")


def test_viability_script_can_require_below_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_prepared(tmp_path)
    plan = prepared / "context1-plan.json"
    plan.write_text(
        json.dumps(
            {
                "layers_supported": 2,
                "bytes": {
                    "chosen_cache_total": 12,
                    "current_o_proj_storage_per_token": 120,
                },
                "build_work": {
                    "fma_per_layer": 128,
                    "fma_total": 256,
                },
            }
        ),
        encoding="utf-8",
    )
    telemetry = prepared / "decode.json"
    telemetry.write_text(
        json.dumps(
            {
                "generated_token_ids": [1, 2, 3],
                "decode_elapsed_seconds": [10.0, 10.0, 10.0],
                "final_logits_elapsed_seconds": [1.0, 1.0, 1.0],
                "decode_attn_output_bytes_read": [120, 120, 120],
                "decode_attn_output_read_seconds": [2.0, 2.0, 2.0],
                "decode_attn_output_projection_kernel_seconds": [3.0, 3.0, 3.0],
            }
        ),
        encoding="utf-8",
    )

    rc = _run_viability_script(
        monkeypatch,
        [
            str(prepared),
            "--decode-telemetry",
            str(telemetry),
            "--context1-collapse-plan",
            str(plan),
            "--target-tok-s",
            "5",
            "--require-below-target",
            "--json",
        ],
    )

    assert rc == 0
    assert '"target_met": false' in capsys.readouterr().out

    rc = _run_viability_script(
        monkeypatch,
        [
            str(prepared),
            "--decode-telemetry",
            str(telemetry),
            "--context1-collapse-plan",
            str(plan),
            "--target-tok-s",
            "0.1",
            "--require-below-target",
        ],
    )

    assert rc == 1
    assert "throughput target is not below threshold" in capsys.readouterr().err


def test_build_metal_viability_report_rejects_bad_layout(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    (prepared / "experts").mkdir(parents=True)
    (prepared / "experts" / "layout.json").write_text(
        json.dumps({"layers": [{"layer": 3, "expert_slot_bytes": 0}]}),
        encoding="utf-8",
    )

    with pytest.raises(MetalViabilityError, match="expert_slot_bytes"):
        build_metal_viability_report(prepared)
