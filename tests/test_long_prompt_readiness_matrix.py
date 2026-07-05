from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "long_prompt_readiness_matrix.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "long_prompt_readiness_matrix_script",
    _SCRIPT_PATH,
)
assert _SPEC is not None
matrix = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(matrix)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "prepared": "prepared",
        "prompt_tokens": [128],
        "max_new_tokens": 1,
        "apply_launch_profile": None,
        "lock_launch_profile": False,
        "require_locked_launch_profile": False,
        "allow_non_accelerated_prefill_launch_audit": False,
        "run_mpsgraph_probe": False,
        "run_mpp_probe": False,
        "probe_timeout_seconds": None,
        "inspect_arg": [],
        "write_raw_dir": None,
        "write_result": None,
        "cwd": None,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_inspect_command_includes_safe_checks_and_extra_args() -> None:
    args = _args(
        prepared="artifact/prepared",
        max_new_tokens=4,
        apply_launch_profile="profile.json",
        lock_launch_profile=True,
        require_locked_launch_profile=True,
        allow_non_accelerated_prefill_launch_audit=True,
        run_mpsgraph_probe=True,
        run_mpp_probe=True,
        probe_timeout_seconds=30,
        inspect_arg=[
            "--prefill-prompt-chunk-tokens",
            "17",
            "--prefill-expert-stage-tiling",
        ],
    )

    command = matrix.build_inspect_command(args, 2048)

    assert command[:4] == [sys.executable, "-m", "largerlm", "inspect-prepared"]
    assert "artifact/prepared" in command
    assert command[command.index("--check-prompt-tokens") + 1] == "2048"
    assert command[command.index("--check-max-new-tokens") + 1] == "4"
    assert "--check-runtime-preflight" in command
    assert "--require-prepared-memory-profile" in command
    assert "--require-glm-4bit" in command
    assert "--require-public-glm-5-2-shape" in command
    assert "--apply-launch-profile" in command
    assert "--lock-launch-profile" in command
    assert "--require-locked-launch-profile" in command
    assert "--allow-non-accelerated-prefill-launch-audit" in command
    assert "--run-mpsgraph-probe" in command
    assert "--run-mpp-probe" in command
    assert command[-3:] == [
        "--prefill-prompt-chunk-tokens",
        "17",
        "--prefill-expert-stage-tiling",
    ]


def test_summarize_payload_extracts_long_prompt_safety_fields() -> None:
    payload = {
        "prefill_backend": {
            "capability": {
                "selectable_accelerated_prefill_backends": ["mpsgraph-f32"],
                "validated_accelerated_prefill_backends": ["mpsgraph-f32"],
                "mps_graph_probe_ok": True,
                "mpp_tensor_ops_symbol_declared": False,
                "prefill_neural_accelerator_status": {
                    "status": "missing_public_mpp_symbols",
                    "reason": "public MPP symbols are missing",
                },
            }
        },
        "request_check": {
            "ok": True,
            "prefill_prompt_chunk_tokens": {
                "configured": 17,
                "resolved": 17,
                "max_safe": 113,
            },
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
                "required_available_memory_bytes": 44 * 1024**3,
                "system_available_memory_bytes": 80 * 1024**3,
                "live_working_set_bytes": 17 * 1024**3,
                "max_live_working_set_bytes": 18 * 1024**3,
                "system_memory_source": "vm_stat",
            },
            "prefill_acceleration_coverage": {
                "ok": True,
                "required": False,
                "accelerated_backends": ["mpsgraph-f32"],
                "accelerated_matrix_count": 75,
                "accelerated_flop_fraction": 0.1,
                "mpsgraph_matrix_count": 75,
                "unsupported_mpsgraph_matrix_count": 0,
                "mpp_tensor_ops_candidate_matrix_count": 10,
                "mpp_tensor_ops_candidate_flop_fraction": 0.9,
                "streamed_routed_expert_mpp_candidate_matrix_count": 225,
                "accelerated_router_gate_only": True,
                "reason": "ok",
            },
            "prefill_routed_expert_read": {
                "prompt_chunk_tokens": 17,
                "chunks_per_prompt": 121,
                "planned_read_bytes": 24 * 1024**4,
                "planned_read_seconds": 3880.25,
                "read_amplification": 64,
                "within_limit": True,
                "within_seconds_limit": True,
                "minimum_chunk_tokens_for_limits": 1,
            },
            "prefill_routed_stage_temp_disk": {
                "prompt_chunk_tokens": 17,
                "expert_stage_tiling": True,
                "max_stage_bytes": 3 * 1024**3,
                "max_compact_stage_bytes": 3 * 1024**3,
                "max_stage_plus_compact_bytes": 6 * 1024**3,
                "max_stage_raw_ranges": 136,
                "max_stage_coalesced_ranges": 136,
                "within_limit": True,
                "within_stage_raw_range_limit": True,
                "within_stage_coalesced_range_limit": True,
            },
            "prefill_prompt_chunk_plan": {
                "configured_is_auto": False,
                "max_safe": {
                    "chunk_tokens": 113,
                    "limiting_cap_names": ["cache_read"],
                    "mpp_tensor_ops_candidate_reachable_under_caps": False,
                    "mpp_tensor_ops_candidate_blocking_cap_summary": {
                        "cache_read": 78
                    },
                    "mpp_tensor_ops_dimension_candidate_matrix_count": 684,
                    "mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable": False,
                    "mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers": [
                        "cache_read_layer_0_mla_kv"
                    ],
                },
            },
        },
    }

    summary = matrix.summarize_payload(
        prompt_tokens=2048,
        returncode=0,
        payload=payload,
        stderr="",
    )

    assert summary["admitted"] is True
    assert summary["runtime_preflight"]["required_available_gib"] == 44
    assert summary["prefill_acceleration"]["mpsgraph_matrix_count"] == 75
    assert summary["prefill_acceleration"]["mpp_candidate_matrix_count"] == 10
    assert summary["backend_probe"]["neural_accelerator_status"] == (
        "missing_public_mpp_symbols"
    )
    assert summary["routed_expert_read"]["planned_read_gib"] == 24576
    assert summary["stage_temp"]["max_stage_plus_compact_gib"] == 6
    assert summary["prefill_prompt_chunk_plan"][
        "selected_mpp_candidate_blocking_cap_summary"
    ] == {"cache_read": 78}


def test_run_matrix_writes_payload_and_raw_inspects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    payloads = {
        128: {
            "request_check": {
                "ok": True,
                "prefill_prompt_chunk_tokens": {
                    "configured": 0,
                    "resolved": 128,
                    "max_safe": 128,
                },
                "runtime_preflight": {
                    "ran": True,
                    "available_memory_ok": True,
                },
            }
        },
        512: {
            "request_check": {
                "ok": False,
                "error": "prompt chunk exceeds cap",
                "prefill_prompt_chunk_tokens": {
                    "configured": 512,
                    "resolved": 512,
                    "max_safe": 128,
                },
                "runtime_preflight": {
                    "ran": True,
                    "available_memory_ok": True,
                },
            }
        },
    }

    def fake_run(command, *, check, capture_output, text, cwd):  # type: ignore[no-untyped-def]
        del check, capture_output, text, cwd
        prompt_tokens = int(command[command.index("--check-prompt-tokens") + 1])
        return subprocess.CompletedProcess(
            command,
            0 if prompt_tokens == 128 else 1,
            stdout=json.dumps(payloads[prompt_tokens]),
            stderr="",
        )

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    args = _args(
        prompt_tokens=[128, 512],
        write_raw_dir=str(raw_dir),
    )
    result = matrix.run_matrix(args)

    assert result["all_admitted"] is False
    assert result["max_admitted_prompt_tokens"] == 128
    assert result["first_rejected_prompt_tokens"] == 512
    assert len(result["entries"]) == 2
    assert (raw_dir / "inspect-128tok.json").exists()
    assert (raw_dir / "inspect-512tok.json").exists()
