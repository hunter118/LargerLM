from __future__ import annotations

import json
import os
import shlex
import subprocess
import struct
from pathlib import Path

import largerlm.artifact_status as artifact_status_module
from largerlm.artifact_status import inspect_checkpoint_artifact
from largerlm.prefill_backend import default_prefill_backend_probe_path
from largerlm.cli import main as cli_main
from largerlm.safety import DiskBudget
from largerlm.safetensors import HEADER_MANIFEST_NAME


def _write_artifact_metadata(root: Path) -> None:
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    index = {
        "metadata": {"total_size": 12},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    (root / HEADER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "source": {
                    "repo": "mlx-community/GLM-5.2-mxfp4",
                    "revision": "main",
                    "endpoint": "https://huggingface.co",
                },
                "index": index,
                "shards": {
                    "model-00001-of-00002.safetensors": {
                        "file_size": 100,
                        "data_start": 40,
                        "header": {},
                    },
                    "model-00002-of-00002.safetensors": {
                        "file_size": 200,
                        "data_start": 50,
                        "header": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _safetensors_bytes(header: dict[str, object], payload_bytes: int) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"\0" * payload_bytes


def _write_valid_header_artifact(root: Path) -> dict[str, bytes]:
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    index = {
        "metadata": {"total_size": 12},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    shard_specs = {
        "model-00001-of-00002.safetensors": (
            "model.layers.0.weight",
            4,
        ),
        "model-00002-of-00002.safetensors": (
            "model.layers.1.weight",
            8,
        ),
    }
    shard_bytes: dict[str, bytes] = {}
    manifest_shards: dict[str, object] = {}
    for shard, (tensor_name, payload_bytes) in shard_specs.items():
        header = {
            tensor_name: {
                "dtype": "U8",
                "shape": [payload_bytes],
                "data_offsets": [0, payload_bytes],
            }
        }
        payload = _safetensors_bytes(header, payload_bytes)
        data_start = len(payload) - payload_bytes
        shard_bytes[shard] = payload
        manifest_shards[shard] = {
            "file_size": len(payload),
            "data_start": data_start,
            "header": header,
        }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    (root / HEADER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "index": index,
                "shards": manifest_shards,
            }
        ),
        encoding="utf-8",
    )
    for shard, payload in shard_bytes.items():
        (root / shard).write_bytes(payload)
    return shard_bytes


def _write_prepared_probe_manifest(prepared: Path) -> None:
    prepared.mkdir(exist_ok=True)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_dir": str(prepared.parent),
                "experts_layout": "experts/layout.json",
                "resident_layout": "resident/layout.json",
                "decode_cache_layout": "decode_cache_layout.json",
                "decode_cache_file": "decode_cache.bin",
            }
        ),
        encoding="utf-8",
    )


def _write_preflight_probe_report(path: Path, model_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "model_dir": str(model_dir),
                "public_glm_5_2_shape": {"matches": True},
            }
        ),
        encoding="utf-8",
    )


def _write_prepare_dry_run_probe_report(
    path: Path,
    *,
    model_dir: Path,
    output_dir: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "ok": True,
                "executed": False,
                "paths": {"output_dir": str(output_dir)},
                "preflight": {
                    "ok": True,
                    "model_dir": str(model_dir),
                    "public_glm_5_2_shape": {"matches": True},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_prefill_backend_probe_report(path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "host_probe_requested": True,
                "host_probe_ran": True,
                "host_probe_ok": True,
                "host_probe_path": str(default_prefill_backend_probe_path()),
                "mps_graph_probe_requested": True,
                "mps_graph_probe_ran": True,
                "mps_graph_probe_ok": True,
                "mps_graph_runtime_available": True,
                "mpp_compile_probe_requested": True,
                "mpp_compile_probe_ran": True,
                "mpp_compile_probe_ok": False,
                "mpp_run_probe_requested": True,
                "mpp_run_probe_ran": True,
                "mpp_run_probe_ok": False,
                "probe_timeout_seconds": 30.0,
                "validated_accelerated_prefill_backends": ["mpsgraph-f32"],
                "suggested_prefill_acceleration_flags": {
                    "runtime_probe_satisfied": True,
                    "runtime_probe_argv": ["--run-mpsgraph-probe"],
                    "prefill_linear_backend": "mpsgraph-f32",
                },
                "prefill_neural_accelerator_status": {
                    "status": "missing_public_mpp_symbols",
                    "recommended_backend": "mpsgraph_prefill_fallback",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_launch_profile_probe(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source": "unit",
                "argv_safe_to_replay": True,
                "argv": ["--min-free-unified-memory-gib", "24"],
            }
        ),
        encoding="utf-8",
    )


def _write_launch_audit_probe(path: Path, *, ok: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "largerlm.launch_audit.v1",
                "launch_audit": {
                    "ok": ok,
                    "failures": [] if ok else ["unit_failure"],
                    "checks": [{"code": "unit", "ok": ok}],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_smoke_result_probe(path: Path) -> None:
    prepared = path.parent
    path.write_text(
        json.dumps(
            {
                "schema": "largerlm.prepared_token_generation_result.v1",
                "source": "generate_prepared_token_ids",
                "prepared_manifest": str(prepared / "manifest.json"),
                "request": {
                    "prompt_token_ids": [0],
                    "max_new_tokens": 1,
                    "launch_audit_path": str(prepared / "launch-audit.json"),
                },
                "token_result": {
                    "prompt_token_ids": [0],
                    "generated_token_ids": [1],
                    "steps": [],
                    "applied_launch_profile": {
                        "path": str(prepared / "launch-profile.json"),
                        "locked": True,
                        "lock_required": True,
                        "matches_prepared": True,
                        "current_prepared_manifest": str(
                            prepared / "manifest.json"
                        ),
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_artifact_status_uses_header_manifest_without_shards(
    tmp_path: Path,
) -> None:
    _write_artifact_metadata(tmp_path)

    status = inspect_checkpoint_artifact(tmp_path)

    assert status.header_manifest_valid is True
    assert status.expected_shard_count == 2
    assert status.present_shard_count == 0
    assert status.missing_shard_count == 2
    assert status.expected_tensor_bytes == 12
    assert status.expected_safetensors_file_bytes == 300
    assert status.remaining_safetensors_file_bytes == 300
    assert status.download_disk_budget is not None
    assert status.download_disk_budget.required_bytes == 300
    assert status.download_disk_ok is True
    assert status.download_complete is False
    assert status.artifact_clean is False
    assert status.can_run_metadata_preflight is True
    assert status.can_attempt_prepare is False
    assert status.preflight_command is not None
    assert "--max-cache-gib" in status.preflight_command
    assert status.preflight_command[
        status.preflight_command.index("--group-size") + 1
    ] == "32"
    assert "--disk-margin-gib" in status.preflight_command
    assert "--unified-memory-gib" in status.preflight_command
    assert "--metadata-only" in status.preflight_command
    assert status.prepare_dry_run_command is not None
    assert "--max-cache-gib" in status.prepare_dry_run_command
    assert status.prepare_dry_run_command[
        status.prepare_dry_run_command.index("--group-size") + 1
    ] == "32"
    assert "--disk-margin-gib" in status.prepare_dry_run_command
    assert "--unified-memory-gib" in status.prepare_dry_run_command
    assert "--metadata-only" in status.prepare_dry_run_command
    assert status.prepare_execute_command is None
    assert status.inspect_prepared_command is None
    assert status.launch_audit_command is None
    assert status.minimal_smoke_command is None
    assert status.prefill_backend_command is not None
    assert "--write-report" in status.prefill_backend_command
    assert status.prefill_backend_command[-2].endswith("prefill-backend-report.json")
    assert status.prefill_backend_report_present is False
    assert status.prefill_backend_report_valid is False
    assert status.prefill_backend_report_error is None
    assert [step.step_id for step in status.bringup_plan] == [
        "fetch_headers",
        "download_precheck",
        "prefill_backend_probe",
        "preflight",
        "prepare_dry_run",
        "download_weights",
        "post_copy_check",
        "prepare_execute",
        "inspect_prepared",
        "launch_audit",
        "minimal_smoke",
    ]
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["download_precheck"].step_status == "complete"
    assert plan_by_id["download_precheck"].command_available is False
    assert plan_by_id["prefill_backend_probe"].step_status == "ready"
    assert plan_by_id["prefill_backend_probe"].writes_artifacts is True
    assert plan_by_id["prefill_backend_probe"].reads_weight_payloads is False
    assert plan_by_id["prefill_backend_probe"].prerequisite_step_ids == (
        "download_precheck",
    )
    assert plan_by_id["download_weights"].step_status == "ready"
    assert plan_by_id["download_weights"].command_available is True
    assert plan_by_id["download_weights"].prerequisite_step_ids == (
        "prepare_dry_run",
    )
    assert plan_by_id["preflight"].command_available is True
    assert plan_by_id["preflight"].prerequisite_step_ids == (
        "prefill_backend_probe",
    )
    assert plan_by_id["prepare_dry_run"].command_available is True
    assert plan_by_id["prepare_dry_run"].prerequisite_step_ids == ("preflight",)
    assert plan_by_id["prepare_execute"].command_available is False
    assert plan_by_id["prepare_execute"].prerequisite_step_ids == (
        "post_copy_check",
    )
    assert plan_by_id["prepare_execute"].blocked_reason == (
        "complete local safetensors shards are not ready"
    )
    assert plan_by_id["minimal_smoke"].runs_model is True
    assert plan_by_id["minimal_smoke"].reads_weight_payloads is True
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prefill_backend_probe"
    assert status.next_bringup_step.command == status.prefill_backend_command
    assert status.hf_download_command == (
        "hf",
        "download",
        "mlx-community/GLM-5.2-mxfp4",
        "--revision",
        "main",
        "--local-dir",
        str(tmp_path),
    )
    assert status.download_precheck_command == (
        "python",
        "-m",
        "largerlm",
        "checkpoint-status",
        str(tmp_path),
        "--repo",
        "mlx-community/GLM-5.2-mxfp4",
        "--revision",
        "main",
        "--endpoint",
        "https://huggingface.co",
        "--download-disk-margin-gib",
        "16",
        "--require-download-disk-ok",
        "--json",
    )
    assert status.post_copy_check_command == (
        "python",
        "-m",
        "largerlm",
        "checkpoint-status",
        str(tmp_path),
        "--repo",
        "mlx-community/GLM-5.2-mxfp4",
        "--revision",
        "main",
        "--endpoint",
        "https://huggingface.co",
        "--verify-local-headers",
        "--require-complete",
        "--require-clean",
        "--json",
    )
    assert status.shards[0].url == (
        "https://huggingface.co/mlx-community/GLM-5.2-mxfp4/resolve/main/"
        "model-00001-of-00002.safetensors"
    )


def test_checkpoint_artifact_status_reports_partial_and_complete_shards(
    tmp_path: Path,
) -> None:
    _write_artifact_metadata(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 199)

    status = inspect_checkpoint_artifact(tmp_path)

    assert status.present_shard_count == 2
    assert status.complete_shard_count == 1
    assert status.partial_shard_count == 1
    assert status.remaining_safetensors_file_bytes == 1
    assert status.download_complete is False
    assert status.download_complete_proven is False
    assert status.can_attempt_prepare is False
    partial = next(shard for shard in status.shards if shard.issue == "truncated")
    assert partial.name == "model-00002-of-00002.safetensors"


def test_checkpoint_status_human_output_prints_metadata_dry_run(
    tmp_path: Path,
    capsys,
) -> None:
    _write_artifact_metadata(tmp_path)

    status = cli_main(["checkpoint-status", str(tmp_path)])

    assert status == 0
    output = capsys.readouterr().out
    assert "preflight command:" in output
    assert "prepare dry-run:" in output
    assert "--metadata-only" in output
    assert "--group-size 32" in output


def test_checkpoint_artifact_status_proves_complete_download(tmp_path: Path) -> None:
    _write_artifact_metadata(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 200)

    status = inspect_checkpoint_artifact(tmp_path)

    assert status.download_complete is True
    assert status.download_complete_proven is True
    assert status.artifact_clean is True
    assert status.remaining_safetensors_file_bytes == 0
    assert status.download_disk_budget is not None
    assert status.download_disk_budget.required_bytes == 0
    assert status.can_run_metadata_preflight is True
    assert status.can_attempt_prepare is True
    assert status.preflight_command is not None
    assert "--metadata-only" not in status.preflight_command
    assert status.prepare_execute_command is None
    assert status.inspect_prepared_command is None
    assert status.launch_audit_command is None
    assert status.minimal_smoke_command is None


def test_checkpoint_artifact_status_verifies_local_headers(tmp_path: Path) -> None:
    _write_valid_header_artifact(tmp_path)
    _write_prefill_backend_probe_report(
        tmp_path / "largerlm-prepared" / "prefill-backend-report.json"
    )

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.local_header_check_requested is True
    assert status.local_header_checked_shard_count == 2
    assert status.local_header_ok_shard_count == 2
    assert status.local_header_error_count == 0
    assert status.local_headers_ok is True
    assert status.can_attempt_prepare is True
    assert status.prefill_backend_report_present is True
    assert status.prefill_backend_report_valid is True
    assert all(shard.header_checked for shard in status.shards)
    assert all(shard.header_ok is True for shard in status.shards)
    assert {shard.header_tensor_count for shard in status.shards} == {1}
    prepared = tmp_path / "largerlm-prepared"
    assert status.prepare_execute_command == (
        "python",
        "-m",
        "largerlm",
        "prepare-glm",
        str(tmp_path),
        "--output-dir",
        str(prepared),
        "--auto-context-from-budget",
        "--quant-bits",
        "4",
        "--group-size",
        "32",
        "--require-public-glm-5-2-shape",
        "--max-cache-gib",
        "16",
        "--disk-margin-gib",
        "32",
        "--unified-memory-gib",
        "128",
        "--auto-cold-read-benchmark",
        "--cold-read-benchmark-mib",
        "1024",
        "--cold-read-benchmark-chunk-mib",
        "8",
        "--execute",
        "--json",
    )
    assert status.inspect_prepared_command == (
        "python",
        "-m",
        "largerlm",
        "inspect-prepared",
        str(prepared),
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--check-prompt-tokens",
        "1",
        "--check-max-new-tokens",
        "1",
        "--check-runtime-preflight",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
        "--prefill-backend-probe-timeout-seconds",
        "30",
        "--write-launch-profile",
        str(prepared / "launch-profile.json"),
        "--json",
    )
    assert status.launch_audit_command == (
        "python",
        "-m",
        "largerlm",
        "inspect-prepared",
        str(prepared),
        "--apply-launch-profile",
        str(prepared / "launch-profile.json"),
        "--lock-launch-profile",
        "--require-locked-launch-profile",
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--allow-non-accelerated-prefill-launch-audit",
        "--prefill-prompt-chunk-tokens",
        "17",
        "--prefill-max-routed-read-amplification",
        "67.2",
        "--prefill-max-routed-read-gib",
        "24097.5",
        "--prefill-max-routed-read-seconds",
        "4074.3",
        "--prefill-max-stage-mib",
        "2731.61",
        "--prefill-max-compact-stage-mib",
        "2731.05",
        "--prefill-max-stage-raw-ranges",
        "143",
        "--prefill-max-stage-coalesced-ranges",
        "143",
        "--prefill-expert-stage-tiling",
        "--check-prompt-tokens",
        "2048",
        "--check-max-new-tokens",
        "1",
        "--check-metal-final-logits",
        "--check-runtime-preflight",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
        "--prefill-backend-probe-timeout-seconds",
        "30",
        "--require-launch-audit",
        "--write-launch-audit",
        str(prepared / "launch-audit.json"),
        "--json",
    )
    assert status.minimal_smoke_command == (
        "python",
        "-m",
        "largerlm",
        "generate-prepared-token-ids",
        str(prepared),
        "--apply-launch-profile",
        str(prepared / "launch-profile.json"),
        "--lock-launch-profile",
        "--require-locked-launch-profile",
        "--require-launch-audit",
        str(prepared / "launch-audit.json"),
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--allow-non-accelerated-prefill-launch-audit",
        "--prefill-prompt-chunk-tokens",
        "17",
        "--prefill-max-routed-read-amplification",
        "67.2",
        "--prefill-max-routed-read-gib",
        "24097.5",
        "--prefill-max-routed-read-seconds",
        "4074.3",
        "--prefill-max-stage-mib",
        "2731.61",
        "--prefill-max-compact-stage-mib",
        "2731.05",
        "--prefill-max-stage-raw-ranges",
        "143",
        "--prefill-max-stage-coalesced-ranges",
        "143",
        "--prefill-expert-stage-tiling",
        "--prefill-static-capacity-per-expert",
        "auto",
        "--metal-final-logits",
        "--run-mpp-probe",
        "--run-mpsgraph-probe",
        "--prefill-backend-probe-timeout-seconds",
        "30",
        "--prompt-token-ids",
        "0",
        "--max-new-tokens",
        "1",
        "--write-result",
        str(prepared / "minimal-smoke.json"),
        "--quiet-runner",
        "--json",
    )
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["fetch_headers"].command_available is False
    assert plan_by_id["fetch_headers"].step_status == "complete"
    assert plan_by_id["download_weights"].command_available is False
    assert plan_by_id["download_weights"].step_status == "complete"
    assert plan_by_id["post_copy_check"].step_status == "complete"
    assert plan_by_id["prepare_execute"].command_available is True
    assert plan_by_id["prepare_execute"].reads_weight_payloads is True
    assert plan_by_id["prepare_execute"].writes_artifacts is True
    assert plan_by_id["launch_audit"].command_available is True
    assert plan_by_id["launch_audit"].writes_artifacts is True
    assert plan_by_id["minimal_smoke"].command_available is True
    assert plan_by_id["minimal_smoke"].prerequisite_step_ids == ("launch_audit",)
    assert plan_by_id["minimal_smoke"].runs_model is True
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "preflight"
    assert status.next_bringup_step.runs_model is False


def test_checkpoint_artifact_status_rejects_stale_prefill_backend_probe_timeout(
    tmp_path: Path,
) -> None:
    _write_valid_header_artifact(tmp_path)
    report_path = tmp_path / "largerlm-prepared" / "prefill-backend-report.json"
    _write_prefill_backend_probe_report(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["probe_timeout_seconds"] = 5.0
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prefill_backend_report_present is True
    assert status.prefill_backend_report_valid is False
    assert status.prefill_backend_report_error is not None
    assert "probe_timeout_seconds must be at least 30" in (
        status.prefill_backend_report_error
    )
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prefill_backend_probe"


def test_checkpoint_artifact_status_rejects_foreign_prefill_backend_probe_path(
    tmp_path: Path,
) -> None:
    _write_valid_header_artifact(tmp_path)
    report_path = tmp_path / "largerlm-prepared" / "prefill-backend-report.json"
    _write_prefill_backend_probe_report(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["host_probe_path"] = "/tmp/other-prefill-backend-probe"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prefill_backend_report_present is True
    assert status.prefill_backend_report_valid is False
    assert status.prefill_backend_report_error is not None
    assert "host_probe_path must be" in status.prefill_backend_report_error
    assert str(default_prefill_backend_probe_path()) in (
        status.prefill_backend_report_error
    )
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prefill_backend_probe"


def test_checkpoint_artifact_status_rejects_stale_prefill_backend_probe_hash(
    tmp_path: Path,
) -> None:
    _write_valid_header_artifact(tmp_path)
    report_path = tmp_path / "largerlm-prepared" / "prefill-backend-report.json"
    _write_prefill_backend_probe_report(report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["host_probe_sha256"] = "0" * 64
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prefill_backend_report_present is True
    assert status.prefill_backend_report_valid is False
    assert status.prefill_backend_report_error is not None
    assert "host_probe_sha256 does not match current probe" in (
        status.prefill_backend_report_error
    )
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prefill_backend_probe"


def test_checkpoint_artifact_status_validates_bringup_artifacts(
    tmp_path: Path,
) -> None:
    _write_valid_header_artifact(tmp_path)
    prepared = tmp_path / "largerlm-prepared"

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prepared_manifest_present is False
    assert status.prepared_manifest_valid is False
    assert status.prepared_manifest_error is None
    assert status.prefill_backend_report_present is False
    assert status.prefill_backend_report_valid is False
    assert status.prefill_backend_report_error is None
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["prefill_backend_probe"].step_status == "ready"
    assert plan_by_id["prepare_execute"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prefill_backend_probe"

    prepared.mkdir()
    (prepared / "prefill-backend-report.json").write_text("{}", encoding="utf-8")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prefill_backend_report_present is True
    assert status.prefill_backend_report_valid is False
    assert status.prefill_backend_report_error is not None
    assert "host_probe_requested must be true" in (
        status.prefill_backend_report_error
    )
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prefill_backend_probe"

    _write_prefill_backend_probe_report(prepared / "prefill-backend-report.json")
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prefill_backend_report_valid is True
    assert status.prepared_manifest_present is True
    assert status.prepared_manifest_valid is False
    assert status.prepared_manifest_error is not None
    assert "version must be 1" in status.prepared_manifest_error
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["prepare_execute"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "preflight"

    (prepared / "preflight-report.json").write_text("{}", encoding="utf-8")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.preflight_report_present is True
    assert status.preflight_report_valid is False
    assert status.preflight_report_error is not None
    assert "ok must be true" in status.preflight_report_error
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "preflight"

    _write_preflight_probe_report(prepared / "preflight-report.json", tmp_path)

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.preflight_report_valid is True
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["preflight"].step_status == "complete"
    assert plan_by_id["prepare_dry_run"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prepare_dry_run"

    _write_prepare_dry_run_probe_report(
        prepared / "prepare-dry-run-report.json",
        model_dir=tmp_path,
        output_dir=prepared,
    )

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prepare_dry_run_report_valid is True
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["prepare_dry_run"].step_status == "complete"
    assert plan_by_id["prepare_execute"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "prepare_execute"

    _write_prepared_probe_manifest(prepared)

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.prepared_manifest_valid is True
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["preflight"].step_status == "complete"
    assert plan_by_id["prepare_dry_run"].step_status == "complete"
    assert plan_by_id["prepare_execute"].step_status == "complete"
    assert plan_by_id["inspect_prepared"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "inspect_prepared"

    (prepared / "launch-profile.json").write_text(
        json.dumps({"argv": []}),
        encoding="utf-8",
    )

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.launch_profile_present is True
    assert status.launch_profile_valid is False
    assert status.launch_profile_error is not None
    assert "non-empty array" in status.launch_profile_error
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "inspect_prepared"

    _write_launch_profile_probe(prepared / "launch-profile.json")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.launch_profile_valid is True
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["inspect_prepared"].step_status == "complete"
    assert plan_by_id["launch_audit"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "launch_audit"

    _write_launch_audit_probe(prepared / "launch-audit.json", ok=False)

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.launch_audit_present is True
    assert status.launch_audit_valid is False
    assert status.launch_audit_error is not None
    assert "launch_audit.ok must be true" in status.launch_audit_error
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "launch_audit"

    _write_launch_audit_probe(prepared / "launch-audit.json", ok=True)

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.launch_audit_valid is True
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["launch_audit"].step_status == "complete"
    assert plan_by_id["minimal_smoke"].step_status == "ready"
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "minimal_smoke"
    assert status.next_bringup_step.runs_model is True

    _write_minimal_smoke_result_probe(prepared / "minimal-smoke.json")
    stale_payload = json.loads(
        (prepared / "minimal-smoke.json").read_text(encoding="utf-8")
    )
    stale_payload["prepared_manifest"] = str(tmp_path / "other" / "manifest.json")
    (prepared / "minimal-smoke.json").write_text(
        json.dumps(stale_payload),
        encoding="utf-8",
    )

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.minimal_smoke_result_present is True
    assert status.minimal_smoke_result_valid is False
    assert status.minimal_smoke_result_error is not None
    assert "prepared_manifest must be" in status.minimal_smoke_result_error
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "minimal_smoke"

    (prepared / "minimal-smoke.json").write_text(
        json.dumps(
            {
                "schema": "largerlm.prepared_token_generation_result.v1",
                "source": "generate_prepared_token_ids",
                "prepared_manifest": str(prepared / "manifest.json"),
                "request": {
                    "prompt_token_ids": [0],
                    "max_new_tokens": 1,
                    "launch_audit_path": str(prepared / "launch-audit.json"),
                },
                "token_result": {
                    "prompt_token_ids": [0],
                    "generated_token_ids": [],
                    "steps": [],
                    "applied_launch_profile": {
                        "path": str(prepared / "launch-profile.json"),
                        "locked": True,
                        "lock_required": True,
                        "matches_prepared": True,
                        "current_prepared_manifest": str(
                            prepared / "manifest.json"
                        ),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.minimal_smoke_result_present is True
    assert status.minimal_smoke_result_valid is False
    assert status.minimal_smoke_result_error is not None
    assert "generated_token_ids must be non-empty" in (
        status.minimal_smoke_result_error
    )
    assert status.next_bringup_step is not None
    assert status.next_bringup_step.step_id == "minimal_smoke"

    _write_minimal_smoke_result_probe(prepared / "minimal-smoke.json")

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.minimal_smoke_result_valid is True
    plan_by_id = {step.step_id: step for step in status.bringup_plan}
    assert plan_by_id["minimal_smoke"].step_status == "complete"
    assert plan_by_id["minimal_smoke"].command_available is False
    assert status.next_bringup_step is None


def test_checkpoint_artifact_status_rejects_corrupt_local_header(
    tmp_path: Path,
    capsys,
) -> None:
    shard_bytes = _write_valid_header_artifact(tmp_path)
    bad_shard = tmp_path / "model-00001-of-00002.safetensors"
    bad_shard.write_bytes(b"x" * len(shard_bytes[bad_shard.name]))

    status = inspect_checkpoint_artifact(tmp_path, verify_local_headers=True)

    assert status.download_complete_proven is True
    assert status.local_headers_ok is False
    assert status.local_header_error_count == 1
    assert status.can_attempt_prepare is False
    bad = next(shard for shard in status.shards if shard.name == bad_shard.name)
    assert bad.header_checked is True
    assert bad.header_ok is False
    assert bad.header_error is not None
    assert "header length" in bad.header_error
    assert any("local safetensors header check failed" in issue for issue in status.issues)

    cli_status = cli_main(
        [
            "checkpoint-status",
            str(tmp_path),
            "--verify-local-headers",
            "--require-complete",
            "--json",
        ]
    )

    assert cli_status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["download_complete_proven"] is True
    assert payload["local_header_error_count"] == 1
    assert payload["can_attempt_prepare"] is False


def test_checkpoint_status_cli_json_and_require_complete(
    tmp_path: Path,
    capsys,
) -> None:
    _write_artifact_metadata(tmp_path)
    missing_json = tmp_path / "missing-shards.json"
    missing_urls = tmp_path / "missing-shards.txt"
    next_json = tmp_path / "next-bringup.json"
    next_sh = tmp_path / "next-bringup.sh"
    external_json = tmp_path / "external-download.json"
    external_sh = tmp_path / "external-download.sh"
    status_json = tmp_path / "checkpoint-status.json"

    status = cli_main(
        [
            "checkpoint-status",
            str(tmp_path),
            "--write-missing-shards-json",
            str(missing_json),
            "--write-missing-shards-urls",
            str(missing_urls),
            "--write-next-bringup-json",
            str(next_json),
            "--write-next-bringup-sh",
            str(next_sh),
            "--write-external-download-json",
            str(external_json),
            "--write-external-download-sh",
            str(external_sh),
            "--write-status-json",
            str(status_json),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["expected_shard_count"] == 2
    assert payload["missing_shard_count"] == 2
    assert payload["remaining_safetensors_file_bytes"] == 300
    assert payload["download_disk_budget"]["required_bytes"] == 300
    assert payload["download_disk_ok"] is True
    assert payload["artifact_clean"] is False
    assert payload["can_run_metadata_preflight"] is True
    assert payload["download_complete_proven"] is False
    assert payload["prepare_execute_command"] is None
    assert payload["inspect_prepared_command"] is None
    assert payload["launch_audit_command"] is None
    assert payload["minimal_smoke_command"] is None
    assert payload["prepared_manifest_present"] is False
    assert payload["prepared_manifest_valid"] is False
    assert payload["prepared_manifest_error"] is None
    assert payload["launch_profile_present"] is False
    assert payload["launch_profile_valid"] is False
    assert payload["launch_profile_error"] is None
    assert payload["launch_audit_present"] is False
    assert payload["launch_audit_valid"] is False
    assert payload["launch_audit_error"] is None
    assert payload["minimal_smoke_result_present"] is False
    assert payload["minimal_smoke_result_valid"] is False
    assert payload["minimal_smoke_result_error"] is None
    assert payload["prefill_backend_report_present"] is False
    assert payload["prefill_backend_report_valid"] is False
    assert payload["prefill_backend_report_error"] is None
    assert payload["prefill_backend_command"] is not None
    assert "--write-report" in payload["prefill_backend_command"]
    assert (
        payload["prefill_backend_command"][-2]
        == str(tmp_path / "largerlm-prepared" / "prefill-backend-report.json")
    )
    assert [step["step_id"] for step in payload["bringup_plan"]] == [
        "fetch_headers",
        "download_precheck",
        "prefill_backend_probe",
        "preflight",
        "prepare_dry_run",
        "download_weights",
        "post_copy_check",
        "prepare_execute",
        "inspect_prepared",
        "launch_audit",
        "minimal_smoke",
    ]
    assert payload["bringup_plan"][-1]["runs_model"] is True
    assert payload["bringup_plan"][-1]["command_available"] is False
    assert payload["bringup_plan"][1]["step_status"] == "complete"
    assert payload["bringup_plan"][2]["step_id"] == "prefill_backend_probe"
    assert payload["bringup_plan"][2]["writes_artifacts"] is True
    assert payload["bringup_plan"][2]["reads_weight_payloads"] is False
    assert payload["next_bringup_step"]["step_id"] == "prefill_backend_probe"
    assert (
        payload["next_bringup_step"]["command"]
        == payload["prefill_backend_command"]
    )
    assert payload["download_precheck_command"][-2:] == [
        "--require-download-disk-ok",
        "--json",
    ]
    assert "--fetch-small-files" in payload["header_fetch_command"]
    assert payload["post_copy_check_command"][-4:] == [
        "--verify-local-headers",
        "--require-complete",
        "--require-clean",
        "--json",
    ]
    status_payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert status_payload["model_dir"] == payload["model_dir"]
    assert status_payload["expected_shard_count"] == 2
    assert status_payload["missing_shard_count"] == 2
    assert status_payload["remaining_safetensors_file_bytes"] == 300
    assert status_payload["next_bringup_step"] == payload["next_bringup_step"]
    assert status_payload["bringup_plan"] == payload["bringup_plan"]
    missing_payload = json.loads(missing_json.read_text(encoding="utf-8"))
    assert missing_payload["remaining_safetensors_file_bytes"] == 300
    assert [entry["name"] for entry in missing_payload["entries"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert missing_urls.read_text(encoding="utf-8").splitlines() == [
        "https://huggingface.co/mlx-community/GLM-5.2-mxfp4/resolve/main/"
        "model-00001-of-00002.safetensors",
        "https://huggingface.co/mlx-community/GLM-5.2-mxfp4/resolve/main/"
        "model-00002-of-00002.safetensors",
    ]
    next_payload = json.loads(next_json.read_text(encoding="utf-8"))
    assert next_payload["version"] == 1
    assert next_payload["ready"] is True
    assert next_payload["step_id"] == "prefill_backend_probe"
    assert next_payload["command"] == payload["prefill_backend_command"]
    assert next_payload["next_bringup_step"]["step_id"] == "prefill_backend_probe"
    assert next_payload["writes_artifacts"] is True
    assert next_payload["reads_weight_payloads"] is False
    script = next_sh.read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "# step_id: prefill_backend_probe" in script
    assert "# writes_artifacts: true" in script
    assert "exec python -m largerlm prefill-backend" in script
    assert "--run-mpsgraph-probe" in script
    assert "--run-mpp-probe" in script
    assert "--write-report" in script
    assert "prefill-backend-report.json" in script
    assert next_sh.stat().st_mode & 0o111
    external_payload = json.loads(external_json.read_text(encoding="utf-8"))
    assert external_payload["schema"] == "largerlm.external_safetensors_download.v1"
    assert external_payload["model_dir"] == str(tmp_path)
    assert external_payload["entry_count"] == 2
    assert external_payload["remaining_safetensors_file_bytes"] == 300
    assert (
        external_payload["post_copy_check_command"]
        == payload["post_copy_check_command"]
    )
    first_external = external_payload["entries"][0]
    assert first_external == {
        "name": "model-00001-of-00002.safetensors",
        "url": (
            "https://huggingface.co/mlx-community/GLM-5.2-mxfp4/resolve/main/"
            "model-00001-of-00002.safetensors"
        ),
        "target_path": str(tmp_path / "model-00001-of-00002.safetensors"),
        "issue": "missing",
        "expected_file_bytes": 100,
        "actual_file_bytes": None,
        "remaining_file_bytes": 100,
        "expected_data_start": 40,
        "expected_header": {},
    }
    external_script = external_sh.read_text(encoding="utf-8")
    assert external_script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert (
        "curl -L --fail --retry 5 --retry-delay 2 --connect-timeout"
        in external_script
    )
    assert "--speed-limit" in external_script
    assert "--speed-time" in external_script
    assert "LARGERLM_DOWNLOAD_CONNECT_TIMEOUT_SECONDS" in external_script
    assert "LARGERLM_DOWNLOAD_LOW_SPEED_LIMIT_BYTES_PER_SECOND" in external_script
    assert "LARGERLM_DOWNLOAD_LOW_SPEED_TIME_SECONDS" in external_script
    assert "Unsupported content type" in external_script
    assert "MODEL_DIR=${LARGERLM_MODEL_DIR:-" in external_script
    assert "LARGERLM_DOWNLOAD_START_INDEX" in external_script
    assert "LARGERLM_DOWNLOAD_END_INDEX" in external_script
    assert "LARGERLM_DOWNLOAD_MAX_BYTES" in external_script
    assert (
        "maybe_download_one 1 https://huggingface.co/mlx-community/GLM-5.2-mxfp4/"
        "resolve/main/model-00001-of-00002.safetensors"
        in external_script
    )
    assert '"$MODEL_DIR"/model-00001-of-00002.safetensors' in external_script
    assert "python -m largerlm checkpoint-status" in external_script
    assert "--verify-local-headers" in external_script
    assert "--require-clean" in external_script
    assert external_sh.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(external_sh)], check=True)

    status = cli_main(["checkpoint-status", str(tmp_path), "--require-complete"])

    assert status == 1


def test_checkpoint_status_next_bringup_uses_bounded_range_downloader(
    tmp_path: Path,
    capsys,
) -> None:
    _write_artifact_metadata(tmp_path)
    prepared = tmp_path / "largerlm-prepared"
    _write_prefill_backend_probe_report(prepared / "prefill-backend-report.json")
    _write_preflight_probe_report(prepared / "preflight-report.json", tmp_path)
    _write_prepare_dry_run_probe_report(
        prepared / "prepare-dry-run-report.json",
        model_dir=tmp_path,
        output_dir=prepared,
    )
    next_json = tmp_path / "next-bringup.json"
    next_sh = tmp_path / "next-bringup.sh"
    external_json = tmp_path / "external-download.json"

    status = cli_main(
        [
            "checkpoint-status",
            str(tmp_path),
            "--write-next-bringup-json",
            str(next_json),
            "--write-next-bringup-sh",
            str(next_sh),
            "--write-external-download-json",
            str(external_json),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["next_bringup_step"]["step_id"] == "download_weights"
    assert payload["next_bringup_step"]["command"][:2] == ["hf", "download"]

    next_payload = json.loads(next_json.read_text(encoding="utf-8"))
    assert next_payload["step_id"] == "download_weights"
    assert next_payload["command"][:3] == [
        "python",
        "scripts/download_safetensors_ranges.py",
        str(external_json),
    ]
    assert next_payload["next_bringup_step"]["command"] == next_payload["command"]
    assert "--model-dir" in next_payload["command"]
    assert next_payload["command"][
        next_payload["command"].index("--model-dir") + 1
    ] == str(tmp_path)
    assert next_payload["command"][
        next_payload["command"].index("--start-index") + 1
    ] == "1"
    assert next_payload["command"][
        next_payload["command"].index("--end-index") + 1
    ] == "1"
    assert next_payload["command"][
        next_payload["command"].index("--max-bytes") + 1
    ] == str(6 * 1024**3)
    assert "--json" in next_payload["command"]

    script = next_sh.read_text(encoding="utf-8")
    assert "# step_id: download_weights" in script
    assert "# download transport: bounded_http_range" in script
    assert "exec python scripts/download_safetensors_ranges.py" in script
    assert shlex.quote(str(external_json)) in script
    assert "--start-index 1 --end-index 1" in script
    subprocess.run(["bash", "-n", str(next_sh)], check=True)


def test_checkpoint_status_external_download_script_runs_with_file_endpoint(
    tmp_path: Path,
    capsys,
) -> None:
    model = tmp_path / "model"
    _write_artifact_metadata(model)
    source_root = tmp_path / "source"
    source = source_root / "tiny-repo" / "resolve" / "main"
    source.mkdir(parents=True)
    (source / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
    (source / "model-00002-of-00002.safetensors").write_bytes(b"b" * 200)
    external_sh = tmp_path / "download.sh"

    cli_status = cli_main(
        [
            "checkpoint-status",
            str(model),
            "--repo",
            "tiny-repo",
            "--endpoint",
            f"file://{source_root}",
            "--write-external-download-sh",
            str(external_sh),
            "--json",
        ]
    )

    assert cli_status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["missing_shard_count"] == 2
    env = os.environ.copy()
    env["LARGERLM_MODEL_DIR"] = str(model)
    env["LARGERLM_DOWNLOAD_RETRIES"] = "2"
    env["LARGERLM_DOWNLOAD_RETRY_SLEEP_SECONDS"] = "0"
    completed = subprocess.run(
        ["bash", str(external_sh)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "ok:" in completed.stdout
    assert (model / "model-00001-of-00002.safetensors").stat().st_size == 100
    assert (model / "model-00002-of-00002.safetensors").stat().st_size == 200
    status = inspect_checkpoint_artifact(
        model,
        repo="tiny-repo",
        endpoint=f"file://{source_root}",
    )
    assert status.download_complete_proven is True
    assert status.missing_shard_count == 0
    assert status.partial_shard_count == 0


def test_checkpoint_status_external_download_script_can_select_safe_batch(
    tmp_path: Path,
    capsys,
) -> None:
    model = tmp_path / "model"
    _write_artifact_metadata(model)
    source_root = tmp_path / "source"
    source = source_root / "tiny-repo" / "resolve" / "main"
    source.mkdir(parents=True)
    (source / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
    (source / "model-00002-of-00002.safetensors").write_bytes(b"b" * 200)
    external_sh = tmp_path / "download.sh"

    cli_status = cli_main(
        [
            "checkpoint-status",
            str(model),
            "--repo",
            "tiny-repo",
            "--endpoint",
            f"file://{source_root}",
            "--write-external-download-sh",
            str(external_sh),
            "--json",
        ]
    )

    assert cli_status == 0
    assert json.loads(capsys.readouterr().out)["missing_shard_count"] == 2
    env = os.environ.copy()
    env["LARGERLM_MODEL_DIR"] = str(model)
    env["LARGERLM_DOWNLOAD_RETRIES"] = "2"
    env["LARGERLM_DOWNLOAD_RETRY_SLEEP_SECONDS"] = "0"
    env["LARGERLM_DOWNLOAD_START_INDEX"] = "2"
    env["LARGERLM_DOWNLOAD_END_INDEX"] = "2"
    completed = subprocess.run(
        ["bash", str(external_sh)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "skip by shard range" in completed.stdout
    assert not (model / "model-00001-of-00002.safetensors").exists()
    assert (model / "model-00002-of-00002.safetensors").stat().st_size == 200

    capped_model = tmp_path / "capped-model"
    _write_artifact_metadata(capped_model)
    env["LARGERLM_MODEL_DIR"] = str(capped_model)
    env.pop("LARGERLM_DOWNLOAD_START_INDEX")
    env.pop("LARGERLM_DOWNLOAD_END_INDEX")
    env["LARGERLM_DOWNLOAD_MAX_BYTES"] = "150"
    capped = subprocess.run(
        ["bash", str(external_sh)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "skip by byte cap" in capped.stdout
    assert (capped_model / "model-00001-of-00002.safetensors").stat().st_size == 100
    assert not (capped_model / "model-00002-of-00002.safetensors").exists()


def test_checkpoint_status_external_download_script_retries_curl_failures(
    tmp_path: Path,
    capsys,
) -> None:
    model = tmp_path / "model"
    _write_artifact_metadata(model)
    source_root = tmp_path / "missing-source"
    external_sh = tmp_path / "download.sh"

    cli_status = cli_main(
        [
            "checkpoint-status",
            str(model),
            "--repo",
            "tiny-repo",
            "--endpoint",
            f"file://{source_root}",
            "--write-external-download-sh",
            str(external_sh),
            "--json",
        ]
    )

    assert cli_status == 0
    assert json.loads(capsys.readouterr().out)["missing_shard_count"] == 2
    env = os.environ.copy()
    env["LARGERLM_MODEL_DIR"] = str(model)
    env["LARGERLM_DOWNLOAD_RETRIES"] = "2"
    env["LARGERLM_DOWNLOAD_RETRY_SLEEP_SECONDS"] = "0"
    completed = subprocess.run(
        ["bash", str(external_sh)],
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 1
    assert "curl failed for" in completed.stderr
    assert "after 2 attempts" in completed.stderr
    assert not (model / "model-00001-of-00002.safetensors").exists()


def test_checkpoint_status_require_clean_rejects_extra_shards(
    tmp_path: Path,
    capsys,
) -> None:
    _write_artifact_metadata(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 200)
    (tmp_path / "unrelated.safetensors").write_bytes(b"extra")

    status = inspect_checkpoint_artifact(tmp_path)

    assert status.download_complete_proven is True
    assert status.extra_shard_count == 1
    assert status.artifact_clean is False
    assert any("unexpected safetensors shards" in issue for issue in status.issues)

    cli_status = cli_main(
        [
            "checkpoint-status",
            str(tmp_path),
            "--require-complete",
            "--require-clean",
            "--json",
        ]
    )

    assert cli_status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["download_complete_proven"] is True
    assert payload["extra_shard_count"] == 1
    assert payload["artifact_clean"] is False


def test_checkpoint_artifact_status_reports_download_disk_shortfall(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _write_artifact_metadata(tmp_path)

    def fake_disk_budget(output_dir, required_bytes, *, safety_margin_bytes):
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=128,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr(artifact_status_module, "disk_budget", fake_disk_budget)

    status = inspect_checkpoint_artifact(
        tmp_path,
        download_disk_safety_margin_bytes=256,
    )

    assert status.remaining_safetensors_file_bytes == 300
    assert status.download_disk_budget == DiskBudget(
        output_dir=tmp_path,
        required_bytes=300,
        available_bytes=128,
        safety_margin_bytes=256,
    )
    assert status.download_disk_ok is False
    assert any("free disk is below" in issue for issue in status.issues)

    cli_status = cli_main(
        [
            "checkpoint-status",
            str(tmp_path),
            "--download-disk-margin-gib",
            "0",
            "--require-download-disk-ok",
            "--json",
        ]
    )

    assert cli_status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["download_disk_ok"] is False
