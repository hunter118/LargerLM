from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import largerlm.prepare as prepare_module
import largerlm.preflight as preflight_module
from largerlm.benchmark import (
    BenchmarkError,
    benchmark_prepared_token_ids,
    summarize_generation,
)
from largerlm.cli import (
    _REQUIRED_LAUNCH_AUDIT_CHECK_CODES,
    _REQUEST_DECODE_ROUTED_READ_AUDIT_FIELDS,
    _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS,
    _REQUEST_PREFILL_CACHE_IO_AUDIT_FIELDS,
    _REQUEST_PREFILL_ROUTED_CHUNK_FRONTIER_AUDIT_FIELDS,
    _REQUEST_PREFILL_STAGE_TEMP_AUDIT_FIELDS,
    _REQUEST_ROUTED_READ_LAUNCH_CHECK_FIELDS,
    _glm_4bit_readiness_audit_details,
    _launch_audit_from_health,
    _launch_profile_targets_match,
    _launch_profile_allows_missing_prepared_identity,
    _prepare_expert_pack_heap_audit_details,
    _prepare_expert_pack_heap_source_from_prepared,
    _prepare_resident_alias_rewrite_audit_details,
    _prepare_resident_alias_rewrite_source_from_prepared,
    _prepared_runtime_profile_audit_details_from_prepared,
    _prepared_ssd_read_audit_details,
    _prepared_ssd_read_audit_details_from_prepared,
    _request_launch_profile_from_health,
    _request_prefill_backend_policy_flags,
    _prefill_routed_chunk_frontier_errors,
    _require_launch_audit_prefill_acceleration_evidence,
    _require_launch_audit_request_profile_binds_read_budgets,
)
from largerlm.cli import main as cli_main
from largerlm.config import load_config
from largerlm.decode_cache import build_decode_cache_layout
from largerlm.generation_guard import GenerationGuardError, LiveMemoryBudget
from largerlm.layout import DEFAULT_EXPERT_COMPONENTS, MXFP4_EXPERT_COMPONENTS, config_sha256
from largerlm.prepare import PrepareError, prepare_glm_checkpoint
from largerlm.prepared import (
    PreparedManifestError,
    load_prepared_manifest,
    validate_layout_backing_files,
)
from largerlm.prompt_prefill import PromptPrefillResult
from largerlm.routed_read import format_routed_read_guard_flag_float
from largerlm.safety import DiskBudget
from largerlm.safetensors import HEADER_MANIFEST_NAME
from largerlm.server import (
    PreparedGenerationApp,
    PreparedServerConfig,
    PreparedServerError,
    _prefill_acceleration_coverage_summary,
    _prefill_acceleration_launch_profile_flags,
    prepared_glm_4bit_readiness,
    prepared_launch_profile_target,
)
from largerlm.text_generator import TextGenerationResult
from largerlm.token_generator import (
    GeneratedStep,
    TokenGenerationResult,
    TokenGeneratorError,
)
from largerlm.tokenizer import RenderedChatPrompt
from test_preflight import _write_checkpoint
from test_token_generator import COMPONENTS, write_config, write_fixture


def test_request_prefill_backend_policy_flags_bind_request_backend() -> None:
    assert _request_prefill_backend_policy_flags("custom-metal") == {
        "source": "prepared_request_check",
        "prefill_linear_backend": "custom-metal",
        "argv": ("--prefill-linear-backend", "custom-metal"),
    }
    assert _request_prefill_backend_policy_flags(
        {
            "configured": "mpsgraph-f32",
            "effective": "mpsgraph-f32",
        }
    ) == {
        "source": "prepared_request_check",
        "prefill_linear_backend": "mpsgraph-f32",
        "argv": ("--prefill-linear-backend", "mpsgraph-f32"),
    }
    assert _request_prefill_backend_policy_flags(
        {
            "configured": "auto",
            "effective": "mps-matrix-f32",
        }
    ) is None
    assert _request_prefill_backend_policy_flags("bogus") is None
    assert _request_prefill_backend_policy_flags(
        {
            "configured": "auto",
            "effective": "auto",
        }
    ) is None


def test_request_launch_profile_honors_custom_prefill_backend() -> None:
    acceleration = {
        "source": "prepared_health",
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }
    health = {
        "suggested_launch_guard_flags": {
            "source": "prepared_health",
            "argv": ("--min-free-unified-memory-gib", "24"),
        },
        "prefill_backend": {
            "capability": {
                "suggested_prefill_acceleration_flags": acceleration,
            },
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
    }
    request_check = {
        "prefill_linear_backend": "custom-metal",
        "suggested_prefill_guard_flags": {
            "source": "prepared_request_check",
            "argv": ("--prefill-prompt-chunk-tokens", "17"),
        },
    }

    profile = _request_launch_profile_from_health(health, request_check)

    assert profile is not None
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_backend_policy_flags"] == {
        "source": "prepared_request_check",
        "prefill_linear_backend": "custom-metal",
        "argv": ("--prefill-linear-backend", "custom-metal"),
    }
    assert "prefill_acceleration_flags" not in profile["sections"]
    assert "--prefill-linear-backend" in profile["argv"]
    assert "custom-metal" in profile["argv"]
    assert "mpsgraph-f32" not in profile["argv"]
    assert "--require-prefill-acceleration" not in profile["argv"]
    assert "--run-mpsgraph-probe" not in profile["argv"]


def test_request_launch_profile_preserves_auto_mixed_prefill_backend() -> None:
    acceleration = {
        "source": "prepared_health",
        "prefill_linear_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": True,
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }
    probe_flags = {
        "source": "prepared_health",
        "run_mpsgraph_probe": True,
        "argv": ("--run-mpsgraph-probe",),
    }
    runtime_policy = {
        "source": "prepared_health",
        "prefill_mpsgraph_min_batch_tokens": 16,
        "prefill_mpsgraph_min_matrix_dim": 32,
        "require_prefill_acceleration": True,
        "argv": (
            "--prefill-mpsgraph-min-batch-tokens",
            "16",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--require-prefill-acceleration",
        ),
    }
    health = {
        "prefill_backend": {
            "capability": {
                "suggested_prefill_acceleration_flags": acceleration,
            },
        },
        "suggested_prefill_backend_probe_flags": probe_flags,
        "suggested_prefill_runtime_policy_flags": runtime_policy,
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
    }
    request_check = {
        "prefill_linear_backend": {
            "configured": "auto",
            "effective": "auto",
            "mpsgraph_matrix_count": 75,
            "custom_metal_matrix_count": 834,
        },
        "suggested_prefill_guard_flags": {
            "source": "prepared_request_check",
            "argv": ("--prefill-prompt-chunk-tokens", "16"),
        },
    }

    profile = _request_launch_profile_from_health(health, request_check)

    assert profile is not None
    assert profile["argv_safe_to_replay"] is True
    assert "prefill_backend_policy_flags" not in profile["sections"]
    filtered = profile["sections"]["prefill_acceleration_flags"]
    assert filtered["prefill_linear_backend_policy"] == "auto"
    assert "prefill_linear_backend" not in filtered
    assert "--prefill-linear-backend" not in profile["argv"]
    assert "mpsgraph-f32" not in profile["argv"]
    assert "--run-mpsgraph-probe" in profile["argv"]
    assert "--prefill-mpsgraph-min-batch-tokens" in profile["argv"]
    assert "--require-prefill-acceleration" in profile["argv"]


def test_health_launch_profile_preserves_auto_mixed_prefill_backend() -> None:
    acceleration = {
        "source": "prepared_health",
        "prefill_linear_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": True,
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }
    config = PreparedServerConfig(
        prepared_path=Path("prepared"),
        runner_path=Path("runner"),
        prefill_linear_backend="auto",
        require_prefill_acceleration=True,
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
    )

    filtered = _prefill_acceleration_launch_profile_flags(config, acceleration)

    assert filtered is not None
    assert filtered["prefill_linear_backend_policy"] == "auto"
    assert "prefill_linear_backend" not in filtered
    assert filtered["argv"] == (
        "--require-prefill-acceleration",
        "--run-mpsgraph-probe",
    )


def test_required_prefill_acceleration_rejects_unsupported_explicit_f32_backend() -> None:
    coverage = _prefill_acceleration_coverage_summary(
        {
            "analyzed": True,
            "effective": "mpsgraph-f32",
            "matrix_count": 3,
            "accelerated_matrix_count": 1,
            "mpsgraph_matrix_count": 1,
            "mps_matrix_matrix_count": 0,
            "custom_metal_matrix_count": 0,
            "unsupported_mpsgraph_matrix_count": 2,
            "total_estimated_flops": 300,
            "accelerated_estimated_flops": 100,
            "custom_metal_estimated_flops": 0,
            "unsupported_mpsgraph_estimated_flops": 200,
        },
        required=True,
    )

    assert coverage["ok"] is False
    assert coverage["any_resident_matrix_accelerated"] is True
    assert coverage["f32_backend_has_unsupported_matrices"] is True
    assert coverage["non_router_matrix_count"] == 3
    assert coverage["non_router_estimated_flops"] == 300
    assert coverage["non_router_accelerated_matrix_count"] == 1
    assert coverage["non_router_unaccelerated_matrix_count"] == 2
    assert coverage["non_router_unaccelerated_estimated_flops"] == 200
    assert coverage["non_router_unaccelerated_flop_fraction"] == pytest.approx(
        2 / 3
    )
    assert coverage[
        "non_router_unaccelerated_streamed_routed_expert_matrix_count"
    ] == 0
    assert coverage[
        "non_router_unaccelerated_streamed_routed_expert_estimated_flops"
    ] == 0
    assert coverage["non_router_unaccelerated_non_streamed_matrix_count"] == 2
    assert coverage["non_router_unaccelerated_non_streamed_estimated_flops"] == 200
    assert coverage["unaccelerated_backend_matrix_counts"] == {
        "unsupported-mpsgraph": 2
    }
    assert coverage["unaccelerated_backend_estimated_flops"] == {
        "unsupported-mpsgraph": 200
    }
    assert "mpsgraph-f32 cannot run 2 resident prefill matrices" in coverage["reason"]


def test_required_prefill_acceleration_rejects_router_gate_only_by_default() -> None:
    coverage = _prefill_acceleration_coverage_summary(
        {
            "analyzed": True,
            "effective": "auto",
            "matrix_count": 1,
            "accelerated_matrix_count": 1,
            "mpsgraph_matrix_count": 1,
            "mps_matrix_matrix_count": 0,
            "custom_metal_matrix_count": 0,
            "unsupported_mpsgraph_matrix_count": 0,
            "total_estimated_flops": 100,
            "accelerated_estimated_flops": 100,
            "custom_metal_estimated_flops": 0,
            "unsupported_mpsgraph_estimated_flops": 0,
            "router_gate_matrix_count": 1,
            "router_gate_estimated_flops": 100,
            "router_gate_accelerated_matrix_count": 1,
            "router_gate_accelerated_estimated_flops": 100,
        },
        required=True,
    )

    assert coverage["ok"] is False
    assert coverage["accelerated_router_gate_only"] is True
    assert coverage["allow_router_gate_only_acceleration"] is False
    assert coverage["non_router_matrix_count"] == 0
    assert coverage["non_router_unaccelerated_estimated_flops"] == 0
    assert "only from MoE router gates" in coverage["reason"]

    allowed = _prefill_acceleration_coverage_summary(
        dict(coverage, analyzed=True),
        required=True,
        allow_router_gate_only_acceleration=True,
    )

    assert allowed["ok"] is True
    assert allowed["accelerated_router_gate_only"] is True
    assert allowed["allow_router_gate_only_acceleration"] is True


def test_launch_audit_prefill_acceleration_evidence_allows_explicit_non_accelerated() -> None:
    audit = {
        "checks": [
            {
                "code": "prefill_acceleration_gate_ok",
                "ok": True,
                "required": False,
                "allow_non_accelerated_prefill_launch_audit": True,
            }
        ]
    }

    _require_launch_audit_prefill_acceleration_evidence(audit)


def test_launch_audit_prefill_acceleration_evidence_rejects_implicit_non_accelerated() -> None:
    audit = {
        "checks": [
            {
                "code": "prefill_acceleration_gate_ok",
                "ok": True,
                "required": False,
            }
        ]
    }

    with pytest.raises(
        Exception,
        match="without allow_non_accelerated_prefill_launch_audit=true",
    ):
        _require_launch_audit_prefill_acceleration_evidence(audit)


def _write_checkpoint_header_manifest(root: Path) -> None:
    shard = root / "model-00001-of-00001.safetensors"
    raw = shard.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    data_start = 8 + header_len
    header = json.loads(raw[8:data_start])
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    (root / HEADER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "index": index,
                "shards": {
                    shard.name: {
                        "file_size": len(raw),
                        "data_start": data_start,
                        "header": header,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_prepared_manifest(root: Path) -> Path:
    prepared = root / "prepared"
    model = root / "model"
    experts = prepared / "experts"
    resident = prepared / "resident"
    experts.mkdir(parents=True)
    resident.mkdir()
    model.mkdir()
    write_config(model / "config.json")
    (experts / "layer_000.bin").write_bytes(b"\0" * 16)
    expert_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "quantization": "mlx-affine-int4",
        "group_size": 8,
        "num_layers": 1,
        "num_experts": 1,
        "component_order": ["gate_proj.weight"],
        "layers": [
            {
                "layer": 0,
                "num_experts": 1,
                "expert_slot_bytes": 16,
                "layer_file": "layer_000.bin",
                "components": [
                    {
                        "name": "gate_proj.weight",
                        "offset": 0,
                        "size": 16,
                        "dtype": "U32",
                        "shape": [4, 1],
                    }
                ],
            }
        ],
    }
    (experts / "layout.json").write_text(json.dumps(expert_layout), encoding="utf-8")
    (resident / "resident.bin").write_bytes(b"\0" * 16)
    resident_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": None,
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": 16,
        "tensors": [
            {
                "name": "model.embed_tokens.weight",
                "offset": 0,
                "size": 16,
                "dtype": "F32",
                "shape": [1, 4],
                "category": "embedding",
            }
        ],
    }
    (resident / "layout.json").write_text(json.dumps(resident_layout), encoding="utf-8")
    cache_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "max_context_tokens": 4,
        "dtype": "BF16",
        "dtype_bytes": 2,
        "alignment": 64,
        "total_bytes": 32,
        "segments": [
            {
                "kind": "mla_kv",
                "layer": 0,
                "offset": 0,
                "width": 4,
                "dtype": "BF16",
                "dtype_bytes": 2,
                "max_context_tokens": 4,
            }
        ],
    }
    (prepared / "decode_cache_layout.json").write_text(
        json.dumps(cache_layout),
        encoding="utf-8",
    )
    (prepared / "decode_cache.bin").write_bytes(b"\0" * 32)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_dir": str(model),
                "experts_layout": "experts/layout.json",
                "resident_layout": "resident/layout.json",
                "decode_cache_layout": "decode_cache_layout.json",
                "decode_cache_file": "decode_cache.bin",
                "max_context_tokens": 4,
            }
        ),
        encoding="utf-8",
    )
    return prepared


def _add_prepared_memory_profile(
    prepared: Path,
    *,
    effective_unified_memory_gib: int = 64,
    system_reserve_gib: int = 16,
    live_working_set_gib: int = 4,
    min_free_unified_memory_gib: int = 16,
) -> None:
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_effective_unified_memory_bytes"] = (
        effective_unified_memory_gib * 1024**3
    )
    payload["prepare_system_reserve_bytes"] = system_reserve_gib * 1024**3
    payload["recommended_max_live_working_set_bytes"] = (
        live_working_set_gib * 1024**3
    )
    payload["recommended_min_free_unified_memory_bytes"] = (
        min_free_unified_memory_gib * 1024**3
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _add_prepared_cold_read_profile(
    prepared: Path,
    *,
    gib_per_second: float = 8.0,
) -> None:
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_cold_read_gib_per_second"] = gib_per_second
    payload["prepare_cold_read_benchmark_path"] = "experts/layer_000.bin"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_sequential_read_benchmark(gib_per_second: float):
    calls: list[dict[str, object]] = []

    def benchmark(
        path: str | Path,
        *,
        bytes_to_read: int | None = None,
        chunk_bytes: int = 8 * 1024**2,
        offset_bytes: int = 0,
        max_chunk_bytes: int | None = 512 * 1024**2,
    ) -> SimpleNamespace:
        measured = int(bytes_to_read or 1024)
        calls.append(
            {
                "path": Path(path),
                "bytes_to_read": bytes_to_read,
                "chunk_bytes": chunk_bytes,
                "offset_bytes": offset_bytes,
                "max_chunk_bytes": max_chunk_bytes,
            }
        )
        return SimpleNamespace(
            path=Path(path),
            file_size_bytes=max(measured, 1),
            offset_bytes=offset_bytes,
            requested_bytes=measured,
            measured_bytes=measured,
            chunk_bytes=chunk_bytes,
            elapsed_seconds=1.0,
            gib_per_second=gib_per_second,
            short_read=False,
        )

    benchmark.calls = calls  # type: ignore[attr-defined]
    return benchmark


def _mock_safe_system_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )


def _set_prepared_expert_layout_quantization(
    prepared: Path,
    quantization: str,
    *,
    group_size: int = 8,
) -> None:
    layout_path = prepared / "experts" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["quantization"] = quantization
    layout["group_size"] = group_size
    layout_path.write_text(json.dumps(layout), encoding="utf-8")


def _write_runtime_guard_prepared_manifest(root: Path) -> Path:
    prepared = root / "prepared-runtime-guard"
    model = root / "model-runtime-guard"
    prepared.mkdir()
    model.mkdir()
    write_config(model / "config.json", mixed_layers=True)
    expert_layout, resident_layout, cache_layout, cache_file = write_fixture(prepared)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model_dir": str(model),
                "experts_layout": str(expert_layout.relative_to(prepared)),
                "resident_layout": str(resident_layout.relative_to(prepared)),
                "decode_cache_layout": str(cache_layout.relative_to(prepared)),
                "decode_cache_file": str(cache_file.relative_to(prepared)),
                "max_context_tokens": 4,
            }
        ),
        encoding="utf-8",
    )
    return prepared


def _write_matching_launch_profile(
    profile_path: Path,
    prepared: Path,
    *,
    argv: list[str],
    source: str = "unit",
    sections: dict[str, object] | None = None,
) -> Path:
    _make_prepared_identity_strong(prepared)
    manifest = load_prepared_manifest(prepared)
    profile_target = prepared_launch_profile_target(manifest)
    payload: dict[str, object] = {
        "source": source,
        "argv_safe_to_replay": True,
        "prepared": profile_target,
        "argv": argv,
    }
    if sections is not None:
        payload["sections"] = sections
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    return profile_path


def _launch_audit_request_guard_argv(
    *,
    prompt_token_count: int = 2,
    max_new_tokens: int = 1,
) -> list[str]:
    argv: list[str] = []
    if prompt_token_count > 1:
        planned_read_bytes = prompt_token_count * 1024
        planned_read_seconds = planned_read_bytes / (16.0 * 1024**3)
        argv.extend(
            [
                "--prefill-prompt-chunk-tokens",
                str(prompt_token_count),
                "--prefill-max-routed-read-amplification",
                "1.05",
                "--prefill-max-routed-read-gib",
                format_routed_read_guard_flag_float(
                    planned_read_bytes / 1024**3 * 1.05
                ),
                "--prefill-ssd-read-gib-s",
                "16",
                "--prefill-max-routed-read-seconds",
                format_routed_read_guard_flag_float(planned_read_seconds * 1.05),
                "--prefill-max-stage-mib",
                format_routed_read_guard_flag_float(4096 / 1024**2 * 1.05),
                "--prefill-max-compact-stage-mib",
                format_routed_read_guard_flag_float(2048 / 1024**2 * 1.05),
                "--prefill-max-stage-raw-ranges",
                "2",
                "--prefill-max-stage-coalesced-ranges",
                "2",
                "--prefill-static-capacity-per-expert",
                "auto",
            ]
        )
    if max_new_tokens > 0:
        decode_read_bytes = 512
        decode_read_seconds = decode_read_bytes / (16.0 * 1024**3)
        if "--prefill-ssd-read-gib-s" not in argv:
            argv.extend(["--prefill-ssd-read-gib-s", "16"])
        argv.extend(
            [
                "--decode-max-routed-read-gib-per-token",
                format_routed_read_guard_flag_float(
                    decode_read_bytes / 1024**3 * 1.05
                ),
                "--decode-max-routed-read-seconds-per-token",
                format_routed_read_guard_flag_float(decode_read_seconds * 1.05),
            ]
        )
    return argv


def _launch_audit_request_guard_argv_with_auto_prompt_chunk(
    *,
    prompt_token_count: int = 2,
    max_new_tokens: int = 1,
) -> list[str]:
    argv = _launch_audit_request_guard_argv(
        prompt_token_count=prompt_token_count,
        max_new_tokens=max_new_tokens,
    )
    chunk_index = argv.index("--prefill-prompt-chunk-tokens")
    argv[chunk_index + 1] = "auto"
    return argv


def _replace_launch_arg_values(
    argv: list[str],
    replacements: dict[str, str],
) -> list[str]:
    replaced = list(argv)
    for flag, value in replacements.items():
        index = replaced.index(flag)
        replaced[index + 1] = value
    return replaced


def _launch_audit_prefill_actual_read_time(
    *,
    prompt_token_count: int = 2,
    cap_seconds: float = 5.0,
) -> dict[str, object]:
    planned_read_bytes = prompt_token_count * 1024
    planned_read_seconds = planned_read_bytes / (16.0 * 1024**3)
    return {
        "source": "benchmark_actual_prefill",
        "total_expert_stage_planned_read_bytes": planned_read_bytes,
        "total_expert_stage_planned_read_seconds": planned_read_seconds,
        "prefill_ssd_read_gib_per_second": 16.0,
        "prefill_max_routed_read_seconds": cap_seconds,
        "total_expert_stage_read_seconds_ok": True,
        "total_expert_stage_copy_seconds_ok": True,
        "prefill_max_stage_raw_ranges": 2,
        "prefill_max_stage_coalesced_ranges": 2,
        "total_expert_stage_raw_ranges": prompt_token_count,
        "total_expert_stage_coalesced_ranges": prompt_token_count,
        "max_expert_stage_raw_ranges": 1,
        "max_expert_stage_coalesced_ranges": 1,
        "total_expert_stage_raw_ranges_ok": True,
        "total_expert_stage_coalesced_ranges_ok": True,
    }


def _launch_audit_prefill_acceleration_coverage(
    *,
    required: bool = False,
    min_accelerated_flop_fraction: float = 0.0,
) -> dict[str, object]:
    return {
        "required": required,
        "analyzed": True,
        "ok": True,
        "min_accelerated_flop_fraction": min_accelerated_flop_fraction,
        "matrix_count": 2,
        "accelerated_matrix_count": 2,
        "mpsgraph_matrix_count": 2,
        "custom_metal_matrix_count": 0,
        "unsupported_mpsgraph_matrix_count": 0,
        "total_estimated_flops": 4096,
        "accelerated_estimated_flops": 4096,
        "custom_metal_estimated_flops": 0,
        "unsupported_mpsgraph_estimated_flops": 0,
        "other_estimated_flops": 0,
        "mpp_candidate_policy": {
            "candidate_backend": "mpp_tensor_ops_prefill",
            "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
            "mpp_tensor_ops_min_batch_tokens": 128,
            "mpp_tensor_ops_min_matrix_dim": 32,
            "selectable_prefill_backend": False,
        },
        "mpp_tensor_ops_candidate_matrix_count": 0,
        "mpp_tensor_ops_candidate_estimated_flops": 0,
        "mpp_tensor_ops_candidate_flop_fraction": 0.0,
        "streamed_routed_expert_layer_count": 0,
        "streamed_routed_expert_matrix_count": 0,
        "streamed_routed_expert_assignments": 0,
        "streamed_routed_expert_estimated_flops": 0,
        "streamed_routed_expert_mpp_candidate_matrix_count": 0,
        "streamed_routed_expert_mpp_candidate_estimated_flops": 0,
        "accelerated_flop_fraction": 1.0,
        "dominant_resident_flops_accelerated": True,
        "accelerated_backends": ("mpsgraph-f32",),
        "any_resident_matrix_accelerated": True,
        "all_resident_matrices_accelerated": True,
        "reason": "",
    }


def _launch_audit_stage_temp_evidence() -> dict[str, object]:
    return {
        "analyzed": True,
        "prompt_chunk_tokens": 64,
        "top_k": 2,
        "chunks_per_prompt": 1,
        "layers": 1,
        "stage_align_bytes": 4096,
        "static_capacity_per_expert": "auto",
        "allow_static_capacity_overflow": False,
        "max_static_capacity_per_expert": 64,
        "static_capacity_strict_overflow_safe": True,
        "max_stage_bytes": 4096,
        "max_stage_limit_bytes": 8192,
        "within_stage_limit": True,
        "max_compact_stage_bytes": 32,
        "max_compact_stage_limit_bytes": 1024,
        "within_compact_stage_limit": True,
        "max_stage_raw_ranges": 1,
        "max_stage_raw_range_limit": 2,
        "within_stage_raw_range_limit": True,
        "max_stage_coalesced_ranges": 1,
        "max_stage_coalesced_range_limit": 2,
        "within_stage_coalesced_range_limit": True,
        "max_stage_plus_compact_bytes": 4128,
        "total_stage_plus_compact_bytes": 4128,
        "max_static_capacity_binary_bytes": 812,
        "total_static_capacity_binary_bytes": 812,
        "max_stage_plus_compact_plus_static_bytes": 4940,
        "total_stage_plus_compact_plus_static_bytes": 4940,
        "within_limit": True,
    }


def _launch_audit_prefill_prompt_chunk_plan(
    *,
    prompt_token_count: int,
) -> dict[str, object]:
    def summary() -> dict[str, object]:
        cap = {
            "name": "prompt_tokens",
            "tokens": prompt_token_count,
            "bytes_available": None,
            "bytes_per_token": None,
            "detail": None,
        }
        runner_cap = {
            "name": "runner_scratch_bytes",
            "tokens": prompt_token_count,
            "bytes_available": 4 * 1024 * 1024,
            "bytes_per_token": 128,
            "detail": "includes resident matrix scratch",
        }
        return {
            "prompt_tokens": prompt_token_count,
            "start_position": 0,
            "raw_tokens": prompt_token_count,
            "chunk_tokens": prompt_token_count,
            "tile_tokens": 1,
            "limiting_cap_tokens": prompt_token_count,
            "limiting_caps": [cap],
            "limiting_cap_names": ["prompt_tokens"],
            "caps": [cap, runner_cap],
            "hidden_dim": 16,
            "per_token_activation_bytes": 128,
            "max_matrix_scratch_bytes": 2 * 1024 * 1024,
            "next_token_matrix_scratch_bytes": None,
            "usable_disk_bytes": 1024**3,
            "per_token_disk_bytes": 4096,
        }

    return {
        "source": "prepared_request_check",
        "configured_is_auto": True,
        "auto": summary(),
        "max_safe": summary(),
    }


def _launch_audit_prefill_actual_linear_backend() -> dict[str, object]:
    return {
        "source": "benchmark_actual_prefill",
        "configured_backend": "auto",
        "auto_policy": {
            "mpsgraph_min_batch_tokens": 64,
            "mpsgraph_min_matrix_dim": 16,
        },
        "linear_backend_counts": {"mpsgraph-f32": 2},
        "linear_backend_flops": {"mpsgraph-f32": 4096},
        "linear_backend_elapsed_seconds": {"mpsgraph-f32": 0.002},
        "linear_backend_estimated_tflops": {
            "mpsgraph-f32": 4096 / 0.002 / 1e12
        },
        "total_linear_estimated_flops": 4096,
        "accelerated_linear_estimated_flops": 4096,
        "custom_linear_estimated_flops": 0,
        "unsupported_linear_estimated_flops": 0,
        "accelerated_linear_flop_fraction": 1.0,
    }


def _launch_audit_decode_actual_read_time(
    *,
    decode_step_count: int = 1,
    read_bytes_per_token: int = 512,
) -> dict[str, object]:
    ssd = 16.0
    planned_bytes = decode_step_count * read_bytes_per_token
    seconds = planned_bytes / (ssd * 1024**3)
    max_seconds_per_token = read_bytes_per_token / (ssd * 1024**3) * 1.05
    return {
        "source": "benchmark_actual_decode",
        "decode_step_count": decode_step_count,
        "decode_read_bytes_per_token": read_bytes_per_token,
        "planned_decode_routed_read_bytes": planned_bytes,
        "actual_decode_routed_read_bytes": planned_bytes,
        "actual_decode_routed_read_bytes_ok": True,
        "planned_decode_routed_read_seconds": seconds,
        "actual_decode_routed_read_seconds": seconds,
        "prefill_ssd_read_gib_per_second": ssd,
        "decode_max_routed_read_seconds_per_token": max_seconds_per_token,
        "total_decode_max_routed_read_seconds": (
            max_seconds_per_token * decode_step_count
        ),
        "total_decode_routed_read_seconds_ok": True,
    }


def _write_launch_audit_artifact(
    audit_path: Path,
    *,
    profile: Path,
    ok: bool = True,
    prompt_token_count: int = 2,
    max_new_tokens: int = 1,
    request_profile_argv: list[str] | None = None,
) -> Path:
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    prepared_for_audit = None
    try:
        prepared_manifest = profile_payload["prepared"]["prepared_manifest"]
        prepared_for_audit = load_prepared_manifest(prepared_manifest)
        readiness = prepared_glm_4bit_readiness(prepared_for_audit)
    except Exception:
        readiness = None
    batch_prefill_prompt = prompt_token_count > 1
    request_check: dict[str, object] = {
        "ok": ok,
        "prompt_token_count": prompt_token_count,
        "max_new_tokens": max_new_tokens,
        "required_context_tokens": prompt_token_count + max_new_tokens,
        "batch_prefill_prompt": batch_prefill_prompt,
        "runtime_preflight": {
            "ran": True,
            "required_for_decode_routed_read_guard": False,
            "requested_context_tokens": prompt_token_count + max_new_tokens,
            "layers": [0],
            "dense_layers": [],
            "max_layer_peak_bytes": 4096,
            "max_layer_cache_read_bytes": 1024,
            "read_bytes_per_token": 512,
            "final_logits_peak_bytes": 128,
            "embedding_row_bytes": 32,
            "embedding_output_bytes": 32,
            "live_working_set_bytes": 4096,
            "resident_backing_bytes": 1024,
            "nonresident_peak_bytes": 3072,
            "extra_live_working_set_bytes": 0,
            "max_live_working_set_bytes": 8192,
            "min_available_memory_bytes": 1024,
            "required_available_memory_bytes": 5120,
            "system_available_memory_bytes": 1024**3,
            "system_total_memory_bytes": 2 * 1024**3,
            "system_memory_source": "unit",
            "available_memory_ok": True,
        },
    }
    if batch_prefill_prompt:
        request_check["prefill_prompt_chunk_tokens"] = {
            "configured": 0,
            "resolved": prompt_token_count,
            "max_safe": prompt_token_count,
        }
        request_check["prefill_prompt_chunk_plan"] = (
            _launch_audit_prefill_prompt_chunk_plan(
                prompt_token_count=prompt_token_count,
            )
        )
        request_check["runtime_preflight"]["prefill_live_memory"] = {
            "prompt_batch_bytes": 1024,
            "runner_scratch_bytes": 1024,
            "cache_read_bytes": 512,
            "cache_write_bytes": 512,
            "stage_copy_bytes": 256,
            "estimated_live_working_set_bytes": 2048,
        }
        request_check["prefill_cache_io"] = {
            "dtype_bytes": 2,
            "mla_cache_width": 4,
            "dsa_index_head_dim": None,
            "dsa_index_topk": None,
            "indexed_attention_layers": 0,
            "full_attention_layers": 1,
            "dsa_full_indexer_layers": 0,
            "causal_rows_per_layer": 3,
            "indexed_rows_per_layer": 0,
            "mla_cache_read_bytes": 24,
            "dsa_index_cache_read_bytes": 0,
            "total_cache_read_bytes": 24,
            "mla_cache_write_bytes": 16,
            "dsa_index_cache_write_bytes": 0,
            "total_cache_write_bytes": 16,
        }
        baseline_read_bytes = prompt_token_count * 1024
        planned_read_bytes = baseline_read_bytes
        ssd_read_gib_per_second = 16.0
        max_read_seconds = 5.0
        planned_read_seconds = planned_read_bytes / (
            ssd_read_gib_per_second * 1024**3
        )
        request_check["prefill_routed_expert_read"] = {
            "analyzed": True,
            "prompt_chunk_tokens": prompt_token_count,
            "top_k": 1,
            "layers": [0],
            "chunks_per_prompt": 1,
            "baseline_read_bytes": baseline_read_bytes,
            "planned_read_bytes": planned_read_bytes,
            "extra_read_bytes": 0,
            "baseline_read_seconds": planned_read_seconds,
            "planned_read_seconds": planned_read_seconds,
            "extra_read_seconds": 0.0,
            "seconds_limit_planned_read_bytes": int(
                max_read_seconds * ssd_read_gib_per_second * 1024**3
            ),
            "effective_planned_read_limit_bytes": planned_read_bytes * 2,
            "minimum_chunk_tokens_for_limits": prompt_token_count,
            "read_amplification": 1.0,
            "max_layer_baseline_read_bytes": baseline_read_bytes,
            "max_layer_planned_read_bytes": planned_read_bytes,
            "max_read_amplification": 2.0,
            "within_amplification_limit": True,
            "max_planned_read_bytes": planned_read_bytes * 2,
            "within_planned_read_limit": True,
            "ssd_read_gib_per_second": ssd_read_gib_per_second,
            "max_read_seconds": max_read_seconds,
            "within_seconds_limit": True,
            "within_limit": True,
        }
        request_check["prefill_routed_chunk_frontier"] = {
            "analyzed": True,
            "prompt_token_count": prompt_token_count,
            "resolved_prompt_chunk_tokens": prompt_token_count,
            "max_safe_prompt_chunk_tokens": prompt_token_count,
            "top_k": 1,
            "layers": 1,
            "stage_align_bytes": 4096,
            "static_capacity_per_expert": "auto",
            "allow_static_capacity_overflow": False,
            "baseline_read_bytes": baseline_read_bytes,
            "saturation_chunk_tokens": 1,
            "candidates": [
                {
                    "prompt_chunk_tokens": 1,
                    "chunks_per_prompt": prompt_token_count,
                    "saturates_all_experts_per_layer": True,
                    "planned_read_bytes": baseline_read_bytes * prompt_token_count,
                    "extra_read_bytes": baseline_read_bytes * (prompt_token_count - 1),
                    "read_amplification": float(prompt_token_count),
                    "max_layer_planned_read_bytes": baseline_read_bytes * prompt_token_count,
                    "max_stage_plus_compact_bytes": 4096,
                    "max_chunk_stage_plus_compact_bytes": 4096,
                    "total_stage_plus_compact_bytes": 4096 * prompt_token_count,
                    "max_static_capacity_binary_bytes": 64,
                    "max_chunk_static_capacity_binary_bytes": 64,
                    "total_static_capacity_binary_bytes": 64 * prompt_token_count,
                    "max_stage_plus_compact_plus_static_bytes": 4160,
                    "max_chunk_stage_plus_compact_plus_static_bytes": 4160,
                    "total_stage_plus_compact_plus_static_bytes": (
                        4160 * prompt_token_count
                    ),
                    "planned_read_seconds": (
                        baseline_read_bytes
                        * prompt_token_count
                        / (ssd_read_gib_per_second * 1024**3)
                    ),
                },
                {
                    "prompt_chunk_tokens": prompt_token_count,
                    "chunks_per_prompt": 1,
                    "saturates_all_experts_per_layer": True,
                    "planned_read_bytes": baseline_read_bytes,
                    "extra_read_bytes": 0,
                    "read_amplification": 1.0,
                    "max_layer_planned_read_bytes": baseline_read_bytes,
                    "max_stage_plus_compact_bytes": 6144,
                    "max_chunk_stage_plus_compact_bytes": 6144,
                    "total_stage_plus_compact_bytes": 6144,
                    "max_static_capacity_binary_bytes": 128,
                    "max_chunk_static_capacity_binary_bytes": 128,
                    "total_static_capacity_binary_bytes": 128,
                    "max_stage_plus_compact_plus_static_bytes": 6272,
                    "max_chunk_stage_plus_compact_plus_static_bytes": 6272,
                    "total_stage_plus_compact_plus_static_bytes": 6272,
                    "planned_read_seconds": planned_read_seconds,
                },
            ],
        }
        request_check["prefill_routed_stage_temp_disk"] = {
            "analyzed": True,
            "prompt_chunk_tokens": prompt_token_count,
            "top_k": 1,
            "chunks_per_prompt": 1,
            "layers": 1,
            "stage_align_bytes": 4096,
            "static_capacity_per_expert": "auto",
            "allow_static_capacity_overflow": False,
            "max_static_capacity_per_expert": prompt_token_count,
            "static_capacity_strict_overflow_safe": True,
            "max_stage_limit_bytes": 8192,
            "max_compact_stage_limit_bytes": 4096,
            "max_stage_raw_ranges": 1,
            "max_stage_raw_range_limit": 2,
            "within_stage_raw_range_limit": True,
            "max_stage_coalesced_ranges": 1,
            "max_stage_coalesced_range_limit": 2,
            "within_stage_coalesced_range_limit": True,
            "max_stage_bytes": 4096,
            "max_compact_stage_bytes": 2048,
            "max_stage_plus_compact_bytes": 6144,
            "total_stage_plus_compact_bytes": 6144,
            "max_static_capacity_binary_bytes": 128,
            "total_static_capacity_binary_bytes": 128,
            "max_stage_plus_compact_plus_static_bytes": 6272,
            "total_stage_plus_compact_plus_static_bytes": 6272,
            "within_stage_limit": True,
            "within_compact_stage_limit": True,
            "within_limit": True,
        }
        request_check["prefill_stage_temp_disk_free"] = {
            "analyzed": True,
            "path": "/private/tmp",
            "required_stage_temp_bytes": 6272,
            "disk_safety_margin_bytes": 0,
            "required_free_bytes": 6272,
            "free_bytes": 1024**3,
            "within_free_space": True,
        }
        request_check["prefill_linear_backend"] = {
            "source": "prepared_request_check",
            "configured": "auto",
            "effective": "auto",
            "analyzed": True,
            "prompt_chunk_tokens": prompt_token_count,
        }
        request_check["prefill_acceleration_coverage"] = (
            _launch_audit_prefill_acceleration_coverage()
        )
    if max_new_tokens > 0:
        decode_read_bytes = 512
        decode_ssd_read_gib_per_second = 16.0
        decode_planned_seconds = decode_read_bytes / (
            decode_ssd_read_gib_per_second * 1024**3
        )
        request_check["decode_routed_expert_read"] = {
            "analyzed": True,
            "read_bytes_per_token": decode_read_bytes,
            "max_read_bytes_per_token": decode_read_bytes * 2,
            "within_read_limit": True,
            "ssd_read_gib_per_second": decode_ssd_read_gib_per_second,
            "planned_read_seconds_per_token": decode_planned_seconds,
            "max_read_seconds_per_token": 5.0,
            "within_seconds_limit": True,
            "within_limit": True,
        }
    payload = {
        "schema": "largerlm.launch_audit.v1",
        "source": "unit",
        "prepared": profile_payload["prepared"],
        "applied_launch_profile": {
            "path": str(profile),
            "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
            "source": profile_payload.get("source"),
            "locked": True,
            "profile_flag_count": 1,
            "lock_checked_flags": ["--min-free-unified-memory-gib"],
            "section_names": tuple(
                sorted(
                    str(name)
                    for name in (
                        profile_payload.get("sections")
                        if isinstance(profile_payload.get("sections"), dict)
                        else {}
                    )
                )
            ),
        },
        "launch_audit": {
            "ok": ok,
            "failures": [] if ok else ["unit_failure"],
            "checks": [
                {
                    "code": code,
                    "ok": ok,
                    "message": "unit launch audit check",
                }
                for code in _REQUIRED_LAUNCH_AUDIT_CHECK_CODES
            ],
        },
        "request_check": request_check,
    }
    profile_sections = profile_payload.get("sections")
    if isinstance(profile_sections, dict) and isinstance(
        profile_sections.get("prefill_actual_read_time"),
        dict,
    ):
        payload["applied_launch_profile"]["prefill_actual_read_time"] = (
            profile_sections["prefill_actual_read_time"]
        )
    if isinstance(profile_sections, dict) and isinstance(
        profile_sections.get("prefill_actual_acceleration_coverage"),
        dict,
    ):
        payload["applied_launch_profile"][
            "prefill_actual_acceleration_coverage"
        ] = profile_sections["prefill_actual_acceleration_coverage"]
    if isinstance(profile_sections, dict) and isinstance(
        profile_sections.get("prefill_actual_acceleration_frontier"),
        dict,
    ):
        payload["applied_launch_profile"][
            "prefill_actual_acceleration_frontier"
        ] = profile_sections["prefill_actual_acceleration_frontier"]
    if isinstance(profile_sections, dict) and isinstance(
        profile_sections.get("prefill_actual_linear_backend"),
        dict,
    ):
        payload["applied_launch_profile"]["prefill_actual_linear_backend"] = (
            profile_sections["prefill_actual_linear_backend"]
        )
    if isinstance(profile_sections, dict) and isinstance(
        profile_sections.get("decode_actual_read_time"),
        dict,
    ):
        payload["applied_launch_profile"]["decode_actual_read_time"] = (
            profile_sections["decode_actual_read_time"]
        )
    if isinstance(readiness, dict) and readiness.get("ok") is True:
        glm_check = next(
            check
            for check in payload["launch_audit"]["checks"]
            if check["code"] == "glm_4bit_ready"
        )
        glm_check.update(_glm_4bit_readiness_audit_details(readiness))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    acceleration_check.update(
        {
            "reason_code": "ok",
            "prefill_backend_configured_backend": "auto",
            "prefill_backend_effective_backend": "auto",
            "recommended_backend": "mpsgraph_prefill_fallback",
            "host_probe_requested": True,
            "host_probe_path": str(
                Path.cwd() / "metal" / "prefill-backend-probe"
            ),
            "host_probe_ran": True,
            "host_probe_ok": True,
            "prefill_backend_probe_timeout_seconds": 5.0,
            "mps_graph_runtime_available": True,
            "mps_graph_probe_requested": True,
            "mps_graph_probe_ran": True,
            "mps_graph_probe_ok": True,
            "metal4_ml_runtime_available": False,
            "mpp_runtime_available": False,
            "prefill_acceleration_runtimes": ("mpsgraph-f32",),
            "selectable_accelerated_prefill_backends": ("mpsgraph-f32",),
            "validated_accelerated_prefill_backends": ("mpsgraph-f32",),
            "prefill_acceleration_runtime_gaps": (),
            "prefill_neural_accelerator_status": {
                "runtime": "mpp_tensor_ops_prefill",
                "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
                "status": "unavailable",
                "ready_for_generation": False,
                "runtime_visible": False,
                "selectable": False,
                "reason": "MPP tensor ops prefill runtime is not available",
            },
            "selectable_prefill_acceleration_available": True,
            "validated_prefill_acceleration_available": True,
        }
    )
    runtime_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_runtime_memory_ok"
    )
    runtime = request_check["runtime_preflight"]
    runtime_check.update(
        {
            field: runtime[field]
            for field in (
                "available_memory_ok",
                "requested_context_tokens",
                "max_layer_peak_bytes",
                "max_layer_cache_read_bytes",
                "read_bytes_per_token",
                "final_logits_peak_bytes",
                "embedding_row_bytes",
                "embedding_output_bytes",
                "live_working_set_bytes",
                "resident_backing_bytes",
                "nonresident_peak_bytes",
                "extra_live_working_set_bytes",
                "max_live_working_set_bytes",
                "min_available_memory_bytes",
                "required_available_memory_bytes",
                "system_available_memory_bytes",
                "system_total_memory_bytes",
                "system_memory_source",
            )
        }
    )
    if runtime.get("prefill_live_memory") is not None:
        runtime_check["prefill_live_memory"] = runtime["prefill_live_memory"]
    prepared_runtime_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prepared_runtime_profile_ok"
    )
    if prepared_for_audit is not None:
        prepared_runtime_check.update(
            {
                key: value
                for key, value in _prepared_runtime_profile_audit_details_from_prepared(
                    prepared_for_audit
                ).items()
                if value is not None
            }
        )
    profile_sections = profile_payload.get("sections")
    ssd_flags = (
        profile_sections.get("prepared_ssd_read_flags")
        if isinstance(profile_sections, dict)
        else None
    )
    ssd_check = {
        "code": "prepared_ssd_read_profile_valid",
        "ok": ok,
        "message": "unit launch audit optional check",
    }
    ssd_details = _prepared_ssd_read_audit_details(
        ssd_flags,
        _prepared_ssd_read_audit_details_from_prepared(prepared_for_audit)
        if prepared_for_audit is not None
        else None,
    )
    ssd_check.update(
        {
            key: value
            for key, value in ssd_details.items()
            if key != "ok" and value is not None
        }
    )
    ssd_check["ok"] = ssd_details.get("ok") is True and ok
    payload["launch_audit"]["checks"].append(ssd_check)
    pack_heap_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prepare_expert_pack_heap_envelope_ok"
    )
    pack_heap = _prepare_expert_pack_heap_audit_details(
        _prepare_expert_pack_heap_source_from_prepared(prepared_for_audit)
        if prepared_for_audit is not None
        else None
    )
    pack_heap_check.update(
        {
            key: value
            for key, value in pack_heap.items()
            if key != "ok" and value is not None
        }
    )
    resident_rewrite_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prepare_resident_alias_rewrite_ok"
    )
    resident_rewrite = _prepare_resident_alias_rewrite_audit_details(
        _prepare_resident_alias_rewrite_source_from_prepared(prepared_for_audit)
        if prepared_for_audit is not None
        else None
    )
    resident_rewrite_check.update(
        {
            key: value
            for key, value in resident_rewrite.items()
            if key != "ok" and value is not None
        }
    )
    cache_io_check = next(
        (
            check
            for check in payload["launch_audit"]["checks"]
            if check["code"] == "request_prefill_cache_io_valid"
        ),
        None,
    )
    if cache_io_check is None:
        cache_io_check = {
            "code": "request_prefill_cache_io_valid",
            "ok": ok,
            "message": "unit launch audit optional check",
        }
        payload["launch_audit"]["checks"].append(cache_io_check)
    cache_io = request_check.get("prefill_cache_io")
    cache_io_check["required"] = False
    cache_io_check["evidence_present"] = isinstance(cache_io, dict)
    if isinstance(cache_io, dict):
        cache_io_check.update(
            {field: cache_io[field] for field in _REQUEST_PREFILL_CACHE_IO_AUDIT_FIELDS}
        )
    frontier_check = next(
        (
            check
            for check in payload["launch_audit"]["checks"]
            if check["code"] == "request_prefill_routed_chunk_frontier_valid"
        ),
        None,
    )
    if frontier_check is None:
        frontier_check = {
            "code": "request_prefill_routed_chunk_frontier_valid",
            "ok": ok,
            "message": "unit launch audit optional check",
        }
        payload["launch_audit"]["checks"].append(frontier_check)
    frontier = request_check.get("prefill_routed_chunk_frontier")
    frontier_check["required"] = False
    frontier_check["evidence_present"] = isinstance(frontier, dict)
    if isinstance(frontier, dict):
        frontier_check.update(
            {
                field: frontier[field]
                for field in _REQUEST_PREFILL_ROUTED_CHUNK_FRONTIER_AUDIT_FIELDS
            }
        )
    chunk_plan_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_prefill_prompt_chunk_plan_ok"
    )
    chunk_plan = request_check.get("prefill_prompt_chunk_plan")
    chunk_tokens = request_check.get("prefill_prompt_chunk_tokens")
    chunk_plan_check["required"] = batch_prefill_prompt
    chunk_plan_check["evidence_present"] = isinstance(chunk_plan, dict)
    chunk_plan_check["request_profile_evidence_present"] = isinstance(
        chunk_plan,
        dict,
    )
    chunk_plan_check["request_profile_plan_matches"] = isinstance(chunk_plan, dict)
    if isinstance(chunk_plan, dict):
        chunk_plan_check["prefill_prompt_chunk_plan"] = chunk_plan
    if isinstance(chunk_tokens, dict):
        chunk_plan_check["prefill_prompt_chunk_tokens"] = chunk_tokens
    routed_budget_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_prefill_routed_read_budget_ok"
    )
    routed_read = request_check.get("prefill_routed_expert_read")
    if isinstance(routed_read, dict):
        routed_budget_check["required"] = True
        routed_budget_check.update(
            {
                field: routed_read[field]
                for field in _REQUEST_ROUTED_READ_LAUNCH_CHECK_FIELDS
            }
        )
    else:
        routed_budget_check["required"] = False
    decode_budget_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_decode_routed_read_budget_ok"
    )
    decode_read = request_check.get("decode_routed_expert_read")
    if isinstance(decode_read, dict):
        decode_budget_check["required"] = True
        decode_budget_check.update(
            {
                field: decode_read[field]
                for field in _REQUEST_DECODE_ROUTED_READ_AUDIT_FIELDS
            }
        )
    else:
        decode_budget_check["required"] = False
    stage_temp_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_prefill_stage_temp_limit_ok"
    )
    stage_temp = request_check.get("prefill_routed_stage_temp_disk")
    if isinstance(stage_temp, dict):
        stage_temp_check["required"] = True
        stage_temp_check.update(
            {
                field: stage_temp[field]
                for field in _REQUEST_PREFILL_STAGE_TEMP_AUDIT_FIELDS
            }
        )
    else:
        stage_temp_check["required"] = False
    stage_disk_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_prefill_stage_temp_disk_ok"
    )
    stage_disk = request_check.get("prefill_stage_temp_disk_free")
    if isinstance(stage_disk, dict):
        stage_disk_check["required"] = True
        stage_disk_check.update(
            {
                "within_free_space": stage_disk["within_free_space"],
                "required_free_bytes": stage_disk["required_free_bytes"],
                "free_bytes": stage_disk["free_bytes"],
                "path": stage_disk["path"],
            }
        )
    else:
        stage_disk_check["required"] = False
    backend_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_prefill_backend_effective_ok"
    )
    request_linear = request_check.get("prefill_linear_backend")
    if isinstance(request_linear, dict):
        backend_check["required"] = True
        backend_check.update(
            {
                "configured": request_linear["configured"],
                "effective": request_linear["effective"],
                "health_effective_backend": "auto",
                "analyzed": request_linear["analyzed"],
            }
        )
    else:
        backend_check["required"] = False
    coverage_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "request_prefill_acceleration_coverage_ok"
    )
    coverage = request_check.get("prefill_acceleration_coverage")
    coverage_check["evidence_present"] = isinstance(coverage, dict)
    if isinstance(coverage, dict):
        coverage_check.update(
            {
                field: coverage[field]
                for field in _REQUEST_PREFILL_ACCELERATION_COVERAGE_AUDIT_FIELDS
            }
        )
    actual_read_time_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "applied_prefill_actual_read_time_ok"
    )
    applied_actual = payload["applied_launch_profile"].get(
        "prefill_actual_read_time"
    )
    actual_required = (
        profile_payload.get("source") == "benchmark_actual"
        and prompt_token_count > 1
    )
    actual_read_time_check["required"] = actual_required
    actual_read_time_check["applied_profile_source"] = profile_payload.get("source")
    actual_read_time_check["evidence_present"] = isinstance(applied_actual, dict)
    if isinstance(applied_actual, dict):
        actual_read_time_check.update(applied_actual)
    actual_accel_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "applied_prefill_actual_acceleration_coverage_valid"
    )
    applied_actual_accel = payload["applied_launch_profile"].get(
        "prefill_actual_acceleration_coverage"
    )
    actual_accel_check["required"] = False
    actual_accel_check["applied_profile_source"] = profile_payload.get("source")
    actual_accel_check["evidence_present"] = isinstance(applied_actual_accel, dict)
    if isinstance(applied_actual_accel, dict):
        actual_accel_check.update(applied_actual_accel)
    decode_actual_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "applied_decode_actual_read_time_ok"
    )
    applied_decode_actual = payload["applied_launch_profile"].get(
        "decode_actual_read_time"
    )
    decode_actual_check["required"] = False
    decode_actual_check["applied_profile_source"] = profile_payload.get("source")
    decode_actual_check["evidence_present"] = isinstance(applied_decode_actual, dict)
    if isinstance(applied_decode_actual, dict):
        decode_actual_check.update(applied_decode_actual)
    request_profile: dict[str, object] = {
        "source": "prepared_request_check",
        "argv_safe_to_replay": True,
        "prepared": profile_payload["prepared"],
        "argv": (
            request_profile_argv
            if request_profile_argv is not None
            else [
                *profile_payload.get("argv", []),
                *_launch_audit_request_guard_argv(
                    prompt_token_count=prompt_token_count,
                    max_new_tokens=max_new_tokens,
                ),
            ]
        ),
    }
    request_profile_sections: dict[str, object] = {}
    chunk_plan = request_check.get("prefill_prompt_chunk_plan")
    if isinstance(chunk_plan, dict):
        request_profile_sections["prefill_prompt_chunk_plan"] = chunk_plan
    if request_profile_sections:
        request_profile["sections"] = request_profile_sections
    payload["request_launch_profile"] = request_profile
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    return audit_path


def _make_prepared_identity_strong(prepared: Path) -> None:
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = config_sha256(Path(str(manifest["model_dir"])))
    _set_prepared_layout_config_sha256(prepared, digest)
    expert_layout = json.loads((prepared / "experts" / "layout.json").read_text())
    manifest["expert_quantization"] = expert_layout["quantization"]
    manifest["expert_group_size"] = expert_layout["group_size"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _assert_printed_applied_launch_profile(output: str, profile: Path) -> None:
    assert f"  applied profile:       {profile}" in output
    assert (
        "  applied profile sha:   "
        f"{hashlib.sha256(profile.read_bytes()).hexdigest()}"
    ) in output
    assert "  applied profile src:   unit" in output
    assert "  applied profile match: True" in output


def _write_mxfp4_glm_config(path: Path) -> None:
    write_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        {
            "hidden_size": 32,
            "intermediate_size": 32,
            "moe_intermediate_size": 32,
            "kv_lora_rank": 32,
            "q_lora_rank": 32,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 8,
            "v_head_dim": 32,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _test_expert_dims(cfg, base: str) -> tuple[int, int]:
    hidden = int(cfg.hidden_size)
    moe_hidden = int(cfg.moe_hidden_size)
    if base in {"gate_proj", "up_proj"}:
        return moe_hidden, hidden
    if base == "down_proj":
        return hidden, moe_hidden
    raise AssertionError(f"unknown expert component base {base}")


def _test_expert_components(
    cfg,
    *,
    quantization: str,
    group_size: int,
) -> list[tuple[str, int, int, str, list[int]]]:
    if quantization in {"largerlm-affine-int4", "mlx-affine-int4"}:
        order = DEFAULT_EXPERT_COMPONENTS
    elif quantization == "mlx-mxfp4":
        order = MXFP4_EXPERT_COMPONENTS
    else:
        raise AssertionError(f"unsupported test expert quantization {quantization}")

    offset = 0
    components: list[tuple[str, int, int, str, list[int]]] = []
    for name in order:
        base, kind = name.rsplit(".", 1)
        out_dim, in_dim = _test_expert_dims(cfg, base)
        if kind == "weight":
            shape = [out_dim, in_dim // 8]
            dtype = "U32"
            size = out_dim * (in_dim // 8) * 4
        elif quantization == "mlx-mxfp4" and kind == "scales":
            shape = [out_dim, in_dim // group_size]
            dtype = "U8"
            size = out_dim * (in_dim // group_size)
        elif kind in {"scales", "biases"}:
            shape = [out_dim, in_dim // group_size]
            dtype = "BF16"
            size = out_dim * (in_dim // group_size) * 2
        else:
            raise AssertionError(f"unknown expert component {name}")
        components.append((name, offset, size, dtype, shape))
        offset += size
    return components


def _make_prepared_glm_4bit_ready(
    prepared: Path,
    *,
    expert_quantization: str = "largerlm-affine-int4",
    expert_group_size: int = 8,
) -> None:
    experts = prepared / "experts"
    resident = prepared / "resident"
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = load_config(Path(str(manifest["model_dir"])))
    digest = config_sha256(Path(str(manifest["model_dir"])))
    moe_layers = tuple(cfg.moe_layers)
    moe_layer_set = set(moe_layers)
    expert_components = _test_expert_components(
        cfg,
        quantization=expert_quantization,
        group_size=expert_group_size,
    )
    slot_bytes = sum(component[2] for component in expert_components)
    layers = []
    for layer_id in moe_layers:
        layer_file = experts / f"layer_{layer_id:03d}.bin"
        layer_file.write_bytes(b"\0" * (int(cfg.routed_experts) * slot_bytes))
        layers.append(
            {
                "layer": layer_id,
                "num_experts": int(cfg.routed_experts),
                "expert_slot_bytes": slot_bytes,
                "layer_file": layer_file.name,
                "components": [
                    {
                        "name": name,
                        "offset": offset,
                        "size": size,
                        "dtype": dtype,
                        "shape": shape,
                    }
                    for name, offset, size, dtype, shape in expert_components
                ],
            }
        )
    expert_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": digest,
        "quantization": expert_quantization,
        "group_size": expert_group_size,
        "num_layers": int(cfg.num_hidden_layers),
        "num_experts": int(cfg.routed_experts),
        "component_order": [component[0] for component in expert_components],
        "layers": layers,
    }
    (experts / "layout.json").write_text(
        json.dumps(expert_layout),
        encoding="utf-8",
    )
    manifest["expert_bytes"] = len(moe_layers) * int(cfg.routed_experts) * slot_bytes
    manifest["expert_quantization"] = expert_quantization
    manifest["expert_group_size"] = expert_group_size
    manifest["model_config_sha256"] = digest
    if expert_quantization == "largerlm-affine-int4":
        manifest["prepare_expert_pack_chunk_size_bytes"] = 1024 * 1024
        manifest["prepare_expert_pack_estimated_peak_heap_bytes"] = 64 * 1024**2
        manifest["prepare_expert_pack_max_heap_bytes"] = 512 * 1024**2
        manifest["prepare_raw_quantization_extra_heap_bytes"] = 64 * 1024
        manifest["prepare_raw_quantization_max_source_block_bytes"] = 128 * 1024
        manifest["prepare_raw_quantization_max_output_block_bytes"] = 64 * 1024
        manifest["prepare_raw_quantization_max_rows_per_block"] = 32
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    resident_tensors: list[dict[str, object]] = []
    offset = 0

    def add_resident_tensor(
        name: str,
        shape: list[int],
        *,
        category: str,
        dtype: str = "F32",
    ) -> None:
        nonlocal offset
        dtype_bytes = 4 if dtype == "F32" else 2
        size = dtype_bytes
        for dim in shape:
            size *= dim
        resident_tensors.append(
            {
                "name": name,
                "offset": offset,
                "size": size,
                "dtype": dtype,
                "shape": shape,
                "category": category,
            }
        )
        offset += size

    hidden = int(cfg.hidden_size)
    q_lora = int(
        cfg.q_lora_rank if cfg.q_lora_rank is not None else cfg.kv_lora_rank or 2
    )
    kv_lora = int(cfg.kv_lora_rank or 0)
    rope = int(cfg.qk_rope_head_dim or 0)
    nope = int(cfg.qk_nope_head_dim or 0)
    heads = int(cfg.num_attention_heads or 0)
    value = int(cfg.v_head_dim or 0)
    intermediate = int(cfg.intermediate_size or hidden)

    add_resident_tensor(
        "model.embed_tokens.weight",
        [4, hidden],
        category="embedding",
    )
    add_resident_tensor("model.norm.weight", [hidden], category="norms")
    dsa_full_layers = {
        layer_id
        for layer_id, indexer_type in enumerate(cfg.indexer_types or ())
        if layer_id < int(cfg.num_hidden_layers)
        and str(indexer_type).lower() == "full"
    }
    for layer_id in range(int(cfg.num_hidden_layers)):
        prefix = f"model.layers.{layer_id}"
        add_resident_tensor(
            f"{prefix}.input_layernorm.weight",
            [hidden],
            category="norms",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.q_a_proj.weight",
            [q_lora, hidden],
            category="attention",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.q_a_layernorm.weight",
            [q_lora],
            category="norms",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.q_b_proj.weight",
            [heads * (nope + rope), q_lora],
            category="attention",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            [kv_lora + rope, hidden],
            category="attention",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.kv_a_layernorm.weight",
            [kv_lora],
            category="norms",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.kv_b_proj.weight",
            [heads * (nope + value), kv_lora],
            category="attention",
        )
        add_resident_tensor(
            f"{prefix}.self_attn.o_proj.weight",
            [hidden, heads * value],
            category="attention",
        )
        add_resident_tensor(
            f"{prefix}.post_attention_layernorm.weight",
            [hidden],
            category="norms",
        )
        if layer_id in dsa_full_layers:
            index_head_dim = int(cfg.index_head_dim or 0)
            index_n_heads = int(cfg.index_n_heads or 0)
            q_lora = int(cfg.q_lora_rank or 0)
            add_resident_tensor(
                f"{prefix}.self_attn.indexer.wk.weight",
                [index_head_dim, hidden],
                category="attention",
            )
            add_resident_tensor(
                f"{prefix}.self_attn.indexer.wq_b.weight",
                [index_n_heads * index_head_dim, q_lora],
                category="attention",
            )
            add_resident_tensor(
                f"{prefix}.self_attn.indexer.weights_proj.weight",
                [index_n_heads, hidden],
                category="attention",
            )
            add_resident_tensor(
                f"{prefix}.self_attn.indexer.k_norm.weight",
                [index_head_dim],
                category="norms",
            )
            add_resident_tensor(
                f"{prefix}.self_attn.indexer.k_norm.bias",
                [index_head_dim],
                category="norms",
            )
        if layer_id in moe_layer_set:
            add_resident_tensor(
                f"{prefix}.mlp.gate.weight",
                [int(cfg.routed_experts), hidden],
                category="routers",
            )
        else:
            add_resident_tensor(
                f"{prefix}.mlp.gate_proj.weight",
                [intermediate, hidden],
                category="dense_mlp",
            )
            add_resident_tensor(
                f"{prefix}.mlp.up_proj.weight",
                [intermediate, hidden],
                category="dense_mlp",
            )
            add_resident_tensor(
                f"{prefix}.mlp.down_proj.weight",
                [hidden, intermediate],
                category="dense_mlp",
            )
    resident_layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "config_sha256": digest,
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": offset,
        "tensors": resident_tensors,
        "router": {
            key: value
            for key, value in {
                "scoring_func": cfg.scoring_func,
                "norm_topk_prob": cfg.norm_topk_prob,
                "routed_scaling_factor": cfg.routed_scaling_factor,
                "n_group": cfg.n_group,
                "topk_group": cfg.topk_group,
                "topk_method": cfg.topk_method,
                "num_experts_per_tok": cfg.num_experts_per_tok,
            }.items()
            if value is not None
        },
    }
    (resident / "layout.json").write_text(json.dumps(resident_layout), encoding="utf-8")
    (resident / "resident.bin").write_bytes(b"\0" * offset)
    cache_layout = build_decode_cache_layout(
        cfg,
        max_context_tokens=int(manifest.get("max_context_tokens") or 4),
    )
    (prepared / "decode_cache_layout.json").write_text(
        json.dumps(cache_layout.to_json()),
        encoding="utf-8",
    )
    (prepared / "decode_cache.bin").write_bytes(b"\0" * cache_layout.total_bytes)
    manifest["decode_cache_bytes"] = cache_layout.total_bytes
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _enable_prepared_full_dsa_indexer(prepared: Path) -> None:
    config_path = prepared.parent / "model" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "index_head_dim": 2,
            "index_n_heads": 1,
            "index_topk": 2,
            "q_lora_rank": 2,
            "indexer_types": ["full", "shared"],
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def _set_prepared_layout_config_sha256(prepared: Path, digest: str) -> None:
    for layout_path in (
        prepared / "experts" / "layout.json",
        prepared / "resident" / "layout.json",
    ):
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        payload["config_sha256"] = digest
        layout_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_simple_model_tokenizer(prepared: Path) -> Path:
    tokenizer_path = prepared.parent / "model" / "simple_tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "type": "largerlm-simple-vocab",
                "split": "characters",
                "tokens": {"A": 0, "B": 1, "C": 2, "<eos>": 3},
                "eos_token": "<eos>",
            }
        ),
        encoding="utf-8",
    )
    return tokenizer_path


def _add_large_bf16_prefill_matrix(prepared: Path) -> None:
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_bin_path = prepared / "resident" / "resident.bin"
    payload = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    offset = int(payload["total_bytes"])
    matrix_size = 32 * 32 * 2
    payload["tensors"].append(
        {
            "name": "model.layers.0.self_attn.q_a_proj.weight",
            "offset": offset,
            "size": matrix_size,
            "dtype": "BF16",
            "shape": [32, 32],
            "category": "attention",
        }
    )
    payload["total_bytes"] = offset + matrix_size
    resident_layout_path.write_text(json.dumps(payload), encoding="utf-8")
    resident_bin_path.write_bytes(resident_bin_path.read_bytes() + b"\0" * matrix_size)


def _mpsgraph_probe_ready_capability(**overrides: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "sdk_path": None,
        "recommended_backend": "mpsgraph_prefill_fallback",
        "host_probe_requested": True,
        "host_probe_path": str(Path.cwd() / "metal" / "prefill-backend-probe"),
        "host_probe_ran": True,
        "host_probe_ok": True,
        "host_probe_error": None,
        "probe_timeout_seconds": 5.0,
        "mps_graph_runtime_available": True,
        "mps_graph_matmul_declared": True,
        "mps_graph_probe_requested": True,
        "mps_graph_probe_ran": True,
        "mps_graph_probe_ok": True,
        "mps_graph_probe_error": None,
        "metal4_ml_runtime_available": False,
        "mpp_runtime_available": False,
        "reasons": (),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _fake_prompt_prefill_result(
    tmp_path: Path,
    *,
    accelerated: bool,
    prompt_token_ids: tuple[int, ...] = (0, 1),
    accelerated_flop_fraction: float | None = None,
    coverage_ok: bool | None = None,
) -> PromptPrefillResult:
    backend_counts = {"mpsgraph-f32": 1} if accelerated else {"custom-metal": 1}
    estimated_flops = 2 * len(prompt_token_ids) * 32 * 32
    fraction = 1.0 if accelerated else 0.0
    if accelerated_flop_fraction is not None:
        fraction = accelerated_flop_fraction
    accelerated_flops = int(estimated_flops * fraction)
    backend_flops = (
        {"mpsgraph-f32": accelerated_flops, "custom-metal": estimated_flops - accelerated_flops}
        if accelerated
        else {"custom-metal": estimated_flops}
    )
    coverage = {
        "required": False,
        "analyzed": True,
        "ok": accelerated if coverage_ok is None else coverage_ok,
        "matrix_count": 1,
        "accelerated_matrix_count": 1 if accelerated else 0,
        "mpsgraph_matrix_count": 1 if accelerated else 0,
        "custom_metal_matrix_count": 0 if accelerated else 1,
        "unsupported_mpsgraph_matrix_count": 0,
        "other_matrix_count": 0,
        "total_estimated_flops": estimated_flops,
        "accelerated_estimated_flops": accelerated_flops if accelerated else 0,
        "custom_metal_estimated_flops": (
            estimated_flops - accelerated_flops if accelerated else estimated_flops
        ),
        "unsupported_mpsgraph_estimated_flops": 0,
        "other_estimated_flops": 0,
        "accelerated_flop_fraction": fraction if accelerated else 0.0,
        "dominant_resident_flops_accelerated": accelerated and fraction >= 0.5,
        "any_resident_matrix_accelerated": accelerated,
        "all_resident_matrices_accelerated": accelerated,
        "reason": "" if accelerated else "no resident prefill matrices used an accelerated backend",
    }
    frontier = {
        "source": "prompt_prefill_actual",
        "analyzed": True,
        "prompt_token_count": len(prompt_token_ids),
        "resolved_prompt_chunk_tokens": len(prompt_token_ids),
        "minimum_accelerated_prompt_chunk_tokens": (
            len(prompt_token_ids) if accelerated else None
        ),
        "suggested_guard_flags": None,
        "candidates": (),
        "reason": "" if accelerated else coverage["reason"],
    }
    return PromptPrefillResult(
        runner_path=tmp_path / "runner",
        expert_layout_path=tmp_path / "experts" / "layout.json",
        resident_layout_path=tmp_path / "resident" / "layout.json",
        cache_layout_path=tmp_path / "decode_cache_layout.json",
        cache_file_path=tmp_path / "decode_cache.bin",
        output_last_hidden_path=tmp_path / "last_hidden.f32",
        output_final_chunk_path=None,
        work_dir=tmp_path,
        kept_work_dir=False,
        elapsed_seconds=0.0,
        prompt_token_ids=prompt_token_ids,
        start_position=0,
        chunk_tokens=len(prompt_token_ids),
        chunk_count=1,
        layers=(),
        dense_layers=(),
        hidden_dim=0,
        total_embedding_read_bytes=0,
        total_embedding_output_bytes=0,
        total_staged_bytes=0,
        total_compact_stage_bytes=0,
        total_compact_stage_materialized_bytes=0,
        max_staged_bytes=0,
        max_compact_stage_bytes=0,
        max_compact_stage_materialized_bytes=0,
        total_stage_plus_compact_bytes=0,
        total_stage_plus_compact_materialized_bytes=0,
        max_stage_plus_compact_bytes=0,
        max_stage_plus_compact_materialized_bytes=0,
        total_expert_stage_serial_read_bytes=0,
        total_expert_stage_unique_requested_bytes=0,
        total_expert_stage_planned_read_bytes=0,
        total_expert_stage_waste_bytes=0,
        total_expert_stage_coalesced_savings_bytes=0,
        total_expert_stage_planned_read_seconds=None,
        prefill_ssd_read_gib_per_second=0.0,
        prefill_max_routed_read_seconds=0.0,
        total_expert_stage_read_seconds_ok=None,
        total_expert_stage_copy_seconds_ok=None,
        prefill_max_stage_raw_ranges=0,
        prefill_max_stage_coalesced_ranges=0,
        total_expert_stage_raw_ranges=0,
        total_expert_stage_coalesced_ranges=0,
        max_expert_stage_raw_ranges=0,
        max_expert_stage_coalesced_ranges=0,
        total_expert_stage_raw_ranges_ok=None,
        total_expert_stage_coalesced_ranges_ok=None,
        total_expert_stage_read_advice_attempted_ranges=0,
        total_expert_stage_read_advice_calls=0,
        total_expert_stage_read_advice_bytes=0,
        total_expert_stage_read_advice_failures=0,
        total_expert_stage_assignment_read_amplification=0.0,
        total_expert_stage_unique_read_amplification=0.0,
        max_expert_stage_unique_read_amplification=0.0,
        max_expert_stage_stage_budget_utilization=0.0,
        total_routed_expert_assignments=0,
        total_routed_unique_expert_slots=0,
        max_routed_unique_experts_per_call=0,
        max_routed_tokens_per_expert=0,
        moe_token_block="auto",
        moe_token_block_mode_counts={},
        max_effective_moe_token_block=0,
        max_moe_max_expert_tokens=0,
        max_moe_batch_buffer_bytes=0,
        max_moe_estimated_peak_bytes=0,
        persistent_moe_plan_server=False,
        persistent_resident_linear_server=False,
        persistent_attention_projection_server=False,
        persistent_attention_output_server=False,
        persistent_shared_expert_server=False,
        persistent_rope_split_server=False,
        persistent_mla_attention_server=False,
        persistent_rmsnorm_server=False,
        moe_output_accumulator="env",
        moe_plan_server_plan_count=0,
        moe_plan_server_job_count=0,
        routed_moe_runner_command_count=0,
        static_capacity_per_expert=None,
        max_static_capacity_per_expert=0,
        total_static_capacity_used_slots=0,
        total_static_capacity_slots=0,
        total_static_capacity_overflow_assignments=0,
        total_static_capacity_binary_bytes=0,
        estimated_peak_bytes=0,
        live_memory_budget=LiveMemoryBudget(
            estimated_live_working_set_bytes=0,
            max_live_working_set_bytes=None,
            min_available_memory_bytes=0,
            system_available_bytes=None,
            system_total_bytes=None,
            system_source=None,
        ),
        linear_backend_counts=backend_counts,
        linear_backend_flops=backend_flops,
        total_linear_matrix_scratch_bytes=0,
        max_linear_matrix_scratch_bytes=0,
        total_linear_matrix_f32_bytes=0,
        total_linear_matrix_raw_conversion_bytes=0,
        total_linear_estimated_flops=estimated_flops,
        accelerated_linear_estimated_flops=estimated_flops if accelerated else 0,
        custom_linear_estimated_flops=0 if accelerated else estimated_flops,
        unsupported_linear_estimated_flops=0,
        accelerated_linear_flop_fraction=1.0 if accelerated else 0.0,
        prefill_acceleration_coverage=coverage,
        prefill_acceleration_frontier=frontier,
        chunks=(),
    )


def _set_prepared_context(prepared: Path, context_tokens: int) -> None:
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["max_context_tokens"] = context_tokens
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache_layout_path = prepared / "decode_cache_layout.json"
    cache_layout = json.loads(cache_layout_path.read_text(encoding="utf-8"))
    cache_layout["max_context_tokens"] = context_tokens
    total_bytes = 0
    for segment in cache_layout.get("segments", []):
        if isinstance(segment, dict):
            segment["max_context_tokens"] = context_tokens
            offset = int(segment["offset"])
            width = int(segment["width"])
            dtype_bytes = int(segment["dtype_bytes"])
            total_bytes = max(total_bytes, offset + width * dtype_bytes * context_tokens)
    cache_layout["total_bytes"] = total_bytes
    cache_layout_path.write_text(json.dumps(cache_layout), encoding="utf-8")
    (prepared / "decode_cache.bin").write_bytes(b"\0" * total_bytes)


def _add_prepared_context_budget_metadata(
    prepared: Path,
    *,
    auto_context_from_budget: bool = False,
    budget_bytes: int | None = None,
    safe_context_tokens: int | None = None,
) -> None:
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_layout = json.loads(
        (prepared / "decode_cache_layout.json").read_text(encoding="utf-8")
    )
    context_tokens = int(cache_layout["max_context_tokens"])
    total_bytes = int(cache_layout["total_bytes"])
    budget_bytes = max(total_bytes, budget_bytes or total_bytes)
    safe_context_tokens = max(context_tokens, safe_context_tokens or context_tokens)
    manifest["prepare_auto_context_from_budget"] = auto_context_from_budget
    manifest["prepare_requested_max_context_tokens"] = (
        None if auto_context_from_budget else context_tokens
    )
    manifest["prepare_resolved_max_context_tokens"] = context_tokens
    manifest["prepare_decode_cache_budget_bytes"] = budget_bytes
    manifest["prepare_decode_cache_safe_context_tokens"] = safe_context_tokens
    manifest["prepare_effective_max_cache_bytes"] = budget_bytes
    manifest["prepare_cache_dtype"] = str(cache_layout["dtype"])
    manifest["prepare_cache_alignment"] = int(cache_layout["alignment"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_prepared_manifest_rejects_context_mismatch(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["max_context_tokens"] = 8
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="max_context_tokens"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_prepare_resolved_context_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_resolved_max_context_tokens"] = 8
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_resolved_max_context_tokens",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_short_cache_file(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    (prepared / "decode_cache.bin").write_bytes(b"\0" * 31)

    with pytest.raises(PreparedManifestError, match="smaller than layout"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_oversized_cache_file(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    (prepared / "decode_cache.bin").write_bytes(b"\0" * 33)

    with pytest.raises(PreparedManifestError, match="larger than layout"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_boolean_integer_field(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["max_context_tokens"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="max_context_tokens must be an integer"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_recorded_cache_bytes_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["decode_cache_bytes"] = 31
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="decode_cache_bytes"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_prepare_cache_budget_below_layout(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_decode_cache_budget_bytes"] = 31
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_decode_cache_budget_bytes",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_auto_context_without_budget_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_auto_context_from_budget"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_auto_context_from_budget requires",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_reads_runtime_guard_recommendations(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recommended_max_live_working_set_bytes"] = 9 * 1024**3
    payload["recommended_min_free_unified_memory_bytes"] = 24 * 1024**3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_prepared_manifest(prepared)

    assert manifest.recommended_max_live_working_set_bytes == 9 * 1024**3
    assert manifest.recommended_min_free_unified_memory_bytes == 24 * 1024**3


def test_load_prepared_manifest_records_validated_storage_bytes(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    manifest = load_prepared_manifest(prepared)

    assert manifest.expert_layout_bytes == 16
    assert manifest.expert_layout_quantization == "mlx-affine-int4"
    assert manifest.expert_layout_group_size == 8
    assert manifest.resident_layout_bytes == 16
    assert manifest.decode_cache_layout_bytes == 32
    assert manifest.decode_cache_file_bytes == 32
    assert manifest.model_config_sha256 is None
    assert manifest.expert_quantization is None
    assert manifest.expert_group_size is None
    assert manifest.prepare_effective_unified_memory_bytes is None
    assert manifest.prepare_cold_read_gib_per_second is None
    assert manifest.prepare_cold_read_source is None
    assert manifest.prepare_cold_read_benchmark_path is None
    assert manifest.prepare_cold_read_benchmark_requested_bytes is None
    assert manifest.prepare_cold_read_benchmark_measured_bytes is None
    assert manifest.prepare_cold_read_benchmark_elapsed_seconds is None
    assert manifest.prepare_flags_applied is None
    assert manifest.prepare_flags_source is None
    assert manifest.prepare_flags_path is None
    assert manifest.prepare_flags_sha256 is None
    assert manifest.prepare_combined_output_required_bytes is None
    assert manifest.prepare_combined_output_available_bytes is None
    assert manifest.prepare_combined_output_disk_margin_bytes is None
    assert manifest.prepare_live_memory_estimated_live_working_set_bytes is None
    assert manifest.prepare_live_memory_min_available_memory_bytes is None
    assert manifest.prepare_live_memory_required_available_memory_bytes is None
    assert manifest.prepare_live_memory_system_available_memory_bytes is None
    assert manifest.prepare_live_memory_system_total_bytes is None
    assert manifest.prepare_live_memory_system_source is None
    assert manifest.prepare_expert_pack_chunk_size_bytes is None
    assert manifest.prepare_expert_pack_estimated_peak_heap_bytes is None
    assert manifest.prepare_expert_pack_max_heap_bytes is None
    assert manifest.prepare_raw_quantization_extra_heap_bytes is None
    assert manifest.prepare_raw_quantization_max_source_block_bytes is None
    assert manifest.prepare_raw_quantization_max_output_block_bytes is None
    assert manifest.prepare_raw_quantization_max_rows_per_block is None
    assert manifest.prepare_resident_component_alias_source_tensor_count is None
    assert manifest.prepare_resident_component_alias_renamed_tensor_count is None
    assert manifest.prepare_resident_component_alias_bytes is None
    assert manifest.prepare_resident_fused_gate_up_source_tensor_count is None
    assert manifest.prepare_resident_fused_gate_up_expanded_tensor_count is None
    assert manifest.prepare_resident_fused_gate_up_expanded_bytes is None
    assert manifest.prepare_hardware_apple_silicon_generation is None
    assert manifest.prepare_hardware_apple_silicon_tier is None
    assert manifest.prepare_public_glm_5_2_shape_required is None
    assert manifest.prepare_public_glm_5_2_shape_matches is None
    assert manifest.prepare_public_glm_5_2_shape_mismatched_fields is None


def test_load_prepared_manifest_rejects_live_memory_required_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_live_memory_estimated_live_working_set_bytes": 1024,
            "prepare_live_memory_min_available_memory_bytes": 2048,
            "prepare_live_memory_required_available_memory_bytes": 4096,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="required_available_memory_bytes does not match",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_combined_disk_budget_shortfall(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_combined_output_required_bytes": 1024,
            "prepare_combined_output_available_bytes": 1024,
            "prepare_combined_output_disk_margin_bytes": 1,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="combined disk budget records insufficient",
    ):
        load_prepared_manifest(prepared)


def test_prepared_launch_profile_target_records_prepare_flags_identity(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_identity_strong(prepared)
    flags = tmp_path / "prepare-flags.json"
    flags.write_text(
        json.dumps(
            {
                "source": "plan",
                "argv_safe_to_replay": True,
                "argv": ["--auto-context-from-budget"],
            }
        ),
        encoding="utf-8",
    )
    expected_sha256 = hashlib.sha256(flags.read_bytes()).hexdigest()
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_flags_applied"] = True
    payload["prepare_flags_source"] = "plan"
    payload["prepare_flags_path"] = str(flags)
    payload["prepare_flags_sha256"] = expected_sha256
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    target = prepared_launch_profile_target(load_prepared_manifest(prepared))

    assert target["identity_strength"] == "strong"
    assert target["prepare_flags_applied"] is True
    assert target["prepare_flags_source"] == "plan"
    assert target["prepare_flags_sha256"] == expected_sha256
    assert "prepare_flags_path" not in target


def test_prepared_launch_profile_target_records_prepare_hardware_identity(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_identity_strong(prepared)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_hardware_chip_name": "Apple M5 Max",
            "prepare_hardware_unified_memory_bytes": 128 * 1024**3,
            "prepare_hardware_gpu_cores": 40,
            "prepare_hardware_apple_silicon_generation": 5,
            "prepare_hardware_apple_silicon_tier": "Max",
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    target = prepared_launch_profile_target(load_prepared_manifest(prepared))

    assert target["identity_strength"] == "strong"
    assert target["prepare_hardware_chip_name"] == "Apple M5 Max"
    assert target["prepare_hardware_unified_memory_bytes"] == 128 * 1024**3
    assert target["prepare_hardware_gpu_cores"] == 40
    assert target["prepare_hardware_apple_silicon_generation"] == 5
    assert target["prepare_hardware_apple_silicon_tier"] == "Max"


def test_prepared_launch_profile_target_records_prepare_safety_identity(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_identity_strong(prepared)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_expert_pack_chunk_size_bytes": 4096,
            "prepare_expert_pack_estimated_peak_heap_bytes": 8192,
            "prepare_expert_pack_max_heap_bytes": 16384,
            "prepare_raw_quantization_extra_heap_bytes": 256,
            "prepare_raw_quantization_max_source_block_bytes": 512,
            "prepare_raw_quantization_max_output_block_bytes": 128,
            "prepare_raw_quantization_max_rows_per_block": 4,
            "prepare_resident_component_alias_source_tensor_count": 3,
            "prepare_resident_component_alias_renamed_tensor_count": 3,
            "prepare_resident_component_alias_bytes": 6,
            "prepare_resident_fused_gate_up_source_tensor_count": 1,
            "prepare_resident_fused_gate_up_expanded_tensor_count": 2,
            "prepare_resident_fused_gate_up_expanded_bytes": 8,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    target = prepared_launch_profile_target(load_prepared_manifest(prepared))

    assert target["identity_strength"] == "strong"
    assert target["prepare_expert_pack_chunk_size_bytes"] == 4096
    assert target["prepare_expert_pack_estimated_peak_heap_bytes"] == 8192
    assert target["prepare_expert_pack_max_heap_bytes"] == 16384
    assert target["prepare_raw_quantization_max_source_block_bytes"] == 512
    assert target["prepare_raw_quantization_max_rows_per_block"] == 4
    assert target["prepare_resident_component_alias_source_tensor_count"] == 3
    assert target["prepare_resident_component_alias_renamed_tensor_count"] == 3
    assert target["prepare_resident_component_alias_bytes"] == 6
    assert target["prepare_resident_fused_gate_up_source_tensor_count"] == 1
    assert target["prepare_resident_fused_gate_up_expanded_tensor_count"] == 2
    assert target["prepare_resident_fused_gate_up_expanded_bytes"] == 8


def test_launch_profile_target_mismatches_when_prepare_safety_identity_changes(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_identity_strong(prepared)
    old_target = prepared_launch_profile_target(load_prepared_manifest(prepared))

    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_resident_component_alias_source_tensor_count": 3,
            "prepare_resident_component_alias_renamed_tensor_count": 3,
            "prepare_resident_component_alias_bytes": 6,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    new_target = prepared_launch_profile_target(load_prepared_manifest(prepared))

    assert old_target["prepare_resident_component_alias_source_tensor_count"] is None
    assert new_target["prepare_resident_component_alias_source_tensor_count"] == 3
    assert not _launch_profile_targets_match(old_target, new_target)


def test_launch_profile_target_matches_json_list_tuple_fields(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_identity_strong(prepared)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_public_glm_5_2_shape_required": True,
            "prepare_public_glm_5_2_shape_matches": True,
            "prepare_public_glm_5_2_shape_mismatched_fields": [],
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    target = prepared_launch_profile_target(load_prepared_manifest(prepared))
    json_target = json.loads(json.dumps(target))

    assert target["prepare_public_glm_5_2_shape_mismatched_fields"] == ()
    assert json_target["prepare_public_glm_5_2_shape_mismatched_fields"] == []
    assert _launch_profile_targets_match(json_target, target)


def test_load_prepared_manifest_rejects_invalid_prepare_flags_sha256(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_flags_applied"] = True
    payload["prepare_flags_source"] = "plan"
    payload["prepare_flags_sha256"] = "not-a-sha"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="prepare_flags_sha256"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_prepare_pack_heap_over_limit(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_expert_pack_estimated_peak_heap_bytes"] = 1024
    payload["prepare_expert_pack_max_heap_bytes"] = 512
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_expert_pack_estimated_peak_heap_bytes exceeds",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_internal_int4_without_pack_heap_evidence(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_expert_layout_quantization(prepared, "largerlm-affine-int4")

    with pytest.raises(
        PreparedManifestError,
        match="requires prepare expert-pack heap evidence",
    ) as exc_info:
        load_prepared_manifest(prepared)

    assert "prepare_expert_pack_chunk_size_bytes" in str(exc_info.value)
    assert "prepare_raw_quantization_max_rows_per_block" in str(exc_info.value)


def test_load_prepared_manifest_accepts_internal_int4_with_pack_heap_evidence(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_expert_layout_quantization(prepared, "largerlm-affine-int4")
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_expert_pack_chunk_size_bytes": 32,
            "prepare_expert_pack_estimated_peak_heap_bytes": 128,
            "prepare_expert_pack_max_heap_bytes": 256,
            "prepare_raw_quantization_extra_heap_bytes": 16,
            "prepare_raw_quantization_max_source_block_bytes": 32,
            "prepare_raw_quantization_max_output_block_bytes": 16,
            "prepare_raw_quantization_max_rows_per_block": 1,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_prepared_manifest(prepared)

    assert manifest.expert_quantization is None
    assert manifest.expert_layout_quantization == "largerlm-affine-int4"
    assert manifest.expert_layout_group_size == 8
    assert manifest.prepare_expert_pack_chunk_size_bytes == 32
    assert manifest.prepare_expert_pack_estimated_peak_heap_bytes == 128
    assert manifest.prepare_expert_pack_max_heap_bytes == 256
    assert manifest.prepare_raw_quantization_extra_heap_bytes == 16
    assert manifest.prepare_raw_quantization_max_source_block_bytes == 32
    assert manifest.prepare_raw_quantization_max_output_block_bytes == 16
    assert manifest.prepare_raw_quantization_max_rows_per_block == 1


def test_prepare_expert_pack_heap_audit_uses_internal_layout_quantization(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_expert_layout_quantization(prepared, "largerlm-affine-int4")
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_expert_pack_chunk_size_bytes": 32,
            "prepare_expert_pack_estimated_peak_heap_bytes": 128,
            "prepare_expert_pack_max_heap_bytes": 256,
            "prepare_raw_quantization_extra_heap_bytes": 16,
            "prepare_raw_quantization_max_source_block_bytes": 32,
            "prepare_raw_quantization_max_output_block_bytes": 16,
            "prepare_raw_quantization_max_rows_per_block": 1,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_prepared_manifest(prepared)

    audit = _prepare_expert_pack_heap_audit_details(
        _prepare_expert_pack_heap_source_from_prepared(manifest)
    )

    assert audit["required"] is True
    assert audit["ok"] is True
    assert audit["expert_quantization"] is None
    assert audit["expert_layout_quantization"] == "largerlm-affine-int4"
    assert audit["effective_expert_quantization"] == "largerlm-affine-int4"


def test_load_prepared_manifest_rejects_raw_quant_bytes_without_rows(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_raw_quantization_max_source_block_bytes"] = 32
    payload["prepare_raw_quantization_max_rows_per_block"] = 0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_raw_quantization_max_rows_per_block=0 conflicts",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_raw_quant_block_over_peak(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_expert_pack_estimated_peak_heap_bytes"] = 64
    payload["prepare_expert_pack_max_heap_bytes"] = 128
    payload["prepare_raw_quantization_max_source_block_bytes"] = 65
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_raw_quantization_max_source_block_bytes exceeds",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_raw_quant_output_not_in_extra_heap(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_expert_pack_chunk_size_bytes"] = 32
    payload["prepare_expert_pack_estimated_peak_heap_bytes"] = 128
    payload["prepare_expert_pack_max_heap_bytes"] = 256
    payload["prepare_raw_quantization_extra_heap_bytes"] = 15
    payload["prepare_raw_quantization_max_output_block_bytes"] = 16
    payload["prepare_raw_quantization_max_rows_per_block"] = 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="extra_heap_bytes is smaller than .*max_output_block_bytes",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_raw_quant_source_overflow_not_in_extra_heap(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_expert_pack_chunk_size_bytes"] = 32
    payload["prepare_expert_pack_estimated_peak_heap_bytes"] = 128
    payload["prepare_expert_pack_max_heap_bytes"] = 256
    payload["prepare_raw_quantization_extra_heap_bytes"] = 40
    payload["prepare_raw_quantization_max_source_block_bytes"] = 64
    payload["prepare_raw_quantization_max_output_block_bytes"] = 16
    payload["prepare_raw_quantization_max_rows_per_block"] = 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="source overflow plus generated output",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_bad_resident_fused_gate_up_stats(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_resident_fused_gate_up_source_tensor_count"] = 2
    payload["prepare_resident_fused_gate_up_expanded_tensor_count"] = 3
    payload["prepare_resident_fused_gate_up_expanded_bytes"] = 128
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="expanded_tensor_count.*source tensor count",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_bad_resident_component_alias_stats(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_resident_component_alias_source_tensor_count"] = 3
    payload["prepare_resident_component_alias_renamed_tensor_count"] = 2
    payload["prepare_resident_component_alias_bytes"] = 128
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="renamed_tensor_count.*source tensor count",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_resident_alias_bytes_over_layout(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_resident_component_alias_source_tensor_count"] = 3
    payload["prepare_resident_component_alias_renamed_tensor_count"] = 3
    payload["prepare_resident_component_alias_bytes"] = 17
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_resident_component_alias_bytes exceeds resident layout bytes",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_resident_alias_bytes_sum_over_layout(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_resident_component_alias_source_tensor_count"] = 3
    payload["prepare_resident_component_alias_renamed_tensor_count"] = 3
    payload["prepare_resident_component_alias_bytes"] = 9
    payload["prepare_resident_fused_gate_up_source_tensor_count"] = 1
    payload["prepare_resident_fused_gate_up_expanded_tensor_count"] = 2
    payload["prepare_resident_fused_gate_up_expanded_bytes"] = 8
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="resident alias rewrite bytes exceed resident layout bytes",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_expert_quantization_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["expert_quantization"] = "largerlm-affine-int4"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="expert_quantization"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_expert_group_size_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["expert_group_size"] = 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="expert_group_size"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_reads_prepare_profile_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "expert_quantization": "mlx-affine-int4",
            "expert_group_size": 8,
            "prepare_hardware_chip_name": "Apple M5 Max",
            "prepare_hardware_unified_memory_bytes": 128 * 1024**3,
            "prepare_hardware_gpu_cores": 40,
            "prepare_hardware_apple_silicon_generation": 5,
            "prepare_hardware_apple_silicon_tier": "Max",
            "prepare_effective_unified_memory_bytes": 128 * 1024**3,
            "prepare_effective_unified_memory_source": "explicit",
            "prepare_system_reserve_bytes": 24 * 1024**3,
            "prepare_cold_read_gib_per_second": 16.5,
            "prepare_cold_read_source": "explicit",
            "prepare_expert_pack_chunk_size_bytes": 17,
            "prepare_expert_pack_estimated_peak_heap_bytes": 4096,
            "prepare_expert_pack_max_heap_bytes": 8192,
            "prepare_raw_quantization_extra_heap_bytes": 64,
            "prepare_raw_quantization_max_source_block_bytes": 32,
            "prepare_raw_quantization_max_output_block_bytes": 32,
            "prepare_raw_quantization_max_rows_per_block": 1,
            "prepare_resident_component_alias_source_tensor_count": 3,
            "prepare_resident_component_alias_renamed_tensor_count": 3,
            "prepare_resident_component_alias_bytes": 6,
            "prepare_resident_fused_gate_up_source_tensor_count": 2,
            "prepare_resident_fused_gate_up_expanded_tensor_count": 4,
            "prepare_resident_fused_gate_up_expanded_bytes": 8,
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_prepared_manifest(prepared)

    assert manifest.expert_quantization == "mlx-affine-int4"
    assert manifest.expert_group_size == 8
    assert manifest.prepare_hardware_chip_name == "Apple M5 Max"
    assert manifest.prepare_hardware_unified_memory_bytes == 128 * 1024**3
    assert manifest.prepare_hardware_gpu_cores == 40
    assert manifest.prepare_hardware_apple_silicon_generation == 5
    assert manifest.prepare_hardware_apple_silicon_tier == "Max"
    assert manifest.prepare_effective_unified_memory_bytes == 128 * 1024**3
    assert manifest.prepare_effective_unified_memory_source == "explicit"
    assert manifest.prepare_system_reserve_bytes == 24 * 1024**3
    assert manifest.prepare_cold_read_gib_per_second == 16.5
    assert manifest.prepare_cold_read_source == "explicit"
    assert manifest.prepare_expert_pack_chunk_size_bytes == 17
    assert manifest.prepare_expert_pack_estimated_peak_heap_bytes == 4096
    assert manifest.prepare_expert_pack_max_heap_bytes == 8192
    assert manifest.prepare_raw_quantization_extra_heap_bytes == 64
    assert manifest.prepare_raw_quantization_max_source_block_bytes == 32
    assert manifest.prepare_raw_quantization_max_output_block_bytes == 32
    assert manifest.prepare_raw_quantization_max_rows_per_block == 1
    assert manifest.prepare_resident_component_alias_source_tensor_count == 3
    assert manifest.prepare_resident_component_alias_renamed_tensor_count == 3
    assert manifest.prepare_resident_component_alias_bytes == 6
    assert manifest.prepare_resident_fused_gate_up_source_tensor_count == 2
    assert manifest.prepare_resident_fused_gate_up_expanded_tensor_count == 4
    assert manifest.prepare_resident_fused_gate_up_expanded_bytes == 8


def test_load_prepared_manifest_reads_prepare_public_glm_5_2_shape_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_public_glm_5_2_shape_required": True,
            "prepare_public_glm_5_2_shape_matches": True,
            "prepare_public_glm_5_2_shape_mismatched_fields": [],
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_prepared_manifest(prepared)

    assert manifest.prepare_public_glm_5_2_shape_required is True
    assert manifest.prepare_public_glm_5_2_shape_matches is True
    assert manifest.prepare_public_glm_5_2_shape_mismatched_fields == ()


def test_load_prepared_manifest_rejects_inconsistent_prepare_public_shape_metadata(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_public_glm_5_2_shape_required": True,
            "prepare_public_glm_5_2_shape_matches": False,
            "prepare_public_glm_5_2_shape_mismatched_fields": ["hidden_size"],
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_public_glm_5_2_shape_required=true",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_matching_public_shape_with_mismatches(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_public_glm_5_2_shape_matches": True,
            "prepare_public_glm_5_2_shape_mismatched_fields": ["hidden_size"],
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="prepare_public_glm_5_2_shape_matches=true conflicts",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_accepts_matching_layout_config_hash(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    digest = config_sha256(tmp_path / "model")
    assert digest is not None
    _set_prepared_layout_config_sha256(prepared, digest)

    manifest = load_prepared_manifest(prepared)

    assert manifest.model_dir == tmp_path / "model"
    assert manifest.model_config_sha256 == digest


def test_load_prepared_manifest_rejects_manifest_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    digest = config_sha256(tmp_path / "model")
    assert digest is not None
    _set_prepared_layout_config_sha256(prepared, digest)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_config_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PreparedManifestError,
        match="manifest model_config_sha256 does not match validated layouts",
    ):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_model_config_hash_drift(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    digest = config_sha256(tmp_path / "model")
    assert digest is not None
    _set_prepared_layout_config_sha256(prepared, digest)
    (tmp_path / "model" / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="config_sha256 mismatch"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_layout_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    expert_layout = prepared / "experts" / "layout.json"
    resident_layout = prepared / "resident" / "layout.json"
    expert_payload = json.loads(expert_layout.read_text(encoding="utf-8"))
    resident_payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    expert_payload["config_sha256"] = "a" * 64
    resident_payload["config_sha256"] = "b" * 64
    expert_layout.write_text(json.dumps(expert_payload), encoding="utf-8")
    resident_layout.write_text(json.dumps(resident_payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="config_sha256 values"):
        load_prepared_manifest(prepared)


def test_validate_layout_backing_rejects_layout_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    expert_layout = prepared / "experts" / "layout.json"
    resident_layout = prepared / "resident" / "layout.json"
    expert_payload = json.loads(expert_layout.read_text(encoding="utf-8"))
    resident_payload = json.loads(resident_layout.read_text(encoding="utf-8"))
    expert_payload["config_sha256"] = "a" * 64
    resident_payload["config_sha256"] = "b" * 64
    expert_layout.write_text(json.dumps(expert_payload), encoding="utf-8")
    resident_layout.write_text(json.dumps(resident_payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="config_sha256 values"):
        validate_layout_backing_files(expert_layout, resident_layout)


def test_load_prepared_manifest_rejects_negative_runtime_guard_recommendation(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recommended_min_free_unified_memory_bytes"] = -1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="recommended_min_free"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_short_expert_layer_file(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    (prepared / "experts" / "layer_000.bin").write_bytes(b"\0" * 15)

    with pytest.raises(PreparedManifestError, match="expert layer file"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_short_resident_file(tmp_path: Path) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    (prepared / "resident" / "resident.bin").write_bytes(b"\0" * 15)

    with pytest.raises(PreparedManifestError, match="resident weight file"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_expert_component_past_slot(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "experts" / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["layers"][0]["components"][0]["offset"] = 15
    payload["layers"][0]["components"][0]["size"] = 2
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="expert layout layer 0 component"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_overlapping_expert_components(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "experts" / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["layers"][0]["components"].append(
        {
            "name": "up_proj.weight",
            "offset": 8,
            "size": 8,
            "dtype": "U32",
            "shape": [4, 1],
        }
    )
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="overlaps"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_resident_tensor_past_total(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "resident" / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["tensors"][0]["offset"] = 15
    payload["tensors"][0]["size"] = 2
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="resident layout tensor"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_overlapping_resident_tensors(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "resident" / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["tensors"].append(
        {
            "name": "model.norm.weight",
            "offset": 8,
            "size": 8,
            "dtype": "F32",
            "shape": [2],
            "category": "norms",
        }
    )
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="overlaps"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_escaped_expert_layer_path(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "experts" / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["layers"][0]["layer_file"] = "../layer_000.bin"
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="relative path"):
        load_prepared_manifest(prepared)


def test_load_prepared_manifest_rejects_escaped_resident_weight_path(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    layout_path = prepared / "resident" / "layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["weight_file"] = "../resident.bin"
    layout_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreparedManifestError, match="relative path"):
        load_prepared_manifest(prepared)


def test_prepare_glm_dry_run_writes_nothing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared" / "nested"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.executed is False
    assert report.expert_pack is not None
    assert report.resident_pack is not None
    assert report.cache_layout is not None
    assert report.public_glm_5_2_shape is not None
    assert report.public_glm_5_2_shape["matches"] is False
    assert "hidden_size" in report.public_glm_5_2_shape["mismatched_fields"]
    assert not output.exists()


def test_prepare_glm_rejects_output_dir_equal_to_model_dir(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)

    with pytest.raises(PrepareError, match="checkpoint model directory"):
        prepare_glm_checkpoint(
            model,
            output_dir=model,
            max_context_tokens=4,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            unified_memory_bytes=128 * 1024**3,
        )


def test_prepare_glm_rejects_output_dir_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared-file"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(PrepareError, match="output_dir must be a directory path"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            unified_memory_bytes=128 * 1024**3,
        )


def test_prepare_glm_require_public_glm_5_2_shape_rejects_non_public(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    with pytest.raises(
        PrepareError,
        match="prepared config does not match the public GLM-5.2 shape",
    ) as excinfo:
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            quantize_raw_to_int4=True,
            require_public_glm_5_2_shape=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            unified_memory_bytes=128 * 1024**3,
        )

    assert "hidden_size" in str(excinfo.value)
    assert "num_hidden_layers" in str(excinfo.value)
    assert not output.exists()


def test_prepare_glm_require_public_glm_5_2_shape_allows_matching_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)

    monkeypatch.setattr(
        prepare_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )
    monkeypatch.setattr(
        preflight_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )

    report = prepare_glm_checkpoint(
        model,
        output_dir=tmp_path / "prepared",
        max_context_tokens=4,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.require_public_glm_5_2_shape is True
    assert report.public_glm_5_2_shape == {
        "matches": True,
        "mismatched_fields": (),
        "checks": {},
    }


def test_prepare_glm_execute_records_required_public_shape_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    monkeypatch.setattr(
        prepare_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )
    monkeypatch.setattr(
        preflight_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_public_glm_5_2_shape_required"] is True
    assert manifest["prepare_public_glm_5_2_shape_matches"] is True
    assert manifest["prepare_public_glm_5_2_shape_mismatched_fields"] == []
    prepared = load_prepared_manifest(output)
    assert prepared.prepare_public_glm_5_2_shape_required is True
    assert prepared.prepare_public_glm_5_2_shape_matches is True
    assert prepared.prepare_public_glm_5_2_shape_mismatched_fields == ()


def test_prepare_glm_execute_records_live_memory_guard_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    def fake_check_live_memory_budget(**kwargs):
        return LiveMemoryBudget(
            estimated_live_working_set_bytes=kwargs[
                "estimated_live_working_set_bytes"
            ],
            max_live_working_set_bytes=kwargs["max_live_working_set_bytes"],
            min_available_memory_bytes=kwargs["min_available_memory_bytes"],
            system_available_bytes=128 * 1024**3,
            system_total_bytes=128 * 1024**3,
            system_source="test",
            nonresident_peak_bytes=kwargs["nonresident_peak_bytes"],
        )

    monkeypatch.setattr(
        prepare_module,
        "check_live_memory_budget",
        fake_check_live_memory_budget,
    )

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        max_pack_heap_bytes=512 * 1024**2,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.prepare_live_memory is not None
    assert report.prepare_live_memory.min_available_memory_bytes == 24 * 1024**3
    assert report.prepare_live_memory.system_available_bytes == 128 * 1024**3
    assert report.prepare_disk_budget is not None
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_combined_output_required_bytes"] == (
        report.expert_pack.layout.total_bytes
        + report.resident_pack.layout.total_bytes
        + report.cache_layout.total_bytes
    )
    assert manifest["prepare_combined_output_available_bytes"] == (
        report.prepare_disk_budget.available_bytes
    )
    assert manifest["prepare_combined_output_disk_margin_bytes"] == 0
    estimated = manifest["prepare_live_memory_estimated_live_working_set_bytes"]
    reserve = manifest["prepare_live_memory_min_available_memory_bytes"]
    assert manifest["prepare_live_memory_required_available_memory_bytes"] == (
        estimated + reserve
    )
    assert manifest["prepare_live_memory_system_available_memory_bytes"] == (
        128 * 1024**3
    )
    assert manifest["prepare_live_memory_system_source"] == "test"
    prepared_manifest = load_prepared_manifest(output)
    assert (
        prepared_manifest.prepare_live_memory_estimated_live_working_set_bytes
        == estimated
    )
    assert (
        prepared_manifest.prepare_live_memory_required_available_memory_bytes
        == estimated + reserve
    )
    assert (
        prepared_manifest.prepare_live_memory_system_available_memory_bytes
        == 128 * 1024**3
    )
    assert prepared_manifest.prepare_live_memory_system_source == "test"
    assert prepared_manifest.prepare_combined_output_required_bytes == (
        manifest["prepare_combined_output_required_bytes"]
    )
    assert prepared_manifest.prepare_combined_output_available_bytes == (
        manifest["prepare_combined_output_available_bytes"]
    )
    assert prepared_manifest.prepare_combined_output_disk_margin_bytes == 0


def test_prepare_glm_execute_rejects_combined_disk_budget_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    def fake_disk_budget(output_dir, required_bytes, *, safety_margin_bytes):
        return DiskBudget(
            output_dir=Path(output_dir),
            required_bytes=required_bytes,
            available_bytes=required_bytes + safety_margin_bytes - 1,
            safety_margin_bytes=safety_margin_bytes,
        )

    monkeypatch.setattr(prepare_module, "disk_budget", fake_disk_budget)

    with pytest.raises(PrepareError, match="combined prepare outputs"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=1,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert not output.exists()


def test_prepare_glm_execute_rejects_incomplete_checkpoint_artifact_before_writes(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    _write_checkpoint_header_manifest(model)
    shard = model / "model-00001-of-00001.safetensors"
    raw = shard.read_bytes()
    shard.write_bytes(raw[:-1])
    output = tmp_path / "prepared"

    with pytest.raises(
        PrepareError,
        match="prepare execute requires a complete, clean checkpoint artifact",
    ) as excinfo:
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    message = str(excinfo.value)
    assert "partial=1" in message
    assert "issue=truncated" in message
    assert not output.exists()


def test_prepare_glm_execute_rejects_low_live_memory_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    def fail_check_live_memory_budget(**_kwargs):
        raise GenerationGuardError(
            "available unified memory 1024 bytes is below required 2048 bytes"
        )

    monkeypatch.setattr(
        prepare_module,
        "check_live_memory_budget",
        fail_check_live_memory_budget,
    )

    with pytest.raises(PrepareError, match="prepare live memory guard failed"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert not output.exists()


def test_prepare_glm_require_public_glm_5_2_shape_requires_4bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    monkeypatch.setattr(
        prepare_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )
    monkeypatch.setattr(
        preflight_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        quant_bits=8,
        quantize_raw_to_int4=True,
        require_public_glm_5_2_shape=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.executed is False
    assert report.expert_pack is None
    assert report.resident_pack is None
    assert report.cache_layout is None
    assert report.preflight.checkpoint_tensor_count == 0
    issue = next(
        item
        for item in report.preflight.issues
        if item.code == "public_glm_5_2_requires_4bit"
    )
    assert issue.severity == "error"
    assert not output.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_context_tokens": True}, "max_context_tokens must be an integer"),
        ({"quant_bits": False}, "quant_bits must be an integer"),
        ({"group_size": 8.5}, "group_size must be an integer"),
        ({"cache_alignment": True}, "cache_alignment must be an integer"),
        ({"max_cache_bytes": 1.5}, "max_cache_bytes must be an integer"),
        (
            {"disk_safety_margin_bytes": False},
            "disk_safety_margin_bytes must be an integer",
        ),
        ({"chunk_size": 1.5}, "chunk_size must be an integer"),
        ({"max_chunk_size": False}, "max_chunk_size must be an integer"),
        ({"max_pack_heap_bytes": 1.5}, "max_pack_heap_bytes must be an integer"),
        ({"unified_memory_bytes": True}, "unified_memory_bytes must be an integer"),
        ({"system_reserve_bytes": 1.5}, "system_reserve_bytes must be an integer"),
        ({"runtime_buffer_bytes": False}, "runtime_buffer_bytes must be an integer"),
        ({"page_cache_fraction": True}, "page_cache_fraction must be numeric"),
        (
            {"cold_read_gib_per_second": False},
            "cold_read_gib_per_second must be numeric",
        ),
        (
            {"auto_cold_read_benchmark": 1},
            "auto_cold_read_benchmark must be a boolean",
        ),
        (
            {"require_public_glm_5_2_shape": 1},
            "require_public_glm_5_2_shape must be a boolean",
        ),
        (
            {"cold_read_benchmark_bytes": False},
            "cold_read_benchmark_bytes must be an integer",
        ),
        (
            {"cold_read_benchmark_chunk_bytes": 1.5},
            "cold_read_benchmark_chunk_bytes must be an integer",
        ),
        (
            {"cold_read_benchmark_chunk_bytes": 513 * 1024**2},
            "cold_read_benchmark_chunk_bytes exceeds safe benchmark chunk limit",
        ),
    ),
)
def test_prepare_glm_rejects_non_integer_budget_inputs(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    params: dict[str, object] = {
        "output_dir": output,
        "max_context_tokens": 4,
        "quantize_raw_to_int4": True,
        "group_size": 8,
        "max_cache_bytes": 1024 * 1024,
        "disk_safety_margin_bytes": 0,
        "unified_memory_bytes": 128 * 1024**3,
    }
    params.update(kwargs)

    with pytest.raises(PrepareError, match=message):
        prepare_glm_checkpoint(model, **params)


def test_prepare_glm_rejects_context_above_model_max_position(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_position_embeddings"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "prepared"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=8,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is False
    assert report.cache_layout is None
    assert not output.exists()
    assert any(
        issue.code == "context_exceeds_model_max" for issue in report.preflight.issues
    )


def test_prepare_glm_auto_context_from_budget(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=None,
        auto_context_from_budget=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.executed is False
    assert report.cache_layout is not None
    assert report.cache_layout.max_context_tokens == 64
    assert report.cache_layout.total_bytes == 1024
    assert report.context_budget is not None
    assert report.context_budget.auto_context_from_budget is True
    assert report.context_budget.requested_max_context_tokens is None
    assert report.context_budget.resolved_max_context_tokens == 64
    assert report.context_budget.decode_cache_budget_bytes == 1024
    assert report.context_budget.decode_cache_safe_context_tokens == 64
    assert report.context_budget.effective_max_cache_bytes == 1024
    assert not output.exists()


def test_prepare_glm_auto_context_respects_model_max_position(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_position_embeddings"] = 32
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "prepared"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=None,
        auto_context_from_budget=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.cache_layout is not None
    assert report.cache_layout.max_context_tokens == 32
    assert report.cache_layout.total_bytes == 512
    assert report.context_budget is not None
    assert report.context_budget.auto_context_from_budget is True
    assert report.context_budget.model_max_position_embeddings == 32
    assert report.context_budget.resolved_max_context_tokens == 32
    assert report.context_budget.decode_cache_safe_context_tokens == 64


def test_prepare_glm_execute_records_auto_context_budget_metadata(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=None,
        auto_context_from_budget=True,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024,
        disk_safety_margin_bytes=0,
        unified_memory_bytes=128 * 1024**3,
    )

    assert report.ok is True
    assert report.cache_layout is not None
    assert report.cache_layout.max_context_tokens == 64

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_auto_context_from_budget"] is True
    assert manifest["prepare_requested_max_context_tokens"] is None
    assert manifest["prepare_resolved_max_context_tokens"] == 64
    assert manifest["prepare_decode_cache_budget_bytes"] == 1024
    assert manifest["prepare_decode_cache_safe_context_tokens"] == 64
    assert manifest["prepare_effective_max_cache_bytes"] == 1024
    assert manifest["prepare_cache_dtype"] == "BF16"
    assert manifest["prepare_cache_alignment"] == 64

    prepared = load_prepared_manifest(output)
    assert prepared.prepare_auto_context_from_budget is True
    assert prepared.prepare_resolved_max_context_tokens == 64
    assert prepared.prepare_decode_cache_budget_bytes == 1024
    assert prepared.prepare_decode_cache_safe_context_tokens == 64
    assert prepared.prepare_effective_max_cache_bytes == 1024


def test_prepare_glm_execute_writes_packed_layouts_and_sparse_cache(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        unified_memory_bytes=128 * 1024**3,
        cold_read_gib_per_second=16.0,
    )

    assert report.ok is True
    assert report.executed is True
    assert (output / "experts" / "layout.json").exists()
    assert (output / "experts" / "layer_001.bin").exists()
    assert (output / "resident" / "layout.json").exists()
    assert (output / "resident" / "resident.bin").exists()
    assert (output / "decode_cache_layout.json").exists()
    assert (output / "decode_cache.bin").stat().st_size == report.cache_layout.total_bytes
    assert (output / "manifest.json").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["recommended_max_live_working_set_bytes"] == (
        8 * 1024**3 + manifest["resident_bytes"]
    )
    assert manifest["recommended_min_free_unified_memory_bytes"] == 24 * 1024**3
    assert manifest["expert_quantization"] == "largerlm-affine-int4"
    assert manifest["expert_group_size"] == 8
    assert manifest["model_config_sha256"] == config_sha256(model)
    assert report.expert_pack is not None
    assert manifest["prepare_expert_pack_chunk_size_bytes"] == (
        report.expert_pack.chunk_size
    )
    assert manifest["prepare_expert_pack_estimated_peak_heap_bytes"] == (
        report.expert_pack.estimated_peak_heap_bytes
    )
    assert manifest["prepare_expert_pack_max_heap_bytes"] == (
        report.expert_pack.max_heap_bytes
    )
    assert manifest["prepare_raw_quantization_extra_heap_bytes"] == (
        report.expert_pack.raw_quantization_extra_heap_bytes
    )
    assert manifest["prepare_raw_quantization_max_source_block_bytes"] == (
        report.expert_pack.raw_quantization_max_source_block_bytes
    )
    assert manifest["prepare_raw_quantization_max_output_block_bytes"] == (
        report.expert_pack.raw_quantization_max_output_block_bytes
    )
    assert manifest["prepare_raw_quantization_max_rows_per_block"] == (
        report.expert_pack.raw_quantization_max_rows_per_block
    )
    assert report.resident_pack is not None
    assert manifest["prepare_resident_component_alias_source_tensor_count"] == (
        report.resident_pack.component_alias_source_tensor_count
    )
    assert manifest["prepare_resident_component_alias_renamed_tensor_count"] == (
        report.resident_pack.component_alias_renamed_tensor_count
    )
    assert manifest["prepare_resident_component_alias_bytes"] == (
        report.resident_pack.component_alias_bytes
    )
    assert manifest["prepare_resident_fused_gate_up_source_tensor_count"] == (
        report.resident_pack.fused_gate_up_source_tensor_count
    )
    assert manifest["prepare_resident_fused_gate_up_expanded_tensor_count"] == (
        report.resident_pack.fused_gate_up_expanded_tensor_count
    )
    assert manifest["prepare_resident_fused_gate_up_expanded_bytes"] == (
        report.resident_pack.fused_gate_up_expanded_bytes
    )
    assert manifest["prepare_effective_unified_memory_bytes"] == 128 * 1024**3
    assert manifest["prepare_effective_unified_memory_source"] == "explicit"
    assert manifest["prepare_system_reserve_bytes"] == 24 * 1024**3
    assert manifest["prepare_cold_read_gib_per_second"] == 16.0
    assert manifest["prepare_cold_read_source"] == "explicit"
    assert manifest["prepare_cold_read_benchmark_path"] is None
    assert manifest["prepare_cold_read_benchmark_measured_bytes"] is None
    assert manifest["prepare_public_glm_5_2_shape_required"] is False
    assert manifest["prepare_public_glm_5_2_shape_matches"] is False
    assert "hidden_size" in manifest["prepare_public_glm_5_2_shape_mismatched_fields"]
    assert isinstance(manifest["prepare_hardware_chip_name"], str)
    assert "prepare_hardware_apple_silicon_generation" in manifest
    assert "prepare_hardware_apple_silicon_tier" in manifest
    prepared_manifest = load_prepared_manifest(output)
    assert prepared_manifest.model_config_sha256 == config_sha256(model)
    assert prepared_manifest.prepare_expert_pack_chunk_size_bytes == (
        report.expert_pack.chunk_size
    )
    assert prepared_manifest.prepare_expert_pack_estimated_peak_heap_bytes == (
        report.expert_pack.estimated_peak_heap_bytes
    )
    assert prepared_manifest.prepare_expert_pack_max_heap_bytes == (
        report.expert_pack.max_heap_bytes
    )
    assert prepared_manifest.prepare_raw_quantization_max_source_block_bytes == (
        report.expert_pack.raw_quantization_max_source_block_bytes
    )
    assert prepared_manifest.prepare_raw_quantization_max_output_block_bytes == (
        report.expert_pack.raw_quantization_max_output_block_bytes
    )
    assert prepared_manifest.prepare_raw_quantization_max_rows_per_block == (
        report.expert_pack.raw_quantization_max_rows_per_block
    )
    assert (
        prepared_manifest.prepare_resident_component_alias_source_tensor_count
        == report.resident_pack.component_alias_source_tensor_count
    )
    assert (
        prepared_manifest.prepare_resident_component_alias_renamed_tensor_count
        == report.resident_pack.component_alias_renamed_tensor_count
    )
    assert (
        prepared_manifest.prepare_resident_component_alias_bytes
        == report.resident_pack.component_alias_bytes
    )
    assert (
        prepared_manifest.prepare_resident_fused_gate_up_source_tensor_count
        == report.resident_pack.fused_gate_up_source_tensor_count
    )
    assert (
        prepared_manifest.prepare_resident_fused_gate_up_expanded_tensor_count
        == report.resident_pack.fused_gate_up_expanded_tensor_count
    )
    assert (
        prepared_manifest.prepare_resident_fused_gate_up_expanded_bytes
        == report.resident_pack.fused_gate_up_expanded_bytes
    )
    assert prepared_manifest.prepare_public_glm_5_2_shape_required is False
    assert prepared_manifest.prepare_public_glm_5_2_shape_matches is False
    assert report.glm_4bit_readiness is not None
    assert report.glm_4bit_readiness["ok"] is True
    assert report.glm_4bit_readiness["issues"] == []


def test_prepare_glm_execute_relative_output_manifest_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    model = Path("model")
    model.mkdir()
    _write_checkpoint(model)
    output = Path("prepared")

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        unified_memory_bytes=128 * 1024**3,
        cold_read_gib_per_second=16.0,
    )

    assert report.ok is True
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experts_layout"] == "experts/layout.json"
    assert manifest["resident_layout"] == "resident/layout.json"
    assert manifest["decode_cache_layout"] == "decode_cache_layout.json"
    assert manifest["decode_cache_file"] == "decode_cache.bin"
    prepared = load_prepared_manifest(output)
    assert prepared.experts_layout == output / "experts" / "layout.json"
    assert prepared.resident_layout == output / "resident" / "layout.json"
    assert prepared.decode_cache_layout == output / "decode_cache_layout.json"
    assert prepared.decode_cache_file == output / "decode_cache.bin"


def test_prepare_glm_execute_auto_benchmarks_packed_expert_read(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    report = prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        unified_memory_bytes=128 * 1024**3,
        auto_cold_read_benchmark=True,
        cold_read_benchmark_bytes=64,
        cold_read_benchmark_chunk_bytes=16,
    )

    assert report.ok is True
    assert report.executed is True
    assert report.cold_read_benchmark is not None
    benchmark = report.cold_read_benchmark
    assert benchmark.path == output / "experts" / "layer_001.bin"
    assert benchmark.requested_bytes == 64
    assert benchmark.measured_bytes == 64
    assert benchmark.chunk_bytes == 16
    assert benchmark.gib_per_second > 0
    assert benchmark.short_read is False

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_cold_read_source"] == "auto_benchmark"
    assert manifest["prepare_cold_read_gib_per_second"] == benchmark.gib_per_second
    assert (
        Path(manifest["prepare_cold_read_benchmark_path"])
        == output / "experts" / "layer_001.bin"
    )
    assert manifest["prepare_cold_read_benchmark_requested_bytes"] == 64
    assert manifest["prepare_cold_read_benchmark_measured_bytes"] == 64
    assert manifest["prepare_cold_read_benchmark_elapsed_seconds"] > 0

    prepared_manifest = load_prepared_manifest(output)
    assert prepared_manifest.prepare_cold_read_source == "auto_benchmark"
    assert (
        prepared_manifest.prepare_cold_read_gib_per_second == benchmark.gib_per_second
    )
    assert (
        prepared_manifest.prepare_cold_read_benchmark_path
        == str(output / "experts" / "layer_001.bin")
    )
    assert prepared_manifest.prepare_cold_read_benchmark_requested_bytes == 64
    assert prepared_manifest.prepare_cold_read_benchmark_measured_bytes == 64
    assert (
        prepared_manifest.prepare_cold_read_benchmark_elapsed_seconds
        == manifest["prepare_cold_read_benchmark_elapsed_seconds"]
    )


def test_prepare_glm_auto_cold_read_benchmark_requires_execute(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)

    with pytest.raises(PrepareError, match="auto_cold_read_benchmark requires"):
        prepare_glm_checkpoint(
            model,
            output_dir=tmp_path / "prepared",
            max_context_tokens=4,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            unified_memory_bytes=128 * 1024**3,
            auto_cold_read_benchmark=True,
        )


def test_prepare_glm_auto_cold_read_benchmark_conflicts_with_explicit_speed(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)

    with pytest.raises(PrepareError, match="conflicts with cold_read_gib_per_second"):
        prepare_glm_checkpoint(
            model,
            output_dir=tmp_path / "prepared",
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            unified_memory_bytes=128 * 1024**3,
            cold_read_gib_per_second=16.0,
            auto_cold_read_benchmark=True,
        )


def test_prepare_write_json_atomic_removes_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "manifest.json"

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace exploded")

    monkeypatch.setattr(type(path), "replace", fail_replace)

    with pytest.raises(OSError, match="replace exploded"):
        prepare_module._write_json_atomic(path, {"ok": True})

    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_prepare_glm_execute_validates_written_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    called: dict[str, Path] = {}

    def fake_load_prepared_manifest(path: str | Path) -> None:
        called["path"] = Path(path)
        raise PreparedManifestError("boom")

    monkeypatch.setattr(
        "largerlm.prepare.load_prepared_manifest",
        fake_load_prepared_manifest,
    )

    with pytest.raises(PrepareError, match="prepared output validation failed: boom"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert called["path"] == output / "manifest.json"
    assert not (output / "manifest.json").exists()
    assert not (output / "decode_cache.bin").exists()
    assert not (output / "decode_cache_layout.json").exists()
    assert not (output / "resident" / "resident.bin").exists()
    assert not (output / "experts" / "layer_001.bin").exists()


def test_prepare_glm_execute_validates_glm_4bit_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    def fail_readiness(prepared) -> dict[str, object]:
        raise PrepareError("prepared GLM 4bit readiness failed after prepare: boom")

    monkeypatch.setattr(
        prepare_module,
        "_validate_prepared_glm_4bit_output",
        fail_readiness,
    )

    with pytest.raises(
        PrepareError,
        match="prepared GLM 4bit readiness failed after prepare: boom",
    ):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert not (output / "manifest.json").exists()
    assert not (output / "decode_cache.bin").exists()
    assert not (output / "decode_cache_layout.json").exists()
    assert not (output / "resident" / "resident.bin").exists()
    assert not (output / "experts" / "layer_001.bin").exists()


def test_prepare_glm_execute_cleans_new_outputs_when_cache_layout_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    original_write_json_atomic = prepare_module._write_json_atomic

    def fail_cache_layout(path: Path, payload: object) -> None:
        if path.name == "decode_cache_layout.json":
            raise OSError("cache layout exploded")
        original_write_json_atomic(path, payload)

    monkeypatch.setattr(prepare_module, "_write_json_atomic", fail_cache_layout)

    with pytest.raises(OSError, match="cache layout exploded"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert not (output / "decode_cache_layout.json").exists()
    assert not (output / "decode_cache_layout.json.tmp").exists()
    assert not (output / "decode_cache.bin").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / "resident" / "resident.bin").exists()
    assert not (output / "experts" / "layer_001.bin").exists()


def test_prepare_glm_execute_cleans_new_outputs_when_manifest_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    original_write_json_atomic = prepare_module._write_json_atomic

    def fail_manifest(path: Path, payload: object) -> None:
        if path.name == "manifest.json":
            raise OSError("manifest exploded")
        original_write_json_atomic(path, payload)

    monkeypatch.setattr(prepare_module, "_write_json_atomic", fail_manifest)

    with pytest.raises(OSError, match="manifest exploded"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert not (output / "manifest.json").exists()
    assert not (output / "manifest.json.tmp").exists()
    assert not (output / "decode_cache.bin").exists()
    assert not (output / "decode_cache_layout.json").exists()
    assert not (output / "resident" / "resident.bin").exists()
    assert not (output / "experts" / "layer_001.bin").exists()


def test_prepare_glm_execute_cleans_new_outputs_when_later_step_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    real_pack_experts = prepare_module.pack_experts

    def fail_execute_pack_experts(*args, **kwargs):
        if kwargs.get("dry_run") is False:
            raise PrepareError("expert pack exploded")
        return real_pack_experts(*args, **kwargs)

    monkeypatch.setattr(prepare_module, "pack_experts", fail_execute_pack_experts)

    with pytest.raises(PrepareError, match="expert pack exploded"):
        prepare_glm_checkpoint(
            model,
            output_dir=output,
            max_context_tokens=4,
            execute=True,
            quantize_raw_to_int4=True,
            group_size=8,
            max_cache_bytes=1024 * 1024,
            disk_safety_margin_bytes=0,
            chunk_size=17,
            unified_memory_bytes=128 * 1024**3,
        )

    assert not (output / "resident" / "resident.bin").exists()
    assert not (output / "resident" / "layout.json").exists()
    assert not (output / "experts" / "layer_001.bin").exists()
    assert not (output / "decode_cache.bin").exists()


def test_prepare_glm_cli_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    report_path = tmp_path / "prepared" / "prepare-dry-run-report.json"

    status = cli_main(
        [
            "prepare-glm",
            str(model),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--max-context-tokens",
            "4",
            "--quantize-bf16-affine-int4",
            "--group-size",
            "8",
            "--max-cache-gib",
            "1",
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--write-report",
            str(report_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload == payload
    assert report_payload["executed"] is False
    assert report_payload["paths"]["output_dir"] == str(tmp_path / "prepared")
    assert report_payload["preflight"]["model_dir"] == str(model)
    assert not report_path.with_name(f"{report_path.name}.tmp").exists()
    public_shape = payload["public_glm_5_2_shape"]
    assert public_shape["matches"] is False
    assert "hidden_size" in public_shape["mismatched_fields"]


def test_prepare_glm_cli_applies_plan_prepare_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    flags = tmp_path / "prepare-flags.json"
    flags.write_text(
        json.dumps(
            {
                "source": "plan",
                "argv_safe_to_replay": True,
                "argv": [
                    "--auto-context-from-budget",
                    "--group-size",
                    "8",
                    "--max-cache-gib",
                    "0.000001",
                    "--system-reserve-gib",
                    "24",
                    "--runtime-buffer-gib",
                    "8",
                    "--page-cache-fraction",
                    "0.6",
                    "--unified-memory-gib",
                    "128",
                    "--cold-read-gib-s",
                    "16",
                ],
            }
        ),
        encoding="utf-8",
    )
    expected_sha256 = hashlib.sha256(flags.read_bytes()).hexdigest()

    status = cli_main(
        [
            "prepare-glm",
            "--apply-prepare-flags",
            str(flags),
            str(model),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--quantize-bf16-affine-int4",
            "--disk-margin-gib",
            "0",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_budget"]["auto_context_from_budget"] is True
    assert payload["cache_layout"]["max_context_tokens"] == 64
    assert payload["context_budget"]["decode_cache_budget_bytes"] == 1073
    assert payload["context_budget"]["decode_cache_safe_context_tokens"] == 67
    assert payload["preflight"]["cold_read_gib_per_second"] == 16.0
    assert payload["prepare_flags"] == {
        "source": "plan",
        "path": str(flags),
        "sha256": expected_sha256,
    }


def test_prepare_glm_cli_records_plan_prepare_flags_in_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    flags = tmp_path / "prepare-flags.json"
    flags.write_text(
        json.dumps(
            {
                "source": "plan",
                "argv_safe_to_replay": True,
                "argv": [
                    "--auto-context-from-budget",
                    "--group-size",
                    "8",
                    "--max-cache-gib",
                    "0.000001",
                    "--system-reserve-gib",
                    "24",
                    "--runtime-buffer-gib",
                    "8",
                    "--page-cache-fraction",
                    "0.6",
                    "--unified-memory-gib",
                    "128",
                    "--cold-read-gib-s",
                    "16",
                ],
            }
        ),
        encoding="utf-8",
    )
    expected_sha256 = hashlib.sha256(flags.read_bytes()).hexdigest()
    output = tmp_path / "prepared"

    status = cli_main(
        [
            "prepare-glm",
            "--apply-prepare-flags",
            str(flags),
            str(model),
            "--output-dir",
            str(output),
            "--quantize-bf16-affine-int4",
            "--disk-margin-gib",
            "0",
            "--execute",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["prepare_flags"]["sha256"] == expected_sha256

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_flags_applied"] is True
    assert manifest["prepare_flags_source"] == "plan"
    assert manifest["prepare_flags_path"] == str(flags)
    assert manifest["prepare_flags_sha256"] == expected_sha256

    prepared = load_prepared_manifest(output)
    assert prepared.prepare_flags_applied is True
    assert prepared.prepare_flags_source == "plan"
    assert prepared.prepare_flags_path == str(flags)
    assert prepared.prepare_flags_sha256 == expected_sha256


def test_prepare_glm_cli_rejects_non_plan_prepare_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    flags = tmp_path / "prepare-flags.json"
    flags.write_text(
        json.dumps(
            {
                "source": "unit",
                "argv_safe_to_replay": True,
                "argv": ["--auto-context-from-budget"],
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "prepare-glm",
            "--apply-prepare-flags",
            str(flags),
            str(model),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "prepare flags must have source='plan'" in captured.err


def test_prepare_glm_cli_require_public_glm_5_2_shape_rejects_non_public(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    status = cli_main(
        [
            "prepare-glm",
            str(model),
            "--output-dir",
            str(output),
            "--max-context-tokens",
            "4",
            "--quantize-bf16-affine-int4",
            "--require-public-glm-5-2-shape",
            "--group-size",
            "8",
            "--max-cache-gib",
            "1",
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "prepared config does not match the public GLM-5.2 shape" in captured.err
    assert "hidden_size" in captured.err
    assert not output.exists()


def test_prepare_glm_cli_require_public_glm_5_2_shape_rejects_non_4bit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        prepare_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )
    monkeypatch.setattr(
        preflight_module,
        "_public_glm_5_2_shape_report",
        lambda config: {"matches": True, "mismatched_fields": (), "checks": {}},
    )

    status = cli_main(
        [
            "prepare-glm",
            str(model),
            "--output-dir",
            str(output),
            "--max-context-tokens",
            "4",
            "--quantize-bf16-affine-int4",
            "--quant-bits",
            "8",
            "--require-public-glm-5-2-shape",
            "--group-size",
            "8",
            "--max-cache-gib",
            "1",
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--execute",
            "--json",
        ]
    )

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["expert_pack"] is None
    assert payload["resident_pack"] is None
    assert payload["cache_layout"] is None
    assert payload["preflight"]["checkpoint_tensor_count"] == 0
    issue = next(
        item
        for item in payload["preflight"]["issues"]
        if item["code"] == "public_glm_5_2_requires_4bit"
    )
    assert issue["severity"] == "error"
    assert not output.exists()


def test_prepare_glm_cli_execute_auto_cold_read_benchmark(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"

    status = cli_main(
        [
            "prepare-glm",
            str(model),
            "--output-dir",
            str(output),
            "--max-context-tokens",
            "4",
            "--quantize-bf16-affine-int4",
            "--group-size",
            "8",
            "--max-cache-gib",
            "1",
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--execute",
            "--auto-cold-read-benchmark",
            "--cold-read-benchmark-mib",
            "0.00006103515625",
            "--cold-read-benchmark-chunk-mib",
            "0.0000152587890625",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    payload = json.loads(captured.out)
    assert payload["cold_read_benchmark"]["requested_bytes"] == 64
    assert payload["cold_read_benchmark"]["chunk_bytes"] == 16
    assert payload["glm_4bit_readiness"]["ok"] is True
    assert payload["glm_4bit_readiness"]["issues"] == []
    assert payload["glm_4bit_readiness"]["decode_cache_layout_ok"] is True

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepare_cold_read_source"] == "auto_benchmark"
    assert manifest["prepare_cold_read_benchmark_requested_bytes"] == 64
    assert manifest["prepare_cold_read_benchmark_measured_bytes"] > 0


def test_prepare_glm_cli_rejects_non_finite_scaled_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)

    status = cli_main(
        [
            "prepare-glm",
            str(model),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--max-context-tokens",
            "4",
            "--quantize-bf16-affine-int4",
            "--group-size",
            "8",
            "--disk-margin-gib",
            "nan",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "--disk-margin-gib must be finite" in captured.err


def test_prepare_glm_cli_auto_context_from_budget(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)

    status = cli_main(
        [
            "prepare-glm",
            str(model),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--auto-context-from-budget",
            "--quantize-bf16-affine-int4",
            "--group-size",
            "8",
            "--max-cache-gib",
            "0.000001",
            "--disk-margin-gib",
            "0",
            "--unified-memory-gib",
            "128",
            "--json",
        ]
    )

    assert status == 0


def test_generate_prepared_commands_use_manifest_paths(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    runner = tmp_path / "runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
import shutil
import sys

def arg(name):
    i = sys.argv.index(name)
    return sys.argv[i + 1]

shutil.copyfile(arg("--input-f32"), arg("--output-f32"))
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        unified_memory_bytes=128 * 1024**3,
    )

    token_status = cli_main(
        [
            "generate-prepared-token-ids",
            str(output),
            "--runner",
            str(runner),
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--top-k",
            "1",
            "--max-k",
            "1",
            "--router-score",
            "raw",
            "--logits-top-k",
            "1",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--min-free-unified-memory-gib",
            "0",
            "--quiet-runner",
        ]
    )
    text_status = cli_main(
        [
            "generate-prepared-text",
            str(output / "manifest.json"),
            "--runner",
            str(runner),
            "--layers",
            "1",
            "--prompt",
            "A",
            "--max-new-tokens",
            "1",
            "--top-k",
            "1",
            "--max-k",
            "1",
            "--router-score",
            "raw",
            "--logits-top-k",
            "1",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--min-free-unified-memory-gib",
            "0",
            "--quiet-runner",
        ]
    )

    assert token_status == 0
    assert text_status == 0


def test_generate_prepared_text_cli_rejects_oversized_prompt_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("AB", encoding="utf-8")

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-file",
            str(prompt_file),
            "--max-prompt-bytes",
            "1",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prompt file" in err
    assert "exceeds --max-prompt-bytes" in err


def test_generate_prepared_token_ids_auto_enables_batch_prefill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--prefill-max-routed-read-amplification",
            "1.75",
            "--prefill-max-routed-read-gib",
            "0.5",
            "--prefill-ssd-read-gib-s",
            "16",
            "--prefill-max-routed-read-seconds",
            "5",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prompt_token_ids"] == (0, 1)
    assert captured["batch_prefill_prompt"] is True
    assert captured["prefill_static_capacity_per_expert"] == "auto"
    assert captured["prefill_max_routed_read_amplification"] == 1.75
    assert captured["prefill_max_routed_read_gib"] == 0.5
    assert captured["prefill_ssd_read_gib_per_second"] == 16.0
    assert captured["prefill_max_routed_read_seconds"] == 5.0
    assert captured["prefill_mpsgraph_min_batch_tokens"] == 64
    assert captured["prefill_mpsgraph_min_matrix_dim"] == 16


@pytest.mark.parametrize(
    ("extra_args", "expected_runtime_preflight"),
    [
        ([], True),
        (["--no-runtime-preflight"], False),
    ],
)
def test_generate_prepared_token_ids_passes_runtime_preflight_to_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_runtime_preflight: bool,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    def fake_admission(*args, **kwargs):
        del args
        captured["runtime_preflight"] = kwargs.get("runtime_preflight")
        return None

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        fake_admission,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
            *extra_args,
        ]
    )

    assert status == 0
    assert captured["runtime_preflight"] is expected_runtime_preflight


def test_generate_prepared_token_ids_writes_schema_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    result_path = tmp_path / "minimal-smoke.json"

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--write-result",
            str(result_path),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "largerlm.prepared_token_generation_result.v1"
    assert payload["source"] == "generate_prepared_token_ids"
    assert payload["prepared_manifest"] == str(prepared / "manifest.json")
    assert payload["request"] == {"prompt_token_ids": [0], "max_new_tokens": 1}
    assert payload["token_result"]["prompt_token_ids"] == [0]
    assert payload["token_result"]["generated_token_ids"] == [2]
    assert payload["token_result"] == stdout_payload


def test_generate_prepared_token_ids_records_prefill_mla_kv_b_cache_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    cache_dir = tmp_path / "mla-kv-b-cache"
    cache_dir.mkdir()
    (cache_dir / "layer-0-kvb-f32.bin").write_bytes(b"0" * 16)
    result_path = tmp_path / "minimal-smoke.json"

    def fake_generate_token_ids(**kwargs):
        assert kwargs["prefill_mla_kv_b_cache_dir"] == str(cache_dir)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--prefill-mla-kv-b-cache-dir",
            str(cache_dir),
            "--write-result",
            str(result_path),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    capsys.readouterr()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["request"] == {
        "prompt_token_ids": [0],
        "max_new_tokens": 1,
        "prefill_mla_kv_b_cache_dir": str(cache_dir),
        "prefill_mla_kv_b_cache_file_count": 1,
        "prefill_mla_kv_b_cache_total_bytes": 16,
    }


def test_generate_prepared_text_writes_schema_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    result_path = tmp_path / "minimal-text-smoke.json"

    def fake_generate_text(**kwargs):
        token_result = TokenGenerationResult(
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="C",
            full_text=kwargs["prompt"] + "C",
            tokenizer_backend="simple",
            tokenizer_path=tmp_path / "model" / "simple_tokenizer.json",
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.cli.generate_text", fake_generate_text)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--tokenizer-backend",
            "simple",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            "--write-result",
            str(result_path),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "largerlm.prepared_text_generation_result.v1"
    assert payload["source"] == "generate_prepared_text"
    assert payload["prepared_manifest"] == str(prepared / "manifest.json")
    assert payload["request"] == {"prompt_token_ids": [0, 1], "max_new_tokens": 1}
    assert payload["text_result"]["prompt"] == "AB"
    assert payload["text_result"]["generated_text"] == "C"
    assert payload["text_result"]["token_result"]["generated_token_ids"] == [2]
    assert payload["text_result"] == stdout_payload


def test_generate_prepared_token_ids_runs_request_admission_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--prefill-prompt-chunk-tokens",
            "999",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared request admission failed" in err
    assert "prefill_prompt_chunk_tokens 999 exceeds safety-capped maximum" in err


def test_inspect_prepared_cli_rejected_prefill_chunk_reports_chunk_plan_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "2",
            "--check-max-new-tokens",
            "1",
            "--prefill-prompt-chunk-tokens",
            "999",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    check = health["request_check"]
    assert check["ok"] is False
    assert (
        "prefill_prompt_chunk_tokens 999 exceeds safety-capped maximum"
        in check["error"]
    )
    chunk = check["prefill_prompt_chunk_tokens"]
    assert chunk["configured"] == 999
    assert chunk["resolved"] == 999
    assert chunk["max_safe"] >= 1
    plan = check["prefill_prompt_chunk_plan"]
    assert plan["source"] == "prepared_request_check"
    assert plan["configured_is_auto"] is False
    assert plan["auto"] is None
    assert plan["max_safe"]["chunk_tokens"] == chunk["max_safe"]
    assert "mpp_tensor_ops_candidate_blocking_cap_names" in plan["max_safe"]
    assert "mpp_tensor_ops_candidate_blocking_cap_summary" in plan["max_safe"]


def test_generate_prepared_token_ids_rejects_nonpassing_request_admission_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fake_inspect_token_request(self, **kwargs):
        del self, kwargs
        return {
            "ok": False,
            "reason": "synthetic live-memory guard failure",
        }

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        fake_inspect_token_request,
    )
    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared request admission failed" in err
    assert "synthetic live-memory guard failure" in err


def test_generate_prepared_token_ids_rejects_concurrent_prepared_generation_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    generated = False

    def fail_generate_token_ids(**kwargs):
        del kwargs
        nonlocal generated
        generated = True
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )
    lock_path = prepared / ".largerlm-selected-replay.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        status = cli_main(
            [
                "generate-prepared-token-ids",
                str(prepared),
                "--runner",
                "unused-runner",
                "--prompt-token-ids",
                "0",
                "--max-new-tokens",
                "1",
                "--quiet-runner",
                "--json",
            ]
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert status == 1
    assert generated is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "another prepared generation is already running" in captured.err
    assert str(lock_path) in captured.err


def test_generate_prepared_token_ids_allows_inherited_prepared_run_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    generated = False

    def fake_generate_token_ids(**kwargs):
        nonlocal generated
        generated = True
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )
    lock_path = prepared / ".largerlm-selected-replay.lock"
    monkeypatch.setenv(
        "LARGERLM_PREPARED_RUN_LOCK_PATH",
        str(lock_path.resolve()),
    )
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        status = cli_main(
            [
                "generate-prepared-token-ids",
                str(prepared),
                "--runner",
                "unused-runner",
                "--prompt-token-ids",
                "0",
                "--max-new-tokens",
                "1",
                "--quiet-runner",
                "--json",
            ]
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert status == 0
    assert generated is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["generated_token_ids"] == [2]


def test_prepared_server_rejects_concurrent_prepared_generation_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    generated = False

    def fail_generate_token_ids(**kwargs):
        del kwargs
        nonlocal generated
        generated = True
        raise AssertionError("server generation should not be called")

    monkeypatch.setattr("largerlm.server.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        lambda *args, **kwargs: {"ok": True},
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )
    lock_path = prepared / ".largerlm-selected-replay.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        with pytest.raises(
            PreparedServerError,
            match="another prepared generation is already running",
        ):
            app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert generated is False


def test_prepared_server_health_reports_available_prepared_run_lock(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    health = app.health()

    assert health["prepared_run_lock"] == {
        "path": str(prepared / ".largerlm-selected-replay.lock"),
        "available": True,
        "busy": False,
        "inherited_lock_marker_matches": False,
        "error": None,
    }


def test_prepared_server_health_reports_busy_prepared_run_lock(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )
    lock_path = prepared / ".largerlm-selected-replay.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        health = app.health()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert health["prepared_run_lock"] == {
        "path": str(lock_path),
        "available": False,
        "busy": True,
        "inherited_lock_marker_matches": False,
        "error": None,
    }


def test_prepared_server_health_reports_prefill_acceleration_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            prefill_linear_backend="mpsgraph-f32",
            require_prefill_acceleration=True,
            prefill_run_mpsgraph_probe=True,
        )
    )

    health = app.health()

    gate = health["prefill_acceleration_requirement"]
    assert gate["required"] is True
    assert gate["ok"] is True
    assert gate["configured_backend"] == "mpsgraph-f32"
    assert gate["mps_graph_probe_requested"] is True
    assert gate["mps_graph_probe_ok"] is True
    assert gate["reason_code"] == "ok"


def test_prepared_server_token_response_reports_request_admission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    request_check = {
        "ok": True,
        "source": "unit",
        "prompt_token_count": 1,
        "max_new_tokens": 1,
    }
    launch_audit_envelope = {
        "schema": "unit.launch_audit_envelope.v1",
        "server_caps_within_envelope": True,
    }

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.server.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        lambda *args, **kwargs: request_check,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            applied_launch_profile={
                "source": "unit",
                "argv_safe_to_replay": True,
                "locked": True,
            },
            launch_audit_envelope=launch_audit_envelope,
        )
    )

    payload = app.generate_token_ids({"prompt_token_ids": [0], "max_new_tokens": 1})

    assert payload["generated_token_ids"] == [2]
    assert payload["request_check"] == request_check
    assert payload["launch_audit_envelope"] == launch_audit_envelope
    assert payload["applied_launch_profile"]["locked"] is True


def test_prepared_server_text_response_reports_request_admission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    request_check = {
        "ok": True,
        "source": "unit",
        "prompt_token_count": 1,
        "max_new_tokens": 1,
    }
    launch_audit_envelope = {
        "schema": "unit.launch_audit_envelope.v1",
        "server_caps_within_envelope": True,
    }

    class FakeTokenizer:
        backend = "unit-tokenizer"
        path = tmp_path

        def encode(self, prompt, *, add_special_tokens=True):
            del prompt, add_special_tokens
            return (0,)

    def fake_generate_text(**kwargs):
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="B",
            full_text=f"{kwargs['prompt']}B",
            tokenizer_backend="unit-tokenizer",
            tokenizer_path=tmp_path,
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=TokenGenerationResult(
                prompt_token_ids=(0,),
                generated_token_ids=(2,),
                steps=(),
                work_dir=tmp_path,
                kept_work_dir=False,
                max_context_tokens=4,
                sampling_temperature=0.0,
                sampling_top_p=1.0,
            ),
        )

    monkeypatch.setattr("largerlm.server.load_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr("largerlm.server.generate_text", fake_generate_text)
    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        lambda *args, **kwargs: request_check,
    )
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
            applied_launch_profile={
                "source": "unit",
                "argv_safe_to_replay": True,
                "locked": True,
            },
            launch_audit_envelope=launch_audit_envelope,
        )
    )

    payload = app.generate_text({"prompt": "A", "max_new_tokens": 1})

    assert payload["generated_text"] == "B"
    assert payload["request_check"] == request_check
    assert payload["launch_audit_envelope"] == launch_audit_envelope
    assert payload["token_result"]["request_check"] == request_check
    assert payload["token_result"]["launch_audit_envelope"] == launch_audit_envelope
    assert payload["applied_launch_profile"]["locked"] is True


def test_prepared_server_openai_completion_reports_request_admission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    request_check = {
        "ok": True,
        "source": "unit",
        "prompt_token_count": 1,
        "max_new_tokens": 1,
    }
    launch_audit_envelope = {
        "schema": "unit.launch_audit_envelope.v1",
        "server_caps_within_envelope": True,
    }
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    def fake_generate_text(payload):
        assert payload["max_new_tokens"] == 1
        return {
            "generated_text": "B",
            "full_text": f"{payload['prompt']}B",
            "generated_token_ids": [2],
            "prompt_token_ids": [0],
            "tokenizer_backend": "unit-tokenizer",
            "token_result": {"generated_token_ids": [2]},
            "applied_launch_profile": {"source": "unit", "locked": True},
            "request_check": request_check,
            "launch_audit_envelope": launch_audit_envelope,
        }

    monkeypatch.setattr(app, "generate_text", fake_generate_text)

    response = app.openai_completion(
        {"model": "unit-model", "prompt": "A", "max_tokens": 1}
    )

    assert response["choices"][0]["text"] == "B"
    assert response["largerlm"]["request_check"] == request_check
    assert response["largerlm"]["launch_audit_envelope"] == launch_audit_envelope
    assert response["largerlm"]["applied_launch_profile"]["locked"] is True


def test_prepared_server_openai_chat_reports_request_admission_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    request_check = {
        "ok": True,
        "source": "unit",
        "prompt_token_count": 1,
        "max_new_tokens": 1,
    }
    launch_audit_envelope = {
        "schema": "unit.launch_audit_envelope.v1",
        "server_caps_within_envelope": True,
    }
    app = PreparedGenerationApp(
        PreparedServerConfig(
            prepared_path=prepared,
            runner_path=Path("unused-runner"),
        )
    )

    monkeypatch.setattr(
        "largerlm.server.render_chat_prompt",
        lambda *args, **kwargs: RenderedChatPrompt(
            text="A",
            backend="unit-template",
            tokenizer_path=tmp_path,
        ),
    )

    def fake_generate_text(payload):
        assert payload["prompt"] == "A"
        assert payload["add_special_tokens"] is False
        return {
            "generated_text": "B",
            "full_text": "AB",
            "generated_token_ids": [2],
            "prompt_token_ids": [0],
            "tokenizer_backend": "unit-tokenizer",
            "token_result": {"generated_token_ids": [2]},
            "applied_launch_profile": {"source": "unit", "locked": True},
            "request_check": request_check,
            "launch_audit_envelope": launch_audit_envelope,
        }

    monkeypatch.setattr(app, "generate_text", fake_generate_text)

    response = app.openai_chat_completion(
        {
            "model": "unit-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
    )

    assert response["choices"][0]["message"]["content"] == "B"
    assert response["largerlm"]["chat_template_backend"] == "unit-template"
    assert response["largerlm"]["request_check"] == request_check
    assert response["largerlm"]["launch_audit_envelope"] == launch_audit_envelope
    assert response["largerlm"]["applied_launch_profile"]["locked"] is True


def test_benchmark_prepared_token_ids_rejects_concurrent_prepared_generation_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    generated = False

    def fail_generate_token_ids(**kwargs):
        del kwargs
        nonlocal generated
        generated = True
        raise AssertionError("benchmark generation should not be called")

    monkeypatch.setattr(
        "largerlm.benchmark.generate_token_ids",
        fail_generate_token_ids,
    )
    lock_path = prepared / ".largerlm-selected-replay.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        with pytest.raises(
            BenchmarkError,
            match="another prepared generation is already running",
        ):
            benchmark_prepared_token_ids(
                prepared,
                runner_path="unused-runner",
                prompt_token_ids=[0],
                max_new_tokens=1,
                preflight_runtime=False,
                batch_prefill_prompt=False,
            )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert generated is False


def test_generate_prepared_text_runs_request_admission_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)

    def fail_generate_text(**kwargs):
        del kwargs
        raise AssertionError("generate_text should not be called")

    monkeypatch.setattr("largerlm.cli.generate_text", fail_generate_text)

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            "--prefill-prompt-chunk-tokens",
            "999",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared request admission failed" in err
    assert "prefill_prompt_chunk_tokens 999 exceeds safety-capped maximum" in err


def test_generate_prepared_text_requires_tokenization_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_text(**kwargs):
        del kwargs
        raise AssertionError("generate_text should not be called")

    monkeypatch.setattr("largerlm.cli.generate_text", fail_generate_text)

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--tokenizer-backend",
            "simple",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared text prompt tokenization failed before request admission" in err


def test_generate_prepared_token_ids_rejects_prefill_chunk_plan_drift_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    _write_simple_model_tokenizer(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--prefill-prompt-chunk-tokens", "2"],
        sections={
            "prefill_prompt_chunk_plan": {
                "source": "prepared_request_check",
                "max_safe": {
                    "chunk_tokens": 999,
                    "limiting_cap_names": ["old_machine"],
                },
            }
        },
    )

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared request admission failed" in err
    assert "prefill chunk-plan drift current_max_safe_below_profile" in err
    assert "profile_max_safe=999" in err
    assert "current_max_safe=2" in err


def test_generate_prepared_token_ids_admission_uses_generation_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    cfg_path = load_prepared_manifest(prepared).model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["num_experts_per_tok"] = 1
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--top-k",
            "2",
            "--max-k",
            "2",
            "--prefill-prompt-chunk-tokens",
            "1",
            "--prefill-max-routed-read-amplification",
            "1.5",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared request admission failed" in err
    assert "read amplification 2" in err
    assert "exceeds cap 1.5" in err


def test_generate_prepared_token_ids_applies_launch_profile_before_overrides(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_runtime_guard_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 64 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 16 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 4 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=48 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    profile_argv = [
        "--max-live-working-set-mib",
        "8192",
        "--min-free-unified-memory-gib",
        "12",
        "--require-prepared-memory-profile",
        "--prefill-prompt-chunk-tokens",
        "2",
        "--prefill-max-routed-read-amplification",
        "1.5",
        "--prefill-max-routed-read-gib",
        "0.75",
        "--prefill-ssd-read-gib-s",
        "16",
        "--prefill-max-routed-read-seconds",
        "3",
        "--compile-mpp-probe",
        "--prefill-mpsgraph-min-batch-tokens",
        "64",
        "--prefill-mpsgraph-min-matrix-dim",
        "16",
        "--prefill-router-hybrid-margin-threshold",
        "1e-05",
        "--prefill-max-stage-mib",
        "6",
        "--prefill-max-compact-stage-mib",
        "5",
        "--prefill-copy-chunk-mib",
        "32",
        "--decode-mla-key-cache",
        "--decode-max-routed-read-gib-per-token",
        "0.25",
        "--decode-max-routed-read-seconds-per-token",
        "0.02",
    ]
    prompt_chunk_plan = {
        "source": "prepared_request_check",
        "configured_is_auto": True,
        "auto": {
            "chunk_tokens": 2,
            "limiting_cap_names": ["prompt_batch_bytes"],
        },
        "max_safe": {
            "chunk_tokens": 2,
            "next_token_matrix_scratch_bytes": 2 * 1024 * 1024,
        },
    }
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=profile_argv,
        sections={"prefill_prompt_chunk_plan": prompt_chunk_plan},
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--prefill-max-routed-read-gib",
            "0.5",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["max_live_working_set_mib"] == 8192.0
    assert captured["min_free_unified_memory_gib"] == 12.0
    assert captured["prefill_prompt_chunk_tokens"] == 2
    assert captured["prefill_max_routed_read_amplification"] == 1.5
    assert captured["prefill_max_routed_read_gib"] == 0.5
    assert captured["prefill_ssd_read_gib_per_second"] == 16.0
    assert captured["prefill_max_routed_read_seconds"] == 3.0
    assert captured["prefill_max_stage_mib"] == 6.0
    assert captured["prefill_max_compact_stage_mib"] == 5.0
    assert captured["prefill_copy_chunk_mib"] == 32.0
    assert captured["prefill_mpsgraph_min_batch_tokens"] == 64
    assert captured["prefill_mpsgraph_min_matrix_dim"] == 16
    assert captured["prefill_router_hybrid_margin_threshold"] == 1e-5
    assert captured["decode_mla_key_cache"] is True
    assert captured["decode_max_routed_read_gib_per_token"] == 0.25
    assert captured["decode_max_routed_read_seconds_per_token"] == 0.02
    applied = payload["applied_launch_profile"]
    assert applied["path"] == str(profile)
    assert applied["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert applied["source"] == "unit"
    assert applied["argv"] == profile_argv
    assert applied["locked"] is False
    assert applied["lock_required"] is False
    assert applied["profile_flag_count"] == 18
    assert applied["lock_checked_flags"] == []
    assert applied["section_names"] == ["prefill_prompt_chunk_plan"]
    assert applied["prefill_prompt_chunk_plan"] == prompt_chunk_plan
    assert applied["matches_prepared"] is True
    drift = payload["prefill_prompt_chunk_plan_drift"]
    assert drift["source"] == "applied_launch_profile"
    assert drift["status"] == "missing_actual_max_safe_plan"
    assert drift["profile_max_safe_chunk_tokens"] == 2
    assert (
        applied["prepared"]["expert_layout_bytes"]
        == load_prepared_manifest(prepared).expert_layout_bytes
    )
    assert applied["current_prepared_manifest"] == str(prepared / "manifest.json")


def test_generate_prepared_token_ids_lock_launch_profile_rejects_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--min-free-unified-memory-gib",
            "12",
            "--prefill-max-routed-read-gib",
            "0.75",
        ],
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-locked-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--min-free-unified-memory-gib",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch profile is locked" in err
    assert "--min-free-unified-memory-gib" in err


def test_generate_prepared_token_ids_apply_launch_profile_enables_metal_final_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--metal-final-logits"],
        sections={
            "final_logits_flags": {
                "source": "prepared_request_check",
                "metal_final_logits": True,
                "argv": ["--metal-final-logits"],
            },
        },
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["metal_final_logits"] is True
    applied = payload["applied_launch_profile"]
    assert applied["locked"] is True
    assert "--metal-final-logits" in applied["argv"]
    assert applied["lock_checked_flags"] == ["--metal-final-logits"]


def test_generate_prepared_token_ids_require_locked_launch_profile_rejects_unlocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--min-free-unified-memory-gib",
            "12",
        ],
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--require-locked-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    assert (
        "--require-locked-launch-profile requires --lock-launch-profile"
        in capsys.readouterr().err
    )


def test_generate_prepared_token_ids_accepts_required_launch_audit_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prompt_token_ids"] == (0, 1)
    assert payload["applied_launch_profile"]["locked"] is True


def test_generate_prepared_token_ids_launch_audit_target_allows_json_list_tuple_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_public_glm_5_2_shape_required"] = True
    manifest["prepare_public_glm_5_2_shape_matches"] = True
    manifest["prepare_public_glm_5_2_shape_mismatched_fields"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0


def test_generate_prepared_token_ids_accepts_benchmark_audit_prefill_read_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    actual_read_time = _launch_audit_prefill_actual_read_time(prompt_token_count=2)
    actual_linear_backend = _launch_audit_prefill_actual_linear_backend()
    decode_read_time = _launch_audit_decode_actual_read_time()
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": actual_read_time,
            "prefill_actual_linear_backend": actual_linear_backend,
            "decode_actual_read_time": decode_read_time,
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prompt_token_ids"] == (0, 1)
    applied = payload["applied_launch_profile"]
    assert applied["source"] == "benchmark_actual"
    assert applied["prefill_actual_read_time"] == actual_read_time
    assert applied["prefill_actual_linear_backend"] == actual_linear_backend
    assert applied["decode_actual_read_time"] == decode_read_time


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_prefill_copy_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    actual_read_time = _launch_audit_prefill_actual_read_time(prompt_token_count=2)
    actual_read_time["total_expert_stage_copy_elapsed_seconds"] = 0.1
    actual_read_time["total_expert_stage_copy_throughput_gib_per_second"] = -1.0
    actual_read_time["total_expert_stage_copy_seconds_ok"] = False
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": actual_read_time,
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit applied prefill read-time evidence is not passing" in err
    assert "total_expert_stage_copy_throughput_gib_per_second=-1.0" in err
    assert "total_expert_stage_copy_seconds_ok=False" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_prefill_range_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    actual_read_time = _launch_audit_prefill_actual_read_time(prompt_token_count=2)
    actual_read_time["prefill_max_stage_raw_ranges"] = 1
    actual_read_time["max_expert_stage_raw_ranges"] = 2
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": actual_read_time,
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit applied prefill read-time evidence is not passing" in err
    assert "max_expert_stage_raw_ranges=2 exceeds cap=1" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_actual_acceleration_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": (
                _launch_audit_prefill_actual_read_time(prompt_token_count=2)
            ),
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_accel_check = next(
        item
        for item in payload["launch_audit"]["checks"]
        if item["code"] == "applied_prefill_actual_acceleration_coverage_valid"
    )
    actual_accel_check["required"] = True
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "missing prefill_actual_acceleration_coverage evidence" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_prefill_linear_backend_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    actual_read_time = _launch_audit_prefill_actual_read_time(prompt_token_count=2)
    actual_linear_backend = _launch_audit_prefill_actual_linear_backend()
    actual_linear_backend["linear_backend_elapsed_seconds"] = {
        "mpsgraph-f32": -0.1
    }
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": actual_read_time,
            "prefill_actual_linear_backend": actual_linear_backend,
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit applied prefill linear-backend evidence is malformed"
        in err
    )
    assert "linear_backend_elapsed_seconds['mpsgraph-f32']=-0.1" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_mismatched_prefill_actual_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    actual_read_time = _launch_audit_prefill_actual_read_time(prompt_token_count=2)
    actual_coverage = _launch_audit_prefill_acceleration_coverage()
    actual_coverage["accelerated_matrix_count"] = 0
    actual_coverage["mpsgraph_matrix_count"] = 0
    actual_coverage["custom_metal_matrix_count"] = 2
    actual_coverage["accelerated_estimated_flops"] = 0
    actual_coverage["custom_metal_estimated_flops"] = 4096
    actual_coverage["accelerated_flop_fraction"] = 0.0
    actual_coverage["dominant_resident_flops_accelerated"] = False
    actual_coverage["accelerated_backends"] = ()
    actual_coverage["any_resident_matrix_accelerated"] = False
    actual_coverage["all_resident_matrices_accelerated"] = False
    actual_coverage["reason"] = "no resident prefill matrices used an accelerated backend"
    actual_linear_backend = _launch_audit_prefill_actual_linear_backend()
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": actual_read_time,
            "prefill_actual_acceleration_coverage": actual_coverage,
            "prefill_actual_linear_backend": actual_linear_backend,
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit applied prefill acceleration coverage does not match "
        "prefill linear-backend evidence"
    ) in err
    assert "accelerated_estimated_flops=0 expected_from_actual_linear=4096" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_prefill_acceleration_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": _launch_audit_prefill_actual_read_time(
                prompt_token_count=2
            ),
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_payload["request_check"]["prefill_acceleration_coverage"][
        "accelerated_flop_fraction"
    ] = 0.5
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit request prefill acceleration coverage evidence is malformed"
        in err
    )
    assert "accelerated_flop_fraction=0.5 expected=1.0" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_empty_required_prefill_acceleration_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": _launch_audit_prefill_actual_read_time(
                prompt_token_count=2
            ),
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    coverage = audit_payload["request_check"]["prefill_acceleration_coverage"]
    coverage["required"] = True
    coverage["accelerated_matrix_count"] = 0
    coverage["mpsgraph_matrix_count"] = 0
    coverage["custom_metal_matrix_count"] = 2
    coverage["accelerated_estimated_flops"] = 0
    coverage["custom_metal_estimated_flops"] = 4096
    coverage["accelerated_flop_fraction"] = 0.0
    coverage["dominant_resident_flops_accelerated"] = False
    coverage["accelerated_backends"] = ()
    coverage["any_resident_matrix_accelerated"] = False
    coverage["all_resident_matrices_accelerated"] = False
    coverage["reason"] = "no resident prefill matrices resolved to an accelerated backend"
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit request prefill acceleration coverage evidence is malformed"
        in err
    )
    assert (
        "ok=True requires any_resident_matrix_accelerated=True when required=True"
        in err
    )
    assert "ok=True requires accelerated_matrix_count>0 when required=True" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_mpp_candidate_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": _launch_audit_prefill_actual_read_time(
                prompt_token_count=2
            ),
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    coverage = audit_payload["request_check"]["prefill_acceleration_coverage"]
    coverage["mpp_tensor_ops_candidate_matrix_count"] = 1
    coverage["mpp_tensor_ops_candidate_estimated_flops"] = 2048
    coverage["mpp_tensor_ops_candidate_flop_fraction"] = 0.0
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit request prefill acceleration coverage evidence is malformed"
        in err
    )
    assert "mpp_tensor_ops_candidate_flop_fraction=0.0 expected=0.5" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_benchmark_prefill_read_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "applied benchmark profile is missing prefill_actual_read_time" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_benchmark_decode_read_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": _launch_audit_prefill_actual_read_time(
                prompt_token_count=2
            ),
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["applied_launch_profile"]["decode_actual_read_time"][
        "actual_decode_routed_read_bytes_ok"
    ] = False
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit applied decode read-time evidence is not passing" in err
    assert "actual_decode_routed_read_bytes_ok=False" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_decode_actual_bytes_over_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        source="benchmark_actual",
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prefill_actual_read_time": _launch_audit_prefill_actual_read_time(
                prompt_token_count=2
            ),
            "decode_actual_read_time": _launch_audit_decode_actual_read_time(),
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["applied_launch_profile"]["decode_actual_read_time"][
        "actual_decode_routed_read_bytes"
    ] = 513
    payload["applied_launch_profile"]["decode_actual_read_time"][
        "actual_decode_routed_read_bytes_ok"
    ] = True
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit applied decode read-time evidence is not passing" in err
    assert "actual_decode_routed_read_bytes=513 exceeds planned=512" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_weak_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    weak_target = prepared_launch_profile_target(load_prepared_manifest(prepared))
    assert weak_target["identity_strength"] == "weak"
    assert "expert_quantization is unavailable for profile matching" in (
        weak_target["identity_warnings"]
    )
    assert "expert_group_size is unavailable for profile matching" in (
        weak_target["identity_warnings"]
    )
    profile = tmp_path / "launch-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "unit",
                "argv_safe_to_replay": True,
                "prepared": weak_target,
                "argv": ["--min-free-unified-memory-gib", "0"],
            }
        ),
        encoding="utf-8",
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit prepared identity is weak" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_probe_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit = payload["launch_audit"]
    audit["checks"] = [
        check
        for check in audit["checks"]
        if check["code"] != "prefill_acceleration_probe_ok"
    ]
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit is missing required checks" in err
    assert "prefill_acceleration_probe_ok" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_storage_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    audit = payload["launch_audit"]
    audit["checks"] = [
        check
        for check in audit["checks"]
        if check["code"] != "prepared_storage_validated"
    ]
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit is missing required checks" in err
    assert "prepared_storage_validated" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_prepare_pack_heap_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_expert_pack_estimated_peak_heap_bytes"] += 4096
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(
                prompt_token_count=2,
                max_new_tokens=1,
            ),
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch profile does not match this prepared package" in err
    assert "prepare_expert_pack_estimated_peak_heap_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_resident_alias_rewrite_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_resident_component_alias_source_tensor_count"] = 3
    manifest["prepare_resident_component_alias_renamed_tensor_count"] = 3
    manifest["prepare_resident_component_alias_bytes"] = 6
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    check = next(
        item
        for item in audit["launch_audit"]["checks"]
        if item["code"] == "prepare_resident_alias_rewrite_ok"
    )
    check["prepare_resident_component_alias_bytes"] = 4
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(
                prompt_token_count=2,
                max_new_tokens=1,
            ),
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "resident alias rewrite evidence does not match" in err
    assert "prepare_resident_component_alias_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_runtime_profile_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 24 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 8 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_min_free_unified_memory_bytes"] = 20 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(
                prompt_token_count=2,
                max_new_tokens=1,
            ),
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile envelope does not match" in err
    assert "prepared_recommended_min_free_unified_memory_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_prepared_ssd_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_cold_read_gib_per_second"] = 16.0
    manifest["prepare_cold_read_source"] = "explicit"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
        sections={
            "prepared_ssd_read_flags": {
                "source": "prepared_manifest",
                "prefill_ssd_read_gib_per_second": 16.0,
                "prepare_cold_read_gib_per_second": 16.0,
                "prepare_cold_read_source": "explicit",
                "matches_prepare_cold_read": True,
                "argv": ["--prefill-ssd-read-gib-s", "16"],
            }
        },
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_cold_read_gib_per_second"] = 12.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(
                prompt_token_count=2,
                max_new_tokens=1,
            ),
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared SSD read-speed envelope does not match" in err
    assert "prepare_cold_read_gib_per_second" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_prefill_acceleration_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    acceleration_check.pop("prefill_neural_accelerator_status")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit prefill acceleration evidence is missing required fields"
        in err
    )
    assert "prefill_neural_accelerator_status" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_prefill_probe_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    acceleration_check.pop("host_probe_path")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit prefill acceleration evidence is missing required fields"
        in err
    )
    assert "host_probe_path" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_validated_prefill_acceleration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    acceleration_check["validated_accelerated_prefill_backends"] = ()
    acceleration_check["validated_prefill_acceleration_available"] = False
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "has no validated accelerated backend" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_mpp_run_probe_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    neural_status = acceleration_check["prefill_neural_accelerator_status"]
    neural_status.update(
        {
            "status": "runtime_executed_not_selectable",
            "runtime_visible": True,
            "mpp_run_probe_requested": True,
            "mpp_run_probe_ran": True,
            "mpp_run_probe_ok": True,
        }
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit prefill neural accelerator MPP run probe evidence "
        "is missing required fields"
    ) in err
    assert "mpp_run_probe_shape" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_mpp_run_probe_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    neural_status = acceleration_check["prefill_neural_accelerator_status"]
    neural_status.update(
        {
            "status": "runtime_executed_not_selectable",
            "runtime_visible": True,
            "mpp_run_probe_requested": True,
            "mpp_run_probe_ran": True,
            "mpp_run_probe_ok": True,
            "mpp_run_probe_kernel_variant": "metal_mpp",
            "mpp_run_probe_shape": "32x32x32",
            "mpp_run_probe_dtype": "float",
            "mpp_run_probe_execution_path": "mpp::tensor_ops::matmul2d",
        }
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert (
        "launch audit prefill neural accelerator MPP run probe evidence "
        "is not passing"
    ) in err
    assert "mpp_run_probe_dtype" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_prefill_acceleration_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    acceleration_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    acceleration_check["mps_graph_probe_ok"] = False
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit MPSGraph prefill evidence is not passing" in err
    assert "mps_graph_probe_ok=False" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_request_routed_read_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_check"].pop("prefill_routed_expert_read")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request is missing prefill_routed_expert_read evidence" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_request_routed_read_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    routed = payload["request_check"]["prefill_routed_expert_read"]
    routed["within_seconds_limit"] = False
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request routed-read evidence is not passing" in err
    assert "within_seconds_limit=False" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_decode_routed_read_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_check"].pop("decode_routed_expert_read")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request is missing decode_routed_expert_read evidence" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_runtime_preflight_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_check"].pop("runtime_preflight")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request is missing runtime_preflight evidence" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_runtime_preflight_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    runtime = payload["request_check"]["runtime_preflight"]
    runtime["system_available_memory_bytes"] = runtime["required_available_memory_bytes"] - 1
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit runtime preflight evidence is not passing" in err
    assert "system_available_memory_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_prefill_live_memory_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    live = payload["request_check"]["runtime_preflight"]["prefill_live_memory"]
    live["estimated_live_working_set_bytes"] = 123
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit runtime preflight evidence is not passing" in err
    assert "prefill_live_memory.estimated_live_working_set_bytes=123" in err
    assert "expected=2048" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_prefill_cache_io_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    cache_io = payload["request_check"]["prefill_cache_io"]
    cache_io["total_cache_read_bytes"] = 25
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request prefill cache I/O evidence is malformed" in err
    assert "total_cache_read_bytes=25 expected=24" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_bad_routed_chunk_frontier_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    frontier = payload["request_check"]["prefill_routed_chunk_frontier"]
    frontier["candidates"][0]["total_stage_plus_compact_plus_static_bytes"] = 1
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request routed chunk frontier evidence is malformed" in err
    assert (
        "candidates[0].total_stage_plus_compact_plus_static_bytes=1"
        in err
    )
    assert "expected=8320" in err


def test_prefill_routed_chunk_frontier_allows_saturation_beyond_prompt() -> None:
    frontier = {
        "analyzed": True,
        "prompt_token_count": 17,
        "resolved_prompt_chunk_tokens": 16,
        "max_safe_prompt_chunk_tokens": 16,
        "top_k": 8,
        "layers": 75,
        "stage_align_bytes": 4096,
        "static_capacity_per_expert": "auto",
        "allow_static_capacity_overflow": False,
        "baseline_read_bytes": 204550963200,
        "saturation_chunk_tokens": 32,
        "candidates": [
            {
                "prompt_chunk_tokens": 16,
                "chunks_per_prompt": 2,
                "saturates_all_experts_per_layer": False,
                "planned_read_bytes": 204550963200,
                "extra_read_bytes": 0,
                "read_amplification": 1.0,
                "max_layer_planned_read_bytes": 2727346176,
                "max_stage_plus_compact_bytes": 5134352384,
                "max_chunk_stage_plus_compact_bytes": 385076428800,
                "total_stage_plus_compact_bytes": 409143705600,
                "max_static_capacity_binary_bytes": 25128,
                "max_chunk_static_capacity_binary_bytes": 1884600,
                "total_static_capacity_binary_bytes": 1897200,
                "max_stage_plus_compact_plus_static_bytes": 5134377512,
                "max_chunk_stage_plus_compact_plus_static_bytes": 385078313400,
                "total_stage_plus_compact_plus_static_bytes": 409145602800,
                "planned_read_seconds": 32.2,
            },
            {
                "prompt_chunk_tokens": 17,
                "chunks_per_prompt": 1,
                "saturates_all_experts_per_layer": False,
                "planned_read_bytes": 204550963200,
                "extra_read_bytes": 0,
                "read_amplification": 1.0,
                "max_layer_planned_read_bytes": 2727346176,
                "max_stage_plus_compact_bytes": 5455249408,
                "max_chunk_stage_plus_compact_bytes": 409143705600,
                "total_stage_plus_compact_bytes": 409143705600,
                "max_static_capacity_binary_bytes": 28328,
                "max_chunk_static_capacity_binary_bytes": 2124600,
                "total_static_capacity_binary_bytes": 2124600,
                "max_stage_plus_compact_plus_static_bytes": 5455277736,
                "max_chunk_stage_plus_compact_plus_static_bytes": 409145830200,
                "total_stage_plus_compact_plus_static_bytes": 409145830200,
                "planned_read_seconds": 32.2,
            },
        ],
    }

    assert _prefill_routed_chunk_frontier_errors(frontier) == ()


def test_generate_prepared_token_ids_required_launch_audit_rejects_current_no_runtime_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "required launch audit needs runtime preflight on this command" in err
    assert "--no-runtime-preflight" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_decode_routed_read_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    decode_read = payload["request_check"]["decode_routed_expert_read"]
    decode_read["within_seconds_limit"] = False
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request decode routed-read evidence is not passing" in err
    assert "within_seconds_limit=False" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_tampered_glm_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    readiness = prepared_glm_4bit_readiness(load_prepared_manifest(prepared))
    assert readiness["ok"] is True
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    glm_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "glm_4bit_ready"
    )
    glm_check["expected_decode_token_routed_expert_read_bytes"] = (
        readiness["expected_decode_token_routed_expert_read_bytes"] + 1
    )
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit GLM 4-bit envelope does not match" in err
    assert "expected_decode_token_routed_expert_read_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_glm_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    glm_check = next(
        check
        for check in payload["launch_audit"]["checks"]
        if check["code"] == "glm_4bit_ready"
    )
    glm_check.pop("expected_decode_token_routed_expert_read_bytes")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit GLM 4-bit envelope is missing required fields" in err
    assert "expected_decode_token_routed_expert_read_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_request_profile_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
        request_profile_argv=[
            "--min-free-unified-memory-gib",
            "0",
            *_launch_audit_request_guard_argv(),
        ],
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "current command does not replay launch audit request profile" in err
    assert "--prefill-prompt-chunk-tokens" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_request_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload.pop("request_launch_profile")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit is missing request launch profile" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_prefill_chunk_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_check"].pop("prefill_prompt_chunk_plan")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "missing prefill_prompt_chunk_plan evidence" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_request_profile_without_prefill_chunk_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_launch_profile"].pop("sections")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "missing prefill_prompt_chunk_plan section" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_prefill_chunk_plan_without_matrix_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    plan = payload["request_check"]["prefill_prompt_chunk_plan"]
    plan["max_safe"].pop("max_matrix_scratch_bytes")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill prompt chunk-plan evidence is malformed" in err
    assert "max_safe.missing=max_matrix_scratch_bytes" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_request_profile_without_prefill_guard_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    argv = payload["request_launch_profile"]["argv"]
    index = argv.index("--prefill-max-routed-read-seconds")
    del argv[index : index + 2]
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request profile is missing prefill routed-read" in err
    assert "--prefill-max-routed-read-seconds" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_request_profile_without_decode_guard_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    argv = payload["request_launch_profile"]["argv"]
    index = argv.index("--decode-max-routed-read-seconds-per-token")
    del argv[index : index + 2]
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request profile is missing decode routed-read" in err
    assert "--decode-max-routed-read-seconds-per-token" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_wide_prefill_guard_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    argv = payload["request_launch_profile"]["argv"]
    argv[argv.index("--prefill-max-routed-read-seconds") + 1] = "999"
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill routed-read guard is wider than audited request evidence" in err
    assert "--prefill-max-routed-read-seconds" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_wide_decode_guard_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    argv = payload["request_launch_profile"]["argv"]
    argv[argv.index("--decode-max-routed-read-seconds-per-token") + 1] = "999"
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "decode routed-read guard is wider than audited request evidence" in err
    assert "--decode-max-routed-read-seconds-per-token" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_missing_stage_temp_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_check"].pop("prefill_routed_stage_temp_disk")
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "missing prefill_routed_stage_temp_disk evidence" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_request_profile_without_stage_temp_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    argv = payload["request_launch_profile"]["argv"]
    index = argv.index("--prefill-max-stage-mib")
    del argv[index : index + 2]
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request profile is missing stage-temp guard flags" in err
    assert "--prefill-max-stage-mib" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_wide_stage_temp_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    argv = payload["request_launch_profile"]["argv"]
    argv[argv.index("--prefill-max-stage-mib") + 1] = "999"
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "stage-temp guard is wider than audited request evidence" in err
    assert "--prefill-max-stage-mib" in err


def test_generate_prepared_token_ids_required_launch_audit_accepts_audit_request_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
        request_profile_argv=[
            "--min-free-unified-memory-gib",
            "0",
            *_launch_audit_request_guard_argv(),
        ],
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(audit_path),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prefill_prompt_chunk_tokens"] == 2
    assert payload["applied_launch_profile"]["path"] == str(audit_path)
    assert payload["applied_launch_profile"]["source"] == "prepared_request_check"
    assert payload["applied_launch_profile"]["locked"] is True


def test_generate_prepared_token_ids_required_launch_audit_rejects_weak_request_profile_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
        request_profile_argv=[
            "--min-free-unified-memory-gib",
            "0",
            "--prefill-prompt-chunk-tokens",
            "2",
        ],
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["request_launch_profile"]["prepared"]["identity_strength"] = "weak"
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(audit_path),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "launch audit request profile prepared identity is weak" in err


def test_generate_prepared_token_ids_required_launch_audit_rejects_larger_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1,2",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "current request exceeds launch audit prompt envelope" in err
    assert "prompt_tokens=3" in err


def test_generate_prepared_text_accepts_required_launch_audit_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    _write_simple_model_tokenizer(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )
    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        token_result = TokenGenerationResult(
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="C",
            full_text=kwargs["prompt"] + "C",
            tokenizer_backend="simple",
            tokenizer_path=tmp_path / "model" / "simple_tokenizer.json",
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.cli.generate_text", fake_generate_text)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-text",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prompt"] == "AB"
    assert payload["applied_launch_profile"]["locked"] is True


def test_generate_prepared_token_ids_applies_prefill_plan_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--prefill-prompt-chunk-tokens",
        "2",
        "--prefill-max-routed-read-amplification",
        "1.5",
        "--prefill-max-routed-read-gib",
        "0.75",
        "--prefill-max-stage-mib",
        "6",
        "--prefill-max-compact-stage-mib",
        "5",
        "--compile-mpp-probe",
        "--prefill-backend-probe-timeout-seconds",
        "12.5",
    ]
    profile = tmp_path / "prefill-plan-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "prefill_plan",
                "argv_safe_to_replay": True,
                "sections": {
                    "prefill_guard_flags": {"source": "prefill_plan"},
                    "prefill_backend_probe_flags": {"source": "prefill_plan"},
                },
                "argv": profile_argv,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prefill_prompt_chunk_tokens"] == 2
    assert captured["prefill_max_routed_read_amplification"] == 1.5
    assert captured["prefill_max_routed_read_gib"] == 0.75
    assert captured["prefill_max_stage_mib"] == 6.0
    assert captured["prefill_max_compact_stage_mib"] == 5.0
    applied = payload["applied_launch_profile"]
    assert applied["source"] == "prefill_plan"
    assert applied["argv"] == profile_argv
    assert applied["prepared"] is None
    assert applied["matches_prepared"] is None
    assert applied["section_names"] == [
        "prefill_backend_probe_flags",
        "prefill_guard_flags",
    ]


def test_generate_prepared_token_ids_applies_plan_launch_and_decode_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 64 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 16 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 8 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 24 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=48 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    profile_argv = [
        "--require-prepared-memory-profile",
        "--max-live-working-set-mib",
        "8192",
        "--min-free-unified-memory-gib",
        "24",
        "--decode-max-routed-read-gib-per-token",
        "12.5",
        "--prefill-ssd-read-gib-s",
        "16",
        "--decode-max-routed-read-seconds-per-token",
        "0.8",
    ]
    profile = tmp_path / "plan-decode-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "plan",
                "argv_safe_to_replay": True,
                "sections": {
                    "launch_guard_flags": {"source": "plan"},
                    "decode_guard_flags": {"source": "plan"},
                },
                "argv": profile_argv,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["max_live_working_set_mib"] == 8192.0
    assert captured["min_free_unified_memory_gib"] == 24.0
    assert captured["decode_max_routed_read_gib_per_token"] == 12.5
    assert captured["prefill_ssd_read_gib_per_second"] == 16.0
    assert captured["decode_max_routed_read_seconds_per_token"] == 0.8
    applied = payload["applied_launch_profile"]
    assert applied["source"] == "plan"
    assert applied["argv"] == profile_argv
    assert applied["prepared"] is None
    assert applied["matches_prepared"] is None
    assert applied["section_names"] == ["decode_guard_flags", "launch_guard_flags"]


def test_plan_profile_allows_memory_guard_without_identity() -> None:
    profile = {
        "source": "plan",
        "argv_safe_to_replay": True,
        "sections": {
            "launch_guard_flags": {"source": "plan"},
            "decode_guard_flags": {"source": "plan"},
        },
        "argv": [
            "--require-prepared-memory-profile",
            "--max-live-working-set-mib",
            "8192",
            "--min-free-unified-memory-gib",
            "24",
            "--decode-max-routed-read-gib-per-token",
            "12.5",
        ],
    }

    assert _launch_profile_allows_missing_prepared_identity(profile) is True


def test_prefill_plan_public_shape_profile_is_allowed_without_identity() -> None:
    profile = {
        "source": "prefill_plan",
        "argv_safe_to_replay": True,
        "sections": {
            "public_glm_5_2_shape_guard_flags": {"source": "prefill_plan"},
        },
        "argv": ["--require-public-glm-5-2-shape"],
    }

    assert _launch_profile_allows_missing_prepared_identity(profile)


def test_prefill_linear_calibration_profile_is_allowed_without_identity() -> None:
    profile = {
        "source": "prefill_linear_calibration",
        "argv_safe_to_replay": True,
        "sections": {
            "prefill_runtime_policy_flags": {
                "source": "prefill_linear_calibration",
            },
        },
        "argv": [
            "--prefill-linear-backend",
            "mps-matrix-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "2",
        ],
    }

    assert _launch_profile_allows_missing_prepared_identity(profile)


def test_prefill_linear_calibration_profile_rejects_non_runtime_sections() -> None:
    profile = {
        "source": "prefill_linear_calibration",
        "argv_safe_to_replay": True,
        "sections": {
            "prefill_guard_flags": {"source": "prefill_plan"},
            "prefill_runtime_policy_flags": {
                "source": "prefill_linear_calibration",
            },
        },
        "argv": [
            "--prefill-linear-backend",
            "mps-matrix-f32",
        ],
    }

    assert _launch_profile_allows_missing_prepared_identity(profile) is False


def test_generate_prepared_token_ids_applies_prefill_plan_calibration_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--prefill-prompt-chunk-tokens",
        "2",
        "--prefill-max-routed-read-amplification",
        "1.5",
        "--prefill-mpsgraph-min-batch-tokens",
        "128",
        "--prefill-mpsgraph-min-matrix-dim",
        "32",
    ]
    profile = tmp_path / "prefill-plan-calibration-output.json"
    profile.write_text(
        json.dumps(
            {
                "prefill_plan": {"model_type": "glm_moe_dsa"},
                "calibration": {"matrix_shapes": [[256, 128]]},
                "combined_launch_profile": {
                    "source": "prefill_plan_calibration",
                    "argv_safe_to_replay": True,
                    "sections": {
                        "prefill_guard_flags": {"source": "prefill_plan"},
                        "prefill_runtime_policy_flags": {
                            "source": "prefill_linear_calibration"
                        },
                    },
                    "argv": profile_argv,
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prefill_prompt_chunk_tokens"] == 2
    assert captured["prefill_max_routed_read_amplification"] == 1.5
    assert captured["prefill_mpsgraph_min_batch_tokens"] == 128
    assert captured["prefill_mpsgraph_min_matrix_dim"] == 32
    applied = payload["applied_launch_profile"]
    assert applied["source"] == "prefill_plan_calibration"
    assert applied["argv"] == profile_argv
    assert applied["prepared"] is None
    assert applied["matches_prepared"] is None
    assert applied["section_names"] == [
        "prefill_guard_flags",
        "prefill_runtime_policy_flags",
    ]


def test_generate_prepared_token_ids_applies_prefill_linear_calibration_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--prefill-linear-backend",
        "mps-matrix-f32",
        "--prefill-mpsgraph-min-batch-tokens",
        "2",
        "--prefill-mpsgraph-min-matrix-dim",
        "2",
    ]
    profile = tmp_path / "prefill-linear-calibration-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "prefill_linear_calibration",
                "argv_safe_to_replay": True,
                "sections": {
                    "prefill_runtime_policy_flags": {
                        "source": "prefill_linear_calibration",
                        "prefill_linear_backend": "mps-matrix-f32",
                        "prefill_mpsgraph_min_batch_tokens": 2,
                        "prefill_mpsgraph_min_matrix_dim": 2,
                    },
                },
                "argv": profile_argv,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["prefill_linear_backend"] == "mps-matrix-f32"
    assert captured["prefill_mpsgraph_min_batch_tokens"] == 2
    assert captured["prefill_mpsgraph_min_matrix_dim"] == 2
    applied = payload["applied_launch_profile"]
    assert applied["source"] == "prefill_linear_calibration"
    assert applied["argv"] == profile_argv
    assert applied["prepared"] is None
    assert applied["matches_prepared"] is None
    assert applied["section_names"] == ["prefill_runtime_policy_flags"]


def test_generate_prepared_token_ids_replays_profile_no_runtime_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--no-runtime-preflight"],
    )
    captured: dict[str, object] = {}

    def fake_admission(*args, **kwargs):
        del args
        captured["admission_runtime_preflight"] = kwargs.get("runtime_preflight")
        return None

    def fake_generate_token_ids(**kwargs):
        captured["generation_preflight_runtime"] = kwargs.get("preflight_runtime")
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        fake_admission,
    )
    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["admission_runtime_preflight"] is False
    assert captured["generation_preflight_runtime"] is False
    applied = payload["applied_launch_profile"]
    assert applied["argv"] == ["--no-runtime-preflight"]
    assert applied["locked"] is True
    assert applied["profile_flag_count"] == 1
    assert applied["lock_checked_flags"] == ["--no-runtime-preflight"]


def test_generate_prepared_token_ids_rejects_prefill_profile_no_runtime_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = tmp_path / "prefill-plan-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "prefill_plan",
                "argv_safe_to_replay": True,
                "sections": {
                    "prefill_guard_flags": {"source": "prefill_plan"},
                },
                "argv": ["--no-runtime-preflight"],
            }
        ),
        encoding="utf-8",
    )

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    assert "launch profile is missing prepared identity metadata" in (
        capsys.readouterr().err
    )


def test_inspect_prepared_cli_applies_prefill_plan_runtime_policy_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--prefill-linear-backend",
        "mpsgraph-f32",
        "--prefill-mpsgraph-min-batch-tokens",
        "64",
        "--prefill-mpsgraph-min-matrix-dim",
        "16",
        "--prefill-min-accelerated-flop-fraction",
        "0.5",
        "--run-mpsgraph-probe",
    ]
    profile = tmp_path / "prefill-plan-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "prefill_plan",
                "argv_safe_to_replay": True,
                "sections": {
                    "prefill_runtime_policy_flags": {"source": "prefill_plan"},
                },
                "argv": profile_argv,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            mps_graph_probe_requested=True,
            mps_graph_probe_ran=True,
            mps_graph_probe_ok=True,
            mps_graph_probe_error=None,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    assert health["prefill_linear_backend"] == "mpsgraph-f32"
    assert health["prefill_mpsgraph_min_batch_tokens"] == 64
    assert health["prefill_mpsgraph_min_matrix_dim"] == 16
    assert health["prefill_min_accelerated_flop_fraction"] == 0.5
    assert health["prefill_acceleration_requirement"]["ok"] is True
    applied = health["applied_launch_profile"]
    assert applied["source"] == "prefill_plan"
    assert applied["section_names"] == ["prefill_runtime_policy_flags"]
    assert applied["matches_prepared"] is None


def test_generate_prepared_token_ids_rejects_launch_profile_without_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = tmp_path / "no-identity-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "unit",
                "argv_safe_to_replay": True,
                "sections": {"prefill_guard_flags": {"source": "unit"}},
                "argv": ["--prefill-prompt-chunk-tokens", "2"],
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    assert "launch profile is missing prepared identity metadata" in (
        capsys.readouterr().err
    )


def test_generate_prepared_token_ids_rejects_non_prefill_plan_profile_without_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = tmp_path / "unsafe-prefill-plan-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "prefill_plan",
                "argv_safe_to_replay": True,
                "sections": {"prefill_guard_flags": {"source": "prefill_plan"}},
                "argv": ["--max-live-working-set-mib", "8192"],
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    assert "launch profile is missing prepared identity metadata" in (
        capsys.readouterr().err
    )


def test_generate_prepared_token_ids_rejects_unsafe_prefill_plan_calibration_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = tmp_path / "unsafe-prefill-plan-calibration-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "prefill_plan_calibration",
                "argv_safe_to_replay": True,
                "sections": {"prefill_guard_flags": {"source": "prefill_plan"}},
                "argv": ["--max-live-working-set-mib", "8192"],
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    assert "launch profile is missing prepared identity metadata" in (
        capsys.readouterr().err
    )


def test_generate_prepared_token_ids_rejects_mismatched_launch_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest = load_prepared_manifest(prepared)
    profile = tmp_path / "launch-profile.json"
    profile.write_text(
        json.dumps(
            {
                "source": "unit",
                "argv_safe_to_replay": True,
                "prepared": {
                    "model_config_sha256": manifest.model_config_sha256,
                    "expert_layout_bytes": manifest.expert_layout_bytes,
                    "resident_layout_bytes": 999,
                    "decode_cache_layout_bytes": manifest.decode_cache_layout_bytes,
                    "decode_cache_file_bytes": manifest.decode_cache_file_bytes,
                    "max_context_tokens": manifest.max_context_tokens,
                    "expert_quantization": manifest.expert_quantization,
                    "expert_group_size": manifest.expert_group_size,
                },
                "argv": [
                    "--max-live-working-set-mib",
                    "8192",
                ],
            }
        ),
        encoding="utf-8",
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    assert "launch profile does not match this prepared package" in capsys.readouterr().err


def test_generate_prepared_token_ids_rejects_prepare_flags_identity_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    flags = tmp_path / "prepare-flags.json"
    flags.write_text(
        json.dumps(
            {
                "source": "plan",
                "argv_safe_to_replay": True,
                "argv": ["--auto-context-from-budget"],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_flags_applied"] = True
    payload["prepare_flags_source"] = "plan"
    payload["prepare_flags_path"] = str(flags)
    payload["prepare_flags_sha256"] = hashlib.sha256(flags.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--max-live-working-set-mib", "8192"],
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_flags_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    stderr = capsys.readouterr().err
    assert "launch profile does not match this prepared package" in stderr
    assert "prepare_flags_sha256" in stderr


def test_generate_prepared_token_ids_rejects_prepare_hardware_identity_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "prepare_hardware_chip_name": "Apple M5 Max",
            "prepare_hardware_unified_memory_bytes": 128 * 1024**3,
            "prepare_hardware_gpu_cores": 40,
            "prepare_hardware_apple_silicon_generation": 5,
            "prepare_hardware_apple_silicon_tier": "Max",
        }
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--max-live-working-set-mib", "8192"],
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_hardware_chip_name"] = "Apple M4 Pro"
    payload["prepare_hardware_gpu_cores"] = 20
    payload["prepare_hardware_apple_silicon_generation"] = 4
    payload["prepare_hardware_apple_silicon_tier"] = "Pro"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    stderr = capsys.readouterr().err
    assert "launch profile does not match this prepared package" in stderr
    assert "prepare_hardware_chip_name" in stderr


def test_generate_prepared_token_ids_rejects_prepare_public_shape_identity_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_public_glm_5_2_shape_required"] = True
    payload["prepare_public_glm_5_2_shape_matches"] = True
    payload["prepare_public_glm_5_2_shape_mismatched_fields"] = []
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--max-live-working-set-mib", "8192"],
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_public_glm_5_2_shape_required"] = False
    payload["prepare_public_glm_5_2_shape_matches"] = False
    payload["prepare_public_glm_5_2_shape_mismatched_fields"] = ["hidden_size"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
        ]
    )

    assert status == 1
    stderr = capsys.readouterr().err
    assert "launch profile does not match this prepared package" in stderr
    assert "prepare_public_glm_5_2_shape_required" in stderr


def test_benchmark_prepared_token_ids_reports_applied_launch_profile_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--max-live-working-set-mib",
        "8192",
        "--min-free-unified-memory-gib",
        "0",
    ]
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=profile_argv,
    )

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
            estimated_read_bytes=1024,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "bench-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-locked-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    applied = payload["applied_launch_profile"]
    assert applied["path"] == str(profile)
    assert applied["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert applied["argv"] == profile_argv
    assert applied["locked"] is True
    assert applied["lock_required"] is True
    assert applied["profile_flag_count"] == 2
    assert applied["lock_checked_flags"] == [
        "--max-live-working-set-mib",
        "--min-free-unified-memory-gib",
    ]
    assert applied["matches_prepared"] is True
    assert payload["token_result"]["applied_launch_profile"] == applied


def test_benchmark_prepared_token_ids_required_launch_audit_rejects_larger_max_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=1,
        max_new_tokens=1,
    )

    def fail_benchmark(**kwargs):
        raise AssertionError("benchmark_prepared_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.benchmark_prepared_token_ids", fail_benchmark)

    status = cli_main(
        [
            "bench-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "2",
            *_launch_audit_request_guard_argv(
                prompt_token_count=1,
                max_new_tokens=1,
            ),
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "current request exceeds launch audit generation envelope" in err
    assert "max_new_tokens=2" in err


def test_benchmark_prepared_token_ids_writes_actual_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 64 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 16 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 4 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=48 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    class RuntimeGuardDict(dict):
        @property
        def read_bytes_per_token(self) -> int:
            return int(self["read_bytes_per_token"])

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(
                GeneratedStep(
                    position=0,
                    input_token_id=0,
                    selected_token_id=2,
                    topk=(),
                    expert_read_bytes=384,
                    decode_layers=(1,),
                ),
            ),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
            estimated_read_bytes=1024,
            runtime_guard=RuntimeGuardDict(read_bytes_per_token=384),
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    profile_path = tmp_path / "benchmark-launch-profile.json"
    status = cli_main(
        [
            "bench-prepared-token-ids",
            str(prepared),
            "--require-prepared-memory-profile",
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--prefill-ssd-read-gib-s",
            "16",
            "--write-launch-profile",
            str(profile_path),
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile == payload["suggested_launch_profile"]
    assert profile["source"] == "benchmark_actual"
    assert profile["argv_safe_to_replay"] is True
    assert profile["prepared"]["model_dir"] == str(
        load_prepared_manifest(prepared).model_dir
    )
    launch_flags = profile["sections"]["launch_guard_flags"]
    assert launch_flags["require_prepared_memory_profile"] is True
    assert launch_flags["max_live_working_set_mib"] == 4 * 1024
    assert launch_flags["min_free_unified_memory_gib"] == 16
    assert profile["sections"]["prepared_ssd_read_flags"][
        "prefill_ssd_read_gib_per_second"
    ] == 16.0
    assert profile["sections"]["decode_guard_flags"] == payload[
        "suggested_decode_guard_flags"
    ]
    decode_actual = profile["sections"]["decode_actual_read_time"]
    assert decode_actual["source"] == "benchmark_actual_decode"
    assert decode_actual["decode_step_count"] == 1
    assert decode_actual["decode_read_bytes_per_token"] == 384
    assert decode_actual["planned_decode_routed_read_bytes"] == 384
    assert decode_actual["actual_decode_routed_read_bytes"] == 384
    assert decode_actual["actual_decode_routed_read_bytes_ok"] is True
    assert decode_actual["planned_decode_routed_read_seconds"] == pytest.approx(
        384 / (16 * 1024**3)
    )
    assert decode_actual["actual_decode_routed_read_seconds"] == pytest.approx(
        384 / (16 * 1024**3)
    )
    assert decode_actual["prefill_ssd_read_gib_per_second"] == 16.0
    assert decode_actual["decode_max_routed_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3) * 1.05
    )
    assert decode_actual["total_decode_max_routed_read_seconds"] == pytest.approx(
        384 / (16 * 1024**3) * 1.05
    )
    assert decode_actual["total_decode_routed_read_seconds_ok"] is True
    assert "--require-prepared-memory-profile" in profile["argv"]
    assert "--max-live-working-set-mib" in profile["argv"]
    assert "--min-free-unified-memory-gib" in profile["argv"]
    assert "--prefill-ssd-read-gib-s" in profile["argv"]
    assert "--decode-max-routed-read-gib-per-token" in profile["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in profile["argv"]


def test_generate_prepared_token_ids_prints_applied_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
            argv=[
                "--max-live-working-set-mib",
                "8193",
                "--min-free-unified-memory-gib",
                "0",
            ],
    )

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
        ]
    )

    assert status == 0
    _assert_printed_applied_launch_profile(capsys.readouterr().out, profile)


def test_generate_prepared_text_prints_applied_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    _write_simple_model_tokenizer(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--max-live-working-set-mib",
            "8193",
            "--min-free-unified-memory-gib",
            "0",
        ],
    )

    def fake_generate_text(**kwargs):
        del kwargs
        token_result = TokenGenerationResult(
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            )
        return TextGenerationResult(
            prompt="AB",
            generated_text="world",
            full_text="AB world",
            tokenizer_backend="fake",
            tokenizer_path=tmp_path / "tokenizer",
            prompt_token_ids=(0,),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=token_result,
        )

    monkeypatch.setattr("largerlm.cli.generate_text", fake_generate_text)

    status = cli_main(
        [
            "generate-prepared-text",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--tokenizer-backend",
            "simple",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 0
    _assert_printed_applied_launch_profile(capsys.readouterr().out, profile)


def test_benchmark_prepared_token_ids_prints_applied_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--max-live-working-set-mib",
            "8192",
            "--min-free-unified-memory-gib",
            "0",
        ],
    )

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
            estimated_read_bytes=1024,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "bench-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
        ]
    )

    assert status == 0
    _assert_printed_applied_launch_profile(capsys.readouterr().out, profile)


def test_generate_prepared_token_ids_uses_manifest_memory_guard_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recommended_max_live_working_set_bytes"] = 7 * 1024**3
    payload["recommended_min_free_unified_memory_bytes"] = 24 * 1024**3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["max_live_working_set_mib"] == 7 * 1024
    assert captured["min_free_unified_memory_gib"] == 24

    captured.clear()
    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--max-live-working-set-mib",
            "0",
            "--min-free-unified-memory-gib",
            "0",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["max_live_working_set_mib"] == 0.0
    assert captured["min_free_unified_memory_gib"] == 0.0


def test_generate_prepared_token_ids_inherits_manifest_ssd_read_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_cold_read_gib_per_second"] = 18.25
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prefill_ssd_read_gib_per_second"] == 18.25

    captured.clear()
    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--no-prepared-ssd-read-default",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prefill_ssd_read_gib_per_second"] == 0.0


def test_generate_prepared_token_ids_resolves_auto_prefill_backend_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_prefill_fallback",
            mps_graph_matmul_declared=True,
            mps_graph_runtime_available=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("host probe could not create a default Metal device",),
        ),
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prefill_linear_backend"] == "custom-metal"


def test_generate_prepared_token_ids_require_prefill_acceleration_allows_mpsgraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            prompt_prefill=_fake_prompt_prefill_result(
                tmp_path,
                accelerated=True,
                prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            ),
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prefill_linear_backend"] == "auto"
    assert captured["require_prefill_acceleration"] is True


def test_generate_prepared_token_ids_require_prefill_acceleration_requires_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
            mps_graph_probe_requested=bool(kwargs.get("run_mpsgraph_probe")),
            mps_graph_probe_ran=bool(kwargs.get("run_mpsgraph_probe")),
            mps_graph_probe_ok=(
                True if kwargs.get("run_mpsgraph_probe") else None
            ),
        ),
    )

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration requirement failed" in err
    assert "requires --run-mpsgraph-probe" in err


def test_generate_prepared_token_ids_require_prefill_acceleration_rejects_actual_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            prompt_prefill=_fake_prompt_prefill_result(
                tmp_path,
                accelerated=False,
                prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            ),
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    assert captured["prefill_linear_backend"] == "auto"
    err = capsys.readouterr().err
    assert "prefill acceleration actual coverage failed" in err
    assert "no resident prefill matrices used an accelerated backend" in err


def test_generate_prepared_token_ids_require_prefill_acceleration_ignores_default_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            prompt_prefill=_fake_prompt_prefill_result(
                tmp_path,
                accelerated=False,
                coverage_ok=True,
                prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            ),
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration actual coverage failed" in err
    assert "no resident prefill matrices used an accelerated backend" in err


def test_generate_prepared_token_ids_rejects_low_accelerated_flop_fraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            prompt_prefill=_fake_prompt_prefill_result(
                tmp_path,
                accelerated=True,
                accelerated_flop_fraction=0.25,
                prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            ),
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0,1",
            "--max-new-tokens",
            "1",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert captured["prefill_min_accelerated_flop_fraction"] == 0.5
    assert "prefill acceleration actual coverage failed" in err
    assert "FLOP fraction 0.25 is below required 0.5" in err


def test_generate_prepared_token_ids_require_prefill_acceleration_rejects_custom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--prefill-linear-backend",
            "custom-metal",
            "--require-prefill-acceleration",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration requirement failed" in err
    assert "custom-metal was configured" in err


def test_generate_prepared_token_ids_require_prefill_acceleration_rejects_mpp_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            mps_graph_runtime_available=False,
            mpp_runtime_available=True,
        ),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration requirement failed" in err
    assert "no selectable MPP prefill backend is implemented" in err


def test_generate_prepared_text_require_prefill_acceleration_allows_mpsgraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    _add_large_bf16_prefill_matrix(prepared)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="C",
            full_text=kwargs["prompt"] + "C",
            tokenizer_backend="simple",
            tokenizer_path=tmp_path / "model" / "simple_tokenizer.json",
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=TokenGenerationResult(
                prompt_token_ids=(0, 1),
                generated_token_ids=(2,),
                steps=(),
                work_dir=tmp_path,
                kept_work_dir=False,
                max_context_tokens=4,
                sampling_temperature=0.0,
                sampling_top_p=1.0,
                prompt_prefill=_fake_prompt_prefill_result(
                    tmp_path,
                    accelerated=True,
                    prompt_token_ids=(0, 1),
                ),
            ),
        )

    monkeypatch.setattr("largerlm.cli.generate_text", fake_generate_text)

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prompt"] == "AB"
    assert captured["auto_batch_prefill_prompt"] is True


def test_generate_prepared_text_resolves_auto_prefill_backend_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    captured: dict[str, object] = {}

    def fake_generate_text(**kwargs):
        captured.update(kwargs)
        return TextGenerationResult(
            prompt=kwargs["prompt"],
            generated_text="C",
            full_text=kwargs["prompt"] + "C",
            tokenizer_backend="simple",
            tokenizer_path=tmp_path / "model" / "simple_tokenizer.json",
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            eos_token_id=None,
            token_result=TokenGenerationResult(
                prompt_token_ids=(0, 1),
                generated_token_ids=(2,),
                steps=(),
                work_dir=tmp_path,
                kept_work_dir=False,
                max_context_tokens=4,
                sampling_temperature=0.0,
                sampling_top_p=1.0,
            ),
        )

    monkeypatch.setattr("largerlm.cli.generate_text", fake_generate_text)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_prefill_fallback",
            mps_graph_matmul_declared=True,
            mps_graph_runtime_available=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("host probe could not create a default Metal device",),
        ),
    )

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt",
            "AB",
            "--max-new-tokens",
            "1",
            "--no-runtime-preflight",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prefill_linear_backend"] == "custom-metal"
    assert captured["auto_batch_prefill_prompt"] is True


def test_generate_prepared_token_ids_require_prefill_acceleration_checks_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            ",".join("0" for _ in range(64)),
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration request coverage failed" in err
    assert "no resident prefill matrices resolved" in err


def test_generate_prepared_token_ids_require_prefill_acceleration_allows_decode_only_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prompt_token_ids"] == (0,)


def test_generate_prepared_text_require_prefill_acceleration_checks_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    def fail_generate_text(**kwargs):
        raise AssertionError("generate_text should not be called")

    monkeypatch.setattr("largerlm.cli.generate_text", fail_generate_text)

    status = cli_main(
        [
            "generate-prepared-text",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt",
            "A" * 64,
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration request coverage failed" in err
    assert "no resident prefill matrices resolved" in err


def test_bench_prepared_token_ids_require_prefill_acceleration_checks_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    def fail_benchmark_prepared_token_ids(*args, **kwargs):
        raise AssertionError("benchmark_prepared_token_ids should not be called")

    monkeypatch.setattr(
        "largerlm.cli.benchmark_prepared_token_ids",
        fail_benchmark_prepared_token_ids,
    )

    status = cli_main(
        [
            "bench-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            ",".join("0" for _ in range(64)),
            "--max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prefill acceleration request coverage failed" in err
    assert "no resident prefill matrices resolved" in err


def test_generate_prepared_token_ids_require_glm_4bit_allows_ready_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--require-glm-4bit",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prompt_token_ids"] == (0,)


def test_generate_prepared_token_ids_applies_launch_profile_require_glm_4bit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--require-glm-4bit"],
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prompt_token_ids"] == (0,)


def test_generate_prepared_token_ids_applies_launch_profile_require_public_glm_5_2_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_memory_profile(prepared)
    _make_prepared_glm_4bit_ready(prepared)
    _mock_safe_system_memory(monkeypatch)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--require-public-glm-5-2-shape"],
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["prompt_token_ids"] == (0,)


def test_generate_prepared_token_ids_require_public_glm_5_2_shape_requires_memory_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--require-public-glm-5-2-shape",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile check failed" in err
    assert "prepared manifest is missing required memory profile fields" in err


def test_generate_prepared_token_ids_rejects_public_glm_5_2_with_missing_dsa_indexer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--require-public-glm-5-2-shape",
            "--allow-missing-dsa-indexer",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    stderr = capsys.readouterr().err
    assert "--require-public-glm-5-2-shape" in stderr
    assert "--allow-missing-dsa-indexer" in stderr


def test_generate_prepared_token_ids_require_glm_4bit_rejects_incomplete_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--require-glm-4bit",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared GLM 4bit readiness failed" in err
    assert "expert layout layers do not match config MoE layers" in err


def test_benchmark_prepared_token_ids_require_glm_4bit_rejects_incomplete_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match="prepared GLM 4bit readiness failed: .*expert layout layers do not match",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
            require_glm_4bit=True,
        )


def test_benchmark_prepared_token_ids_require_glm_4bit_allows_ready_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0],
        max_new_tokens=1,
        require_glm_4bit=True,
    )

    assert captured["prompt_token_ids"] == (0,)
    assert result.generated_tokens == 1
    profile = result.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "benchmark_actual"
    assert profile["sections"]["glm_4bit_guard_flags"] == {
        "source": "benchmark_actual",
        "require_glm_4bit": True,
        "argv": ("--require-glm-4bit",),
    }
    assert "--require-glm-4bit" in profile["argv"]


def test_benchmark_prepared_token_ids_require_public_glm_5_2_shape_rejects_non_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match="prepared config does not match the public GLM-5.2 shape",
    ) as excinfo:
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
            require_public_glm_5_2_shape=True,
        )
    assert "hidden_size" in str(excinfo.value)
    assert "num_hidden_layers" in str(excinfo.value)


def test_benchmark_prepared_token_ids_require_public_glm_5_2_shape_requires_memory_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match="prepared manifest is missing required memory profile fields",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
            require_public_glm_5_2_shape=True,
        )


def test_benchmark_prepared_token_ids_require_public_glm_5_2_shape_allows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_memory_profile(prepared)
    _make_prepared_glm_4bit_ready(prepared)
    _mock_safe_system_memory(monkeypatch)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0],
        max_new_tokens=1,
        require_public_glm_5_2_shape=True,
    )

    assert captured["prompt_token_ids"] == (0,)
    assert result.generated_tokens == 1
    profile = result.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "benchmark_actual"
    assert profile["sections"]["glm_4bit_guard_flags"]["require_glm_4bit"] is True
    assert profile["sections"]["public_glm_5_2_shape_guard_flags"] == {
        "source": "benchmark_actual",
        "require_public_glm_5_2_shape": True,
        "argv": ("--require-public-glm-5-2-shape",),
    }
    assert "--require-glm-4bit" in profile["argv"]
    assert "--require-public-glm-5-2-shape" in profile["argv"]


def test_benchmark_prepared_token_ids_rejects_public_glm_5_2_with_missing_dsa_indexer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match="require_public_glm_5_2_shape cannot be used with allow_missing_dsa_indexer",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
            require_public_glm_5_2_shape=True,
            allow_missing_dsa_indexer=True,
        )


def test_benchmark_summary_profile_preserves_prefill_acceleration_gates(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest = load_prepared_manifest(prepared)
    token_result = TokenGenerationResult(
        prompt_token_ids=(0,),
        generated_token_ids=(2,),
        steps=(),
        work_dir=tmp_path,
        kept_work_dir=False,
        max_context_tokens=4,
        sampling_temperature=0.0,
        sampling_top_p=1.0,
        elapsed_seconds=1.0,
    )

    result = summarize_generation(
        manifest,
        token_result,
        require_prefill_acceleration=True,
        prefill_min_accelerated_flop_fraction=0.5,
        prefill_linear_backend="mpsgraph-f32",
        prefill_mpsgraph_min_batch_tokens=128,
        prefill_mpsgraph_min_matrix_dim=32,
        compile_mpp_probe=True,
        run_mpp_probe=True,
        run_mpsgraph_probe=True,
    )

    profile = result.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "benchmark_actual"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_backend_probe_flags"] == {
        "source": "benchmark_actual",
        "compile_mpp_probe": True,
        "run_mpp_probe": True,
        "run_mpsgraph_probe": True,
        "argv": (
            "--compile-mpp-probe",
            "--run-mpp-probe",
            "--run-mpsgraph-probe",
        ),
    }
    assert profile["sections"]["prefill_runtime_policy_flags"] == {
        "source": "benchmark_actual",
        "prefill_mpsgraph_min_batch_tokens": 128,
        "prefill_mpsgraph_min_matrix_dim": 32,
        "prefill_min_accelerated_flop_fraction": 0.5,
        "require_prefill_acceleration": True,
        "argv": (
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--require-prefill-acceleration",
        ),
    }
    assert profile["sections"]["prefill_acceleration_flags"] == {
        "source": "benchmark_actual",
        "prefill_linear_backend": "mpsgraph-f32",
        "argv": ("--prefill-linear-backend", "mpsgraph-f32"),
    }
    assert "prefill_actual_read_time" not in profile["sections"]
    assert "--compile-mpp-probe" in profile["argv"]
    assert "--run-mpsgraph-probe" in profile["argv"]
    assert "--require-prefill-acceleration" in profile["argv"]
    assert "--prefill-linear-backend" in profile["argv"]


def test_benchmark_prepared_token_ids_uses_config_tie_embedding_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = prepared.parent / "model" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tie_word_embeddings"] = False
    config["vocab_size"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0],
        max_new_tokens=1,
        preflight_runtime=False,
    )

    assert captured["allow_tied_embeddings"] is False
    assert captured["expected_vocab_size"] == 4
    assert captured["expected_hidden_size"] == 8
    assert result.generated_tokens == 1


def test_generate_prepared_token_ids_rejects_low_prepared_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile check failed" in err
    assert "current system total memory is below" in err


def test_generate_prepared_token_ids_requires_prepared_memory_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--require-prepared-memory-profile",
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile check failed" in err
    assert "prepared manifest is missing required memory profile fields" in err
    assert "recommended_max_live_working_set_bytes" in err


def test_generate_prepared_token_ids_requires_prepared_memory_profile_allows_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 64 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 16 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 4 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=48 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
        )

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.cli._require_prepared_request_admission",
        lambda *args, **kwargs: None,
    )

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--require-prepared-memory-profile",
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 0
    assert captured["max_live_working_set_mib"] == 4 * 1024
    assert captured["min_free_unified_memory_gib"] == 16


def test_generate_prepared_token_ids_rejects_low_recommended_available_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_max_live_working_set_bytes"] = 40 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=48 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.cli.generate_token_ids", fail_generate_token_ids)

    status = cli_main(
        [
            "generate-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile check failed" in err
    assert "current available memory is below the prepared recommended" in err


def test_bench_prepared_token_ids_rejects_low_prepared_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("benchmark generation should not be called")

    monkeypatch.setattr(
        "largerlm.benchmark.generate_token_ids",
        fail_generate_token_ids,
    )

    status = cli_main(
        [
            "bench-prepared-token-ids",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--quiet-runner",
            "--json",
        ]
    )

    assert status == 1
    assert "prepared runtime profile check failed" in capsys.readouterr().err


def test_benchmark_prepared_token_ids_requires_prepared_memory_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("benchmark generation should not be called")

    monkeypatch.setattr(
        "largerlm.benchmark.generate_token_ids",
        fail_generate_token_ids,
    )

    with pytest.raises(
        BenchmarkError,
        match="prepared manifest is missing required memory profile fields",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
            require_prepared_memory_profile=True,
        )


def test_serve_prepared_require_glm_4bit_rejects_incomplete_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--quiet-runner",
        ]
    )

    assert status == 1
    assert "prepared GLM 4bit readiness failed" in capsys.readouterr().err


def test_serve_prepared_require_public_glm_5_2_shape_rejects_non_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)

    def fail_run_prepared_server(config) -> None:
        del config
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-public-glm-5-2-shape",
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared public GLM-5.2 readiness failed" in err
    assert "prepared config does not match the public GLM-5.2 shape" in err


def test_serve_prepared_require_public_glm_5_2_shape_allows_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_memory_profile(prepared)
    _make_prepared_glm_4bit_ready(prepared)
    _mock_safe_system_memory(monkeypatch)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-public-glm-5-2-shape",
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert captured["config"].require_public_glm_5_2_shape is True
    assert captured["config"].require_prepared_memory_profile is True


def test_serve_prepared_accepts_required_launch_audit_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
    )
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
        ]
    )

    assert status == 0
    config = captured["config"]
    assert config.applied_launch_profile["locked"] is True
    assert config.applied_launch_profile["sha256"] == hashlib.sha256(
        profile.read_bytes()
    ).hexdigest()
    assert config.max_prompt_tokens == 2
    assert config.max_new_tokens_cap == 1
    assert config.launch_audit_envelope == {
        "schema": "largerlm.launch_audit_server_envelope.v1",
        "artifact_path": str(audit_path),
        "artifact_source": "unit",
        "applied_launch_profile_sha256": hashlib.sha256(
            profile.read_bytes()
        ).hexdigest(),
        "audited_prompt_token_count": 2,
        "audited_max_new_tokens": 1,
        "audited_required_context_tokens": 3,
        "audited_runtime_required_available_memory_bytes": 5120,
        "audited_runtime_system_available_memory_bytes": 1024**3,
        "audited_runtime_system_total_memory_bytes": 2 * 1024**3,
        "audited_runtime_system_memory_source": "unit",
        "server_max_prompt_tokens": 2,
        "server_max_new_tokens_cap": 1,
        "server_caps_within_envelope": True,
        "server_configured_max_live_working_set_bytes": 8 * 1024**3,
        "server_configured_min_free_unified_memory_bytes": 0,
        "server_configured_required_available_memory_bytes": 8 * 1024**3,
        "server_system_available_memory_bytes": 96 * 1024**3,
        "server_system_total_memory_bytes": 128 * 1024**3,
        "server_system_memory_source": "test",
        "server_memory_guard_ok": True,
    }


def test_serve_prepared_required_launch_audit_rejects_missing_request_profile_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        request_profile_argv=[
            "--min-free-unified-memory-gib",
            "0",
            *_launch_audit_request_guard_argv(),
        ],
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "current command does not replay launch audit request profile" in err
    assert "--prefill-prompt-chunk-tokens" in err


def test_serve_prepared_required_launch_audit_rejects_current_low_memory_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=4 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "current server memory is below launch audit guard" in err
    assert f"required={8 * 1024**3!r}" in err


def test_serve_prepared_required_launch_audit_accepts_request_profile_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        request_profile_argv=[
            "--min-free-unified-memory-gib",
            "0",
            *_launch_audit_request_guard_argv(),
        ],
    )
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert captured["config"].prefill_prompt_chunk_tokens == 2


def test_serve_prepared_required_launch_audit_rejects_stale_applied_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    request_profile_argv = [
        "--min-free-unified-memory-gib",
        "0",
        *_launch_audit_request_guard_argv(),
    ]
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        request_profile_argv=request_profile_argv,
    )
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    request_profile = tmp_path / "request-launch-profile.json"
    request_profile.write_text(
        json.dumps(audit_payload["request_launch_profile"]),
        encoding="utf-8",
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(request_profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "server launch audit was created for a different applied launch profile" in err
    assert "regenerate the launch audit with the exact server launch profile" in err


def test_serve_prepared_required_launch_audit_accepts_auto_request_profile_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    request_argv = [
        "--min-free-unified-memory-gib",
        "0",
        *_launch_audit_request_guard_argv_with_auto_prompt_chunk(),
    ]
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=request_argv,
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        request_profile_argv=request_argv,
    )
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert captured["config"].prefill_prompt_chunk_tokens == 0


def test_serve_prepared_required_launch_audit_accepts_stricter_request_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    request_argv = [
        "--min-free-unified-memory-gib",
        "0",
        *_launch_audit_request_guard_argv_with_auto_prompt_chunk(),
    ]
    strict_argv = _replace_launch_arg_values(
        request_argv,
        {
            "--prefill-max-stage-mib": "0.001",
            "--prefill-max-compact-stage-mib": "0.001",
            "--prefill-max-stage-raw-ranges": "1",
            "--prefill-max-stage-coalesced-ranges": "1",
        },
    )
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=strict_argv,
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        request_profile_argv=request_argv,
    )
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-prompt-tokens",
            "2",
            "--max-new-tokens-cap",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert captured["config"].prefill_prompt_chunk_tokens == 0
    assert captured["config"].prefill_max_stage_mib == 0.001


def test_serve_prepared_required_launch_audit_rejects_caps_above_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
        prompt_token_count=2,
        max_new_tokens=1,
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            *_launch_audit_request_guard_argv(),
            "--quiet-runner",
        ]
    )

    assert status == 1
    assert (
        "server prompt token cap exceeds launch audit prompt envelope"
        in capsys.readouterr().err
    )


def test_serve_prepared_required_launch_audit_needs_locked_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=["--min-free-unified-memory-gib", "0"],
    )
    audit_path = _write_launch_audit_artifact(
        tmp_path / "launch-audit.json",
        profile=profile,
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            "--apply-launch-profile",
            str(profile),
            "--require-launch-audit",
            str(audit_path),
            str(prepared),
            "--runner",
            "unused-runner",
            "--quiet-runner",
        ]
    )

    assert status == 1
    assert (
        "required launch audit needs --lock-launch-profile"
        in capsys.readouterr().err
    )


def test_serve_prepared_rejects_low_prepared_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--quiet-runner",
        ]
    )

    assert status == 1
    assert "prepared runtime profile check failed" in capsys.readouterr().err


def test_serve_prepared_requires_prepared_memory_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--require-prepared-memory-profile",
            "--runner",
            "unused-runner",
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile check failed" in err
    assert "prepared manifest is missing required memory profile fields" in err


def test_serve_prepared_rejects_unverified_prepared_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recommended_max_live_working_set_bytes"] = 40 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: None,
    )

    def fail_run_prepared_server(config) -> None:
        raise AssertionError("server should not start")

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "prepared runtime profile check failed" in err
    assert "could not verify prepared runtime profile" in err


def test_serve_prepared_cli_passes_prefill_linear_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    cache_dir = tmp_path / "mla-kv-b-cache"
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)
    monkeypatch.setattr(
        "largerlm.cli.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prefill-prompt-chunk-tokens",
            "32",
            "--prefill-max-prompt-batch-mib",
            "256",
            "--prefill-max-cache-write-mib",
            "512",
            "--prefill-max-stage-mib",
            "768",
            "--prefill-max-compact-stage-mib",
            "384",
            "--prefill-copy-chunk-mib",
            "4",
            "--prefill-stage-disk-margin-mib",
            "1024",
            "--prefill-max-routed-read-amplification",
            "1.75",
            "--prefill-max-routed-read-gib",
            "0.5",
            "--prefill-ssd-read-gib-s",
            "16",
            "--prefill-max-routed-read-seconds",
            "5",
            "--expert-read-advise-merge-gap-kib",
            "128",
            "--expert-read-advise-align-kib",
            "4",
            "--prefill-moe-token-block",
            "16",
            "--prefill-mla-key-cache",
            "--decode-mla-key-cache",
            "--prefill-mla-kv-b-cache-dir",
            str(cache_dir),
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpp-probe",
            "--run-mpsgraph-probe",
            "--prefill-backend-probe-timeout-seconds",
            "12.5",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-router-hybrid-margin-threshold",
            "1e-05",
            "--metal-final-logits",
            "--quiet-runner",
        ]
    )

    assert status == 0
    config = captured["config"]
    assert config.prefill_prompt_chunk_tokens == 32
    assert config.prefill_max_prompt_batch_mib == 256.0
    assert config.prefill_max_cache_write_mib == 512.0
    assert config.prefill_max_stage_mib == 768.0
    assert config.prefill_max_compact_stage_mib == 384.0
    assert config.prefill_copy_chunk_mib == 4.0
    assert config.prefill_stage_disk_margin_mib == 1024.0
    assert config.prefill_max_routed_read_amplification == 1.75
    assert config.prefill_max_routed_read_gib == 0.5
    assert config.prefill_ssd_read_gib_per_second == 16.0
    assert config.prefill_max_routed_read_seconds == 5.0
    assert config.expert_read_advise_merge_gap_kib == 128
    assert config.expert_read_advise_align_kib == 4
    assert config.prefill_moe_token_block == 16
    assert config.prefill_mla_key_cache is True
    assert config.decode_mla_key_cache is True
    assert config.prefill_mla_kv_b_cache_dir == cache_dir
    assert config.prefill_linear_backend == "mpsgraph-f32"
    assert config.require_prefill_acceleration is True
    assert config.prefill_run_mpp_probe is True
    assert config.prefill_run_mpsgraph_probe is True
    assert config.prefill_backend_probe_timeout_seconds == 12.5
    assert config.prefill_mpsgraph_min_batch_tokens == 64
    assert config.prefill_mpsgraph_min_matrix_dim == 16
    assert config.prefill_router_hybrid_margin_threshold == 1e-5
    assert config.metal_final_logits is True


def test_serve_prepared_check_ssd_read_speed_allows_current_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_cold_read_profile(prepared, gib_per_second=8.0)
    fake_benchmark = _fake_sequential_read_benchmark(6.5)
    captured: dict[str, object] = {}

    def fake_run_prepared_server(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("largerlm.cli.benchmark_sequential_read", fake_benchmark)
    monkeypatch.setattr("largerlm.cli.run_prepared_server", fake_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-ssd-read-speed",
            "--ssd-read-speed-min-ratio",
            "0.75",
            "--ssd-read-speed-bytes-mib",
            "1",
            "--ssd-read-speed-chunk-mib",
            "1",
            "--ssd-read-speed-max-chunk-mib",
            "2",
            "--quiet-runner",
        ]
    )

    assert status == 0
    assert captured["config"].prepared_path == prepared
    assert fake_benchmark.calls == [  # type: ignore[attr-defined]
        {
            "path": prepared / "experts" / "layer_000.bin",
            "bytes_to_read": 1024**2,
            "chunk_bytes": 1024**2,
            "offset_bytes": 0,
            "max_chunk_bytes": 2 * 1024**2,
        }
    ]


def test_serve_prepared_check_ssd_read_speed_rejects_slow_current_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_cold_read_profile(prepared, gib_per_second=8.0)

    def fail_run_prepared_server(config) -> None:
        del config
        raise AssertionError("server should not start")

    monkeypatch.setattr(
        "largerlm.cli.benchmark_sequential_read",
        _fake_sequential_read_benchmark(5.5),
    )
    monkeypatch.setattr("largerlm.cli.run_prepared_server", fail_run_prepared_server)

    status = cli_main(
        [
            "serve-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-ssd-read-speed",
            "--ssd-read-speed-min-ratio",
            "0.75",
            "--ssd-read-speed-bytes-mib",
            "1",
            "--ssd-read-speed-chunk-mib",
            "1",
            "--quiet-runner",
        ]
    )

    assert status == 1
    err = capsys.readouterr().err
    assert "current SSD read speed check failed before starting serve-prepared" in err
    assert "actual_gib_per_second=5.5" in err
    assert "required_gib_per_second=6.0" in err


def test_inspect_prepared_cli_reports_health_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recommended_max_live_working_set_bytes"] = 7 * 1024**3
    payload["recommended_min_free_unified_memory_bytes"] = 20 * 1024**3
    payload["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    payload["prepare_effective_unified_memory_source"] = "explicit"
    payload["prepare_system_reserve_bytes"] = 24 * 1024**3
    payload["prepare_auto_context_from_budget"] = True
    payload["prepare_resolved_max_context_tokens"] = 4
    payload["prepare_decode_cache_budget_bytes"] = 64
    payload["prepare_decode_cache_safe_context_tokens"] = 8
    payload["prepare_effective_max_cache_bytes"] = 64
    payload["prepare_cache_dtype"] = "BF16"
    payload["prepare_cache_alignment"] = 64
    payload["prepare_cold_read_gib_per_second"] = 16.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "largerlm.server._system_disk_usage",
        lambda path: {
            "path": str(path),
            "total_bytes": 1024**3,
            "used_bytes": 1024,
            "free_bytes": 1024**3 - 1024,
        },
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )
    runtime_capture: dict[str, object] = {}

    def fake_check_generation_runtime(**kwargs):
        runtime_capture.update(kwargs)
        return SimpleNamespace(
            requested_context_tokens=kwargs["requested_context_tokens"],
            layers=(0,),
            dense_layers=tuple(kwargs["dense_layers"] or ()),
            max_layer_peak_bytes=1234,
            max_layer_cache_read_bytes=456,
            read_bytes_per_token=789,
            final_logits_budget=SimpleNamespace(estimated_peak_bytes=2048),
            embedding_budget=SimpleNamespace(row_bytes=32, output_bytes=32),
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=64 * 1024**2,
                max_live_working_set_bytes=7 * 1024**3,
                min_available_memory_bytes=20 * 1024**3,
                system_available_bytes=96 * 1024**3,
                system_total_bytes=128 * 1024**3,
                system_source="test",
            ),
        )

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fake_check_generation_runtime,
    )
    profile_path = tmp_path / "launch-profile.json"

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--check-prompt-tokens",
            "2",
            "--check-max-new-tokens",
            "1",
            "--prefill-static-capacity-per-expert",
            "auto",
            "--check-runtime-preflight",
            "--write-launch-profile",
            str(profile_path),
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert "--prefill-static-capacity-per-expert" in profile["argv"]
    replay_status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--apply-launch-profile",
            str(profile_path),
            "--lock-launch-profile",
            "--require-locked-launch-profile",
            "--json",
        ]
    )
    assert replay_status == 0
    replay_health = json.loads(capsys.readouterr().out)
    assert replay_health["applied_launch_profile"]["locked"] is True
    assert health["max_live_working_set_mib"] == 7 * 1024
    assert health["min_free_unified_memory_gib"] == 20
    assert health["prefill_ssd_read_gib_per_second"] == 16.0
    assert health["system_memory"]["available_bytes"] == 96 * 1024**3
    assert health["memory_guard"] == {
        "configured_max_live_working_set_bytes": 7 * 1024**3,
        "configured_min_free_unified_memory_bytes": 20 * 1024**3,
        "configured_required_available_memory_bytes": 27 * 1024**3,
        "system_available_memory_bytes": 96 * 1024**3,
        "system_total_memory_bytes": 128 * 1024**3,
        "system_memory_source": "test",
        "available_ok": True,
    }
    assert health["prefill_backend"]["configured_backend"] == "mpsgraph-f32"
    assert health["prefill_backend"]["warnings"] == []
    assert health["prepared_storage"] == {
        "prepared_storage_validated": True,
        "expert_layout_backing_validated": True,
        "resident_layout_backing_validated": True,
        "decode_cache_file_exact_size": True,
        "expert_layout_bytes": 16,
        "resident_layout_bytes": 16,
        "decode_cache_layout_bytes": 32,
        "decode_cache_file_bytes": 32,
        "decode_cache_file_extra_bytes": 0,
        "prepared_total_layout_bytes": 64,
        "prepared_total_file_bytes": 64,
        "resident_and_cache_layout_bytes": 48,
        "recommended_max_live_working_set_bytes": 7 * 1024**3,
        "recommended_min_free_unified_memory_bytes": 20 * 1024**3,
        "recommended_required_available_memory_bytes": 27 * 1024**3,
        "expert_quantization": None,
        "expert_group_size": None,
        "expert_layout_quantization": "mlx-affine-int4",
        "expert_layout_group_size": 8,
        "prepare_hardware_chip_name": None,
        "prepare_hardware_unified_memory_bytes": None,
        "prepare_hardware_gpu_cores": None,
        "prepare_hardware_apple_silicon_generation": None,
        "prepare_hardware_apple_silicon_tier": None,
        "prepare_effective_unified_memory_bytes": 128 * 1024**3,
        "prepare_effective_unified_memory_source": "explicit",
        "prepare_system_reserve_bytes": 24 * 1024**3,
        "prepare_auto_context_from_budget": True,
        "prepare_requested_max_context_tokens": None,
        "prepare_resolved_max_context_tokens": 4,
        "prepare_decode_cache_budget_bytes": 64,
        "prepare_decode_cache_safe_context_tokens": 8,
        "prepare_effective_max_cache_bytes": 64,
        "prepare_model_max_position_embeddings": None,
        "prepare_cache_dtype": "BF16",
        "prepare_cache_alignment": 64,
        "prepare_flags_applied": None,
        "prepare_flags_source": None,
        "prepare_flags_path": None,
        "prepare_flags_sha256": None,
        "prepare_expert_pack_chunk_size_bytes": None,
        "prepare_expert_pack_estimated_peak_heap_bytes": None,
        "prepare_expert_pack_max_heap_bytes": None,
        "prepare_raw_quantization_extra_heap_bytes": None,
        "prepare_raw_quantization_max_source_block_bytes": None,
        "prepare_raw_quantization_max_output_block_bytes": None,
        "prepare_raw_quantization_max_rows_per_block": None,
        "prepare_resident_component_alias_source_tensor_count": None,
        "prepare_resident_component_alias_renamed_tensor_count": None,
        "prepare_resident_component_alias_bytes": None,
        "prepare_resident_fused_gate_up_source_tensor_count": None,
        "prepare_resident_fused_gate_up_expanded_tensor_count": None,
        "prepare_resident_fused_gate_up_expanded_bytes": None,
        "prepare_public_glm_5_2_shape_required": None,
        "prepare_public_glm_5_2_shape_matches": None,
        "prepare_public_glm_5_2_shape_mismatched_fields": None,
        "prepare_cold_read_gib_per_second": 16.0,
        "prepare_cold_read_source": None,
        "prepare_cold_read_benchmark_path": None,
        "prepare_cold_read_benchmark_requested_bytes": None,
        "prepare_cold_read_benchmark_measured_bytes": None,
        "prepare_cold_read_benchmark_elapsed_seconds": None,
        "model_config_sha256": None,
    }
    assert health["prepared_runtime_profile"] == {
        "prepare_effective_unified_memory_bytes": 128 * 1024**3,
        "prepare_effective_unified_memory_source": "explicit",
        "prepare_system_reserve_bytes": 24 * 1024**3,
        "prepared_recommended_max_live_working_set_bytes": 7 * 1024**3,
        "prepared_recommended_min_free_unified_memory_bytes": 20 * 1024**3,
        "prepared_recommended_required_available_memory_bytes": 27 * 1024**3,
        "system_total_memory_bytes": 128 * 1024**3,
        "system_available_memory_bytes": 96 * 1024**3,
        "system_memory_source": "test",
        "system_total_meets_prepare_effective_unified_memory": True,
        "system_available_meets_prepare_system_reserve": True,
        "system_available_meets_prepared_recommended_required_available": True,
        "profile_ok": True,
        "warnings": [],
    }
    ssd_flags = health["suggested_prepared_ssd_read_flags"]
    assert ssd_flags == {
        "source": "prepared_manifest",
        "prefill_ssd_read_gib_per_second": 16.0,
        "prepare_cold_read_gib_per_second": 16.0,
        "prepare_cold_read_source": None,
        "matches_prepare_cold_read": True,
        "argv": ["--prefill-ssd-read-gib-s", "16"],
    }
    assert health["request_check"]["ok"] is True
    assert health["request_check"]["prompt_token_count"] == 2
    assert health["request_check"]["max_new_tokens"] == 1
    assert health["request_check"]["required_context_tokens"] == 3
    assert health["request_check"]["batch_prefill_prompt"] is True
    assert health["request_check"]["prefill_prompt_chunk_tokens"]["resolved"] == 2
    cache_io = health["request_check"]["prefill_cache_io"]
    assert cache_io["mla_cache_width"] == 4
    assert cache_io["indexed_attention_layers"] == 0
    assert cache_io["full_attention_layers"] == 2
    assert cache_io["causal_rows_per_layer"] == 3
    assert cache_io["mla_cache_read_bytes"] == 48
    assert cache_io["total_cache_read_bytes"] == 48
    assert cache_io["mla_cache_write_bytes"] == 32
    assert cache_io["total_cache_write_bytes"] == 32
    frontier = health["request_check"]["prefill_routed_chunk_frontier"]
    assert frontier["analyzed"] is True
    assert frontier["resolved_prompt_chunk_tokens"] == 2
    assert frontier["prompt_token_count"] == 2
    assert frontier["saturation_chunk_tokens"] == 1
    assert any(
        candidate["prompt_chunk_tokens"] == 2
        and candidate["planned_read_bytes"] == 16
        and candidate["total_static_capacity_binary_bytes"] == 68
        and candidate["total_stage_plus_compact_plus_static_bytes"] == 4196
        for candidate in frontier["candidates"]
    )
    assert health["prefill_backend"]["auto_policy"]["mpsgraph_min_batch_tokens"] == 64
    assert health["prefill_backend"]["auto_policy"]["mpsgraph_min_matrix_dim"] == 16
    policy = health["suggested_prefill_runtime_policy_flags"]
    assert policy == {
        "source": "prepared_health",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_mpsgraph_min_batch_tokens": 64,
        "prefill_mpsgraph_min_matrix_dim": 16,
        "argv": [
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
        ],
    }
    assert health["request_check"]["runtime_preflight"]["ran"] is True
    assert health["request_check"]["runtime_preflight"]["live_working_set_bytes"] == (
        64 * 1024**2
    )
    assert health["request_check"]["runtime_preflight"]["embedding_row_bytes"] == 32
    assert health["request_check"]["runtime_preflight"]["embedding_output_bytes"] == 32
    assert health["request_check"]["runtime_preflight"][
        "required_available_memory_bytes"
    ] == (20 * 1024**3 + 64 * 1024**2)
    assert health["request_check"]["runtime_preflight"][
        "system_available_memory_bytes"
    ] == 96 * 1024**3
    assert health["request_check"]["runtime_preflight"][
        "system_total_memory_bytes"
    ] == 128 * 1024**3
    assert health["request_check"]["runtime_preflight"]["system_memory_source"] == "test"
    assert health["request_check"]["runtime_preflight"]["available_memory_ok"] is True
    prefill_live = health["request_check"]["runtime_preflight"]["prefill_live_memory"]
    assert prefill_live["prompt_batch_bytes"] == 1024 * 1024**2
    assert prefill_live["runner_scratch_bytes"] == 4096 * 1024**2
    assert prefill_live["cache_read_bytes"] == 256 * 1024**2
    assert prefill_live["cache_write_bytes"] == 4096 * 1024**2
    assert prefill_live["stage_copy_bytes"] == 8 * 1024**2
    assert prefill_live["estimated_live_working_set_bytes"] == 8192 * 1024**2
    assert runtime_capture["requested_context_tokens"] == 3
    assert runtime_capture["max_live_working_set_mib"] == 7 * 1024
    assert runtime_capture["min_free_unified_memory_mib"] == 20 * 1024
    assert runtime_capture["extra_live_working_set_bytes"] == 8192 * 1024**2
    profile = json.loads((tmp_path / "launch-profile.json").read_text(encoding="utf-8"))
    assert profile == health["request_launch_profile"]
    assert profile["source"] == "prepared_request_check"
    assert profile["argv_safe_to_replay"] is True
    assert profile["prepared"]["expert_layout_bytes"] == 16
    assert profile["prepared"]["decode_cache_file_bytes"] == 32
    assert profile["sections"]["prefill_backend_policy_flags"] == {
        "source": "prepared_request_check",
        "prefill_linear_backend": "mpsgraph-f32",
        "argv": ["--prefill-linear-backend", "mpsgraph-f32"],
    }
    assert profile["sections"]["prefill_runtime_policy_flags"] == policy
    assert profile["sections"]["prepared_ssd_read_flags"] == ssd_flags
    assert "--prefill-prompt-chunk-tokens" in profile["argv"]
    assert "--prefill-ssd-read-gib-s" in profile["argv"]
    assert "--prefill-linear-backend" in profile["argv"]
    assert "--prefill-mpsgraph-min-batch-tokens" in profile["argv"]
    assert "--max-live-working-set-mib" in profile["argv"]


def test_inspect_prepared_cli_reports_applied_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--max-live-working-set-mib",
        "8192",
        "--min-free-unified-memory-gib",
        "0",
    ]
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=profile_argv,
        sections={"launch_guard_flags": {"source": "unit"}},
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    applied = health["applied_launch_profile"]
    assert applied["path"] == str(profile)
    assert applied["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert applied["source"] == "unit"
    assert applied["argv"] == profile_argv
    assert applied["locked"] is False
    assert applied["lock_required"] is False
    assert applied["profile_flag_count"] == 2
    assert applied["lock_checked_flags"] == []
    assert applied["section_names"] == ["launch_guard_flags"]
    assert applied["matches_prepared"] is True
    assert applied["prepared"]["decode_cache_file_bytes"] == 32
    assert health["max_live_working_set_mib"] == 8192.0
    assert health["min_free_unified_memory_gib"] == 0.0


def test_inspect_prepared_cli_require_launch_audit_reports_missing_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-launch-audit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    audit = health["launch_audit"]
    assert audit["ok"] is False
    failures = set(audit["failures"])
    assert "locked_launch_profile" in failures
    assert "prepared_identity_strong" in failures
    assert "prepared_memory_profile_required" in failures
    assert "prepared_memory_profile_ok" in failures
    assert "prepared_context_budget_profile_ok" in failures
    assert "memory_guard_free_reserve" in failures
    assert "glm_4bit_required" in failures
    assert "public_glm_5_2_shape_required" in failures
    assert "prefill_acceleration_required" in failures
    assert "request_check_requested" in failures
    assert "request_runtime_preflight_ran" in failures


def test_inspect_prepared_cli_launch_audit_reports_prepare_flags_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    flags = tmp_path / "prepare-flags.json"
    flags.write_text(
        json.dumps(
            {
                "source": "plan",
                "argv_safe_to_replay": True,
                "argv": ["--auto-context-from-budget"],
            }
        ),
        encoding="utf-8",
    )
    expected_sha256 = hashlib.sha256(flags.read_bytes()).hexdigest()
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_flags_applied"] = True
    manifest["prepare_flags_source"] = "plan"
    manifest["prepare_flags_path"] = str(flags)
    manifest["prepare_flags_sha256"] = expected_sha256
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-launch-audit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    audit = health["launch_audit"]
    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "prepare_flags_provenance_ok"
    )
    assert check["ok"] is True
    assert check["prepare_flags_source"] == "plan"
    assert check["prepare_flags_path"] == str(flags)
    assert check["prepare_flags_sha256"] == expected_sha256


def test_launch_audit_records_public_glm_5_2_dsa_schedule_evidence() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=None,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    public_shape = {
        "matches": True,
        "mismatched_fields": (),
        "dsa_full_indexer_layer_count": 21,
        "expected_dsa_full_indexer_layer_count": 21,
        "dsa_full_indexer_layers": (0, 1, 2, 6, 10, 14, 18, 22),
        "dsa_schedule": {
            "index_topk_freq": 4,
            "index_skip_topk_offset": 3,
            "full_indexer_layer_count": 21,
            "first_full_indexer_layers": (0, 1, 2, 6, 10, 14, 18, 22),
            "last_full_indexer_layers": (62, 66, 70, 74),
        },
        "checks": {
            "index_topk_freq": {"actual": 4, "expected": 4, "matches": True},
            "index_skip_topk_offset": {
                "actual": 3,
                "expected": 3,
                "matches": True,
            },
            "num_nextn_predict_layers": {
                "actual": 1,
                "expected": 1,
                "matches": True,
            },
            "full_indexer_layers": {
                "actual": (0, 1, 2, 6),
                "expected": (0, 1, 2, 6),
                "matches": True,
            },
        },
    }
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
            "argv": (),
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_memory_profile_requirement": {"ok": True},
        "prepared_context_budget_requirement": {"ok": True},
        "prepared_runtime_profile": {"profile_ok": True},
        "memory_guard": {
            "available_ok": True,
            "configured_min_free_unified_memory_bytes": 1,
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
            "decode_cache_file_extra_bytes": 0,
        },
        "glm_4bit_readiness": {
            "ok": True,
            "matches_public_glm_5_2_shape": True,
            "public_glm_5_2_shape": public_shape,
        },
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "public_glm_5_2_shape_ok"
    )
    assert check["ok"] is True
    assert check["public_glm_5_2_shape_mismatched_fields"] == ()
    assert check["dsa_full_indexer_layer_count"] == 21
    assert check["expected_dsa_full_indexer_layer_count"] == 21
    assert check["dsa_schedule"]["index_topk_freq"] == 4
    assert check["dsa_schedule"]["index_skip_topk_offset"] == 3
    assert check["public_glm_5_2_shape_check_index_topk_freq"] == {
        "actual": 4,
        "expected": 4,
        "matches": True,
    }
    assert check["public_glm_5_2_shape_check_num_nextn_predict_layers"] == {
        "actual": 1,
        "expected": 1,
        "matches": True,
    }


def test_launch_audit_fails_without_prepared_storage_validation() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=False,
        require_glm_4bit=False,
        require_public_glm_5_2_shape=False,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=None,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": False,
            "decode_cache_file_extra_bytes": 64,
        },
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item for item in audit["checks"] if item["code"] == "prepared_storage_validated"
    )
    assert check["ok"] is False
    assert check["decode_cache_file_exact_size"] is False
    assert check["decode_cache_file_extra_bytes"] == 64
    assert "prepared_storage_validated" in audit["failures"]


def test_launch_audit_fails_internal_int4_without_prepare_pack_heap_evidence() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=False,
        require_glm_4bit=False,
        require_public_glm_5_2_shape=False,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=None,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
            "expert_quantization": "largerlm-affine-int4",
            "expert_group_size": 8,
        },
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "prepare_expert_pack_heap_envelope_ok"
    )
    assert check["ok"] is False
    assert check["required"] is True
    assert check["evidence_present"] is False
    assert "prepare_expert_pack_chunk_size_bytes" in check["missing_fields"]
    assert "prepare_expert_pack_heap_envelope_ok" in audit["failures"]


def test_launch_audit_fails_bad_resident_alias_rewrite_evidence() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=False,
        require_glm_4bit=False,
        require_public_glm_5_2_shape=False,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=None,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
            "prepare_resident_component_alias_source_tensor_count": 3,
            "prepare_resident_component_alias_renamed_tensor_count": 2,
            "prepare_resident_component_alias_bytes": 1024,
        },
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "prepare_resident_alias_rewrite_ok"
    )
    assert check["ok"] is False
    assert check["rewrite_present"] is True
    assert check["alias_required"] is True
    assert check["alias_counts_match"] is False
    assert "prepare_resident_alias_rewrite_ok" in audit["failures"]


def test_launch_audit_requires_prefill_routed_read_seconds_budget() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=64,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
            "argv": (),
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
        },
        "prepared_memory_profile_requirement": {"ok": True},
        "prepared_context_budget_requirement": {"ok": True},
        "prepared_runtime_profile": {"profile_ok": True},
        "memory_guard": {
            "available_ok": True,
            "configured_min_free_unified_memory_bytes": 1,
        },
        "glm_4bit_readiness": {
            "ok": True,
            "matches_public_glm_5_2_shape": True,
        },
        "request_check": {
            "ok": True,
            "batch_prefill_prompt": True,
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
            },
            "prefill_routed_expert_read": {
                "analyzed": True,
                "within_limit": True,
                "baseline_read_bytes": 1024,
                "planned_read_bytes": 1024,
                "extra_read_bytes": 0,
                "read_amplification": 1.0,
                "max_read_amplification": 0.0,
                "within_amplification_limit": True,
                "max_planned_read_bytes": 0,
                "within_planned_read_limit": True,
                "ssd_read_gib_per_second": 16.0,
                "planned_read_seconds": 1024 / (16.0 * 1024**3),
                "max_read_seconds": 0.0,
                "within_seconds_limit": True,
            },
            "prefill_routed_stage_temp_disk": {
                "analyzed": True,
                "within_limit": True,
            },
            "prefill_stage_temp_disk_free": {
                "analyzed": True,
                "within_free_space": True,
            },
            "prefill_acceleration_coverage": (
                _launch_audit_prefill_acceleration_coverage()
            ),
        },
        "request_launch_profile": {"argv_safe_to_replay": True},
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "request_prefill_routed_read_budget_ok"
    )
    assert check["ok"] is False
    assert check["required"] is True
    assert check["ssd_read_gib_per_second"] == 16.0
    assert check["max_read_seconds"] == 0.0
    assert "request_prefill_routed_read_budget_ok" in audit["failures"]


def test_launch_audit_requires_static_capacity_stage_temp_evidence() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=64,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    stage_temp = _launch_audit_stage_temp_evidence()
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
            "argv": (),
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
        },
        "prepared_memory_profile_requirement": {"ok": True},
        "prepared_context_budget_requirement": {"ok": True},
        "prepared_runtime_profile": {"profile_ok": True},
        "memory_guard": {
            "available_ok": True,
            "configured_min_free_unified_memory_bytes": 1,
        },
        "glm_4bit_readiness": {
            "ok": True,
            "matches_public_glm_5_2_shape": True,
        },
        "request_check": {
            "ok": True,
            "batch_prefill_prompt": True,
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
            },
            "prefill_routed_stage_temp_disk": stage_temp,
            "prefill_stage_temp_disk_free": {
                "analyzed": True,
                "path": "/private/tmp",
                "required_stage_temp_bytes": 4940,
                "disk_safety_margin_bytes": 1024,
                "required_free_bytes": 5964,
                "within_free_space": True,
                "free_bytes": 1024**3,
            },
            "prefill_acceleration_coverage": (
                _launch_audit_prefill_acceleration_coverage()
            ),
        },
        "request_launch_profile": {"argv_safe_to_replay": True},
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "request_prefill_stage_temp_limit_ok"
    )
    assert check["ok"] is True
    assert all(field in stage_temp for field in _REQUEST_PREFILL_STAGE_TEMP_AUDIT_FIELDS)
    assert check["max_stage_plus_compact_bytes"] == 4128
    assert check["max_static_capacity_binary_bytes"] == 812
    assert check["max_stage_plus_compact_plus_static_bytes"] == 4940
    assert check["static_capacity_per_expert"] == "auto"
    assert check["allow_static_capacity_overflow"] is False
    assert "errors" not in check


def test_launch_audit_rejects_legacy_stage_temp_without_static_envelope() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=64,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    legacy_stage_temp = {
        "analyzed": True,
        "prompt_chunk_tokens": 64,
        "top_k": 2,
        "max_stage_bytes": 4096,
        "max_stage_limit_bytes": 8192,
        "within_stage_limit": True,
        "max_compact_stage_bytes": 32,
        "max_compact_stage_limit_bytes": 1024,
        "within_compact_stage_limit": True,
        "within_limit": True,
    }
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
            "argv": (),
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
        },
        "prepared_memory_profile_requirement": {"ok": True},
        "prepared_context_budget_requirement": {"ok": True},
        "prepared_runtime_profile": {"profile_ok": True},
        "memory_guard": {
            "available_ok": True,
            "configured_min_free_unified_memory_bytes": 1,
        },
        "glm_4bit_readiness": {
            "ok": True,
            "matches_public_glm_5_2_shape": True,
        },
        "request_check": {
            "ok": True,
            "batch_prefill_prompt": True,
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
            },
            "prefill_routed_stage_temp_disk": legacy_stage_temp,
            "prefill_stage_temp_disk_free": {
                "analyzed": True,
                "path": "/private/tmp",
                "required_stage_temp_bytes": 4128,
                "disk_safety_margin_bytes": 1024,
                "required_free_bytes": 5152,
                "within_free_space": True,
                "free_bytes": 1024**3,
            },
            "prefill_acceleration_coverage": (
                _launch_audit_prefill_acceleration_coverage()
            ),
        },
        "request_launch_profile": {"argv_safe_to_replay": True},
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "request_prefill_stage_temp_limit_ok"
    )
    assert check["ok"] is False
    assert "request_prefill_stage_temp_limit_ok" in audit["failures"]
    assert "missing=max_stage_plus_compact_plus_static_bytes" in check["errors"]


def test_launch_audit_stage_temp_profile_allows_prompt_only_request() -> None:
    stage_temp = _launch_audit_stage_temp_evidence()
    artifact = {
        "request_check": {
            "prefill_routed_stage_temp_disk": stage_temp,
        },
    }
    audit = {
        "checks": (
            {
                "code": "request_prefill_stage_temp_limit_ok",
                "required": True,
            },
        )
    }
    request_profile = {
        "argv": (
            "--prefill-max-stage-mib",
            str(stage_temp["max_stage_bytes"] / 1024**2),
            "--prefill-max-compact-stage-mib",
            str(stage_temp["max_compact_stage_bytes"] / 1024**2),
            "--prefill-max-stage-raw-ranges",
            str(stage_temp["max_stage_raw_ranges"]),
            "--prefill-max-stage-coalesced-ranges",
            str(stage_temp["max_stage_coalesced_ranges"]),
            "--prefill-static-capacity-per-expert",
            "auto",
        )
    }

    _require_launch_audit_request_profile_binds_read_budgets(
        artifact,
        audit,
        request_profile,
    )


def test_launch_audit_records_applied_benchmark_prefill_read_time_evidence() -> None:
    actual_read_time = _launch_audit_prefill_actual_read_time(prompt_token_count=64)
    actual_acceleration_coverage = _launch_audit_prefill_acceleration_coverage()
    actual_linear_backend = _launch_audit_prefill_actual_linear_backend()
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=64,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
            "source": "benchmark_actual",
            "argv": (),
            "prefill_actual_read_time": actual_read_time,
            "prefill_actual_acceleration_coverage": actual_acceleration_coverage,
            "prefill_actual_linear_backend": actual_linear_backend,
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
        },
        "prepared_memory_profile_requirement": {"ok": True},
        "prepared_context_budget_requirement": {"ok": True},
        "prepared_runtime_profile": {"profile_ok": True},
        "memory_guard": {
            "available_ok": True,
            "configured_min_free_unified_memory_bytes": 1,
        },
        "glm_4bit_readiness": {
            "ok": True,
            "matches_public_glm_5_2_shape": True,
        },
        "request_check": {
            "ok": True,
            "batch_prefill_prompt": True,
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
            },
            "prefill_linear_backend": {
                "configured": "auto",
                "effective": "auto",
                "analyzed": True,
            },
            "prefill_routed_expert_read": {
                "analyzed": True,
                "within_limit": True,
                "baseline_read_bytes": 64 * 1024,
                "planned_read_bytes": 64 * 1024,
                "extra_read_bytes": 0,
                "read_amplification": 1.0,
                "max_read_amplification": 2.0,
                "within_amplification_limit": True,
                "max_planned_read_bytes": 128 * 1024,
                "within_planned_read_limit": True,
                "ssd_read_gib_per_second": 16.0,
                "planned_read_seconds": actual_read_time[
                    "total_expert_stage_planned_read_seconds"
                ],
                "max_read_seconds": 5.0,
                "within_seconds_limit": True,
            },
            "prefill_routed_stage_temp_disk": {
                "analyzed": True,
                "within_limit": True,
            },
            "prefill_stage_temp_disk_free": {
                "analyzed": True,
                "within_free_space": True,
            },
            "prefill_acceleration_coverage": (
                _launch_audit_prefill_acceleration_coverage()
            ),
        },
        "request_launch_profile": {"argv_safe_to_replay": True},
        "prefill_backend": {"effective_backend": "auto"},
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "applied_prefill_actual_read_time_ok"
    )
    assert check["ok"] is True
    assert check["required"] is True
    assert check["evidence_present"] is True
    assert check["source"] == "benchmark_actual_prefill"
    assert check["total_expert_stage_planned_read_bytes"] == 64 * 1024
    assert check["prefill_max_routed_read_seconds"] == 5.0
    assert check["total_expert_stage_copy_seconds_ok"] is True
    actual_accel_check = next(
        item
        for item in audit["checks"]
        if item["code"] == "applied_prefill_actual_acceleration_coverage_valid"
    )
    assert actual_accel_check["ok"] is True
    assert actual_accel_check["required"] is False
    assert actual_accel_check["evidence_present"] is True
    assert actual_accel_check["accelerated_backends"] == ("mpsgraph-f32",)
    assert actual_accel_check["accelerated_flop_fraction"] == 1.0
    linear_check = next(
        item
        for item in audit["checks"]
        if item["code"] == "applied_prefill_actual_linear_backend_valid"
    )
    assert linear_check["ok"] is True
    assert linear_check["required"] is False
    assert linear_check["evidence_present"] is True
    assert linear_check["configured_backend"] == "auto"
    assert linear_check["linear_backend_elapsed_seconds"] == {
        "mpsgraph-f32": 0.002
    }
    coverage_check = next(
        item
        for item in audit["checks"]
        if item["code"] == "request_prefill_acceleration_coverage_ok"
    )
    assert coverage_check["ok"] is True
    assert coverage_check["evidence_present"] is True
    assert coverage_check["accelerated_backends"] == ("mpsgraph-f32",)
    assert coverage_check["total_estimated_flops"] == 4096
    assert coverage_check["accelerated_flop_fraction"] == 1.0


def test_launch_audit_requires_request_prefill_backend_effective_match() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=2,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "prefill_backend": {
            "configured_backend": "auto",
            "effective_backend": "custom-metal",
        },
        "request_check": {
            "ok": True,
            "batch_prefill_prompt": True,
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
            },
            "prefill_linear_backend": {
                "configured": "auto",
                "effective": "auto",
                "analyzed": True,
            },
            "prefill_acceleration_coverage": (
                _launch_audit_prefill_acceleration_coverage()
            ),
        },
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "request_prefill_backend_effective_ok"
    )
    assert check["ok"] is False
    assert check["required"] is True
    assert check["effective"] == "auto"
    assert check["health_effective_backend"] == "custom-metal"
    assert "request_prefill_backend_effective_ok" in audit["failures"]


def test_launch_audit_requires_decode_routed_read_seconds_budget() -> None:
    args = SimpleNamespace(
        require_prepared_memory_profile=True,
        require_glm_4bit=True,
        require_public_glm_5_2_shape=True,
        require_prefill_acceleration=False,
        prefill_min_accelerated_flop_fraction=0.0,
        check_prompt_tokens=1,
        check_prompt=None,
        check_prompt_file=None,
        check_chat_messages=None,
        check_chat_messages_file=None,
    )
    health = {
        "applied_launch_profile": {
            "locked": True,
            "profile_flag_count": 1,
            "argv": (),
        },
        "suggested_launch_profile": {
            "prepared": {"identity_strength": "strong"},
        },
        "prepared_storage": {
            "prepared_storage_validated": True,
            "expert_layout_backing_validated": True,
            "resident_layout_backing_validated": True,
            "decode_cache_file_exact_size": True,
        },
        "prepared_memory_profile_requirement": {"ok": True},
        "prepared_context_budget_requirement": {"ok": True},
        "prepared_runtime_profile": {"profile_ok": True},
        "memory_guard": {
            "available_ok": True,
            "configured_min_free_unified_memory_bytes": 1,
        },
        "glm_4bit_readiness": {
            "ok": True,
            "matches_public_glm_5_2_shape": True,
        },
        "request_check": {
            "ok": True,
            "prompt_token_count": 1,
            "max_new_tokens": 1,
            "batch_prefill_prompt": False,
            "runtime_preflight": {
                "ran": True,
                "available_memory_ok": True,
            },
            "decode_routed_expert_read": {
                "analyzed": True,
                "read_bytes_per_token": 1024,
                "max_read_bytes_per_token": 2048,
                "within_read_limit": True,
                "ssd_read_gib_per_second": 16.0,
                "planned_read_seconds_per_token": 1024 / (16.0 * 1024**3),
                "max_read_seconds_per_token": 0.0,
                "within_seconds_limit": True,
                "within_limit": True,
            },
            "prefill_routed_stage_temp_disk": {
                "analyzed": True,
                "within_limit": True,
            },
            "prefill_stage_temp_disk_free": {
                "analyzed": True,
                "within_free_space": True,
            },
            "prefill_acceleration_coverage": (
                _launch_audit_prefill_acceleration_coverage()
            ),
        },
        "request_launch_profile": {"argv_safe_to_replay": True},
    }

    audit = _launch_audit_from_health(args, health)

    check = next(
        item
        for item in audit["checks"]
        if item["code"] == "request_decode_routed_read_budget_ok"
    )
    assert check["ok"] is False
    assert check["required"] is True
    assert check["ssd_read_gib_per_second"] == 16.0
    assert check["max_read_seconds_per_token"] == 0.0
    assert "request_decode_routed_read_budget_ok" in audit["failures"]


def test_inspect_prepared_cli_require_launch_audit_requires_mpsgraph_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _make_prepared_glm_4bit_ready(prepared)
    _add_prepared_context_budget_metadata(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 24 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 9 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 20 * 1024**3
    manifest["prepare_cold_read_gib_per_second"] = 16.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            mps_graph_probe_ran=False,
            mps_graph_probe_ok=None,
            mps_graph_probe_error=None,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--max-live-working-set-mib",
            str(9 * 1024),
            "--min-free-unified-memory-gib",
            "20",
            "--require-prepared-memory-profile",
            "--require-glm-4bit",
            "--require-public-glm-5-2-shape",
            "--require-prefill-acceleration",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "1",
            "--prefill-ssd-read-gib-s",
            "16",
            "--prefill-max-routed-read-seconds",
            "5",
        ],
    )

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "64",
            "--check-max-new-tokens",
            "1",
            "--check-runtime-preflight",
            "--require-launch-audit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    audit = health["launch_audit"]
    assert "prefill_acceleration_probe_ok" in audit["failures"]
    assert "prefill_acceleration_profile_replays_probe" in audit["failures"]
    probe_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prefill_acceleration_probe_ok"
    )
    assert probe_check["ok"] is False
    assert probe_check["validated_accelerated_prefill_backends"] == []
    assert probe_check["validated_prefill_acceleration_available"] is False
    assert probe_check["mps_graph_probe_requested"] is False
    assert probe_check.get("mps_graph_probe_ok") is None
    replay_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prefill_acceleration_profile_replays_probe"
    )
    assert replay_check["ok"] is False
    assert replay_check["has_run_mpsgraph_probe"] is False


def test_inspect_prepared_cli_require_launch_audit_rejects_profile_without_probe_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _make_prepared_glm_4bit_ready(prepared)
    _add_prepared_context_budget_metadata(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 24 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 9 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 20 * 1024**3
    manifest["prepare_cold_read_gib_per_second"] = 16.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            mps_graph_probe_ran=True,
            mps_graph_probe_ok=True,
            mps_graph_probe_error=None,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--max-live-working-set-mib",
            str(9 * 1024),
            "--min-free-unified-memory-gib",
            "20",
            "--require-prepared-memory-profile",
            "--require-glm-4bit",
            "--require-public-glm-5-2-shape",
            "--require-prefill-acceleration",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--prefill-mpsgraph-min-batch-tokens",
            "2",
            "--prefill-mpsgraph-min-matrix-dim",
            "1",
            "--prefill-ssd-read-gib-s",
            "16",
        ],
    )

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "64",
            "--check-max-new-tokens",
            "1",
            "--check-runtime-preflight",
            "--run-mpsgraph-probe",
            "--require-launch-audit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    audit = health["launch_audit"]
    assert "prefill_acceleration_probe_ok" not in audit["failures"]
    assert "prefill_acceleration_profile_replays_probe" in audit["failures"]
    probe_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prefill_acceleration_probe_ok"
    )
    assert probe_check["ok"] is True
    assert probe_check["validated_accelerated_prefill_backends"] == ["mpsgraph-f32"]
    assert probe_check["validated_prefill_acceleration_available"] is True
    replay_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prefill_acceleration_profile_replays_probe"
    )
    assert replay_check["ok"] is False
    assert replay_check["has_run_mpsgraph_probe"] is False


def test_inspect_prepared_cli_require_launch_audit_accepts_locked_glm_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _make_prepared_glm_4bit_ready(prepared)
    _add_prepared_context_budget_metadata(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 24 * 1024**3
    manifest["recommended_max_live_working_set_bytes"] = 9 * 1024**3
    manifest["recommended_min_free_unified_memory_bytes"] = 20 * 1024**3
    manifest["prepare_cold_read_gib_per_second"] = 16.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            mps_graph_probe_ran=True,
            mps_graph_probe_ok=True,
            mps_graph_probe_error=None,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    profile_argv = [
        "--max-live-working-set-mib",
        str(9 * 1024),
        "--min-free-unified-memory-gib",
        "20",
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--require-prefill-acceleration",
        "--prefill-min-accelerated-flop-fraction",
        "0.2",
        "--prefill-mpsgraph-min-batch-tokens",
        "2",
        "--prefill-mpsgraph-min-matrix-dim",
        "1",
        "--prefill-ssd-read-gib-s",
        "16",
        "--prefill-max-routed-read-seconds",
        "5",
        "--decode-max-routed-read-gib-per-token",
        "1",
        "--decode-max-routed-read-seconds-per-token",
        "5",
        "--run-mpsgraph-probe",
    ]
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=profile_argv,
    )
    audit_path = tmp_path / "launch-audit.json"

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            "--lock-launch-profile",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "64",
            "--check-max-new-tokens",
            "1",
            "--check-runtime-preflight",
            "--run-mpsgraph-probe",
            "--require-launch-audit",
            "--write-launch-audit",
            str(audit_path),
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    audit = health["launch_audit"]
    assert audit["ok"] is True
    assert audit["failures"] == []
    assert all(check["ok"] is True for check in audit["checks"])
    check_codes = {check["code"] for check in audit["checks"]}
    assert "request_prefill_stage_temp_limit_ok" in check_codes
    assert "request_prefill_stage_temp_disk_ok" in check_codes
    assert "request_prefill_routed_read_budget_ok" in check_codes
    assert "request_decode_routed_read_budget_ok" in check_codes
    assert "prepared_storage_validated" in check_codes
    assert "prepared_ssd_read_profile_valid" in check_codes
    assert "prepare_expert_pack_heap_envelope_ok" in check_codes
    assert "prepare_resident_alias_rewrite_ok" in check_codes
    assert "prefill_acceleration_probe_ok" in check_codes
    assert "prefill_acceleration_profile_replays_probe" in check_codes
    acceleration_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prefill_acceleration_gate_ok"
    )
    assert acceleration_check["ok"] is True
    assert acceleration_check["reason_code"] == "ok"
    assert acceleration_check["recommended_backend"] == "mpsgraph_prefill_fallback"
    assert acceleration_check["prefill_acceleration_runtimes"] == ["mpsgraph-f32"]
    assert acceleration_check["selectable_accelerated_prefill_backends"] == [
        "mpsgraph-f32"
    ]
    assert acceleration_check["mps_graph_probe_requested"] is True
    assert acceleration_check["mps_graph_probe_ran"] is True
    assert acceleration_check["mps_graph_probe_ok"] is True
    assert acceleration_check["host_probe_requested"] is True
    assert acceleration_check["host_probe_path"]
    assert acceleration_check["host_probe_ran"] is True
    assert acceleration_check["host_probe_ok"] is True
    assert acceleration_check["prefill_backend_probe_timeout_seconds"] == 5.0
    assert acceleration_check["mpp_runtime_available"] is False
    assert acceleration_check["prefill_neural_accelerator_status"]["runtime"] == (
        "mpp_tensor_ops_prefill"
    )
    assert acceleration_check["prefill_neural_accelerator_status"]["status"] == (
        "unavailable"
    )
    expected_slot_bytes = sum(component[2] for component in COMPONENTS)
    glm_check = next(
        check for check in audit["checks"] if check["code"] == "glm_4bit_ready"
    )
    assert glm_check["ok"] is True
    assert glm_check["expert_layout_quantization"] == "largerlm-affine-int4"
    assert glm_check["expert_layout_group_size"] == 8
    assert glm_check["expert_layer_file_count"] == 2
    assert glm_check["unique_expert_layer_file_count"] == 2
    assert glm_check["expert_layer_files_exact_size"] is True
    assert glm_check["prepared_expert_layer_file_bytes"] == 4 * expected_slot_bytes
    assert glm_check["expected_total_expert_bytes"] == 4 * expected_slot_bytes
    assert (
        glm_check["expected_decode_token_routed_expert_read_bytes"]
        == 4 * expected_slot_bytes
    )
    assert (
        glm_check["expected_full_prompt_routed_expert_sweep_bytes"]
        == 4 * expected_slot_bytes
    )
    assert glm_check["decode_cache_layout_ok"] is True
    readiness = health["glm_4bit_readiness"]
    assert (
        glm_check["decode_cache_layout_total_bytes"]
        == readiness["decode_cache_layout_total_bytes"]
    )
    assert glm_check["decode_cache_mla_kv_segments_checked"] == 2
    assert glm_check["decode_cache_mla_kv_segments_ok"] == 2
    storage_check = next(
        check for check in audit["checks"] if check["code"] == "prepared_storage_validated"
    )
    assert storage_check["ok"] is True
    assert storage_check["expert_layout_backing_validated"] is True
    assert storage_check["resident_layout_backing_validated"] is True
    assert storage_check["decode_cache_file_exact_size"] is True
    assert storage_check["decode_cache_file_extra_bytes"] == 0
    ssd_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prepared_ssd_read_profile_valid"
    )
    assert ssd_check["ok"] is True
    assert ssd_check["source"] == "prepared_manifest"
    assert ssd_check["prefill_ssd_read_gib_per_second"] == 16.0
    assert ssd_check["prepare_cold_read_gib_per_second"] == 16.0
    assert ssd_check["matches_prepare_cold_read"] is True
    runtime_profile_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prepared_runtime_profile_ok"
    )
    assert runtime_profile_check["ok"] is True
    assert runtime_profile_check["prepare_effective_unified_memory_bytes"] == (
        128 * 1024**3
    )
    assert runtime_profile_check["prepare_system_reserve_bytes"] == 24 * 1024**3
    assert runtime_profile_check["prepared_recommended_max_live_working_set_bytes"] == (
        9 * 1024**3
    )
    assert runtime_profile_check[
        "prepared_recommended_min_free_unified_memory_bytes"
    ] == (20 * 1024**3)
    assert runtime_profile_check[
        "prepared_recommended_required_available_memory_bytes"
    ] == (29 * 1024**3)
    pack_heap_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prepare_expert_pack_heap_envelope_ok"
    )
    assert pack_heap_check["ok"] is True
    assert pack_heap_check["required"] is True
    assert pack_heap_check["evidence_present"] is True
    assert pack_heap_check["all_fields_present"] is True
    assert pack_heap_check["within_pack_heap_limit"] is True
    assert pack_heap_check["raw_quantization_rows_cover_bytes"] is True
    assert pack_heap_check["expert_quantization"] == "largerlm-affine-int4"
    assert pack_heap_check["prepare_expert_pack_estimated_peak_heap_bytes"] == (
        64 * 1024**2
    )
    assert pack_heap_check["prepare_expert_pack_max_heap_bytes"] == 512 * 1024**2
    assert pack_heap_check["prepare_raw_quantization_max_rows_per_block"] == 32
    resident_rewrite_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "prepare_resident_alias_rewrite_ok"
    )
    assert resident_rewrite_check["ok"] is True
    assert resident_rewrite_check["rewrite_present"] is False
    assert resident_rewrite_check["alias_counts_match"] is True
    assert resident_rewrite_check["fused_counts_match"] is True
    assert health["applied_launch_profile"]["locked"] is True
    assert health["prepared_memory_profile_requirement"]["ok"] is True
    assert health["request_check"]["ok"] is True
    assert health["request_check"]["runtime_preflight"]["ran"] is True
    routed_read_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "request_prefill_routed_read_budget_ok"
    )
    assert routed_read_check["ok"] is True
    assert routed_read_check["required"] is True
    assert routed_read_check["analyzed"] is True
    assert routed_read_check["ssd_read_gib_per_second"] == 16.0
    assert routed_read_check["max_read_seconds"] == 5.0
    assert routed_read_check["planned_read_seconds"] is not None
    assert routed_read_check["within_seconds_limit"] is True
    assert routed_read_check["within_limit"] is True
    decode_read_check = next(
        check
        for check in audit["checks"]
        if check["code"] == "request_decode_routed_read_budget_ok"
    )
    assert decode_read_check["ok"] is True
    assert decode_read_check["required"] is True
    assert decode_read_check["analyzed"] is True
    assert decode_read_check["ssd_read_gib_per_second"] == 16.0
    assert decode_read_check["max_read_bytes_per_token"] == 1024**3
    assert decode_read_check["max_read_seconds_per_token"] == 5.0
    assert decode_read_check["planned_read_seconds_per_token"] is not None
    assert decode_read_check["within_seconds_limit"] is True
    assert decode_read_check["within_limit"] is True
    assert (
        health["request_check"]["prefill_routed_stage_temp_disk"]["within_limit"]
        is True
    )
    assert (
        health["request_check"]["prefill_stage_temp_disk_free"]["within_free_space"]
        is True
    )
    assert (
        health["request_check"]["prefill_acceleration_coverage"]["ok"] is True
    )
    artifact = json.loads(audit_path.read_text(encoding="utf-8"))
    assert artifact["schema"] == "largerlm.launch_audit.v1"
    assert artifact["launch_audit"] == audit
    assert artifact["applied_launch_profile"]["sha256"] == hashlib.sha256(
        profile.read_bytes()
    ).hexdigest()
    assert artifact["prepared"]["max_context_tokens"] == 256


def test_inspect_prepared_cli_applies_prefill_runtime_policy_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile_argv = [
        "--prefill-linear-backend",
        "mpsgraph-f32",
        "--prefill-mpsgraph-min-batch-tokens",
        "64",
        "--prefill-mpsgraph-min-matrix-dim",
        "16",
        "--prefill-min-accelerated-flop-fraction",
        "0.5",
        "--run-mpsgraph-probe",
    ]
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=profile_argv,
        sections={
            "prefill_backend_probe_flags": {"source": "unit"},
            "prefill_runtime_policy_flags": {"source": "unit"},
        },
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    assert health["prefill_linear_backend"] == "mpsgraph-f32"
    assert health["prefill_mpsgraph_min_batch_tokens"] == 64
    assert health["prefill_mpsgraph_min_matrix_dim"] == 16
    assert health["prefill_min_accelerated_flop_fraction"] == 0.5
    assert health["prefill_acceleration_requirement"]["ok"] is True
    applied = health["applied_launch_profile"]
    assert applied["argv"] == profile_argv
    assert applied["section_names"] == [
        "prefill_backend_probe_flags",
        "prefill_runtime_policy_flags",
    ]
    assert applied["matches_prepared"] is True


def test_inspect_prepared_cli_writes_applied_calibration_backend_in_safe_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    calibration_profile = tmp_path / "calibration-profile.json"
    calibration_profile.write_text(
        json.dumps(
            {
                "source": "prefill_linear_calibration",
                "argv_safe_to_replay": True,
                "sections": {
                    "prefill_runtime_policy_flags": {
                        "source": "prefill_linear_calibration",
                        "prefill_linear_backend": "custom-metal",
                        "argv": [
                            "--prefill-linear-backend",
                            "custom-metal",
                        ],
                    }
                },
                "argv": [
                    "--prefill-linear-backend",
                    "custom-metal",
                ],
            }
        ),
        encoding="utf-8",
    )
    written_profile = tmp_path / "prepared-launch-profile.json"

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(calibration_profile),
            str(prepared),
            "--runner",
            "unused-runner",
            "--write-launch-profile",
            str(written_profile),
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    assert health["prefill_linear_backend"] == "custom-metal"
    policy = health["suggested_prefill_runtime_policy_flags"]
    assert policy == {
        "source": "prepared_health",
        "prefill_linear_backend": "custom-metal",
        "argv": [
            "--prefill-linear-backend",
            "custom-metal",
        ],
    }
    profile = json.loads(written_profile.read_text(encoding="utf-8"))
    assert profile == health["suggested_launch_profile"]
    assert profile["prepared"]["prepared_manifest"] == str(
        prepared / "manifest.json"
    )
    assert profile["sections"]["prefill_runtime_policy_flags"] == policy
    assert "--prefill-linear-backend" in profile["argv"]
    assert "custom-metal" in profile["argv"]


def test_inspect_prepared_cli_compile_mpp_probe_reaches_backend_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend_probe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpp_tensor_ops_prefill",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=True,
            mpp_tensor_ops_symbol_declared=True,
            mpp_runtime_available=True,
            mpp_compile_probe_requested=bool(kwargs.get("compile_mpp_probe")),
            mpp_compile_probe_ran=True,
            mpp_compile_probe_ok=True,
            mpp_compile_variant="metal_mpp",
            mpp_compile_error=None,
            reasons=(),
        )

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fake_backend_probe)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--compile-mpp-probe",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    capability = health["prefill_backend"]["capability"]
    assert captured["compile_mpp_probe"] is True
    assert health["prefill_compile_mpp_probe"] is True
    assert health["prefill_backend"]["auto_policy"]["compile_mpp_probe"] is True
    assert capability["mpp_compile_probe_requested"] is True
    assert capability["mpp_compile_probe_ok"] is True
    assert capability["mpp_compile_variant"] == "metal_mpp"
    neural_status = capability["prefill_neural_accelerator_status"]
    assert neural_status["status"] == "runtime_visible_not_selectable"
    assert neural_status["mpp_tensor_ops_symbol_declared"] is True
    assert neural_status["mpp_compile_probe_requested"] is True
    assert neural_status["ready_for_generation"] is False
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "compile_mpp_probe": True,
        "argv": ["--compile-mpp-probe"],
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert "--compile-mpp-probe" in profile["argv"]


def test_inspect_prepared_cli_run_mpp_probe_reaches_backend_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend_probe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpp_tensor_ops_prefill",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=True,
            mpp_tensor_ops_symbol_declared=True,
            mpp_runtime_available=True,
            mpp_compile_probe_requested=True,
            mpp_compile_probe_ran=True,
            mpp_compile_probe_ok=True,
            mpp_compile_variant="metal_mpp",
            mpp_compile_error=None,
            mpp_run_probe_requested=bool(kwargs.get("run_mpp_probe")),
            mpp_run_probe_ran=True,
            mpp_run_probe_ok=True,
            mpp_run_probe_error=None,
            mpp_run_probe_max_abs_error=0.0,
            reasons=(),
        )

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fake_backend_probe)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--run-mpp-probe",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    capability = health["prefill_backend"]["capability"]
    assert captured["run_mpp_probe"] is True
    assert health["prefill_run_mpp_probe"] is True
    assert health["prefill_backend"]["auto_policy"]["run_mpp_probe"] is True
    assert capability["mpp_run_probe_requested"] is True
    assert capability["mpp_run_probe_ok"] is True
    neural_status = capability["prefill_neural_accelerator_status"]
    assert neural_status["status"] == "runtime_executed_not_selectable"
    assert neural_status["ready_for_generation"] is False
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "run_mpp_probe": True,
        "argv": ["--run-mpp-probe"],
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert "--run-mpp-probe" in profile["argv"]


def test_inspect_prepared_cli_mpsgraph_probe_reaches_backend_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_backend_probe(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_runtime_available=True,
            mps_graph_matmul_declared=True,
            mps_graph_probe_ran=True,
            mps_graph_probe_ok=True,
            mps_graph_probe_error=None,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            mpp_compile_probe_ran=False,
            mpp_compile_probe_ok=None,
            mpp_compile_variant=None,
            mpp_compile_error=None,
            reasons=(),
        )

    monkeypatch.setattr("largerlm.server.inspect_prefill_backend", fake_backend_probe)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--run-mpsgraph-probe",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    capability = health["prefill_backend"]["capability"]
    assert captured["run_mpsgraph_probe"] is True
    assert health["prefill_run_mpsgraph_probe"] is True
    assert health["prefill_backend"]["auto_policy"]["run_mpsgraph_probe"] is True
    assert capability["mps_graph_probe_requested"] is True
    assert capability["mps_graph_probe_ok"] is True
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "run_mpsgraph_probe": True,
        "argv": ["--run-mpsgraph-probe"],
    }
    profile = health["suggested_launch_profile"]
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert "--run-mpsgraph-probe" in profile["argv"]


def test_inspect_prepared_cli_writes_prefill_runtime_policy_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )
    profile_path = tmp_path / "launch-profile.json"

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--prefill-router-hybrid-margin-threshold",
            "1e-05",
            "--prefill-mla-key-cache",
            "--run-mpsgraph-probe",
            "--write-launch-profile",
            str(profile_path),
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    policy = health["suggested_prefill_runtime_policy_flags"]
    assert policy == {
        "source": "prepared_health",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_mpsgraph_min_batch_tokens": 64,
        "prefill_mpsgraph_min_matrix_dim": 16,
        "prefill_min_accelerated_flop_fraction": 0.5,
        "prefill_router_hybrid_margin_threshold": 1e-5,
        "prefill_mla_key_cache": True,
        "argv": [
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "16",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--prefill-router-hybrid-margin-threshold",
            "1e-05",
            "--prefill-mla-key-cache",
        ],
    }
    assert health["prefill_mla_key_cache"] is True
    assert "request_check" not in health
    probe_flags = health["suggested_prefill_backend_probe_flags"]
    assert probe_flags == {
        "source": "prepared_health",
        "run_mpsgraph_probe": True,
        "argv": ["--run-mpsgraph-probe"],
    }
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile == health["suggested_launch_profile"]
    assert profile["source"] == "prepared_health"
    assert profile["sections"]["prefill_backend_probe_flags"] == probe_flags
    assert profile["sections"]["prefill_runtime_policy_flags"] == policy
    assert "--prefill-linear-backend" in profile["argv"]
    assert "--prefill-mpsgraph-min-batch-tokens" in profile["argv"]
    assert "--prefill-mpsgraph-min-matrix-dim" in profile["argv"]
    assert "--prefill-min-accelerated-flop-fraction" in profile["argv"]
    assert "--prefill-router-hybrid-margin-threshold" in profile["argv"]
    assert "--run-mpsgraph-probe" in profile["argv"]


def test_inspect_prepared_cli_prints_applied_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    profile = _write_matching_launch_profile(
        tmp_path / "launch-profile.json",
        prepared,
        argv=[
            "--max-live-working-set-mib",
            "8192",
            "--min-free-unified-memory-gib",
            "0",
        ],
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            "--apply-launch-profile",
            str(profile),
            str(prepared),
            "--runner",
            "unused-runner",
        ]
    )

    assert status == 0
    _assert_printed_applied_launch_profile(capsys.readouterr().out, profile)


def test_inspect_prepared_cli_fails_when_current_memory_is_below_prepare_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    manifest["prepare_effective_unified_memory_source"] = "explicit"
    manifest["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    profile = health["prepared_runtime_profile"]
    assert profile["profile_ok"] is False
    assert profile["system_total_meets_prepare_effective_unified_memory"] is False
    assert profile["system_available_meets_prepare_system_reserve"] is False
    assert profile["warnings"] == [
        "current system total memory is below the prepared effective unified-memory budget",
        "current available memory is below the prepared system reserve",
    ]


def test_inspect_prepared_cli_requires_prepared_memory_profile_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--require-prepared-memory-profile",
            "--runner",
            "unused-runner",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    requirement = health["prepared_memory_profile_requirement"]
    assert requirement["ok"] is False
    assert (
        "prepared manifest is missing required memory profile fields"
        in requirement["error"]
    )
    assert "recommended_min_free_unified_memory_bytes" in requirement["error"]


def test_inspect_prepared_cli_reports_glm_4bit_readiness_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--prefill-ssd-read-gib-s",
            "16",
            "--decode-mla-key-cache",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["issues"] == []
    assert readiness["model_type"] == "glm_moe_dsa"
    assert readiness["hidden_size"] == 8
    assert readiness["vocab_size"] is None
    assert readiness["moe_intermediate_size"] == 8
    assert readiness["num_hidden_layers"] == 2
    assert readiness["moe_layer_count"] == 2
    assert readiness["routed_experts"] == 2
    assert readiness["experts_per_token"] == 2
    assert readiness["tie_word_embeddings"] is None
    digest = config_sha256(prepared.parent / "model")
    assert readiness["model_config_sha256"] == digest
    assert readiness["expert_layout_config_sha256"] == digest
    assert readiness["resident_layout_config_sha256"] == digest
    assert readiness["expert_layout_model_type"] == "glm_moe_dsa"
    assert readiness["resident_layout_model_type"] == "glm_moe_dsa"
    assert readiness["expert_layout_quantization"] == "largerlm-affine-int4"
    assert readiness["expert_layout_group_size"] == 8
    assert readiness["expert_layout_model_layer_count"] == 2
    assert readiness["expert_layout_moe_layer_count"] == 2
    assert readiness["expert_layout_layer_count"] == 2
    assert readiness["expected_expert_slot_bytes"] == sum(
        component[2] for component in COMPONENTS
    )
    expected_slot_bytes = sum(component[2] for component in COMPONENTS)
    assert readiness["expected_expert_layer_bytes"] == 2 * expected_slot_bytes
    assert readiness["expected_total_expert_bytes"] == 4 * expected_slot_bytes
    assert (
        readiness["expected_decode_token_routed_expert_read_bytes"]
        == 4 * expected_slot_bytes
    )
    suggested_decode = health["suggested_decode_guard_flags"]
    assert suggested_decode["source"] == "prepared_health"
    assert suggested_decode["decode_read_bytes_per_token"] == 4 * expected_slot_bytes
    assert suggested_decode["decode_read_seconds_per_token"] == pytest.approx(
        (4 * expected_slot_bytes) / (16 * 1024**3)
    )
    assert "--decode-max-routed-read-gib-per-token" in suggested_decode["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in suggested_decode["argv"]
    assert suggested_decode["decode_mla_key_cache"] is True
    assert "--decode-mla-key-cache" in suggested_decode["argv"]
    suggested_glm = health["suggested_glm_4bit_guard_flags"]
    assert suggested_glm == {
        "source": "prepared_health",
        "require_glm_4bit": True,
        "argv": ["--require-glm-4bit"],
    }
    launch_profile = health["suggested_launch_profile"]
    assert launch_profile["source"] == "prepared_health"
    assert launch_profile["argv_safe_to_replay"] is True
    assert launch_profile["sections"]["glm_4bit_guard_flags"] == suggested_glm
    assert launch_profile["sections"]["decode_guard_flags"] == suggested_decode
    assert "--require-glm-4bit" in launch_profile["argv"]
    assert "--decode-max-routed-read-gib-per-token" in launch_profile["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in launch_profile["argv"]
    assert "--decode-mla-key-cache" in launch_profile["argv"]
    assert (
        readiness["expected_full_prompt_routed_expert_sweep_bytes"]
        == 4 * expected_slot_bytes
    )
    assert readiness["prepared_expert_layout_bytes"] == 4 * expected_slot_bytes
    assert readiness["prepared_expert_layer_file_bytes"] == 4 * expected_slot_bytes
    assert readiness["expert_layer_file_count"] == 2
    assert readiness["unique_expert_layer_file_count"] == 2
    assert readiness["expert_layer_files_exact_size"] is True
    assert readiness["prepared_resident_layout_bytes"] > 0
    assert readiness["resident_layout_total_bytes"] == readiness[
        "prepared_resident_layout_bytes"
    ]
    assert readiness["resident_weight_file_bytes"] == readiness[
        "prepared_resident_layout_bytes"
    ]
    assert readiness["resident_weight_file_exact_size"] is True
    assert readiness["prepared_decode_cache_file_bytes"] == 2 * 64
    assert readiness["decode_cache_layout_ok"] is True
    assert readiness["decode_cache_layout_total_bytes"] == 2 * 64
    assert readiness["decode_cache_context_tokens"] == 4
    assert readiness["decode_cache_model_type"] == "glm_moe_dsa"
    assert readiness["decode_cache_dtype"] == "BF16"
    assert readiness["decode_cache_segment_count"] == 2
    assert readiness["decode_cache_mla_kv_segments_checked"] == 2
    assert readiness["decode_cache_mla_kv_segments_ok"] == 2
    assert readiness["decode_cache_dsa_index_segments_checked"] == 0
    assert readiness["decode_cache_dsa_index_segments_ok"] == 0
    assert readiness["resident_layout_tensor_count"] == 22
    assert readiness["resident_embedding"] is True
    assert readiness["resident_final_norm"] is True
    assert readiness["resident_lm_head"] is False
    assert readiness["resident_tied_lm_head"] is True
    assert readiness["resident_attention_layers_checked"] == 2
    assert readiness["resident_attention_layers_ok"] == 2
    assert readiness["resident_router_layers_checked"] == 2
    assert readiness["resident_router_layers_ok"] == 2
    assert readiness["resident_router_metadata_present"] is True
    assert readiness["resident_router_metadata_ok"] is True
    assert readiness["resident_router_metadata_checked_fields"] == [
        "scoring_func",
        "num_experts_per_tok",
    ]
    assert readiness["resident_dense_layers_checked"] == 0
    assert readiness["resident_shared_layers_checked"] == 0
    assert readiness["resident_indexer_layers_checked"] == 0
    assert readiness["resident_routed_expert_tensor_count"] == 0
    assert readiness["matches_public_glm_5_2_shape"] is False
    public_shape = readiness["public_glm_5_2_shape"]
    assert public_shape["matches"] is False
    assert "hidden_size" in public_shape["mismatched_fields"]
    assert "num_hidden_layers" in public_shape["mismatched_fields"]
    assert public_shape["checks"]["hidden_size"] == {
        "actual": 8,
        "expected": 6144,
        "matches": False,
    }


def test_inspect_prepared_cli_require_glm_4bit_rejects_decode_cache_shape_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    layout_path = prepared / "decode_cache_layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    layout["segments"][1]["width"] = 3
    layout["segments"][1]["token_stride_bytes"] = 6
    layout["segments"][1]["total_bytes"] = 24
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["decode_cache_layout_ok"] is False
    assert any(
        "decode cache layout layer 1 mla_kv width does not match config" in issue
        for issue in readiness["issues"]
    )


def test_inspect_prepared_cli_require_glm_4bit_checks_dsa_decode_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _enable_prepared_full_dsa_indexer(prepared)
    _make_prepared_glm_4bit_ready(prepared)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["decode_cache_layout_ok"] is True
    assert readiness["decode_cache_segment_count"] == 3
    assert readiness["decode_cache_layout_total_bytes"] == 3 * 64
    assert readiness["prepared_decode_cache_file_bytes"] == 3 * 64
    assert readiness["decode_cache_mla_kv_segments_checked"] == 2
    assert readiness["decode_cache_mla_kv_segments_ok"] == 2
    assert readiness["decode_cache_dsa_index_segments_checked"] == 1
    assert readiness["decode_cache_dsa_index_segments_ok"] == 1
    assert readiness["resident_indexer_layers_checked"] == 1
    assert readiness["resident_indexer_layers_ok"] == 1


def test_inspect_prepared_cli_require_glm_4bit_rejects_extra_resident_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_path = prepared / "resident" / "resident.bin"
    resident_path.write_bytes(resident_path.read_bytes() + b"\0")

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is False
    assert readiness["resident_layout_total_bytes"] == readiness[
        "prepared_resident_layout_bytes"
    ]
    assert readiness["resident_weight_file_bytes"] == (
        readiness["resident_layout_total_bytes"] + 1
    )
    assert readiness["resident_weight_file_exact_size"] is False
    assert any(
        "resident weight file bytes" in issue
        and "do not match layout total_bytes" in issue
        for issue in readiness["issues"]
    )


def test_inspect_prepared_cli_require_glm_4bit_rejects_dsa_cache_shape_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _enable_prepared_full_dsa_indexer(prepared)
    _make_prepared_glm_4bit_ready(prepared)
    layout_path = prepared / "decode_cache_layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    dsa_segment = next(
        segment for segment in layout["segments"] if segment["kind"] == "dsa_index"
    )
    dsa_segment["width"] = 3
    dsa_segment["token_stride_bytes"] = 6
    dsa_segment["total_bytes"] = 24
    layout_path.write_text(json.dumps(layout), encoding="utf-8")

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["decode_cache_layout_ok"] is False
    assert readiness["decode_cache_dsa_index_segments_checked"] == 1
    assert readiness["decode_cache_dsa_index_segments_ok"] == 0
    assert any(
        "decode cache layout layer 0 dsa_index width does not match config" in issue
        for issue in readiness["issues"]
    )


def test_inspect_prepared_cli_require_public_glm_5_2_shape_rejects_non_public(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-public-glm-5-2-shape",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["matches_public_glm_5_2_shape"] is False
    assert "hidden_size" in readiness["public_glm_5_2_shape"]["mismatched_fields"]
    assert health["suggested_public_glm_5_2_shape_guard_flags"] is None


def test_inspect_prepared_cli_writes_public_glm_5_2_shape_launch_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_prepared_memory_profile(prepared)
    _make_prepared_glm_4bit_ready(prepared)
    _mock_safe_system_memory(monkeypatch)
    monkeypatch.setattr("largerlm.server._is_public_glm_5_2_shape", lambda config: True)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )
    profile_path = tmp_path / "launch-profile.json"

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-public-glm-5-2-shape",
            "--write-launch-profile",
            str(profile_path),
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    readiness = health["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["matches_public_glm_5_2_shape"] is True
    guard = health["suggested_public_glm_5_2_shape_guard_flags"]
    assert guard == {
        "source": "prepared_health",
        "require_public_glm_5_2_shape": True,
        "argv": ["--require-public-glm-5-2-shape"],
    }
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile == health["suggested_launch_profile"]
    assert profile["sections"]["public_glm_5_2_shape_guard_flags"] == guard
    assert "--require-public-glm-5-2-shape" in profile["argv"]


def test_inspect_prepared_cli_require_glm_4bit_allows_dense_prefix_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    write_config(prepared.parent / "model" / "config.json", mixed_layers=True)
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 0
    readiness = json.loads(capsys.readouterr().out)["glm_4bit_readiness"]
    assert readiness["ok"] is True
    assert readiness["issues"] == []
    assert readiness["num_hidden_layers"] == 2
    assert readiness["moe_layer_count"] == 1
    assert readiness["expert_layout_model_layer_count"] == 2
    assert readiness["expert_layout_moe_layer_count"] == 1
    assert readiness["expert_layout_layer_count"] == 1
    assert readiness["resident_router_layers_checked"] == 1
    assert readiness["resident_router_layers_ok"] == 1
    assert readiness["resident_dense_layers_checked"] == 1
    assert readiness["resident_dense_layers_ok"] == 1


def test_inspect_prepared_cli_require_glm_4bit_rejects_resident_routed_expert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    resident_layout_path = prepared / "resident" / "layout.json"
    resident_layout = json.loads(resident_layout_path.read_text(encoding="utf-8"))
    routed_tensor = {
        "name": "model.layers.1.mlp.experts.0.w1.weight",
        "offset": resident_layout["total_bytes"],
        "size": 16,
        "dtype": "BF16",
        "shape": [8],
        "category": "routed_experts",
    }
    resident_layout["tensors"].append(routed_tensor)
    resident_layout["total_bytes"] += 16
    resident_layout_path.write_text(json.dumps(resident_layout), encoding="utf-8")
    (prepared / "resident" / "resident.bin").write_bytes(
        b"\0" * resident_layout["total_bytes"]
    )
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    readiness = json.loads(capsys.readouterr().out)["glm_4bit_readiness"]
    assert readiness["ok"] is False
    assert readiness["resident_routed_expert_tensor_count"] == 1
    assert any(
        "resident layout contains routed expert tensors" in issue
        and "model.layers.1.mlp.experts.0.w1.weight" in issue
        for issue in readiness["issues"]
    )


def test_inspect_prepared_cli_require_glm_4bit_rejects_expert_byte_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _make_prepared_glm_4bit_ready(prepared)
    expert_layout_path = prepared / "experts" / "layout.json"
    expert_layout = json.loads(expert_layout_path.read_text(encoding="utf-8"))
    for layer in expert_layout["layers"]:
        layer["num_experts"] = 1
    expert_layout_path.write_text(json.dumps(expert_layout), encoding="utf-8")
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expert_bytes"] = (
        len(expert_layout["layers"]) * sum(component[2] for component in COMPONENTS)
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    readiness = json.loads(capsys.readouterr().out)["glm_4bit_readiness"]
    assert readiness["ok"] is False
    expected_total = (
        len(expert_layout["layers"])
        * int(expert_layout["num_experts"])
        * sum(component[2] for component in COMPONENTS)
    )
    assert readiness["expected_total_expert_bytes"] == expected_total
    assert readiness["prepared_expert_layout_bytes"] < expected_total
    assert any("prepared expert bytes" in issue for issue in readiness["issues"])


def test_inspect_prepared_cli_require_prefill_acceleration_reports_failure_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_tile",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("no accelerated backend",),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-prefill-acceleration",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    gate = health["prefill_acceleration_requirement"]
    assert gate["required"] is True
    assert gate["ok"] is False
    assert gate["configured_backend"] == "auto"
    assert gate["reason"] == "no accelerated prefill backend is available"
    assert gate["reason_code"] == "no_accelerated_backend"


def test_inspect_prepared_cli_require_glm_4bit_rejects_incomplete_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    readiness = json.loads(capsys.readouterr().out)["glm_4bit_readiness"]
    assert readiness["ok"] is False
    assert (
        "expert layout layers do not match config MoE layers"
        in readiness["issues"]
    )
    assert any("missing component up_proj.weight" in item for item in readiness["issues"])


def test_inspect_prepared_cli_require_glm_4bit_rejects_unsupported_config_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    config_path = prepared.parent / "model" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["attention_bias"] = True
    payload["hidden_act"] = "gelu"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    _make_prepared_glm_4bit_ready(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom-metal",
            mps_graph_matmul_declared=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--require-glm-4bit",
            "--json",
        ]
    )

    assert status == 1
    readiness = json.loads(capsys.readouterr().out)["glm_4bit_readiness"]
    assert readiness["ok"] is False
    assert (
        "config attention_bias=true is not supported by the GLM runner"
        in readiness["issues"]
    )
    assert (
        "config hidden_act 'gelu' is not supported by the GLM runner"
        in readiness["issues"]
    )


def test_inspect_prepared_cli_prints_runtime_memory_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fake_check_generation_runtime(**kwargs):
        return SimpleNamespace(
            requested_context_tokens=kwargs["requested_context_tokens"],
            layers=(0,),
            dense_layers=tuple(kwargs["dense_layers"] or ()),
            max_layer_peak_bytes=1234,
            max_layer_cache_read_bytes=456,
            read_bytes_per_token=789,
            final_logits_budget=SimpleNamespace(estimated_peak_bytes=2048),
            embedding_budget=SimpleNamespace(row_bytes=32, output_bytes=32),
            live_memory_budget=SimpleNamespace(
                estimated_live_working_set_bytes=64 * 1024**2,
                max_live_working_set_bytes=8 * 1024**3,
                min_available_memory_bytes=24 * 1024**3,
                system_available_bytes=96 * 1024**3,
                system_total_bytes=128 * 1024**3,
                system_source="test",
            ),
        )

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fake_check_generation_runtime,
    )
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=128 * 1024**3,
            available_bytes=96 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "1",
            "--check-max-new-tokens",
            "1",
            "--check-runtime-preflight",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "runtime required mem:" in output
    assert "runtime available mem:" in output
    assert "runtime memory ok:" in output
    assert "guard required mem:" in output
    assert "guard memory ok:" in output
    assert "prepared layout bytes:" in output
    assert "resident+cache bytes:" in output
    assert "recommended required:" in output
    assert "True" in output


def test_inspect_prepared_cli_reports_prefill_linear_backend_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="mpsgraph_prefill_fallback",
            mps_graph_matmul_declared=True,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=(),
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prefill-linear-backend",
            "auto",
            "--prefill-mpsgraph-min-batch-tokens",
            "64",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--check-prompt-tokens",
            "64",
            "--check-max-new-tokens",
            "1",
            "--check-metal-final-logits",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    linear = health["request_check"]["prefill_linear_backend"]
    assert linear["configured"] == "auto"
    assert linear["effective"] == "auto"
    assert linear["analyzed"] is True
    assert linear["prompt_chunk_tokens"] == 64
    routed_flops = 6 * 64 * 2 * 8 * 8
    assert linear["matrix_count"] == 4
    assert linear["mpsgraph_matrix_count"] == 1
    assert linear["custom_metal_matrix_count"] == 3
    assert linear["custom_metal_estimated_flops"] == routed_flops
    assert linear["streamed_routed_expert_layer_count"] == 1
    assert linear["streamed_routed_expert_matrix_count"] == 3
    assert linear["streamed_routed_expert_assignments"] == 64 * 2
    assert linear["streamed_routed_expert_estimated_flops"] == routed_flops
    assert linear["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert linear["mpp_tensor_ops_candidate_estimated_flops"] == 0
    assert linear["mpp_tensor_ops_candidate_flop_fraction"] == 0.0
    assert linear["mpp_tensor_ops_candidate_backend_counts"] == {}
    assert linear["mpp_tensor_ops_candidate_backend_flops"] == {}
    assert linear["total_matrix_scratch_bytes"] == 2 * 1024 * 1024 + 32 * 32 * 2
    assert linear["total_matrix_raw_conversion_bytes"] == 32 * 32 * 2
    assert linear["auto_policy"]["mpsgraph_min_batch_tokens"] == 64
    assert linear["auto_policy"]["mpsgraph_min_matrix_dim"] == 32
    assert linear["mpp_candidate_policy"] == {
        "candidate_backend": "mpp_tensor_ops_prefill",
        "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
        "mpp_tensor_ops_min_batch_tokens": 128,
        "mpp_tensor_ops_min_matrix_dim": 32,
        "selectable_prefill_backend": False,
    }
    assert linear["top_matrices"] == [
        {
            "rank": 1,
            "name": "model.layers.0.self_attn.q_a_proj.weight",
            "rows": 32,
            "cols": 32,
            "dtype": "BF16",
            "size_bytes": 32 * 32 * 2,
            "resolved_backend": "mpsgraph-f32",
            "estimated_flops": 2 * 64 * 32 * 32,
            "mpp_tensor_ops_candidate": False,
            "matrix_scratch_bytes": 2 * 1024 * 1024 + 32 * 32 * 2,
            "matrix_raw_conversion_bytes": 32 * 32 * 2,
        },
        {
            "rank": 2,
            "name": "streamed_routed_expert_mlp",
            "rows": 8,
            "cols": 8,
            "dtype": "streamed-expert",
            "size_bytes": 0,
            "resolved_backend": "custom-metal",
            "estimated_flops": routed_flops,
            "mpp_tensor_ops_candidate": False,
            "matrix_scratch_bytes": 0,
            "matrix_raw_conversion_bytes": 0,
            "streamed_routed_expert_layer_count": 1,
            "streamed_routed_expert_assignments": 64 * 2,
        },
    ]
    coverage = health["request_check"]["prefill_acceleration_coverage"]
    assert coverage["required"] is False
    assert coverage["ok"] is True
    assert coverage["matrix_count"] == 4
    assert coverage["accelerated_matrix_count"] == 1
    assert coverage["mpsgraph_matrix_count"] == 1
    assert coverage["custom_metal_matrix_count"] == 3
    assert coverage["custom_metal_estimated_flops"] == routed_flops
    assert coverage["total_estimated_flops"] == 2 * 64 * 32 * 32 + routed_flops
    assert coverage["accelerated_flop_fraction"] == pytest.approx(
        (2 * 64 * 32 * 32) / (2 * 64 * 32 * 32 + routed_flops)
    )
    assert coverage["streamed_routed_expert_estimated_flops"] == routed_flops
    assert coverage["mpp_tensor_ops_candidate_backend_counts"] == {}
    assert coverage["accelerated_backends"] == ["mpsgraph-f32"]
    assert coverage["any_resident_matrix_accelerated"] is True
    assert coverage["all_resident_matrices_accelerated"] is False
    accel_frontier = health["request_check"]["prefill_acceleration_frontier"]
    assert accel_frontier["analyzed"] is True
    assert accel_frontier["resolved_prompt_chunk_tokens"] == 64
    assert accel_frontier["max_safe_prompt_chunk_tokens"] == 64
    assert accel_frontier["minimum_accelerated_prompt_chunk_tokens"] == 64
    assert accel_frontier["suggested_guard_flags"]["argv"] == [
        "--prefill-prompt-chunk-tokens",
        "64",
    ]
    frontier_candidates = {
        item["prompt_chunk_tokens"]: item for item in accel_frontier["candidates"]
    }
    assert sorted(frontier_candidates) == [1, 64]
    assert frontier_candidates[1]["mpsgraph_matrix_count"] == 0
    assert frontier_candidates[1]["custom_metal_matrix_count"] == 4
    assert frontier_candidates[1]["streamed_routed_expert_estimated_flops"] == (
        6 * 1 * 2 * 8 * 8
    )
    assert frontier_candidates[64]["mpsgraph_matrix_count"] == 1
    assert frontier_candidates[64]["custom_metal_matrix_count"] == 3
    assert frontier_candidates[64]["streamed_routed_expert_estimated_flops"] == (
        routed_flops
    )
    assert frontier_candidates[64]["total_matrix_scratch_bytes"] == (
        2 * 1024 * 1024 + 32 * 32 * 2
    )
    routed = health["request_check"]["prefill_routed_expert_read"]
    assert routed["analyzed"] is True
    assert routed["prompt_chunk_tokens"] == 64
    assert routed["top_k"] == 2
    assert routed["layers"] == 1
    assert routed["chunks_per_prompt"] == 1
    assert routed["baseline_read_bytes"] == 16
    assert routed["planned_read_bytes"] == 16
    assert routed["extra_read_bytes"] == 0
    assert routed["read_amplification"] == 1.0
    stage_temp = health["request_check"]["prefill_routed_stage_temp_disk"]
    assert stage_temp["analyzed"] is True
    assert stage_temp["prompt_chunk_tokens"] == 64
    assert stage_temp["top_k"] == 2
    assert stage_temp["layers"] == 1
    assert stage_temp["chunks_per_prompt"] == 1
    assert stage_temp["stage_align_bytes"] == 4096
    assert stage_temp["static_capacity_per_expert"] == "auto"
    assert stage_temp["allow_static_capacity_overflow"] is False
    assert stage_temp["max_static_capacity_per_expert"] == 64
    assert stage_temp["static_capacity_strict_overflow_safe"] is True
    assert stage_temp["max_stage_plus_compact_bytes"] == 4128
    assert stage_temp["max_static_capacity_binary_bytes"] == 812
    assert stage_temp["max_stage_plus_compact_plus_static_bytes"] == 4940
    assert stage_temp["total_stage_plus_compact_bytes"] == 4128
    assert stage_temp["total_static_capacity_binary_bytes"] == 812
    assert stage_temp["total_stage_plus_compact_plus_static_bytes"] == 4940
    assert stage_temp["within_limit"] is True
    suggested = health["request_check"]["suggested_guard_flags"]
    assert suggested["source"] == "prepared_request_check"
    assert suggested["prefill_prompt_chunk_tokens"] == 64
    assert suggested["prefill_max_routed_read_amplification"] == pytest.approx(1.05)
    assert suggested["prefill_max_routed_read_gib"] == pytest.approx(
        16 / 1024**3 * 1.05
    )
    assert suggested["argv"][:4] == [
        "--prefill-prompt-chunk-tokens",
        "64",
        "--prefill-max-routed-read-amplification",
        "1.05",
    ]
    stage_suggested = health["request_check"]["suggested_stage_temp_guard_flags"]
    assert stage_suggested["source"] == "prepared_request_check"
    assert stage_suggested["prefill_prompt_chunk_tokens"] == 64
    assert stage_suggested["prefill_max_stage_mib"] == pytest.approx(
        4112 / 1024**2 * 1.05
    )
    assert stage_suggested["prefill_max_compact_stage_mib"] == pytest.approx(
        16 / 1024**2 * 1.05
    )
    assert stage_suggested["profile_max_stage_plus_compact_bytes"] == 4128
    assert stage_suggested["profile_total_stage_plus_compact_bytes"] == 4128
    assert stage_suggested["profile_max_static_capacity_binary_bytes"] == 812
    assert stage_suggested["profile_total_static_capacity_binary_bytes"] == 812
    assert stage_suggested["profile_max_stage_plus_compact_plus_static_bytes"] == 4940
    assert (
        stage_suggested["profile_total_stage_plus_compact_plus_static_bytes"] == 4940
    )
    assert stage_suggested["argv"][:4] == [
        "--prefill-prompt-chunk-tokens",
        "64",
        "--prefill-max-stage-mib",
        f"{4112 / 1024**2 * 1.05:.6g}",
    ]
    combined = health["request_check"]["suggested_prefill_guard_flags"]
    assert combined["source"] == "prepared_request_check"
    assert combined["prefill_prompt_chunk_tokens"] == 64
    assert combined["routed_read_guard"] == suggested
    assert combined["stage_temp_guard"] == stage_suggested
    assert combined["argv"].count("--prefill-prompt-chunk-tokens") == 1
    assert combined["argv"][:6] == [
        "--prefill-prompt-chunk-tokens",
        "64",
        "--prefill-max-routed-read-amplification",
        "1.05",
        "--prefill-max-routed-read-gib",
        f"{16 / 1024**3 * 1.05:.6g}",
    ]
    assert "--prefill-max-stage-mib" in combined["argv"]
    assert "--prefill-max-compact-stage-mib" in combined["argv"]
    chunk_plan = health["request_check"]["prefill_prompt_chunk_plan"]
    assert chunk_plan["source"] == "prepared_request_check"
    assert chunk_plan["configured_is_auto"] is True
    assert chunk_plan["auto"]["chunk_tokens"] == 64
    assert chunk_plan["auto"]["limiting_cap_names"]
    assert chunk_plan["max_safe"]["chunk_tokens"] == 64
    assert chunk_plan["max_safe"]["next_token_matrix_scratch_bytes"] is None
    request_profile = health["request_launch_profile"]
    assert request_profile["source"] == "prepared_request_check"
    assert request_profile["argv_safe_to_replay"] is True
    final_logits_flags = request_profile["sections"]["final_logits_flags"]
    assert final_logits_flags == {
        "source": "prepared_request_check",
        "metal_final_logits": True,
        "argv": ["--metal-final-logits"],
    }
    profile_prefill_flags = request_profile["sections"]["prefill_guard_flags"]
    assert profile_prefill_flags == {
        **combined,
        "prefill_prompt_chunk_tokens": "auto",
        "prefill_prompt_chunk_tokens_policy": "auto",
        "resolved_prefill_prompt_chunk_tokens": 64,
        "argv": [
            "--prefill-prompt-chunk-tokens",
            "auto",
            *combined["argv"][2:],
        ],
    }
    assert request_profile["sections"]["prefill_prompt_chunk_plan"] == chunk_plan
    assert "--metal-final-logits" in request_profile["argv"]
    assert request_profile["argv"].count("--prefill-prompt-chunk-tokens") == 1
    chunk_flag_index = request_profile["argv"].index("--prefill-prompt-chunk-tokens")
    assert request_profile["argv"][chunk_flag_index + 1] == "auto"
    assert "--prefill-max-routed-read-gib" in request_profile["argv"]
    assert "--prefill-max-stage-mib" in request_profile["argv"]
    assert "--prefill-max-compact-stage-mib" in request_profile["argv"]


def test_inspect_prepared_cli_require_prefill_acceleration_checks_request_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prefill-linear-backend",
            "auto",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--check-prompt-tokens",
            "64",
            "--check-max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--run-mpsgraph-probe",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    assert health["prefill_acceleration_requirement"]["ok"] is True
    request = health["request_check"]
    assert request["ok"] is False
    assert "prefill acceleration coverage failed" in request["error"]
    assert "no resident prefill matrices resolved" in request["error"]


def test_inspect_prepared_cli_prefill_acceleration_failure_suggests_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 512)
    _add_large_bf16_prefill_matrix(prepared)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(
            sdk_path=tmp_path / "MacOSX.sdk",
        ),
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--prefill-linear-backend",
            "auto",
            "--prefill-prompt-chunk-tokens",
            "64",
            "--prefill-mpsgraph-min-batch-tokens",
            "128",
            "--prefill-mpsgraph-min-matrix-dim",
            "32",
            "--check-prompt-tokens",
            "256",
            "--check-max-new-tokens",
            "1",
            "--require-prefill-acceleration",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
            "--run-mpsgraph-probe",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    request = health["request_check"]
    assert request["ok"] is False
    assert "prefill acceleration coverage failed" in request["error"]
    assert "prefill_prompt_chunk_tokens>=128" in request["error"]
    assert request["prefill_acceleration_frontier"]["suggested_guard_flags"] == {
        "prefill_prompt_chunk_tokens": 128,
        "require_prefill_acceleration": True,
        "prefill_min_accelerated_flop_fraction": 0.5,
        "argv": [
            "--prefill-prompt-chunk-tokens",
            "128",
            "--require-prefill-acceleration",
            "--prefill-min-accelerated-flop-fraction",
            "0.5",
        ],
    }


def test_inspect_prepared_cli_request_check_reports_rejection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "4",
            "--check-max-new-tokens",
            "1",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    assert health["request_check"]["ok"] is False
    assert "exceeds server cap" in health["request_check"]["error"]


def test_inspect_prepared_cli_runtime_preflight_reports_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fake_check_generation_runtime(**kwargs):
        raise GenerationGuardError("estimated live working set exceeds limit")

    monkeypatch.setattr(
        "largerlm.server.check_generation_runtime",
        fake_check_generation_runtime,
    )

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "2",
            "--check-max-new-tokens",
            "1",
            "--check-runtime-preflight",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    assert health["request_check"]["ok"] is False
    assert "estimated live working set exceeds limit" in health["request_check"]["error"]


def test_inspect_prepared_cli_can_tokenize_prompt_for_request_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--tokenizer-backend",
            "simple",
            "--check-prompt",
            "AB",
            "--check-max-new-tokens",
            "1",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    assert health["request_check"]["ok"] is True
    assert health["request_check"]["prompt_source"] == "text"
    assert health["request_check"]["prompt_token_count"] == 2
    assert health["request_check"]["required_context_tokens"] == 3
    assert health["request_check"]["tokenizer"]["backend"] == "simple"
    assert health["request_check"]["tokenizer"]["add_special_tokens"] is True


def test_inspect_prepared_cli_can_render_chat_for_request_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    tokenizer_path = _write_simple_model_tokenizer(prepared)
    captured: dict[str, object] = {}

    def fake_render_chat_prompt(
        tokenizer_path_arg,
        messages,
        *,
        backend,
        trust_remote_code,
        add_generation_prompt,
    ):
        captured.update(
            {
                "tokenizer_path": Path(tokenizer_path_arg),
                "messages": messages,
                "backend": backend,
                "trust_remote_code": trust_remote_code,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return RenderedChatPrompt(
            text="AB",
            backend="simple",
            tokenizer_path=Path(tokenizer_path_arg),
        )

    monkeypatch.setattr("largerlm.cli.render_chat_prompt", fake_render_chat_prompt)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--tokenizer-backend",
            "simple",
            "--check-chat-messages",
            '[{"role":"user","content":"hi"}]',
            "--check-max-new-tokens",
            "1",
            "--json",
        ]
    )

    assert status == 0
    health = json.loads(capsys.readouterr().out)
    assert captured["tokenizer_path"] == tokenizer_path.parent
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["add_generation_prompt"] is True
    assert health["request_check"]["ok"] is True
    assert health["request_check"]["prompt_source"] == "chat"
    assert health["request_check"]["prompt_token_count"] == 2
    assert health["request_check"]["chat_template_backend"] == "simple"
    assert health["request_check"]["tokenizer"]["add_special_tokens"] is False
    assert health["request_check"]["tokenizer"]["add_generation_prompt"] is True


def test_inspect_prepared_cli_rejects_multiple_prompt_check_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--check-prompt-tokens",
            "2",
            "--check-prompt",
            "AB",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    assert health["request_check"]["ok"] is False
    assert "mutually exclusive" in health["request_check"]["error"]


def test_inspect_prepared_cli_rejects_oversized_check_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _write_simple_model_tokenizer(prepared)

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--tokenizer-backend",
            "simple",
            "--max-request-bytes",
            "1",
            "--check-prompt",
            "AB",
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    assert health["request_check"]["ok"] is False
    assert "--check-prompt exceeds --max-request-bytes" in health["request_check"]["error"]


def test_inspect_prepared_cli_rejects_oversized_chat_messages_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    messages_path = tmp_path / "messages.json"
    messages_path.write_text('[{"role":"user","content":"hi"}]', encoding="utf-8")

    status = cli_main(
        [
            "inspect-prepared",
            str(prepared),
            "--runner",
            "unused-runner",
            "--max-request-bytes",
            "4",
            "--check-chat-messages-file",
            str(messages_path),
            "--json",
        ]
    )

    assert status == 1
    health = json.loads(capsys.readouterr().out)
    assert health["request_check"]["ok"] is False
    assert "chat messages file" in health["request_check"]["error"]
    assert "exceeds --max-request-bytes" in health["request_check"]["error"]


def test_benchmark_prepared_token_ids_auto_enables_batch_prefill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        prompt_prefill = None
        if kwargs.get("batch_prefill_prompt") is not False:
            prompt_prefill = SimpleNamespace(
                chunk_tokens=2,
                chunk_count=2,
                estimated_peak_bytes=8192,
                total_embedding_read_bytes=128,
                total_embedding_output_bytes=64,
                total_staged_bytes=2048,
                total_compact_stage_bytes=1024,
                total_compact_stage_materialized_bytes=0,
                max_staged_bytes=1024,
                max_compact_stage_bytes=512,
                max_compact_stage_materialized_bytes=0,
                total_stage_plus_compact_bytes=3072,
                total_stage_plus_compact_materialized_bytes=2048,
                max_stage_plus_compact_bytes=1536,
                max_stage_plus_compact_materialized_bytes=1024,
                linear_backend_counts={"mpsgraph-f32": 2},
                linear_backend_flops={"mpsgraph-f32": 4096},
                linear_backend_elapsed_seconds={"mpsgraph-f32": 0.002},
                linear_backend_estimated_tflops={
                    "mpsgraph-f32": 4096 / 0.002 / 1e12
                },
                total_linear_estimated_flops=4096,
                accelerated_linear_estimated_flops=4096,
                custom_linear_estimated_flops=0,
                unsupported_linear_estimated_flops=0,
                accelerated_linear_flop_fraction=1.0,
                total_routed_expert_assignments=16,
                total_routed_unique_expert_slots=12,
                max_routed_unique_experts_per_call=8,
                max_routed_tokens_per_expert=4,
                total_expert_stage_serial_read_bytes=4096,
                total_expert_stage_unique_requested_bytes=3072,
                total_expert_stage_planned_read_bytes=3584,
                total_expert_stage_planned_read_seconds=3584 / (16 * 1024**3),
                total_expert_stage_copy_elapsed_seconds=0.25,
                total_expert_stage_copy_throughput_gib_per_second=(
                    (2048 / 1024**3) / 0.25
                ),
                prefill_ssd_read_gib_per_second=16.0,
                prefill_max_routed_read_seconds=5.0,
                total_expert_stage_read_seconds_ok=True,
                total_expert_stage_copy_seconds_ok=True,
                prefill_max_stage_raw_ranges=8,
                prefill_max_stage_coalesced_ranges=4,
                total_expert_stage_raw_ranges=6,
                total_expert_stage_coalesced_ranges=3,
                max_expert_stage_raw_ranges=4,
                max_expert_stage_coalesced_ranges=2,
                total_expert_stage_raw_ranges_ok=True,
                total_expert_stage_coalesced_ranges_ok=True,
                total_expert_stage_waste_bytes=512,
                total_expert_stage_coalesced_savings_bytes=512,
                total_expert_stage_read_advice_attempted_ranges=4,
                total_expert_stage_read_advice_calls=4,
                total_expert_stage_read_advice_bytes=3584,
                total_expert_stage_read_advice_failures=0,
                total_expert_stage_assignment_read_amplification=0.875,
                total_expert_stage_unique_read_amplification=1.1666666667,
                max_expert_stage_unique_read_amplification=1.25,
                max_expert_stage_stage_budget_utilization=0.5,
                max_effective_moe_token_block=3,
                max_moe_max_expert_tokens=5,
                max_moe_batch_buffer_bytes=600,
                max_moe_estimated_peak_bytes=7000,
                static_capacity_per_expert="auto",
                max_static_capacity_per_expert=3,
                total_static_capacity_used_slots=9,
                total_static_capacity_slots=12,
                total_static_capacity_overflow_assignments=0,
                total_static_capacity_binary_bytes=1234,
                total_linear_matrix_scratch_bytes=4096,
                max_linear_matrix_scratch_bytes=2048,
                total_linear_matrix_f32_bytes=1024,
                total_linear_matrix_raw_conversion_bytes=512,
            )
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
            estimated_read_bytes=4096,
            estimated_embedding_read_bytes=128,
            estimated_expert_read_bytes=3584,
            estimated_cache_read_bytes=384,
            estimated_logits_read_bytes=0,
            runtime_guard=SimpleNamespace(read_bytes_per_token=384),
            prompt_prefill=prompt_prefill,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0, 1],
        max_new_tokens=1,
        preflight_runtime=False,
        prefill_max_routed_read_amplification=1.75,
        prefill_max_routed_read_gib=0.5,
        prefill_ssd_read_gib_per_second=16.0,
        prefill_max_routed_read_seconds=5.0,
        prefill_linear_backend="auto",
        prefill_mpsgraph_min_batch_tokens=64,
        prefill_mpsgraph_min_matrix_dim=16,
    )
    assert captured["batch_prefill_prompt"] is True
    assert captured["prefill_static_capacity_per_expert"] == "auto"
    assert captured["prefill_max_routed_read_amplification"] == 1.75
    assert captured["prefill_max_routed_read_gib"] == 0.5
    assert captured["prefill_ssd_read_gib_per_second"] == 16.0
    assert captured["prefill_max_routed_read_seconds"] == 5.0
    assert captured["prefill_linear_backend"] == "custom-metal"
    assert captured["prefill_mpsgraph_min_batch_tokens"] == 64
    assert captured["prefill_mpsgraph_min_matrix_dim"] == 16
    assert result.estimated_read_bytes == 4096
    assert result.estimated_embedding_read_bytes == 128
    assert result.estimated_expert_read_bytes == 3584
    assert result.estimated_cache_read_bytes == 384
    assert result.estimated_logits_read_bytes == 0
    assert result.prompt_prefill_chunk_tokens == 2
    assert result.prompt_prefill_chunk_count == 2
    assert result.prompt_prefill_estimated_peak_bytes == 8192
    assert result.prompt_prefill_total_embedding_read_bytes == 128
    assert result.prompt_prefill_total_embedding_output_bytes == 64
    assert result.prompt_prefill_total_staged_bytes == 2048
    assert result.prompt_prefill_total_compact_stage_bytes == 1024
    assert result.prompt_prefill_total_compact_stage_materialized_bytes == 0
    assert result.prompt_prefill_max_staged_bytes == 1024
    assert result.prompt_prefill_max_compact_stage_bytes == 512
    assert result.prompt_prefill_max_compact_stage_materialized_bytes == 0
    assert result.prompt_prefill_total_stage_plus_compact_bytes == 3072
    assert result.prompt_prefill_total_stage_plus_compact_materialized_bytes == 2048
    assert result.prompt_prefill_max_stage_plus_compact_bytes == 1536
    assert result.prompt_prefill_max_stage_plus_compact_materialized_bytes == 1024
    assert result.linear_backend_counts == {"mpsgraph-f32": 2}
    assert result.total_routed_expert_assignments == 16
    assert result.total_routed_unique_expert_slots == 12
    assert result.max_routed_unique_experts_per_call == 8
    assert result.max_routed_tokens_per_expert == 4
    assert result.total_expert_stage_serial_read_bytes == 4096
    assert result.total_expert_stage_unique_requested_bytes == 3072
    assert result.total_expert_stage_planned_read_bytes == 3584
    assert result.total_expert_stage_planned_read_seconds == pytest.approx(
        3584 / (16 * 1024**3)
    )
    assert result.total_expert_stage_copy_elapsed_seconds == 0.25
    assert result.total_expert_stage_copy_throughput_gib_per_second == pytest.approx(
        (2048 / 1024**3) / 0.25
    )
    assert result.prefill_ssd_read_gib_per_second == 16.0
    assert result.prefill_max_routed_read_seconds == 5.0
    assert result.total_expert_stage_read_seconds_ok is True
    assert result.prefill_max_stage_raw_ranges == 8
    assert result.prefill_max_stage_coalesced_ranges == 4
    assert result.total_expert_stage_raw_ranges == 6
    assert result.total_expert_stage_coalesced_ranges == 3
    assert result.max_expert_stage_raw_ranges == 4
    assert result.max_expert_stage_coalesced_ranges == 2
    assert result.total_expert_stage_raw_ranges_ok is True
    assert result.total_expert_stage_coalesced_ranges_ok is True
    assert result.total_expert_stage_waste_bytes == 512
    assert result.total_expert_stage_coalesced_savings_bytes == 512
    assert result.total_expert_stage_read_advice_attempted_ranges == 4
    assert result.total_expert_stage_read_advice_calls == 4
    assert result.total_expert_stage_read_advice_bytes == 3584
    assert result.total_expert_stage_read_advice_failures == 0
    assert result.total_expert_stage_assignment_read_amplification == 0.875
    assert result.total_expert_stage_unique_read_amplification == pytest.approx(
        1.1666666667
    )
    assert result.max_expert_stage_unique_read_amplification == 1.25
    assert result.max_expert_stage_stage_budget_utilization == 0.5
    assert result.max_effective_moe_token_block == 3
    assert result.max_moe_max_expert_tokens == 5
    assert result.max_moe_batch_buffer_bytes == 600
    assert result.max_moe_estimated_peak_bytes == 7000
    assert result.prefill_static_capacity_per_expert == "auto"
    assert result.max_static_capacity_per_expert == 3
    assert result.total_static_capacity_used_slots == 9
    assert result.total_static_capacity_slots == 12
    assert result.total_static_capacity_overflow_assignments == 0
    assert result.total_static_capacity_binary_bytes == 1234
    assert result.total_linear_matrix_scratch_bytes == 4096
    assert result.max_linear_matrix_scratch_bytes == 2048
    assert result.total_linear_matrix_f32_bytes == 1024
    assert result.total_linear_matrix_raw_conversion_bytes == 512
    assert result.linear_backend_flops == {"mpsgraph-f32": 4096}
    assert result.linear_backend_elapsed_seconds == {"mpsgraph-f32": 0.002}
    assert result.linear_backend_estimated_tflops == pytest.approx(
        {"mpsgraph-f32": 4096 / 0.002 / 1e12}
    )
    assert result.total_linear_estimated_flops == 4096
    assert result.accelerated_linear_estimated_flops == 4096
    assert result.custom_linear_estimated_flops == 0
    assert result.unsupported_linear_estimated_flops == 0
    assert result.accelerated_linear_flop_fraction == 1.0
    coverage = result.prefill_acceleration_coverage
    assert coverage is not None
    assert coverage["required"] is False
    assert coverage["min_accelerated_flop_fraction"] == 0.0
    assert coverage["ok"] is True
    assert coverage["matrix_count"] == 2
    assert coverage["accelerated_matrix_count"] == 2
    assert coverage["mpsgraph_matrix_count"] == 2
    assert coverage["custom_metal_matrix_count"] == 0
    assert coverage["total_estimated_flops"] == 4096
    assert coverage["accelerated_estimated_flops"] == 4096
    assert coverage["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert coverage["mpp_tensor_ops_candidate_estimated_flops"] == 0
    assert coverage["mpp_tensor_ops_candidate_flop_fraction"] == 0.0
    assert coverage["accelerated_flop_fraction"] == 1.0
    assert coverage["dominant_resident_flops_accelerated"] is True
    assert coverage["any_resident_matrix_accelerated"] is True
    assert coverage["all_resident_matrices_accelerated"] is True
    accel_frontier = result.prefill_acceleration_frontier
    assert accel_frontier is not None
    assert accel_frontier["source"] == "benchmark_actual_prefill"
    assert accel_frontier["resolved_prompt_chunk_tokens"] == 2
    assert accel_frontier["minimum_accelerated_prompt_chunk_tokens"] == 2
    accel_candidates = accel_frontier["candidates"]
    by_chunk = {item["prompt_chunk_tokens"]: item for item in accel_candidates}
    assert sorted(by_chunk) == [1, 2, 64]
    assert by_chunk[1]["viable_for_request"] is True
    assert by_chunk[1]["matrix_count"] == 0
    assert by_chunk[2]["is_resolved"] is True
    assert by_chunk[2]["viable_for_request"] is True
    assert by_chunk[2]["matrix_count"] == 2
    assert by_chunk[2]["accelerated_matrix_count"] == 2
    assert by_chunk[2]["mpsgraph_matrix_count"] == 2
    assert by_chunk[2]["custom_metal_matrix_count"] == 0
    assert by_chunk[2]["other_matrix_count"] == 0
    assert by_chunk[2]["total_estimated_flops"] == 4096
    assert by_chunk[2]["accelerated_estimated_flops"] == 4096
    assert by_chunk[2]["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert by_chunk[2]["accelerated_flop_fraction"] == 1.0
    assert by_chunk[2]["dominant_resident_flops_accelerated"] is True
    assert by_chunk[2]["any_resident_matrix_accelerated"] is True
    assert by_chunk[2]["all_resident_matrices_accelerated"] is True
    assert by_chunk[2]["total_matrix_scratch_bytes"] == 4096
    assert by_chunk[2]["total_matrix_raw_conversion_bytes"] == 512
    assert by_chunk[64]["is_auto_mpsgraph_threshold"] is True
    assert by_chunk[64]["viable_for_request"] is False
    assert by_chunk[64]["exceeds_prompt_tokens"] is True
    suggested = result.suggested_guard_flags
    assert suggested is not None
    assert suggested["source"] == "benchmark_actual_prefill"
    assert suggested["prefill_prompt_chunk_tokens"] == 2
    assert suggested["prefill_max_routed_read_amplification"] == pytest.approx(
        1.1666666667 * 1.05
    )
    assert suggested["prefill_max_routed_read_gib"] == pytest.approx(
        3584 / 1024**3 * 1.05
    )
    assert suggested["prefill_ssd_read_gib_per_second"] == 16.0
    assert suggested["planned_routed_read_seconds"] == pytest.approx(
        3584 / (16 * 1024**3)
    )
    assert suggested["prefill_max_routed_read_seconds"] == pytest.approx(
        3584 / (16 * 1024**3) * 1.05
    )
    assert "--prefill-max-routed-read-seconds" in suggested["argv"]
    stage_suggested = result.suggested_stage_temp_guard_flags
    assert stage_suggested is not None
    assert stage_suggested["source"] == "benchmark_actual_prefill"
    assert stage_suggested["prefill_prompt_chunk_tokens"] == 2
    assert stage_suggested["prefill_max_stage_mib"] == pytest.approx(
        1024 / 1024**2 * 1.05
    )
    assert stage_suggested["prefill_max_compact_stage_mib"] == pytest.approx(
        512 / 1024**2 * 1.05
    )
    assert stage_suggested["profile_max_stage_plus_compact_bytes"] == 1536
    assert stage_suggested["profile_total_stage_plus_compact_bytes"] == 3072
    assert stage_suggested["profile_total_static_capacity_binary_bytes"] == 1234
    assert (
        stage_suggested["profile_total_stage_plus_compact_plus_static_bytes"] == 4306
    )
    assert stage_suggested["profile_max_stage_raw_ranges"] == 4
    assert stage_suggested["prefill_max_stage_raw_ranges"] == 5
    assert stage_suggested["profile_max_stage_coalesced_ranges"] == 2
    assert stage_suggested["prefill_max_stage_coalesced_ranges"] == 3
    assert "--prefill-max-stage-mib" in stage_suggested["argv"]
    assert "--prefill-max-compact-stage-mib" in stage_suggested["argv"]
    assert "--prefill-max-stage-raw-ranges" in stage_suggested["argv"]
    assert "--prefill-max-stage-coalesced-ranges" in stage_suggested["argv"]
    combined = result.suggested_prefill_guard_flags
    assert combined is not None
    assert combined["source"] == "benchmark_actual_prefill"
    assert combined["prefill_prompt_chunk_tokens"] == 2
    assert combined["routed_read_guard"] == suggested
    assert combined["stage_temp_guard"] == stage_suggested
    assert combined["argv"].count("--prefill-prompt-chunk-tokens") == 1
    assert "--prefill-max-routed-read-seconds" in combined["argv"]
    assert "--prefill-max-stage-mib" in combined["argv"]
    assert "--prefill-max-compact-stage-mib" in combined["argv"]
    assert "--prefill-max-stage-raw-ranges" in combined["argv"]
    assert "--prefill-max-stage-coalesced-ranges" in combined["argv"]
    decode_suggested = result.suggested_decode_guard_flags
    assert decode_suggested is not None
    assert decode_suggested["source"] == "benchmark_actual_decode"
    assert decode_suggested["decode_read_bytes_per_token"] == 384
    assert decode_suggested["decode_max_routed_read_gib_per_token"] == pytest.approx(
        384 / 1024**3 * 1.05
    )
    assert decode_suggested["decode_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3)
    )
    assert decode_suggested["decode_max_routed_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3) * 1.05
    )
    assert "--decode-max-routed-read-gib-per-token" in decode_suggested["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in decode_suggested["argv"]
    profile = result.suggested_launch_profile
    assert profile is not None
    assert profile["source"] == "benchmark_actual"
    assert profile["argv_safe_to_replay"] is True
    assert profile["sections"]["prefill_guard_flags"] == combined
    assert profile["sections"]["decode_guard_flags"] == decode_suggested
    assert profile["sections"]["prefill_actual_acceleration_coverage"] == coverage
    assert (
        profile["sections"]["prefill_actual_acceleration_frontier"]
        == accel_frontier
    )
    actual_read_time = profile["sections"]["prefill_actual_read_time"]
    assert actual_read_time["source"] == "benchmark_actual_prefill"
    assert actual_read_time["total_expert_stage_planned_read_bytes"] == 3584
    assert actual_read_time[
        "total_expert_stage_planned_read_seconds"
    ] == pytest.approx(3584 / (16 * 1024**3))
    assert actual_read_time["prefill_ssd_read_gib_per_second"] == 16.0
    assert actual_read_time["prefill_max_routed_read_seconds"] == 5.0
    assert actual_read_time["total_expert_stage_read_seconds_ok"] is True
    assert actual_read_time["total_expert_stage_copy_seconds_ok"] is True
    assert actual_read_time["prefill_max_stage_raw_ranges"] == 8
    assert actual_read_time["prefill_max_stage_coalesced_ranges"] == 4
    assert actual_read_time["total_expert_stage_raw_ranges"] == 6
    assert actual_read_time["total_expert_stage_coalesced_ranges"] == 3
    assert actual_read_time["max_expert_stage_raw_ranges"] == 4
    assert actual_read_time["max_expert_stage_coalesced_ranges"] == 2
    assert actual_read_time["total_expert_stage_raw_ranges_ok"] is True
    assert actual_read_time["total_expert_stage_coalesced_ranges_ok"] is True
    assert actual_read_time["total_expert_stage_copy_elapsed_seconds"] == 0.25
    assert actual_read_time[
        "total_expert_stage_copy_throughput_gib_per_second"
    ] == pytest.approx((2048 / 1024**3) / 0.25)
    actual_linear = profile["sections"]["prefill_actual_linear_backend"]
    assert actual_linear["source"] == "benchmark_actual_prefill"
    assert actual_linear["configured_backend"] == "custom-metal"
    assert actual_linear["auto_policy"] == {
        "mpsgraph_min_batch_tokens": 64,
        "mpsgraph_min_matrix_dim": 16,
    }
    assert actual_linear["linear_backend_counts"] == {"mpsgraph-f32": 2}
    assert actual_linear["linear_backend_flops"] == {"mpsgraph-f32": 4096}
    assert actual_linear["linear_backend_elapsed_seconds"] == {
        "mpsgraph-f32": 0.002
    }
    assert actual_linear["linear_backend_estimated_tflops"] == pytest.approx(
        {"mpsgraph-f32": 4096 / 0.002 / 1e12}
    )
    assert actual_linear["total_linear_estimated_flops"] == 4096
    assert actual_linear["accelerated_linear_estimated_flops"] == 4096
    assert actual_linear["custom_linear_estimated_flops"] == 0
    assert actual_linear["unsupported_linear_estimated_flops"] == 0
    assert actual_linear["accelerated_linear_flop_fraction"] == 1.0
    assert "decode_actual_read_time" not in profile["sections"]
    assert profile["prepared"]["model_dir"] == str(
        load_prepared_manifest(prepared).model_dir
    )
    assert "--prefill-max-routed-read-seconds" in profile["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in profile["argv"]
    frontier = result.routed_chunk_frontier
    assert frontier is not None
    assert frontier["source"] == "benchmark_actual_prefill"
    assert frontier["resolved_prompt_chunk_tokens"] == 2
    assert frontier["prompt_token_count"] == 2
    assert frontier["top_k"] == 8
    assert frontier["saturation_chunk_tokens"] == 1
    assert frontier["baseline_read_bytes"] == 16
    assert any(
        candidate["prompt_chunk_tokens"] == 2
        and candidate["planned_read_bytes"] == 16
        for candidate in frontier["candidates"]
    )

    captured.clear()
    no_prefill_result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0, 1],
        max_new_tokens=1,
        preflight_runtime=False,
        batch_prefill_prompt=False,
    )
    assert captured["batch_prefill_prompt"] is False
    assert "prefill_static_capacity_per_expert" not in captured
    assert no_prefill_result.suggested_guard_flags is None
    assert no_prefill_result.suggested_stage_temp_guard_flags is None
    assert no_prefill_result.suggested_prefill_guard_flags is None
    assert no_prefill_result.routed_chunk_frontier is None
    assert no_prefill_result.prefill_acceleration_coverage is None
    assert no_prefill_result.prefill_acceleration_frontier is None


def test_benchmark_prepared_token_ids_resolves_auto_prefill_backend_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: SimpleNamespace(
            sdk_path=tmp_path / "MacOSX.sdk",
            recommended_backend="custom_metal_prefill_fallback",
            mps_graph_matmul_declared=True,
            mps_graph_runtime_available=False,
            metal4_ml_runtime_available=False,
            mpp_runtime_available=False,
            reasons=("host probe could not create a default Metal device",),
        ),
    )

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0, 1],
        max_new_tokens=1,
        preflight_runtime=False,
        prefill_linear_backend="auto",
    )

    assert captured["prefill_linear_backend"] == "custom-metal"
    assert result.generated_tokens == 1


def test_benchmark_prepared_token_ids_runs_request_admission_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match=(
            "prepared request admission failed: "
            "prefill_prompt_chunk_tokens 999 exceeds safety-capped maximum"
        ),
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0, 1],
            max_new_tokens=1,
            prefill_prompt_chunk_tokens=999,
        )


def test_benchmark_prepared_token_ids_rejects_nonpassing_admission_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    def fake_inspect_token_request(self, **kwargs):
        del self, kwargs
        return {
            "ok": False,
            "reason": "synthetic live-memory guard failure",
        }

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        fake_inspect_token_request,
    )
    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match=(
            "prepared request admission failed: "
            "synthetic live-memory guard failure"
        ),
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0, 1],
            max_new_tokens=1,
        )


@pytest.mark.parametrize(
    ("generation_kwargs", "expected_runtime_preflight"),
    [
        ({}, True),
        ({"preflight_runtime": False}, False),
    ],
)
def test_benchmark_prepared_token_ids_passes_runtime_preflight_to_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_kwargs: dict[str, object],
    expected_runtime_preflight: bool,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    captured: dict[str, object] = {}

    def fake_inspect_token_request(self, **kwargs):
        del self
        captured["runtime_preflight"] = kwargs.get("runtime_preflight")
        return {"ok": True}

    def fake_generate_token_ids(**kwargs):
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr(
        "largerlm.server.PreparedGenerationApp.inspect_token_request",
        fake_inspect_token_request,
    )
    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0],
        max_new_tokens=1,
        **generation_kwargs,
    )

    assert result.generated_tokens == 1
    assert captured["runtime_preflight"] is expected_runtime_preflight


def test_benchmark_prepared_token_ids_admission_uses_generation_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_runtime_guard_prepared_manifest(tmp_path)
    cfg_path = load_prepared_manifest(prepared).model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["num_experts_per_tok"] = 1
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    def fail_generate_token_ids(**kwargs):
        del kwargs
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match=(
            "prepared request admission failed: .*read amplification 2.*"
            "exceeds cap 1.5"
        ),
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0, 1],
            max_new_tokens=1,
            preflight_runtime=False,
            top_k=2,
            max_k=2,
            prefill_prompt_chunk_tokens=1,
            prefill_max_routed_read_amplification=1.5,
        )


def test_benchmark_prepared_token_ids_acceleration_frontier_suggests_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _set_prepared_context(prepared, 256)
    _add_large_bf16_prefill_matrix(prepared)

    def fake_generate_token_ids(**kwargs):
        prompt_prefill = SimpleNamespace(
            chunk_tokens=32,
            chunk_count=4,
            estimated_peak_bytes=8192,
            total_embedding_read_bytes=128,
            total_embedding_output_bytes=64,
            total_staged_bytes=0,
            total_compact_stage_bytes=0,
            total_compact_stage_materialized_bytes=0,
            max_staged_bytes=0,
            max_compact_stage_bytes=0,
            max_compact_stage_materialized_bytes=0,
            total_stage_plus_compact_bytes=0,
            total_stage_plus_compact_materialized_bytes=0,
            max_stage_plus_compact_bytes=0,
            max_stage_plus_compact_materialized_bytes=0,
            total_expert_stage_read_advice_attempted_ranges=0,
            total_expert_stage_read_advice_calls=0,
            total_expert_stage_read_advice_bytes=0,
            total_expert_stage_read_advice_failures=0,
            linear_backend_counts={"custom-metal": 1},
            linear_backend_flops={"custom-metal": 65536},
            total_linear_estimated_flops=65536,
            accelerated_linear_estimated_flops=0,
            custom_linear_estimated_flops=65536,
            unsupported_linear_estimated_flops=0,
            accelerated_linear_flop_fraction=0.0,
            total_linear_matrix_scratch_bytes=2 * 1024 * 1024,
            max_linear_matrix_scratch_bytes=2 * 1024 * 1024,
            total_linear_matrix_f32_bytes=0,
            total_linear_matrix_raw_conversion_bytes=0,
        )
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=256,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
            estimated_read_bytes=1,
            prompt_prefill=prompt_prefill,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    result = benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0] * 128,
        max_new_tokens=1,
        preflight_runtime=False,
        prefill_prompt_chunk_tokens=32,
        prefill_mpsgraph_min_batch_tokens=64,
        prefill_mpsgraph_min_matrix_dim=32,
    )

    coverage = result.prefill_acceleration_coverage
    assert coverage is not None
    assert coverage["any_resident_matrix_accelerated"] is False
    assert coverage["custom_metal_estimated_flops"] == 65536
    assert coverage["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert coverage["accelerated_flop_fraction"] == 0.0
    frontier = result.prefill_acceleration_frontier
    assert frontier is not None
    assert frontier["minimum_accelerated_prompt_chunk_tokens"] == 64
    assert frontier["suggested_guard_flags"] == {
        "prefill_prompt_chunk_tokens": 64,
        "argv": (
            "--prefill-prompt-chunk-tokens",
            "64",
        ),
    }
    by_chunk = {item["prompt_chunk_tokens"]: item for item in frontier["candidates"]}
    assert sorted(by_chunk) == [1, 32, 64, 128]
    assert by_chunk[32]["is_resolved"] is True
    assert by_chunk[32]["mpp_tensor_ops_candidate_matrix_count"] == 0
    assert by_chunk[128]["mpp_tensor_ops_candidate_matrix_count"] == 1
    assert by_chunk[128]["mpp_tensor_ops_candidate_estimated_flops"] == (
        2 * 128 * 32 * 32
    )
    assert by_chunk[128]["mpp_tensor_ops_candidate_flop_fraction"] == 1.0
    assert by_chunk[128]["mpp_tensor_ops_candidate_backend_counts"] == {
        "mpsgraph-f32": 1
    }
    assert by_chunk[128]["mpp_tensor_ops_candidate_backend_flops"] == {
        "mpsgraph-f32": 2 * 128 * 32 * 32
    }
    assert by_chunk[32]["custom_metal_matrix_count"] == 1
    assert by_chunk[32]["custom_metal_estimated_flops"] == 65536
    assert by_chunk[64]["mpsgraph_matrix_count"] == 1
    assert by_chunk[64]["accelerated_estimated_flops"] == 131072
    assert by_chunk[64]["accelerated_flop_fraction"] == 1.0
    assert by_chunk[64]["any_resident_matrix_accelerated"] is True


def test_benchmark_prepared_token_ids_require_prefill_acceleration_rejects_actual_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)

    def fake_generate_token_ids(**kwargs):
        del kwargs
        return TokenGenerationResult(
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            prompt_prefill=SimpleNamespace(
                prefill_acceleration_coverage={
                    "ok": False,
                    "reason": "no resident prefill matrices used an accelerated backend",
                },
                prefill_acceleration_frontier=None,
            ),
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    with pytest.raises(
        BenchmarkError,
        match="prefill acceleration actual coverage failed: no resident prefill matrices",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0, 1],
            max_new_tokens=1,
            preflight_runtime=False,
            require_prefill_acceleration=True,
            prefill_linear_backend="mpsgraph-f32",
            run_mpsgraph_probe=True,
        )


def test_benchmark_prepared_token_ids_rejects_low_accelerated_flop_fraction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    _add_large_bf16_prefill_matrix(prepared)

    def fake_generate_token_ids(**kwargs):
        del kwargs
        return TokenGenerationResult(
            prompt_token_ids=(0, 1),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            prompt_prefill=SimpleNamespace(
                prefill_acceleration_coverage={
                    "ok": True,
                    "any_resident_matrix_accelerated": True,
                    "accelerated_flop_fraction": 0.25,
                },
                prefill_acceleration_frontier=None,
            ),
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)
    monkeypatch.setattr(
        "largerlm.server.inspect_prefill_backend",
        lambda **kwargs: _mpsgraph_probe_ready_capability(),
    )

    with pytest.raises(
        BenchmarkError,
        match="prefill acceleration actual coverage failed: .*0.25.*0.5",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0, 1],
            max_new_tokens=1,
            preflight_runtime=False,
            prefill_min_accelerated_flop_fraction=0.5,
            prefill_linear_backend="mpsgraph-f32",
            run_mpsgraph_probe=True,
        )


def test_benchmark_prepared_token_ids_preserves_token_validation(
    tmp_path: Path,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)

    with pytest.raises(TokenGeneratorError, match="prompt_token_ids"):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0, 1.5],
            max_new_tokens=1,
            preflight_runtime=False,
            num_heads=1,
            qk_nope_dim=1,
            rope_dim=1,
            v_head_dim=1,
        )


def test_benchmark_prepared_token_ids_uses_manifest_memory_guard_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recommended_max_live_working_set_bytes"] = 5 * 1024**3
    payload["recommended_min_free_unified_memory_bytes"] = 20 * 1024**3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_generate_token_ids(**kwargs):
        captured.update(kwargs)
        return TokenGenerationResult(
            prompt_token_ids=tuple(kwargs["prompt_token_ids"]),
            generated_token_ids=(2,),
            steps=(),
            work_dir=tmp_path,
            kept_work_dir=False,
            max_context_tokens=4,
            sampling_temperature=0.0,
            sampling_top_p=1.0,
            elapsed_seconds=1.0,
        )

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fake_generate_token_ids)

    benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0],
        max_new_tokens=1,
        preflight_runtime=False,
    )
    assert captured["max_live_working_set_mib"] == 5 * 1024
    assert captured["min_free_unified_memory_gib"] == 20

    captured.clear()
    benchmark_prepared_token_ids(
        prepared,
        runner_path="unused-runner",
        prompt_token_ids=[0],
        max_new_tokens=1,
        max_live_working_set_mib=0.0,
        min_free_unified_memory_gib=0.0,
        preflight_runtime=False,
    )
    assert captured["max_live_working_set_mib"] == 0.0
    assert captured["min_free_unified_memory_gib"] == 0.0


def test_benchmark_prepared_token_ids_rejects_below_prepare_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["prepare_effective_unified_memory_bytes"] = 128 * 1024**3
    payload["prepare_effective_unified_memory_source"] = "explicit"
    payload["prepare_system_reserve_bytes"] = 48 * 1024**3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "largerlm.server.system_memory_snapshot",
        lambda: SimpleNamespace(
            total_bytes=64 * 1024**3,
            available_bytes=32 * 1024**3,
            page_size=16 * 1024,
            source="test",
        ),
    )

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match="prepared runtime profile check failed: current system total memory",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
        )


def test_benchmark_prepared_token_ids_rejects_unverified_prepare_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _write_minimal_prepared_manifest(tmp_path)
    manifest_path = prepared / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recommended_max_live_working_set_bytes"] = 40 * 1024**3
    payload["recommended_min_free_unified_memory_bytes"] = 16 * 1024**3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("largerlm.server.system_memory_snapshot", lambda: None)

    def fail_generate_token_ids(**kwargs):
        raise AssertionError("generate_token_ids should not be called")

    monkeypatch.setattr("largerlm.benchmark.generate_token_ids", fail_generate_token_ids)

    with pytest.raises(
        BenchmarkError,
        match="prepared runtime profile check failed: could not verify",
    ):
        benchmark_prepared_token_ids(
            prepared,
            runner_path="unused-runner",
            prompt_token_ids=[0],
            max_new_tokens=1,
        )


def test_bench_prepared_token_ids_reports_telemetry(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_checkpoint(model)
    output = tmp_path / "prepared"
    runner = tmp_path / "runner.py"
    runner.write_text(
        """#!/usr/bin/env python3
import shutil
import sys

def arg(name):
    i = sys.argv.index(name)
    return sys.argv[i + 1]

shutil.copyfile(arg("--input-f32"), arg("--output-f32"))
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    prepare_glm_checkpoint(
        model,
        output_dir=output,
        max_context_tokens=4,
        execute=True,
        quantize_raw_to_int4=True,
        group_size=8,
        max_cache_bytes=1024 * 1024,
        disk_safety_margin_bytes=0,
        chunk_size=17,
        unified_memory_bytes=128 * 1024**3,
    )

    result = benchmark_prepared_token_ids(
        output,
        runner_path=runner,
        prompt_token_ids=[0],
        max_new_tokens=1,
        layers={1},
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        top_k=1,
        max_k=1,
        router_score="raw",
        rms_norm_eps=1e-5,
        logits_top_k=1,
        max_slot_mib=1,
        max_router_mib=1,
        max_resident_matrix_mib=1,
        max_cache_file_mib=1,
        max_cache_read_mib=1,
        max_runner_scratch_mib=64,
        min_free_unified_memory_gib=0,
        echo_runner_output=False,
    )
    cli_status = cli_main(
        [
            "bench-prepared-token-ids",
            str(output),
            "--runner",
            str(runner),
            "--layers",
            "1",
            "--prompt-token-ids",
            "0",
            "--max-new-tokens",
            "1",
            "--top-k",
            "1",
            "--max-k",
            "1",
            "--router-score",
            "raw",
            "--logits-top-k",
            "1",
            "--max-slot-mib",
            "1",
            "--max-router-mib",
            "1",
            "--max-resident-matrix-mib",
            "1",
            "--max-cache-file-mib",
            "1",
            "--max-cache-read-mib",
            "1",
            "--max-runner-scratch-mib",
            "64",
            "--min-free-unified-memory-gib",
            "0",
            "--quiet-runner",
            "--json",
        ]
    )

    assert result.generated_tokens == 1
    assert result.tokens_per_second >= 0.0
    assert result.estimated_read_bytes > 0
    if result.prompt_prefill_chunk_count:
        assert result.suggested_guard_flags is not None
    assert cli_status == 0
