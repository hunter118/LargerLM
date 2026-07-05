from __future__ import annotations

import json
import math
import operator
from dataclasses import dataclass, replace
from pathlib import Path

from .config import load_config
from .decode_cache import (
    DecodeCacheError,
    DecodeCacheInitResult,
    DecodeCacheLayout,
    build_decode_cache_layout,
    init_decode_cache_file,
)
from .disk_benchmark import (
    DiskBenchmarkError,
    MAX_SEQUENTIAL_READ_CHUNK_BYTES,
    SequentialReadBenchmark,
    benchmark_sequential_read,
)
from .generation_guard import (
    GenerationGuardError,
    LiveMemoryBudget,
    check_live_memory_budget,
)
from .packer import PackReport, pack_experts
from .prepared import PreparedManifestError, load_prepared_manifest
from .preflight import GlmPreflightReport, preflight_glm_checkpoint
from .resident import ResidentPackReport, pack_resident_weights
from .safety import DiskBudget, disk_budget


class PrepareError(RuntimeError):
    """Raised when prepare-glm cannot safely continue."""


def _integer_param(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise PrepareError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise PrepareError(f"{name} must be an integer") from exc


def _positive_integer_param(name: str, value: object) -> int:
    parsed = _integer_param(name, value)
    if parsed <= 0:
        raise PrepareError(f"{name} must be positive")
    return parsed


def _nonnegative_integer_param(name: str, value: object) -> int:
    parsed = _integer_param(name, value)
    if parsed < 0:
        raise PrepareError(f"{name} must be non-negative")
    return parsed


def _optional_positive_integer_param(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    return _positive_integer_param(name, value)


def _optional_nonnegative_integer_param(name: str, value: object | None) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer_param(name, value)


def _numeric_param(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise PrepareError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PrepareError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise PrepareError(f"{name} must be finite")
    return parsed


def _fraction_param(name: str, value: object) -> float:
    parsed = _numeric_param(name, value)
    if parsed < 0.0 or parsed > 1.0:
        raise PrepareError(f"{name} must be between 0 and 1")
    return parsed


def _optional_positive_numeric_param(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    parsed = _numeric_param(name, value)
    if parsed <= 0.0:
        raise PrepareError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class PreparePaths:
    output_dir: Path
    experts_dir: Path
    resident_dir: Path
    cache_layout_path: Path
    cache_file_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PrepareContextBudget:
    auto_context_from_budget: bool
    requested_max_context_tokens: int | None
    resolved_max_context_tokens: int | None
    decode_cache_budget_bytes: int | None
    decode_cache_safe_context_tokens: int | None
    effective_max_cache_bytes: int | None
    model_max_position_embeddings: int | None
    cache_dtype: str
    cache_alignment: int


@dataclass(frozen=True)
class PrepareFlagsProvenance:
    source: str
    path: str | None
    sha256: str | None


@dataclass(frozen=True)
class PrepareGlmReport:
    ok: bool
    executed: bool
    paths: PreparePaths
    preflight: GlmPreflightReport
    expert_pack: PackReport | None
    resident_pack: ResidentPackReport | None
    cache_layout: DecodeCacheLayout | None
    cache_init: DecodeCacheInitResult | None
    context_budget: PrepareContextBudget | None = None
    prepare_flags: PrepareFlagsProvenance | None = None
    prepare_disk_budget: DiskBudget | None = None
    prepare_live_memory: LiveMemoryBudget | None = None
    cold_read_benchmark: SequentialReadBenchmark | None = None
    glm_4bit_readiness: dict[str, object] | None = None
    public_glm_5_2_shape: dict[str, object] | None = None
    require_public_glm_5_2_shape: bool = False


def _existing_outputs(
    paths: PreparePaths,
    *,
    expert_report: PackReport,
) -> list[Path]:
    candidates = [
        paths.cache_layout_path,
        paths.cache_file_path,
        paths.manifest_path,
        paths.resident_dir / "layout.json",
        paths.resident_dir / "resident.bin",
        paths.experts_dir / "layout.json",
    ]
    for layer in expert_report.layout.layers:
        candidates.append(paths.experts_dir / layer.layer_file)
    return [path for path in candidates if path.exists()]


def _cleanup_prepare_outputs(
    paths: PreparePaths,
    *,
    expert_report: PackReport,
    preserve: set[Path],
) -> None:
    for path in reversed(_existing_outputs(paths, expert_report=expert_report)):
        if path in preserve:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def _write_cache_layout(path: Path, layout: DecodeCacheLayout, *, force: bool) -> None:
    if path.exists() and not force:
        raise PrepareError(f"{path} already exists; use --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, layout.to_json())


def _resolve_path_for_guard(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _validate_prepare_output_dir(model_dir: Path, output_dir: Path) -> None:
    model_resolved = _resolve_path_for_guard(model_dir)
    output_resolved = _resolve_path_for_guard(output_dir)
    if output_resolved == model_resolved:
        raise PrepareError(
            "output_dir must not be the checkpoint model directory; use a "
            "dedicated prepared output subdirectory"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise PrepareError("output_dir must be a directory path")


def _require_execute_checkpoint_artifact_ready(model_dir: Path) -> None:
    from .artifact_status import inspect_checkpoint_artifact

    status = inspect_checkpoint_artifact(model_dir, verify_local_headers=True)
    ready = (
        status.download_complete
        and status.artifact_clean
        and status.local_headers_ok is not False
        and (
            not status.header_manifest_valid
            or (
                status.download_complete_proven
                and status.local_headers_ok is True
            )
        )
    )
    if ready:
        return

    details = [
        f"expected shards={status.expected_shard_count}",
        f"present={status.present_shard_count}",
        f"complete={status.complete_shard_count}",
        f"missing={status.missing_shard_count}",
        f"partial={status.partial_shard_count}",
        f"extra={status.extra_shard_count}",
    ]
    if status.remaining_safetensors_file_bytes is not None:
        details.append(
            f"remaining safetensors bytes={status.remaining_safetensors_file_bytes}"
        )
    if status.local_header_check_requested:
        details.append(
            "local headers="
            f"{status.local_header_ok_shard_count}/{status.local_header_checked_shard_count} ok"
        )

    first_bad_shard = next(
        (
            shard
            for shard in status.shards
            if (
                not shard.present
                or shard.complete is False
                or shard.header_ok is False
            )
        ),
        None,
    )
    if first_bad_shard is not None:
        shard_details = [first_bad_shard.name]
        if first_bad_shard.issue is not None:
            shard_details.append(f"issue={first_bad_shard.issue}")
        if first_bad_shard.actual_file_bytes is not None:
            shard_details.append(f"actual={first_bad_shard.actual_file_bytes}")
        if first_bad_shard.expected_file_bytes is not None:
            shard_details.append(f"expected={first_bad_shard.expected_file_bytes}")
        if first_bad_shard.header_error is not None:
            shard_details.append(f"header_error={first_bad_shard.header_error}")
        details.append("first bad shard: " + ", ".join(shard_details))

    raise PrepareError(
        "prepare execute requires a complete, clean checkpoint artifact; "
        + "; ".join(details)
        + ". Run checkpoint-status --verify-local-headers --require-complete "
        "--require-clean before --execute."
    )


def _report_model_config_sha256(report: PrepareGlmReport) -> str | None:
    digests: list[str] = []
    if report.expert_pack is not None:
        digest = report.expert_pack.layout.config_sha256
        if digest is not None:
            digests.append(digest)
    if report.resident_pack is not None:
        digest = report.resident_pack.layout.config_sha256
        if digest is not None:
            digests.append(digest)
    unique = tuple(dict.fromkeys(digests))
    if len(unique) > 1:
        raise PrepareError("expert and resident layouts disagree on config_sha256")
    return unique[0] if unique else None


def _manifest_cold_read_metadata(report: PrepareGlmReport) -> dict[str, object]:
    benchmark = report.cold_read_benchmark
    if benchmark is not None:
        return {
            "prepare_cold_read_gib_per_second": benchmark.gib_per_second,
            "prepare_cold_read_source": "auto_benchmark",
            "prepare_cold_read_benchmark_path": str(benchmark.path),
            "prepare_cold_read_benchmark_requested_bytes": benchmark.requested_bytes,
            "prepare_cold_read_benchmark_measured_bytes": benchmark.measured_bytes,
            "prepare_cold_read_benchmark_elapsed_seconds": (
                benchmark.elapsed_seconds
            ),
        }
    return {
        "prepare_cold_read_gib_per_second": (
            report.preflight.cold_read_gib_per_second
        ),
        "prepare_cold_read_source": (
            "explicit" if report.preflight.cold_read_gib_per_second is not None else None
        ),
        "prepare_cold_read_benchmark_path": None,
        "prepare_cold_read_benchmark_requested_bytes": None,
        "prepare_cold_read_benchmark_measured_bytes": None,
        "prepare_cold_read_benchmark_elapsed_seconds": None,
    }


def _manifest_context_budget_metadata(report: PrepareGlmReport) -> dict[str, object]:
    budget = report.context_budget
    if budget is None:
        return {
            "prepare_auto_context_from_budget": None,
            "prepare_requested_max_context_tokens": None,
            "prepare_resolved_max_context_tokens": (
                report.cache_layout.max_context_tokens if report.cache_layout else None
            ),
            "prepare_decode_cache_budget_bytes": None,
            "prepare_decode_cache_safe_context_tokens": None,
            "prepare_effective_max_cache_bytes": None,
            "prepare_model_max_position_embeddings": None,
            "prepare_cache_dtype": (
                report.cache_layout.dtype if report.cache_layout else None
            ),
            "prepare_cache_alignment": (
                report.cache_layout.alignment if report.cache_layout else None
            ),
        }
    return {
        "prepare_auto_context_from_budget": budget.auto_context_from_budget,
        "prepare_requested_max_context_tokens": budget.requested_max_context_tokens,
        "prepare_resolved_max_context_tokens": budget.resolved_max_context_tokens,
        "prepare_decode_cache_budget_bytes": budget.decode_cache_budget_bytes,
        "prepare_decode_cache_safe_context_tokens": (
            budget.decode_cache_safe_context_tokens
        ),
        "prepare_effective_max_cache_bytes": budget.effective_max_cache_bytes,
        "prepare_model_max_position_embeddings": budget.model_max_position_embeddings,
        "prepare_cache_dtype": budget.cache_dtype,
        "prepare_cache_alignment": budget.cache_alignment,
    }


def _manifest_prepare_flags_metadata(report: PrepareGlmReport) -> dict[str, object]:
    provenance = report.prepare_flags
    if provenance is None:
        return {
            "prepare_flags_applied": False,
            "prepare_flags_source": None,
            "prepare_flags_path": None,
            "prepare_flags_sha256": None,
        }
    return {
        "prepare_flags_applied": True,
        "prepare_flags_source": provenance.source,
        "prepare_flags_path": provenance.path,
        "prepare_flags_sha256": provenance.sha256,
    }


def _manifest_prepare_live_memory_metadata(
    report: PrepareGlmReport,
) -> dict[str, object]:
    live = report.prepare_live_memory
    if live is None:
        return {
            "prepare_live_memory_estimated_live_working_set_bytes": None,
            "prepare_live_memory_min_available_memory_bytes": None,
            "prepare_live_memory_required_available_memory_bytes": None,
            "prepare_live_memory_system_available_memory_bytes": None,
            "prepare_live_memory_system_total_bytes": None,
            "prepare_live_memory_system_source": None,
        }
    required_available = (
        live.estimated_live_working_set_bytes
        + live.min_available_memory_bytes
    )
    return {
        "prepare_live_memory_estimated_live_working_set_bytes": (
            live.estimated_live_working_set_bytes
        ),
        "prepare_live_memory_min_available_memory_bytes": (
            live.min_available_memory_bytes
        ),
        "prepare_live_memory_required_available_memory_bytes": required_available,
        "prepare_live_memory_system_available_memory_bytes": (
            live.system_available_bytes
        ),
        "prepare_live_memory_system_total_bytes": live.system_total_bytes,
        "prepare_live_memory_system_source": live.system_source,
    }


def _manifest_prepare_disk_budget_metadata(
    report: PrepareGlmReport,
) -> dict[str, object]:
    budget = report.prepare_disk_budget
    if budget is None:
        return {
            "prepare_combined_output_required_bytes": None,
            "prepare_combined_output_available_bytes": None,
            "prepare_combined_output_disk_margin_bytes": None,
        }
    return {
        "prepare_combined_output_required_bytes": budget.required_bytes,
        "prepare_combined_output_available_bytes": budget.available_bytes,
        "prepare_combined_output_disk_margin_bytes": budget.safety_margin_bytes,
    }


def _manifest_expert_pack_metadata(report: PrepareGlmReport) -> dict[str, object]:
    pack = report.expert_pack
    if pack is None:
        return {
            "prepare_expert_pack_chunk_size_bytes": None,
            "prepare_expert_pack_estimated_peak_heap_bytes": None,
            "prepare_expert_pack_max_heap_bytes": None,
            "prepare_raw_quantization_extra_heap_bytes": None,
            "prepare_raw_quantization_max_source_block_bytes": None,
            "prepare_raw_quantization_max_output_block_bytes": None,
            "prepare_raw_quantization_max_rows_per_block": None,
        }
    return {
        "prepare_expert_pack_chunk_size_bytes": pack.chunk_size,
        "prepare_expert_pack_estimated_peak_heap_bytes": (
            pack.estimated_peak_heap_bytes
        ),
        "prepare_expert_pack_max_heap_bytes": pack.max_heap_bytes,
        "prepare_raw_quantization_extra_heap_bytes": (
            pack.raw_quantization_extra_heap_bytes
        ),
        "prepare_raw_quantization_max_source_block_bytes": (
            pack.raw_quantization_max_source_block_bytes
        ),
        "prepare_raw_quantization_max_output_block_bytes": (
            pack.raw_quantization_max_output_block_bytes
        ),
        "prepare_raw_quantization_max_rows_per_block": (
            pack.raw_quantization_max_rows_per_block
        ),
    }


def _manifest_resident_pack_metadata(report: PrepareGlmReport) -> dict[str, object]:
    pack = report.resident_pack
    if pack is None:
        return {
            "prepare_resident_component_alias_source_tensor_count": None,
            "prepare_resident_component_alias_renamed_tensor_count": None,
            "prepare_resident_component_alias_bytes": None,
            "prepare_resident_fused_gate_up_source_tensor_count": None,
            "prepare_resident_fused_gate_up_expanded_tensor_count": None,
            "prepare_resident_fused_gate_up_expanded_bytes": None,
        }
    return {
        "prepare_resident_component_alias_source_tensor_count": (
            pack.component_alias_source_tensor_count
        ),
        "prepare_resident_component_alias_renamed_tensor_count": (
            pack.component_alias_renamed_tensor_count
        ),
        "prepare_resident_component_alias_bytes": pack.component_alias_bytes,
        "prepare_resident_fused_gate_up_source_tensor_count": (
            pack.fused_gate_up_source_tensor_count
        ),
        "prepare_resident_fused_gate_up_expanded_tensor_count": (
            pack.fused_gate_up_expanded_tensor_count
        ),
        "prepare_resident_fused_gate_up_expanded_bytes": (
            pack.fused_gate_up_expanded_bytes
        ),
    }


def _manifest_public_glm_5_2_shape_metadata(
    report: PrepareGlmReport,
) -> dict[str, object]:
    shape = report.public_glm_5_2_shape
    matches = shape.get("matches") is True if isinstance(shape, dict) else None
    mismatched_fields: tuple[str, ...] | None = None
    if isinstance(shape, dict):
        raw_fields = shape.get("mismatched_fields")
        if isinstance(raw_fields, (list, tuple)):
            mismatched_fields = tuple(
                field for field in raw_fields if isinstance(field, str)
            )
    return {
        "prepare_public_glm_5_2_shape_required": bool(
            report.require_public_glm_5_2_shape
        ),
        "prepare_public_glm_5_2_shape_matches": matches,
        "prepare_public_glm_5_2_shape_mismatched_fields": mismatched_fields,
    }


def _manifest_path_value(path: Path, target: Path) -> str:
    try:
        return str(target.relative_to(path.parent))
    except ValueError:
        return str(target)


def _write_manifest(path: Path, report: PrepareGlmReport, *, force: bool) -> None:
    if path.exists() and not force:
        raise PrepareError(f"{path} already exists; use --force to overwrite")
    payload = {
        "version": 1,
        "executed": report.executed,
        "model_dir": str(report.preflight.model_dir.resolve()),
        "model_config_sha256": _report_model_config_sha256(report),
        "experts_layout": _manifest_path_value(
            path, report.paths.experts_dir / "layout.json"
        ),
        "resident_layout": _manifest_path_value(
            path, report.paths.resident_dir / "layout.json"
        ),
        "decode_cache_layout": _manifest_path_value(path, report.paths.cache_layout_path),
        "decode_cache_file": _manifest_path_value(path, report.paths.cache_file_path),
        "max_context_tokens": (
            report.cache_layout.max_context_tokens if report.cache_layout else None
        ),
        "expert_bytes": (
            report.expert_pack.layout.total_bytes if report.expert_pack else None
        ),
        "resident_bytes": (
            report.resident_pack.layout.total_bytes if report.resident_pack else None
        ),
        "decode_cache_bytes": (
            report.cache_layout.total_bytes if report.cache_layout else None
        ),
        "recommended_max_live_working_set_bytes": (
            report.preflight.recommended_max_live_working_set_bytes
        ),
        "recommended_min_free_unified_memory_bytes": (
            report.preflight.recommended_min_free_unified_memory_bytes
        ),
        "expert_quantization": (
            report.expert_pack.layout.quantization if report.expert_pack else None
        ),
        "expert_group_size": (
            report.expert_pack.layout.group_size if report.expert_pack else None
        ),
        "prepare_hardware_chip_name": report.preflight.hardware_chip_name,
        "prepare_hardware_unified_memory_bytes": (
            report.preflight.hardware_unified_memory_bytes
        ),
        "prepare_hardware_gpu_cores": report.preflight.hardware_gpu_cores,
        "prepare_hardware_apple_silicon_generation": (
            report.preflight.hardware_apple_silicon_generation
        ),
        "prepare_hardware_apple_silicon_tier": (
            report.preflight.hardware_apple_silicon_tier
        ),
        "prepare_effective_unified_memory_bytes": (
            report.preflight.effective_unified_memory_bytes
        ),
        "prepare_effective_unified_memory_source": (
            report.preflight.effective_unified_memory_source
        ),
        "prepare_system_reserve_bytes": (
            report.preflight.effective_system_reserve_bytes
        ),
    }
    payload = (
        payload
        | _manifest_cold_read_metadata(report)
        | _manifest_context_budget_metadata(report)
        | _manifest_prepare_flags_metadata(report)
        | _manifest_prepare_disk_budget_metadata(report)
        | _manifest_prepare_live_memory_metadata(report)
        | _manifest_expert_pack_metadata(report)
        | _manifest_resident_pack_metadata(report)
        | _manifest_public_glm_5_2_shape_metadata(report)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, payload)


def _prepared_glm_4bit_readiness_failure_detail(
    readiness: dict[str, object],
) -> str:
    issues = readiness.get("issues")
    if isinstance(issues, list) and issues:
        return "; ".join(str(issue) for issue in issues[:5])
    return "readiness check did not pass"


def _validate_prepared_glm_4bit_output(prepared) -> dict[str, object]:
    from .server import prepared_glm_4bit_readiness

    readiness = prepared_glm_4bit_readiness(prepared)
    if readiness.get("ok") is not True:
        raise PrepareError(
            "prepared GLM 4bit readiness failed after prepare: "
            f"{_prepared_glm_4bit_readiness_failure_detail(readiness)}"
    )
    return readiness


def _check_prepare_execute_live_memory(
    *,
    preflight: GlmPreflightReport,
    expert_report: PackReport,
    resident_report: ResidentPackReport,
    cold_read_benchmark_chunk_bytes: int = 0,
) -> LiveMemoryBudget:
    estimated_live_bytes = max(
        int(expert_report.estimated_peak_heap_bytes),
        int(resident_report.estimated_peak_heap_bytes),
        int(cold_read_benchmark_chunk_bytes),
    )
    try:
        return check_live_memory_budget(
            estimated_live_working_set_bytes=estimated_live_bytes,
            max_live_working_set_bytes=None,
            min_available_memory_bytes=int(
                preflight.effective_system_reserve_bytes
            ),
            nonresident_peak_bytes=estimated_live_bytes,
        )
    except GenerationGuardError as exc:
        raise PrepareError(f"prepare live memory guard failed: {exc}") from exc


def _check_prepare_execute_disk_budget(
    *,
    paths: PreparePaths,
    expert_report: PackReport,
    resident_report: ResidentPackReport,
    cache_layout: DecodeCacheLayout,
    disk_safety_margin_bytes: int,
) -> DiskBudget:
    required_bytes = (
        int(expert_report.layout.total_bytes)
        + int(resident_report.layout.total_bytes)
        + int(cache_layout.total_bytes)
    )
    budget = disk_budget(
        paths.output_dir,
        required_bytes,
        safety_margin_bytes=disk_safety_margin_bytes,
    )
    if not budget.ok:
        raise PrepareError(
            "not enough free disk for combined prepare outputs: "
            f"need {required_bytes + disk_safety_margin_bytes} bytes including "
            f"margin, have {budget.available_bytes} bytes"
        )
    return budget


def _public_glm_5_2_shape_report(config) -> dict[str, object]:
    from .server import _public_glm_5_2_shape_report

    return _public_glm_5_2_shape_report(config)


def _public_glm_5_2_shape_failure_detail(report: dict[str, object]) -> str | None:
    fields = report.get("mismatched_fields")
    if not isinstance(fields, (list, tuple)) or not fields:
        return None
    preview = ", ".join(str(field) for field in tuple(fields)[:6])
    if len(fields) > 6:
        preview += f", +{len(fields) - 6} more"
    return preview


def _require_public_glm_5_2_shape(report: dict[str, object]) -> None:
    if report.get("matches") is True:
        return
    detail = _public_glm_5_2_shape_failure_detail(report)
    suffix = f" ({detail})" if detail else ""
    raise PrepareError(
        "prepared config does not match the public GLM-5.2 shape"
        f"{suffix}"
    )


def _largest_expert_layer_file(
    paths: PreparePaths,
    expert_report: PackReport,
) -> Path:
    candidates: list[Path] = []
    for layer in expert_report.layout.layers:
        path = paths.experts_dir / layer.layer_file
        if path.exists():
            candidates.append(path)
    if not candidates:
        raise PrepareError("no packed expert layer file is available for benchmarking")
    try:
        return max(candidates, key=lambda path: path.stat().st_size)
    except OSError as exc:
        raise PrepareError(f"failed to inspect packed expert layer files: {exc}") from exc


def _largest_cache_context_that_fits(
    cfg,
    *,
    upper_bound: int,
    dtype: str,
    alignment: int,
    max_cache_bytes: int | None,
) -> int:
    if upper_bound <= 0:
        return 0
    if max_cache_bytes is None:
        return upper_bound
    lo = 0
    hi = upper_bound
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            build_decode_cache_layout(
                cfg,
                max_context_tokens=mid,
                dtype=dtype,
                alignment=alignment,
                max_cache_bytes=max_cache_bytes,
            )
        except DecodeCacheError:
            hi = mid - 1
        else:
            lo = mid
    return lo


def prepare_glm_checkpoint(
    model_dir: str | Path,
    *,
    output_dir: str | Path,
    max_context_tokens: int | None,
    auto_context_from_budget: bool = False,
    execute: bool = False,
    force: bool = False,
    tokenizer_path: str | Path | None = None,
    tokenizer_backend: str = "auto",
    trust_remote_code: bool = False,
    load_tokenizer_backend: bool = False,
    quant_bits: int = 4,
    group_size: int = 64,
    quantize_raw_to_int4: bool = False,
    require_public_glm_5_2_shape: bool = False,
    cache_dtype: str = "BF16",
    cache_alignment: int = 64,
    max_cache_bytes: int | None = None,
    disk_safety_margin_bytes: int = 16 * 1024**3,
    chunk_size: int = 8 * 1024**2,
    max_chunk_size: int = 64 * 1024**2,
    max_pack_heap_bytes: int = 512 * 1024**2,
    unified_memory_bytes: int | None = None,
    system_reserve_bytes: int | None = None,
    runtime_buffer_bytes: int = 8 * 1024**3,
    page_cache_fraction: float = 0.60,
    cold_read_gib_per_second: float | None = None,
    auto_cold_read_benchmark: bool = False,
    cold_read_benchmark_bytes: int = 1024 * 1024**2,
    cold_read_benchmark_chunk_bytes: int = 8 * 1024**2,
    prepare_flags: PrepareFlagsProvenance | None = None,
    prefer_header_manifest: bool = False,
) -> PrepareGlmReport:
    root = Path(model_dir)
    out = Path(output_dir)
    _validate_prepare_output_dir(root, out)
    paths = PreparePaths(
        output_dir=out,
        experts_dir=out / "experts",
        resident_dir=out / "resident",
        cache_layout_path=out / "decode_cache_layout.json",
        cache_file_path=out / "decode_cache.bin",
        manifest_path=out / "manifest.json",
    )
    if auto_context_from_budget and max_context_tokens is not None:
        raise PrepareError("--auto-context-from-budget conflicts with --max-context-tokens")
    if not auto_context_from_budget and max_context_tokens is None:
        raise PrepareError("--max-context-tokens is required unless auto context is enabled")
    max_context_tokens = _optional_positive_integer_param(
        "max_context_tokens",
        max_context_tokens,
    )
    quant_bits = _positive_integer_param("quant_bits", quant_bits)
    group_size = _positive_integer_param("group_size", group_size)
    cache_alignment = _positive_integer_param("cache_alignment", cache_alignment)
    max_cache_bytes = _optional_nonnegative_integer_param(
        "max_cache_bytes",
        max_cache_bytes,
    )
    disk_safety_margin_bytes = _nonnegative_integer_param(
        "disk_safety_margin_bytes",
        disk_safety_margin_bytes,
    )
    chunk_size = _positive_integer_param("chunk_size", chunk_size)
    max_chunk_size = _positive_integer_param("max_chunk_size", max_chunk_size)
    max_pack_heap_bytes = _positive_integer_param(
        "max_pack_heap_bytes",
        max_pack_heap_bytes,
    )
    unified_memory_bytes = _optional_positive_integer_param(
        "unified_memory_bytes",
        unified_memory_bytes,
    )
    system_reserve_bytes = _optional_nonnegative_integer_param(
        "system_reserve_bytes",
        system_reserve_bytes,
    )
    runtime_buffer_bytes = _positive_integer_param(
        "runtime_buffer_bytes",
        runtime_buffer_bytes,
    )
    page_cache_fraction = _fraction_param("page_cache_fraction", page_cache_fraction)
    cold_read_gib_per_second = _optional_positive_numeric_param(
        "cold_read_gib_per_second",
        cold_read_gib_per_second,
    )
    if type(require_public_glm_5_2_shape) is not bool:
        raise PrepareError("require_public_glm_5_2_shape must be a boolean")
    if type(auto_cold_read_benchmark) is not bool:
        raise PrepareError("auto_cold_read_benchmark must be a boolean")
    if type(prefer_header_manifest) is not bool:
        raise PrepareError("prefer_header_manifest must be a boolean")
    if execute and prefer_header_manifest:
        raise PrepareError("--metadata-only cannot be used with --execute")
    if execute:
        _require_execute_checkpoint_artifact_ready(root)
    cold_read_benchmark_bytes = _positive_integer_param(
        "cold_read_benchmark_bytes",
        cold_read_benchmark_bytes,
    )
    cold_read_benchmark_chunk_bytes = _positive_integer_param(
        "cold_read_benchmark_chunk_bytes",
        cold_read_benchmark_chunk_bytes,
    )
    max_cold_read_benchmark_chunk_bytes = min(
        max_pack_heap_bytes,
        MAX_SEQUENTIAL_READ_CHUNK_BYTES,
    )
    if cold_read_benchmark_chunk_bytes > max_cold_read_benchmark_chunk_bytes:
        raise PrepareError(
            "cold_read_benchmark_chunk_bytes exceeds safe benchmark chunk limit "
            f"{max_cold_read_benchmark_chunk_bytes}"
        )
    if auto_cold_read_benchmark and not execute:
        raise PrepareError("auto_cold_read_benchmark requires execute=True")
    if auto_cold_read_benchmark and cold_read_gib_per_second is not None:
        raise PrepareError(
            "auto_cold_read_benchmark conflicts with cold_read_gib_per_second"
        )
    cfg_for_shape = load_config(root)
    public_shape_report = _public_glm_5_2_shape_report(cfg_for_shape)
    if require_public_glm_5_2_shape:
        _require_public_glm_5_2_shape(public_shape_report)
    effective_max_cache_bytes = max_cache_bytes
    requested_max_context_tokens = max_context_tokens
    auto_context_budget_bytes: int | None = None
    auto_context_safe_context_tokens: int | None = None
    model_max_position_embeddings = (
        int(cfg_for_shape.max_position_embeddings)
        if cfg_for_shape.max_position_embeddings is not None
        else None
    )
    if auto_context_from_budget:
        sizing_preflight = preflight_glm_checkpoint(
            root,
            tokenizer_path=tokenizer_path,
            tokenizer_backend=tokenizer_backend,
            trust_remote_code=trust_remote_code,
            load_tokenizer_backend=load_tokenizer_backend,
            quant_bits=quant_bits,
            group_size=group_size,
            quantize_raw_to_int4=quantize_raw_to_int4,
            require_public_glm_5_2_shape=require_public_glm_5_2_shape,
            max_context_tokens=None,
            max_cache_bytes=max_cache_bytes,
            output_dir=out,
            disk_safety_margin_bytes=disk_safety_margin_bytes,
            unified_memory_bytes=unified_memory_bytes,
            system_reserve_bytes=system_reserve_bytes,
            runtime_buffer_bytes=runtime_buffer_bytes,
            page_cache_fraction=page_cache_fraction,
            cold_read_gib_per_second=cold_read_gib_per_second,
            prefer_header_manifest=prefer_header_manifest,
        )
        if not sizing_preflight.ok:
            sizing_plan = sizing_preflight.plan
            return PrepareGlmReport(
                ok=False,
                executed=False,
                paths=paths,
                preflight=sizing_preflight,
                expert_pack=None,
                resident_pack=None,
                cache_layout=None,
                cache_init=None,
                context_budget=PrepareContextBudget(
                    auto_context_from_budget=True,
                    requested_max_context_tokens=requested_max_context_tokens,
                    resolved_max_context_tokens=None,
                    decode_cache_budget_bytes=(
                        sizing_plan.decode_cache_budget_bytes
                        if sizing_plan is not None
                        else None
                    ),
                    decode_cache_safe_context_tokens=(
                        sizing_plan.decode_cache_safe_context_tokens
                        if sizing_plan is not None
                        else None
                    ),
                    effective_max_cache_bytes=effective_max_cache_bytes,
                    model_max_position_embeddings=model_max_position_embeddings,
                    cache_dtype=cache_dtype,
                    cache_alignment=cache_alignment,
                ),
                prepare_flags=prepare_flags,
                public_glm_5_2_shape=public_shape_report,
                require_public_glm_5_2_shape=require_public_glm_5_2_shape,
            )
        sizing_plan = sizing_preflight.plan
        auto_context_budget_bytes = (
            sizing_plan.decode_cache_budget_bytes if sizing_plan is not None else None
        )
        auto_context_safe_context_tokens = (
            sizing_plan.decode_cache_safe_context_tokens if sizing_plan is not None else None
        )
        safe_context = (
            sizing_plan.decode_cache_safe_context_tokens if sizing_plan is not None else None
        )
        if safe_context is None:
            raise PrepareError(
                "auto context requires MLA/DSA cache dimensions and a cache budget"
            )
        cfg = load_config(root)
        if model_max_position_embeddings is not None:
            safe_context = min(safe_context, model_max_position_embeddings)
        if effective_max_cache_bytes is None and sizing_plan is not None:
            effective_max_cache_bytes = sizing_plan.decode_cache_budget_bytes
        safe_context = _largest_cache_context_that_fits(
            cfg,
            upper_bound=int(safe_context),
            dtype=cache_dtype,
            alignment=cache_alignment,
            max_cache_bytes=effective_max_cache_bytes,
        )
        if safe_context <= 0:
            raise PrepareError("auto context resolved to zero tokens")
        max_context_tokens = int(safe_context)

    assert max_context_tokens is not None
    preflight = preflight_glm_checkpoint(
        root,
        tokenizer_path=tokenizer_path,
        tokenizer_backend=tokenizer_backend,
        trust_remote_code=trust_remote_code,
        load_tokenizer_backend=load_tokenizer_backend,
        quant_bits=quant_bits,
        group_size=group_size,
        quantize_raw_to_int4=quantize_raw_to_int4,
        require_public_glm_5_2_shape=require_public_glm_5_2_shape,
        max_context_tokens=max_context_tokens,
        max_cache_bytes=effective_max_cache_bytes,
        output_dir=out,
        disk_safety_margin_bytes=disk_safety_margin_bytes,
        unified_memory_bytes=unified_memory_bytes,
        system_reserve_bytes=system_reserve_bytes,
        runtime_buffer_bytes=runtime_buffer_bytes,
        page_cache_fraction=page_cache_fraction,
        cold_read_gib_per_second=cold_read_gib_per_second,
        prefer_header_manifest=prefer_header_manifest,
    )
    plan = preflight.plan
    context_budget = PrepareContextBudget(
        auto_context_from_budget=auto_context_from_budget,
        requested_max_context_tokens=requested_max_context_tokens,
        resolved_max_context_tokens=max_context_tokens,
        decode_cache_budget_bytes=(
            plan.decode_cache_budget_bytes
            if plan is not None
            else auto_context_budget_bytes
        ),
        decode_cache_safe_context_tokens=(
            plan.decode_cache_safe_context_tokens
            if plan is not None
            else auto_context_safe_context_tokens
        ),
        effective_max_cache_bytes=effective_max_cache_bytes,
        model_max_position_embeddings=model_max_position_embeddings,
        cache_dtype=cache_dtype,
        cache_alignment=cache_alignment,
    )
    if not preflight.ok:
        return PrepareGlmReport(
            ok=False,
            executed=False,
            paths=paths,
            preflight=preflight,
            expert_pack=None,
            resident_pack=None,
            cache_layout=None,
            cache_init=None,
            context_budget=context_budget,
            prepare_flags=prepare_flags,
            public_glm_5_2_shape=public_shape_report,
            require_public_glm_5_2_shape=require_public_glm_5_2_shape,
        )

    cfg = cfg_for_shape
    cache_layout = build_decode_cache_layout(
        cfg,
        max_context_tokens=max_context_tokens,
        dtype=cache_dtype,
        alignment=cache_alignment,
        max_cache_bytes=effective_max_cache_bytes,
    )
    expert_report = pack_experts(
        root,
        paths.experts_dir,
        dry_run=True,
        force=force,
        chunk_size=chunk_size,
        max_chunk_size=max_chunk_size,
        max_heap_bytes=max_pack_heap_bytes,
        disk_safety_margin_bytes=disk_safety_margin_bytes,
        quantize_raw_to_int4=quantize_raw_to_int4,
        group_size=group_size,
        prefer_header_manifest=prefer_header_manifest,
    )
    resident_report = pack_resident_weights(
        root,
        paths.resident_dir,
        dry_run=True,
        force=force,
        chunk_size=chunk_size,
        max_chunk_size=max_chunk_size,
        max_heap_bytes=max_pack_heap_bytes,
        disk_safety_margin_bytes=disk_safety_margin_bytes,
        prefer_header_manifest=prefer_header_manifest,
    )

    if not execute:
        return PrepareGlmReport(
            ok=True,
            executed=False,
            paths=paths,
            preflight=preflight,
            expert_pack=expert_report,
            resident_pack=resident_report,
            cache_layout=cache_layout,
            cache_init=None,
            context_budget=context_budget,
            prepare_flags=prepare_flags,
            public_glm_5_2_shape=public_shape_report,
            require_public_glm_5_2_shape=require_public_glm_5_2_shape,
        )

    pre_existing = set(_existing_outputs(paths, expert_report=expert_report))
    if pre_existing and not force:
        existing_preview = sorted(pre_existing, key=str)
        preview = ", ".join(str(path) for path in existing_preview[:6])
        more = "" if len(existing_preview) <= 6 else f", +{len(existing_preview) - 6} more"
        raise PrepareError(f"prepare outputs already exist: {preview}{more}; use --force")

    prepare_disk_budget = _check_prepare_execute_disk_budget(
        paths=paths,
        expert_report=expert_report,
        resident_report=resident_report,
        cache_layout=cache_layout,
        disk_safety_margin_bytes=disk_safety_margin_bytes,
    )
    prepare_live_memory = _check_prepare_execute_live_memory(
        preflight=preflight,
        expert_report=expert_report,
        resident_report=resident_report,
        cold_read_benchmark_chunk_bytes=(
            cold_read_benchmark_chunk_bytes if auto_cold_read_benchmark else 0
        ),
    )

    try:
        resident_report = pack_resident_weights(
            root,
            paths.resident_dir,
            dry_run=False,
            force=force,
            chunk_size=chunk_size,
            max_chunk_size=max_chunk_size,
            max_heap_bytes=max_pack_heap_bytes,
            disk_safety_margin_bytes=disk_safety_margin_bytes,
        )
        expert_report = pack_experts(
            root,
            paths.experts_dir,
            dry_run=False,
            force=force,
            chunk_size=chunk_size,
            max_chunk_size=max_chunk_size,
            max_heap_bytes=max_pack_heap_bytes,
            disk_safety_margin_bytes=disk_safety_margin_bytes,
            quantize_raw_to_int4=quantize_raw_to_int4,
            group_size=group_size,
        )
        _write_cache_layout(paths.cache_layout_path, cache_layout, force=force)
        cache_init = init_decode_cache_file(
            paths.cache_layout_path,
            paths.cache_file_path,
            force=force,
            max_cache_bytes=effective_max_cache_bytes,
            disk_safety_margin_bytes=disk_safety_margin_bytes,
        )
        cold_read_benchmark = None
        if auto_cold_read_benchmark:
            try:
                cold_read_benchmark = benchmark_sequential_read(
                    _largest_expert_layer_file(paths, expert_report),
                    bytes_to_read=cold_read_benchmark_bytes,
                    chunk_bytes=cold_read_benchmark_chunk_bytes,
                    max_chunk_bytes=max_cold_read_benchmark_chunk_bytes,
                )
            except DiskBenchmarkError as exc:
                raise PrepareError(f"cold read benchmark failed: {exc}") from exc
        report = PrepareGlmReport(
            ok=True,
            executed=True,
            paths=paths,
            preflight=preflight,
            expert_pack=expert_report,
            resident_pack=resident_report,
            cache_layout=cache_layout,
            cache_init=cache_init,
            context_budget=context_budget,
            prepare_flags=prepare_flags,
            prepare_disk_budget=prepare_disk_budget,
            prepare_live_memory=prepare_live_memory,
            cold_read_benchmark=cold_read_benchmark,
            public_glm_5_2_shape=public_shape_report,
            require_public_glm_5_2_shape=require_public_glm_5_2_shape,
        )
        _write_manifest(paths.manifest_path, report, force=force)
        try:
            prepared = load_prepared_manifest(paths.manifest_path)
        except PreparedManifestError as exc:
            raise PrepareError(f"prepared output validation failed: {exc}") from exc
        readiness = _validate_prepared_glm_4bit_output(prepared)
        report = replace(report, glm_4bit_readiness=readiness)
    except Exception:
        _cleanup_prepare_outputs(
            paths,
            expert_report=expert_report,
            preserve=pre_existing,
        )
        raise
    return report
