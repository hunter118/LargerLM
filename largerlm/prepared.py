from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .decode_cache import DecodeCacheError, load_decode_cache_layout
from .layout import config_sha256


class PreparedManifestError(RuntimeError):
    """Raised when a prepared LargerLM manifest cannot be used."""


@dataclass(frozen=True)
class PreparedManifest:
    manifest_path: Path
    model_dir: Path
    experts_layout: Path
    resident_layout: Path
    decode_cache_layout: Path
    decode_cache_file: Path
    max_context_tokens: int | None
    recommended_max_live_working_set_bytes: int | None = None
    recommended_min_free_unified_memory_bytes: int | None = None
    expert_quantization: str | None = None
    expert_group_size: int | None = None
    prepare_hardware_chip_name: str | None = None
    prepare_hardware_unified_memory_bytes: int | None = None
    prepare_hardware_gpu_cores: int | None = None
    prepare_hardware_apple_silicon_generation: int | None = None
    prepare_hardware_apple_silicon_tier: str | None = None
    prepare_effective_unified_memory_bytes: int | None = None
    prepare_effective_unified_memory_source: str | None = None
    prepare_system_reserve_bytes: int | None = None
    prepare_cold_read_gib_per_second: float | None = None
    prepare_cold_read_source: str | None = None
    prepare_cold_read_benchmark_path: str | None = None
    prepare_cold_read_benchmark_requested_bytes: int | None = None
    prepare_cold_read_benchmark_measured_bytes: int | None = None
    prepare_cold_read_benchmark_elapsed_seconds: float | None = None
    prepare_auto_context_from_budget: bool | None = None
    prepare_requested_max_context_tokens: int | None = None
    prepare_resolved_max_context_tokens: int | None = None
    prepare_decode_cache_budget_bytes: int | None = None
    prepare_decode_cache_safe_context_tokens: int | None = None
    prepare_effective_max_cache_bytes: int | None = None
    prepare_model_max_position_embeddings: int | None = None
    prepare_cache_dtype: str | None = None
    prepare_cache_alignment: int | None = None
    prepare_flags_applied: bool | None = None
    prepare_flags_source: str | None = None
    prepare_flags_path: str | None = None
    prepare_flags_sha256: str | None = None
    prepare_combined_output_required_bytes: int | None = None
    prepare_combined_output_available_bytes: int | None = None
    prepare_combined_output_disk_margin_bytes: int | None = None
    prepare_live_memory_estimated_live_working_set_bytes: int | None = None
    prepare_live_memory_min_available_memory_bytes: int | None = None
    prepare_live_memory_required_available_memory_bytes: int | None = None
    prepare_live_memory_system_available_memory_bytes: int | None = None
    prepare_live_memory_system_total_bytes: int | None = None
    prepare_live_memory_system_source: str | None = None
    prepare_expert_pack_chunk_size_bytes: int | None = None
    prepare_expert_pack_estimated_peak_heap_bytes: int | None = None
    prepare_expert_pack_max_heap_bytes: int | None = None
    prepare_raw_quantization_extra_heap_bytes: int | None = None
    prepare_raw_quantization_max_source_block_bytes: int | None = None
    prepare_raw_quantization_max_output_block_bytes: int | None = None
    prepare_raw_quantization_max_rows_per_block: int | None = None
    prepare_resident_component_alias_source_tensor_count: int | None = None
    prepare_resident_component_alias_renamed_tensor_count: int | None = None
    prepare_resident_component_alias_bytes: int | None = None
    prepare_resident_fused_gate_up_source_tensor_count: int | None = None
    prepare_resident_fused_gate_up_expanded_tensor_count: int | None = None
    prepare_resident_fused_gate_up_expanded_bytes: int | None = None
    prepare_public_glm_5_2_shape_required: bool | None = None
    prepare_public_glm_5_2_shape_matches: bool | None = None
    prepare_public_glm_5_2_shape_mismatched_fields: tuple[str, ...] | None = None
    expert_layout_bytes: int | None = None
    expert_layout_quantization: str | None = None
    expert_layout_group_size: int | None = None
    resident_layout_bytes: int | None = None
    decode_cache_layout_bytes: int | None = None
    decode_cache_file_bytes: int | None = None
    model_config_sha256: str | None = None


@dataclass(frozen=True)
class LayoutBackingValidation:
    expert_layout_bytes: int
    resident_layout_bytes: int
    expert_config_sha256: str | None = None
    resident_config_sha256: str | None = None
    expert_quantization: str | None = None
    expert_group_size: int | None = None


@dataclass(frozen=True)
class _ValidatedLayoutBacking:
    total_bytes: int
    config_sha256: str | None
    quantization: str | None = None
    group_size: int | None = None


_INTERNAL_AFFINE_INT4_QUANTIZATION = "largerlm-affine-int4"

_PREPARE_EXPERT_PACK_HEAP_EVIDENCE_FIELDS = (
    "prepare_expert_pack_chunk_size_bytes",
    "prepare_expert_pack_estimated_peak_heap_bytes",
    "prepare_expert_pack_max_heap_bytes",
    "prepare_raw_quantization_extra_heap_bytes",
    "prepare_raw_quantization_max_source_block_bytes",
    "prepare_raw_quantization_max_output_block_bytes",
    "prepare_raw_quantization_max_rows_per_block",
)


def _path_from_manifest(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreparedManifestError(f"manifest missing {field}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _optional_int(payload: dict[str, Any], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreparedManifestError(f"manifest field {field} must be an integer")
    return int(value)


def _optional_nonnegative_int(payload: dict[str, Any], field: str) -> int | None:
    value = _optional_int(payload, field)
    if value is not None and value < 0:
        raise PreparedManifestError(f"manifest field {field} must be non-negative")
    return value


def _optional_positive_int(payload: dict[str, Any], field: str) -> int | None:
    value = _optional_int(payload, field)
    if value is not None and value <= 0:
        raise PreparedManifestError(f"manifest field {field} must be positive")
    return value


def _optional_str(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PreparedManifestError(f"manifest field {field} must be a non-empty string")
    return value


def _optional_sha256_hex(payload: dict[str, Any], field: str) -> str | None:
    value = _optional_str(payload, field)
    if value is None:
        return None
    if len(value) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in value):
        raise PreparedManifestError(f"manifest field {field} must be a SHA-256 hex digest")
    return value.lower()


def _optional_bool(payload: dict[str, Any], field: str) -> bool | None:
    value = payload.get(field)
    if value is None:
        return None
    if type(value) is not bool:
        raise PreparedManifestError(f"manifest field {field} must be a boolean")
    return value


def _optional_str_tuple(payload: dict[str, Any], field: str) -> tuple[str, ...] | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise PreparedManifestError(f"manifest field {field} must be a string array")
    return tuple(value)


def _optional_positive_float(payload: dict[str, Any], field: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreparedManifestError(f"manifest field {field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PreparedManifestError(f"manifest field {field} must be finite")
    if parsed <= 0:
        raise PreparedManifestError(f"manifest field {field} must be positive")
    return parsed


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreparedManifestError(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PreparedManifestError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreparedManifestError(f"{label} must be a JSON object")
    return payload


def _positive_int(value: Any, field: str, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PreparedManifestError(f"{label} missing positive integer {field}")
    return int(value)


def _nonnegative_int(value: Any, field: str, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PreparedManifestError(f"{label} missing non-negative integer {field}")
    return int(value)


def _layout_backing_path(layout_path: Path, filename: str, label: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise PreparedManifestError(
            f"{label} must be a relative path inside the layout directory"
        )
    return layout_path.parent / path


def _layout_config_sha256(payload: dict[str, Any], label: str) -> str | None:
    value = payload.get("config_sha256")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PreparedManifestError(f"{label} config_sha256 must be a non-empty string")
    return value


def _layout_optional_str(
    payload: dict[str, Any],
    field: str,
    label: str,
) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PreparedManifestError(f"{label} {field} must be a non-empty string")
    return value


def _layout_optional_positive_int(
    payload: dict[str, Any],
    field: str,
    label: str,
) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise PreparedManifestError(f"{label} {field} must be a positive integer")
    return int(value)


def _require_backing_file_size(path: Path, required_bytes: int, label: str) -> None:
    try:
        actual = path.stat().st_size
    except OSError as exc:
        raise PreparedManifestError(f"failed to stat {label} {path}: {exc}") from exc
    if actual < required_bytes:
        raise PreparedManifestError(
            f"{label} is smaller than layout total_bytes: {actual} < {required_bytes}"
        )


def _layout_total_bytes(payload: dict[str, Any], label: str) -> int:
    actual = payload.get("total_bytes")
    if type(actual) is int and actual >= 0:
        return int(actual)
    raw_layers = payload.get("layers")
    if isinstance(raw_layers, list):
        total = 0
        for item in raw_layers:
            if not isinstance(item, dict):
                raise PreparedManifestError(f"{label} layers must be objects")
            num_experts = item.get("num_experts")
            slot_bytes = item.get("expert_slot_bytes")
            if type(num_experts) is not int or type(slot_bytes) is not int:
                raise PreparedManifestError(
                    f"{label} missing integer total_bytes or layer expert sizes"
                )
            if num_experts < 0 or slot_bytes < 0:
                raise PreparedManifestError(
                    f"{label} missing non-negative total_bytes or layer expert sizes"
                )
            total += int(num_experts) * int(slot_bytes)
        return total
    raise PreparedManifestError(f"{label} missing integer total_bytes")


def _validate_layout_spans(entries: Any, *, label: str, total_bytes: int) -> None:
    if not isinstance(entries, list):
        raise PreparedManifestError(f"{label}s must be an array")
    spans: list[tuple[int, int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise PreparedManifestError(f"{label}s must be objects")
        name = entry.get("name")
        suffix = f" {name}" if isinstance(name, str) and name else ""
        offset = _nonnegative_int(entry.get("offset"), "offset", f"{label}{suffix}")
        size = _nonnegative_int(entry.get("size"), "size", f"{label}{suffix}")
        if offset + size > total_bytes:
            raise PreparedManifestError(
                f"{label}{suffix} extends beyond layout size: "
                f"{offset + size} > {total_bytes}"
            )
        spans.append((offset, offset + size, f"{label}{suffix}"))
    spans.sort(key=lambda item: (item[0], item[1], item[2]))
    previous_end = 0
    previous_label = ""
    has_previous = False
    for start, end, span_label in spans:
        if has_previous and start < previous_end:
            raise PreparedManifestError(
                f"{span_label} overlaps {previous_label}: "
                f"{start} < {previous_end}"
            )
        if not has_previous or end > previous_end:
            previous_end = end
            previous_label = span_label
            has_previous = True


def _validate_expert_backing(layout_path: Path) -> _ValidatedLayoutBacking:
    payload = _load_json_object(layout_path, "expert layout")
    quantization = _layout_optional_str(payload, "quantization", "expert layout")
    group_size = _layout_optional_positive_int(
        payload,
        "group_size",
        "expert layout",
    )
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, list):
        raise PreparedManifestError("expert layout missing layers array")
    total = 0
    for item in raw_layers:
        if not isinstance(item, dict):
            raise PreparedManifestError("expert layout layers must be objects")
        layer_file = item.get("layer_file")
        if not isinstance(layer_file, str) or not layer_file:
            raise PreparedManifestError("expert layout layer missing layer_file")
        num_experts = _positive_int(
            item.get("num_experts"),
            "num_experts",
            "expert layout",
        )
        slot_bytes = _positive_int(
            item.get("expert_slot_bytes"),
            "expert_slot_bytes",
            "expert layout",
        )
        layer_id = item.get("layer")
        layer_label = (
            f"expert layout layer {layer_id}"
            if type(layer_id) is int
            else "expert layout layer"
        )
        _validate_layout_spans(
            item.get("components"),
            label=f"{layer_label} component",
            total_bytes=slot_bytes,
        )
        required = num_experts * slot_bytes
        layer_path = _layout_backing_path(
            layout_path,
            layer_file,
            f"expert layer file {layer_file}",
        )
        _require_backing_file_size(
            layer_path,
            required,
            f"expert layer file {layer_file}",
        )
        total += required
    return _ValidatedLayoutBacking(
        total_bytes=total,
        config_sha256=_layout_config_sha256(payload, "expert layout"),
        quantization=quantization,
        group_size=group_size,
    )


def _validate_resident_backing(layout_path: Path) -> _ValidatedLayoutBacking:
    payload = _load_json_object(layout_path, "resident layout")
    weight_file = payload.get("weight_file")
    if not isinstance(weight_file, str) or not weight_file:
        raise PreparedManifestError("resident layout missing weight_file")
    total = _layout_total_bytes(payload, "resident layout")
    _validate_layout_spans(
        payload.get("tensors"),
        label="resident layout tensor",
        total_bytes=total,
    )
    weight_path = _layout_backing_path(
        layout_path,
        weight_file,
        f"resident weight file {weight_file}",
    )
    _require_backing_file_size(
        weight_path,
        total,
        f"resident weight file {weight_file}",
    )
    return _ValidatedLayoutBacking(
        total_bytes=total,
        config_sha256=_layout_config_sha256(payload, "resident layout"),
    )


def _validate_recorded_bytes(
    *,
    label: str,
    expected: int | None,
    actual: int,
) -> None:
    if expected is None:
        return
    if actual != expected:
        raise PreparedManifestError(
            f"manifest {label} bytes {expected} do not match layout total_bytes {actual}"
        )


def _validate_resident_alias_rewrite_bytes(
    *,
    manifest: "PreparedManifest",
    resident_layout_bytes: int,
) -> None:
    alias_bytes = manifest.prepare_resident_component_alias_bytes or 0
    fused_bytes = manifest.prepare_resident_fused_gate_up_expanded_bytes or 0
    if alias_bytes > resident_layout_bytes:
        raise PreparedManifestError(
            "manifest prepare_resident_component_alias_bytes exceeds resident "
            f"layout bytes: {alias_bytes} > {resident_layout_bytes}"
        )
    if fused_bytes > resident_layout_bytes:
        raise PreparedManifestError(
            "manifest prepare_resident_fused_gate_up_expanded_bytes exceeds "
            f"resident layout bytes: {fused_bytes} > {resident_layout_bytes}"
        )
    rewrite_bytes = alias_bytes + fused_bytes
    if rewrite_bytes > resident_layout_bytes:
        raise PreparedManifestError(
            "manifest resident alias rewrite bytes exceed resident layout bytes: "
            f"{rewrite_bytes} > {resident_layout_bytes}"
        )


def _validate_prepare_pack_heap_envelope(
    *,
    estimated_peak_bytes: int | None,
    chunk_size_bytes: int | None,
    raw_extra_heap_bytes: int | None,
    raw_max_source_block_bytes: int | None,
    raw_max_output_block_bytes: int | None,
) -> None:
    if estimated_peak_bytes is not None:
        for field, value in (
            (
                "prepare_raw_quantization_extra_heap_bytes",
                raw_extra_heap_bytes,
            ),
            (
                "prepare_raw_quantization_max_source_block_bytes",
                raw_max_source_block_bytes,
            ),
            (
                "prepare_raw_quantization_max_output_block_bytes",
                raw_max_output_block_bytes,
            ),
        ):
            if value is not None and value > estimated_peak_bytes:
                raise PreparedManifestError(
                    f"manifest {field} exceeds "
                    "prepare_expert_pack_estimated_peak_heap_bytes"
                )
    if (
        raw_extra_heap_bytes is not None
        and raw_max_output_block_bytes is not None
        and raw_extra_heap_bytes < raw_max_output_block_bytes
    ):
        raise PreparedManifestError(
            "manifest prepare_raw_quantization_extra_heap_bytes is smaller than "
            "prepare_raw_quantization_max_output_block_bytes"
        )
    if (
        chunk_size_bytes is None
        or raw_extra_heap_bytes is None
        or raw_max_source_block_bytes is None
        or raw_max_output_block_bytes is None
    ):
        return
    required_extra_heap = (
        max(0, raw_max_source_block_bytes - chunk_size_bytes)
        + raw_max_output_block_bytes
    )
    if raw_extra_heap_bytes < required_extra_heap:
        raise PreparedManifestError(
            "manifest prepare_raw_quantization_extra_heap_bytes does not cover "
            "raw source overflow plus generated output block bytes"
        )


def _validate_required_prepare_pack_heap_evidence(
    *,
    manifest: "PreparedManifest",
    expert_quantization: str | None,
) -> None:
    if expert_quantization != _INTERNAL_AFFINE_INT4_QUANTIZATION:
        return
    missing = [
        field
        for field in _PREPARE_EXPERT_PACK_HEAP_EVIDENCE_FIELDS
        if getattr(manifest, field) is None
    ]
    if missing:
        raise PreparedManifestError(
            "largerlm-affine-int4 prepared manifest requires prepare "
            "expert-pack heap evidence: "
            + ", ".join(missing)
        )


def validate_layout_backing_files(
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
) -> LayoutBackingValidation:
    expert = _validate_expert_backing(Path(expert_layout_path))
    resident = _validate_resident_backing(Path(resident_layout_path))
    if (
        expert.config_sha256 is not None
        and resident.config_sha256 is not None
        and expert.config_sha256 != resident.config_sha256
    ):
        raise PreparedManifestError(
            "expert and resident layout config_sha256 values do not match"
        )
    return LayoutBackingValidation(
        expert_layout_bytes=expert.total_bytes,
        resident_layout_bytes=resident.total_bytes,
        expert_config_sha256=expert.config_sha256,
        resident_config_sha256=resident.config_sha256,
        expert_quantization=expert.quantization,
        expert_group_size=expert.group_size,
    )


def _validate_model_config_sha256(
    *,
    model_dir: Path,
    backing: LayoutBackingValidation,
) -> str | None:
    recorded = [
        value
        for value in (
            backing.expert_config_sha256,
            backing.resident_config_sha256,
        )
        if value is not None
    ]
    if not recorded:
        return None
    if len(set(recorded)) != 1:
        raise PreparedManifestError(
            "prepared expert/resident layout config_sha256 values do not match"
        )
    current = config_sha256(model_dir)
    if current is None:
        raise PreparedManifestError(
            "prepared model config.json is missing; cannot verify config_sha256"
        )
    expected = recorded[0]
    if current != expected:
        raise PreparedManifestError(
            "prepared model config_sha256 mismatch: "
            f"current {current} != layout {expected}"
        )
    return current


def validate_layout_model_config_sha256(
    *,
    model_config_path: str | Path,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
) -> str | None:
    backing = validate_layout_backing_files(expert_layout_path, resident_layout_path)
    return _validate_model_config_sha256(
        model_dir=Path(model_config_path),
        backing=backing,
    )


def load_prepared_manifest(path_or_dir: str | Path) -> PreparedManifest:
    manifest_path = Path(path_or_dir)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreparedManifestError(f"failed to read prepared manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PreparedManifestError(f"failed to parse prepared manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreparedManifestError("prepared manifest must be a JSON object")
    if payload.get("version") != 1:
        raise PreparedManifestError(f"unsupported prepared manifest version {payload.get('version')}")

    base = manifest_path.parent
    max_context_tokens = _optional_int(payload, "max_context_tokens")
    expert_bytes = _optional_int(payload, "expert_bytes")
    resident_bytes = _optional_int(payload, "resident_bytes")
    decode_cache_bytes = _optional_int(payload, "decode_cache_bytes")
    recommended_max_live = _optional_nonnegative_int(
        payload,
        "recommended_max_live_working_set_bytes",
    )
    recommended_min_free = _optional_nonnegative_int(
        payload,
        "recommended_min_free_unified_memory_bytes",
    )
    expert_group_size = _optional_positive_int(payload, "expert_group_size")
    prepare_hardware_memory = _optional_nonnegative_int(
        payload,
        "prepare_hardware_unified_memory_bytes",
    )
    prepare_hardware_gpu_cores = _optional_nonnegative_int(
        payload,
        "prepare_hardware_gpu_cores",
    )
    prepare_hardware_apple_silicon_generation = _optional_positive_int(
        payload,
        "prepare_hardware_apple_silicon_generation",
    )
    prepare_effective_memory = _optional_nonnegative_int(
        payload,
        "prepare_effective_unified_memory_bytes",
    )
    prepare_system_reserve = _optional_nonnegative_int(
        payload,
        "prepare_system_reserve_bytes",
    )
    prepare_requested_context = _optional_positive_int(
        payload,
        "prepare_requested_max_context_tokens",
    )
    prepare_resolved_context = _optional_positive_int(
        payload,
        "prepare_resolved_max_context_tokens",
    )
    prepare_cache_budget = _optional_nonnegative_int(
        payload,
        "prepare_decode_cache_budget_bytes",
    )
    prepare_safe_context = _optional_nonnegative_int(
        payload,
        "prepare_decode_cache_safe_context_tokens",
    )
    prepare_effective_max_cache = _optional_nonnegative_int(
        payload,
        "prepare_effective_max_cache_bytes",
    )
    prepare_model_max_position = _optional_positive_int(
        payload,
        "prepare_model_max_position_embeddings",
    )
    prepare_cache_alignment = _optional_positive_int(
        payload,
        "prepare_cache_alignment",
    )
    prepare_flags_applied = _optional_bool(payload, "prepare_flags_applied")
    prepare_flags_source = _optional_str(payload, "prepare_flags_source")
    prepare_flags_path = _optional_str(payload, "prepare_flags_path")
    prepare_flags_sha256 = _optional_sha256_hex(payload, "prepare_flags_sha256")
    prepare_combined_output_required = _optional_nonnegative_int(
        payload,
        "prepare_combined_output_required_bytes",
    )
    prepare_combined_output_available = _optional_nonnegative_int(
        payload,
        "prepare_combined_output_available_bytes",
    )
    prepare_combined_output_margin = _optional_nonnegative_int(
        payload,
        "prepare_combined_output_disk_margin_bytes",
    )
    prepare_live_estimated = _optional_nonnegative_int(
        payload,
        "prepare_live_memory_estimated_live_working_set_bytes",
    )
    prepare_live_min_available = _optional_nonnegative_int(
        payload,
        "prepare_live_memory_min_available_memory_bytes",
    )
    prepare_live_required_available = _optional_nonnegative_int(
        payload,
        "prepare_live_memory_required_available_memory_bytes",
    )
    prepare_live_system_available = _optional_nonnegative_int(
        payload,
        "prepare_live_memory_system_available_memory_bytes",
    )
    prepare_live_system_total = _optional_nonnegative_int(
        payload,
        "prepare_live_memory_system_total_bytes",
    )
    prepare_live_system_source = _optional_str(
        payload,
        "prepare_live_memory_system_source",
    )
    prepare_pack_chunk_size = _optional_positive_int(
        payload,
        "prepare_expert_pack_chunk_size_bytes",
    )
    prepare_pack_estimated_peak = _optional_positive_int(
        payload,
        "prepare_expert_pack_estimated_peak_heap_bytes",
    )
    prepare_pack_max_heap = _optional_positive_int(
        payload,
        "prepare_expert_pack_max_heap_bytes",
    )
    prepare_raw_extra_heap = _optional_nonnegative_int(
        payload,
        "prepare_raw_quantization_extra_heap_bytes",
    )
    prepare_raw_max_source_block = _optional_nonnegative_int(
        payload,
        "prepare_raw_quantization_max_source_block_bytes",
    )
    prepare_raw_max_output_block = _optional_nonnegative_int(
        payload,
        "prepare_raw_quantization_max_output_block_bytes",
    )
    prepare_raw_max_rows_per_block = _optional_nonnegative_int(
        payload,
        "prepare_raw_quantization_max_rows_per_block",
    )
    resident_alias_source_count = _optional_nonnegative_int(
        payload,
        "prepare_resident_component_alias_source_tensor_count",
    )
    resident_alias_renamed_count = _optional_nonnegative_int(
        payload,
        "prepare_resident_component_alias_renamed_tensor_count",
    )
    resident_alias_bytes = _optional_nonnegative_int(
        payload,
        "prepare_resident_component_alias_bytes",
    )
    resident_fused_source_count = _optional_nonnegative_int(
        payload,
        "prepare_resident_fused_gate_up_source_tensor_count",
    )
    resident_fused_expanded_count = _optional_nonnegative_int(
        payload,
        "prepare_resident_fused_gate_up_expanded_tensor_count",
    )
    resident_fused_expanded_bytes = _optional_nonnegative_int(
        payload,
        "prepare_resident_fused_gate_up_expanded_bytes",
    )
    public_shape_required = _optional_bool(
        payload,
        "prepare_public_glm_5_2_shape_required",
    )
    public_shape_matches = _optional_bool(
        payload,
        "prepare_public_glm_5_2_shape_matches",
    )
    public_shape_mismatched_fields = _optional_str_tuple(
        payload,
        "prepare_public_glm_5_2_shape_mismatched_fields",
    )
    if prepare_flags_applied is False and any(
        value is not None
        for value in (
            prepare_flags_source,
            prepare_flags_path,
            prepare_flags_sha256,
        )
    ):
        raise PreparedManifestError(
            "prepare_flags_applied=false conflicts with prepare flags metadata"
        )
    if prepare_flags_applied is True and prepare_flags_source is None:
        raise PreparedManifestError(
            "prepare_flags_applied=true requires prepare_flags_source"
        )
    combined_disk_values = (
        prepare_combined_output_required,
        prepare_combined_output_available,
        prepare_combined_output_margin,
    )
    if any(value is not None for value in combined_disk_values):
        if any(value is None for value in combined_disk_values):
            raise PreparedManifestError(
                "prepare combined disk budget metadata requires required, "
                "available, and margin bytes"
            )
        assert prepare_combined_output_required is not None
        assert prepare_combined_output_available is not None
        assert prepare_combined_output_margin is not None
        if (
            prepare_combined_output_available
            < prepare_combined_output_required + prepare_combined_output_margin
        ):
            raise PreparedManifestError(
                "prepare combined disk budget records insufficient available bytes"
            )
    live_memory_values = (
        prepare_live_estimated,
        prepare_live_min_available,
        prepare_live_required_available,
        prepare_live_system_available,
        prepare_live_system_total,
        prepare_live_system_source,
    )
    if any(value is not None for value in live_memory_values):
        if (
            prepare_live_estimated is None
            or prepare_live_min_available is None
            or prepare_live_required_available is None
        ):
            raise PreparedManifestError(
                "prepare live-memory metadata requires estimated, minimum, "
                "and required available bytes"
            )
        expected_required = prepare_live_estimated + prepare_live_min_available
        if prepare_live_required_available != expected_required:
            raise PreparedManifestError(
                "prepare_live_memory_required_available_memory_bytes does not "
                "match estimated live working set plus minimum available memory"
            )
        if (
            prepare_live_system_available is not None
            and prepare_live_system_total is not None
            and prepare_live_system_available > prepare_live_system_total
        ):
            raise PreparedManifestError(
                "prepare live-memory system available bytes exceed total bytes"
            )
    if (
        prepare_pack_estimated_peak is not None
        and prepare_pack_max_heap is not None
        and prepare_pack_estimated_peak > prepare_pack_max_heap
    ):
        raise PreparedManifestError(
            "manifest prepare_expert_pack_estimated_peak_heap_bytes exceeds "
            "prepare_expert_pack_max_heap_bytes"
        )
    if prepare_raw_max_rows_per_block == 0 and any(
        value not in (None, 0)
        for value in (
            prepare_raw_extra_heap,
            prepare_raw_max_source_block,
            prepare_raw_max_output_block,
        )
    ):
        raise PreparedManifestError(
            "manifest prepare_raw_quantization_max_rows_per_block=0 conflicts "
            "with non-zero raw quantization block bytes"
        )
    _validate_prepare_pack_heap_envelope(
        estimated_peak_bytes=prepare_pack_estimated_peak,
        chunk_size_bytes=prepare_pack_chunk_size,
        raw_extra_heap_bytes=prepare_raw_extra_heap,
        raw_max_source_block_bytes=prepare_raw_max_source_block,
        raw_max_output_block_bytes=prepare_raw_max_output_block,
    )
    resident_fused_values = (
        resident_fused_source_count,
        resident_fused_expanded_count,
        resident_fused_expanded_bytes,
    )
    resident_alias_values = (
        resident_alias_source_count,
        resident_alias_renamed_count,
        resident_alias_bytes,
    )
    if any(value not in (None, 0) for value in resident_alias_values):
        if resident_alias_source_count in (None, 0):
            raise PreparedManifestError(
                "manifest resident component alias bytes require "
                "prepare_resident_component_alias_source_tensor_count"
            )
        if resident_alias_renamed_count != resident_alias_source_count:
            raise PreparedManifestError(
                "manifest prepare_resident_component_alias_renamed_tensor_count "
                "must equal source tensor count"
            )
        if resident_alias_bytes in (None, 0):
            raise PreparedManifestError(
                "manifest resident component alias rewrite requires "
                "prepare_resident_component_alias_bytes"
            )
    if any(value not in (None, 0) for value in resident_fused_values):
        if resident_fused_source_count in (None, 0):
            raise PreparedManifestError(
                "manifest resident fused gate/up expansion bytes require "
                "prepare_resident_fused_gate_up_source_tensor_count"
            )
        if resident_fused_expanded_count != resident_fused_source_count * 2:
            raise PreparedManifestError(
                "manifest prepare_resident_fused_gate_up_expanded_tensor_count "
                "must equal source tensor count * 2"
            )
        if resident_fused_expanded_bytes in (None, 0):
            raise PreparedManifestError(
                "manifest resident fused gate/up expansion requires "
                "prepare_resident_fused_gate_up_expanded_bytes"
            )
    if public_shape_required is True and public_shape_matches is not True:
        raise PreparedManifestError(
            "prepare_public_glm_5_2_shape_required=true requires "
            "prepare_public_glm_5_2_shape_matches=true"
        )
    if public_shape_matches is True and public_shape_mismatched_fields:
        raise PreparedManifestError(
            "prepare_public_glm_5_2_shape_matches=true conflicts with "
            "prepare_public_glm_5_2_shape_mismatched_fields"
        )
    manifest_model_config_sha256 = _optional_str(payload, "model_config_sha256")
    manifest = PreparedManifest(
        manifest_path=manifest_path,
        model_dir=_path_from_manifest(base, payload.get("model_dir"), "model_dir"),
        experts_layout=_path_from_manifest(base, payload.get("experts_layout"), "experts_layout"),
        resident_layout=_path_from_manifest(base, payload.get("resident_layout"), "resident_layout"),
        decode_cache_layout=_path_from_manifest(
            base,
            payload.get("decode_cache_layout"),
            "decode_cache_layout",
        ),
        decode_cache_file=_path_from_manifest(
            base,
            payload.get("decode_cache_file"),
            "decode_cache_file",
        ),
        max_context_tokens=max_context_tokens,
        recommended_max_live_working_set_bytes=recommended_max_live,
        recommended_min_free_unified_memory_bytes=recommended_min_free,
        expert_quantization=_optional_str(payload, "expert_quantization"),
        expert_group_size=expert_group_size,
        prepare_hardware_chip_name=_optional_str(
            payload,
            "prepare_hardware_chip_name",
        ),
        prepare_hardware_unified_memory_bytes=prepare_hardware_memory,
        prepare_hardware_gpu_cores=prepare_hardware_gpu_cores,
        prepare_hardware_apple_silicon_generation=(
            prepare_hardware_apple_silicon_generation
        ),
        prepare_hardware_apple_silicon_tier=_optional_str(
            payload,
            "prepare_hardware_apple_silicon_tier",
        ),
        prepare_effective_unified_memory_bytes=prepare_effective_memory,
        prepare_effective_unified_memory_source=_optional_str(
            payload,
            "prepare_effective_unified_memory_source",
        ),
        prepare_system_reserve_bytes=prepare_system_reserve,
        prepare_cold_read_gib_per_second=_optional_positive_float(
            payload,
            "prepare_cold_read_gib_per_second",
        ),
        prepare_cold_read_source=_optional_str(
            payload,
            "prepare_cold_read_source",
        ),
        prepare_cold_read_benchmark_path=_optional_str(
            payload,
            "prepare_cold_read_benchmark_path",
        ),
        prepare_cold_read_benchmark_requested_bytes=_optional_nonnegative_int(
            payload,
            "prepare_cold_read_benchmark_requested_bytes",
        ),
        prepare_cold_read_benchmark_measured_bytes=_optional_nonnegative_int(
            payload,
            "prepare_cold_read_benchmark_measured_bytes",
        ),
        prepare_cold_read_benchmark_elapsed_seconds=_optional_positive_float(
            payload,
            "prepare_cold_read_benchmark_elapsed_seconds",
        ),
        prepare_auto_context_from_budget=_optional_bool(
            payload,
            "prepare_auto_context_from_budget",
        ),
        prepare_requested_max_context_tokens=prepare_requested_context,
        prepare_resolved_max_context_tokens=prepare_resolved_context,
        prepare_decode_cache_budget_bytes=prepare_cache_budget,
        prepare_decode_cache_safe_context_tokens=prepare_safe_context,
        prepare_effective_max_cache_bytes=prepare_effective_max_cache,
        prepare_model_max_position_embeddings=prepare_model_max_position,
        prepare_cache_dtype=_optional_str(payload, "prepare_cache_dtype"),
        prepare_cache_alignment=prepare_cache_alignment,
        prepare_flags_applied=prepare_flags_applied,
        prepare_flags_source=prepare_flags_source,
        prepare_flags_path=prepare_flags_path,
        prepare_flags_sha256=prepare_flags_sha256,
        prepare_combined_output_required_bytes=(
            prepare_combined_output_required
        ),
        prepare_combined_output_available_bytes=(
            prepare_combined_output_available
        ),
        prepare_combined_output_disk_margin_bytes=(
            prepare_combined_output_margin
        ),
        prepare_live_memory_estimated_live_working_set_bytes=(
            prepare_live_estimated
        ),
        prepare_live_memory_min_available_memory_bytes=(
            prepare_live_min_available
        ),
        prepare_live_memory_required_available_memory_bytes=(
            prepare_live_required_available
        ),
        prepare_live_memory_system_available_memory_bytes=(
            prepare_live_system_available
        ),
        prepare_live_memory_system_total_bytes=prepare_live_system_total,
        prepare_live_memory_system_source=prepare_live_system_source,
        prepare_expert_pack_chunk_size_bytes=prepare_pack_chunk_size,
        prepare_expert_pack_estimated_peak_heap_bytes=prepare_pack_estimated_peak,
        prepare_expert_pack_max_heap_bytes=prepare_pack_max_heap,
        prepare_raw_quantization_extra_heap_bytes=prepare_raw_extra_heap,
        prepare_raw_quantization_max_source_block_bytes=(
            prepare_raw_max_source_block
        ),
        prepare_raw_quantization_max_output_block_bytes=(
            prepare_raw_max_output_block
        ),
        prepare_raw_quantization_max_rows_per_block=(
            prepare_raw_max_rows_per_block
        ),
        prepare_resident_component_alias_source_tensor_count=(
            resident_alias_source_count
        ),
        prepare_resident_component_alias_renamed_tensor_count=(
            resident_alias_renamed_count
        ),
        prepare_resident_component_alias_bytes=resident_alias_bytes,
        prepare_resident_fused_gate_up_source_tensor_count=(
            resident_fused_source_count
        ),
        prepare_resident_fused_gate_up_expanded_tensor_count=(
            resident_fused_expanded_count
        ),
        prepare_resident_fused_gate_up_expanded_bytes=(
            resident_fused_expanded_bytes
        ),
        prepare_public_glm_5_2_shape_required=public_shape_required,
        prepare_public_glm_5_2_shape_matches=public_shape_matches,
        prepare_public_glm_5_2_shape_mismatched_fields=public_shape_mismatched_fields,
        model_config_sha256=manifest_model_config_sha256,
    )
    missing = [
        path
        for path in (
            manifest.model_dir,
            manifest.experts_layout,
            manifest.resident_layout,
            manifest.decode_cache_layout,
            manifest.decode_cache_file,
        )
        if not path.exists()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        more = "" if len(missing) <= 5 else f", +{len(missing) - 5} more"
        raise PreparedManifestError(f"prepared manifest references missing paths: {preview}{more}")
    try:
        cache_layout = load_decode_cache_layout(manifest.decode_cache_layout)
    except DecodeCacheError as exc:
        raise PreparedManifestError(str(exc)) from exc
    if (
        manifest.max_context_tokens is not None
        and manifest.max_context_tokens != cache_layout.max_context_tokens
    ):
        raise PreparedManifestError(
            "manifest max_context_tokens "
            f"{manifest.max_context_tokens} does not match decode cache layout "
            f"{cache_layout.max_context_tokens}"
        )
    if (
        manifest.prepare_resolved_max_context_tokens is not None
        and manifest.prepare_resolved_max_context_tokens
        != cache_layout.max_context_tokens
    ):
        raise PreparedManifestError(
            "manifest prepare_resolved_max_context_tokens "
            f"{manifest.prepare_resolved_max_context_tokens} does not match "
            f"decode cache layout max_context_tokens {cache_layout.max_context_tokens}"
        )
    if (
        manifest.prepare_cache_dtype is not None
        and manifest.prepare_cache_dtype != cache_layout.dtype
    ):
        raise PreparedManifestError(
            "manifest prepare_cache_dtype "
            f"{manifest.prepare_cache_dtype} does not match decode cache layout "
            f"dtype {cache_layout.dtype}"
        )
    if (
        manifest.prepare_cache_alignment is not None
        and manifest.prepare_cache_alignment != cache_layout.alignment
    ):
        raise PreparedManifestError(
            "manifest prepare_cache_alignment "
            f"{manifest.prepare_cache_alignment} does not match decode cache "
            f"layout alignment {cache_layout.alignment}"
        )
    if decode_cache_bytes is not None and decode_cache_bytes != cache_layout.total_bytes:
        raise PreparedManifestError(
            "manifest decode_cache_bytes "
            f"{decode_cache_bytes} does not match decode cache layout total_bytes "
            f"{cache_layout.total_bytes}"
        )
    if (
        manifest.prepare_decode_cache_budget_bytes is not None
        and cache_layout.total_bytes > manifest.prepare_decode_cache_budget_bytes
    ):
        raise PreparedManifestError(
            "manifest prepare_decode_cache_budget_bytes "
            f"{manifest.prepare_decode_cache_budget_bytes} is smaller than decode "
            f"cache layout total_bytes {cache_layout.total_bytes}"
        )
    if (
        manifest.prepare_effective_max_cache_bytes is not None
        and cache_layout.total_bytes > manifest.prepare_effective_max_cache_bytes
    ):
        raise PreparedManifestError(
            "manifest prepare_effective_max_cache_bytes "
            f"{manifest.prepare_effective_max_cache_bytes} is smaller than decode "
            f"cache layout total_bytes {cache_layout.total_bytes}"
        )
    if (
        manifest.prepare_decode_cache_safe_context_tokens is not None
        and cache_layout.max_context_tokens
        > manifest.prepare_decode_cache_safe_context_tokens
    ):
        raise PreparedManifestError(
            "manifest prepare_decode_cache_safe_context_tokens "
            f"{manifest.prepare_decode_cache_safe_context_tokens} is smaller than "
            f"decode cache layout max_context_tokens {cache_layout.max_context_tokens}"
        )
    if manifest.prepare_auto_context_from_budget is True:
        missing_auto_context_fields = [
            name
            for name, value in (
                (
                    "prepare_resolved_max_context_tokens",
                    manifest.prepare_resolved_max_context_tokens,
                ),
                (
                    "prepare_decode_cache_budget_bytes",
                    manifest.prepare_decode_cache_budget_bytes,
                ),
                (
                    "prepare_decode_cache_safe_context_tokens",
                    manifest.prepare_decode_cache_safe_context_tokens,
                ),
            )
            if value is None
        ]
        if missing_auto_context_fields:
            preview = ", ".join(missing_auto_context_fields)
            raise PreparedManifestError(
                "manifest prepare_auto_context_from_budget requires " + preview
            )
    try:
        cache_file_bytes = manifest.decode_cache_file.stat().st_size
    except OSError as exc:
        raise PreparedManifestError(
            f"failed to stat decode cache file {manifest.decode_cache_file}: {exc}"
        ) from exc
    if cache_file_bytes < cache_layout.total_bytes:
        raise PreparedManifestError(
            "prepared decode cache file is smaller than layout total_bytes: "
            f"{cache_file_bytes} < {cache_layout.total_bytes}"
        )
    if cache_file_bytes > cache_layout.total_bytes:
        raise PreparedManifestError(
            "prepared decode cache file is larger than layout total_bytes: "
            f"{cache_file_bytes} > {cache_layout.total_bytes}"
        )
    backing = validate_layout_backing_files(
        manifest.experts_layout,
        manifest.resident_layout,
    )
    if manifest.expert_quantization is not None:
        if backing.expert_quantization is None:
            raise PreparedManifestError(
                "manifest records expert_quantization but expert layout "
                "quantization metadata is unavailable"
            )
        if manifest.expert_quantization != backing.expert_quantization:
            raise PreparedManifestError(
                "manifest expert_quantization does not match expert layout: "
                f"{manifest.expert_quantization} != {backing.expert_quantization}"
            )
    if manifest.expert_group_size is not None:
        if backing.expert_group_size is None:
            raise PreparedManifestError(
                "manifest records expert_group_size but expert layout group_size "
                "metadata is unavailable"
            )
        if manifest.expert_group_size != backing.expert_group_size:
            raise PreparedManifestError(
                "manifest expert_group_size does not match expert layout: "
                f"{manifest.expert_group_size} != {backing.expert_group_size}"
            )
    effective_expert_quantization = (
        manifest.expert_quantization or backing.expert_quantization
    )
    _validate_required_prepare_pack_heap_evidence(
        manifest=manifest,
        expert_quantization=effective_expert_quantization,
    )
    model_config_digest = _validate_model_config_sha256(
        model_dir=manifest.model_dir,
        backing=backing,
    )
    if manifest.model_config_sha256 is not None:
        if model_config_digest is None:
            raise PreparedManifestError(
                "manifest records model_config_sha256 but layout config_sha256 "
                "metadata is unavailable"
            )
        if manifest.model_config_sha256 != model_config_digest:
            raise PreparedManifestError(
                "manifest model_config_sha256 does not match validated layouts: "
                f"{manifest.model_config_sha256} != {model_config_digest}"
            )
    _validate_recorded_bytes(
        label="expert layout",
        expected=expert_bytes,
        actual=backing.expert_layout_bytes,
    )
    _validate_recorded_bytes(
        label="resident layout",
        expected=resident_bytes,
        actual=backing.resident_layout_bytes,
    )
    _validate_resident_alias_rewrite_bytes(
        manifest=manifest,
        resident_layout_bytes=backing.resident_layout_bytes,
    )
    return replace(
        manifest,
        expert_layout_bytes=backing.expert_layout_bytes,
        expert_layout_quantization=backing.expert_quantization,
        expert_layout_group_size=backing.expert_group_size,
        resident_layout_bytes=backing.resident_layout_bytes,
        decode_cache_layout_bytes=cache_layout.total_bytes,
        decode_cache_file_bytes=cache_file_bytes,
        model_config_sha256=model_config_digest,
    )
