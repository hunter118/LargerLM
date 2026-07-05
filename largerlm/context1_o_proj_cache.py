from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .safety import DiskBudget, disk_budget


class Context1OProjCacheError(RuntimeError):
    """Raised when context=1 collapsed attention-output cache work is unsafe."""


CACHE_SCHEMA = "largerlm.context1_o_proj_bv_cache.v1"
BUILD_REPORT_SCHEMA = "largerlm.context1_o_proj_bv_cache_build.v1"
PROGRESS_SCHEMA = "largerlm.context1_o_proj_bv_cache_progress.v1"

DEFAULT_CACHE_DIRNAME = "context1-o-proj-bv-cache"
DEFAULT_CACHE_BIN = "context1_o_proj_bv.bin"
DEFAULT_CACHE_LAYOUT = "layout.json"
DEFAULT_PROGRESS = "progress.json"
DEFAULT_METAL_BINARY = Path(__file__).resolve().parents[1] / "metal" / "glm_moe_infer"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Context1OProjCacheError(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Context1OProjCacheError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Context1OProjCacheError(f"{label} must be a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise Context1OProjCacheError(f"failed to write JSON {path}: {exc}") from exc


def _json_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _json_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _json_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _json_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    return None


def _progress_layer_result_summary(
    layout: "Context1OProjCacheLayout",
    *,
    completed_layers: set[int],
    layer_results: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    per_layer_fma = layout.hidden_dim * layout.attention_value_dim * layout.kv_lora_dim
    fma_total = per_layer_fma * len(layout.layers)
    measured_layers: set[int] = set()
    measured_fma_total = 0
    measured_source_bytes_read = 0
    measured_cache_bytes_written = 0
    elapsed_seconds = 0.0
    read_seconds = 0.0
    kernel_seconds = 0.0
    write_seconds = 0.0
    max_estimated_live_bytes = 0
    max_estimated_metal_builder_live_bytes = 0
    valid_layers = set(layout.layers)
    for item in layer_results:
        layer_id = _json_int(item.get("layer"))
        if layer_id is None or layer_id not in valid_layers:
            continue
        measured_layers.add(layer_id)
        measured_fma_total += _json_int(item.get("fma_count")) or per_layer_fma
        measured_source_bytes_read += _json_int(item.get("source_bytes_read")) or 0
        measured_cache_bytes_written += (
            _json_int(item.get("cache_bytes_written"))
            or (layout.hidden_dim * layout.kv_lora_dim * layout.dtype_bytes)
        )
        elapsed_seconds += float(_json_number(item.get("elapsed_seconds")) or 0.0)
        read_seconds += float(_json_number(item.get("read_seconds")) or 0.0)
        kernel_seconds += float(_json_number(item.get("kernel_seconds")) or 0.0)
        write_seconds += float(_json_number(item.get("write_seconds")) or 0.0)
        live = _json_int(item.get("estimated_live_working_set_bytes"))
        if live is not None:
            max_estimated_live_bytes = max(max_estimated_live_bytes, live)
        builder_live = _json_int(item.get("estimated_metal_builder_live_bytes"))
        if builder_live is not None:
            max_estimated_metal_builder_live_bytes = max(
                max_estimated_metal_builder_live_bytes,
                builder_live,
            )
    completed_fma_total = per_layer_fma * len(completed_layers)
    remaining_fma_total = max(fma_total - completed_fma_total, 0)
    fma_per_second = (
        measured_fma_total / elapsed_seconds
        if measured_fma_total > 0 and elapsed_seconds > 0.0
        else None
    )
    return {
        "measured_layer_count": len(measured_layers),
        "measured_layers": sorted(measured_layers),
        "measured_fma_total": measured_fma_total,
        "measured_source_bytes_read": measured_source_bytes_read,
        "measured_cache_bytes_written": measured_cache_bytes_written,
        "measured_elapsed_seconds": elapsed_seconds,
        "measured_read_seconds": read_seconds,
        "measured_kernel_seconds": kernel_seconds,
        "measured_write_seconds": write_seconds,
        "measured_fma_per_second": fma_per_second,
        "measured_gfma_per_second": (
            fma_per_second / 1.0e9 if fma_per_second is not None else None
        ),
        "completed_fma_total": completed_fma_total,
        "remaining_fma_total": remaining_fma_total,
        "completed_fma_fraction": (
            completed_fma_total / fma_total if fma_total > 0 else None
        ),
        "estimated_full_build_seconds_from_measured": (
            fma_total / fma_per_second
            if fma_per_second and fma_per_second > 0.0
            else None
        ),
        "estimated_remaining_build_seconds_from_measured": (
            remaining_fma_total / fma_per_second
            if fma_per_second and fma_per_second > 0.0
            else None
        ),
        "max_estimated_live_working_set_bytes": max_estimated_live_bytes,
        "max_estimated_metal_builder_live_bytes": max_estimated_metal_builder_live_bytes,
    }


def _format_context1_o_proj_build_gfma(fma: int) -> str:
    return f"{fma / 1.0e9:.9g}"


def _context1_o_proj_resume_suggestion(
    layout: "Context1OProjCacheLayout",
    *,
    progress_exists: bool,
    backend: str | None,
    missing_layers: tuple[int, ...],
) -> dict[str, Any] | None:
    if not missing_layers:
        return None
    next_layers = (missing_layers[0],)
    per_layer_fma = layout.hidden_dim * layout.attention_value_dim * layout.kv_lora_dim
    min_max_build_fma = per_layer_fma * len(next_layers)
    base: dict[str, Any] = {
        "available": False,
        "source": "context1_o_proj_cache_progress",
        "selection": "next_incomplete_layers",
        "recommended_next_layers": len(next_layers),
        "next_layer": next_layers[0],
        "next_layers": list(next_layers),
        "remaining_layer_count": len(missing_layers),
        "min_max_build_fma": min_max_build_fma,
        "min_max_build_gfma": min_max_build_fma / 1.0e9,
        "min_max_build_gfma_arg": _format_context1_o_proj_build_gfma(
            min_max_build_fma
        ),
        "requires_explicit_execute": True,
    }
    if not progress_exists:
        base["reason"] = "progress file is missing; no resumable completed-layer state"
        return base
    if not layout.source_prepared_manifest:
        base["reason"] = "cache layout missing source_prepared_manifest"
        return base

    resolved_backend = backend if backend in {"reference", "metal"} else "metal"
    prepared_dir = str(Path(layout.source_prepared_manifest).parent)
    output_dir = str(layout.layout_path.parent)
    dry_run_argv = (
        "context1-o-proj-cache",
        prepared_dir,
        "--output-dir",
        output_dir,
        "--backend",
        resolved_backend,
        "--build-next-layers",
        str(len(next_layers)),
        "--max-build-gfma",
        base["min_max_build_gfma_arg"],
    )
    execute_argv = dry_run_argv + ("--execute",)
    base.update(
        {
            "available": True,
            "backend": resolved_backend,
            "prepared_dir": prepared_dir,
            "output_dir": output_dir,
            "dry_run_argv": list(dry_run_argv),
            "execute_argv": list(execute_argv),
            "execute_argv_requires_explicit_review": True,
        }
    )
    return base


def _require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise Context1OProjCacheError(f"{name} must be an integer")
    return int(value)


def _require_positive_int(value: Any, name: str) -> int:
    parsed = _require_int(value, name)
    if parsed <= 0:
        raise Context1OProjCacheError(f"{name} must be positive")
    return parsed


def _require_nonnegative_int(value: Any, name: str) -> int:
    parsed = _require_int(value, name)
    if parsed < 0:
        raise Context1OProjCacheError(f"{name} must be non-negative")
    return parsed


def _shape(tensor: dict[str, Any], rank: int, label: str) -> tuple[int, ...]:
    raw = tensor.get("shape")
    if not isinstance(raw, list) or len(raw) != rank:
        raise Context1OProjCacheError(f"{label} must have rank {rank}")
    dims: list[int] = []
    for index, dim in enumerate(raw):
        if type(dim) is not int or dim <= 0:
            raise Context1OProjCacheError(
                f"{label} shape[{index}] must be a positive integer"
            )
        dims.append(int(dim))
    return tuple(dims)


def _tensor_size(tensor: dict[str, Any], label: str) -> int:
    return _require_nonnegative_int(tensor.get("size"), f"{label} size")


def _tensor_offset(tensor: dict[str, Any], label: str) -> int:
    return _require_nonnegative_int(tensor.get("offset"), f"{label} offset")


def _tensors_by_name(layout: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise Context1OProjCacheError("resident layout missing tensors array")
    by_name: dict[str, dict[str, Any]] = {}
    for item in tensors:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            by_name[name] = item
    return by_name


def _companion(
    tensors: dict[str, dict[str, Any]],
    weight_name: str,
) -> dict[str, Any]:
    if not weight_name.endswith(".weight"):
        raise Context1OProjCacheError(f"{weight_name} is not a .weight tensor")
    scale_name = weight_name[: -len(".weight")] + ".scales"
    try:
        return tensors[scale_name]
    except KeyError as exc:
        raise Context1OProjCacheError(f"missing companion scale tensor {scale_name}") from exc


def _mxfp4_matrix_dims(
    tensors: dict[str, dict[str, Any]],
    weight: dict[str, Any],
    label: str,
) -> tuple[int, int, int]:
    if weight.get("dtype") != "U32":
        raise Context1OProjCacheError(f"{label} must be U32")
    scales = _companion(tensors, str(weight["name"]))
    if scales.get("dtype") != "U8":
        raise Context1OProjCacheError(f"{label}.scales must be U8")
    out_dim, packed_cols = _shape(weight, 2, label)
    scale_rows, groups = _shape(scales, 2, f"{label}.scales")
    if scale_rows != out_dim:
        raise Context1OProjCacheError(f"{label} scales row count mismatch")
    in_dim = packed_cols * 8
    if groups <= 0 or in_dim % groups != 0:
        raise Context1OProjCacheError(f"{label} scale groups do not divide input")
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise Context1OProjCacheError(f"{label} group size must be a multiple of 8")
    expected_weight = out_dim * packed_cols * 4
    expected_scales = out_dim * groups
    if _tensor_size(weight, label) != expected_weight:
        raise Context1OProjCacheError(f"{label} size does not match packed shape")
    if _tensor_size(scales, f"{label}.scales") != expected_scales:
        raise Context1OProjCacheError(f"{label}.scales size does not match shape")
    return out_dim, in_dim, group_size


def _mxfp4_tensor3d_dims(
    tensors: dict[str, dict[str, Any]],
    weight: dict[str, Any],
    label: str,
) -> tuple[int, int, int, int]:
    if weight.get("dtype") != "U32":
        raise Context1OProjCacheError(f"{label} must be U32")
    scales = _companion(tensors, str(weight["name"]))
    if scales.get("dtype") != "U8":
        raise Context1OProjCacheError(f"{label}.scales must be U8")
    dim0, dim1, packed_dim2 = _shape(weight, 3, label)
    scale0, scale1, groups = _shape(scales, 3, f"{label}.scales")
    if (scale0, scale1) != (dim0, dim1):
        raise Context1OProjCacheError(f"{label} scales prefix dims mismatch")
    dim2 = packed_dim2 * 8
    if groups <= 0 or dim2 % groups != 0:
        raise Context1OProjCacheError(f"{label} scale groups do not divide dim2")
    group_size = dim2 // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise Context1OProjCacheError(f"{label} group size must be a multiple of 8")
    expected_weight = dim0 * dim1 * packed_dim2 * 4
    expected_scales = dim0 * dim1 * groups
    if _tensor_size(weight, label) != expected_weight:
        raise Context1OProjCacheError(f"{label} size does not match packed shape")
    if _tensor_size(scales, f"{label}.scales") != expected_scales:
        raise Context1OProjCacheError(f"{label}.scales size does not match shape")
    return dim0, dim1, dim2, group_size


def _mxfp4_e2m1_to_f32(value: int) -> float:
    mag = value & 0x7
    if mag == 0:
        return 0.0
    lookup = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    sign = -1.0 if value & 0x8 else 1.0
    return sign * lookup[mag]


def _mxfp4_e8m0_to_f32(value: int) -> float:
    return struct.unpack("<f", ((value & 0xFF) << 23).to_bytes(4, "little"))[0]


def _f32_to_bf16_bytes(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", float(value)), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def _write_scalar(out: bytearray, offset: int, value: float, dtype: str) -> None:
    if dtype == "F32":
        out[offset : offset + 4] = struct.pack("<f", float(value))
    elif dtype == "BF16":
        out[offset : offset + 2] = _f32_to_bf16_bytes(float(value))
    else:
        raise Context1OProjCacheError(f"unsupported cache dtype {dtype}")


def _read_exact(fd: int, size: int, offset: int, label: str) -> bytes:
    raw = os.pread(fd, size, offset)
    if len(raw) != size:
        raise Context1OProjCacheError(
            f"short read for {label}: got {len(raw)}, expected {size}"
        )
    return raw


def _check_tensor_span(file_bytes: int, tensor: dict[str, Any], label: str) -> None:
    offset = _tensor_offset(tensor, label)
    size = _tensor_size(tensor, label)
    if offset + size > file_bytes:
        raise Context1OProjCacheError(
            f"{label} extends beyond resident file: {offset + size} > {file_bytes}"
        )


@dataclass(frozen=True)
class CollapsibleLayer:
    layer: int
    hidden_dim: int
    attention_value_dim: int
    num_heads: int
    v_head_dim: int
    qk_nope_dim: int
    kv_lora_dim: int
    o_proj_group_size: int
    unembed_out_group_size: int
    o_proj_weight: dict[str, Any]
    o_proj_scales: dict[str, Any]
    unembed_out_weight: dict[str, Any]
    unembed_out_scales: dict[str, Any]
    embed_q_weight: dict[str, Any]
    embed_q_scales: dict[str, Any]


@dataclass(frozen=True)
class Context1OProjCachePlan:
    prepared_dir: Path
    prepared_manifest: Path
    resident_layout_path: Path
    resident_weight_path: Path
    output_dir: Path
    cache_layout_path: Path
    cache_file_path: Path
    progress_path: Path
    dtype: str
    dtype_bytes: int
    layers: tuple[CollapsibleLayer, ...]
    hidden_dim: int
    attention_value_dim: int
    num_heads: int
    v_head_dim: int
    qk_nope_dim: int
    kv_lora_dim: int
    total_bytes: int
    fma_total: int
    current_o_proj_storage_per_token: int
    config_sha256: str | None

    def to_report(self) -> dict[str, Any]:
        layer_fma = [_layer_fma(layer) for layer in self.layers]
        return {
            "schema": BUILD_REPORT_SCHEMA,
            "prepared_dir": str(self.prepared_dir),
            "prepared_manifest": str(self.prepared_manifest),
            "resident_layout": str(self.resident_layout_path),
            "resident_weight": str(self.resident_weight_path),
            "output_dir": str(self.output_dir),
            "cache_layout": str(self.cache_layout_path),
            "cache_file": str(self.cache_file_path),
            "progress": str(self.progress_path),
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "layers": [layer.layer for layer in self.layers],
            "layer_count": len(self.layers),
            "dims": {
                "hidden_dim": self.hidden_dim,
                "attention_value_dim": self.attention_value_dim,
                "num_heads": self.num_heads,
                "v_head_dim": self.v_head_dim,
                "qk_nope_dim": self.qk_nope_dim,
                "kv_lora_dim": self.kv_lora_dim,
            },
            "bytes": {
                "total_bytes": self.total_bytes,
                "per_layer_bytes": self.hidden_dim * self.kv_lora_dim * self.dtype_bytes,
                "current_o_proj_storage_per_token": self.current_o_proj_storage_per_token,
            },
            "build_work": {
                "fma_total": self.fma_total,
                "two_op_flops_total": self.fma_total * 2,
                "max_layer_fma": max(layer_fma, default=0),
                "per_layer_fma": layer_fma,
            },
            "production_default": False,
            "context_limit": "decode/context_length_1_only",
            "format": CACHE_SCHEMA,
        }


@dataclass(frozen=True)
class Context1OProjCacheBuildResult:
    plan: Context1OProjCachePlan
    executed: bool
    completed_layers: tuple[int, ...]
    skipped_layers: tuple[int, ...]
    elapsed_seconds: float
    backend: str
    requested_build_layers: tuple[int, ...] | None = None
    requested_build_next_layers: int | None = None
    disk_budget: DiskBudget | None = None
    max_metal_builder_live_bytes: int | None = None
    layer_results: tuple[dict[str, Any], ...] = ()

    def _selected_build_layers(self) -> tuple[CollapsibleLayer, ...]:
        if self.requested_build_layers is None:
            return self.plan.layers
        requested = set(self.requested_build_layers)
        return tuple(layer for layer in self.plan.layers if layer.layer in requested)

    def selected_build_report(self) -> dict[str, Any]:
        selected = self._selected_build_layers()
        per_layer_bytes = _layer_cache_bytes(self.plan)
        per_layer = [
            {
                "layer": layer.layer,
                "fma": _layer_fma(layer),
                "cache_bytes": per_layer_bytes,
                "source_bytes": _layer_source_bytes(layer),
                "estimated_metal_builder_live_bytes": _metal_builder_live_bytes(
                    self.plan,
                    layer,
                ),
            }
            for layer in selected
        ]
        fma_total = sum(item["fma"] for item in per_layer)
        source_bytes = sum(item["source_bytes"] for item in per_layer)
        max_builder_live_bytes = max(
            (item["estimated_metal_builder_live_bytes"] for item in per_layer),
            default=0,
        )
        return {
            "full_plan": self.requested_build_layers is None,
            "layers": [item["layer"] for item in per_layer],
            "layer_count": len(per_layer),
            "cache_bytes": per_layer_bytes * len(per_layer),
            "source_bytes": source_bytes,
            "fma_total": fma_total,
            "two_op_flops_total": fma_total * 2,
            "max_layer_fma": max((item["fma"] for item in per_layer), default=0),
            "max_estimated_metal_builder_live_bytes": max_builder_live_bytes,
            "min_max_build_fma": fma_total,
            "per_layer": per_layer,
        }

    def layer_result_summary(self) -> dict[str, Any]:
        plan_layers = {layer.layer: layer for layer in self.plan.layers}
        measured_layers: set[int] = set()
        measured_fma_total = 0
        measured_source_bytes_read = 0
        measured_cache_bytes_written = 0
        elapsed_seconds = 0.0
        read_seconds = 0.0
        kernel_seconds = 0.0
        write_seconds = 0.0
        max_estimated_live_bytes = 0
        max_estimated_metal_builder_live_bytes = 0
        for item in self.layer_results:
            layer_id = _json_int(item.get("layer"))
            if layer_id is None or layer_id not in plan_layers:
                continue
            layer = plan_layers[layer_id]
            measured_layers.add(layer_id)
            measured_fma_total += _json_int(item.get("fma_count")) or _layer_fma(layer)
            measured_source_bytes_read += (
                _json_int(item.get("source_bytes_read")) or _layer_source_bytes(layer)
            )
            measured_cache_bytes_written += (
                _json_int(item.get("cache_bytes_written")) or _layer_cache_bytes(self.plan)
            )
            elapsed_seconds += float(_json_number(item.get("elapsed_seconds")) or 0.0)
            read_seconds += float(_json_number(item.get("read_seconds")) or 0.0)
            kernel_seconds += float(_json_number(item.get("kernel_seconds")) or 0.0)
            write_seconds += float(_json_number(item.get("write_seconds")) or 0.0)
            live = _json_int(item.get("estimated_live_working_set_bytes"))
            if live is not None:
                max_estimated_live_bytes = max(max_estimated_live_bytes, live)
            max_estimated_metal_builder_live_bytes = max(
                max_estimated_metal_builder_live_bytes,
                _json_int(item.get("estimated_metal_builder_live_bytes"))
                or _metal_builder_live_bytes(self.plan, layer),
            )
        completed_layer_set = set(self.completed_layers)
        completed_fma_total = sum(
            _layer_fma(layer)
            for layer in self.plan.layers
            if layer.layer in completed_layer_set
        )
        remaining_fma_total = max(self.plan.fma_total - completed_fma_total, 0)
        fma_per_second = (
            measured_fma_total / elapsed_seconds
            if measured_fma_total > 0 and elapsed_seconds > 0.0
            else None
        )
        estimated_full_seconds = (
            self.plan.fma_total / fma_per_second
            if fma_per_second and fma_per_second > 0.0
            else None
        )
        estimated_remaining_seconds = (
            remaining_fma_total / fma_per_second
            if fma_per_second and fma_per_second > 0.0
            else None
        )
        return {
            "measured_layer_count": len(measured_layers),
            "measured_layers": sorted(measured_layers),
            "measured_fma_total": measured_fma_total,
            "measured_source_bytes_read": measured_source_bytes_read,
            "measured_cache_bytes_written": measured_cache_bytes_written,
            "measured_elapsed_seconds": elapsed_seconds,
            "measured_read_seconds": read_seconds,
            "measured_kernel_seconds": kernel_seconds,
            "measured_write_seconds": write_seconds,
            "measured_fma_per_second": fma_per_second,
            "measured_gfma_per_second": (
                fma_per_second / 1.0e9 if fma_per_second is not None else None
            ),
            "completed_fma_total": completed_fma_total,
            "remaining_fma_total": remaining_fma_total,
            "completed_fma_fraction": (
                completed_fma_total / self.plan.fma_total
                if self.plan.fma_total > 0
                else None
            ),
            "estimated_full_build_seconds_from_measured": estimated_full_seconds,
            "estimated_remaining_build_seconds_from_measured": (
                estimated_remaining_seconds
            ),
            "max_estimated_live_working_set_bytes": max_estimated_live_bytes,
            "max_estimated_metal_builder_live_bytes": (
                max_estimated_metal_builder_live_bytes
            ),
        }

    def to_json(self) -> dict[str, Any]:
        payload = self.plan.to_report()
        payload.update(
            {
                "executed": self.executed,
                "completed_layers": list(self.completed_layers),
                "skipped_layers": list(self.skipped_layers),
                "elapsed_seconds": self.elapsed_seconds,
                "backend": self.backend,
                "requested_build_layers": (
                    list(self.requested_build_layers)
                    if self.requested_build_layers is not None
                    else None
                ),
                "requested_build_next_layers": self.requested_build_next_layers,
                "selected_build": self.selected_build_report(),
                "max_metal_builder_live_bytes": self.max_metal_builder_live_bytes,
                "layer_results": [dict(item) for item in self.layer_results],
                "layer_result_summary": self.layer_result_summary(),
                "disk_budget": (
                    None
                    if self.disk_budget is None
                    else {
                        "output_dir": str(self.disk_budget.output_dir),
                        "required_bytes": self.disk_budget.required_bytes,
                        "available_bytes": self.disk_budget.available_bytes,
                        "safety_margin_bytes": self.disk_budget.safety_margin_bytes,
                        "ok": self.disk_budget.ok,
                    }
                ),
            }
        )
        return payload


@dataclass(frozen=True)
class Context1OProjCacheTensor:
    name: str
    layer: int
    offset: int
    size: int
    dtype: str
    shape: tuple[int, int]
    category: str


@dataclass(frozen=True)
class Context1OProjCacheLayout:
    layout_path: Path
    cache_file_path: Path
    dtype: str
    dtype_bytes: int
    total_bytes: int
    hidden_dim: int
    attention_value_dim: int
    num_heads: int
    v_head_dim: int
    qk_nope_dim: int
    kv_lora_dim: int
    tensors: tuple[Context1OProjCacheTensor, ...]
    config_sha256: str | None
    source_prepared_manifest: str | None
    source_resident_layout: str | None

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(tensor.layer for tensor in self.tensors)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "layout_path": str(self.layout_path),
            "cache_file": str(self.cache_file_path),
            "dtype": self.dtype,
            "dtype_bytes": self.dtype_bytes,
            "total_bytes": self.total_bytes,
            "dims": {
                "hidden_dim": self.hidden_dim,
                "attention_value_dim": self.attention_value_dim,
                "num_heads": self.num_heads,
                "v_head_dim": self.v_head_dim,
                "qk_nope_dim": self.qk_nope_dim,
                "kv_lora_dim": self.kv_lora_dim,
            },
            "layers": list(self.layers),
            "tensor_count": len(self.tensors),
            "config_sha256": self.config_sha256,
            "source_prepared_manifest": self.source_prepared_manifest,
            "source_resident_layout": self.source_resident_layout,
        }


@dataclass(frozen=True)
class Context1OProjCacheProgress:
    progress_path: Path
    exists: bool
    complete: bool | None
    completed_layers: tuple[int, ...]
    missing_layers: tuple[int, ...]
    backend: str | None = None
    updated_at_unix: float | None = None
    layer_results: tuple[dict[str, Any], ...] = ()
    total_layers: int | None = None
    missing_layer_count: int | None = None
    next_missing_layer: int | None = None
    layer_result_summary: dict[str, Any] | None = None
    suggested_resume_build: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "complete": self.complete,
            "progress_path": str(self.progress_path),
            "backend": self.backend,
            "completed_layers": list(self.completed_layers),
            "missing_layers": list(self.missing_layers),
            "total_layers": self.total_layers,
            "missing_layer_count": self.missing_layer_count,
            "next_missing_layer": self.next_missing_layer,
            "updated_at_unix": self.updated_at_unix,
            "layer_results": [dict(item) for item in self.layer_results],
            "layer_result_summary": self.layer_result_summary,
            "suggested_resume_build": self.suggested_resume_build,
        }


def _optional_str(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise Context1OProjCacheError(f"{field} must be a non-empty string")
    return value


def _relative_layout_file(layout_path: Path, filename: str, label: str) -> Path:
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise Context1OProjCacheError(f"{label} must be relative inside the layout directory")
    return layout_path.parent / path


def _validate_nonoverlapping_spans(
    tensors: tuple[Context1OProjCacheTensor, ...],
    *,
    total_bytes: int,
) -> None:
    spans = sorted((item.offset, item.offset + item.size, item.name) for item in tensors)
    previous_end = 0
    previous_name = ""
    for start, end, name in spans:
        if end > total_bytes:
            raise Context1OProjCacheError(
                f"cache tensor {name} extends beyond total_bytes: {end} > {total_bytes}"
            )
        if start < previous_end:
            raise Context1OProjCacheError(
                f"cache tensor {name} overlaps {previous_name}: {start} < {previous_end}"
            )
        previous_end = end
        previous_name = name


def _layout_dims(payload: dict[str, Any]) -> dict[str, int]:
    dims = payload.get("dims")
    if not isinstance(dims, dict):
        raise Context1OProjCacheError("cache layout missing dims")
    fields = (
        "hidden_dim",
        "attention_value_dim",
        "num_heads",
        "v_head_dim",
        "qk_nope_dim",
        "kv_lora_dim",
    )
    return {field: _require_positive_int(dims.get(field), f"dims.{field}") for field in fields}


def load_context1_o_proj_cache_layout(
    layout_path: str | Path,
    *,
    prepared_dir: str | Path | None = None,
    require_cache_file: bool = True,
) -> Context1OProjCacheLayout:
    path = Path(layout_path)
    payload = _read_json(path, "context=1 o_proj cache layout")
    if payload.get("schema") != CACHE_SCHEMA:
        raise Context1OProjCacheError(
            f"cache layout schema must be {CACHE_SCHEMA}"
        )
    version = _require_positive_int(payload.get("version"), "version")
    if version != 1:
        raise Context1OProjCacheError(f"unsupported cache layout version {version}")
    dtype = _optional_str(payload, "dtype")
    if dtype not in {"BF16", "F32"}:
        raise Context1OProjCacheError("cache dtype must be BF16 or F32")
    expected_dtype_bytes = 2 if dtype == "BF16" else 4
    dtype_bytes = _require_positive_int(payload.get("dtype_bytes"), "dtype_bytes")
    if dtype_bytes != expected_dtype_bytes:
        raise Context1OProjCacheError(
            f"dtype_bytes {dtype_bytes} does not match dtype {dtype}"
        )
    total_bytes = _require_nonnegative_int(payload.get("total_bytes"), "total_bytes")
    weight_file = _optional_str(payload, "weight_file")
    assert weight_file is not None
    cache_file = _relative_layout_file(path, weight_file, "weight_file")
    dims = _layout_dims(payload)
    raw_tensors = payload.get("tensors")
    if not isinstance(raw_tensors, list) or not raw_tensors:
        raise Context1OProjCacheError("cache layout tensors must be a non-empty array")

    tensors: list[Context1OProjCacheTensor] = []
    seen_layers: set[int] = set()
    expected_size = dims["hidden_dim"] * dims["kv_lora_dim"] * dtype_bytes
    for index, raw in enumerate(raw_tensors):
        if not isinstance(raw, dict):
            raise Context1OProjCacheError("cache layout tensors must be objects")
        name = _optional_str(raw, "name")
        assert name is not None
        layer = _require_nonnegative_int(raw.get("layer"), f"tensors[{index}].layer")
        expected_name = f"model.layers.{layer}.self_attn.context1_o_proj_bv.weight"
        if name != expected_name:
            raise Context1OProjCacheError(
                f"cache tensor name {name!r} does not match expected {expected_name!r}"
            )
        if layer in seen_layers:
            raise Context1OProjCacheError(f"duplicate cache tensor for layer {layer}")
        seen_layers.add(layer)
        tensor_dtype = _optional_str(raw, "dtype")
        if tensor_dtype != dtype:
            raise Context1OProjCacheError(
                f"cache tensor {name} dtype {tensor_dtype!r} does not match {dtype}"
            )
        category = _optional_str(raw, "category")
        if category != "context1_attention_output":
            raise Context1OProjCacheError(
                f"cache tensor {name} has invalid category {category!r}"
            )
        shape = _shape(raw, 2, name)
        if shape != (dims["hidden_dim"], dims["kv_lora_dim"]):
            raise Context1OProjCacheError(
                f"cache tensor {name} shape {shape} does not match layout dims"
            )
        offset = _require_nonnegative_int(raw.get("offset"), f"{name} offset")
        size = _require_nonnegative_int(raw.get("size"), f"{name} size")
        if size != expected_size:
            raise Context1OProjCacheError(
                f"cache tensor {name} size {size} does not match expected {expected_size}"
            )
        tensors.append(
            Context1OProjCacheTensor(
                name=name,
                layer=layer,
                offset=offset,
                size=size,
                dtype=dtype,
                shape=(shape[0], shape[1]),
                category=category,
            )
        )
    tensors_tuple = tuple(sorted(tensors, key=lambda item: item.layer))
    if tuple(t.layer for t in tensors) != tuple(t.layer for t in tensors_tuple):
        raise Context1OProjCacheError("cache tensors must be sorted by numeric layer")
    _validate_nonoverlapping_spans(tensors_tuple, total_bytes=total_bytes)
    expected_total = expected_size * len(tensors_tuple)
    if total_bytes != expected_total:
        raise Context1OProjCacheError(
            f"total_bytes {total_bytes} does not match tensor payload {expected_total}"
        )
    if require_cache_file:
        try:
            actual = cache_file.stat().st_size
        except OSError as exc:
            raise Context1OProjCacheError(f"failed to stat cache file {cache_file}: {exc}") from exc
        if actual < total_bytes:
            raise Context1OProjCacheError(
                f"cache file is smaller than total_bytes: {actual} < {total_bytes}"
            )

    config_sha = _optional_str(payload, "config_sha256")
    loaded = Context1OProjCacheLayout(
        layout_path=path,
        cache_file_path=cache_file,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        total_bytes=total_bytes,
        hidden_dim=dims["hidden_dim"],
        attention_value_dim=dims["attention_value_dim"],
        num_heads=dims["num_heads"],
        v_head_dim=dims["v_head_dim"],
        qk_nope_dim=dims["qk_nope_dim"],
        kv_lora_dim=dims["kv_lora_dim"],
        tensors=tensors_tuple,
        config_sha256=config_sha,
        source_prepared_manifest=_optional_str(payload, "source_prepared_manifest"),
        source_resident_layout=_optional_str(payload, "source_resident_layout"),
    )
    if prepared_dir is not None:
        plan = plan_context1_o_proj_cache(
            prepared_dir,
            output_dir=path.parent,
            dtype=dtype,
            layers=loaded.layers,
        )
        if (
            loaded.hidden_dim,
            loaded.attention_value_dim,
            loaded.num_heads,
            loaded.v_head_dim,
            loaded.qk_nope_dim,
            loaded.kv_lora_dim,
        ) != (
            plan.hidden_dim,
            plan.attention_value_dim,
            plan.num_heads,
            plan.v_head_dim,
            plan.qk_nope_dim,
            plan.kv_lora_dim,
        ):
            raise Context1OProjCacheError("cache layout dims do not match prepared artifact")
        if loaded.total_bytes != plan.total_bytes:
            raise Context1OProjCacheError("cache layout bytes do not match prepared artifact")
        if loaded.config_sha256 is not None and plan.config_sha256 is not None:
            if loaded.config_sha256 != plan.config_sha256:
                raise Context1OProjCacheError(
                    "cache layout config_sha256 does not match prepared artifact"
                )
    return loaded


def load_context1_o_proj_cache_progress(
    layout: Context1OProjCacheLayout,
    *,
    require_complete: bool = True,
) -> Context1OProjCacheProgress:
    progress_path = layout.layout_path.parent / DEFAULT_PROGRESS
    if not progress_path.exists():
        missing = layout.layers
        return Context1OProjCacheProgress(
            progress_path=progress_path,
            exists=False,
            complete=None,
            completed_layers=(),
            missing_layers=missing,
            total_layers=len(layout.layers),
            missing_layer_count=len(layout.layers),
            next_missing_layer=layout.layers[0] if layout.layers else None,
            layer_result_summary=_progress_layer_result_summary(
                layout,
                completed_layers=set(),
                layer_results=(),
            ),
            suggested_resume_build=_context1_o_proj_resume_suggestion(
                layout,
                progress_exists=False,
                backend=None,
                missing_layers=missing,
            ),
        )
    payload = _read_json(progress_path, "context1 o_proj cache progress")
    if payload.get("schema") != PROGRESS_SCHEMA:
        raise Context1OProjCacheError("progress schema mismatch")
    if payload.get("cache_schema") != CACHE_SCHEMA:
        raise Context1OProjCacheError("progress cache schema mismatch")
    if payload.get("dtype") != layout.dtype or payload.get("total_bytes") != layout.total_bytes:
        raise Context1OProjCacheError("progress does not match cache layout")
    raw_completed = payload.get("completed_layers")
    if not isinstance(raw_completed, list) or not all(
        type(item) is int for item in raw_completed
    ):
        raise Context1OProjCacheError(
            "progress completed_layers must be an integer array"
        )
    expected_layers = set(layout.layers)
    completed_layers = set(raw_completed)
    extra = sorted(completed_layers - expected_layers)
    if extra:
        raise Context1OProjCacheError(f"progress contains unplanned layers: {extra}")
    missing = tuple(sorted(expected_layers - completed_layers))
    complete = not missing
    if missing and require_complete:
        raise Context1OProjCacheError(
            f"progress is incomplete; missing layers: {list(missing)}"
        )
    backend = payload.get("backend")
    if backend is not None and not isinstance(backend, str):
        raise Context1OProjCacheError("progress backend must be a string")
    updated_at = payload.get("updated_at_unix")
    if updated_at is not None:
        if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)):
            raise Context1OProjCacheError("progress updated_at_unix must be numeric")
        updated_at = float(updated_at)
    layer_results = _progress_layer_results_from_payload(
        payload,
        planned_layers=expected_layers,
    )
    next_missing_layer = missing[0] if missing else None
    return Context1OProjCacheProgress(
        progress_path=progress_path,
        exists=True,
        complete=complete,
        completed_layers=tuple(sorted(completed_layers)),
        missing_layers=missing,
        backend=backend,
        updated_at_unix=updated_at,
        layer_results=layer_results,
        total_layers=len(layout.layers),
        missing_layer_count=len(missing),
        next_missing_layer=next_missing_layer,
        layer_result_summary=_progress_layer_result_summary(
            layout,
            completed_layers=completed_layers,
            layer_results=layer_results,
        ),
        suggested_resume_build=_context1_o_proj_resume_suggestion(
            layout,
            progress_exists=True,
            backend=backend,
            missing_layers=missing,
        ),
    )


def discover_context1_o_proj_layers(
    resident_layout: dict[str, Any],
) -> tuple[CollapsibleLayer, ...]:
    tensors = _tensors_by_name(resident_layout)
    prefix = "model.layers."
    suffix = ".self_attn.o_proj.weight"
    layer_ids = sorted(
        {
            int(name[len(prefix) : -len(suffix)])
            for name in tensors
            if name.startswith(prefix)
            and name.endswith(suffix)
            and name[len(prefix) : -len(suffix)].isdigit()
        }
    )
    layers: list[CollapsibleLayer] = []
    for layer_id in layer_ids:
        attn = f"model.layers.{layer_id}.self_attn"
        o_proj_name = f"{attn}.o_proj.weight"
        o_proj = tensors[o_proj_name]
        o_scales = _companion(tensors, o_proj_name)
        unembed = tensors[f"{attn}.unembed_out.weight"]
        unembed_scales = _companion(tensors, str(unembed["name"]))
        embed = tensors[f"{attn}.embed_q.weight"]
        embed_scales = _companion(tensors, str(embed["name"]))
        hidden_dim, o_in_dim, o_group = _mxfp4_matrix_dims(
            tensors,
            o_proj,
            f"layer {layer_id} o_proj",
        )
        heads, embed_kv, embed_qk, _embed_group = _mxfp4_tensor3d_dims(
            tensors,
            embed,
            f"layer {layer_id} embed_q",
        )
        u_heads, v_head, kv_lora, u_group = _mxfp4_tensor3d_dims(
            tensors,
            unembed,
            f"layer {layer_id} unembed_out",
        )
        if heads != u_heads or embed_kv != kv_lora:
            raise Context1OProjCacheError(
                f"layer {layer_id} embed_q/unembed_out logical dims mismatch"
            )
        if o_in_dim != heads * v_head:
            raise Context1OProjCacheError(
                f"layer {layer_id} o_proj input {o_in_dim} does not match "
                f"heads*v_head {heads * v_head}"
            )
        layers.append(
            CollapsibleLayer(
                layer=layer_id,
                hidden_dim=hidden_dim,
                attention_value_dim=o_in_dim,
                num_heads=heads,
                v_head_dim=v_head,
                qk_nope_dim=embed_qk,
                kv_lora_dim=kv_lora,
                o_proj_group_size=o_group,
                unembed_out_group_size=u_group,
                o_proj_weight=o_proj,
                o_proj_scales=o_scales,
                unembed_out_weight=unembed,
                unembed_out_scales=unembed_scales,
                embed_q_weight=embed,
                embed_q_scales=embed_scales,
            )
        )
    if not layers:
        raise Context1OProjCacheError("no collapsible context=1 o_proj layers found")
    first = layers[0]
    expected = (
        first.hidden_dim,
        first.attention_value_dim,
        first.num_heads,
        first.v_head_dim,
        first.qk_nope_dim,
        first.kv_lora_dim,
    )
    for layer in layers[1:]:
        actual = (
            layer.hidden_dim,
            layer.attention_value_dim,
            layer.num_heads,
            layer.v_head_dim,
            layer.qk_nope_dim,
            layer.kv_lora_dim,
        )
        if actual != expected:
            raise Context1OProjCacheError(
                "mixed-shape context=1 collapse is not supported yet"
            )
    return tuple(layers)


def plan_context1_o_proj_cache(
    prepared_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    dtype: str = "BF16",
    max_cache_bytes: int | None = None,
    layers: tuple[int, ...] | None = None,
) -> Context1OProjCachePlan:
    dtype = dtype.upper()
    if dtype not in {"BF16", "F32"}:
        raise Context1OProjCacheError("dtype must be BF16 or F32")
    dtype_bytes = 2 if dtype == "BF16" else 4
    if max_cache_bytes is not None:
        max_cache_bytes = _require_nonnegative_int(max_cache_bytes, "max_cache_bytes")

    prepared = Path(prepared_dir)
    manifest_path = prepared / "manifest.json"
    manifest = _read_json(manifest_path, "prepared manifest")
    resident_layout_rel = manifest.get("resident_layout", "resident/layout.json")
    if not isinstance(resident_layout_rel, str) or not resident_layout_rel:
        raise Context1OProjCacheError("manifest resident_layout must be a string")
    resident_layout_path = Path(resident_layout_rel)
    if not resident_layout_path.is_absolute():
        resident_layout_path = prepared / resident_layout_path
    resident_layout = _read_json(resident_layout_path, "resident layout")
    weight_file = resident_layout.get("weight_file")
    if not isinstance(weight_file, str) or not weight_file:
        raise Context1OProjCacheError("resident layout missing weight_file")
    resident_weight_path = Path(weight_file)
    if not resident_weight_path.is_absolute():
        resident_weight_path = resident_layout_path.parent / resident_weight_path
    if not resident_weight_path.exists():
        raise Context1OProjCacheError(f"resident weight file not found: {resident_weight_path}")
    resident_bytes = resident_weight_path.stat().st_size

    all_layers = discover_context1_o_proj_layers(resident_layout)
    if layers is not None:
        selected = set(layers)
        missing = sorted(selected - {item.layer for item in all_layers})
        if missing:
            raise Context1OProjCacheError(f"requested layers not found: {missing}")
        collapse_layers = tuple(item for item in all_layers if item.layer in selected)
    else:
        collapse_layers = all_layers
    for layer in collapse_layers:
        for tensor, label in (
            (layer.o_proj_weight, f"layer {layer.layer} o_proj.weight"),
            (layer.o_proj_scales, f"layer {layer.layer} o_proj.scales"),
            (layer.unembed_out_weight, f"layer {layer.layer} unembed_out.weight"),
            (layer.unembed_out_scales, f"layer {layer.layer} unembed_out.scales"),
        ):
            _check_tensor_span(resident_bytes, tensor, label)

    first = collapse_layers[0]
    layer_bytes = first.hidden_dim * first.kv_lora_dim * dtype_bytes
    total_bytes = layer_bytes * len(collapse_layers)
    if max_cache_bytes is not None and total_bytes > max_cache_bytes:
        raise Context1OProjCacheError(
            f"context=1 o_proj cache {total_bytes} bytes exceeds limit {max_cache_bytes}"
        )
    out_dir = Path(output_dir) if output_dir is not None else prepared / DEFAULT_CACHE_DIRNAME
    cache_layout_path = out_dir / DEFAULT_CACHE_LAYOUT
    cache_file_path = out_dir / DEFAULT_CACHE_BIN
    progress_path = out_dir / DEFAULT_PROGRESS
    fma_total = sum(_layer_fma(layer) for layer in collapse_layers)
    current_o_proj = sum(
        _tensor_size(layer.o_proj_weight, f"layer {layer.layer} o_proj.weight")
        + _tensor_size(layer.o_proj_scales, f"layer {layer.layer} o_proj.scales")
        for layer in collapse_layers
    )
    config_sha = resident_layout.get("config_sha256")
    if config_sha is not None and not isinstance(config_sha, str):
        raise Context1OProjCacheError("resident layout config_sha256 must be a string")
    return Context1OProjCachePlan(
        prepared_dir=prepared,
        prepared_manifest=manifest_path,
        resident_layout_path=resident_layout_path,
        resident_weight_path=resident_weight_path,
        output_dir=out_dir,
        cache_layout_path=cache_layout_path,
        cache_file_path=cache_file_path,
        progress_path=progress_path,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        layers=collapse_layers,
        hidden_dim=first.hidden_dim,
        attention_value_dim=first.attention_value_dim,
        num_heads=first.num_heads,
        v_head_dim=first.v_head_dim,
        qk_nope_dim=first.qk_nope_dim,
        kv_lora_dim=first.kv_lora_dim,
        total_bytes=total_bytes,
        fma_total=fma_total,
        current_o_proj_storage_per_token=current_o_proj,
        config_sha256=config_sha,
    )


def _cache_layout(plan: Context1OProjCachePlan) -> dict[str, Any]:
    offset = 0
    tensors: list[dict[str, Any]] = []
    per_layer = plan.hidden_dim * plan.kv_lora_dim * plan.dtype_bytes
    for layer in plan.layers:
        tensors.append(
            {
                "name": f"model.layers.{layer.layer}.self_attn.context1_o_proj_bv.weight",
                "layer": layer.layer,
                "offset": offset,
                "size": per_layer,
                "dtype": plan.dtype,
                "shape": [plan.hidden_dim, plan.kv_lora_dim],
                "category": "context1_attention_output",
            }
        )
        offset += per_layer
    return {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "config_sha256": plan.config_sha256,
        "source_prepared_manifest": str(plan.prepared_manifest),
        "source_resident_layout": str(plan.resident_layout_path),
        "source_resident_weight": str(plan.resident_weight_path),
        "context_limit": "decode/context_length_1_only",
        "dtype": plan.dtype,
        "dtype_bytes": plan.dtype_bytes,
        "weight_file": plan.cache_file_path.name,
        "total_bytes": plan.total_bytes,
        "dims": {
            "hidden_dim": plan.hidden_dim,
            "attention_value_dim": plan.attention_value_dim,
            "num_heads": plan.num_heads,
            "v_head_dim": plan.v_head_dim,
            "qk_nope_dim": plan.qk_nope_dim,
            "kv_lora_dim": plan.kv_lora_dim,
        },
        "tensors": tensors,
    }


def _progress_payload(
    plan: Context1OProjCachePlan,
    *,
    completed: set[int],
    backend: str,
    layer_results: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "schema": PROGRESS_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "dtype": plan.dtype,
        "total_bytes": plan.total_bytes,
        "fma_total": plan.fma_total,
        "backend": backend,
        "layers": [layer.layer for layer in plan.layers],
        "completed_layers": sorted(completed),
        "layer_results": [dict(item) for item in layer_results],
        "cache_file": str(plan.cache_file_path),
        "updated_at_unix": time.time(),
    }


def _progress_layer_results_from_payload(
    payload: dict[str, Any],
    *,
    planned_layers: set[int],
) -> tuple[dict[str, Any], ...]:
    raw = payload.get("layer_results", [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise Context1OProjCacheError("progress layer_results must be an array")
    results: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise Context1OProjCacheError("progress layer_results must contain objects")
        layer = item.get("layer")
        if type(layer) is not int:
            raise Context1OProjCacheError("progress layer_results layer must be an integer")
        if layer not in planned_layers:
            raise Context1OProjCacheError(
                f"progress layer_results contains unplanned layer: {layer}"
            )
        results.append(dict(item))
    return tuple(results)


def _read_progress_payload(
    path: Path,
    plan: Context1OProjCachePlan,
    backend: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path, "context1 o_proj cache progress")
    if payload.get("schema") != PROGRESS_SCHEMA:
        raise Context1OProjCacheError("progress schema mismatch")
    if payload.get("cache_schema") != CACHE_SCHEMA:
        raise Context1OProjCacheError("progress cache schema mismatch")
    if payload.get("dtype") != plan.dtype or payload.get("total_bytes") != plan.total_bytes:
        raise Context1OProjCacheError("progress does not match requested cache plan")
    if payload.get("backend") != backend:
        raise Context1OProjCacheError("progress backend does not match requested backend")
    raw = payload.get("completed_layers")
    if not isinstance(raw, list) or not all(type(item) is int for item in raw):
        raise Context1OProjCacheError("progress completed_layers must be an integer array")
    planned = {layer.layer for layer in plan.layers}
    completed = set(raw)
    extra = sorted(completed - planned)
    if extra:
        raise Context1OProjCacheError(f"progress contains unplanned layers: {extra}")
    _progress_layer_results_from_payload(payload, planned_layers=planned)
    return payload


def _read_progress(path: Path, plan: Context1OProjCachePlan, backend: str) -> set[int]:
    payload = _read_progress_payload(path, plan, backend)
    if payload is None:
        return set()
    raw = payload["completed_layers"]
    completed = set(raw)
    return completed


def _read_progress_layer_results(
    path: Path,
    plan: Context1OProjCachePlan,
    backend: str,
) -> tuple[dict[str, Any], ...]:
    payload = _read_progress_payload(path, plan, backend)
    if payload is None:
        return ()
    planned = {layer.layer for layer in plan.layers}
    return _progress_layer_results_from_payload(payload, planned_layers=planned)


def _decode_mxfp4_matrix_rows(
    fd: int,
    weight: dict[str, Any],
    scales: dict[str, Any],
    *,
    row_start: int,
    row_count: int,
    in_dim: int,
    group_size: int,
    label: str,
) -> list[list[float]]:
    out_dim, packed_cols = _shape(weight, 2, label)
    scale_rows, groups = _shape(scales, 2, f"{label}.scales")
    if row_start < 0 or row_count < 0 or row_start + row_count > out_dim:
        raise Context1OProjCacheError(f"{label} row range is out of bounds")
    if scale_rows != out_dim or packed_cols * 8 != in_dim:
        raise Context1OProjCacheError(f"{label} shape changed during build")
    packed_per_group = group_size // 8
    weight_row_bytes = packed_cols * 4
    scale_row_bytes = groups
    weight_raw = _read_exact(
        fd,
        row_count * weight_row_bytes,
        _tensor_offset(weight, label) + row_start * weight_row_bytes,
        label,
    )
    scale_raw = _read_exact(
        fd,
        row_count * scale_row_bytes,
        _tensor_offset(scales, f"{label}.scales") + row_start * scale_row_bytes,
        f"{label}.scales",
    )
    rows: list[list[float]] = []
    for local_row in range(row_count):
        row: list[float] = []
        weight_base = local_row * weight_row_bytes
        scale_base = local_row * scale_row_bytes
        for group in range(groups):
            scale = _mxfp4_e8m0_to_f32(scale_raw[scale_base + group])
            packed_base = weight_base + group * packed_per_group * 4
            for packed_index in range(packed_per_group):
                packed = struct.unpack_from("<I", weight_raw, packed_base + packed_index * 4)[0]
                for nibble_index in range(8):
                    code = (packed >> (nibble_index * 4)) & 0xF
                    row.append(_mxfp4_e2m1_to_f32(code) * scale)
        rows.append(row)
    return rows


def _decode_mxfp4_tensor3d_as_flat_rows(
    fd: int,
    weight: dict[str, Any],
    scales: dict[str, Any],
    *,
    dim0: int,
    dim1: int,
    dim2: int,
    group_size: int,
    label: str,
) -> list[list[float]]:
    raw_dim0, raw_dim1, packed_dim2 = _shape(weight, 3, label)
    scale0, scale1, groups = _shape(scales, 3, f"{label}.scales")
    if (raw_dim0, raw_dim1, packed_dim2 * 8) != (dim0, dim1, dim2):
        raise Context1OProjCacheError(f"{label} shape changed during build")
    if (scale0, scale1) != (dim0, dim1):
        raise Context1OProjCacheError(f"{label}.scales shape changed during build")
    packed_per_group = group_size // 8
    vector_weight_bytes = packed_dim2 * 4
    vector_scale_bytes = groups
    weight_raw = _read_exact(
        fd,
        _tensor_size(weight, label),
        _tensor_offset(weight, label),
        label,
    )
    scale_raw = _read_exact(
        fd,
        _tensor_size(scales, f"{label}.scales"),
        _tensor_offset(scales, f"{label}.scales"),
        f"{label}.scales",
    )
    rows: list[list[float]] = []
    for head in range(dim0):
        for value_index in range(dim1):
            row: list[float] = []
            vector = head * dim1 + value_index
            weight_base = vector * vector_weight_bytes
            scale_base = vector * vector_scale_bytes
            for group in range(groups):
                scale = _mxfp4_e8m0_to_f32(scale_raw[scale_base + group])
                packed_base = weight_base + group * packed_per_group * 4
                for packed_index in range(packed_per_group):
                    packed = struct.unpack_from("<I", weight_raw, packed_base + packed_index * 4)[0]
                    for nibble_index in range(8):
                        code = (packed >> (nibble_index * 4)) & 0xF
                        row.append(_mxfp4_e2m1_to_f32(code) * scale)
            rows.append(row)
    return rows


def _build_layer_reference(
    fd: int,
    plan: Context1OProjCachePlan,
    layer: CollapsibleLayer,
    *,
    row_tile: int,
) -> bytes:
    bv = _decode_mxfp4_tensor3d_as_flat_rows(
        fd,
        layer.unembed_out_weight,
        layer.unembed_out_scales,
        dim0=layer.num_heads,
        dim1=layer.v_head_dim,
        dim2=layer.kv_lora_dim,
        group_size=layer.unembed_out_group_size,
        label=f"layer {layer.layer} unembed_out",
    )
    out = bytearray(plan.hidden_dim * plan.kv_lora_dim * plan.dtype_bytes)
    for row_start in range(0, plan.hidden_dim, row_tile):
        row_count = min(row_tile, plan.hidden_dim - row_start)
        o_rows = _decode_mxfp4_matrix_rows(
            fd,
            layer.o_proj_weight,
            layer.o_proj_scales,
            row_start=row_start,
            row_count=row_count,
            in_dim=layer.attention_value_dim,
            group_size=layer.o_proj_group_size,
            label=f"layer {layer.layer} o_proj",
        )
        for local_row, o_row in enumerate(o_rows):
            global_row = row_start + local_row
            for kv in range(plan.kv_lora_dim):
                acc = 0.0
                for value_index, o_value in enumerate(o_row):
                    acc += o_value * bv[value_index][kv]
                write_offset = (
                    global_row * plan.kv_lora_dim + kv
                ) * plan.dtype_bytes
                _write_scalar(out, write_offset, acc, plan.dtype)
    return bytes(out)


def _layer_fma(layer: CollapsibleLayer) -> int:
    return layer.hidden_dim * layer.attention_value_dim * layer.kv_lora_dim


def _layer_source_bytes(layer: CollapsibleLayer) -> int:
    return (
        _tensor_size(layer.o_proj_weight, f"layer {layer.layer} o_proj.weight")
        + _tensor_size(layer.o_proj_scales, f"layer {layer.layer} o_proj.scales")
        + _tensor_size(layer.unembed_out_weight, f"layer {layer.layer} unembed_out.weight")
        + _tensor_size(layer.unembed_out_scales, f"layer {layer.layer} unembed_out.scales")
    )


def _layer_cache_bytes(plan: Context1OProjCachePlan) -> int:
    return plan.hidden_dim * plan.kv_lora_dim * plan.dtype_bytes


def _metal_builder_live_bytes(plan: Context1OProjCachePlan, layer: CollapsibleLayer) -> int:
    return _layer_source_bytes(layer) + _layer_cache_bytes(plan)


def _base_layer_result(
    plan: Context1OProjCachePlan,
    layer: CollapsibleLayer,
    *,
    backend: str,
) -> dict[str, Any]:
    return {
        "layer": layer.layer,
        "backend": backend,
        "source_bytes_read": _layer_source_bytes(layer),
        "cache_bytes_written": _layer_cache_bytes(plan),
        "fma_count": _layer_fma(layer),
        "estimated_metal_builder_live_bytes": _metal_builder_live_bytes(plan, layer),
    }


def _reference_layer_result(
    plan: Context1OProjCachePlan,
    layer: CollapsibleLayer,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    result = _base_layer_result(plan, layer, backend="reference")
    result["elapsed_seconds"] = elapsed_seconds
    return result


def _metal_layer_result(
    plan: Context1OProjCachePlan,
    layer: CollapsibleLayer,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = _base_layer_result(plan, layer, backend="metal")
    int_fields = (
        "source_bytes_read",
        "cache_bytes_written",
        "fma_count",
        "estimated_live_working_set_bytes",
    )
    number_fields = (
        "max_live_working_set_mib",
        "read_seconds",
        "kernel_seconds",
        "write_seconds",
        "elapsed_seconds",
        "output0",
    )
    for field in int_fields:
        value = _json_int(payload.get(field))
        if value is not None:
            result[field] = value
    for field in number_fields:
        value = _json_number(payload.get(field))
        if value is not None:
            result[field] = value
    live_ok = _json_boolish(payload.get("live_working_set_ok"))
    if live_ok is not None:
        result["live_working_set_ok"] = live_ok
    device_name = payload.get("device_name")
    if isinstance(device_name, str):
        result["device_name"] = device_name
    return result


def _run_metal_layer_builder(
    plan: Context1OProjCachePlan,
    layer: CollapsibleLayer,
    *,
    metal_binary: str | Path,
    max_build_fma: int,
    max_metal_builder_live_bytes: int,
) -> dict[str, Any]:
    binary = Path(metal_binary)
    if not binary.exists():
        raise Context1OProjCacheError(f"Metal builder binary not found: {binary}")
    source_bytes = _layer_source_bytes(layer)
    per_layer_bytes = plan.hidden_dim * plan.kv_lora_dim * plan.dtype_bytes
    max_read_mib = max(
        1,
        math.ceil(max(source_bytes, per_layer_bytes) / (1024 * 1024)),
    )
    cmd = [
        str(binary),
        "--build-context1-o-proj-cache-layer",
        "--resident-layout",
        str(plan.resident_layout_path),
        "--probe-layer",
        str(layer.layer),
        "--context1-o-proj-cache-layout",
        str(plan.cache_layout_path),
        "--max-cache-read-mib",
        str(max_read_mib),
        "--max-context1-o-proj-build-gfma",
        f"{max_build_fma / 1.0e9:.9g}",
        "--max-live-working-set-mib",
        f"{max_metal_builder_live_bytes / (1024 * 1024):.9g}",
        "--json",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise Context1OProjCacheError(
            f"Metal context1 cache builder failed for layer {layer.layer}: {detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Context1OProjCacheError(
            f"Metal context1 cache builder returned invalid JSON for layer {layer.layer}: "
            f"{completed.stdout.strip()}"
        ) from exc
    if payload.get("ok") is not True:
        raise Context1OProjCacheError(
            f"Metal context1 cache builder reported failure for layer {layer.layer}: "
            f"{payload}"
        )
    return payload


def build_context1_o_proj_cache(
    prepared_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    dtype: str = "BF16",
    execute: bool = False,
    force: bool = False,
    backend: str = "reference",
    max_cache_bytes: int | None = None,
    max_build_fma: int = 50_000_000,
    disk_safety_margin_bytes: int = 16 * 1024**3,
    max_metal_builder_live_bytes: int = 512 * 1024**2,
    row_tile: int = 8,
    layers: tuple[int, ...] | None = None,
    build_layers: tuple[int, ...] | None = None,
    build_next_layers: int | None = None,
    metal_binary: str | Path | None = None,
) -> Context1OProjCacheBuildResult:
    max_build_fma = _require_nonnegative_int(max_build_fma, "max_build_fma")
    disk_safety_margin_bytes = _require_nonnegative_int(
        disk_safety_margin_bytes,
        "disk_safety_margin_bytes",
    )
    max_metal_builder_live_bytes = _require_positive_int(
        max_metal_builder_live_bytes,
        "max_metal_builder_live_bytes",
    )
    if build_layers is not None and build_next_layers is not None:
        raise Context1OProjCacheError(
            "build_layers and build_next_layers are mutually exclusive"
        )
    if build_next_layers is not None:
        build_next_layers = _require_positive_int(
            build_next_layers,
            "build_next_layers",
        )
    row_tile = _require_positive_int(row_tile, "row_tile")
    if backend not in {"reference", "metal"}:
        raise Context1OProjCacheError(
            "backend must be reference or metal"
        )
    plan = plan_context1_o_proj_cache(
        prepared_dir,
        output_dir=output_dir,
        dtype=dtype,
        max_cache_bytes=max_cache_bytes,
        layers=layers,
    )
    planned_layers = {layer.layer for layer in plan.layers}
    completed_for_selection: set[int] = set()
    layer_results_for_selection: tuple[dict[str, Any], ...] = ()
    if build_next_layers is not None and not force:
        completed_for_selection = _read_progress(plan.progress_path, plan, backend)
        layer_results_for_selection = _read_progress_layer_results(
            plan.progress_path,
            plan,
            backend,
        )
    if build_layers is not None:
        requested_build_layers = tuple(sorted(set(build_layers)))
        if not requested_build_layers:
            raise Context1OProjCacheError("requested build layers must not be empty")
        missing = sorted(set(requested_build_layers) - planned_layers)
        if missing:
            raise Context1OProjCacheError(
                f"requested build layers not found in cache plan: {missing}"
            )
        layers_to_build = tuple(
            layer for layer in plan.layers if layer.layer in set(requested_build_layers)
        )
    elif build_next_layers is not None:
        requested_build_layers = tuple(
            layer.layer
            for layer in plan.layers
            if layer.layer not in completed_for_selection
        )[:build_next_layers]
        layers_to_build = tuple(
            layer for layer in plan.layers if layer.layer in set(requested_build_layers)
        )
    else:
        requested_build_layers = None
        layers_to_build = plan.layers
    build_fma_total = sum(_layer_fma(layer) for layer in layers_to_build)
    budget = disk_budget(
        plan.output_dir,
        plan.total_bytes,
        safety_margin_bytes=disk_safety_margin_bytes,
    )
    if not execute:
        return Context1OProjCacheBuildResult(
            plan=plan,
            executed=False,
            completed_layers=tuple(sorted(completed_for_selection)),
            skipped_layers=(),
            elapsed_seconds=0.0,
            backend=backend,
            requested_build_layers=requested_build_layers,
            requested_build_next_layers=build_next_layers,
            disk_budget=budget,
            max_metal_builder_live_bytes=max_metal_builder_live_bytes,
            layer_results=layer_results_for_selection,
        )
    if build_fma_total > max_build_fma:
        raise Context1OProjCacheError(
            f"context=1 o_proj cache build requires {build_fma_total} FMA, "
            f"exceeds max_build_fma={max_build_fma}; raise the cap intentionally"
        )
    if not budget.ok:
        raise Context1OProjCacheError(
            "not enough free disk for context=1 o_proj cache: "
            f"need {plan.total_bytes + disk_safety_margin_bytes} bytes including "
            f"margin, have {budget.available_bytes} bytes"
        )
    if backend == "metal":
        builder_live_max = max(
            (_metal_builder_live_bytes(plan, layer) for layer in layers_to_build),
            default=0,
        )
        if builder_live_max > max_metal_builder_live_bytes:
            raise Context1OProjCacheError(
                f"context=1 o_proj Metal builder estimated live bytes "
                f"{builder_live_max} exceeds max_metal_builder_live_bytes="
                f"{max_metal_builder_live_bytes}; raise the cap intentionally"
            )
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in (plan.cache_layout_path, plan.cache_file_path, plan.progress_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise Context1OProjCacheError(f"failed to remove {path}: {exc}") from exc
    elif plan.cache_layout_path.exists() and not plan.progress_path.exists():
        raise Context1OProjCacheError(
            f"cache layout already exists: {plan.cache_layout_path}; pass force=True"
        )

    completed = _read_progress(plan.progress_path, plan, backend)
    layer_results = list(
        _read_progress_layer_results(plan.progress_path, plan, backend)
    )
    if completed:
        try:
            existing_bytes = plan.cache_file_path.stat().st_size
        except OSError as exc:
            raise Context1OProjCacheError(
                f"progress exists but cache file is not readable: {plan.cache_file_path}"
            ) from exc
        if existing_bytes < plan.total_bytes:
            raise Context1OProjCacheError(
                f"progress exists but cache file is smaller than planned: "
                f"{existing_bytes} < {plan.total_bytes}"
            )
    mode = "r+b" if plan.cache_file_path.exists() else "w+b"
    started = time.perf_counter()
    skipped: list[int] = []
    per_layer = plan.hidden_dim * plan.kv_lora_dim * plan.dtype_bytes
    layer_offsets = {layer.layer: index * per_layer for index, layer in enumerate(plan.layers)}
    with plan.cache_file_path.open(mode) as out:
        out.truncate(plan.total_bytes)
    _write_json_atomic(plan.cache_layout_path, _cache_layout(plan))
    if backend == "reference":
        with plan.cache_file_path.open("r+b") as out:
            fd = os.open(plan.resident_weight_path, os.O_RDONLY)
            try:
                for layer in layers_to_build:
                    if layer.layer in completed:
                        skipped.append(layer.layer)
                        continue
                    layer_started = time.perf_counter()
                    layer_bytes = _build_layer_reference(
                        fd,
                        plan,
                        layer,
                        row_tile=row_tile,
                    )
                    if len(layer_bytes) != per_layer:
                        raise Context1OProjCacheError(
                            f"layer {layer.layer} produced {len(layer_bytes)} bytes, "
                            f"expected {per_layer}"
                        )
                    out.seek(layer_offsets[layer.layer])
                    out.write(layer_bytes)
                    out.flush()
                    completed.add(layer.layer)
                    layer_results.append(
                        _reference_layer_result(
                            plan,
                            layer,
                            elapsed_seconds=time.perf_counter() - layer_started,
                        )
                    )
                    _write_json_atomic(
                        plan.progress_path,
                        _progress_payload(
                            plan,
                            completed=completed,
                            backend=backend,
                            layer_results=tuple(layer_results),
                        ),
                    )
            finally:
                os.close(fd)
    else:
        binary = DEFAULT_METAL_BINARY if metal_binary is None else metal_binary
        for layer in layers_to_build:
            if layer.layer in completed:
                skipped.append(layer.layer)
                continue
            payload = _run_metal_layer_builder(
                plan,
                layer,
                metal_binary=binary,
                max_build_fma=max_build_fma,
                max_metal_builder_live_bytes=max_metal_builder_live_bytes,
            )
            completed.add(layer.layer)
            layer_results.append(_metal_layer_result(plan, layer, payload))
            _write_json_atomic(
                plan.progress_path,
                _progress_payload(
                    plan,
                    completed=completed,
                    backend=backend,
                    layer_results=tuple(layer_results),
                ),
            )
    elapsed = time.perf_counter() - started
    return Context1OProjCacheBuildResult(
        plan=plan,
        executed=True,
        completed_layers=tuple(sorted(completed)),
        skipped_layers=tuple(skipped),
        elapsed_seconds=elapsed,
        backend=backend,
        requested_build_layers=requested_build_layers,
        requested_build_next_layers=build_next_layers,
        disk_budget=budget,
        max_metal_builder_live_bytes=max_metal_builder_live_bytes,
        layer_results=tuple(layer_results),
    )
