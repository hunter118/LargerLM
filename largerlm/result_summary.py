from __future__ import annotations

import json
import hashlib
import math
import shlex
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class ResultSummaryError(RuntimeError):
    """Raised when a LargerLM result artifact cannot be summarized."""


_TEXT_DUPLICATE_NAMED_ELAPSED_FIELDS = {
    "copy_elapsed_seconds",
    "stage_copy_elapsed_seconds",
    "total_expert_stage_copy_elapsed_seconds",
}
_PROFILE_RECOMMENDATION_MIN_TOTAL_SPEEDUP_RATIO = 0.98
_PROFILE_RECOMMENDATION_LARGE_TOTAL_SPEEDUP_RATIO = 0.85
_PROFILE_RECOMMENDATION_MAX_TOTAL_REGRESSION_RATIO = 1.02
_PROFILE_RECOMMENDATION_TENSOR_REGRESSION_RATIO = 1.10
_PROFILE_RECOMMENDATION_TENSOR_REGRESSION_SECONDS = 0.25
_PROFILE_RECOMMENDATION_NET_WIN_REGRESSION_MULTIPLIER = 2.0
_RUNNER_PROCESS_FUSION_MIN_UNIQUE_COMMANDS = 500
_JSON_OBJECT_FILE_MAX_BYTES = 16 * 1024 * 1024
_PREFILL_LINEAR_ACCELERATED_BACKENDS = ("mpsgraph-f32", "mps-matrix-f32")
_ROUTED_MOE_TIMING_FIELDS = (
    ("sort", "moe_timing_sort_seconds"),
    ("setup", "moe_timing_setup_seconds"),
    ("expert_read", "moe_timing_expert_read_seconds"),
    ("input_read", "moe_timing_input_read_seconds"),
    ("output_read", "moe_timing_output_read_seconds"),
    ("kernel", "moe_timing_kernel_seconds"),
    ("mxfp4_swiglu_kernel", "moe_timing_mxfp4_swiglu_kernel_seconds"),
    ("mxfp4_down_add_kernel", "moe_timing_mxfp4_down_add_kernel_seconds"),
    ("output_write", "moe_timing_output_write_seconds"),
    ("final_read", "moe_timing_final_read_seconds"),
    ("total", "moe_timing_total_seconds"),
)
_ROUTED_MOE_OUTER_WALL_FIELDS = (
    ("total", "wall_outer_total_elapsed_seconds"),
    ("plan", "wall_outer_plan_elapsed_seconds"),
    ("tile_router", "wall_outer_tile_router_elapsed_seconds"),
    ("tile_input", "wall_outer_tile_input_elapsed_seconds"),
    ("stage", "wall_outer_stage_elapsed_seconds"),
    ("staged_moe", "wall_outer_staged_moe_elapsed_seconds"),
    ("scatter", "wall_outer_scatter_elapsed_seconds"),
    ("cleanup", "wall_outer_cleanup_elapsed_seconds"),
    ("output_validation", "wall_outer_output_validation_elapsed_seconds"),
)


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _sum_numeric_mapping(value: object) -> float:
    if not isinstance(value, dict):
        return 0.0
    total = 0.0
    for item in value.values():
        number = _finite_float(item)
        if number is not None:
            total += number
    return total


def _sum_nonnegative_backend_ints(
    value: object,
    *,
    backends: tuple[str, ...] | None = None,
) -> int:
    mapping = _as_mapping(value)
    total = 0
    for backend, item in mapping.items():
        if not isinstance(backend, str):
            continue
        if backends is not None and backend not in backends:
            continue
        parsed = _nonnegative_int(item)
        if parsed is not None:
            total += parsed
    return total


def _positive_backend_int_mapping(value: object) -> dict[str, int]:
    result: dict[str, int] = {}
    for backend, item in _as_mapping(value).items():
        if not isinstance(backend, str):
            continue
        parsed = _nonnegative_int(item)
        if parsed is not None and parsed > 0:
            result[backend] = parsed
    return dict(sorted(result.items()))


def _router_gate_component_acceleration(
    component_stats: object,
) -> dict[str, int]:
    router = _as_mapping(
        _as_mapping(component_stats).get("moe.router_gate_proj")
    )
    counts = router.get("linear_backend_counts")
    flops = router.get("linear_backend_flops")
    return {
        "router_gate_accelerated_matrix_count": _sum_nonnegative_backend_ints(
            counts,
            backends=_PREFILL_LINEAR_ACCELERATED_BACKENDS,
        ),
        "router_gate_accelerated_estimated_flops": _sum_nonnegative_backend_ints(
            flops,
            backends=_PREFILL_LINEAR_ACCELERATED_BACKENDS,
        ),
    }


def _linear_component_stat_rows(
    value: object,
    *,
    top_limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows: list[dict[str, Any]] = []
    for component, raw_payload in value.items():
        if not isinstance(component, str) or not isinstance(raw_payload, dict):
            continue
        counts = _as_mapping(raw_payload.get("linear_backend_counts"))
        flops = _as_mapping(raw_payload.get("linear_backend_flops"))
        elapsed = _as_mapping(raw_payload.get("linear_backend_elapsed_seconds"))
        tflops = _as_mapping(raw_payload.get("linear_backend_estimated_tflops"))
        count_total = sum(
            int(item)
            for item in counts.values()
            if _nonnegative_int(item) is not None
        )
        flops_total = sum(
            int(item)
            for item in flops.values()
            if _nonnegative_int(item) is not None
        )
        elapsed_total = _sum_numeric_mapping(elapsed)
        if count_total <= 0 and flops_total <= 0 and elapsed_total <= 0.0:
            continue
        row: dict[str, Any] = {
            "component": component,
            "count": count_total,
            "elapsed_seconds": elapsed_total,
            "estimated_flops": flops_total,
            "linear_backend_counts": dict(sorted(counts.items())),
            "linear_backend_elapsed_seconds": dict(sorted(elapsed.items())),
            "linear_backend_flops": dict(sorted(flops.items())),
        }
        if tflops:
            row["linear_backend_estimated_tflops"] = dict(sorted(tflops.items()))
        if elapsed_total > 0.0 and flops_total > 0:
            row["estimated_tflops"] = flops_total / elapsed_total / 1e12
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            float(item.get("elapsed_seconds") or 0.0),
            int(item.get("estimated_flops") or 0),
            str(item.get("component") or ""),
        ),
        reverse=True,
    )[:top_limit]


def _coarse_elapsed_path(path: str) -> str:
    parts: list[str] = []
    for raw in path.split("."):
        if "[" in raw:
            raw = raw.split("[", 1)[0]
        if raw:
            parts.append(raw)
    return ".".join(parts[-4:]) if parts else path


def _is_container_elapsed(path: str) -> bool:
    if path == "token_result":
        return True
    if path == "token_result.prompt_prefill":
        return True
    if path.startswith("token_result.steps[") and path.count(".") == 1:
        return True
    return False


def _elapsed_record_hint(record: dict[str, Any]) -> dict[str, Any]:
    hint: dict[str, Any] = {"elapsed_seconds": record["elapsed_seconds"]}
    for key in (
        "backend",
        "layer",
        "tensor",
        "matrix_name",
        "name",
        "kind",
        "batch_tokens",
        "tokens",
        "planned_read_bytes",
        "staged_bytes",
        "requested_bytes",
    ):
        if key in record:
            hint[key] = record[key]
    hint["path"] = record["path"]
    return hint


def _tensor_suffix_group(tensor: object) -> str | None:
    if not isinstance(tensor, str) or not tensor:
        return None
    marker = ".layers."
    if marker not in tensor:
        return tensor
    suffix = tensor.split(marker, 1)[1]
    if "." not in suffix:
        return tensor
    return suffix.split(".", 1)[1]


def _tensor_suffix_group_summary(
    records: list[dict[str, Any]],
    *,
    top_limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        suffix = _tensor_suffix_group(record.get("tensor"))
        if suffix is None:
            continue
        elapsed = _finite_float(record.get("elapsed_seconds"))
        if elapsed is None:
            continue
        group = grouped.setdefault(
            suffix,
            {
                "tensor_suffix": suffix,
                "count": 0,
                "elapsed_seconds": 0.0,
                "backend_counts": Counter(),
                "backend_elapsed_seconds": defaultdict(float),
                "top_record": None,
            },
        )
        group["count"] += 1
        group["elapsed_seconds"] += elapsed
        backend = record.get("backend")
        if isinstance(backend, str) and backend:
            group["backend_counts"][backend] += 1
            group["backend_elapsed_seconds"][backend] += elapsed
        top_record = group.get("top_record")
        top_elapsed = (
            _finite_float(top_record.get("elapsed_seconds"))
            if isinstance(top_record, dict)
            else None
        )
        if top_elapsed is None or elapsed > top_elapsed:
            group["top_record"] = _elapsed_record_hint(record)

    result: list[dict[str, Any]] = []
    for group in grouped.values():
        result.append(
            {
                "tensor_suffix": group["tensor_suffix"],
                "count": group["count"],
                "elapsed_seconds": group["elapsed_seconds"],
                "backend_counts": dict(sorted(group["backend_counts"].items())),
                "backend_elapsed_seconds": dict(
                    sorted(group["backend_elapsed_seconds"].items())
                ),
                "top_record": group["top_record"],
            }
        )
    return sorted(
        result,
        key=lambda item: float(item["elapsed_seconds"]),
        reverse=True,
    )[:top_limit]


def _collect_elapsed_records(payload: object) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            elapsed = _finite_float(value.get("elapsed_seconds"))
            if elapsed is not None:
                record: dict[str, Any] = {
                    "elapsed_seconds": elapsed,
                    "path": path,
                }
                for key in (
                    "backend",
                    "layer",
                    "tensor",
                    "matrix_name",
                    "name",
                    "kind",
                    "batch_tokens",
                    "tokens",
                    "planned_read_bytes",
                    "staged_bytes",
                    "requested_bytes",
                ):
                    if key in value:
                        record[key] = value[key]
                records.append(record)
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                walk(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "")
    return records


def _collect_named_elapsed_fields(payload: object) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key.endswith("_elapsed_seconds"):
                    elapsed = _finite_float(item)
                    if elapsed is not None:
                        records.append(
                            {
                                "field": key,
                                "elapsed_seconds": elapsed,
                                "path": child_path,
                            }
                        )
                walk(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "")
    return records


def _decode_steps_summary(
    steps: list[object],
    *,
    top_limit: int,
) -> dict[str, Any]:
    step_records: list[dict[str, Any]] = []
    total_layer_elapsed = 0.0
    total_attention_elapsed = 0.0
    total_mlp_elapsed = 0.0
    total_mla_timing: dict[str, float] = defaultdict(float)
    total_mlp_stages: dict[str, float] = defaultdict(float)
    total_mlp_preload_enabled_count = 0
    total_mlp_preload_selected_bytes = 0
    total_mlp_mxfp4_fused_decode_enabled_count = 0
    layer_count = 0
    top_attention_candidates: list[dict[str, Any]] = []
    top_mlp_candidates: list[dict[str, Any]] = []
    top_mla_kernel_candidates: list[dict[str, Any]] = []

    for step_index, step in enumerate(steps):
        step_map = _as_mapping(step)
        decode_layers = _as_list(step_map.get("decode_layers"))
        if not decode_layers:
            continue
        step_layer_elapsed = 0.0
        step_attention_elapsed = 0.0
        step_mlp_elapsed = 0.0
        step_mla_timing: dict[str, float] = defaultdict(float)
        step_mlp_stages: dict[str, float] = defaultdict(float)
        step_mlp_preload_enabled_count = 0
        step_mlp_preload_selected_bytes = 0
        step_mlp_mxfp4_fused_decode_enabled_count = 0
        step_top_attention: list[dict[str, Any]] = []
        step_top_mlp: list[dict[str, Any]] = []
        step_top_mla_kernel: list[dict[str, Any]] = []
        for layer in decode_layers:
            layer_map = _as_mapping(layer)
            elapsed = _finite_float(layer_map.get("elapsed_seconds"))
            attention = _finite_float(layer_map.get("attention_elapsed_seconds"))
            mlp = _finite_float(layer_map.get("mlp_elapsed_seconds"))
            mla_timing = _as_mapping(
                layer_map.get("mla_attention_timing_elapsed_seconds")
            )
            mlp_stages = _as_mapping(layer_map.get("mlp_stage_elapsed_seconds"))
            mlp_diagnostics = _as_mapping(layer_map.get("mlp_diagnostics"))
            mla_kernel = _finite_float(mla_timing.get("kernel"))
            mla_total = _finite_float(mla_timing.get("total"))
            if elapsed is not None:
                step_layer_elapsed += elapsed
            if attention is not None:
                step_attention_elapsed += attention
            if mlp is not None:
                step_mlp_elapsed += mlp
            for name, value in mla_timing.items():
                number = _finite_float(value)
                if number is not None:
                    step_mla_timing[str(name)] += number
                    total_mla_timing[str(name)] += number
            for name, value in mlp_stages.items():
                number = _finite_float(value)
                if number is not None:
                    step_mlp_stages[str(name)] += number
                    total_mlp_stages[str(name)] += number
            if bool(mlp_diagnostics.get("preload_selected_enabled")):
                step_mlp_preload_enabled_count += 1
                total_mlp_preload_enabled_count += 1
            if bool(mlp_diagnostics.get("mxfp4_fused_decode_enabled")):
                step_mlp_mxfp4_fused_decode_enabled_count += 1
                total_mlp_mxfp4_fused_decode_enabled_count += 1
            preload_bytes = _nonnegative_int(
                mlp_diagnostics.get("preload_selected_bytes")
            )
            if preload_bytes is not None:
                step_mlp_preload_selected_bytes += preload_bytes
                total_mlp_preload_selected_bytes += preload_bytes
            layer_id = _nonnegative_int(layer_map.get("layer"))
            attention_record = {
                "step_index": step_index,
                "position": step_map.get("position"),
                "layer": layer_id,
                "elapsed_seconds": elapsed,
                "attention_elapsed_seconds": attention,
                "mlp_elapsed_seconds": mlp,
                "mla_kernel_elapsed_seconds": mla_kernel,
                "mla_total_elapsed_seconds": mla_total,
            }
            if attention is not None:
                step_top_attention.append(attention_record)
                top_attention_candidates.append(attention_record)
            if mlp is not None:
                step_top_mlp.append(attention_record)
                top_mlp_candidates.append(attention_record)
            if mla_kernel is not None:
                step_top_mla_kernel.append(attention_record)
                top_mla_kernel_candidates.append(attention_record)
        step_top_attention = sorted(
            step_top_attention,
            key=lambda item: float(item.get("attention_elapsed_seconds") or 0.0),
            reverse=True,
        )[:top_limit]
        step_top_mla_kernel = sorted(
            step_top_mla_kernel,
            key=lambda item: float(item.get("mla_kernel_elapsed_seconds") or 0.0),
            reverse=True,
        )[:top_limit]
        step_top_mlp = sorted(
            step_top_mlp,
            key=lambda item: float(item.get("mlp_elapsed_seconds") or 0.0),
            reverse=True,
        )[:top_limit]
        record = {
            "step_index": step_index,
            "position": step_map.get("position"),
            "input_token_id": step_map.get("input_token_id"),
            "selected_token_id": step_map.get("selected_token_id"),
            "elapsed_seconds": _finite_float(step_map.get("elapsed_seconds")),
            "logits_elapsed_seconds": _finite_float(
                step_map.get("logits_elapsed_seconds")
            ),
            "expert_read_bytes": _nonnegative_int(step_map.get("expert_read_bytes")),
            "cache_read_bytes": _nonnegative_int(step_map.get("cache_read_bytes")),
            "logits_read_bytes": _nonnegative_int(step_map.get("logits_read_bytes")),
            "decode_layer_count": len(decode_layers),
            "decode_layer_elapsed_seconds": step_layer_elapsed or None,
            "attention_elapsed_seconds": step_attention_elapsed or None,
            "mlp_elapsed_seconds": step_mlp_elapsed or None,
            "mla_attention_timing_elapsed_seconds": (
                dict(sorted(step_mla_timing.items())) if step_mla_timing else {}
            ),
            "mlp_stage_elapsed_seconds": (
                dict(sorted(step_mlp_stages.items())) if step_mlp_stages else {}
            ),
            "mlp_preload_selected_enabled_count": (
                step_mlp_preload_enabled_count or None
            ),
            "mlp_preload_selected_bytes": step_mlp_preload_selected_bytes or None,
            "mlp_mxfp4_fused_decode_enabled_count": (
                step_mlp_mxfp4_fused_decode_enabled_count or None
            ),
            "top_attention_layers": step_top_attention,
            "top_mlp_layers": step_top_mlp,
            "top_mla_kernel_layers": step_top_mla_kernel,
        }
        step_records.append(record)
        total_layer_elapsed += step_layer_elapsed
        total_attention_elapsed += step_attention_elapsed
        total_mlp_elapsed += step_mlp_elapsed
        layer_count += len(decode_layers)

    top_steps = sorted(
        step_records,
        key=lambda item: float(
            item.get("decode_layer_elapsed_seconds")
            or item.get("elapsed_seconds")
            or 0.0
        ),
        reverse=True,
    )[:top_limit]
    top_attention_layers = sorted(
        top_attention_candidates,
        key=lambda item: float(item.get("attention_elapsed_seconds") or 0.0),
        reverse=True,
    )[:top_limit]
    top_mla_kernel_layers = sorted(
        top_mla_kernel_candidates,
        key=lambda item: float(item.get("mla_kernel_elapsed_seconds") or 0.0),
        reverse=True,
    )[:top_limit]
    top_mlp_layers = sorted(
        top_mlp_candidates,
        key=lambda item: float(item.get("mlp_elapsed_seconds") or 0.0),
        reverse=True,
    )[:top_limit]
    return {
        "present": bool(step_records),
        "generated_step_count": len(steps),
        "step_count_with_decode_layers": len(step_records),
        "decode_layer_count": layer_count,
        "decode_layer_elapsed_seconds": total_layer_elapsed or None,
        "attention_elapsed_seconds": total_attention_elapsed or None,
        "mlp_elapsed_seconds": total_mlp_elapsed or None,
        "mla_attention_timing_elapsed_seconds": (
            dict(sorted(total_mla_timing.items())) if total_mla_timing else {}
        ),
        "mlp_stage_elapsed_seconds": (
            dict(sorted(total_mlp_stages.items())) if total_mlp_stages else {}
        ),
        "mlp_preload_selected_enabled_count": (
            total_mlp_preload_enabled_count or None
        ),
        "mlp_preload_selected_bytes": total_mlp_preload_selected_bytes or None,
        "mlp_mxfp4_fused_decode_enabled_count": (
            total_mlp_mxfp4_fused_decode_enabled_count or None
        ),
        "top_slowest_steps": top_steps,
        "top_attention_layers": top_attention_layers,
        "top_mlp_layers": top_mlp_layers,
        "top_mla_kernel_layers": top_mla_kernel_layers,
    }


def _runner_command_group(command: object) -> str | None:
    if not isinstance(command, list) or not command:
        return None
    if not all(isinstance(item, str) for item in command):
        return None
    executable = Path(command[0]).name
    if executable == "largerlm-runner":
        for item in command[1:]:
            if item.startswith("--run-"):
                return item
        return executable
    if executable == "python-rope-singleton":
        return executable
    return None


def _collect_runner_command_records(payload: object) -> dict[str, Any]:
    record_groups: Counter[str] = Counter()
    unique_groups: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    seen_commands: set[tuple[str, ...]] = set()
    record_count = 0

    def walk(value: object) -> None:
        nonlocal record_count
        if isinstance(value, dict):
            command = value.get("command")
            group = _runner_command_group(command)
            if group is not None and isinstance(command, list):
                record_count += 1
                record_groups[group] += 1
                command_key = tuple(str(item) for item in command)
                if command_key not in seen_commands:
                    seen_commands.add(command_key)
                    unique_groups[group] += 1
                    examples.setdefault(group, [str(item) for item in command[:10]])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return {
        "record_count": record_count,
        "unique_command_count": len(seen_commands),
        "duplicate_record_count": record_count - len(seen_commands),
        "group_counts": dict(sorted(record_groups.items())),
        "unique_group_counts": dict(sorted(unique_groups.items())),
        "top_groups": [
            {
                "group": group,
                "count": count,
                "example_argv_prefix": examples.get(group, []),
            }
            for group, count in record_groups.most_common()
        ],
        "top_unique_groups": [
            {
                "group": group,
                "count": count,
                "record_count": record_groups.get(group, 0),
                "example_argv_prefix": examples.get(group, []),
            }
            for group, count in unique_groups.most_common()
        ],
    }


def _first_nonnegative_int(*values: object) -> int | None:
    for value in values:
        parsed = _nonnegative_int(value)
        if parsed is not None:
            return parsed
    return None


def _first_finite_float(*values: object) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _put_known(record: dict[str, Any], key: str, value: object) -> None:
    if value is not None:
        record[key] = value


def _sequence_count(value: object) -> int | None:
    items = _as_list(value)
    return len(items) if items else None


def _nonnegative_int_list(value: object) -> list[int] | None:
    items = _as_list(value)
    if not items:
        return None
    result: list[int] = []
    for item in items:
        parsed = _nonnegative_int(item)
        if parsed is None:
            return None
        result.append(parsed)
    return result


def _sum_int_records(records: list[dict[str, Any]], key: str) -> int | None:
    total = 0
    seen = False
    for record in records:
        value = _nonnegative_int(record.get(key))
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def _max_int_records(records: list[dict[str, Any]], key: str) -> int | None:
    values = [
        value
        for record in records
        if (value := _nonnegative_int(record.get(key))) is not None
    ]
    return max(values) if values else None


def _sum_float_records(records: list[dict[str, Any]], key: str) -> float | None:
    total = 0.0
    seen = False
    for record in records:
        value = _finite_float(record.get(key))
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def _min_float_records(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for record in records
        if (value := _finite_float(record.get(key))) is not None
    ]
    return min(values) if values else None


def _top_records_by_numeric_field(
    records: list[dict[str, Any]],
    key: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    sortable = [
        record
        for record in records
        if _finite_float(record.get(key)) is not None
        or _nonnegative_int(record.get(key)) is not None
    ]
    return [
        dict(record)
        for record in sorted(
            sortable,
            key=lambda item: float(item.get(key) or 0),
            reverse=True,
        )[:limit]
    ]


def _collect_routed_moe_layer_records(
    prompt_prefill: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk_position, raw_chunk in enumerate(_as_list(prompt_prefill.get("chunks"))):
        chunk = _as_mapping(raw_chunk)
        chunk_index = _first_nonnegative_int(chunk.get("chunk_index"), chunk_position)
        for layer_position, raw_layer in enumerate(_as_list(chunk.get("layers"))):
            layer = _as_mapping(raw_layer)
            staged = _as_mapping(layer.get("staged_mlp"))
            if not staged:
                continue
            elapsed = _finite_float(staged.get("routed_moe_elapsed_seconds"))
            if elapsed is None:
                continue
            stage_result = _as_mapping(staged.get("stage_result"))
            batch_plan = _as_mapping(stage_result.get("batch_plan"))
            io_summary = _as_mapping(stage_result.get("io_summary"))
            staged_moe = _as_mapping(staged.get("staged_moe"))
            tiled_staged_moe = _as_mapping(staged.get("tiled_staged_moe"))
            selected_expert_count = _first_nonnegative_int(
                _sequence_count(stage_result.get("selected_experts")),
                _sequence_count(batch_plan.get("selected_experts")),
                _sequence_count(staged_moe.get("selected_experts")),
                io_summary.get("selected_expert_count"),
                io_summary.get("slot_count"),
            )

            record: dict[str, Any] = {
                "elapsed_seconds": elapsed,
                "chunk_index": chunk_index,
                "layer_index": layer_position,
            }
            _put_known(
                record,
                "layer",
                _first_nonnegative_int(staged.get("layer"), layer.get("layer")),
            )
            _put_known(
                record,
                "batch_tokens",
                _first_nonnegative_int(
                    staged.get("batch_tokens"),
                    chunk.get("batch_tokens"),
                    batch_plan.get("batch_tokens"),
                    io_summary.get("batch_tokens"),
                ),
            )
            _put_known(record, "top_k", _first_nonnegative_int(staged.get("top_k")))
            _put_known(record, "max_k", _first_nonnegative_int(staged.get("max_k")))
            _put_known(
                record,
                "assignments",
                _first_nonnegative_int(
                    batch_plan.get("total_assignments"),
                    io_summary.get("total_assignments"),
                    staged.get("static_capacity_used_slots"),
                    staged_moe.get("static_capacity_used_slots"),
                ),
            )
            _put_known(record, "selected_expert_count", selected_expert_count)
            selected_experts = (
                _nonnegative_int_list(stage_result.get("selected_experts"))
                or _nonnegative_int_list(batch_plan.get("selected_experts"))
                or _nonnegative_int_list(staged_moe.get("selected_experts"))
            )
            if selected_experts is not None:
                record["selected_experts"] = selected_experts
            _put_known(
                record,
                "stage_planned_read_bytes",
                _first_nonnegative_int(
                    staged_moe.get("stage_planned_read_bytes"),
                    stage_result.get("planned_read_bytes"),
                    batch_plan.get("planned_read_bytes"),
                    io_summary.get("planned_read_bytes"),
                ),
            )
            _put_known(
                record,
                "stage_staged_bytes",
                _first_nonnegative_int(
                    staged_moe.get("stage_staged_bytes"),
                    stage_result.get("staged_bytes"),
                    io_summary.get("staged_bytes"),
                ),
            )
            _put_known(
                record,
                "stage_unique_requested_bytes",
                _first_nonnegative_int(
                    staged_moe.get("stage_unique_requested_bytes"),
                    batch_plan.get("unique_requested_bytes"),
                    io_summary.get("unique_requested_bytes"),
                ),
            )
            _put_known(
                record,
                "stage_waste_bytes",
                _first_nonnegative_int(
                    staged_moe.get("stage_waste_bytes"),
                    batch_plan.get("waste_bytes"),
                    io_summary.get("waste_bytes"),
                ),
            )
            _put_known(
                record,
                "stage_raw_range_count",
                _first_nonnegative_int(
                    staged_moe.get("stage_raw_range_count"),
                    batch_plan.get("raw_range_count"),
                    io_summary.get("raw_range_count"),
                ),
            )
            _put_known(
                record,
                "stage_coalesced_range_count",
                _first_nonnegative_int(
                    staged_moe.get("stage_coalesced_range_count"),
                    batch_plan.get("coalesced_range_count"),
                    io_summary.get("coalesced_range_count"),
                ),
            )
            _put_known(
                record,
                "compact_stage_bytes",
                _first_nonnegative_int(staged.get("compact_stage_bytes")),
            )
            _put_known(
                record,
                "compact_stage_materialized_bytes",
                _first_nonnegative_int(
                    staged.get("compact_stage_materialized_bytes")
                ),
            )
            _put_known(
                record,
                "stage_plus_compact_bytes",
                _first_nonnegative_int(staged.get("stage_plus_compact_bytes")),
            )
            _put_known(
                record,
                "stage_plus_compact_materialized_bytes",
                _first_nonnegative_int(
                    staged.get("stage_plus_compact_materialized_bytes")
                ),
            )
            _put_known(
                record,
                "stage_copy_elapsed_seconds",
                _first_finite_float(
                    staged.get("stage_copy_elapsed_seconds"),
                    stage_result.get("copy_elapsed_seconds"),
                    io_summary.get("copy_elapsed_seconds"),
                ),
            )
            _put_known(
                record,
                "stage_copy_throughput_gib_per_second",
                _first_finite_float(
                    staged.get("stage_copy_throughput_gib_per_second"),
                    stage_result.get("copy_throughput_gib_per_second"),
                    io_summary.get("copy_throughput_gib_per_second"),
                ),
            )
            _put_known(
                record,
                "copy_chunk_bytes",
                _first_nonnegative_int(
                    staged.get("copy_chunk_bytes"),
                    staged_moe.get("copy_chunk_bytes"),
                    stage_result.get("copy_chunk_bytes"),
                ),
            )
            _put_known(
                record,
                "static_capacity_per_expert",
                _first_nonnegative_int(staged.get("static_capacity_per_expert")),
            )
            _put_known(
                record,
                "static_capacity_used_slots",
                _first_nonnegative_int(
                    staged.get("static_capacity_used_slots"),
                    staged_moe.get("static_capacity_used_slots"),
                ),
            )
            _put_known(
                record,
                "static_capacity_total_slots",
                _first_nonnegative_int(
                    staged.get("static_capacity_total_slots"),
                    staged_moe.get("static_capacity_total_slots"),
                ),
            )
            used_slots = _nonnegative_int(record.get("static_capacity_used_slots"))
            total_slots = _nonnegative_int(record.get("static_capacity_total_slots"))
            if used_slots is not None and total_slots is not None and total_slots > 0:
                record["static_capacity_utilization"] = used_slots / total_slots
            _put_known(
                record,
                "effective_moe_token_block",
                _first_nonnegative_int(
                    staged.get("effective_moe_token_block"),
                    staged_moe.get("effective_moe_token_block"),
                ),
            )
            for key in (
                "moe_token_block_mode",
                "compact_stage_storage",
                "moe_output_accumulator",
            ):
                value = staged.get(key)
                if value is None:
                    value = staged_moe.get(key)
                if isinstance(value, str) and value:
                    record[key] = value
            _put_known(
                record,
                "moe_output_accumulator_bytes",
                _first_nonnegative_int(
                    staged.get("moe_output_accumulator_bytes"),
                    staged_moe.get("moe_output_accumulator_bytes"),
                ),
            )
            _put_known(
                record,
                "moe_max_expert_tokens",
                _first_nonnegative_int(
                    staged.get("moe_max_expert_tokens"),
                    staged_moe.get("moe_max_expert_tokens"),
                ),
            )
            _put_known(
                record,
                "moe_batch_buffer_bytes",
                _first_nonnegative_int(
                    staged.get("moe_batch_buffer_bytes"),
                    staged_moe.get("moe_batch_buffer_bytes"),
                ),
            )
            _put_known(
                record,
                "moe_estimated_peak_bytes",
                _first_nonnegative_int(
                    staged.get("moe_estimated_peak_bytes"),
                    staged_moe.get("moe_estimated_peak_bytes"),
                ),
            )
            for _phase, key in _ROUTED_MOE_TIMING_FIELDS:
                _put_known(
                    record,
                    key,
                    _first_finite_float(staged.get(key), staged_moe.get(key)),
                )
            for key in (
                "wall_total_elapsed_seconds",
                "wall_compact_stage_elapsed_seconds",
                "wall_routes_elapsed_seconds",
                "wall_static_capacity_elapsed_seconds",
                "wall_runner_elapsed_seconds",
                "wall_output_validation_elapsed_seconds",
            ):
                _put_known(
                    record,
                    key,
                    _first_finite_float(staged.get(key), staged_moe.get(key)),
                )
            for record_key, source_key in (
                (
                    "wall_outer_total_elapsed_seconds",
                    "wall_total_elapsed_seconds",
                ),
                ("wall_outer_plan_elapsed_seconds", "wall_plan_elapsed_seconds"),
                (
                    "wall_outer_tile_router_elapsed_seconds",
                    "wall_tile_router_elapsed_seconds",
                ),
                (
                    "wall_outer_tile_input_elapsed_seconds",
                    "wall_tile_input_elapsed_seconds",
                ),
                ("wall_outer_stage_elapsed_seconds", "wall_stage_elapsed_seconds"),
                (
                    "wall_outer_staged_moe_elapsed_seconds",
                    "wall_staged_moe_elapsed_seconds",
                ),
                (
                    "wall_outer_scatter_elapsed_seconds",
                    "wall_scatter_elapsed_seconds",
                ),
                (
                    "wall_outer_cleanup_elapsed_seconds",
                    "wall_cleanup_elapsed_seconds",
                ),
                (
                    "wall_outer_output_validation_elapsed_seconds",
                    "wall_output_validation_elapsed_seconds",
                ),
            ):
                _put_known(
                    record,
                    record_key,
                    _first_finite_float(
                        staged.get(record_key),
                        tiled_staged_moe.get(source_key),
                    ),
                )
            runner_total = _finite_float(record.get("moe_timing_total_seconds"))
            if runner_total is not None and elapsed > 0.0:
                non_runner_elapsed = max(0.0, elapsed - runner_total)
                record["moe_non_runner_elapsed_seconds"] = non_runner_elapsed
                record["moe_non_runner_elapsed_fraction"] = (
                    non_runner_elapsed / elapsed
                )
                record["moe_runner_total_elapsed_fraction"] = (
                    runner_total / elapsed
                )
            router_margin = _as_mapping(staged.get("router_margin_summary"))
            if router_margin:
                record["router_margin_summary"] = dict(router_margin)
                for source_key, record_key in (
                    ("min_effective_score_margin", "router_min_effective_score_margin"),
                    ("min_topk_score_margin", "router_min_topk_score_margin"),
                    ("min_group_score_margin", "router_min_group_score_margin"),
                ):
                    _put_known(
                        record,
                        record_key,
                        _first_finite_float(router_margin.get(source_key)),
                    )
            records.append(record)
    return records


def _collect_mla_key_cache_summary(
    prompt_prefill: dict[str, Any],
) -> dict[str, Any] | None:
    return _collect_mla_cache_summary(
        prompt_prefill,
        enabled_field="mla_key_cache",
        bytes_field="mla_key_cache_bytes",
        total_bytes_field="total_mla_key_cache_bytes",
    )


def _collect_mla_value_cache_summary(
    prompt_prefill: dict[str, Any],
) -> dict[str, Any] | None:
    return _collect_mla_cache_summary(
        prompt_prefill,
        enabled_field="mla_value_cache",
        bytes_field="mla_value_cache_bytes",
        total_bytes_field="total_mla_value_cache_bytes",
    )


def _collect_mla_cache_summary(
    prompt_prefill: dict[str, Any],
    *,
    enabled_field: str,
    bytes_field: str,
    total_bytes_field: str,
) -> dict[str, Any] | None:
    layer_count = 0
    enabled_layer_count = 0
    disabled_layer_count = 0
    total_bytes = 0
    observed = False
    for raw_chunk in _as_list(prompt_prefill.get("chunks")):
        chunk = _as_mapping(raw_chunk)
        for raw_layer in _as_list(chunk.get("layers")):
            layer = _as_mapping(raw_layer)
            attention = _as_mapping(layer.get("attention"))
            if not attention:
                continue
            enabled = attention.get(enabled_field)
            cache_bytes = _nonnegative_int(attention.get(bytes_field))
            if isinstance(enabled, bool) or cache_bytes is not None:
                observed = True
                layer_count += 1
                if enabled is True:
                    enabled_layer_count += 1
                elif enabled is False:
                    disabled_layer_count += 1
                if cache_bytes is not None:
                    total_bytes += cache_bytes
    if not observed:
        return None
    return {
        "observed": True,
        "layer_count": layer_count,
        "enabled_layer_count": enabled_layer_count,
        "disabled_layer_count": disabled_layer_count,
        "all_layers_enabled": layer_count > 0 and enabled_layer_count == layer_count,
        total_bytes_field: total_bytes,
    }


def _collect_mla_attention_layers(
    prompt_prefill: dict[str, Any],
    *,
    top_limit: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for chunk_index, raw_chunk in enumerate(_as_list(prompt_prefill.get("chunks"))):
        chunk = _as_mapping(raw_chunk)
        for layer_index, raw_layer in enumerate(_as_list(chunk.get("layers"))):
            layer = _as_mapping(raw_layer)
            attention = _as_mapping(layer.get("attention"))
            if not attention:
                continue
            mla = _as_mapping(attention.get("mla_attention"))
            elapsed = _first_finite_float(
                attention.get("mla_attention_elapsed_seconds"),
                mla.get("elapsed_seconds"),
            )
            if elapsed is None:
                continue
            record: dict[str, Any] = {
                "chunk_index": chunk_index,
                "layer_index": layer_index,
                "elapsed_seconds": elapsed,
            }
            layer_id = _first_nonnegative_int(
                mla.get("layer"),
                layer.get("layer"),
                layer_index,
            )
            if layer_id is not None:
                record["layer"] = layer_id
            for key in (
                "batch_tokens",
                "context_length",
                "start_position",
                "num_heads",
                "qk_nope_dim",
                "rope_dim",
                "v_head_dim",
                "kv_lora_dim",
                "cache_read_bytes",
                "cache_f32_bytes",
                "kv_b_matrix_bytes",
                "kv_b_f32_bytes",
                "output_bytes",
                "mla_key_cache_bytes",
                "mla_value_cache_bytes",
                "estimated_peak_bytes",
            ):
                value = _nonnegative_int(mla.get(key))
                if value is not None:
                    record[key] = value
            for key in (
                "mla_key_cache",
                "mla_value_cache",
                "indexed",
                "rope_interleave",
            ):
                value = mla.get(key)
                if isinstance(value, bool):
                    record[key] = value
            for key in (
                "attention_value_source",
            ):
                value = mla.get(key)
                if isinstance(value, str) and value:
                    record[key] = value
            for key in (
                "attention_scale",
                "rope_theta",
                "mla_timing_input_elapsed_seconds",
                "mla_timing_cache_read_elapsed_seconds",
                "mla_timing_value_read_elapsed_seconds",
                "mla_timing_metal_setup_elapsed_seconds",
                "mla_timing_kernel_elapsed_seconds",
                "mla_timing_write_elapsed_seconds",
                "mla_timing_total_elapsed_seconds",
            ):
                value = _finite_float(mla.get(key))
                if value is not None:
                    record[key] = value
            records.append(record)

    timing_fields = (
        "mla_timing_input_elapsed_seconds",
        "mla_timing_cache_read_elapsed_seconds",
        "mla_timing_value_read_elapsed_seconds",
        "mla_timing_metal_setup_elapsed_seconds",
        "mla_timing_kernel_elapsed_seconds",
        "mla_timing_write_elapsed_seconds",
        "mla_timing_total_elapsed_seconds",
    )
    timing_totals: dict[str, float] = {}
    for key in timing_fields:
        total = sum(
            value
            for record in records
            if (value := _finite_float(record.get(key))) is not None
        )
        if total > 0.0:
            timing_totals[key] = total
    return {
        "count": len(records),
        "elapsed_seconds": sum(
            value
            for record in records
            if (value := _finite_float(record.get("elapsed_seconds"))) is not None
        )
        or None,
        "timing_elapsed_seconds": timing_totals or None,
        "top_slowest_layers": _top_records_by_numeric_field(
            records,
            "elapsed_seconds",
            limit=top_limit,
        ),
    }


def _compact_mla_cache_summary(
    value: object,
    *,
    total_bytes_field: str,
) -> dict[str, Any] | None:
    raw = _as_mapping(value)
    if raw.get("observed") is not True:
        return None
    layer_count = _nonnegative_int(raw.get("layer_count"))
    enabled_layer_count = _nonnegative_int(raw.get("enabled_layer_count"))
    disabled_layer_count = _nonnegative_int(raw.get("disabled_layer_count"))
    total_bytes = _nonnegative_int(raw.get(total_bytes_field))
    if (
        layer_count is None
        or enabled_layer_count is None
        or disabled_layer_count is None
        or total_bytes is None
    ):
        return None
    return {
        "observed": True,
        "layer_count": layer_count,
        "enabled_layer_count": enabled_layer_count,
        "disabled_layer_count": disabled_layer_count,
        "all_layers_enabled": layer_count > 0 and enabled_layer_count == layer_count,
        total_bytes_field: total_bytes,
    }


def _result_prompt_token_ids(value: object) -> list[int] | None:
    items = _as_list(value)
    if not items:
        return None
    token_ids: list[int] = []
    for item in items:
        parsed = _nonnegative_int(item)
        if parsed is None:
            return None
        token_ids.append(parsed)
    return token_ids


def _path_is_file(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    try:
        return Path(path).is_file()
    except OSError:
        return False


def _path_is_dir(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    try:
        return Path(path).is_dir()
    except OSError:
        return False


def _paths_refer_to_same_file(left: object, right: object) -> bool | None:
    if not isinstance(left, str) or not left:
        return None
    if not isinstance(right, str) or not right:
        return None
    if left == right:
        return True
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(
            right
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return False


def _directory_file_stats(path: object) -> dict[str, int] | None:
    if not isinstance(path, str) or not path:
        return None
    try:
        root = Path(path)
        if not root.is_dir():
            return None
        file_count = 0
        total_bytes = 0
        for item in root.rglob("*"):
            try:
                if not item.is_file():
                    continue
                file_count += 1
                total_bytes += item.stat().st_size
            except OSError:
                continue
        return {"file_count": file_count, "total_bytes": total_bytes}
    except OSError:
        return None


def _file_sha256(path: object) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    try:
        with Path(path).open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except OSError:
        return None


def _json_object_file(path: object) -> dict[str, Any] | None:
    if not isinstance(path, str) or not path:
        return None
    try:
        file_path = Path(path)
        if file_path.stat().st_size > _JSON_OBJECT_FILE_MAX_BYTES:
            return None
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _first_nonempty_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _http_result_wrapper_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    response = _as_mapping(payload.get("response"))
    if not response and _as_mapping(payload.get("largerlm")):
        response = payload
    largerlm = _as_mapping(response.get("largerlm"))
    token_result = _as_mapping(largerlm.get("token_result"))
    response_token_result = _as_mapping(response.get("token_result"))
    if not token_result and response_token_result:
        token_result = response_token_result
    if not token_result and (
        response.get("generated_token_ids") is not None
        or response.get("prompt_prefill") is not None
    ):
        token_result = response
    if not token_result:
        return payload, None

    request_check = _as_mapping(token_result.get("request_check")) or _as_mapping(
        largerlm.get("request_check")
    ) or _as_mapping(
        response.get("request_check")
    )
    launch_envelope = _as_mapping(
        token_result.get("launch_audit_envelope")
    ) or _as_mapping(largerlm.get("launch_audit_envelope")) or _as_mapping(
        response.get("launch_audit_envelope")
    )
    applied_profile = _as_mapping(
        token_result.get("applied_launch_profile")
    ) or _as_mapping(largerlm.get("applied_launch_profile")) or _as_mapping(
        response.get("applied_launch_profile")
    )
    usage = _as_mapping(response.get("usage"))
    request_payload = _as_mapping(payload.get("request"))

    prompt_ids = _as_list(token_result.get("prompt_token_ids")) or _as_list(
        response.get("prompt_token_ids")
    )
    generated_ids = _as_list(token_result.get("generated_token_ids")) or _as_list(
        response.get("generated_token_ids")
    )
    prompt_tokens = _first_nonnegative_int(
        request_check.get("prompt_token_count"),
        request_payload.get("prompt_tokens"),
        usage.get("prompt_tokens"),
        len(prompt_ids) if prompt_ids else None,
    )
    max_new_tokens = _first_nonnegative_int(
        request_check.get("max_new_tokens"),
        request_payload.get("max_new_tokens"),
        usage.get("completion_tokens"),
        len(generated_ids) if generated_ids else None,
    )
    prepared = _as_mapping(applied_profile.get("prepared"))
    prepared_manifest = _first_nonempty_str(
        applied_profile.get("current_prepared_manifest"),
        prepared.get("prepared_manifest"),
    )
    normalized = {
        "prepared_manifest": prepared_manifest,
        "request": {
            "prompt_tokens": prompt_tokens,
            "prompt_token_ids": prompt_ids,
            "max_new_tokens": max_new_tokens,
            "launch_audit_path": launch_envelope.get("artifact_path"),
            "prefill_mla_kv_b_cache_dir": _first_nonempty_str(
                request_payload.get("prefill_mla_kv_b_cache_dir"),
                request_check.get("prefill_mla_kv_b_cache_dir"),
            ),
            "prefill_mla_kv_b_cache_file_count": _first_nonnegative_int(
                request_payload.get("prefill_mla_kv_b_cache_file_count"),
                request_check.get("prefill_mla_kv_b_cache_file_count"),
            ),
            "prefill_mla_kv_b_cache_total_bytes": _first_nonnegative_int(
                request_payload.get("prefill_mla_kv_b_cache_total_bytes"),
                request_check.get("prefill_mla_kv_b_cache_total_bytes"),
            ),
        },
        "token_result": token_result,
    }
    wrapper = {
        "schema": payload.get("schema"),
        "endpoint": payload.get("endpoint")
        or (
            "/v1/chat/completions"
            if response.get("object") == "chat.completion"
            else "/v1/completions"
            if response.get("object") == "text_completion"
            else None
        ),
        "url": payload.get("url"),
        "status": payload.get("status"),
        "client_elapsed_seconds": _first_finite_float(
            payload.get("elapsed_seconds"),
            payload.get("client_elapsed_seconds"),
        ),
        "response_object": response.get("object"),
        "model": response.get("model"),
        "usage": usage if usage else None,
        "chat_template_backend": largerlm.get("chat_template_backend"),
        "tokenizer_backend": largerlm.get("tokenizer_backend"),
    }
    return normalized, wrapper


def _primary_token_result(payload: dict[str, Any]) -> dict[str, Any]:
    token_result = _as_mapping(payload.get("token_result"))
    if token_result:
        return token_result
    text_result = _as_mapping(payload.get("text_result"))
    nested = _as_mapping(text_result.get("token_result"))
    if nested:
        return nested
    return {}


def _launch_binding_token_result(payload: dict[str, Any]) -> dict[str, Any]:
    token_result = _primary_token_result(payload)
    if token_result:
        return token_result
    if (
        payload.get("generated_token_ids") is not None
        or payload.get("prompt_prefill") is not None
        or payload.get("applied_launch_profile") is not None
    ):
        return payload
    return {}


def _result_launch_binding(payload: dict[str, Any]) -> dict[str, Any]:
    request = _as_mapping(payload.get("request"))
    token_result = _launch_binding_token_result(payload)
    largerlm = _as_mapping(payload.get("largerlm"))
    request_check = (
        _as_mapping(token_result.get("request_check"))
        or _as_mapping(largerlm.get("request_check"))
        or _as_mapping(payload.get("request_check"))
    )
    launch_envelope = (
        _as_mapping(token_result.get("launch_audit_envelope"))
        or _as_mapping(largerlm.get("launch_audit_envelope"))
        or _as_mapping(payload.get("launch_audit_envelope"))
        or _as_mapping(request_check.get("launch_audit_envelope"))
    )
    applied = (
        _as_mapping(token_result.get("applied_launch_profile"))
        or _as_mapping(largerlm.get("applied_launch_profile"))
        or _as_mapping(payload.get("applied_launch_profile"))
    )
    prepared = _as_mapping(applied.get("prepared"))
    current_prepared_manifest = _first_nonempty_str(
        applied.get("current_prepared_manifest"),
        prepared.get("prepared_manifest"),
    )
    prepared_manifest = _first_nonempty_str(
        payload.get("prepared_manifest"),
        current_prepared_manifest,
        prepared.get("prepared_manifest"),
    )
    launch_profile_path = applied.get("path")
    launch_audit_path = _first_nonempty_str(
        request.get("launch_audit_path"),
        launch_envelope.get("artifact_path"),
        request_check.get("launch_audit_path"),
    )
    replay_mla_kv_b_cache_dir = _first_nonempty_str(
        request.get("prefill_mla_kv_b_cache_dir"),
        request_check.get("prefill_mla_kv_b_cache_dir"),
    )
    replay_mla_kv_b_cache_expected_file_count = _first_nonnegative_int(
        request.get("prefill_mla_kv_b_cache_file_count"),
        request_check.get("prefill_mla_kv_b_cache_file_count"),
    )
    replay_mla_kv_b_cache_expected_total_bytes = _first_nonnegative_int(
        request.get("prefill_mla_kv_b_cache_total_bytes"),
        request_check.get("prefill_mla_kv_b_cache_total_bytes"),
    )
    profile_sha256 = applied.get("sha256")
    prepared_manifest_matches = (
        current_prepared_manifest == prepared_manifest
        if isinstance(current_prepared_manifest, str)
        and isinstance(prepared_manifest, str)
        else None
    )
    locked = applied.get("locked") is True
    lock_required = applied.get("lock_required") is True
    matches_prepared = applied.get("matches_prepared") is True
    argv_safe_to_replay = applied.get("argv_safe_to_replay") is True
    prompt_token_ids = _result_prompt_token_ids(request.get("prompt_token_ids"))
    if prompt_token_ids is None and not request:
        prompt_token_ids = _result_prompt_token_ids(
            token_result.get("prompt_token_ids")
        )
    if prompt_token_ids is None and not request:
        prompt_token_ids = _result_prompt_token_ids(
            request_check.get("prompt_token_ids")
        )
    generated_token_ids = _as_list(token_result.get("generated_token_ids"))
    max_new_tokens = _first_nonnegative_int(
        request.get("max_new_tokens"),
        request_check.get("max_new_tokens"),
        len(generated_token_ids) if generated_token_ids else None,
    )
    prepared_dir = (
        str(Path(prepared_manifest).parent)
        if isinstance(prepared_manifest, str) and prepared_manifest
        else None
    )
    prepared_manifest_exists = _path_is_file(prepared_manifest)
    launch_profile_exists = _path_is_file(launch_profile_path)
    launch_audit_exists = _path_is_file(launch_audit_path)
    replay_mla_kv_b_cache_dir_exists = (
        _path_is_dir(replay_mla_kv_b_cache_dir)
        if replay_mla_kv_b_cache_dir is not None
        else None
    )
    replay_mla_kv_b_cache_stats = (
        _directory_file_stats(replay_mla_kv_b_cache_dir)
        if replay_mla_kv_b_cache_dir_exists
        else None
    )
    replay_mla_kv_b_cache_current_file_count = (
        replay_mla_kv_b_cache_stats.get("file_count")
        if replay_mla_kv_b_cache_stats is not None
        else None
    )
    replay_mla_kv_b_cache_current_total_bytes = (
        replay_mla_kv_b_cache_stats.get("total_bytes")
        if replay_mla_kv_b_cache_stats is not None
        else None
    )
    replay_mla_kv_b_cache_ready = (
        None
        if replay_mla_kv_b_cache_dir is None
        else (
            replay_mla_kv_b_cache_dir_exists is True
            and (
                replay_mla_kv_b_cache_expected_file_count is None
                or (
                    replay_mla_kv_b_cache_current_file_count is not None
                    and replay_mla_kv_b_cache_current_file_count
                    >= replay_mla_kv_b_cache_expected_file_count
                )
            )
            and (
                replay_mla_kv_b_cache_expected_total_bytes is None
                or (
                    replay_mla_kv_b_cache_current_total_bytes is not None
                    and replay_mla_kv_b_cache_current_total_bytes
                    >= replay_mla_kv_b_cache_expected_total_bytes
                )
            )
        )
    )
    launch_profile_file_sha256 = _file_sha256(launch_profile_path)
    profile_sha256_matches_file = (
        launch_profile_file_sha256 == profile_sha256
        if isinstance(profile_sha256, str)
        and profile_sha256
        and launch_profile_file_sha256 is not None
        else None
    )
    launch_audit_payload = (
        _json_object_file(launch_audit_path) if launch_audit_exists else None
    )
    launch_audit_json_valid = launch_audit_payload is not None
    launch_audit_root = _as_mapping(
        launch_audit_payload.get("launch_audit")
        if launch_audit_payload is not None
        else None
    )
    launch_audit_applied = _as_mapping(
        launch_audit_payload.get("applied_launch_profile")
        if launch_audit_payload is not None
        else None
    )
    if not launch_audit_applied:
        launch_audit_applied = _as_mapping(
            launch_audit_root.get("applied_launch_profile")
        )
    launch_audit_schema = (
        launch_audit_payload.get("schema")
        if launch_audit_payload is not None
        and isinstance(launch_audit_payload.get("schema"), str)
        else None
    )
    launch_audit_ok = (
        launch_audit_root.get("ok") is True
        if launch_audit_payload is not None
        else None
    )
    launch_audit_profile_path = _first_nonempty_str(
        launch_audit_applied.get("path"),
        launch_audit_payload.get("applied_launch_profile_path")
        if launch_audit_payload is not None
        else None,
        launch_audit_root.get("applied_launch_profile_path"),
        launch_audit_payload.get("launch_profile_path")
        if launch_audit_payload is not None
        else None,
        launch_audit_root.get("launch_profile_path"),
    )
    launch_audit_profile_sha256 = _first_nonempty_str(
        launch_audit_applied.get("sha256"),
        launch_audit_payload.get("applied_launch_profile_sha256")
        if launch_audit_payload is not None
        else None,
        launch_audit_root.get("applied_launch_profile_sha256"),
        launch_audit_payload.get("launch_profile_sha256")
        if launch_audit_payload is not None
        else None,
        launch_audit_root.get("launch_profile_sha256"),
    )
    launch_audit_profile_path_matches = _paths_refer_to_same_file(
        launch_audit_profile_path, launch_profile_path
    )
    launch_audit_profile_sha256_matches = (
        launch_audit_profile_sha256 == profile_sha256
        if launch_audit_profile_sha256 is not None
        and isinstance(profile_sha256, str)
        and bool(profile_sha256)
        else None
    )
    launch_audit_binding_matches = (
        launch_audit_schema == "largerlm.launch_audit.v1"
        and launch_audit_ok is True
        and launch_audit_profile_path_matches is True
        and launch_audit_profile_sha256_matches is True
    )
    safe_to_replay = (
        isinstance(launch_profile_path, str)
        and bool(launch_profile_path)
        and locked
        and lock_required
        and matches_prepared
        and argv_safe_to_replay
        and prepared_manifest_matches is not False
    )
    replay_ready = (
        safe_to_replay
        and isinstance(launch_audit_path, str)
        and bool(launch_audit_path)
        and isinstance(prepared_dir, str)
        and bool(prepared_dir)
        and prompt_token_ids is not None
        and max_new_tokens is not None
    )
    replay_files_ready = (
        replay_ready
        and prepared_manifest_exists
        and launch_profile_exists
        and launch_audit_exists
        and profile_sha256_matches_file is True
        and launch_audit_binding_matches
        and replay_mla_kv_b_cache_ready is not False
    )
    if replay_ready:
        replay_argv = [
            "python",
            "-m",
            "largerlm",
            "generate-prepared-token-ids",
            "--apply-launch-profile",
            launch_profile_path,
            "--lock-launch-profile",
            "--require-locked-launch-profile",
            "--require-launch-audit",
            launch_audit_path,
            prepared_dir,
        ]
        if replay_mla_kv_b_cache_dir is not None:
            replay_argv.extend(
                ["--prefill-mla-kv-b-cache-dir", replay_mla_kv_b_cache_dir]
            )
        replay_argv.extend(
            [
                "--prompt-token-ids",
                ",".join(str(token_id) for token_id in prompt_token_ids or ()),
                "--max-new-tokens",
                str(max_new_tokens),
            ]
        )
    else:
        replay_argv = None
    return {
        "prepared_manifest": (
            prepared_manifest if isinstance(prepared_manifest, str) else None
        ),
        "prepared_dir": prepared_dir,
        "prepared_manifest_exists": prepared_manifest_exists,
        "current_prepared_manifest": (
            current_prepared_manifest
            if isinstance(current_prepared_manifest, str)
            else None
        ),
        "prepared_manifest_matches": prepared_manifest_matches,
        "launch_profile_path": (
            launch_profile_path if isinstance(launch_profile_path, str) else None
        ),
        "launch_profile_exists": launch_profile_exists,
        "launch_profile_file_sha256": launch_profile_file_sha256,
        "launch_profile_sha256_matches_file": profile_sha256_matches_file,
        "launch_audit_path": (
            launch_audit_path if isinstance(launch_audit_path, str) else None
        ),
        "launch_audit_exists": launch_audit_exists,
        "launch_audit_json_valid": launch_audit_json_valid,
        "launch_audit_schema": launch_audit_schema,
        "launch_audit_ok": launch_audit_ok,
        "launch_audit_profile_path": launch_audit_profile_path,
        "launch_audit_profile_path_matches": launch_audit_profile_path_matches,
        "launch_audit_profile_sha256": launch_audit_profile_sha256,
        "launch_audit_profile_sha256_matches": launch_audit_profile_sha256_matches,
        "launch_audit_binding_matches": launch_audit_binding_matches,
        "applied_launch_profile_sha256": (
            profile_sha256 if isinstance(profile_sha256, str) else None
        ),
        "applied_launch_profile_locked": locked,
        "applied_launch_profile_lock_required": lock_required,
        "applied_launch_profile_matches_prepared": matches_prepared,
        "applied_launch_profile_argv_safe_to_replay": argv_safe_to_replay,
        "safe_to_replay": safe_to_replay,
        "replay_ready": replay_ready,
        "replay_files_ready": replay_files_ready,
        "replay_prompt_token_count": len(prompt_token_ids or ()),
        "replay_max_new_tokens": max_new_tokens,
        "replay_mla_kv_b_cache_dir": replay_mla_kv_b_cache_dir,
        "replay_mla_kv_b_cache_dir_exists": replay_mla_kv_b_cache_dir_exists,
        "replay_mla_kv_b_cache_expected_file_count": (
            replay_mla_kv_b_cache_expected_file_count
        ),
        "replay_mla_kv_b_cache_current_file_count": (
            replay_mla_kv_b_cache_current_file_count
        ),
        "replay_mla_kv_b_cache_expected_total_bytes": (
            replay_mla_kv_b_cache_expected_total_bytes
        ),
        "replay_mla_kv_b_cache_current_total_bytes": (
            replay_mla_kv_b_cache_current_total_bytes
        ),
        "replay_mla_kv_b_cache_ready": replay_mla_kv_b_cache_ready,
        "replay_generate_token_ids_argv": replay_argv,
        "replay_generate_token_ids_command": (
            shlex.join(replay_argv) if replay_argv is not None else None
        ),
    }


def _routed_moe_layers_summary(
    records: list[dict[str, Any]],
    *,
    top_limit: int,
) -> dict[str, Any]:
    elapsed = _sum_float_records(records, "elapsed_seconds")
    assignments = _sum_int_records(records, "assignments")
    stage_planned_read = _sum_int_records(records, "stage_planned_read_bytes")
    stage_plus_compact = _sum_int_records(records, "stage_plus_compact_bytes")
    stage_plus_compact_materialized = _sum_int_records(
        records,
        "stage_plus_compact_materialized_bytes",
    )
    stage_plus_compact_for_share = (
        stage_plus_compact_materialized
        if stage_plus_compact_materialized is not None
        else stage_plus_compact
    )
    static_used = _sum_int_records(records, "static_capacity_used_slots")
    static_total = _sum_int_records(records, "static_capacity_total_slots")
    stage_copy_elapsed = _sum_float_records(records, "stage_copy_elapsed_seconds")
    router_min_effective_margin = _min_float_records(
        records,
        "router_min_effective_score_margin",
    )
    router_min_topk_margin = _min_float_records(records, "router_min_topk_score_margin")
    router_min_group_margin = _min_float_records(
        records,
        "router_min_group_score_margin",
    )
    router_margin_observed_records = [
        record
        for record in records
        if _as_mapping(record.get("router_margin_summary"))
    ]
    router_near_tie_counts: dict[str, int] = {}
    for record in router_margin_observed_records:
        margin = _as_mapping(record.get("router_margin_summary"))
        counts = _as_mapping(margin.get("effective_near_tie_counts"))
        for key, value in counts.items():
            parsed = _nonnegative_int(value)
            if parsed is not None:
                router_near_tie_counts[str(key)] = (
                    router_near_tie_counts.get(str(key), 0) + parsed
                )
    router_policy_records = [
        _as_mapping(_as_mapping(record.get("router_margin_summary")).get("router_gate_policy"))
        for record in router_margin_observed_records
        if _as_mapping(
            _as_mapping(record.get("router_margin_summary")).get("router_gate_policy")
        )
    ]
    router_policy_summary: dict[str, Any] | None = None
    if router_policy_records:
        decisions: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        thresholds: list[float] = []
        custom_elapsed_values: list[float] = []
        fallback_elapsed_values: list[float] = []
        extra_custom_probe_values: list[float] = []
        command_count = 0
        for policy in router_policy_records:
            decision = policy.get("decision")
            if isinstance(decision, str) and decision:
                decisions[decision] += 1
            mode = policy.get("mode")
            if isinstance(mode, str) and mode:
                modes[mode] += 1
            threshold = _finite_float(policy.get("margin_threshold"))
            if threshold is not None:
                thresholds.append(threshold)
            commands = _nonnegative_int(policy.get("command_count"))
            if commands is not None:
                command_count += commands
            custom_elapsed = _finite_float(policy.get("custom_elapsed_seconds"))
            if custom_elapsed is not None:
                custom_elapsed_values.append(custom_elapsed)
                if decision == "mpsgraph-f32-fallback":
                    extra_custom_probe_values.append(custom_elapsed)
            fallback_elapsed = _finite_float(policy.get("fallback_elapsed_seconds"))
            if fallback_elapsed is not None:
                fallback_elapsed_values.append(fallback_elapsed)
        custom_elapsed_total = (
            sum(custom_elapsed_values) if custom_elapsed_values else None
        )
        fallback_elapsed_total = (
            sum(fallback_elapsed_values) if fallback_elapsed_values else None
        )
        router_policy_elapsed_total = (
            (custom_elapsed_total or 0.0) + (fallback_elapsed_total or 0.0)
            if custom_elapsed_values or fallback_elapsed_values
            else None
        )
        router_policy_summary = {
            "layer_count": len(router_policy_records),
            "decision_counts": dict(sorted(decisions.items())) or None,
            "mode_counts": dict(sorted(modes.items())) or None,
            "margin_threshold": (
                thresholds[0]
                if thresholds and all(value == thresholds[0] for value in thresholds)
                else None
            ),
            "margin_threshold_min": min(thresholds) if thresholds else None,
            "margin_threshold_max": max(thresholds) if thresholds else None,
            "command_count": command_count or None,
            "custom_elapsed_seconds": custom_elapsed_total,
            "fallback_elapsed_seconds": fallback_elapsed_total,
            "total_elapsed_seconds": router_policy_elapsed_total,
            "extra_custom_probe_elapsed_seconds": (
                sum(extra_custom_probe_values) if extra_custom_probe_values else None
            ),
        }
    moe_timing_elapsed: dict[str, float] = {}
    for phase, key in _ROUTED_MOE_TIMING_FIELDS:
        phase_elapsed = _sum_float_records(records, key)
        if phase_elapsed is not None:
            moe_timing_elapsed[phase] = phase_elapsed
    moe_wall_elapsed: dict[str, float] = {}
    for phase, key in (
        ("total", "wall_total_elapsed_seconds"),
        ("compact_stage", "wall_compact_stage_elapsed_seconds"),
        ("routes", "wall_routes_elapsed_seconds"),
        ("static_capacity", "wall_static_capacity_elapsed_seconds"),
        ("runner", "wall_runner_elapsed_seconds"),
        ("output_validation", "wall_output_validation_elapsed_seconds"),
    ):
        phase_elapsed = _sum_float_records(records, key)
        if phase_elapsed is not None:
            moe_wall_elapsed[phase] = phase_elapsed
    moe_wall_fraction = (
        {
            phase: phase_elapsed / elapsed
            for phase, phase_elapsed in moe_wall_elapsed.items()
            if elapsed is not None and elapsed > 0.0
        }
        if elapsed is not None and elapsed > 0.0
        else {}
    )
    moe_wall_total_elapsed = moe_wall_elapsed.get("total")
    moe_wall_residual_elapsed = (
        max(0.0, elapsed - moe_wall_total_elapsed)
        if (
            elapsed is not None
            and moe_wall_total_elapsed is not None
            and elapsed >= 0.0
            and moe_wall_total_elapsed >= 0.0
        )
        else None
    )
    moe_wall_residual_fraction = (
        moe_wall_residual_elapsed / elapsed
        if (
            moe_wall_residual_elapsed is not None
            and elapsed is not None
            and elapsed > 0.0
        )
        else None
    )
    moe_non_runner_elapsed = _sum_float_records(
        records,
        "moe_non_runner_elapsed_seconds",
    )
    moe_non_runner_fraction = (
        moe_non_runner_elapsed / elapsed
        if (
            moe_non_runner_elapsed is not None
            and elapsed is not None
            and elapsed > 0.0
        )
        else None
    )
    moe_runner_total_elapsed = moe_timing_elapsed.get("total")
    moe_runner_total_fraction = (
        moe_runner_total_elapsed / elapsed
        if (
            moe_runner_total_elapsed is not None
            and elapsed is not None
            and elapsed > 0.0
        )
        else None
    )
    moe_timing_fraction = (
        {
            phase: phase_elapsed / elapsed
            for phase, phase_elapsed in moe_timing_elapsed.items()
            if elapsed is not None and elapsed > 0.0
        }
        if elapsed is not None and elapsed > 0.0
        else {}
    )
    moe_timing_top_phase = (
        max(
            (
                (phase, seconds)
                for phase, seconds in moe_timing_elapsed.items()
                if phase != "total"
            ),
            key=lambda item: item[1],
            default=None,
        )
        if moe_timing_elapsed
        else None
    )
    moe_timing_top_phase_fraction = (
        moe_timing_top_phase[1] / elapsed
        if moe_timing_top_phase is not None
        and elapsed is not None
        and elapsed > 0.0
        else None
    )
    mxfp4_swiglu_kernel_elapsed = moe_timing_elapsed.get("mxfp4_swiglu_kernel")
    mxfp4_down_add_kernel_elapsed = moe_timing_elapsed.get("mxfp4_down_add_kernel")
    mxfp4_kernel_elapsed = moe_timing_elapsed.get("kernel")
    mxfp4_split_elapsed = {
        "swiglu": mxfp4_swiglu_kernel_elapsed,
        "down_add": mxfp4_down_add_kernel_elapsed,
    }
    mxfp4_split_elapsed = {
        phase: seconds
        for phase, seconds in mxfp4_split_elapsed.items()
        if seconds is not None
    }
    mxfp4_split_total = (
        sum(mxfp4_split_elapsed.values()) if mxfp4_split_elapsed else None
    )
    mxfp4_split_fraction = (
        {
            phase: seconds / mxfp4_kernel_elapsed
            for phase, seconds in mxfp4_split_elapsed.items()
        }
        if (
            mxfp4_split_elapsed
            and mxfp4_kernel_elapsed is not None
            and mxfp4_kernel_elapsed > 0.0
        )
        else {}
    )
    if (
        mxfp4_split_total is not None
        and mxfp4_kernel_elapsed is not None
        and mxfp4_kernel_elapsed > 0.0
    ):
        mxfp4_split_fraction["total"] = mxfp4_split_total / mxfp4_kernel_elapsed
    mxfp4_split_top_phase = (
        max(mxfp4_split_elapsed.items(), key=lambda item: item[1])
        if mxfp4_split_elapsed
        else None
    )
    mxfp4_split_top_phase_fraction = (
        mxfp4_split_fraction.get(mxfp4_split_top_phase[0])
        if mxfp4_split_top_phase is not None
        else None
    )
    token_block_limited_records = [
        record
        for record in records
        if (
            (effective := _nonnegative_int(record.get("effective_moe_token_block")))
            is not None
            and (max_tokens := _nonnegative_int(record.get("moe_max_expert_tokens")))
            is not None
            and effective < max_tokens
        )
    ]
    token_block_observed_records = [
        record
        for record in records
        if _nonnegative_int(record.get("effective_moe_token_block")) is not None
        and _nonnegative_int(record.get("moe_max_expert_tokens")) is not None
    ]
    copy_to_routed_ratio = (
        stage_copy_elapsed / elapsed
        if stage_copy_elapsed is not None and elapsed is not None and elapsed > 0.0
        else None
    )
    stage_copy_throughput = (
        (stage_planned_read / 1024**3) / stage_copy_elapsed
        if stage_planned_read is not None
        and stage_copy_elapsed is not None
        and stage_copy_elapsed > 0.0
        else None
    )
    max_copy_chunk_bytes = _max_int_records(records, "copy_chunk_bytes")
    top_stage = _top_records_by_numeric_field(
        records,
        (
            "stage_plus_compact_materialized_bytes"
            if stage_plus_compact_materialized is not None
            else "stage_plus_compact_bytes"
        ),
        limit=1,
    )
    top_stage_bytes_for_share = (
        _nonnegative_int(top_stage[0].get("stage_plus_compact_materialized_bytes"))
        if top_stage
        else None
    )
    if top_stage_bytes_for_share is None and top_stage:
        top_stage_bytes_for_share = _nonnegative_int(
            top_stage[0].get("stage_plus_compact_bytes")
        )
    top_stage_share = (
        top_stage_bytes_for_share
        / stage_plus_compact_for_share
        if top_stage
        and stage_plus_compact_for_share is not None
        and stage_plus_compact_for_share > 0
        and top_stage_bytes_for_share is not None
        else None
    )
    notes: list[str] = []
    if token_block_observed_records and not token_block_limited_records:
        notes.append(
            "effective token blocks already match per-expert token fanout; "
            "larger --prefill-moe-token-block is unlikely to help this run"
        )
    elif token_block_limited_records:
        notes.append(
            "some routed MoE layers used a token block below per-expert fanout; "
            "inspect scratch limits before increasing --prefill-moe-token-block"
        )
    if copy_to_routed_ratio is not None and copy_to_routed_ratio >= 0.35:
        notes.append(
            "expert staging copy time is significant relative to routed runner time; "
            "stage I/O, compact staging, or copy avoidance may be worth testing"
        )
    if top_stage_share is not None and top_stage_share >= 0.05:
        notes.append(
            "a single routed layer carries a noticeable share of stage+compact "
            "bytes; include it in replay and kernel-tuning targets"
        )
    kernel_ratio = moe_timing_fraction.get("kernel")
    if kernel_ratio is not None and kernel_ratio >= 0.50:
        notes.append(
            "routed MoE runner timing is kernel dominated; prioritize Metal "
            "dequant/matvec tiling before more SSD staging work"
        )
    if (
        mxfp4_split_top_phase is not None
        and mxfp4_split_top_phase_fraction is not None
        and mxfp4_split_top_phase_fraction >= 0.55
    ):
        if mxfp4_split_top_phase[0] == "swiglu":
            notes.append(
                "MXFP4 split timing is dominated by the gate/up/SwiGLU side; "
                "prioritize routed expert dot-product scheduling before "
                "down/add accumulator work"
            )
        elif mxfp4_split_top_phase[0] == "down_add":
            notes.append(
                "MXFP4 split timing is dominated by down/add; prioritize the "
                "route-weighted output projection and accumulator path"
            )
    routed_io_ratio = sum(
        moe_timing_fraction.get(phase, 0.0)
        for phase in (
            "expert_read",
            "input_read",
            "output_read",
            "output_write",
            "final_read",
        )
    )
    if routed_io_ratio >= 0.25:
        notes.append(
            "routed MoE runner timing has a substantial file-I/O share; inspect "
            "expert streaming and accumulator read/write paths"
        )
    if moe_non_runner_fraction is not None and moe_non_runner_fraction >= 0.40:
        notes.append(
            "routed MoE elapsed is dominated by work outside runner timing; "
            "prioritize persistent runner or cross-layer fusion before more "
            "single-kernel tuning"
        )
    if (
        moe_wall_residual_fraction is not None
        and moe_wall_residual_fraction >= 0.15
    ):
        notes.append(
            "routed MoE elapsed has a large residual outside the recorded wall "
            "phase split; instrument outer Python/file orchestration before "
            "promoting more local kernel changes"
        )
    output_accum_ratio = (
        moe_timing_fraction.get("output_read", 0.0)
        + moe_timing_fraction.get("output_write", 0.0)
    )
    output_accum_seconds = (
        moe_timing_elapsed.get("output_read", 0.0)
        + moe_timing_elapsed.get("output_write", 0.0)
    )
    if output_accum_ratio >= 0.15:
        notes.append(
            "output accumulator read/write time is noticeable; an in-memory or "
            "GPU-side accumulator path may be worth prototyping under memory caps"
        )
    accumulator_counts = Counter(
        value
        for record in records
        if isinstance((value := record.get("moe_output_accumulator")), str)
        and value
    )
    top_slowest_layers = _top_records_by_numeric_field(
        records,
        "elapsed_seconds",
        limit=top_limit,
    )
    top_kernel_layer = (
        top_slowest_layers[0] if top_slowest_layers else {}
    )
    suggested_stage_copy_experiments: list[dict[str, object]] = []
    if (
        copy_to_routed_ratio is not None
        and copy_to_routed_ratio >= 0.35
        and stage_copy_throughput is not None
        and stage_copy_throughput < 5.0
        and max_copy_chunk_bytes is not None
        and max_copy_chunk_bytes <= 64 * 1024 * 1024
    ):
        candidate_mib = (32, 64) if max_copy_chunk_bytes <= 16 * 1024 * 1024 else (128,)
        for mib in candidate_mib:
            bytes_value = mib * 1024 * 1024
            if bytes_value > max_copy_chunk_bytes:
                suggested_stage_copy_experiments.append(
                    {
                        "argv": ["--prefill-copy-chunk-mib", str(mib)],
                        "requires_bakeoff": True,
                        "promotion_gate": "result-bakeoff-total-latency-win",
                        "reason": (
                            "stage copy is a large share of runtime and current "
                            "copy chunks are small; rerun under the same memory "
                            "guards to test SSD throughput, then promote only "
                            "if result-bakeoff shows a total-latency win"
                        ),
                    }
                )
    suggested_output_accumulator_experiments: list[dict[str, object]] = []
    if (
        accumulator_counts.get("file", 0) > 0
        and output_accum_seconds >= 0.25
        and output_accum_ratio >= 0.01
    ):
        suggested_output_accumulator_experiments.append(
            {
                "env": {"LARGERLM_MOE_BATCH_ACCUMULATOR": "memory"},
                "requires_bakeoff": True,
                "promotion_gate": "result-bakeoff-total-latency-win",
                "reason": (
                    "file-backed output accumulator I/O is measurable; rerun "
                    "under the same launch and memory guards with the opt-in "
                    "memory accumulator, then promote only if a paired result "
                    "bakeoff shows a total-latency win"
                ),
            }
        )
    suggested_moe_kernel_experiments: list[dict[str, object]] = []
    if (
        mxfp4_split_top_phase is not None
        and mxfp4_split_top_phase_fraction is not None
        and mxfp4_split_top_phase_fraction >= 0.55
    ):
        kernel_argv = [
            "python",
            "scripts/glm_moe_tile_sweep.py",
            "--repeat",
            "6",
            "--order",
            "interleave",
        ]
        layer = _nonnegative_int(top_kernel_layer.get("layer"))
        batch_tokens = _nonnegative_int(top_kernel_layer.get("batch_tokens"))
        selected_experts = _as_list(top_kernel_layer.get("selected_experts"))
        experts = [
            str(expert)
            for expert in selected_experts
            if _nonnegative_int(expert) is not None
        ]
        if layer is not None:
            kernel_argv.extend(["--layer", str(layer)])
        if batch_tokens is not None:
            kernel_argv.extend(["--batch-tokens", str(batch_tokens)])
        if experts:
            kernel_argv.extend(["--experts", ",".join(experts)])
        if mxfp4_split_top_phase[0] == "swiglu":
            kernel_argv.extend(
                [
                    "--tiles",
                    "1",
                    "--vector-swiglu-modes",
                    "off,on",
                    "--group32-modes",
                    "auto,off",
                ]
            )
            reason = (
                "split timing says the gate/up/SwiGLU side dominates; run a "
                "bounded vector-SwiGLU A/B against the default group32/auto "
                "path on the hottest routed layer, then promote only after "
                "numerical agreement and a full replay bakeoff win"
            )
        else:
            kernel_argv.extend(
                [
                    "--tiles",
                    "1,2,4",
                    "--group32-modes",
                    "off,auto",
                ]
            )
            reason = (
                "split timing says the down/add side dominates; run a bounded "
                "tile/group32 A/B on the hottest routed layer, then promote "
                "only after numerical agreement and a full replay bakeoff win"
            )
        suggested_moe_kernel_experiments.append(
            {
                "env": {"LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING": "1"},
                "argv": kernel_argv,
                "scope": "bounded-layer-microbench",
                "requires_bakeoff": True,
                "promotion_gate": (
                    "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
                ),
                "reason": reason,
            }
        )
    return {
        "count": len(records),
        "elapsed_seconds": elapsed,
        "total_assignments": assignments,
        "total_selected_expert_slots": _sum_int_records(
            records,
            "selected_expert_count",
        ),
        "max_selected_expert_count": _max_int_records(
            records,
            "selected_expert_count",
        ),
        "total_stage_planned_read_bytes": stage_planned_read,
        "total_stage_staged_bytes": _sum_int_records(records, "stage_staged_bytes"),
        "total_compact_stage_bytes": _sum_int_records(
            records,
            "compact_stage_bytes",
        ),
        "total_compact_stage_materialized_bytes": _sum_int_records(
            records,
            "compact_stage_materialized_bytes",
        ),
        "total_stage_plus_compact_bytes": stage_plus_compact,
        "total_stage_plus_compact_materialized_bytes": (
            stage_plus_compact_materialized
        ),
        "total_stage_copy_elapsed_seconds": stage_copy_elapsed,
        "stage_copy_throughput_gib_per_second": stage_copy_throughput,
        "router_margin_layer_count": len(router_margin_observed_records),
        "router_min_effective_score_margin": router_min_effective_margin,
        "router_min_topk_score_margin": router_min_topk_margin,
        "router_min_group_score_margin": router_min_group_margin,
        "router_effective_near_tie_counts": router_near_tie_counts or None,
        "router_gate_policy": router_policy_summary,
        "moe_timing_elapsed_seconds": moe_timing_elapsed or None,
        "moe_timing_elapsed_fraction": moe_timing_fraction or None,
        "moe_wall_elapsed_seconds": moe_wall_elapsed or None,
        "moe_wall_elapsed_fraction": moe_wall_fraction or None,
        "moe_wall_residual_elapsed_seconds": moe_wall_residual_elapsed,
        "moe_wall_residual_elapsed_fraction": moe_wall_residual_fraction,
        "moe_runner_total_elapsed_seconds": moe_runner_total_elapsed,
        "moe_runner_total_elapsed_fraction": moe_runner_total_fraction,
        "moe_non_runner_elapsed_seconds": moe_non_runner_elapsed,
        "moe_non_runner_elapsed_fraction": moe_non_runner_fraction,
        "moe_timing_top_phase": (
            moe_timing_top_phase[0] if moe_timing_top_phase is not None else None
        ),
        "moe_timing_top_phase_seconds": (
            moe_timing_top_phase[1] if moe_timing_top_phase is not None else None
        ),
        "moe_timing_top_phase_fraction": moe_timing_top_phase_fraction,
        "moe_mxfp4_kernel_split_elapsed_seconds": (
            (
                mxfp4_split_elapsed
                | {"total": mxfp4_split_total}
            )
            if mxfp4_split_total is not None
            else None
        ),
        "moe_mxfp4_kernel_split_fraction": mxfp4_split_fraction or None,
        "moe_mxfp4_kernel_split_top_phase": (
            mxfp4_split_top_phase[0]
            if mxfp4_split_top_phase is not None
            else None
        ),
        "moe_mxfp4_kernel_split_top_phase_seconds": (
            mxfp4_split_top_phase[1]
            if mxfp4_split_top_phase is not None
            else None
        ),
        "moe_mxfp4_kernel_split_top_phase_fraction": (
            mxfp4_split_top_phase_fraction
        ),
        "output_accumulator_counts": dict(accumulator_counts) or None,
        "max_copy_chunk_bytes": max_copy_chunk_bytes,
        "total_static_capacity_used_slots": static_used,
        "total_static_capacity_slots": static_total,
        "static_capacity_utilization": (
            static_used / static_total
            if static_used is not None and static_total is not None and static_total > 0
            else None
        ),
        "max_effective_moe_token_block": _max_int_records(
            records,
            "effective_moe_token_block",
        ),
        "max_moe_batch_buffer_bytes": _max_int_records(
            records,
            "moe_batch_buffer_bytes",
        ),
        "max_moe_estimated_peak_bytes": _max_int_records(
            records,
            "moe_estimated_peak_bytes",
        ),
        "top_slowest_layers": top_slowest_layers,
        "top_stage_plus_compact_layers": _top_records_by_numeric_field(
            records,
            "stage_plus_compact_bytes",
            limit=top_limit,
        ),
        "top_stage_plus_compact_materialized_layers": (
            _top_records_by_numeric_field(
                records,
                "stage_plus_compact_materialized_bytes",
                limit=top_limit,
            )
        ),
        "bottleneck_hints": {
            "token_block_observed_layer_count": len(token_block_observed_records),
            "token_block_limited_layer_count": len(token_block_limited_records),
            "token_block_status": (
                "not_observed"
                if not token_block_observed_records
                else (
                    "limited"
                    if token_block_limited_records
                    else "saturates_expert_fanout"
                )
            ),
            "stage_copy_to_routed_elapsed_ratio": copy_to_routed_ratio,
            "stage_copy_throughput_gib_per_second": stage_copy_throughput,
            "moe_timing_top_phase": (
                moe_timing_top_phase[0] if moe_timing_top_phase is not None else None
            ),
            "moe_timing_top_phase_seconds": (
                moe_timing_top_phase[1] if moe_timing_top_phase is not None else None
            ),
            "moe_timing_top_phase_fraction": moe_timing_top_phase_fraction,
            "moe_runner_total_elapsed_fraction": moe_runner_total_fraction,
            "moe_non_runner_elapsed_seconds": moe_non_runner_elapsed,
            "moe_non_runner_elapsed_fraction": moe_non_runner_fraction,
            "moe_wall_residual_elapsed_seconds": moe_wall_residual_elapsed,
            "moe_wall_residual_elapsed_fraction": moe_wall_residual_fraction,
            "moe_mxfp4_kernel_split_top_phase": (
                mxfp4_split_top_phase[0]
                if mxfp4_split_top_phase is not None
                else None
            ),
            "moe_mxfp4_kernel_split_top_phase_fraction": (
                mxfp4_split_top_phase_fraction
            ),
            "moe_timing_io_elapsed_fraction": routed_io_ratio
            if moe_timing_fraction
            else None,
            "moe_timing_output_accumulator_elapsed_fraction": output_accum_ratio
            if moe_timing_fraction
            else None,
            "top_stage_plus_compact_layer_share": top_stage_share,
            "suggested_stage_copy_experiments": suggested_stage_copy_experiments,
            "suggested_output_accumulator_experiments": (
                suggested_output_accumulator_experiments
            ),
            "suggested_moe_kernel_experiments": (
                suggested_moe_kernel_experiments
            ),
            "notes": notes,
        },
    }


def _prefill_stage_io_hotspot_summary_rows(
    value: object,
    *,
    limit: int = 5,
) -> list[dict[str, object]]:
    int_fields = {
        "chunk_index",
        "layer",
        "tile_index",
        "batch_tokens",
        "selected_expert_count",
        "total_assignments",
        "raw_range_count",
        "coalesced_range_count",
        "planned_read_bytes",
        "staged_bytes",
        "unique_requested_bytes",
        "waste_bytes",
        "copy_chunk_bytes",
        "copy_read_calls",
        "copy_write_calls",
    }
    float_fields = {
        "copy_elapsed_seconds",
        "copy_throughput_gib_per_second",
        "copy_average_read_bytes",
        "stage_budget_utilization",
        "unique_read_amplification",
    }
    rows: list[dict[str, object]] = []
    for item in _as_list(value):
        item_map = _as_mapping(item)
        if not item_map:
            continue
        row: dict[str, object] = {}
        for field in int_fields:
            parsed = _nonnegative_int(item_map.get(field))
            if parsed is not None:
                row[field] = parsed
        selected_experts = _nonnegative_int_list(item_map.get("selected_experts"))
        if selected_experts is not None:
            row["selected_experts"] = selected_experts
        for field in float_fields:
            parsed_float = _finite_float(item_map.get(field))
            if parsed_float is not None:
                row[field] = parsed_float
        if row:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def _prefill_actual_stage_io_summary(
    actual_read_time: dict[str, Any],
) -> dict[str, object]:
    rich_fields = {
        "expert_stage_io_stage_count",
        "expert_stage_copy_hotspots",
        "expert_stage_range_hotspots",
        "total_expert_stage_serial_read_bytes",
        "total_expert_stage_unique_requested_bytes",
        "total_expert_stage_waste_bytes",
        "total_expert_stage_coalesced_savings_bytes",
        "total_expert_stage_read_advice_attempted_ranges",
        "total_expert_stage_read_advice_calls",
        "total_expert_stage_read_advice_bytes",
        "total_expert_stage_read_advice_failures",
        "total_expert_stage_copy_read_calls",
        "total_expert_stage_copy_write_calls",
        "total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib",
        "total_expert_stage_assignment_read_amplification",
        "total_expert_stage_unique_read_amplification",
        "max_expert_stage_unique_read_amplification",
        "max_expert_stage_stage_budget_utilization",
        "total_expert_stage_copy_average_read_bytes",
        "total_expert_stage_copy_average_write_bytes",
    }
    if not actual_read_time or not any(
        field in actual_read_time for field in rich_fields
    ):
        return {}
    summary: dict[str, object] = {}
    source = actual_read_time.get("source")
    if isinstance(source, str):
        summary["source"] = source
    int_fields = {
        "expert_stage_io_stage_count": "stage_count",
        "total_expert_stage_serial_read_bytes": "serial_read_bytes",
        "total_expert_stage_unique_requested_bytes": "unique_requested_bytes",
        "total_expert_stage_planned_read_bytes": "planned_read_bytes",
        "total_expert_stage_waste_bytes": "waste_bytes",
        "total_expert_stage_coalesced_savings_bytes": "coalesced_savings_bytes",
        "total_expert_stage_raw_ranges": "raw_ranges",
        "total_expert_stage_coalesced_ranges": "coalesced_ranges",
        "total_expert_stage_read_advice_attempted_ranges": (
            "read_advice_attempted_ranges"
        ),
        "total_expert_stage_read_advice_calls": "read_advice_calls",
        "total_expert_stage_read_advice_bytes": "read_advice_bytes",
        "total_expert_stage_read_advice_failures": "read_advice_failures",
        "total_expert_stage_copy_read_calls": "copy_read_calls",
        "total_expert_stage_copy_write_calls": "copy_write_calls",
    }
    for source_field, target_field in int_fields.items():
        value = _nonnegative_int(actual_read_time.get(source_field))
        if value is not None:
            summary[target_field] = value
    counterfactuals: dict[str, int] = {}
    for key, value in _as_mapping(
        actual_read_time.get(
            "total_expert_stage_copy_read_call_counterfactuals_by_chunk_mib"
        )
    ).items():
        parsed = _nonnegative_int(value)
        if parsed is not None:
            counterfactuals[str(key)] = parsed
    if counterfactuals:
        summary["copy_read_call_counterfactuals_by_chunk_mib"] = dict(
            sorted(counterfactuals.items(), key=lambda item: int(item[0]))
        )
    float_fields = {
        "total_expert_stage_assignment_read_amplification": (
            "assignment_read_amplification"
        ),
        "total_expert_stage_unique_read_amplification": "unique_read_amplification",
        "max_expert_stage_unique_read_amplification": (
            "max_unique_read_amplification"
        ),
        "max_expert_stage_stage_budget_utilization": "max_stage_budget_utilization",
        "total_expert_stage_copy_average_read_bytes": "copy_average_read_bytes",
        "total_expert_stage_copy_average_write_bytes": "copy_average_write_bytes",
    }
    for source_field, target_field in float_fields.items():
        value = _finite_float(actual_read_time.get(source_field))
        if value is not None:
            summary[target_field] = value
    copy_hotspots = _prefill_stage_io_hotspot_summary_rows(
        actual_read_time.get("expert_stage_copy_hotspots")
    )
    if copy_hotspots:
        summary["copy_hotspots"] = copy_hotspots
    range_hotspots = _prefill_stage_io_hotspot_summary_rows(
        actual_read_time.get("expert_stage_range_hotspots")
    )
    if range_hotspots:
        summary["range_hotspots"] = range_hotspots
    return summary


def summarize_result_payload(
    payload: dict[str, Any],
    *,
    source: str | None = None,
    top_limit: int = 12,
    experiment_result_base: str | Path | None = None,
) -> dict[str, Any]:
    payload, wrapper = _http_result_wrapper_payload(payload)
    token_result = _primary_token_result(payload) or payload
    request = _as_mapping(payload.get("request"))
    prompt_prefill = _as_mapping(token_result.get("prompt_prefill"))
    actual_read_time = _as_mapping(token_result.get("prefill_actual_read_time"))
    actual_coverage = _as_mapping(
        token_result.get("prefill_actual_acceleration_coverage")
    ) or _as_mapping(prompt_prefill.get("prefill_acceleration_coverage"))
    actual_frontier = _as_mapping(
        token_result.get("prefill_actual_acceleration_frontier")
    ) or _as_mapping(prompt_prefill.get("prefill_acceleration_frontier"))
    actual_linear = _as_mapping(token_result.get("prefill_actual_linear_backend"))
    steps = _as_list(token_result.get("steps"))
    generated = _as_list(token_result.get("generated_token_ids"))
    prompt_ids = _as_list(token_result.get("prompt_token_ids"))

    total_elapsed = _finite_float(token_result.get("elapsed_seconds"))
    prompt_elapsed = _finite_float(prompt_prefill.get("elapsed_seconds"))
    linear_elapsed = _as_mapping(
        actual_linear.get("linear_backend_elapsed_seconds")
    ) or _as_mapping(prompt_prefill.get("linear_backend_elapsed_seconds"))
    linear_counts = _as_mapping(actual_linear.get("linear_backend_counts")) or _as_mapping(
        prompt_prefill.get("linear_backend_counts")
    )
    linear_flops = _as_mapping(actual_linear.get("linear_backend_flops")) or _as_mapping(
        prompt_prefill.get("linear_backend_flops")
    )
    linear_tflops = _as_mapping(
        actual_linear.get("linear_backend_estimated_tflops")
    ) or _as_mapping(prompt_prefill.get("linear_backend_estimated_tflops"))
    linear_component_stats = _as_mapping(
        actual_linear.get("linear_backend_component_stats")
    ) or _as_mapping(prompt_prefill.get("linear_backend_component_stats"))
    top_linear_components = _linear_component_stat_rows(
        linear_component_stats,
        top_limit=top_limit,
    )
    linear_elapsed_total = _sum_numeric_mapping(linear_elapsed)
    logits_elapsed_total = 0.0
    step_elapsed_total = 0.0
    for step in steps:
        step_map = _as_mapping(step)
        elapsed = _finite_float(step_map.get("elapsed_seconds"))
        if elapsed is not None:
            step_elapsed_total += elapsed
        logits = _finite_float(step_map.get("logits_elapsed_seconds"))
        if logits is not None:
            logits_elapsed_total += logits
    decode_steps = _decode_steps_summary(steps, top_limit=top_limit)

    stage_copy_elapsed = _finite_float(
        prompt_prefill.get("total_expert_stage_copy_elapsed_seconds")
    )
    known_prompt_seconds = (
        prompt_elapsed
        if prompt_elapsed is not None
        else linear_elapsed_total + (stage_copy_elapsed or 0.0)
    )
    known_subphase_seconds = known_prompt_seconds + logits_elapsed_total
    unattributed = (
        max(0.0, total_elapsed - known_subphase_seconds)
        if total_elapsed is not None
        else None
    )

    elapsed_records = _collect_elapsed_records(payload)
    named_elapsed_records = _collect_named_elapsed_fields(payload)
    runner_command_records = _collect_runner_command_records(payload)
    routed_moe_layer_records = _collect_routed_moe_layer_records(prompt_prefill)
    routed_moe_layers = _routed_moe_layers_summary(
        routed_moe_layer_records,
        top_limit=top_limit,
    )
    mla_key_cache = _collect_mla_key_cache_summary(
        prompt_prefill
    ) or _compact_mla_cache_summary(
        prompt_prefill.get("mla_key_cache"),
        total_bytes_field="total_mla_key_cache_bytes",
    )
    mla_value_cache = _collect_mla_value_cache_summary(
        prompt_prefill
    ) or _compact_mla_cache_summary(
        prompt_prefill.get("mla_value_cache"),
        total_bytes_field="total_mla_value_cache_bytes",
    )
    mla_attention_layers = _collect_mla_attention_layers(
        prompt_prefill,
        top_limit=top_limit,
    )
    leaf_records = [
        record for record in elapsed_records if not _is_container_elapsed(record["path"])
    ]
    tensor_suffix_groups = _tensor_suffix_group_summary(
        leaf_records,
        top_limit=top_limit,
    )
    backend_seconds: dict[str, float] = defaultdict(float)
    backend_counts: Counter[str] = Counter()
    group_seconds: dict[str, float] = defaultdict(float)
    group_counts: Counter[str] = Counter()
    for record in leaf_records:
        elapsed = float(record["elapsed_seconds"])
        backend = record.get("backend")
        if isinstance(backend, str):
            backend_seconds[backend] += elapsed
            backend_counts[backend] += 1
        group = _coarse_elapsed_path(str(record["path"]))
        group_seconds[group] += elapsed
        group_counts[group] += 1

    top_groups = []
    for group, seconds in sorted(
        group_seconds.items(), key=lambda item: item[1], reverse=True
    )[:top_limit]:
        top_groups.append(
            {
                "path": group,
                "elapsed_seconds": seconds,
                "count": group_counts[group],
            }
        )

    top_records = [
        _elapsed_record_hint(record)
        for record in sorted(
            leaf_records, key=lambda item: float(item["elapsed_seconds"]), reverse=True
        )[:top_limit]
    ]
    field_seconds: dict[str, float] = defaultdict(float)
    field_counts: Counter[str] = Counter()
    for record in named_elapsed_records:
        field = str(record["field"])
        field_seconds[field] += float(record["elapsed_seconds"])
        field_counts[field] += 1
    top_named_fields = []
    for field, seconds in sorted(
        field_seconds.items(), key=lambda item: item[1], reverse=True
    ):
        if field in _TEXT_DUPLICATE_NAMED_ELAPSED_FIELDS:
            continue
        top_named_fields.append(
            {
                "field": field,
                "elapsed_seconds": seconds,
                "count": field_counts[field],
            }
        )
        if len(top_named_fields) >= top_limit:
            break
    filtered_named_records = [
        record
        for record in named_elapsed_records
        if record["field"] not in _TEXT_DUPLICATE_NAMED_ELAPSED_FIELDS
    ]
    top_named_records = [
        {
            "field": record["field"],
            "elapsed_seconds": record["elapsed_seconds"],
            "path": record["path"],
        }
        for record in sorted(
            filtered_named_records,
            key=lambda item: float(item["elapsed_seconds"]),
            reverse=True,
        )[:top_limit]
    ]
    field_top_records: dict[str, dict[str, object]] = {}
    for record in named_elapsed_records:
        field = str(record["field"])
        elapsed = float(record["elapsed_seconds"])
        previous = field_top_records.get(field)
        if previous is None or elapsed > float(previous.get("elapsed_seconds") or 0.0):
            field_top_records[field] = {
                "field": field,
                "elapsed_seconds": elapsed,
                "path": record["path"],
            }

    planned_read_bytes = _nonnegative_int(
        prompt_prefill.get("total_expert_stage_planned_read_bytes")
    )
    copy_throughput = _finite_float(
        prompt_prefill.get("total_expert_stage_copy_throughput_gib_per_second")
    )
    accelerated_fraction = _finite_float(
        actual_linear.get("accelerated_linear_flop_fraction")
    )
    if accelerated_fraction is None:
        accelerated_fraction = _finite_float(
            prompt_prefill.get("accelerated_linear_flop_fraction")
        )
    actual_accelerated_fraction = _finite_float(
        actual_coverage.get("accelerated_flop_fraction")
    )
    frontier_status = _prefill_acceleration_frontier_status(
        actual_coverage,
        actual_frontier,
    )
    streamed_routed_flops = _nonnegative_int(
        actual_coverage.get("streamed_routed_expert_estimated_flops")
    )
    streamed_routed_matrix_count = _nonnegative_int(
        actual_coverage.get("streamed_routed_expert_matrix_count")
    )
    accelerated_count = _nonnegative_int(
        actual_coverage.get("accelerated_matrix_count")
    )
    if accelerated_count is None:
        accelerated_count = _sum_nonnegative_backend_ints(
            linear_counts,
            backends=_PREFILL_LINEAR_ACCELERATED_BACKENDS,
        )
    accelerated_flops = _nonnegative_int(
        actual_coverage.get("accelerated_estimated_flops")
    )
    if accelerated_flops is None:
        accelerated_flops = _sum_nonnegative_backend_ints(
            linear_flops,
            backends=_PREFILL_LINEAR_ACCELERATED_BACKENDS,
        )
    total_matrix_count = _nonnegative_int(actual_coverage.get("matrix_count"))
    if total_matrix_count is None:
        total_matrix_count = _sum_nonnegative_backend_ints(linear_counts)
    total_estimated_flops = _nonnegative_int(
        actual_coverage.get("total_estimated_flops")
    )
    if total_estimated_flops is None:
        total_estimated_flops = _sum_nonnegative_backend_ints(linear_flops)
    custom_count = _nonnegative_int(
        actual_coverage.get("custom_metal_matrix_count")
    )
    if custom_count is None:
        custom_count = _nonnegative_int(linear_counts.get("custom-metal")) or 0
    unsupported_count = _nonnegative_int(
        actual_coverage.get("unsupported_mpsgraph_matrix_count")
    )
    if unsupported_count is None:
        unsupported_count = (
            _nonnegative_int(linear_counts.get("unsupported-mpsgraph")) or 0
        )
    other_count = _nonnegative_int(actual_coverage.get("other_matrix_count"))
    if other_count is None:
        other_count = max(
            0,
            total_matrix_count
            - (accelerated_count or 0)
            - custom_count
            - unsupported_count,
        )
    custom_flops = _nonnegative_int(
        actual_coverage.get("custom_metal_estimated_flops")
    )
    if custom_flops is None:
        custom_flops = _nonnegative_int(linear_flops.get("custom-metal")) or 0
    unsupported_flops = _nonnegative_int(
        actual_coverage.get("unsupported_mpsgraph_estimated_flops")
    )
    if unsupported_flops is None:
        unsupported_flops = (
            _nonnegative_int(linear_flops.get("unsupported-mpsgraph")) or 0
        )
    other_flops = _nonnegative_int(actual_coverage.get("other_estimated_flops"))
    if other_flops is None:
        other_flops = max(
            0,
            total_estimated_flops
            - (accelerated_flops or 0)
            - custom_flops
            - unsupported_flops,
        )
    router_gate_acceleration_analyzed = any(
        field in actual_coverage
        for field in (
            "router_gate_accelerated_matrix_count",
            "router_gate_accelerated_estimated_flops",
            "accelerated_router_gate_only",
        )
    ) or "moe.router_gate_proj" in linear_component_stats
    router_gate_component = _router_gate_component_acceleration(
        linear_component_stats
    )
    router_gate_accelerated_count = _nonnegative_int(
        actual_coverage.get("router_gate_accelerated_matrix_count")
    )
    if router_gate_accelerated_count is None:
        router_gate_accelerated_count = router_gate_component[
            "router_gate_accelerated_matrix_count"
        ]
    router_gate_accelerated_flops = _nonnegative_int(
        actual_coverage.get("router_gate_accelerated_estimated_flops")
    )
    if router_gate_accelerated_flops is None:
        router_gate_accelerated_flops = router_gate_component[
            "router_gate_accelerated_estimated_flops"
        ]
    router_gate_matrix_count = _nonnegative_int(
        actual_coverage.get("router_gate_matrix_count")
    )
    router_gate_estimated_flops = _nonnegative_int(
        actual_coverage.get("router_gate_estimated_flops")
    )
    non_router_accelerated_count = _nonnegative_int(
        actual_coverage.get("non_router_accelerated_matrix_count")
    )
    if non_router_accelerated_count is None and accelerated_count is not None:
        non_router_accelerated_count = max(
            0,
            accelerated_count - (router_gate_accelerated_count or 0),
        )
    non_router_accelerated_flops = _nonnegative_int(
        actual_coverage.get("non_router_accelerated_estimated_flops")
    )
    if non_router_accelerated_flops is None and accelerated_flops is not None:
        non_router_accelerated_flops = max(
            0,
            accelerated_flops - (router_gate_accelerated_flops or 0),
        )
    non_router_matrix_count = _nonnegative_int(
        actual_coverage.get("non_router_matrix_count")
    )
    if non_router_matrix_count is None:
        non_router_matrix_count = max(
            0,
            total_matrix_count - (router_gate_matrix_count or 0),
        )
    non_router_estimated_flops = _nonnegative_int(
        actual_coverage.get("non_router_estimated_flops")
    )
    if non_router_estimated_flops is None:
        non_router_estimated_flops = max(
            0,
            total_estimated_flops - (router_gate_estimated_flops or 0),
        )
    non_router_unaccelerated_count = _nonnegative_int(
        actual_coverage.get("non_router_unaccelerated_matrix_count")
    )
    if (
        non_router_unaccelerated_count is None
        and non_router_matrix_count is not None
    ):
        non_router_unaccelerated_count = max(
            0,
            non_router_matrix_count - (non_router_accelerated_count or 0),
        )
    non_router_unaccelerated_flops = _nonnegative_int(
        actual_coverage.get("non_router_unaccelerated_estimated_flops")
    )
    if (
        non_router_unaccelerated_flops is None
        and non_router_estimated_flops is not None
    ):
        non_router_unaccelerated_flops = max(
            0,
            non_router_estimated_flops - (non_router_accelerated_flops or 0),
        )
    streamed_unaccelerated_count = _nonnegative_int(
        actual_coverage.get(
            "non_router_unaccelerated_streamed_routed_expert_matrix_count"
        )
    )
    if (
        streamed_unaccelerated_count is None
        and streamed_routed_matrix_count is not None
        and non_router_unaccelerated_count is not None
    ):
        streamed_unaccelerated_count = min(
            streamed_routed_matrix_count,
            non_router_unaccelerated_count,
        )
    streamed_unaccelerated_flops = _nonnegative_int(
        actual_coverage.get(
            "non_router_unaccelerated_streamed_routed_expert_estimated_flops"
        )
    )
    if (
        streamed_unaccelerated_flops is None
        and streamed_routed_flops is not None
        and non_router_unaccelerated_flops is not None
    ):
        streamed_unaccelerated_flops = min(
            streamed_routed_flops,
            non_router_unaccelerated_flops,
        )
    non_streamed_unaccelerated_count = _nonnegative_int(
        actual_coverage.get("non_router_unaccelerated_non_streamed_matrix_count")
    )
    if (
        non_streamed_unaccelerated_count is None
        and non_router_unaccelerated_count is not None
    ):
        non_streamed_unaccelerated_count = max(
            0,
            non_router_unaccelerated_count - (streamed_unaccelerated_count or 0),
        )
    non_streamed_unaccelerated_flops = _nonnegative_int(
        actual_coverage.get(
            "non_router_unaccelerated_non_streamed_estimated_flops"
        )
    )
    if (
        non_streamed_unaccelerated_flops is None
        and non_router_unaccelerated_flops is not None
    ):
        non_streamed_unaccelerated_flops = max(
            0,
            non_router_unaccelerated_flops - (streamed_unaccelerated_flops or 0),
        )
    unaccelerated_backend_counts = _positive_backend_int_mapping(
        actual_coverage.get("unaccelerated_backend_matrix_counts")
    )
    if not unaccelerated_backend_counts:
        unaccelerated_backend_counts = _positive_backend_int_mapping(
            {
                "custom-metal": custom_count,
                "unsupported-mpsgraph": unsupported_count,
                "other": other_count,
            }
        )
    unaccelerated_backend_flops = _positive_backend_int_mapping(
        actual_coverage.get("unaccelerated_backend_estimated_flops")
    )
    if not unaccelerated_backend_flops:
        unaccelerated_backend_flops = _positive_backend_int_mapping(
            {
                "custom-metal": custom_flops,
                "unsupported-mpsgraph": unsupported_flops,
                "other": other_flops,
            }
        )
    non_router_unaccelerated_fraction = _finite_float(
        actual_coverage.get("non_router_unaccelerated_flop_fraction")
    )
    if (
        non_router_unaccelerated_fraction is None
        and non_router_unaccelerated_flops is not None
        and total_estimated_flops is not None
        and total_estimated_flops > 0
    ):
        non_router_unaccelerated_fraction = (
            non_router_unaccelerated_flops / total_estimated_flops
        )
    router_gate_only = actual_coverage.get("accelerated_router_gate_only")
    if not isinstance(router_gate_only, bool):
        router_gate_only = (
            accelerated_count is not None
            and accelerated_flops is not None
            and accelerated_count > 0
            and router_gate_accelerated_count == accelerated_count
            and router_gate_accelerated_flops == accelerated_flops
        )
    router_gate_flop_share = _finite_float(
        actual_coverage.get("accelerated_router_gate_flop_share")
    )
    if router_gate_flop_share is None and accelerated_flops is not None:
        router_gate_flop_share = (
            (router_gate_accelerated_flops or 0) / accelerated_flops
            if accelerated_flops > 0
            else 0.0
        )
    if not router_gate_acceleration_analyzed:
        router_gate_accelerated_count = None
        router_gate_accelerated_flops = None
        non_router_accelerated_count = None
        non_router_accelerated_flops = None
        router_gate_only = None
        router_gate_flop_share = None
    routed_moe_elapsed = _finite_float(
        field_seconds.get("routed_moe_elapsed_seconds")
    )
    routed_moe_tflops = (
        streamed_routed_flops / routed_moe_elapsed / 1e12
        if streamed_routed_flops is not None
        and routed_moe_elapsed is not None
        and routed_moe_elapsed > 0.0
        else None
    )
    custom_linear_elapsed = _finite_float(linear_elapsed.get("custom-metal"))
    routed_moe_custom_fraction = (
        routed_moe_elapsed / custom_linear_elapsed
        if routed_moe_elapsed is not None
        and custom_linear_elapsed is not None
        and custom_linear_elapsed > 0.0
        else None
    )
    prompt_tokens = _nonnegative_int(request.get("prompt_tokens")) or len(prompt_ids)
    max_new_tokens = _nonnegative_int(request.get("max_new_tokens")) or len(generated)
    prompt_prefill_summary = {
        "present": bool(prompt_prefill),
        "elapsed_seconds": prompt_elapsed,
        "chunk_count": prompt_prefill.get("chunk_count"),
        "chunk_tokens": prompt_prefill.get("chunk_tokens"),
        "linear_backend_counts": dict(sorted(linear_counts.items())),
        "linear_backend_elapsed_seconds": dict(sorted(linear_elapsed.items())),
        "linear_backend_flops": dict(sorted(linear_flops.items())),
        "linear_backend_estimated_tflops": dict(sorted(linear_tflops.items())),
        "top_linear_components": top_linear_components,
        "linear_elapsed_seconds": linear_elapsed_total or None,
        "accelerated_linear_flop_fraction": accelerated_fraction,
        "expert_stage_planned_read_bytes": planned_read_bytes,
        "expert_stage_planned_read_gib": (
            planned_read_bytes / 1024**3 if planned_read_bytes is not None else None
        ),
        "expert_stage_copy_elapsed_seconds": stage_copy_elapsed,
        "expert_stage_copy_throughput_gib_per_second": copy_throughput,
        "expert_stage_io": _prefill_actual_stage_io_summary(actual_read_time),
        "max_stage_plus_compact_bytes": prompt_prefill.get(
            "max_stage_plus_compact_bytes"
        ),
        "max_routed_unique_experts_per_call": prompt_prefill.get(
            "max_routed_unique_experts_per_call"
        ),
        "total_routed_unique_expert_slots": prompt_prefill.get(
            "total_routed_unique_expert_slots"
        ),
        "persistent_moe_plan_server": prompt_prefill.get(
            "persistent_moe_plan_server"
        ),
        "persistent_resident_linear_server": prompt_prefill.get(
            "persistent_resident_linear_server"
        ),
        "persistent_attention_projection_server": prompt_prefill.get(
            "persistent_attention_projection_server"
        ),
        "persistent_attention_output_server": prompt_prefill.get(
            "persistent_attention_output_server"
        ),
        "persistent_shared_expert_server": prompt_prefill.get(
            "persistent_shared_expert_server"
        ),
        "persistent_rope_split_server": prompt_prefill.get(
            "persistent_rope_split_server"
        ),
        "persistent_mla_attention_server": prompt_prefill.get(
            "persistent_mla_attention_server"
        ),
        "persistent_rmsnorm_server": prompt_prefill.get(
            "persistent_rmsnorm_server"
        ),
        "moe_plan_server_plan_count": prompt_prefill.get(
            "moe_plan_server_plan_count"
        ),
        "routed_moe_runner_command_count": prompt_prefill.get(
            "routed_moe_runner_command_count"
        ),
        "mla_key_cache": mla_key_cache,
        "mla_value_cache": mla_value_cache,
        "mla_attention_layers": mla_attention_layers,
    }

    summary = {
        "schema": "largerlm.result_summary.v1",
        "source": source,
        "result_wrapper": wrapper,
        "launch_binding": _result_launch_binding(payload),
        "generated_token_ids": generated,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "total_elapsed_seconds": total_elapsed,
        "step_count": len(steps),
        "step_elapsed_seconds": step_elapsed_total or None,
        "logits_elapsed_seconds": logits_elapsed_total or None,
        "decode_steps": decode_steps,
        "prompt_prefill": prompt_prefill_summary,
        "prefill_plan_signature": _prefill_plan_signature_from_prompt(
            prompt_prefill_summary
        ),
        "prefill_actual": {
            "read_time": actual_read_time,
            "acceleration_coverage": actual_coverage,
            "acceleration_frontier": actual_frontier,
            "acceleration_frontier_status": frontier_status,
            "linear_backend": actual_linear,
            "accelerated_flop_fraction": actual_accelerated_fraction,
            "total_estimated_flops": total_estimated_flops,
            "accelerated_estimated_flops": accelerated_flops,
            "streamed_routed_expert_estimated_flops": streamed_routed_flops,
            "streamed_routed_expert_matrix_count": streamed_routed_matrix_count,
            "router_gate_acceleration_analyzed": router_gate_acceleration_analyzed,
            "router_gate_accelerated_matrix_count": router_gate_accelerated_count,
            "router_gate_accelerated_estimated_flops": router_gate_accelerated_flops,
            "non_router_matrix_count": non_router_matrix_count,
            "non_router_estimated_flops": non_router_estimated_flops,
            "non_router_accelerated_matrix_count": non_router_accelerated_count,
            "non_router_accelerated_estimated_flops": non_router_accelerated_flops,
            "non_router_unaccelerated_matrix_count": (
                non_router_unaccelerated_count
            ),
            "non_router_unaccelerated_estimated_flops": (
                non_router_unaccelerated_flops
            ),
            "non_router_unaccelerated_flop_fraction": (
                non_router_unaccelerated_fraction
            ),
            "non_router_unaccelerated_streamed_routed_expert_matrix_count": (
                streamed_unaccelerated_count
            ),
            "non_router_unaccelerated_streamed_routed_expert_estimated_flops": (
                streamed_unaccelerated_flops
            ),
            "non_router_unaccelerated_non_streamed_matrix_count": (
                non_streamed_unaccelerated_count
            ),
            "non_router_unaccelerated_non_streamed_estimated_flops": (
                non_streamed_unaccelerated_flops
            ),
            "unaccelerated_backend_matrix_counts": unaccelerated_backend_counts,
            "unaccelerated_backend_estimated_flops": unaccelerated_backend_flops,
            "accelerated_router_gate_flop_share": router_gate_flop_share,
            "accelerated_router_gate_only": router_gate_only,
            "routed_moe_elapsed_seconds": routed_moe_elapsed,
            "routed_moe_estimated_tflops": routed_moe_tflops,
            "routed_moe_custom_elapsed_fraction": routed_moe_custom_fraction,
            "routed_moe_layers": routed_moe_layers,
        },
        "known_subphase_elapsed_seconds": known_subphase_seconds or None,
        "unattributed_elapsed_seconds": unattributed,
        "elapsed_records": {
            "count": len(elapsed_records),
            "leaf_count": len(leaf_records),
            "backend_elapsed_seconds": dict(sorted(backend_seconds.items())),
            "backend_counts": dict(sorted(backend_counts.items())),
            "top_groups": top_groups,
            "top_records": top_records,
            "tensor_suffix_groups": tensor_suffix_groups,
        },
        "named_elapsed_fields": {
            "count": len(named_elapsed_records),
            "field_elapsed_seconds": dict(sorted(field_seconds.items())),
            "field_counts": dict(sorted(field_counts.items())),
            "top_fields": top_named_fields,
            "top_records": top_named_records,
            "field_top_records": dict(sorted(field_top_records.items())),
        },
        "runner_command_records": runner_command_records,
    }
    optimization_targets = _result_optimization_targets(
        summary,
        top_limit=top_limit,
    )
    _attach_suggested_experiment_results(
        optimization_targets,
        base_dir=experiment_result_base,
    )
    summary["optimization_targets"] = optimization_targets
    return summary


def summarize_result_file(path: str | Path, *, top_limit: int = 12) -> dict[str, Any]:
    result_path = Path(path)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResultSummaryError(f"failed to read result JSON: {result_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultSummaryError(f"invalid result JSON: {result_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultSummaryError("result JSON must contain an object")
    return summarize_result_payload(
        payload,
        source=str(result_path),
        top_limit=top_limit,
        experiment_result_base=result_path.parent,
    )


def _elapsed_ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    return candidate / baseline


def _comparison_row(
    name: str,
    baseline: object,
    candidate: object,
    *,
    kind: str,
) -> dict[str, Any] | None:
    base = _finite_float(baseline)
    cand = _finite_float(candidate)
    if base is None and cand is None:
        return None
    ratio = _elapsed_ratio(cand, base)
    return {
        "name": name,
        "kind": kind,
        "baseline_seconds": base,
        "candidate_seconds": cand,
        "delta_seconds": (
            cand - base if cand is not None and base is not None else None
        ),
        "ratio": ratio,
    }


def _summary_path_float(summary: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: object = summary
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _finite_float(value)


def _prefill_plan_signature_from_prompt(prompt: object) -> dict[str, Any]:
    prompt = _as_mapping(prompt)
    if prompt.get("present") is not True:
        return {"present": False}
    return {
        "present": True,
        "chunk_count": prompt.get("chunk_count"),
        "chunk_tokens": prompt.get("chunk_tokens"),
        "linear_backend_counts": dict(
            sorted(_as_mapping(prompt.get("linear_backend_counts")).items())
        ),
        "linear_backend_flops": dict(
            sorted(_as_mapping(prompt.get("linear_backend_flops")).items())
        ),
        "expert_stage_planned_read_bytes": prompt.get(
            "expert_stage_planned_read_bytes"
        ),
        "max_stage_plus_compact_bytes": prompt.get("max_stage_plus_compact_bytes"),
        "mla_key_cache": prompt.get("mla_key_cache"),
        "mla_value_cache": prompt.get("mla_value_cache"),
    }


def _prefill_plan_signature(summary: dict[str, Any]) -> dict[str, Any]:
    cached = summary.get("prefill_plan_signature")
    if isinstance(cached, dict):
        return cached
    return _prefill_plan_signature_from_prompt(summary.get("prompt_prefill"))


def _add_common_mapping_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    kind: str,
    baseline: object,
    candidate: object,
) -> None:
    base_map = _as_mapping(baseline)
    cand_map = _as_mapping(candidate)
    for key in sorted(set(base_map) | set(cand_map)):
        row = _comparison_row(
            f"{prefix}.{key}",
            base_map.get(key),
            cand_map.get(key),
            kind=kind,
        )
        if row is not None:
            rows.append(row)


def _tensor_suffix_elapsed_map(elapsed_records: object) -> dict[str, float]:
    groups = _as_list(_as_mapping(elapsed_records).get("tensor_suffix_groups"))
    result: dict[str, float] = {}
    for raw_group in groups:
        group = _as_mapping(raw_group)
        suffix = group.get("tensor_suffix")
        seconds = _finite_float(group.get("elapsed_seconds"))
        if isinstance(suffix, str) and suffix and seconds is not None:
            result[suffix] = seconds
    return result


def _change_rows_by_direction(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    slower: bool,
    ratio_threshold: float,
    min_abs_delta_seconds: float = 0.0,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != kind:
            continue
        delta = _finite_float(row.get("delta_seconds"))
        ratio = _finite_float(row.get("ratio"))
        if delta is None or ratio is None:
            continue
        if abs(delta) < min_abs_delta_seconds:
            continue
        if slower:
            if delta <= 0.0 or ratio < ratio_threshold:
                continue
        elif delta >= 0.0 or ratio > ratio_threshold:
            continue
        selected.append(dict(row))
    return sorted(
        selected,
        key=lambda item: abs(float(item["delta_seconds"])),
        reverse=True,
    )[:limit]


def _profile_recommendation(
    *,
    total_row: dict[str, Any] | None,
    workload_match: bool,
    generated_match: bool,
    possible_system_slowdown: bool,
    rows: list[dict[str, Any]],
    top_limit: int,
) -> dict[str, Any]:
    total_ratio = _finite_float(total_row.get("ratio")) if total_row else None
    total_delta = _finite_float(total_row.get("delta_seconds")) if total_row else None
    tensor_regressions = _change_rows_by_direction(
        rows,
        kind="tensor_suffix_elapsed",
        slower=True,
        ratio_threshold=_PROFILE_RECOMMENDATION_MAX_TOTAL_REGRESSION_RATIO,
        limit=top_limit,
    )
    tensor_improvements = _change_rows_by_direction(
        rows,
        kind="tensor_suffix_elapsed",
        slower=False,
        ratio_threshold=_PROFILE_RECOMMENDATION_MIN_TOTAL_SPEEDUP_RATIO,
        limit=top_limit,
    )
    large_tensor_regressions = _change_rows_by_direction(
        rows,
        kind="tensor_suffix_elapsed",
        slower=True,
        ratio_threshold=_PROFILE_RECOMMENDATION_TENSOR_REGRESSION_RATIO,
        min_abs_delta_seconds=_PROFILE_RECOMMENDATION_TENSOR_REGRESSION_SECONDS,
        limit=top_limit,
    )

    reasons: list[str] = []
    decision = "inconclusive"
    candidate_promotable = False
    large_tensor_regression_seconds = sum(
        max(0.0, float(row.get("delta_seconds") or 0.0))
        for row in large_tensor_regressions
    )
    total_win_seconds = (
        -total_delta if total_delta is not None and total_delta < 0.0 else 0.0
    )
    large_total_win_overrides_tensor_regressions = (
        total_ratio is not None
        and total_delta is not None
        and total_ratio <= _PROFILE_RECOMMENDATION_LARGE_TOTAL_SPEEDUP_RATIO
        and total_win_seconds
        > large_tensor_regression_seconds
        * _PROFILE_RECOMMENDATION_NET_WIN_REGRESSION_MULTIPLIER
    )
    if not workload_match or not generated_match:
        reasons.append("workload_not_comparable")
    elif possible_system_slowdown:
        reasons.append("possible_system_slowdown")
    elif total_ratio is None:
        reasons.append("missing_total_elapsed_ratio")
    elif total_ratio <= _PROFILE_RECOMMENDATION_MIN_TOTAL_SPEEDUP_RATIO:
        if not large_tensor_regressions:
            decision = "prefer_candidate"
            candidate_promotable = True
            reasons.append("candidate_total_elapsed_faster")
        elif large_total_win_overrides_tensor_regressions:
            decision = "prefer_candidate"
            candidate_promotable = True
            reasons.append("candidate_large_total_win_overrides_tensor_regressions")
        else:
            decision = "inconclusive"
            reasons.append("candidate_total_faster_but_large_tensor_regressions")
    elif total_ratio >= _PROFILE_RECOMMENDATION_MAX_TOTAL_REGRESSION_RATIO:
        decision = "prefer_baseline"
        reasons.append("candidate_total_elapsed_slower")
    else:
        decision = "tie"
        reasons.append("total_elapsed_within_two_percent")

    return {
        "decision": decision,
        "candidate_promotable": candidate_promotable,
        "baseline_preferred": decision == "prefer_baseline",
        "reasons": reasons,
        "total_ratio": total_ratio,
        "total_delta_seconds": total_delta,
        "min_total_speedup_ratio": _PROFILE_RECOMMENDATION_MIN_TOTAL_SPEEDUP_RATIO,
        "large_total_speedup_ratio": (
            _PROFILE_RECOMMENDATION_LARGE_TOTAL_SPEEDUP_RATIO
        ),
        "max_total_regression_ratio": (
            _PROFILE_RECOMMENDATION_MAX_TOTAL_REGRESSION_RATIO
        ),
        "large_tensor_regression_ratio": (
            _PROFILE_RECOMMENDATION_TENSOR_REGRESSION_RATIO
        ),
        "large_tensor_regression_seconds": (
            _PROFILE_RECOMMENDATION_TENSOR_REGRESSION_SECONDS
        ),
        "large_tensor_regression_total_seconds": large_tensor_regression_seconds,
        "net_win_regression_multiplier": (
            _PROFILE_RECOMMENDATION_NET_WIN_REGRESSION_MULTIPLIER
        ),
        "top_tensor_regressions": tensor_regressions,
        "top_tensor_improvements": tensor_improvements,
        "large_tensor_regressions": large_tensor_regressions,
    }


def compare_result_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    top_limit: int = 12,
    system_slowdown_ratio: float = 1.8,
    allow_prefill_policy_change: bool = False,
) -> dict[str, Any]:
    if top_limit < 1:
        raise ResultSummaryError("top_limit must be >= 1")
    if not math.isfinite(system_slowdown_ratio) or system_slowdown_ratio <= 1.0:
        raise ResultSummaryError("system_slowdown_ratio must be > 1")

    base_prompt = _as_mapping(baseline.get("prompt_prefill"))
    cand_prompt = _as_mapping(candidate.get("prompt_prefill"))
    base_elapsed = _as_mapping(baseline.get("elapsed_records"))
    cand_elapsed = _as_mapping(candidate.get("elapsed_records"))
    base_named = _as_mapping(baseline.get("named_elapsed_fields"))
    cand_named = _as_mapping(candidate.get("named_elapsed_fields"))

    rows: list[dict[str, Any]] = []
    for name, path in (
        ("total_elapsed_seconds", ("total_elapsed_seconds",)),
        (
            "prompt_prefill.elapsed_seconds",
            ("prompt_prefill", "elapsed_seconds"),
        ),
        (
            "prompt_prefill.linear_elapsed_seconds",
            ("prompt_prefill", "linear_elapsed_seconds"),
        ),
        (
            "prompt_prefill.expert_stage_copy_elapsed_seconds",
            ("prompt_prefill", "expert_stage_copy_elapsed_seconds"),
        ),
        ("logits_elapsed_seconds", ("logits_elapsed_seconds",)),
    ):
        row = _comparison_row(
            name,
            _summary_path_float(baseline, path),
            _summary_path_float(candidate, path),
            kind="summary",
        )
        if row is not None:
            rows.append(row)

    _add_common_mapping_rows(
        rows,
        prefix="linear_backend_elapsed_seconds",
        kind="linear_backend",
        baseline=base_prompt.get("linear_backend_elapsed_seconds"),
        candidate=cand_prompt.get("linear_backend_elapsed_seconds"),
    )
    _add_common_mapping_rows(
        rows,
        prefix="backend_elapsed_seconds",
        kind="elapsed_backend",
        baseline=base_elapsed.get("backend_elapsed_seconds"),
        candidate=cand_elapsed.get("backend_elapsed_seconds"),
    )
    _add_common_mapping_rows(
        rows,
        prefix="tensor_suffix_elapsed_seconds",
        kind="tensor_suffix_elapsed",
        baseline=_tensor_suffix_elapsed_map(base_elapsed),
        candidate=_tensor_suffix_elapsed_map(cand_elapsed),
    )
    base_field_elapsed = {
        key: value
        for key, value in _as_mapping(base_named.get("field_elapsed_seconds")).items()
        if key not in _TEXT_DUPLICATE_NAMED_ELAPSED_FIELDS
    }
    cand_field_elapsed = {
        key: value
        for key, value in _as_mapping(cand_named.get("field_elapsed_seconds")).items()
        if key not in _TEXT_DUPLICATE_NAMED_ELAPSED_FIELDS
    }
    _add_common_mapping_rows(
        rows,
        prefix="named_elapsed_fields",
        kind="named_elapsed_field",
        baseline=base_field_elapsed,
        candidate=cand_field_elapsed,
    )

    rows_with_delta = [
        row for row in rows if _finite_float(row.get("delta_seconds")) is not None
    ]
    top_changes = sorted(
        rows_with_delta,
        key=lambda item: abs(float(item["delta_seconds"])),
        reverse=True,
    )[:top_limit]

    sentinel_names = {
        "total_elapsed_seconds",
        "prompt_prefill.elapsed_seconds",
        "prompt_prefill.linear_elapsed_seconds",
        "prompt_prefill.expert_stage_copy_elapsed_seconds",
        "linear_backend_elapsed_seconds.custom-metal",
        "linear_backend_elapsed_seconds.fused-metal",
        "linear_backend_elapsed_seconds.mpsgraph-f32",
        "named_elapsed_fields.mla_attention_elapsed_seconds",
        "named_elapsed_fields.projections_elapsed_seconds",
        "named_elapsed_fields.attention_output_elapsed_seconds",
        "named_elapsed_fields.rope_elapsed_seconds",
        "named_elapsed_fields.cache_write_elapsed_seconds",
        "named_elapsed_fields.runner_backend_elapsed_seconds",
        "named_elapsed_fields.runner_matrix_f32_elapsed_seconds",
        "named_elapsed_fields.runner_accelerator_elapsed_seconds",
        "logits_elapsed_seconds",
    }
    sentinel_ratios = [
        float(row["ratio"])
        for row in rows
        if row.get("name") in sentinel_names
        and _finite_float(row.get("ratio")) is not None
    ]
    slow_sentinels = [ratio for ratio in sentinel_ratios if ratio >= system_slowdown_ratio]
    median_ratio = statistics.median(sentinel_ratios) if sentinel_ratios else None
    required_slow = max(4, math.ceil(len(sentinel_ratios) * 0.6))
    possible_system_slowdown = (
        median_ratio is not None
        and median_ratio >= system_slowdown_ratio
        and len(slow_sentinels) >= required_slow
    )

    generated_match = _as_list(baseline.get("generated_token_ids")) == _as_list(
        candidate.get("generated_token_ids")
    )
    baseline_prefill_plan = _prefill_plan_signature(baseline)
    candidate_prefill_plan = _prefill_plan_signature(candidate)
    prefill_plan_match = baseline_prefill_plan == candidate_prefill_plan
    request_shape_match = (
        baseline.get("prompt_tokens") == candidate.get("prompt_tokens")
        and baseline.get("max_new_tokens") == candidate.get("max_new_tokens")
    )
    workload_match = request_shape_match and (
        prefill_plan_match or allow_prefill_policy_change
    )
    comparison_mode = (
        "prefill_policy_experiment"
        if allow_prefill_policy_change and not prefill_plan_match
        else "strict_prefill_plan"
    )

    total_row = next(
        (row for row in rows if row.get("name") == "total_elapsed_seconds"),
        None,
    )
    recommendation = _profile_recommendation(
        total_row=total_row,
        workload_match=workload_match,
        generated_match=generated_match,
        possible_system_slowdown=possible_system_slowdown,
        rows=rows,
        top_limit=top_limit,
    )
    return {
        "schema": "largerlm.result_comparison.v1",
        "baseline": baseline.get("source"),
        "candidate": candidate.get("source"),
        "workload": {
            "prompt_tokens_match": baseline.get("prompt_tokens")
            == candidate.get("prompt_tokens"),
            "max_new_tokens_match": baseline.get("max_new_tokens")
            == candidate.get("max_new_tokens"),
            "generated_token_ids_match": generated_match,
            "request_shape_match": request_shape_match,
            "comparable": workload_match and generated_match,
            "comparison_mode": comparison_mode,
            "prefill_policy_change_allowed": allow_prefill_policy_change,
            "baseline_generated_token_ids": baseline.get("generated_token_ids"),
            "candidate_generated_token_ids": candidate.get("generated_token_ids"),
            "prefill_plan_match": prefill_plan_match,
            "baseline_prefill_plan": baseline_prefill_plan,
            "candidate_prefill_plan": candidate_prefill_plan,
        },
        "total": total_row,
        "possible_system_slowdown": {
            "detected": possible_system_slowdown,
            "threshold_ratio": system_slowdown_ratio,
            "sentinel_count": len(sentinel_ratios),
            "slow_sentinel_count": len(slow_sentinels),
            "required_slow_sentinel_count": required_slow,
            "median_sentinel_ratio": median_ratio,
        },
        "changes": {
            "count": len(rows),
            "top": top_changes,
        },
        "profile_recommendation": recommendation,
    }


def compare_result_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    top_limit: int = 12,
    system_slowdown_ratio: float = 1.8,
    allow_prefill_policy_change: bool = False,
) -> dict[str, Any]:
    baseline = summarize_result_file(baseline_path, top_limit=max(top_limit, 12))
    candidate = summarize_result_file(candidate_path, top_limit=max(top_limit, 12))
    return compare_result_summaries(
        baseline,
        candidate,
        top_limit=top_limit,
        system_slowdown_ratio=system_slowdown_ratio,
        allow_prefill_policy_change=allow_prefill_policy_change,
    )


def _launch_profile_prefill_moe_output_accumulator(
    summary: dict[str, Any],
) -> str | None:
    launch_binding = _as_mapping(summary.get("launch_binding"))
    launch_profile_path = launch_binding.get("launch_profile_path")
    if not isinstance(launch_profile_path, str) or not launch_profile_path:
        return None
    profile = _json_object_file(launch_profile_path)
    if profile is None:
        return None

    sections = _as_mapping(profile.get("sections"))
    section = _as_mapping(sections.get("prefill_moe_output_accumulator_flags"))
    section_mode = _first_nonempty_str(
        section.get("prefill_moe_output_accumulator")
    )
    if isinstance(section_mode, str):
        mode = section_mode.strip().lower()
        if mode in {"env", "file", "memory"}:
            return mode

    argv = _as_list(profile.get("argv"))
    for index, value in enumerate(argv):
        if value != "--prefill-moe-output-accumulator":
            continue
        if index + 1 >= len(argv):
            return None
        next_value = argv[index + 1]
        if not isinstance(next_value, str):
            return None
        mode = next_value.strip().lower()
        if mode in {"env", "file", "memory"}:
            return mode
        return None
    return None


def _result_replay_required_environment(summary: dict[str, Any]) -> dict[str, str]:
    prefill_actual = _as_mapping(summary.get("prefill_actual"))
    routed_layers = _as_mapping(prefill_actual.get("routed_moe_layers"))
    accumulator_counts = _as_mapping(routed_layers.get("output_accumulator_counts"))
    memory_count = _nonnegative_int(accumulator_counts.get("memory"))
    if memory_count is not None and memory_count > 0:
        if _launch_profile_prefill_moe_output_accumulator(summary) == "memory":
            return {}
        return {"LARGERLM_MOE_BATCH_ACCUMULATOR": "memory"}
    return {}


def result_bakeoff_files(
    baseline_path: str | Path,
    candidate_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    top_limit: int = 12,
    system_slowdown_ratio: float = 1.8,
    promote_only_replay_files_ready: bool = False,
    allow_prefill_policy_change: bool = False,
) -> dict[str, Any]:
    if not candidate_paths:
        raise ResultSummaryError("at least one candidate result is required")
    if top_limit < 1:
        raise ResultSummaryError("top_limit must be >= 1")
    if not math.isfinite(system_slowdown_ratio) or system_slowdown_ratio <= 1.0:
        raise ResultSummaryError("system_slowdown_ratio must be > 1")

    baseline = summarize_result_file(baseline_path, top_limit=max(top_limit, 12))
    candidates: list[dict[str, Any]] = []
    for candidate_path in candidate_paths:
        candidate = summarize_result_file(candidate_path, top_limit=max(top_limit, 12))
        comparison = compare_result_summaries(
            baseline,
            candidate,
            top_limit=top_limit,
            system_slowdown_ratio=system_slowdown_ratio,
            allow_prefill_policy_change=allow_prefill_policy_change,
        )
        recommendation = _as_mapping(comparison.get("profile_recommendation"))
        total = _as_mapping(comparison.get("total"))
        total_ratio = _finite_float(total.get("ratio"))
        total_delta = _finite_float(total.get("delta_seconds"))
        launch_binding = _as_mapping(candidate.get("launch_binding"))
        required_environment = _result_replay_required_environment(candidate)
        performance_promotable = recommendation.get("candidate_promotable") is True
        replay_ready = launch_binding.get("replay_ready") is True
        replay_files_ready = launch_binding.get("replay_files_ready") is True
        candidate_promotable = performance_promotable and (
            not promote_only_replay_files_ready or replay_files_ready
        )
        reasons = _as_list(recommendation.get("reasons"))
        if (
            performance_promotable
            and promote_only_replay_files_ready
            and not replay_files_ready
        ):
            reasons = [*reasons, "candidate_replay_files_not_ready"]
        candidates.append(
            {
                "path": str(candidate_path),
                "decision": recommendation.get("decision"),
                "candidate_promotable": candidate_promotable,
                "performance_promotable": performance_promotable,
                "replay_ready": replay_ready,
                "replay_files_ready": replay_files_ready,
                "reasons": reasons,
                "total_ratio": total_ratio,
                "total_delta_seconds": total_delta,
                "launch_binding": launch_binding or candidate.get("launch_binding"),
                "required_environment": required_environment or None,
                "comparison": comparison,
            }
        )

    promotable_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_promotable") is True
        and _finite_float(candidate.get("total_ratio")) is not None
    ]
    winner = (
        min(
            promotable_candidates,
            key=lambda item: float(item["total_ratio"]),
        )
        if promotable_candidates
        else None
    )
    winner_summary = (
        {
            "role": "candidate",
            "path": winner["path"],
            "decision": winner["decision"],
            "total_ratio": winner["total_ratio"],
            "total_delta_seconds": winner["total_delta_seconds"],
            "reasons": winner["reasons"],
            "launch_binding": winner.get("launch_binding"),
            "required_environment": winner.get("required_environment"),
        }
        if winner is not None
        else None
    )
    baseline_required_environment = _result_replay_required_environment(baseline)
    selected = (
        winner_summary
        if winner_summary is not None
        else {
            "role": "baseline",
            "path": str(baseline_path),
            "decision": "retain_baseline",
            "total_ratio": 1.0,
            "total_delta_seconds": 0.0,
            "reasons": ["no_promotable_candidate"],
            "launch_binding": baseline.get("launch_binding"),
            "required_environment": baseline_required_environment or None,
        }
    )
    return {
        "schema": "largerlm.result_bakeoff.v1",
        "baseline": str(baseline_path),
        "baseline_launch_binding": baseline.get("launch_binding"),
        "baseline_required_environment": baseline_required_environment or None,
        "promote_only_replay_files_ready": bool(promote_only_replay_files_ready),
        "allow_prefill_policy_change": bool(allow_prefill_policy_change),
        "candidate_count": len(candidates),
        "baseline_retained": winner is None,
        "winner": winner_summary,
        "selected": selected,
        "candidates": candidates,
    }


def _format_gib_from_bytes(value: object) -> str | None:
    parsed = _nonnegative_int(value)
    if parsed is None:
        return None
    return f"{parsed / 1024**3:.3f}GiB"


def _format_compact_bytes(value: object) -> str | None:
    parsed = _nonnegative_int(value)
    if parsed is None:
        return None
    units = (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    )
    for suffix, scale in units:
        if parsed >= scale:
            return f"{parsed / scale:.3f}{suffix}"
    return f"{parsed}B"


def _format_stage_io_hotspot(row: object) -> str | None:
    row_map = _as_mapping(row)
    if not row_map:
        return None
    chunk = _nonnegative_int(row_map.get("chunk_index"))
    layer = _nonnegative_int(row_map.get("layer"))
    tile = _nonnegative_int(row_map.get("tile_index"))
    if chunk is None and layer is None and tile is None:
        label = "stage"
    else:
        label_parts = []
        if chunk is not None:
            label_parts.append(f"c{chunk}")
        if layer is not None:
            label_parts.append(f"L{layer}")
        if tile is not None:
            label_parts.append(f"t{tile}")
        label = "/".join(label_parts)
    details: list[str] = []
    copy_elapsed = _finite_float(row_map.get("copy_elapsed_seconds"))
    if copy_elapsed is not None:
        details.append(f"{copy_elapsed:.3g}s")
    raw_ranges = _nonnegative_int(row_map.get("raw_range_count"))
    coalesced_ranges = _nonnegative_int(row_map.get("coalesced_range_count"))
    if raw_ranges is not None or coalesced_ranges is not None:
        details.append(f"{raw_ranges or 0}/{coalesced_ranges or 0}r")
    planned = _format_compact_bytes(row_map.get("planned_read_bytes"))
    if planned is not None:
        details.append(planned)
    copy_calls = _nonnegative_int(row_map.get("copy_read_calls"))
    if copy_calls is not None:
        details.append(f"{copy_calls}calls")
    return f"{label}({','.join(details)})" if details else label


def _format_stage_io_hotspot_list(value: object) -> str | None:
    parts = []
    for row in _as_list(value):
        text = _format_stage_io_hotspot(row)
        if text is not None:
            parts.append(text)
    return ",".join(parts) if parts else None


def _positive_int_mapping(value: object) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, item in _as_mapping(value).items():
        parsed = _nonnegative_int(item)
        if parsed is not None and parsed > 0:
            result[str(key)] = parsed
    return dict(sorted(result.items()))


def _prefill_acceleration_frontier_status(
    coverage: object,
    frontier: object,
) -> dict[str, Any] | None:
    coverage_map = _as_mapping(coverage)
    frontier_map = _as_mapping(frontier)
    if not coverage_map and not frontier_map:
        return None

    resolved_candidate: dict[str, Any] = {}
    for raw_candidate in _as_list(frontier_map.get("candidates")):
        candidate = _as_mapping(raw_candidate)
        if candidate.get("is_resolved") is True:
            resolved_candidate = candidate
            break
    if not resolved_candidate:
        candidates = [_as_mapping(item) for item in _as_list(frontier_map.get("candidates"))]
        viable_candidates = [
            item
            for item in candidates
            if item.get("viable_for_request") is True
            and _nonnegative_int(item.get("mpp_tensor_ops_candidate_matrix_count"))
            is not None
        ]
        if viable_candidates:
            resolved_candidate = max(
                viable_candidates,
                key=lambda item: (
                    _finite_float(item.get("mpp_tensor_ops_candidate_flop_fraction"))
                    or 0.0,
                    _nonnegative_int(item.get("mpp_tensor_ops_candidate_matrix_count"))
                    or 0,
                ),
            )

    policy = (
        _as_mapping(frontier_map.get("mpp_candidate_policy"))
        or _as_mapping(coverage_map.get("mpp_candidate_policy"))
        or _as_mapping(resolved_candidate.get("mpp_candidate_policy"))
    )
    candidate_count = _first_nonnegative_int(
        coverage_map.get("mpp_tensor_ops_candidate_matrix_count"),
        resolved_candidate.get("mpp_tensor_ops_candidate_matrix_count"),
    )
    total_count = _first_nonnegative_int(
        coverage_map.get("matrix_count"),
        resolved_candidate.get("matrix_count"),
    )
    candidate_flops = _first_nonnegative_int(
        coverage_map.get("mpp_tensor_ops_candidate_estimated_flops"),
        resolved_candidate.get("mpp_tensor_ops_candidate_estimated_flops"),
    )
    total_flops = _first_nonnegative_int(
        coverage_map.get("total_estimated_flops"),
        resolved_candidate.get("total_estimated_flops"),
    )
    candidate_fraction = _first_finite_float(
        coverage_map.get("mpp_tensor_ops_candidate_flop_fraction"),
        resolved_candidate.get("mpp_tensor_ops_candidate_flop_fraction"),
    )
    if (
        candidate_fraction is None
        and candidate_flops is not None
        and total_flops is not None
        and total_flops > 0
    ):
        candidate_fraction = candidate_flops / total_flops

    backend_counts = _positive_int_mapping(
        coverage_map.get("mpp_tensor_ops_candidate_backend_counts")
    ) or _positive_int_mapping(
        resolved_candidate.get("mpp_tensor_ops_candidate_backend_counts")
    )
    backend_flops = _positive_int_mapping(
        coverage_map.get("mpp_tensor_ops_candidate_backend_flops")
    ) or _positive_int_mapping(
        resolved_candidate.get("mpp_tensor_ops_candidate_backend_flops")
    )
    streamed_count = _first_nonnegative_int(
        coverage_map.get("streamed_routed_expert_mpp_candidate_matrix_count"),
        resolved_candidate.get("streamed_routed_expert_matrix_count"),
    )
    streamed_flops = _first_nonnegative_int(
        coverage_map.get("streamed_routed_expert_mpp_candidate_estimated_flops"),
        resolved_candidate.get("streamed_routed_expert_estimated_flops"),
    )
    selectable_value = policy.get("selectable_prefill_backend")
    selectable = selectable_value if isinstance(selectable_value, bool) else None
    prompt_tokens = _first_nonnegative_int(
        frontier_map.get("prompt_token_count"),
        resolved_candidate.get("prompt_chunk_tokens"),
    )
    resolved_chunk = _first_nonnegative_int(
        frontier_map.get("resolved_prompt_chunk_tokens"),
        resolved_candidate.get("prompt_chunk_tokens"),
    )
    min_tokens = _first_nonnegative_int(policy.get("mpp_tensor_ops_min_batch_tokens"))
    min_dim = _first_nonnegative_int(policy.get("mpp_tensor_ops_min_matrix_dim"))
    configured_backend = frontier_map.get("configured_backend")
    if not isinstance(configured_backend, str):
        configured_backend = None
    frontier_reason = frontier_map.get("reason")
    if not isinstance(frontier_reason, str):
        frontier_reason = None

    observed = candidate_count is not None or bool(policy)
    if not observed:
        return None
    if (
        candidate_count is not None
        and candidate_count <= 0
        and resolved_chunk is not None
        and min_tokens is not None
        and resolved_chunk < min_tokens
    ):
        status = "prompt_below_mpp_threshold"
        blocker = "resolved prompt chunk is below the MPP tensor-op candidate threshold"
    elif candidate_count is not None and candidate_count <= 0:
        status = "no_mpp_candidate_shapes"
        blocker = "no MPP-sized prefill GEMM shapes were observed"
    elif selectable is True:
        status = "mpp_backend_selectable"
        blocker = None
    elif selectable is False:
        status = "mpp_backend_not_selectable"
        blocker = (
            "candidate shapes are present but mpp_tensor_ops_prefill is not "
            "selectable in this build"
        )
    else:
        status = "mpp_selectability_unknown"
        blocker = "MPP backend selectability was not reported"

    return {
        "status": status,
        "blocker": blocker,
        "frontier_reason": frontier_reason,
        "configured_backend": configured_backend,
        "prompt_token_count": prompt_tokens,
        "resolved_prompt_chunk_tokens": resolved_chunk,
        "mpp_tensor_ops_min_batch_tokens": min_tokens,
        "mpp_tensor_ops_min_matrix_dim": min_dim,
        "mpp_tensor_ops_selectable": selectable,
        "mpp_tensor_ops_candidate_matrix_count": candidate_count,
        "matrix_count": total_count,
        "mpp_tensor_ops_candidate_estimated_flops": candidate_flops,
        "total_estimated_flops": total_flops,
        "mpp_tensor_ops_candidate_flop_fraction": candidate_fraction,
        "mpp_tensor_ops_candidate_backend_counts": backend_counts,
        "mpp_tensor_ops_candidate_backend_flops": backend_flops,
        "streamed_routed_expert_mpp_candidate_matrix_count": streamed_count,
        "streamed_routed_expert_mpp_candidate_estimated_flops": streamed_flops,
    }


def _target_elapsed_fraction(seconds: float | None, total: float | None) -> float | None:
    if seconds is None or total is None or total <= 0.0:
        return None
    return seconds / total


def _optimization_target(
    *,
    target: str,
    kind: str,
    rank_score: float,
    suggested_next_step: str,
    elapsed_seconds: float | None = None,
    total_elapsed_seconds: float | None = None,
    evidence: dict[str, object] | None = None,
    suggested_experiments: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": target,
        "kind": kind,
        "rank_score": rank_score,
        "suggested_next_step": suggested_next_step,
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    fraction = _target_elapsed_fraction(elapsed_seconds, total_elapsed_seconds)
    if fraction is not None:
        payload["elapsed_fraction"] = fraction
    if evidence:
        payload["evidence"] = evidence
    if suggested_experiments:
        payload["suggested_experiments"] = suggested_experiments
    return payload


def _bounded_mib_cap_from_bytes(
    *values: object,
    floor_mib: int,
    ceiling_mib: int,
    headroom: float,
    round_to_mib: int = 16,
) -> int | None:
    byte_values = [
        parsed
        for value in values
        if (parsed := _nonnegative_int(value)) is not None and parsed > 0
    ]
    if not byte_values:
        return None
    mib = math.ceil(max(byte_values) * headroom / (1024 * 1024))
    if round_to_mib > 1:
        mib = int(math.ceil(mib / round_to_mib) * round_to_mib)
    return min(max(mib, floor_mib), ceiling_mib)


def _routed_moe_microbench_experiment(
    routed_layers: dict[str, Any],
) -> dict[str, object] | None:
    """Build a bounded one-layer MoE microbench command from measured telemetry."""

    for raw_layer in _as_list(routed_layers.get("top_slowest_layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        batch_tokens = _nonnegative_int(layer.get("batch_tokens"))
        if layer_id is None or batch_tokens is None or batch_tokens <= 0:
            continue
        expert_ids = [
            str(expert)
            for expert in _as_list(layer.get("selected_experts"))
            if _nonnegative_int(expert) is not None
        ]
        if not expert_ids:
            continue

        stage_cap_mib = _bounded_mib_cap_from_bytes(
            layer.get("stage_planned_read_bytes"),
            layer.get("stage_staged_bytes"),
            layer.get("stage_unique_requested_bytes"),
            floor_mib=64,
            ceiling_mib=1024,
            headroom=1.20,
        )
        compact_cap_mib = _bounded_mib_cap_from_bytes(
            layer.get("compact_stage_bytes"),
            layer.get("stage_plus_compact_materialized_bytes"),
            floor_mib=64,
            ceiling_mib=1024,
            headroom=1.20,
        )
        scratch_cap_mib = _bounded_mib_cap_from_bytes(
            layer.get("moe_estimated_peak_bytes"),
            layer.get("moe_batch_buffer_bytes"),
            layer.get("moe_output_accumulator_bytes"),
            floor_mib=256,
            ceiling_mib=1024,
            headroom=2.00,
        )
        argv: list[str] = [
            "python",
            "scripts/glm_moe_tile_sweep.py",
            "--repeat",
            "4",
            "--order",
            "interleave",
            "--layer",
            str(layer_id),
            "--batch-tokens",
            str(batch_tokens),
            "--experts",
            ",".join(expert_ids),
            "--tiles",
            "1,2",
            "--vector-swiglu-modes",
            "off,on",
            "--group32-modes",
            "auto,off",
        ]
        if stage_cap_mib is not None:
            argv.extend(["--max-stage-mib", str(stage_cap_mib)])
        if compact_cap_mib is not None:
            argv.extend(["--max-compact-stage-mib", str(compact_cap_mib)])
        if scratch_cap_mib is not None:
            argv.extend(["--max-runner-scratch-mib", str(scratch_cap_mib)])
        copy_chunk_bytes = _nonnegative_int(layer.get("copy_chunk_bytes"))
        if copy_chunk_bytes is not None and copy_chunk_bytes > 0:
            copy_chunk_mib = max(1, copy_chunk_bytes // (1024 * 1024))
            argv.extend(["--copy-chunk-mib", str(copy_chunk_mib)])
        write_result = (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            f"glm-moe-layer{layer_id}-optimization-target-"
            f"{batch_tokens}tok.json"
        )
        argv.extend(["--write-result", write_result])
        safety: dict[str, object] = {
            "source": "routed_moe.top_slowest_layers[0]",
            "layer": layer_id,
            "batch_tokens": batch_tokens,
            "selected_expert_count": len(expert_ids),
            "selected_experts": [int(expert) for expert in expert_ids],
        }
        if stage_cap_mib is not None:
            safety["max_stage_mib"] = stage_cap_mib
        if compact_cap_mib is not None:
            safety["max_compact_stage_mib"] = compact_cap_mib
        if scratch_cap_mib is not None:
            safety["max_runner_scratch_mib"] = scratch_cap_mib
        peak_bytes = _nonnegative_int(layer.get("moe_estimated_peak_bytes"))
        if peak_bytes is not None:
            safety["observed_moe_estimated_peak_bytes"] = peak_bytes
        stage_bytes = _nonnegative_int(layer.get("stage_planned_read_bytes"))
        if stage_bytes is not None:
            safety["observed_stage_planned_read_bytes"] = stage_bytes
        return {
            "env": {"LARGERLM_MOE_MXFP4_SPLIT_KERNEL_TIMING": "1"},
            "argv": argv,
            "scope": "bounded-layer-microbench",
            "write_result": write_result,
            "requires_bakeoff": True,
            "promotion_gate": (
                "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
            ),
            "safety": safety,
            "reason": (
                "routed MoE is a ranked bottleneck; run a bounded tile/group32/"
                "vector-SwiGLU microbench on the measured slowest layer, then "
                "promote only after numerical agreement and a full replay bakeoff"
            ),
        }
    return None


def _attention_output_microbench_experiment(
    summary: dict[str, Any],
) -> dict[str, object] | None:
    elapsed = _as_mapping(summary.get("elapsed_records"))
    for raw_group in _as_list(elapsed.get("tensor_suffix_groups")):
        group = _as_mapping(raw_group)
        if group.get("tensor_suffix") != "self_attn.o_proj.weight":
            continue
        top_record = _as_mapping(group.get("top_record"))
        layer_id = _nonnegative_int(top_record.get("layer"))
        batch_tokens = _nonnegative_int(top_record.get("batch_tokens"))
        if batch_tokens is None:
            prompt = _as_mapping(summary.get("prompt_prefill"))
            batch_tokens = _nonnegative_int(prompt.get("chunk_tokens"))
        if batch_tokens is None:
            batch_tokens = _nonnegative_int(summary.get("prompt_tokens"))
        if layer_id is None or batch_tokens is None or batch_tokens <= 0:
            return None
        write_result = (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            f"glm-resident-o-proj-layer{layer_id}-optimization-target-"
            f"{batch_tokens}tok.json"
        )
        argv = [
            "python",
            "scripts/glm_resident_mxfp4_group32_sweep.py",
            "--repeat",
            "4",
            "--order",
            "interleave",
            "--layer",
            str(layer_id),
            "--batch-tokens",
            str(batch_tokens),
            "--group32-modes",
            "off,auto",
            "--max-resident-matrix-mib",
            "256",
            "--max-runner-scratch-mib",
            "256",
            "--write-result",
            write_result,
        ]
        safety: dict[str, object] = {
            "source": "elapsed_records.tensor_suffix_groups.self_attn.o_proj.weight",
            "layer": layer_id,
            "batch_tokens": batch_tokens,
            "max_resident_matrix_mib": 256,
            "max_runner_scratch_mib": 256,
        }
        elapsed_seconds = _finite_float(group.get("elapsed_seconds"))
        if elapsed_seconds is not None:
            safety["observed_group_elapsed_seconds"] = elapsed_seconds
        top_elapsed = _finite_float(top_record.get("elapsed_seconds"))
        if top_elapsed is not None:
            safety["observed_top_record_elapsed_seconds"] = top_elapsed
        return {
            "argv": argv,
            "scope": "bounded-resident-projection-microbench",
            "write_result": write_result,
            "requires_bakeoff": True,
            "promotion_gate": (
                "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
            ),
            "safety": safety,
            "reason": (
                "attention o_proj is a ranked resident-projection target; run a "
                "bounded group32/default microbench on the measured slowest "
                "layer, then promote only after numerical agreement and a full "
                "replay bakeoff"
            ),
        }
    return None


def _layer_from_elapsed_record_path(path: object) -> int | None:
    if not isinstance(path, str) or not path:
        return None
    marker = ".layers["
    index = path.find(marker)
    if index < 0:
        return None
    start = index + len(marker)
    end = path.find("]", start)
    if end < 0:
        return None
    try:
        layer = int(path[start:end])
    except ValueError:
        return None
    return layer if layer >= 0 else None


def _attention_projections_microbench_experiment(
    summary: dict[str, Any],
) -> dict[str, object] | None:
    named = _as_mapping(summary.get("named_elapsed_fields"))
    field_top = _as_mapping(named.get("field_top_records"))
    projection_record = _as_mapping(field_top.get("projections_elapsed_seconds"))
    layer_id = _layer_from_elapsed_record_path(projection_record.get("path"))
    prompt = _as_mapping(summary.get("prompt_prefill"))
    batch_tokens = _nonnegative_int(prompt.get("chunk_tokens"))
    if batch_tokens is None:
        batch_tokens = _nonnegative_int(summary.get("prompt_tokens"))
    if layer_id is None or batch_tokens is None or batch_tokens <= 1:
        return None
    write_result = (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        f"glm-attn-proj-layer{layer_id}-fusion-sweep-{batch_tokens}tok.json"
    )
    argv = [
        "python",
        "scripts/glm_attention_projection_fusion_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--layer",
        str(layer_id),
        "--batch-tokens",
        str(batch_tokens),
        "--modes",
        "fused,separate",
        "--max-resident-matrix-mib",
        "256",
        "--max-runner-scratch-mib",
        "256",
        "--write-result",
        write_result,
    ]
    safety: dict[str, object] = {
        "source": (
            "named_elapsed_fields.field_top_records."
            "projections_elapsed_seconds"
        ),
        "layer": layer_id,
        "batch_tokens": batch_tokens,
        "max_resident_matrix_mib": 256,
        "max_runner_scratch_mib": 256,
    }
    elapsed = _finite_float(projection_record.get("elapsed_seconds"))
    if elapsed is not None:
        safety["observed_top_record_elapsed_seconds"] = elapsed
    return {
        "argv": argv,
        "scope": "bounded-attention-projection-fusion-microbench",
        "write_result": write_result,
        "requires_bakeoff": True,
        "promotion_gate": (
            "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
        ),
        "safety": safety,
        "reason": (
            "attention projections are a ranked resident-projection target; run "
            "a bounded fused-vs-separate projection microbench on the measured "
            "slowest layer, then promote only after numerical agreement and a "
            "full replay bakeoff"
        ),
    }


def _rope_split_microbench_experiment(
    summary: dict[str, Any],
) -> dict[str, object] | None:
    named = _as_mapping(summary.get("named_elapsed_fields"))
    field_top = _as_mapping(named.get("field_top_records"))
    rope_record = _as_mapping(field_top.get("rope_elapsed_seconds"))
    layer_id = _layer_from_elapsed_record_path(rope_record.get("path"))
    prompt = _as_mapping(summary.get("prompt_prefill"))
    batch_tokens = _nonnegative_int(prompt.get("chunk_tokens"))
    if batch_tokens is None:
        batch_tokens = _nonnegative_int(summary.get("prompt_tokens"))
    mla_layers = _as_mapping(prompt.get("mla_attention_layers"))
    layer_source = _as_mapping(
        next(iter(_as_list(mla_layers.get("top_slowest_layers"))), {})
    )
    num_heads = _nonnegative_int(layer_source.get("num_heads"))
    qk_nope_dim = _nonnegative_int(layer_source.get("qk_nope_dim"))
    rope_dim = _nonnegative_int(layer_source.get("rope_dim"))
    start_position = _nonnegative_int(layer_source.get("start_position")) or 0
    if (
        layer_id is None
        or batch_tokens is None
        or batch_tokens <= 1
        or num_heads is None
        or qk_nope_dim is None
        or rope_dim is None
    ):
        return None
    write_result = (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        f"glm-rope-split-layer{layer_id}-fusion-sweep-{batch_tokens}tok.json"
    )
    argv = [
        "python",
        "scripts/glm_rope_split_sweep.py",
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--batch-tokens",
        str(batch_tokens),
        "--num-heads",
        str(num_heads),
        "--qk-nope-dim",
        str(qk_nope_dim),
        "--rope-dim",
        str(rope_dim),
        "--start-position",
        str(start_position),
        "--max-runner-scratch-mib",
        "256",
    ]
    rope_theta = _finite_float(layer_source.get("rope_theta"))
    if rope_theta is not None:
        argv.extend(["--rope-theta", f"{rope_theta:.9g}"])
    if layer_source.get("rope_interleave") is True:
        argv.append("--rope-interleave")
    argv.extend(["--write-result", write_result])
    safety: dict[str, object] = {
        "source": "named_elapsed_fields.field_top_records.rope_elapsed_seconds",
        "layer": layer_id,
        "batch_tokens": batch_tokens,
        "num_heads": num_heads,
        "qk_nope_dim": qk_nope_dim,
        "rope_dim": rope_dim,
        "max_runner_scratch_mib": 256,
    }
    elapsed = _finite_float(rope_record.get("elapsed_seconds"))
    if elapsed is not None:
        safety["observed_top_record_elapsed_seconds"] = elapsed
    return {
        "argv": argv,
        "scope": "bounded-rope-split-fusion-microbench",
        "write_result": write_result,
        "requires_bakeoff": True,
        "promotion_gate": (
            "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
        ),
        "safety": safety,
        "reason": (
            "RoPE split remains a ranked attention-kernel target; run a bounded "
            "fused-vs-old split/RoPE microbench with measured GLM dimensions, "
            "then promote only after numerical agreement and a full replay bakeoff"
        ),
    }


def _cache_write_microbench_experiment(
    summary: dict[str, Any],
) -> dict[str, object] | None:
    named = _as_mapping(summary.get("named_elapsed_fields"))
    field_top = _as_mapping(named.get("field_top_records"))
    cache_record = _as_mapping(field_top.get("cache_write_elapsed_seconds"))
    layer_id = _layer_from_elapsed_record_path(cache_record.get("path"))
    prompt = _as_mapping(summary.get("prompt_prefill"))
    batch_tokens = _nonnegative_int(prompt.get("chunk_tokens"))
    if batch_tokens is None:
        batch_tokens = _nonnegative_int(summary.get("prompt_tokens"))
    if layer_id is None or batch_tokens is None or batch_tokens <= 0:
        return None
    launch_binding = _as_mapping(summary.get("launch_binding"))
    prepared_dir = _first_nonempty_str(
        launch_binding.get("prepared_dir"),
        "artifacts/glm-5.2-mxfp4/largerlm-prepared",
    )
    write_result = (
        "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
        f"glm-cache-write-layer{layer_id}-chunk-sweep-{batch_tokens}tok.json"
    )
    argv = [
        "python",
        "scripts/glm_prefill_cache_write_sweep.py",
        "--prepared-dir",
        prepared_dir,
        "--layer",
        str(layer_id),
        "--batch-tokens",
        str(batch_tokens),
        "--repeat",
        "4",
        "--order",
        "interleave",
        "--chunk-modes",
        "default,one-row,16KiB,256KiB,1MiB",
        "--max-cache-file-mib",
        "256",
        "--max-cache-write-mib",
        "256",
        "--write-result",
        write_result,
    ]
    safety: dict[str, object] = {
        "source": "named_elapsed_fields.field_top_records.cache_write_elapsed_seconds",
        "layer": layer_id,
        "batch_tokens": batch_tokens,
        "prepared_dir": prepared_dir,
        "synthetic_cache_layout": True,
        "max_cache_file_mib": 256,
        "max_cache_write_mib": 256,
    }
    elapsed = _finite_float(cache_record.get("elapsed_seconds"))
    if elapsed is not None:
        safety["observed_top_record_elapsed_seconds"] = elapsed
    return {
        "argv": argv,
        "scope": "bounded-cache-write-chunk-sweep",
        "write_result": write_result,
        "requires_bakeoff": True,
        "promotion_gate": (
            "microbench-byte-identical-plus-result-bakeoff-total-latency-win"
        ),
        "safety": safety,
        "reason": (
            "cache write is a ranked decode-cache I/O target; run a bounded "
            "synthetic cache write chunk sweep with measured GLM cache width, "
            "then promote only after byte-identical output and a full replay "
            "bakeoff"
        ),
    }


def _mla_attention_cache_experiment(
    summary: dict[str, Any],
) -> dict[str, object] | None:
    prompt = _as_mapping(summary.get("prompt_prefill"))
    mla_layers = _as_mapping(prompt.get("mla_attention_layers"))
    for raw_layer in _as_list(mla_layers.get("top_slowest_layers")):
        layer = _as_mapping(raw_layer)
        layer_id = _nonnegative_int(layer.get("layer"))
        batch_tokens = _nonnegative_int(layer.get("batch_tokens"))
        context_length = _nonnegative_int(layer.get("context_length"))
        num_heads = _nonnegative_int(layer.get("num_heads"))
        qk_nope_dim = _nonnegative_int(layer.get("qk_nope_dim"))
        rope_dim = _nonnegative_int(layer.get("rope_dim"))
        v_head_dim = _nonnegative_int(layer.get("v_head_dim"))
        kv_lora_dim = _nonnegative_int(layer.get("kv_lora_dim"))
        if (
            layer_id is None
            or batch_tokens is None
            or context_length is None
            or num_heads is None
            or qk_nope_dim is None
            or rope_dim is None
            or v_head_dim is None
            or kv_lora_dim is None
        ):
            continue
        start_position = _nonnegative_int(layer.get("start_position")) or 0
        argv = [
            "python",
            "scripts/glm_mla_attention_cache_sweep.py",
            "--repeat",
            "4",
            "--order",
            "interleave",
            "--layer",
            str(layer_id),
            "--context-length",
            str(context_length),
            "--start-position",
            str(start_position),
            "--batch-tokens",
            str(batch_tokens),
            "--num-heads",
            str(num_heads),
            "--qk-nope-dim",
            str(qk_nope_dim),
            "--rope-dim",
            str(rope_dim),
            "--v-head-dim",
            str(v_head_dim),
            "--kv-lora-dim",
            str(kv_lora_dim),
            "--cache-modes",
            "key-value,key-only,value-only,none",
            "--max-cache-read-mib",
            "256",
            "--max-resident-matrix-mib",
            "256",
            "--max-runner-scratch-mib",
            "256",
        ]
        rope_theta = _finite_float(layer.get("rope_theta"))
        if rope_theta is not None:
            argv.extend(["--rope-theta", f"{rope_theta:.9g}"])
        if layer.get("rope_interleave") is True:
            argv.append("--rope-interleave")
        else:
            argv.append("--no-rope-interleave")
        write_result = (
            "artifacts/glm-5.2-mxfp4/largerlm-prepared/"
            f"glm-mla-layer{layer_id}-cache-sweep-{batch_tokens}tok.json"
        )
        argv.extend(["--write-result", write_result])
        safety: dict[str, object] = {
            "source": "prompt_prefill.mla_attention_layers.top_slowest_layers[0]",
            "layer": layer_id,
            "context_length": context_length,
            "batch_tokens": batch_tokens,
            "max_cache_read_mib": 256,
            "max_resident_matrix_mib": 256,
            "max_runner_scratch_mib": 256,
        }
        for source_key, target_key in (
            ("elapsed_seconds", "observed_elapsed_seconds"),
            ("mla_timing_total_elapsed_seconds", "observed_timing_total_seconds"),
            ("mla_timing_value_read_elapsed_seconds", "observed_value_read_seconds"),
            ("mla_timing_kernel_elapsed_seconds", "observed_kernel_seconds"),
        ):
            value = _finite_float(layer.get(source_key))
            if value is not None:
                safety[target_key] = value
        for source_key in (
            "estimated_peak_bytes",
            "mla_key_cache_bytes",
            "mla_value_cache_bytes",
            "cache_read_bytes",
        ):
            value = _nonnegative_int(layer.get(source_key))
            if value is not None:
                safety[f"observed_{source_key}"] = value
        return {
            "argv": argv,
            "scope": "bounded-mla-attention-cache-microbench",
            "write_result": write_result,
            "requires_bakeoff": True,
            "promotion_gate": (
                "microbench-numerical-match-plus-result-bakeoff-total-latency-win"
            ),
            "safety": safety,
            "reason": (
                "MLA attention is a ranked attention-kernel target; run a "
                "bounded key/value cache-mode microbench on the measured "
                "slowest layer, then promote only after numerical agreement "
                "and a full replay bakeoff"
            ),
        }
    return None


def _experiment_write_result_path(experiment: dict[str, Any]) -> str | None:
    explicit = experiment.get("write_result")
    if isinstance(explicit, str) and explicit:
        return explicit
    argv = experiment.get("argv")
    if not isinstance(argv, list):
        return None
    for index, item in enumerate(argv[:-1]):
        if item == "--write-result":
            path = argv[index + 1]
            return path if isinstance(path, str) and path else None
    return None


def _resolve_experiment_result_path(
    path: str,
    *,
    base_dir: str | Path | None,
) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.is_file():
        return cwd_candidate
    if base_dir is not None:
        base_candidate = Path(base_dir).expanduser() / candidate
        if base_candidate.is_file():
            return base_candidate
        return base_candidate
    return cwd_candidate


def _experiment_config_comparison_summary(
    payload: dict[str, Any],
) -> dict[str, object] | None:
    comparison = _as_mapping(payload.get("config_comparison"))
    if not comparison:
        return None
    candidate_for_full_replay = comparison.get("candidate_for_full_replay")
    requires_full_replay_bakeoff = comparison.get("requires_full_replay_bakeoff")
    rows = _as_list(comparison.get("rows"))
    result: dict[str, object] = {
        "row_count": len(rows),
        "baseline": _first_nonempty_str(
            comparison.get("baseline_config"),
            comparison.get("baseline_mode"),
        ),
        "candidate": _first_nonempty_str(
            comparison.get("candidate_config"),
            comparison.get("candidate_mode"),
        ),
        "fastest": _first_nonempty_str(
            comparison.get("fastest_kernel_config"),
            comparison.get("fastest_backend_mode"),
            comparison.get("fastest_total_mode"),
            comparison.get("fastest_wall_mode"),
            comparison.get("fastest_runner_total_config"),
        ),
        "reasons": [
            reason
            for reason in _as_list(comparison.get("reasons"))
            if isinstance(reason, str) and reason
        ],
    }
    if isinstance(candidate_for_full_replay, bool):
        result["candidate_for_full_replay"] = candidate_for_full_replay
    if isinstance(requires_full_replay_bakeoff, bool):
        result["requires_full_replay_bakeoff"] = requires_full_replay_bakeoff
    for key in (
        "max_promotion_drift",
        "min_promotion_speedup_ratio",
        "min_promotion_sample_count",
    ):
        value = comparison.get(key)
        if _finite_float(value) is not None or _nonnegative_int(value) is not None:
            result[key] = value
    return {key: value for key, value in result.items() if value is not None}


def _experiment_result_summary(
    path: str,
    *,
    base_dir: str | Path | None,
) -> dict[str, object]:
    resolved = _resolve_experiment_result_path(path, base_dir=base_dir)
    result: dict[str, object] = {
        "path": path,
        "resolved_path": str(resolved),
        "present": False,
    }
    if not resolved.is_file():
        return result
    payload = _json_object_file(str(resolved))
    if payload is None:
        result["error"] = "result_json_unreadable_or_too_large"
        return result
    result["present"] = True
    schema = payload.get("schema")
    if isinstance(schema, str) and schema:
        result["schema"] = schema
    comparison = _experiment_config_comparison_summary(payload)
    if comparison is not None:
        result["config_comparison"] = comparison
    return result


def _attach_suggested_experiment_results(
    targets: list[dict[str, object]],
    *,
    base_dir: str | Path | None,
) -> None:
    if base_dir is None:
        return
    for target in targets:
        experiments = target.get("suggested_experiments")
        if not isinstance(experiments, list):
            continue
        for raw_experiment in experiments:
            experiment = _as_mapping(raw_experiment)
            if not experiment:
                continue
            write_result = _experiment_write_result_path(experiment)
            if write_result is None:
                continue
            experiment.setdefault("write_result", write_result)
            experiment["result"] = _experiment_result_summary(
                write_result,
                base_dir=base_dir,
            )


def _result_optimization_targets(
    summary: dict[str, Any],
    *,
    top_limit: int,
) -> list[dict[str, object]]:
    """Return compact, ranked next optimization targets for this result."""

    total = _finite_float(summary.get("total_elapsed_seconds"))
    prompt = _as_mapping(summary.get("prompt_prefill"))
    actual = _as_mapping(summary.get("prefill_actual"))
    named = _as_mapping(summary.get("named_elapsed_fields"))
    field_seconds = _as_mapping(named.get("field_elapsed_seconds"))
    targets: list[dict[str, object]] = []

    runner_commands = _as_mapping(summary.get("runner_command_records"))
    runner_unique_commands = _nonnegative_int(
        runner_commands.get("unique_command_count")
    )
    if (
        runner_unique_commands is not None
        and runner_unique_commands >= _RUNNER_PROCESS_FUSION_MIN_UNIQUE_COMMANDS
    ):
        runner_record_count = _nonnegative_int(runner_commands.get("record_count"))
        runner_duplicate_count = _nonnegative_int(
            runner_commands.get("duplicate_record_count")
        )
        top_unique_groups: list[dict[str, object]] = []
        for raw_group in _as_list(runner_commands.get("top_unique_groups"))[:6]:
            group = _as_mapping(raw_group)
            name = group.get("group")
            count = _nonnegative_int(group.get("count"))
            record_count = _nonnegative_int(group.get("record_count"))
            if isinstance(name, str) and name and count is not None:
                row: dict[str, object] = {"group": name, "unique_count": count}
                if record_count is not None:
                    row["record_count"] = record_count
                top_unique_groups.append(row)
        rank_score = (total or 1.0) * min(
            1.0,
            runner_unique_commands
            / float(_RUNNER_PROCESS_FUSION_MIN_UNIQUE_COMMANDS * 4),
        )
        targets.append(
            _optimization_target(
                target="runner_process_fusion",
                kind="runner_process_orchestration",
                rank_score=rank_score,
                suggested_next_step=(
                    "prototype a persistent runner/plan-server boundary for the "
                    "largest repeated prompt command groups; prove with 128/512 "
                    "locked replay before any 2048 long run"
                ),
                total_elapsed_seconds=total,
                evidence={
                    "unique_command_count": runner_unique_commands,
                    "record_count": runner_record_count,
                    "duplicate_record_count": runner_duplicate_count,
                    "top_unique_groups": top_unique_groups,
                    "threshold_unique_command_count": (
                        _RUNNER_PROCESS_FUSION_MIN_UNIQUE_COMMANDS
                    ),
                },
            )
        )

    frontier = _as_mapping(actual.get("acceleration_frontier_status"))
    frontier_status = frontier.get("status")
    candidate_fraction = _finite_float(
        frontier.get("mpp_tensor_ops_candidate_flop_fraction")
    )
    candidate_count = _nonnegative_int(
        frontier.get("mpp_tensor_ops_candidate_matrix_count")
    )
    if (
        isinstance(frontier_status, str)
        and frontier_status in {"mpp_backend_not_selectable", "mpp_backend_selectable"}
        and candidate_count is not None
        and candidate_count > 0
    ):
        rank_score = (total or 1.0) * (candidate_fraction or 0.0)
        if frontier_status == "mpp_backend_not_selectable":
            suggested = (
                "make mpp_tensor_ops_prefill selectable or keep it blocked with "
                "explicit probe evidence"
            )
        else:
            suggested = (
                "run a guarded mpp_tensor_ops_prefill A/B and promote only with "
                "result-bakeoff"
            )
        targets.append(
            _optimization_target(
                target="mpp_tensor_ops_prefill",
                kind="prefill_acceleration_frontier",
                rank_score=rank_score,
                suggested_next_step=suggested,
                total_elapsed_seconds=total,
                evidence={
                    "frontier_status": frontier_status,
                    "candidate_matrix_count": candidate_count,
                    "matrix_count": _nonnegative_int(frontier.get("matrix_count")),
                    "candidate_flop_fraction": candidate_fraction,
                    "blocker": frontier.get("blocker"),
                },
            )
        )

    non_router_fraction = _finite_float(
        actual.get("non_router_unaccelerated_flop_fraction")
    )
    non_router_flops = _nonnegative_int(
        actual.get("non_router_unaccelerated_estimated_flops")
    )
    non_router_count = _nonnegative_int(
        actual.get("non_router_unaccelerated_matrix_count")
    )
    if non_router_fraction is not None and non_router_fraction >= 0.25:
        targets.append(
            _optimization_target(
                target="prefill_non_router_acceleration_gap",
                kind="prefill_acceleration_gap",
                rank_score=(total or 1.0) * non_router_fraction,
                suggested_next_step=(
                    "calibrate or implement acceleration for non-router prefill "
                    "GEMMs; verify with locked replay"
                ),
                total_elapsed_seconds=total,
                evidence={
                    "matrix_count": non_router_count,
                    "estimated_flops": non_router_flops,
                    "flop_fraction": non_router_fraction,
                    "backend_flops": actual.get(
                        "unaccelerated_backend_estimated_flops"
                    ),
                },
            )
        )

    routed_elapsed = _finite_float(actual.get("routed_moe_elapsed_seconds"))
    if routed_elapsed is not None and routed_elapsed > 0.0:
        persistent_moe_plan_server = prompt.get("persistent_moe_plan_server") is True
        moe_plan_server_plan_count = _nonnegative_int(
            prompt.get("moe_plan_server_plan_count")
        )
        routed_moe_runner_command_count = _nonnegative_int(
            prompt.get("routed_moe_runner_command_count")
        )
        routed_layers = _as_mapping(actual.get("routed_moe_layers"))
        hints = _as_mapping(routed_layers.get("bottleneck_hints"))
        non_runner_fraction = _finite_float(
            hints.get("moe_non_runner_elapsed_fraction")
        )
        runner_total_fraction = _finite_float(
            hints.get("moe_runner_total_elapsed_fraction")
        )
        suggested_experiments = [
            _as_mapping(experiment)
            for experiment in _as_list(hints.get("suggested_moe_kernel_experiments"))
            if _as_mapping(experiment)
        ]
        if not suggested_experiments:
            fallback_experiment = _routed_moe_microbench_experiment(routed_layers)
            if fallback_experiment is not None:
                suggested_experiments.append(fallback_experiment)
        if non_runner_fraction is not None and non_runner_fraction >= 0.40:
            if (
                persistent_moe_plan_server
                and routed_moe_runner_command_count == 0
            ):
                routed_kind = "moe_orchestration_overhead"
                routed_next_step = (
                    "batch or fuse tiled MoE plan submissions and reduce "
                    "Python/file orchestration around the persistent runner; "
                    "gate with locked replay"
                )
            else:
                routed_kind = "process_boundary"
                routed_next_step = (
                    "prototype a persistent routed-MoE runner or cross-layer fusion "
                    "to remove per-layer Python/file/subprocess overhead; gate with "
                    "locked replay"
                )
        else:
            routed_kind = "streamed_moe_kernel"
            routed_next_step = (
                "microbench routed-MoE kernel/tiling candidates and require "
                "a locked replay bakeoff before promotion"
            )
        targets.append(
            _optimization_target(
                target="routed_moe",
                kind=routed_kind,
                rank_score=routed_elapsed,
                elapsed_seconds=routed_elapsed,
                total_elapsed_seconds=total,
                suggested_next_step=routed_next_step,
                evidence={
                    "estimated_tflops": actual.get("routed_moe_estimated_tflops"),
                    "custom_elapsed_fraction": actual.get(
                        "routed_moe_custom_elapsed_fraction"
                    ),
                    "non_runner_elapsed_fraction": non_runner_fraction,
                    "runner_total_elapsed_fraction": runner_total_fraction,
                    "persistent_moe_plan_server": persistent_moe_plan_server,
                    "moe_plan_server_plan_count": moe_plan_server_plan_count,
                    "routed_moe_runner_command_count": (
                        routed_moe_runner_command_count
                    ),
                    "token_block_status": hints.get("token_block_status"),
                    "top_phase": hints.get("moe_timing_top_phase"),
                    "top_phase_fraction": hints.get(
                        "moe_timing_top_phase_fraction"
                    ),
                },
                suggested_experiments=suggested_experiments,
            )
        )

    field_targets = (
        (
            "mla_attention",
            "attention_kernel",
            "profile MLA attention value-read/cache/kernel split before changing math",
            "mla_attention_elapsed_seconds",
        ),
        (
            "attention_output",
            "resident_projection",
            "microbench attention o_proj fused/resident variants and bake off",
            "attention_output_elapsed_seconds",
        ),
        (
            "attention_projections",
            "resident_projection",
            "microbench q/kv projection batching and fusion candidates",
            "projections_elapsed_seconds",
        ),
        (
            "rope_split",
            "attention_kernel",
            "keep fused split-RoPE guarded and test only against locked replay",
            "rope_elapsed_seconds",
        ),
        (
            "cache_write",
            "decode_cache_io",
            "profile cache write batching only when it exceeds attention kernels",
            "cache_write_elapsed_seconds",
        ),
    )
    for target, kind, suggested, field in field_targets:
        elapsed = _finite_float(field_seconds.get(field))
        if elapsed is None or elapsed <= 0.0:
            continue
        suggested_experiments: list[dict[str, object]] = []
        if target == "mla_attention":
            experiment = _mla_attention_cache_experiment(summary)
            if experiment is not None:
                suggested_experiments.append(experiment)
        if target == "attention_output":
            experiment = _attention_output_microbench_experiment(summary)
            if experiment is not None:
                suggested_experiments.append(experiment)
        if target == "attention_projections":
            experiment = _attention_projections_microbench_experiment(summary)
            if experiment is not None:
                suggested_experiments.append(experiment)
        if target == "rope_split":
            experiment = _rope_split_microbench_experiment(summary)
            if experiment is not None:
                suggested_experiments.append(experiment)
        if target == "cache_write":
            experiment = _cache_write_microbench_experiment(summary)
            if experiment is not None:
                suggested_experiments.append(experiment)
        targets.append(
            _optimization_target(
                target=target,
                kind=kind,
                rank_score=elapsed,
                elapsed_seconds=elapsed,
                total_elapsed_seconds=total,
                suggested_next_step=suggested,
                evidence={
                    "field": field,
                    "count": _nonnegative_int(
                        _as_mapping(named.get("field_counts")).get(field)
                    ),
                },
                suggested_experiments=suggested_experiments,
            )
        )

    stage_copy = _finite_float(prompt.get("expert_stage_copy_elapsed_seconds"))
    stage_copy_throughput = _finite_float(
        prompt.get("expert_stage_copy_throughput_gib_per_second")
    )
    if stage_copy is not None and stage_copy > 0.0:
        routed_ratio = (
            stage_copy / routed_elapsed
            if routed_elapsed is not None and routed_elapsed > 0.0
            else None
        )
        if stage_copy >= 1.0 or (
            routed_ratio is not None and routed_ratio >= 0.2
        ):
            targets.append(
                _optimization_target(
                    target="expert_stage_copy",
                    kind="ssd_streaming",
                    rank_score=stage_copy,
                    elapsed_seconds=stage_copy,
                    total_elapsed_seconds=total,
                    suggested_next_step=(
                        "try copy chunk/read-advice experiments only under "
                        "result-bakeoff promotion gates"
                    ),
                    evidence={
                        "planned_read_bytes": prompt.get(
                            "expert_stage_planned_read_bytes"
                        ),
                        "throughput_gib_per_second": stage_copy_throughput,
                        "copy_to_routed_elapsed_ratio": routed_ratio,
                    },
                )
            )

    targets.sort(key=lambda item: float(item.get("rank_score") or 0.0), reverse=True)
    return targets[:top_limit]


def format_result_summary_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    source = summary.get("source")
    if source:
        lines.append(f"result: {source}")
    wrapper = _as_mapping(summary.get("result_wrapper"))
    if wrapper:
        client_elapsed = _finite_float(wrapper.get("client_elapsed_seconds"))
        client_text = f"{client_elapsed:.3f}s" if client_elapsed is not None else "n/a"
        lines.append(
            "wrapper: "
            f"schema={wrapper.get('schema')} "
            f"endpoint={wrapper.get('endpoint')} "
            f"status={wrapper.get('status')} "
            f"client_elapsed={client_text}"
        )
    launch_binding = _as_mapping(summary.get("launch_binding"))
    if launch_binding:
        lines.append(
            "launch binding: "
            f"profile={launch_binding.get('launch_profile_path')} "
            f"audit={launch_binding.get('launch_audit_path')} "
            f"audit_ok={bool(launch_binding.get('launch_audit_ok'))} "
            f"audit_bound={bool(launch_binding.get('launch_audit_binding_matches'))} "
            f"safe_to_replay={bool(launch_binding.get('safe_to_replay'))} "
            f"replay_ready={bool(launch_binding.get('replay_ready'))} "
            f"files_ready={bool(launch_binding.get('replay_files_ready'))}"
        )
    total = _finite_float(summary.get("total_elapsed_seconds"))
    if total is not None:
        lines.append(f"total elapsed: {total:.3f}s")
    lines.append(
        "tokens: "
        f"prompt={summary.get('prompt_tokens')} "
        f"generated={len(_as_list(summary.get('generated_token_ids')))}"
    )
    decode = _as_mapping(summary.get("decode_steps"))
    if decode.get("present") is True:
        layer_elapsed = _finite_float(decode.get("decode_layer_elapsed_seconds"))
        attention_elapsed = _finite_float(decode.get("attention_elapsed_seconds"))
        mlp_elapsed = _finite_float(decode.get("mlp_elapsed_seconds"))
        lines.append(
            "decode layers: "
            f"steps={decode.get('step_count_with_decode_layers')}/"
            f"{decode.get('generated_step_count')} "
            f"layers={decode.get('decode_layer_count')} "
            f"elapsed={_format_seconds(layer_elapsed)} "
            f"attention={_format_seconds(attention_elapsed)} "
            f"mlp={_format_seconds(mlp_elapsed)}"
        )
        top_attention = _as_list(decode.get("top_attention_layers"))
        if top_attention:
            parts: list[str] = []
            for item in top_attention[:5]:
                layer = _as_mapping(item)
                seconds = _finite_float(layer.get("attention_elapsed_seconds"))
                layer_id = layer.get("layer")
                if seconds is not None:
                    parts.append(f"L{layer_id}={seconds:.3f}s")
            if parts:
                lines.append("top decode attention layers: " + ", ".join(parts))
        top_mlp = _as_list(decode.get("top_mlp_layers"))
        if top_mlp:
            parts = []
            for item in top_mlp[:5]:
                layer = _as_mapping(item)
                seconds = _finite_float(layer.get("mlp_elapsed_seconds"))
                layer_id = layer.get("layer")
                if seconds is not None:
                    parts.append(f"L{layer_id}={seconds:.3f}s")
            if parts:
                lines.append("top decode MLP layers: " + ", ".join(parts))
        mlp_stages = _as_mapping(decode.get("mlp_stage_elapsed_seconds"))
        if mlp_stages:
            parts = []
            for key, label in (
                ("expert_kernel", "expert_kernel"),
                ("expert_read", "expert_read"),
                ("shared", "shared"),
                ("router", "router"),
                ("rmsnorm", "rmsnorm"),
                ("residual", "residual"),
                ("output", "output"),
                ("total", "total"),
            ):
                seconds = _finite_float(mlp_stages.get(key))
                if seconds is not None and seconds > 0.0:
                    parts.append(f"{label}={seconds:.3f}s")
            if parts:
                lines.append("decode MLP timing: " + " ".join(parts))
        preload_count = _nonnegative_int(
            decode.get("mlp_preload_selected_enabled_count")
        )
        preload_bytes = _nonnegative_int(decode.get("mlp_preload_selected_bytes"))
        if preload_count:
            detail = f"enabled={preload_count}/{decode.get('decode_layer_count')}"
            if preload_bytes is not None:
                detail += f" bytes={_format_compact_bytes(preload_bytes)}"
            lines.append("decode MLP selected preload: " + detail)
        fused_count = _nonnegative_int(
            decode.get("mlp_mxfp4_fused_decode_enabled_count")
        )
        if fused_count:
            lines.append(
                "decode MLP MXFP4 fused decode: "
                f"enabled={fused_count}/{decode.get('decode_layer_count')}"
            )
        mla_timing = _as_mapping(
            decode.get("mla_attention_timing_elapsed_seconds")
        )
        if mla_timing:
            parts = []
            for key, label in (
                ("kernel", "kernel"),
                ("kernel_weights", "weights"),
                ("kernel_values", "values"),
                ("total", "total"),
                ("value_read", "value_read"),
                ("cache_read", "cache_read"),
                ("metal_setup", "setup"),
                ("write", "write"),
            ):
                seconds = _finite_float(mla_timing.get(key))
                if seconds is not None and (
                    seconds > 0.0
                    or key not in {"kernel_weights", "kernel_values"}
                ):
                    parts.append(f"{label}={seconds:.3f}s")
            if parts:
                lines.append("decode MLA timing: " + " ".join(parts))
        top_mla_kernel = _as_list(decode.get("top_mla_kernel_layers"))
        if top_mla_kernel:
            parts = []
            for item in top_mla_kernel[:5]:
                layer = _as_mapping(item)
                seconds = _finite_float(layer.get("mla_kernel_elapsed_seconds"))
                layer_id = layer.get("layer")
                if seconds is not None:
                    parts.append(f"L{layer_id}={seconds:.3f}s")
            if parts:
                lines.append("top decode MLA kernel layers: " + ", ".join(parts))

    prompt = _as_mapping(summary.get("prompt_prefill"))
    if prompt.get("present"):
        prompt_elapsed = _finite_float(prompt.get("elapsed_seconds"))
        elapsed_text = f"{prompt_elapsed:.3f}s" if prompt_elapsed is not None else "n/a"
        prompt_parts = [
            f"chunks={prompt.get('chunk_count')}",
            f"chunk_tokens={prompt.get('chunk_tokens')}",
            f"elapsed={elapsed_text}",
        ]
        if prompt.get("persistent_resident_linear_server") is True:
            prompt_parts.append("persistent_linear_server=yes")
        if prompt.get("persistent_attention_projection_server") is True:
            prompt_parts.append("persistent_attention_projection_server=yes")
        if prompt.get("persistent_attention_output_server") is True:
            prompt_parts.append("persistent_attention_output_server=yes")
        if prompt.get("persistent_shared_expert_server") is True:
            prompt_parts.append("persistent_shared_expert_server=yes")
        if prompt.get("persistent_rope_split_server") is True:
            prompt_parts.append("persistent_rope_split_server=yes")
        if prompt.get("persistent_mla_attention_server") is True:
            prompt_parts.append("persistent_mla_attention_server=yes")
        if prompt.get("persistent_rmsnorm_server") is True:
            prompt_parts.append("persistent_rmsnorm_server=yes")
        lines.append("prompt prefill: " + " ".join(prompt_parts))
        planned_gib = _finite_float(prompt.get("expert_stage_planned_read_gib"))
        copy_elapsed = _finite_float(prompt.get("expert_stage_copy_elapsed_seconds"))
        throughput = _finite_float(
            prompt.get("expert_stage_copy_throughput_gib_per_second")
        )
        if planned_gib is not None or copy_elapsed is not None:
            planned_text = (
                f"{planned_gib:.3f}GiB" if planned_gib is not None else "n/a"
            )
            copy_text = (
                f"{copy_elapsed:.3f}s" if copy_elapsed is not None else "n/a"
            )
            throughput_text = (
                f"{throughput:.3f}GiB/s" if throughput is not None else "n/a"
            )
            lines.append(
                "expert stage: "
                f"planned={planned_text} "
                f"copy={copy_text} "
                f"throughput={throughput_text}"
            )
        stage_io = _as_mapping(prompt.get("expert_stage_io"))
        if stage_io:
            parts = []
            for field, label in (
                ("serial_read_bytes", "serial"),
                ("unique_requested_bytes", "unique"),
                ("planned_read_bytes", "planned"),
                ("waste_bytes", "waste"),
                ("coalesced_savings_bytes", "savings"),
            ):
                text = _format_compact_bytes(stage_io.get(field))
                if text is not None:
                    parts.append(f"{label}={text}")
            unique_amp = _finite_float(stage_io.get("unique_read_amplification"))
            if unique_amp is not None:
                parts.append(f"unique_amp={unique_amp:.3f}x")
            assignment_amp = _finite_float(
                stage_io.get("assignment_read_amplification")
            )
            if assignment_amp is not None:
                parts.append(f"serial_ratio={assignment_amp:.3f}x")
            raw_ranges = _nonnegative_int(stage_io.get("raw_ranges"))
            coalesced_ranges = _nonnegative_int(stage_io.get("coalesced_ranges"))
            if raw_ranges is not None or coalesced_ranges is not None:
                parts.append(f"ranges={raw_ranges or 0}/{coalesced_ranges or 0}")
            advice_attempts = _nonnegative_int(
                stage_io.get("read_advice_attempted_ranges")
            )
            advice_calls = _nonnegative_int(stage_io.get("read_advice_calls"))
            advice_failures = _nonnegative_int(
                stage_io.get("read_advice_failures")
            )
            if advice_attempts is not None or advice_calls is not None:
                detail = f"advice={advice_calls or 0}/{advice_attempts or 0}"
                if advice_failures is not None:
                    detail += f" failures={advice_failures}"
                parts.append(detail)
            copy_read_calls = _nonnegative_int(stage_io.get("copy_read_calls"))
            copy_write_calls = _nonnegative_int(stage_io.get("copy_write_calls"))
            if copy_read_calls is not None or copy_write_calls is not None:
                parts.append(
                    f"copy_calls={copy_read_calls or 0}/{copy_write_calls or 0}"
                )
            copy_avg_read_value = _finite_float(
                stage_io.get("copy_average_read_bytes")
            )
            copy_avg_write_value = _finite_float(
                stage_io.get("copy_average_write_bytes")
            )
            copy_avg_read = (
                _format_compact_bytes(int(copy_avg_read_value))
                if copy_avg_read_value is not None and copy_avg_read_value >= 0
                else None
            )
            copy_avg_write = (
                _format_compact_bytes(int(copy_avg_write_value))
                if copy_avg_write_value is not None and copy_avg_write_value >= 0
                else None
            )
            if copy_avg_read is not None or copy_avg_write is not None:
                parts.append(
                    f"copy_avg={copy_avg_read or 'n/a'}/{copy_avg_write or 'n/a'}"
                )
            counterfactuals = _as_mapping(
                stage_io.get("copy_read_call_counterfactuals_by_chunk_mib")
            )
            counterfactual_parts = []
            for chunk_mib, calls in sorted(
                counterfactuals.items(),
                key=lambda item: int(item[0]),
            ):
                parsed_calls = _nonnegative_int(calls)
                if parsed_calls is not None:
                    counterfactual_parts.append(f"{chunk_mib}MiB={parsed_calls}")
            if counterfactual_parts:
                parts.append("copy_calls_if=" + ",".join(counterfactual_parts))
            stage_util = _finite_float(stage_io.get("max_stage_budget_utilization"))
            if stage_util is not None:
                parts.append(f"stage_budget={stage_util:.1%}")
            if parts:
                lines.append("expert stage io: " + " ".join(parts))
            hotspot_parts = []
            copy_hotspots = _format_stage_io_hotspot_list(
                stage_io.get("copy_hotspots")
            )
            if copy_hotspots is not None:
                hotspot_parts.append(f"copy={copy_hotspots}")
            range_hotspots = _format_stage_io_hotspot_list(
                stage_io.get("range_hotspots")
            )
            if range_hotspots is not None:
                hotspot_parts.append(f"ranges={range_hotspots}")
            if hotspot_parts:
                lines.append("expert stage hotspots: " + " ".join(hotspot_parts))
        mla_key_cache = _as_mapping(prompt.get("mla_key_cache"))
        if mla_key_cache.get("observed") is True:
            layer_count = _nonnegative_int(mla_key_cache.get("layer_count"))
            enabled_count = _nonnegative_int(
                mla_key_cache.get("enabled_layer_count")
            )
            bytes_text = _format_compact_bytes(
                mla_key_cache.get("total_mla_key_cache_bytes")
            )
            if layer_count is not None and enabled_count is not None:
                lines.append(
                    "MLA key cache: "
                    f"enabled={enabled_count}/{layer_count} "
                    f"bytes={bytes_text or 'n/a'}"
                )
        mla_value_cache = _as_mapping(prompt.get("mla_value_cache"))
        if mla_value_cache.get("observed") is True:
            layer_count = _nonnegative_int(mla_value_cache.get("layer_count"))
            enabled_count = _nonnegative_int(
                mla_value_cache.get("enabled_layer_count")
            )
            bytes_text = _format_compact_bytes(
                mla_value_cache.get("total_mla_value_cache_bytes")
            )
            if layer_count is not None and enabled_count is not None:
                lines.append(
                    "MLA value cache: "
                    f"enabled={enabled_count}/{layer_count} "
                    f"bytes={bytes_text or 'n/a'}"
                )
        linear_elapsed = _as_mapping(prompt.get("linear_backend_elapsed_seconds"))
        linear_counts = _as_mapping(prompt.get("linear_backend_counts"))
        if linear_elapsed:
            parts = []
            for backend, seconds in sorted(linear_elapsed.items()):
                count = linear_counts.get(backend, "?")
                value = _finite_float(seconds)
                if value is not None:
                    parts.append(f"{backend}={value:.3f}s/{count}")
            if parts:
                lines.append("linear elapsed: " + ", ".join(parts))
        top_components = _as_list(prompt.get("top_linear_components"))
        if top_components:
            lines.append("linear component hot spots:")
            for raw_component in top_components[:6]:
                item = _as_mapping(raw_component)
                component = item.get("component")
                seconds = _finite_float(item.get("elapsed_seconds"))
                count = _nonnegative_int(item.get("count"))
                flops = _nonnegative_int(item.get("estimated_flops"))
                if not isinstance(component, str) or seconds is None:
                    continue
                parts = [f"{seconds:.3f}s"]
                if count is not None:
                    parts.append(f"x{count}")
                if flops is not None and flops > 0:
                    parts.append(f"{flops / 1e12:.3f}Tflop")
                tflops = _finite_float(item.get("estimated_tflops"))
                if tflops is not None:
                    parts.append(f"{tflops:.4g}TF/s")
                lines.append("  " + " ".join(parts) + f" {component}")
    actual = _as_mapping(summary.get("prefill_actual"))
    coverage = _as_mapping(actual.get("acceleration_coverage"))
    if coverage:
        ok = coverage.get("ok")
        fraction = _finite_float(actual.get("accelerated_flop_fraction"))
        total_flops = _nonnegative_int(actual.get("total_estimated_flops"))
        accelerated_flops = _nonnegative_int(
            actual.get("accelerated_estimated_flops")
        )
        streamed_flops = _nonnegative_int(
            actual.get("streamed_routed_expert_estimated_flops")
        )
        parts = [f"ok={ok}"]
        if fraction is not None:
            parts.append(f"fraction={fraction:.4g}")
        if total_flops is not None and accelerated_flops is not None:
            parts.append(f"flops={accelerated_flops:,}/{total_flops:,}")
        if streamed_flops is not None:
            parts.append(f"streamed={streamed_flops:,}")
        lines.append("prefill actual accel: " + " ".join(parts))
        router_analyzed = actual.get("router_gate_acceleration_analyzed") is True
        if router_analyzed:
            router_parts = ["analyzed=true"]
            router_count = _nonnegative_int(
                actual.get("router_gate_accelerated_matrix_count")
            )
            router_flops = _nonnegative_int(
                actual.get("router_gate_accelerated_estimated_flops")
            )
            non_router_flops = _nonnegative_int(
                actual.get("non_router_accelerated_estimated_flops")
            )
            router_share = _finite_float(
                actual.get("accelerated_router_gate_flop_share")
            )
            router_only = actual.get("accelerated_router_gate_only")
            if router_count is not None:
                router_parts.append(f"count={router_count}")
            if router_flops is not None:
                router_parts.append(f"flops={router_flops:,}")
            if non_router_flops is not None:
                router_parts.append(f"non_router_flops={non_router_flops:,}")
            if router_share is not None:
                router_parts.append(f"share={router_share:.1%}")
            if isinstance(router_only, bool):
                router_parts.append(f"only={router_only}")
            lines.append("prefill router-gate accel: " + " ".join(router_parts))
            gap_flops = _nonnegative_int(
                actual.get("non_router_unaccelerated_estimated_flops")
            )
            gap_count = _nonnegative_int(
                actual.get("non_router_unaccelerated_matrix_count")
            )
            gap_fraction = _finite_float(
                actual.get("non_router_unaccelerated_flop_fraction")
            )
            streamed_gap_flops = _nonnegative_int(
                actual.get(
                    "non_router_unaccelerated_streamed_routed_expert_estimated_flops"
                )
            )
            non_streamed_gap_flops = _nonnegative_int(
                actual.get(
                    "non_router_unaccelerated_non_streamed_estimated_flops"
                )
            )
            backend_gap_flops = _positive_backend_int_mapping(
                actual.get("unaccelerated_backend_estimated_flops")
            )
            if (
                gap_flops is not None
                and gap_flops > 0
                or gap_count is not None
                and gap_count > 0
            ):
                gap_parts = []
                if gap_count is not None:
                    gap_parts.append(f"matrices={gap_count}")
                if gap_flops is not None:
                    gap_parts.append(f"flops={gap_flops:,}")
                if gap_fraction is not None:
                    gap_parts.append(f"fraction={gap_fraction:.1%}")
                if streamed_gap_flops is not None:
                    gap_parts.append(f"streamed={streamed_gap_flops:,}")
                if non_streamed_gap_flops is not None:
                    gap_parts.append(f"non_streamed={non_streamed_gap_flops:,}")
                if backend_gap_flops:
                    backend_text = ",".join(
                        f"{backend}:{flops}"
                        for backend, flops in backend_gap_flops.items()
                    )
                    gap_parts.append(f"backends={backend_text}")
                lines.append(
                    "prefill non-router accel gap: " + " ".join(gap_parts)
                )
        elif _nonnegative_int(actual.get("accelerated_estimated_flops")):
            lines.append("prefill router-gate accel: analyzed=false")
        frontier_status = _as_mapping(actual.get("acceleration_frontier_status"))
        if frontier_status:
            frontier_parts = []
            status = frontier_status.get("status")
            if isinstance(status, str) and status:
                frontier_parts.append(f"status={status}")
            candidate_count = _nonnegative_int(
                frontier_status.get("mpp_tensor_ops_candidate_matrix_count")
            )
            matrix_count = _nonnegative_int(frontier_status.get("matrix_count"))
            if candidate_count is not None:
                if matrix_count is not None and matrix_count > 0:
                    frontier_parts.append(f"candidate={candidate_count}/{matrix_count}")
                else:
                    frontier_parts.append(f"candidate={candidate_count}")
            candidate_fraction = _finite_float(
                frontier_status.get("mpp_tensor_ops_candidate_flop_fraction")
            )
            if candidate_fraction is not None:
                frontier_parts.append(f"flops={candidate_fraction:.1%}")
            selectable = frontier_status.get("mpp_tensor_ops_selectable")
            if isinstance(selectable, bool):
                frontier_parts.append(f"selectable={selectable}")
            min_tokens = _nonnegative_int(
                frontier_status.get("mpp_tensor_ops_min_batch_tokens")
            )
            min_dim = _nonnegative_int(
                frontier_status.get("mpp_tensor_ops_min_matrix_dim")
            )
            if min_tokens is not None and min_dim is not None:
                frontier_parts.append(f"policy=tokens>={min_tokens},dim>={min_dim}")
            if frontier_parts:
                lines.append("prefill MPP frontier: " + " ".join(frontier_parts))
            backend_counts = _positive_int_mapping(
                frontier_status.get("mpp_tensor_ops_candidate_backend_counts")
            )
            if backend_counts:
                formatted = [
                    f"{backend}={count}"
                    for backend, count in sorted(backend_counts.items())
                ]
                lines.append("prefill MPP backends: " + ",".join(formatted))
            blocker = frontier_status.get("blocker")
            if isinstance(blocker, str) and blocker:
                lines.append(f"prefill MPP blocker: {blocker}")
        targets = _as_list(summary.get("optimization_targets"))
        if targets:
            target_parts: list[str] = []
            for raw_target in targets[:5]:
                target = _as_mapping(raw_target)
                name = target.get("target")
                if not isinstance(name, str) or not name:
                    continue
                kind = target.get("kind")
                elapsed = _finite_float(target.get("elapsed_seconds"))
                fraction = _finite_float(target.get("elapsed_fraction"))
                evidence = _as_mapping(target.get("evidence"))
                text = name
                if isinstance(kind, str) and kind:
                    text += f"[{kind}]"
                if elapsed is not None:
                    text += f"={elapsed:.3f}s"
                    if fraction is not None:
                        text += f"/{fraction:.1%}"
                else:
                    flop_fraction = _finite_float(
                        evidence.get("candidate_flop_fraction")
                    )
                    if flop_fraction is not None:
                        text += f"={flop_fraction:.1%}flops"
                    else:
                        unique_commands = _nonnegative_int(
                            evidence.get("unique_command_count")
                        )
                        if unique_commands is not None:
                            text += f"={unique_commands}cmds"
                target_parts.append(text)
            if target_parts:
                lines.append("optimization targets: " + " ".join(target_parts))
            experiment_parts: list[str] = []
            experiment_result_parts: list[str] = []
            for raw_target in targets:
                target = _as_mapping(raw_target)
                name = target.get("target")
                if not isinstance(name, str) or not name:
                    continue
                experiments = _as_list(target.get("suggested_experiments"))
                if not experiments:
                    continue
                experiment = _as_mapping(experiments[0])
                env = _as_mapping(experiment.get("env"))
                argv = experiment.get("argv")
                env_prefix = " ".join(
                    f"{key}={value}"
                    for key, value in sorted(env.items())
                    if isinstance(key, str) and key and value is not None
                )
                argv_text = (
                    " ".join(str(part) for part in argv)
                    if isinstance(argv, list) and argv
                    else ""
                )
                command = " ".join(
                    part for part in (env_prefix, argv_text) if part
                )
                if command:
                    experiment_parts.append(f"{name}: {command}")
                result = _as_mapping(experiment.get("result"))
                if result.get("present") is not True and "error" not in result:
                    continue
                result_text = name
                if result.get("present") is True:
                    result_text += ": present"
                else:
                    result_text += ": unreadable"
                schema = result.get("schema")
                if isinstance(schema, str) and schema:
                    result_text += f" schema={schema}"
                comparison = _as_mapping(result.get("config_comparison"))
                if comparison:
                    candidate = comparison.get("candidate")
                    candidate_flag = comparison.get("candidate_for_full_replay")
                    baseline = comparison.get("baseline")
                    fastest = comparison.get("fastest")
                    if isinstance(candidate_flag, bool):
                        result_text += f" candidate={candidate_flag}"
                    if isinstance(candidate, str) and candidate:
                        result_text += f" candidate_config={candidate}"
                    if isinstance(baseline, str) and baseline:
                        result_text += f" baseline={baseline}"
                    if isinstance(fastest, str) and fastest:
                        result_text += f" fastest={fastest}"
                    reasons = [
                        reason
                        for reason in _as_list(comparison.get("reasons"))
                        if isinstance(reason, str) and reason
                    ]
                    if reasons:
                        result_text += " reasons=" + ",".join(reasons[:2])
                error = result.get("error")
                if isinstance(error, str) and error:
                    result_text += f" error={error}"
                experiment_result_parts.append(result_text)
            if experiment_parts:
                lines.append(
                    "optimization target experiments "
                    "(bounded; require bakeoff): "
                    + "; ".join(experiment_parts)
                )
            if experiment_result_parts:
                lines.append(
                    "optimization target experiment results: "
                    + "; ".join(experiment_result_parts)
                )
        routed_elapsed = _finite_float(actual.get("routed_moe_elapsed_seconds"))
        routed_tflops = _finite_float(actual.get("routed_moe_estimated_tflops"))
        routed_custom_fraction = _finite_float(
            actual.get("routed_moe_custom_elapsed_fraction")
        )
        if routed_elapsed is not None:
            routed_parts = [f"elapsed={routed_elapsed:.3f}s"]
            if routed_tflops is not None:
                routed_parts.append(f"{routed_tflops:.4g} TFLOP/s")
            if routed_custom_fraction is not None:
                routed_parts.append(f"custom_share={routed_custom_fraction:.1%}")
            if prompt.get("persistent_moe_plan_server") is True:
                routed_parts.append("persistent_server=yes")
                plan_count = _nonnegative_int(
                    prompt.get("moe_plan_server_plan_count")
                )
                runner_commands = _nonnegative_int(
                    prompt.get("routed_moe_runner_command_count")
                )
                if plan_count is not None:
                    routed_parts.append(f"plans={plan_count}")
                if runner_commands is not None:
                    routed_parts.append(f"runner_launches={runner_commands}")
            non_runner_fraction = _finite_float(
                _as_mapping(actual.get("routed_moe_layers")).get(
                    "moe_non_runner_elapsed_fraction"
                )
            )
            if non_runner_fraction is not None:
                routed_parts.append(f"non_runner={non_runner_fraction:.1%}")
            lines.append("routed moe: " + " ".join(routed_parts))
        routed_layers = _as_mapping(actual.get("routed_moe_layers"))
        routed_wall_elapsed = _as_mapping(
            routed_layers.get("moe_wall_elapsed_seconds")
        )
        if routed_wall_elapsed:
            wall_parts = []
            for key, label in (
                ("total", "total"),
                ("runner", "runner_wall"),
                ("compact_stage", "compact"),
                ("routes", "routes"),
                ("static_capacity", "static"),
                ("output_validation", "validate"),
            ):
                value = _finite_float(routed_wall_elapsed.get(key))
                if value is not None:
                    wall_parts.append(f"{label}={value:.3f}s")
            residual = _finite_float(
                routed_layers.get("moe_wall_residual_elapsed_seconds")
            )
            if residual is not None:
                wall_parts.append(f"residual={residual:.3f}s")
            if wall_parts:
                lines.append("routed moe wall: " + " ".join(wall_parts))
        router_margin_layer_count = _nonnegative_int(
            routed_layers.get("router_margin_layer_count")
        )
        if router_margin_layer_count:
            margin_parts = [f"layers={router_margin_layer_count}"]
            for key, label in (
                ("router_min_effective_score_margin", "effective_min"),
                ("router_min_topk_score_margin", "topk_min"),
                ("router_min_group_score_margin", "group_min"),
            ):
                value = _finite_float(routed_layers.get(key))
                if value is not None:
                    margin_parts.append(f"{label}={value:.4g}")
            near_ties = _as_mapping(
                routed_layers.get("router_effective_near_tie_counts")
            )
            if near_ties:
                formatted = []
                for key, value in sorted(near_ties.items()):
                    parsed = _nonnegative_int(value)
                    if parsed is not None and parsed > 0:
                        formatted.append(f"{key}:{parsed}")
            if formatted:
                margin_parts.append("near_tie=" + ",".join(formatted))
            lines.append("router margins: " + " ".join(margin_parts))
        router_policy = _as_mapping(routed_layers.get("router_gate_policy"))
        if router_policy:
            policy_parts = []
            layer_count = _nonnegative_int(router_policy.get("layer_count"))
            if layer_count is not None:
                policy_parts.append(f"layers={layer_count}")
            threshold = _finite_float(router_policy.get("margin_threshold"))
            if threshold is not None:
                policy_parts.append(f"threshold={threshold:.4g}")
            decisions = _as_mapping(router_policy.get("decision_counts"))
            if decisions:
                formatted = []
                for key, value in sorted(decisions.items()):
                    parsed = _nonnegative_int(value)
                    if parsed is not None:
                        formatted.append(f"{key}:{parsed}")
                if formatted:
                    policy_parts.append("decisions=" + ",".join(formatted))
            total_policy_elapsed = _finite_float(
                router_policy.get("total_elapsed_seconds")
            )
            if total_policy_elapsed is not None:
                policy_parts.append(f"elapsed={total_policy_elapsed:.3f}s")
            extra_probe_elapsed = _finite_float(
                router_policy.get("extra_custom_probe_elapsed_seconds")
            )
            if extra_probe_elapsed is not None:
                policy_parts.append(f"fallback_custom_probe={extra_probe_elapsed:.3f}s")
            if policy_parts:
                lines.append("router hybrid: " + " ".join(policy_parts))
        bottleneck_hints = _as_mapping(routed_layers.get("bottleneck_hints"))
        if bottleneck_hints:
            hint_parts = []
            token_status = bottleneck_hints.get("token_block_status")
            if isinstance(token_status, str) and token_status:
                hint_parts.append(f"token_block={token_status}")
            copy_ratio = _finite_float(
                bottleneck_hints.get("stage_copy_to_routed_elapsed_ratio")
            )
            if copy_ratio is not None:
                hint_parts.append(f"stage_copy/routed={copy_ratio:.1%}")
            timing_top_phase = bottleneck_hints.get("moe_timing_top_phase")
            timing_top_fraction = _finite_float(
                bottleneck_hints.get("moe_timing_top_phase_fraction")
            )
            if isinstance(timing_top_phase, str) and timing_top_phase:
                if timing_top_fraction is not None:
                    hint_parts.append(
                        f"moe_top={timing_top_phase}:{timing_top_fraction:.1%}"
                    )
                else:
                    hint_parts.append(f"moe_top={timing_top_phase}")
            non_runner_fraction = _finite_float(
                bottleneck_hints.get("moe_non_runner_elapsed_fraction")
            )
            if non_runner_fraction is not None:
                hint_parts.append(f"non_runner={non_runner_fraction:.1%}")
            top_stage_share = _finite_float(
                bottleneck_hints.get("top_stage_plus_compact_layer_share")
            )
            if top_stage_share is not None:
                hint_parts.append(f"top_stage_share={top_stage_share:.1%}")
            if hint_parts:
                lines.append("routed moe hints: " + " ".join(hint_parts))
            split_elapsed = _as_mapping(
                routed_layers.get("moe_mxfp4_kernel_split_elapsed_seconds")
            )
            if split_elapsed:
                split_parts: list[str] = []
                swiglu_seconds = _finite_float(split_elapsed.get("swiglu"))
                down_add_seconds = _finite_float(split_elapsed.get("down_add"))
                total_split_seconds = _finite_float(split_elapsed.get("total"))
                if swiglu_seconds is not None:
                    split_parts.append(f"swiglu={swiglu_seconds:.3f}s")
                if down_add_seconds is not None:
                    split_parts.append(f"down_add={down_add_seconds:.3f}s")
                if total_split_seconds is not None:
                    split_parts.append(f"total={total_split_seconds:.3f}s")
                split_top = routed_layers.get("moe_mxfp4_kernel_split_top_phase")
                split_top_fraction = _finite_float(
                    routed_layers.get("moe_mxfp4_kernel_split_top_phase_fraction")
                )
                if isinstance(split_top, str) and split_top:
                    if split_top_fraction is not None:
                        split_parts.append(f"top={split_top}:{split_top_fraction:.1%}")
                    else:
                        split_parts.append(f"top={split_top}")
                if split_parts:
                    lines.append("routed moe MXFP4 split: " + " ".join(split_parts))
            copy_experiments = _as_list(
                bottleneck_hints.get("suggested_stage_copy_experiments")
            )
            if copy_experiments:
                argv_items: list[str] = []
                for experiment in copy_experiments:
                    item = _as_mapping(experiment)
                    argv = item.get("argv")
                    if isinstance(argv, list) and argv:
                        argv_items.append(" ".join(str(part) for part in argv))
                if argv_items:
                    lines.append(
                        "routed moe copy experiments "
                        "(A/B only; require result-bakeoff win): "
                        + "; ".join(argv_items)
                    )
            accumulator_experiments = _as_list(
                bottleneck_hints.get("suggested_output_accumulator_experiments")
            )
            if accumulator_experiments:
                env_items: list[str] = []
                for experiment in accumulator_experiments:
                    item = _as_mapping(experiment)
                    env = _as_mapping(item.get("env"))
                    for key, value in env.items():
                        if isinstance(key, str) and key and value is not None:
                            env_items.append(f"{key}={value}")
                if env_items:
                    lines.append(
                        "routed moe accumulator experiments "
                        "(A/B only; require result-bakeoff win): "
                        + "; ".join(env_items)
                    )
            kernel_experiments = _as_list(
                bottleneck_hints.get("suggested_moe_kernel_experiments")
            )
            if kernel_experiments:
                command_items: list[str] = []
                for experiment in kernel_experiments:
                    item = _as_mapping(experiment)
                    env = _as_mapping(item.get("env"))
                    argv = item.get("argv")
                    env_prefix = " ".join(
                        f"{key}={value}"
                        for key, value in sorted(env.items())
                        if isinstance(key, str) and key and value is not None
                    )
                    argv_text = (
                        " ".join(str(part) for part in argv)
                        if isinstance(argv, list) and argv
                        else ""
                    )
                    command = " ".join(
                        part for part in (env_prefix, argv_text) if part
                    )
                    if command:
                        command_items.append(command)
                if command_items:
                    lines.append(
                        "routed moe kernel experiments "
                        "(microbench first; require result-bakeoff win): "
                        + "; ".join(command_items)
                    )
        top_routed_layers = _as_list(routed_layers.get("top_slowest_layers"))
        if top_routed_layers:
            lines.append("routed moe slow layers:")
            for raw_layer in top_routed_layers:
                item = _as_mapping(raw_layer)
                seconds = _finite_float(item.get("elapsed_seconds"))
                if seconds is None:
                    continue
                parts = [f"{seconds:.3f}s"]
                layer = _nonnegative_int(item.get("layer"))
                if layer is not None:
                    parts.append(f"layer={layer}")
                chunk_index = _nonnegative_int(item.get("chunk_index"))
                if chunk_index is not None:
                    parts.append(f"chunk={chunk_index}")
                assignments = _nonnegative_int(item.get("assignments"))
                if assignments is not None:
                    parts.append(f"assignments={assignments}")
                expert_count = _nonnegative_int(item.get("selected_expert_count"))
                if expert_count is not None:
                    parts.append(f"experts={expert_count}")
                materialized_stage_plus = _format_gib_from_bytes(
                    item.get("stage_plus_compact_materialized_bytes")
                )
                logical_stage_plus = _format_gib_from_bytes(
                    item.get("stage_plus_compact_bytes")
                )
                if materialized_stage_plus is not None:
                    parts.append(f"materialized={materialized_stage_plus}")
                elif logical_stage_plus is not None:
                    parts.append(f"stage+compact={logical_stage_plus}")
                router_margin = _finite_float(
                    item.get("router_min_effective_score_margin")
                )
                if router_margin is not None:
                    parts.append(f"router_margin={router_margin:.4g}")
                storage = item.get("compact_stage_storage")
                if isinstance(storage, str) and storage:
                    parts.append(f"compact={storage}")
                accumulator = item.get("moe_output_accumulator")
                if isinstance(accumulator, str) and accumulator:
                    parts.append(f"accum={accumulator}")
                token_block = _nonnegative_int(item.get("effective_moe_token_block"))
                if token_block is not None:
                    parts.append(f"token_block={token_block}")
                layer_timing_parts: list[str] = []
                for phase, label in (
                    ("kernel", "kernel"),
                    ("mxfp4_swiglu_kernel", "swiglu"),
                    ("mxfp4_down_add_kernel", "down_add"),
                    ("expert_read", "expert_read"),
                    ("output_read", "output_read"),
                    ("output_write", "output_write"),
                ):
                    seconds_phase = _finite_float(
                        item.get(f"moe_timing_{phase}_seconds")
                    )
                    if seconds_phase is not None:
                        layer_timing_parts.append(f"{label}={seconds_phase:.3f}s")
                if layer_timing_parts:
                    parts.append("timing=" + ",".join(layer_timing_parts))
                used_slots = _nonnegative_int(item.get("static_capacity_used_slots"))
                total_slots = _nonnegative_int(item.get("static_capacity_total_slots"))
                if used_slots is not None and total_slots is not None:
                    parts.append(f"static={used_slots}/{total_slots}")
                lines.append("  " + " ".join(parts))

    known = _finite_float(summary.get("known_subphase_elapsed_seconds"))
    unknown = _finite_float(summary.get("unattributed_elapsed_seconds"))
    if known is not None or unknown is not None:
        if known is not None and unknown is not None:
            lines.append(f"known/unattributed: {known:.3f}s / {unknown:.3f}s")
        else:
            lines.append("known/unattributed: n/a")

    runner_commands = _as_mapping(summary.get("runner_command_records"))
    runner_record_count = _nonnegative_int(runner_commands.get("record_count"))
    runner_unique_count = _nonnegative_int(
        runner_commands.get("unique_command_count")
    )
    runner_duplicate_count = _nonnegative_int(
        runner_commands.get("duplicate_record_count")
    )
    runner_top_groups = _as_list(runner_commands.get("top_unique_groups"))
    if not runner_top_groups:
        runner_top_groups = _as_list(runner_commands.get("top_groups"))
    if runner_record_count:
        group_text: list[str] = []
        for raw_group in runner_top_groups[:6]:
            item = _as_mapping(raw_group)
            group = item.get("group")
            count = _nonnegative_int(item.get("count"))
            if isinstance(group, str) and count is not None:
                group_text.append(f"{group}={count}")
        if group_text:
            count_text = (
                f"unique={runner_unique_count} records={runner_record_count}"
                if runner_unique_count is not None
                else f"total={runner_record_count}"
            )
            if runner_duplicate_count:
                count_text += f" duplicate_records={runner_duplicate_count}"
            lines.append(
                "runner command records: "
                f"{count_text} "
                f"top={', '.join(group_text)}"
            )
        else:
            lines.append(f"runner command records: total={runner_record_count}")

    elapsed_records = _as_mapping(summary.get("elapsed_records"))
    groups = _as_list(elapsed_records.get("top_groups"))
    tensor_groups = _as_list(elapsed_records.get("tensor_suffix_groups"))
    if tensor_groups:
        lines.append("tensor hot spots:")
        for group in tensor_groups:
            item = _as_mapping(group)
            seconds = _finite_float(item.get("elapsed_seconds"))
            count = item.get("count")
            suffix = item.get("tensor_suffix")
            if seconds is None or not isinstance(suffix, str):
                continue
            backend_counts = _as_mapping(item.get("backend_counts"))
            backend_text = ", ".join(
                f"{backend}={backend_counts[backend]}"
                for backend in sorted(backend_counts)
            )
            parts = [f"{seconds:.3f}s", f"x{count}"]
            if backend_text:
                parts.append(backend_text)
            parts.append(suffix)
            lines.append("  " + " ".join(parts))
    if groups:
        lines.append("top elapsed groups:")
        for group in groups:
            item = _as_mapping(group)
            seconds = _finite_float(item.get("elapsed_seconds"))
            count = item.get("count")
            path = item.get("path")
            if seconds is not None:
                lines.append(f"  {seconds:.3f}s x{count} {path}")
    named = _as_mapping(summary.get("named_elapsed_fields"))
    fields = _as_list(named.get("top_fields"))
    if fields:
        lines.append("top named elapsed fields:")
        for field in fields:
            item = _as_mapping(field)
            seconds = _finite_float(item.get("elapsed_seconds"))
            count = item.get("count")
            name = item.get("field")
            if (
                seconds is not None
                and isinstance(name, str)
                and name not in _TEXT_DUPLICATE_NAMED_ELAPSED_FIELDS
            ):
                lines.append(f"  {seconds:.3f}s x{count} {name}")
    return "\n".join(lines)


def _format_seconds(value: object) -> str:
    number = _finite_float(value)
    return f"{number:.3f}s" if number is not None else "n/a"


def _format_ratio(value: object) -> str:
    number = _finite_float(value)
    return f"{number:.3f}x" if number is not None else "n/a"


def _format_prefill_plan_signature(plan: object) -> str:
    item = _as_mapping(plan)
    if item.get("present") is not True:
        return "present=False"
    parts = [
        f"chunks={item.get('chunk_count')}x{item.get('chunk_tokens')}",
    ]
    counts = _as_mapping(item.get("linear_backend_counts"))
    if counts:
        parts.append(
            "backends="
            + ",".join(f"{backend}={counts[backend]}" for backend in sorted(counts))
        )
    read_bytes = _nonnegative_int(item.get("expert_stage_planned_read_bytes"))
    if read_bytes is not None:
        parts.append(f"expert_read={read_bytes}B")
    stage_bytes = _nonnegative_int(item.get("max_stage_plus_compact_bytes"))
    if stage_bytes is not None:
        parts.append(f"max_stage_plus_compact={stage_bytes}B")
    for label, key in (
        ("mla_key_cache", "mla_key_cache"),
        ("mla_value_cache", "mla_value_cache"),
    ):
        cache = _as_mapping(item.get(key))
        if cache.get("observed") is not True:
            continue
        layer_count = _nonnegative_int(cache.get("layer_count"))
        enabled_count = _nonnegative_int(cache.get("enabled_layer_count"))
        if layer_count is not None and enabled_count is not None:
            parts.append(f"{label}={enabled_count}/{layer_count}")
    return " ".join(parts)


def format_result_comparison_text(comparison: dict[str, Any]) -> str:
    lines: list[str] = []
    baseline = comparison.get("baseline")
    candidate = comparison.get("candidate")
    lines.append(f"baseline: {baseline}")
    lines.append(f"candidate: {candidate}")
    workload = _as_mapping(comparison.get("workload"))
    lines.append(
        "workload comparable: "
        f"{bool(workload.get('comparable'))} "
        f"(generated ids match={bool(workload.get('generated_token_ids_match'))}, "
        f"prefill plan match={bool(workload.get('prefill_plan_match'))}, "
        f"mode={workload.get('comparison_mode') or 'strict_prefill_plan'})"
    )
    if workload and workload.get("prefill_plan_match") is False:
        lines.append(
            "prefill plan: "
            f"baseline {_format_prefill_plan_signature(workload.get('baseline_prefill_plan'))}; "
            f"candidate {_format_prefill_plan_signature(workload.get('candidate_prefill_plan'))}"
        )
    total = _as_mapping(comparison.get("total"))
    if total:
        delta = _finite_float(total.get("delta_seconds"))
        delta_text = f"{delta:+.3f}s" if delta is not None else "n/a"
        lines.append(
            "total: "
            f"{_format_seconds(total.get('candidate_seconds'))} vs "
            f"{_format_seconds(total.get('baseline_seconds'))} "
            f"delta={delta_text} ratio={_format_ratio(total.get('ratio'))}"
        )
    slowdown = _as_mapping(comparison.get("possible_system_slowdown"))
    lines.append(
        "possible system slowdown: "
        f"{bool(slowdown.get('detected'))} "
        f"(median sentinel ratio={_format_ratio(slowdown.get('median_sentinel_ratio'))})"
    )
    recommendation = _as_mapping(comparison.get("profile_recommendation"))
    if recommendation:
        reasons = _as_list(recommendation.get("reasons"))
        reason_text = ", ".join(str(reason) for reason in reasons) or "n/a"
        lines.append(
            "profile recommendation: "
            f"{recommendation.get('decision')} "
            f"(candidate_promotable={bool(recommendation.get('candidate_promotable'))}; "
            f"reasons={reason_text})"
        )
    changes = _as_mapping(comparison.get("changes"))
    top = _as_list(changes.get("top"))
    if top:
        lines.append("top changes:")
        for item in top:
            row = _as_mapping(item)
            delta = _finite_float(row.get("delta_seconds"))
            delta_text = f"{delta:+.3f}s" if delta is not None else "n/a"
            lines.append(
                "  "
                f"{row.get('name')}: "
                f"{_format_seconds(row.get('candidate_seconds'))} vs "
                f"{_format_seconds(row.get('baseline_seconds'))} "
                f"delta={delta_text} ratio={_format_ratio(row.get('ratio'))}"
            )
    return "\n".join(lines)


def format_result_bakeoff_text(bakeoff: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"baseline: {bakeoff.get('baseline')}")
    lines.append(f"candidate count: {bakeoff.get('candidate_count')}")
    if bakeoff.get("promote_only_replay_files_ready") is True:
        lines.append("promotion requires replay files ready: True")
    if bakeoff.get("allow_prefill_policy_change") is True:
        lines.append("prefill policy changes allowed: True")
    selected = _as_mapping(bakeoff.get("selected"))
    if selected:
        lines.append(
            "selected: "
            f"role={selected.get('role')} "
            f"path={selected.get('path')} "
            f"ratio={_format_ratio(selected.get('total_ratio'))} "
            f"delta={_format_seconds(selected.get('total_delta_seconds'))}"
        )
        binding = _as_mapping(selected.get("launch_binding"))
        if binding:
            lines.append(
                "selected launch: "
                f"profile={binding.get('launch_profile_path')} "
                f"audit={binding.get('launch_audit_path')} "
                f"audit_ok={bool(binding.get('launch_audit_ok'))} "
                f"audit_bound={bool(binding.get('launch_audit_binding_matches'))} "
                f"safe_to_replay={bool(binding.get('safe_to_replay'))} "
                f"replay_ready={bool(binding.get('replay_ready'))} "
                f"files_ready={bool(binding.get('replay_files_ready'))}"
            )
        required_env = _as_mapping(selected.get("required_environment"))
        if required_env:
            lines.append(
                "selected env: "
                + " ".join(
                    f"{key}={required_env[key]}" for key in sorted(required_env)
                )
            )
    winner = _as_mapping(bakeoff.get("winner"))
    if winner:
        lines.append(
            "winner: "
            f"{winner.get('path')} "
            f"ratio={_format_ratio(winner.get('total_ratio'))} "
            f"delta={_format_seconds(winner.get('total_delta_seconds'))}"
        )
        binding = _as_mapping(winner.get("launch_binding"))
        if binding:
            lines.append(
                "winner launch: "
                f"profile={binding.get('launch_profile_path')} "
                f"audit={binding.get('launch_audit_path')} "
                f"audit_ok={bool(binding.get('launch_audit_ok'))} "
                f"audit_bound={bool(binding.get('launch_audit_binding_matches'))} "
                f"safe_to_replay={bool(binding.get('safe_to_replay'))} "
                f"replay_ready={bool(binding.get('replay_ready'))} "
                f"files_ready={bool(binding.get('replay_files_ready'))}"
            )
        required_env = _as_mapping(winner.get("required_environment"))
        if required_env:
            lines.append(
                "winner env: "
                + " ".join(
                    f"{key}={required_env[key]}" for key in sorted(required_env)
                )
            )
    else:
        lines.append("winner: none")
    lines.append(f"baseline retained: {bool(bakeoff.get('baseline_retained'))}")
    if bakeoff.get("baseline_retained") is True:
        binding = _as_mapping(bakeoff.get("baseline_launch_binding"))
        if binding:
            lines.append(
                "baseline launch: "
                f"profile={binding.get('launch_profile_path')} "
                f"audit={binding.get('launch_audit_path')} "
                f"audit_ok={bool(binding.get('launch_audit_ok'))} "
                f"audit_bound={bool(binding.get('launch_audit_binding_matches'))} "
                f"safe_to_replay={bool(binding.get('safe_to_replay'))} "
                f"replay_ready={bool(binding.get('replay_ready'))} "
                f"files_ready={bool(binding.get('replay_files_ready'))}"
            )
    candidates = _as_list(bakeoff.get("candidates"))
    if candidates:
        lines.append("candidates:")
        for raw_candidate in candidates:
            candidate = _as_mapping(raw_candidate)
            reasons = _as_list(candidate.get("reasons"))
            reason_text = ", ".join(str(reason) for reason in reasons) or "n/a"
            delta = _finite_float(candidate.get("total_delta_seconds"))
            delta_text = f"{delta:+.3f}s" if delta is not None else "n/a"
            lines.append(
                "  "
                f"{candidate.get('path')}: "
                f"decision={candidate.get('decision')} "
                f"promotable={bool(candidate.get('candidate_promotable'))} "
                f"performance_promotable={bool(candidate.get('performance_promotable'))} "
                f"replay_files_ready={bool(candidate.get('replay_files_ready'))} "
                f"ratio={_format_ratio(candidate.get('total_ratio'))} "
                f"delta={delta_text} "
                f"reasons={reason_text}"
            )
    return "\n".join(lines)
