from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import first_shared_indexer_without_previous_full
from .decode_cache import DecodeCacheError, load_decode_cache_layout
from .prefill_execute import (
    PrefillExecuteError,
    run_prefill_attention_block_batch,
    run_prefill_dense_mlp_block_batch,
    run_prefill_routed_mlp_block_batch,
)
from .runtime_check import LayerRuntimeBudget, check_layer_runtime


class DecodeDriverError(RuntimeError):
    """Raised when a decode-layer driver preflight or subprocess fails."""


@dataclass(frozen=True)
class DecodeLayerRecord:
    layer: int
    kind: str
    input_path: Path
    output_path: Path
    command: tuple[str, ...]
    expert_read_bytes: int = 0
    attention_read_bytes: int = 0
    cache_read_bytes: int = 0
    dsa_index_cache_read_bytes: int = 0
    mla_cache_read_bytes: int = 0
    estimated_peak_bytes: int = 0
    router_stage_peak_bytes: int = 0
    moe_stage_peak_bytes: int = 0
    elapsed_seconds: float = 0.0
    attention_elapsed_seconds: float = 0.0
    mlp_elapsed_seconds: float = 0.0
    attention_stage_elapsed_seconds: dict[str, float] = field(default_factory=dict)
    mlp_stage_elapsed_seconds: dict[str, float] = field(default_factory=dict)
    mlp_diagnostics: dict[str, object] = field(default_factory=dict)
    mla_attention_timing_elapsed_seconds: dict[str, float] = field(default_factory=dict)
    mla_attention_diagnostics: dict[str, object] = field(default_factory=dict)
    input_in_memory: bool = False
    output_in_memory: bool = False
    composed: bool = False
    dsa_indexer_mode: str = "none"
    dsa_rope_interleave: bool = False
    dsa_indices_u32_path: Path | None = None


@dataclass(frozen=True)
class DecodeLayersResult:
    expert_layout_path: Path
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    input_path: Path
    output_path: Path
    work_dir: Path
    kept_work_dir: bool
    position: int
    context_length: int
    layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    budgets: tuple[LayerRuntimeBudget, ...]
    records: tuple[DecodeLayerRecord, ...]


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DecodeDriverError(f"failed to read {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecodeDriverError(f"failed to parse {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DecodeDriverError(f"{p} must contain a JSON object")
    return payload


def layers_from_expert_layout(expert_layout_path: str | Path) -> tuple[int, ...]:
    payload = _load_json(expert_layout_path)
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise DecodeDriverError("expert layout missing layers array")
    out: list[int] = []
    for item in layers:
        if isinstance(item, dict) and type(item.get("layer")) is int:
            out.append(int(item["layer"]))
    if not out:
        raise DecodeDriverError("expert layout contains no runnable layers")
    return tuple(sorted(out))


def _num(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.9g}"


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise DecodeDriverError(f"{label} must be an integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed < 0:
        raise DecodeDriverError(f"{label} must be non-negative")
    return parsed


def _optional_report_seconds(record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise DecodeDriverError(f"decoder layers report {key} must be numeric") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise DecodeDriverError(f"decoder layers report {key} must be non-negative")
    return seconds


def _optional_report_bool(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    if value is None:
        return False
    if type(value) is not bool:
        raise DecodeDriverError(f"decoder layers report {key} must be boolean")
    return bool(value)


def _optional_report_nonnegative_int(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if value is None:
        return 0
    if type(value) is not int or value < 0:
        raise DecodeDriverError(
            f"decoder layers report {key} must be a non-negative integer"
        )
    return int(value)


def _positive_int(value: object, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed <= 0:
        raise DecodeDriverError(f"{label} must be positive")
    return parsed


def _optional_positive_int(value: object | None, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _numeric_limit(name: str, value: float | int) -> float:
    if isinstance(value, bool):
        raise DecodeDriverError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DecodeDriverError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise DecodeDriverError(f"{name} must be finite")
    return parsed


def _positive_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed <= 0:
        raise DecodeDriverError(f"{name} must be positive")
    return parsed


def _nonnegative_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed < 0:
        raise DecodeDriverError(f"{name} must be non-negative")
    return parsed


def _optional_positive_limit(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    return _positive_limit(name, value)


def _optional_float_flag(cmd: list[str], name: str, value: float | None) -> None:
    if value is not None:
        cmd.extend([name, _num(value)])


def _optional_int_flag(cmd: list[str], name: str, value: int | None) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def _normalize_layer_set(layers: Iterable[int] | None) -> set[int]:
    if layers is None:
        return set()
    return {_nonnegative_int(layer, "layers") for layer in layers}


def _run_command(
    cmd: list[str],
    *,
    echo_output: bool,
    env: dict[str, str] | None = None,
) -> None:
    run_env = None if env is None else {**os.environ, **env}
    completed = subprocess.run(cmd, text=True, capture_output=True, env=run_env)
    if echo_output:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    if completed.returncode != 0:
        detail = ""
        if completed.stdout:
            detail += f"\nstdout:\n{completed.stdout[-4000:]}"
        if completed.stderr:
            detail += f"\nstderr:\n{completed.stderr[-4000:]}"
        raise DecodeDriverError(
            f"decoder layer command failed with exit {completed.returncode}: "
            f"{' '.join(cmd)}{detail}"
        )


def run_decode_layers(
    *,
    runner_path: str | Path,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    layers: Iterable[int] | None = None,
    dense_layers: Iterable[int] | None = None,
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
    position: int,
    context_length: int,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    kv_lora_dim: int | None = None,
    mla_kv_b_cache_dir: str | Path | None = None,
    mla_key_cache: bool = False,
    cache_position_offset: int = 0,
    attention_scale: float | None = None,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    top_k: int = 8,
    max_k: int = 8,
    router_score: str = "sigmoid",
    routed_scaling_factor: float | None = None,
    norm_topk_prob: bool = False,
    no_norm_topk_prob: bool = False,
    router_n_group: int | None = None,
    router_topk_group: int | None = None,
    ignore_router_bias: bool = False,
    include_shared_expert: bool = False,
    rms_norm_eps: float = 1e-5,
    max_slot_mib: float = 256.0,
    max_router_mib: float = 64.0,
    max_resident_matrix_mib: float = 512.0,
    max_cache_file_mib: float = 32768.0,
    max_cache_write_mib: float = 4096.0,
    max_cache_read_mib: float = 256.0,
    max_runner_scratch_mib: float = 4096.0,
    expert_read_advise_merge_gap_kib: int = 0,
    expert_read_advise_align_kib: int = 0,
    cache_dtype_bytes: int = 2,
    dsa_indexer_types: Iterable[str] | None = None,
    dsa_index_topk: int | None = None,
    dsa_index_n_heads: int | None = None,
    dsa_index_head_dim: int | None = None,
    dsa_qk_rope_dim: int | None = None,
    dsa_rope_interleave: bool = False,
    dsa_layer_norm_eps: float = 1e-6,
    write_dsa_future_cache: bool = True,
    final_logits_topk_path: str | Path | None = None,
    final_logits_top_k: int = 1,
    final_logits_chunk_rows: int | None = None,
    final_logits_max_chunk_mib: float = 64.0,
    final_logits_rms_norm_eps: float | None = None,
    final_logits_allow_tied_embeddings: bool = True,
    final_logits_skip_final_norm: bool = False,
    echo_runner_output: bool = True,
) -> DecodeLayersResult:
    if type(write_dsa_future_cache) is not bool:
        raise DecodeDriverError("write_dsa_future_cache must be a boolean")
    if type(final_logits_allow_tied_embeddings) is not bool:
        raise DecodeDriverError("final_logits_allow_tied_embeddings must be a boolean")
    if type(final_logits_skip_final_norm) is not bool:
        raise DecodeDriverError("final_logits_skip_final_norm must be a boolean")
    position = _nonnegative_int(position, "position")
    context_length = _positive_int(context_length, "context_length")
    num_heads = _positive_int(num_heads, "num_heads")
    qk_nope_dim = _positive_int(qk_nope_dim, "qk_nope_dim")
    rope_dim = _positive_int(rope_dim, "rope_dim")
    v_head_dim = _positive_int(v_head_dim, "v_head_dim")
    kv_lora_dim = _optional_positive_int(kv_lora_dim, "kv_lora_dim")
    if isinstance(mla_kv_b_cache_dir, bool):
        raise DecodeDriverError("mla_kv_b_cache_dir must be a path when provided")
    if type(mla_key_cache) is not bool:
        raise DecodeDriverError("mla_key_cache must be a boolean")
    mla_kv_b_cache_path = (
        Path(mla_kv_b_cache_dir) if mla_kv_b_cache_dir is not None else None
    )
    cache_position_offset = _nonnegative_int(
        cache_position_offset,
        "cache_position_offset",
    )
    top_k = _positive_int(top_k, "top_k")
    max_k = _positive_int(max_k, "max_k")
    router_n_group = _optional_positive_int(router_n_group, "router_n_group")
    router_topk_group = _optional_positive_int(
        router_topk_group,
        "router_topk_group",
    )
    cache_dtype_bytes = _positive_int(cache_dtype_bytes, "cache_dtype_bytes")
    dsa_index_topk = _optional_positive_int(dsa_index_topk, "dsa_index_topk")
    dsa_index_n_heads = _optional_positive_int(
        dsa_index_n_heads,
        "dsa_index_n_heads",
    )
    dsa_index_head_dim = _optional_positive_int(
        dsa_index_head_dim,
        "dsa_index_head_dim",
    )
    dsa_qk_rope_dim = _optional_positive_int(
        dsa_qk_rope_dim,
        "dsa_qk_rope_dim",
    )
    attention_scale = _optional_positive_limit("attention_scale", attention_scale)
    rope_theta = _positive_limit("rope_theta", rope_theta)
    rms_norm_eps = _nonnegative_limit("rms_norm_eps", rms_norm_eps)
    dsa_layer_norm_eps = _positive_limit(
        "dsa_layer_norm_eps",
        dsa_layer_norm_eps,
    )
    final_topk_path = Path(final_logits_topk_path) if final_logits_topk_path else None
    if final_topk_path is not None:
        final_logits_top_k = _positive_int(final_logits_top_k, "final_logits_top_k")
        if final_logits_top_k > 64:
            raise DecodeDriverError("final_logits_top_k must be in 1..64")
        final_logits_chunk_rows = _optional_positive_int(
            final_logits_chunk_rows,
            "final_logits_chunk_rows",
        )
        final_logits_max_chunk_mib = _positive_limit(
            "final_logits_max_chunk_mib",
            final_logits_max_chunk_mib,
        )
        if final_logits_rms_norm_eps is None:
            final_logits_rms_norm_eps = rms_norm_eps
        final_logits_rms_norm_eps = _nonnegative_limit(
            "final_logits_rms_norm_eps",
            final_logits_rms_norm_eps,
        )
    routed_scaling_factor = _optional_positive_limit(
        "routed_scaling_factor",
        routed_scaling_factor,
    )
    if position >= context_length:
        raise DecodeDriverError("position must be smaller than context_length")
    if top_k > max_k or max_k > 64:
        raise DecodeDriverError("top_k/max_k must satisfy 1 <= top_k <= max_k <= 64")
    if router_score not in {"sigmoid", "softmax", "raw"}:
        raise DecodeDriverError("router_score must be sigmoid, softmax, or raw")
    if norm_topk_prob and no_norm_topk_prob:
        raise DecodeDriverError("norm_topk_prob and no_norm_topk_prob conflict")
    max_slot_mib = _positive_limit("max_slot_mib", max_slot_mib)
    max_router_mib = _positive_limit("max_router_mib", max_router_mib)
    max_resident_matrix_mib = _positive_limit(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_cache_file_mib = _positive_limit("max_cache_file_mib", max_cache_file_mib)
    max_cache_write_mib = _positive_limit("max_cache_write_mib", max_cache_write_mib)
    max_cache_read_mib = _positive_limit("max_cache_read_mib", max_cache_read_mib)
    max_runner_scratch_mib = _positive_limit(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    expert_read_advise_merge_gap_kib = _nonnegative_int(
        expert_read_advise_merge_gap_kib,
        "expert_read_advise_merge_gap_kib",
    )
    expert_read_advise_align_kib = _nonnegative_int(
        expert_read_advise_align_kib,
        "expert_read_advise_align_kib",
    )

    expert_layer_ids = set(layers_from_expert_layout(expert_layout_path))
    dense_layer_ids = _normalize_layer_set(dense_layers)
    if expert_layer_ids & dense_layer_ids:
        overlap = ",".join(str(layer) for layer in sorted(expert_layer_ids & dense_layer_ids))
        raise DecodeDriverError(f"dense layers overlap expert layout layers: {overlap}")
    if layers is not None:
        requested_layers = _normalize_layer_set(layers)
    else:
        requested_layers = set(expert_layer_ids) | set(dense_layer_ids)
    unknown_layers = requested_layers - expert_layer_ids - dense_layer_ids
    if unknown_layers:
        unknown = ",".join(str(layer) for layer in sorted(unknown_layers))
        raise DecodeDriverError(f"layers not found in dense or expert layouts: {unknown}")
    layer_ids = tuple(sorted(requested_layers))
    if not layer_ids:
        raise DecodeDriverError("no layers selected")
    skip_dsa_composition = write_dsa_future_cache is False and (
        context_length == 1
        or (dsa_index_topk is not None and context_length <= dsa_index_topk)
    )
    dsa_types = tuple(str(item).lower() for item in dsa_indexer_types or ())
    invalid_dsa_types = sorted(set(dsa_types) - {"none", "full", "shared"})
    if invalid_dsa_types:
        joined = ",".join(invalid_dsa_types)
        raise DecodeDriverError(f"invalid DSA indexer_type values: {joined}")
    if dsa_types and len(dsa_types) <= max(layer_ids):
        raise DecodeDriverError("dsa_indexer_types does not cover selected layers")
    if not skip_dsa_composition:
        bad_shared_layer = first_shared_indexer_without_previous_full(
            dsa_types,
            selected_layers=layer_ids,
        )
        if bad_shared_layer is not None:
            raise DecodeDriverError(
                f"selected DSA layer {bad_shared_layer} is shared but no previous "
                "selected full-indexer layer is available"
            )
    if skip_dsa_composition:
        active_dsa_modes = set()
    else:
        active_dsa_modes = {
            dsa_types[layer]
            for layer in layer_ids
            if dsa_types
            and layer < len(dsa_types)
            and dsa_types[layer] in {"full", "shared"}
        }
    if active_dsa_modes:
        if dsa_index_topk is None or dsa_index_topk <= 0:
            raise DecodeDriverError("DSA index_topk must be positive when DSA is enabled")
        if dsa_index_n_heads is None or dsa_index_n_heads <= 0:
            raise DecodeDriverError("DSA index_n_heads must be positive when DSA is enabled")
        if dsa_layer_norm_eps <= 0:
            raise DecodeDriverError("dsa_layer_norm_eps must be positive")

    cache_layout = Path(cache_layout_path)
    resolved_dsa_index_head_dim = dsa_index_head_dim
    if active_dsa_modes:
        try:
            cache_metadata = load_decode_cache_layout(cache_layout)
        except DecodeCacheError as exc:
            raise DecodeDriverError(f"failed to read decode cache layout: {exc}") from exc
        dsa_segment_by_layer = {
            segment.layer: segment
            for segment in cache_metadata.segments
            if segment.kind == "dsa_index"
        }
        full_layers = {layer for layer in layer_ids if dsa_types[layer] == "full"}
        missing_cache_layers = sorted(full_layers - set(dsa_segment_by_layer))
        if missing_cache_layers:
            joined = ",".join(str(layer) for layer in missing_cache_layers)
            raise DecodeDriverError(
                "DSA full-indexer layers are missing dsa_index cache segments: "
                f"{joined}"
            )
        full_widths = {
            dsa_segment_by_layer[layer].width
            for layer in full_layers
            if layer in dsa_segment_by_layer
        }
        if resolved_dsa_index_head_dim is None and len(full_widths) == 1:
            resolved_dsa_index_head_dim = next(iter(full_widths))
        if resolved_dsa_index_head_dim is not None:
            mismatched_width_layers = sorted(
                layer
                for layer in full_layers
                if layer in dsa_segment_by_layer
                and dsa_segment_by_layer[layer].width != resolved_dsa_index_head_dim
            )
            if mismatched_width_layers:
                joined = ",".join(str(layer) for layer in mismatched_width_layers)
                raise DecodeDriverError(
                    "dsa_index_head_dim does not match dsa_index cache width for layers: "
                    f"{joined}"
                )

    expert_layout = Path(expert_layout_path)
    resident_layout = Path(resident_layout_path)
    cache_file = Path(cache_file_path)
    current_input = Path(input_path)
    final_output = Path(output_path)
    runner = Path(runner_path)

    max_slot_bytes = int(max_slot_mib * 1024**2)
    max_router_bytes = int(max_router_mib * 1024**2)
    max_resident_matrix_bytes = int(max_resident_matrix_mib * 1024**2)
    max_cache_read_bytes = int(max_cache_read_mib * 1024**2)
    max_runner_scratch_bytes = int(max_runner_scratch_mib * 1024**2)

    budgets = tuple(
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=layer,
            dense_mlp=layer in dense_layer_ids,
            top_k=top_k,
            max_k=max_k,
            max_slot_bytes=max_slot_bytes,
            max_router_bytes=max_router_bytes,
            max_resident_matrix_bytes=max_resident_matrix_bytes,
            max_cache_read_bytes=max_cache_read_bytes,
            max_runner_scratch_bytes=max_runner_scratch_bytes,
            include_shared_expert=include_shared_expert,
            include_decoder_layer=True,
            context_length=context_length,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            cache_dtype_bytes=cache_dtype_bytes,
            dsa_indexer_mode=(
                dsa_types[layer]
                if (
                    not skip_dsa_composition
                    and dsa_types
                    and dsa_types[layer] in {"full", "shared"}
                )
                else "none"
            ),
            dsa_index_topk=dsa_index_topk,
            dsa_index_head_dim=resolved_dsa_index_head_dim,
        )
        for layer in layer_ids
    )
    budget_by_layer = {budget.layer: budget for budget in budgets}

    created_work_dir = False
    if work_dir is None:
        work_root = Path(tempfile.mkdtemp(prefix="largerlm-decode-layers-", dir="/private/tmp"))
        created_work_dir = True
    else:
        work_root = Path(work_dir)
        work_root.mkdir(parents=True, exist_ok=True)

    records: list[DecodeLayerRecord] = []
    previous_dsa_indices: Path | None = None

    if final_topk_path is not None and not (
        skip_dsa_composition and runner.name == "largerlm-runner"
    ):
        raise DecodeDriverError(
            "in-runner final logits require batched largerlm-runner decoder layers"
        )

    def build_result() -> DecodeLayersResult:
        return DecodeLayersResult(
            expert_layout_path=expert_layout,
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout,
            cache_file_path=cache_file,
            input_path=Path(input_path),
            output_path=final_output,
            work_dir=work_root,
            kept_work_dir=keep_work_dir or not created_work_dir,
            position=position,
            context_length=context_length,
            layers=layer_ids,
            dense_layers=tuple(layer for layer in layer_ids if layer in dense_layer_ids),
            budgets=budgets,
            records=tuple(records),
        )

    try:
        if skip_dsa_composition and runner.name == "largerlm-runner":
            report_path = work_root / "decoder_layers_report.json"
            batch_work_dir = work_root / "metal_layers"
            selected_dense_layers = tuple(
                layer for layer in layer_ids if layer in dense_layer_ids
            )
            cmd = [
                str(runner),
                "--layout",
                str(expert_layout),
                "--run-decoder-layers",
                "--layers",
                ",".join(str(layer) for layer in layer_ids),
                "--resident-layout",
                str(resident_layout),
                "--cache-layout",
                str(cache_layout),
                "--cache-file",
                str(cache_file),
                "--input-f32",
                str(current_input),
                "--position",
                str(position),
                "--context-length",
                str(context_length),
                "--num-heads",
                str(num_heads),
                "--qk-nope-dim",
                str(qk_nope_dim),
                "--rope-dim",
                str(rope_dim),
                "--v-head-dim",
                str(v_head_dim),
                "--rms-norm-eps",
                _num(rms_norm_eps),
                "--output-f32",
                str(final_output),
                "--work-dir",
                str(batch_work_dir),
                "--output-report-json",
                str(report_path),
                "--max-cache-file-mib",
                _num(max_cache_file_mib),
                "--max-cache-read-mib",
                _num(max_cache_read_mib),
                "--max-resident-matrix-mib",
                _num(max_resident_matrix_mib),
                "--max-runner-scratch-mib",
                _num(max_runner_scratch_mib),
                "--top-k",
                str(top_k),
                "--max-k",
                str(max_k),
                "--router-score",
                router_score,
                "--max-slot-mib",
                _num(max_slot_mib),
                "--max-router-mib",
                _num(max_router_mib),
            ]
            if selected_dense_layers:
                cmd.extend(
                    [
                        "--dense-layers",
                        ",".join(str(layer) for layer in selected_dense_layers),
                    ]
                )
            if final_topk_path is not None:
                final_topk_path.parent.mkdir(parents=True, exist_ok=True)
                cmd.extend(
                    [
                        "--output-topk-json",
                        str(final_topk_path),
                        "--final-logits-top-k",
                        str(final_logits_top_k),
                        "--final-logits-max-chunk-mib",
                        _num(final_logits_max_chunk_mib),
                        "--final-logits-rms-norm-eps",
                        _num(final_logits_rms_norm_eps),
                    ]
                )
                if final_logits_chunk_rows is not None:
                    cmd.extend(
                        [
                            "--final-logits-chunk-rows",
                            str(final_logits_chunk_rows),
                        ]
                    )
                if not final_logits_allow_tied_embeddings:
                    cmd.append("--no-tied-embeddings")
                if final_logits_skip_final_norm:
                    cmd.append("--skip-final-norm")
            if kv_lora_dim is not None:
                cmd.extend(["--kv-lora-dim", str(kv_lora_dim)])
            if mla_kv_b_cache_path is not None:
                cmd.extend(["--mla-kv-b-cache-dir", str(mla_kv_b_cache_path)])
            if cache_position_offset:
                cmd.extend(["--cache-position-offset", str(cache_position_offset)])
            _optional_float_flag(cmd, "--attention-scale", attention_scale)
            if rope_theta != 10000.0:
                cmd.extend(["--rope-theta", _num(rope_theta)])
            if rope_interleave:
                cmd.append("--rope-interleave")
            _optional_float_flag(cmd, "--routed-scaling-factor", routed_scaling_factor)
            if norm_topk_prob:
                cmd.append("--norm-topk-prob")
            if no_norm_topk_prob:
                cmd.append("--no-norm-topk-prob")
            _optional_int_flag(cmd, "--router-n-group", router_n_group)
            _optional_int_flag(cmd, "--router-topk-group", router_topk_group)
            if ignore_router_bias:
                cmd.append("--ignore-router-bias")
            if include_shared_expert:
                cmd.append("--include-shared-expert")
            if not echo_runner_output:
                cmd.append("--quiet-inner-layers")
            if expert_read_advise_merge_gap_kib > 0:
                cmd.extend(
                    [
                        "--expert-read-advise-merge-gap-kib",
                        str(expert_read_advise_merge_gap_kib),
                    ]
                )
            if expert_read_advise_align_kib > 0:
                cmd.extend(
                    [
                        "--expert-read-advise-align-kib",
                        str(expert_read_advise_align_kib),
                    ]
                )

            runner_env = {"LARGERLM_MLA_KEY_CACHE": "1"} if mla_key_cache else None
            _run_command(cmd, echo_output=echo_runner_output, env=runner_env)
            report = _load_json(report_path)
            report_records = report.get("records")
            if not isinstance(report_records, list) or len(report_records) != len(layer_ids):
                raise DecodeDriverError("decoder layers report has invalid records")
            for index, layer in enumerate(layer_ids):
                record_payload = report_records[index]
                if not isinstance(record_payload, dict):
                    raise DecodeDriverError("decoder layers report record must be an object")
                input_value = record_payload.get("input_path")
                output_value = record_payload.get("output_path")
                elapsed_value = record_payload.get("elapsed_seconds", 0.0)
                if not isinstance(input_value, str) or not isinstance(output_value, str):
                    raise DecodeDriverError("decoder layers report paths must be strings")
                try:
                    elapsed_seconds = float(elapsed_value)
                except (TypeError, ValueError) as exc:
                    raise DecodeDriverError(
                        "decoder layers report elapsed_seconds must be numeric"
                    ) from exc
                attention_stages = {
                    "projections": _optional_report_seconds(
                        record_payload,
                        "attention_projections_elapsed_seconds",
                    ),
                    "split_q": _optional_report_seconds(
                        record_payload,
                        "split_q_elapsed_seconds",
                    ),
                    "dummy_k": _optional_report_seconds(
                        record_payload,
                        "dummy_k_elapsed_seconds",
                    ),
                    "rope": _optional_report_seconds(
                        record_payload,
                        "rope_elapsed_seconds",
                    ),
                    "mla_attention": _optional_report_seconds(
                        record_payload,
                        "mla_attention_elapsed_seconds",
                    ),
                    "attention_output": _optional_report_seconds(
                        record_payload,
                        "attention_output_elapsed_seconds",
                    ),
                }
                if not any(attention_stages.values()):
                    attention_stages = {}
                mla_timing = {
                    "input": _optional_report_seconds(
                        record_payload,
                        "mla_timing_input_elapsed_seconds",
                    ),
                    "cache_read": _optional_report_seconds(
                        record_payload,
                        "mla_timing_cache_read_elapsed_seconds",
                    ),
                    "value_read": _optional_report_seconds(
                        record_payload,
                        "mla_timing_value_read_elapsed_seconds",
                    ),
                    "metal_setup": _optional_report_seconds(
                        record_payload,
                        "mla_timing_metal_setup_elapsed_seconds",
                    ),
                    "kernel": _optional_report_seconds(
                        record_payload,
                        "mla_timing_kernel_elapsed_seconds",
                    ),
                    "kernel_weights": _optional_report_seconds(
                        record_payload,
                        "mla_timing_kernel_weights_elapsed_seconds",
                    ),
                    "kernel_values": _optional_report_seconds(
                        record_payload,
                        "mla_timing_kernel_values_elapsed_seconds",
                    ),
                    "write": _optional_report_seconds(
                        record_payload,
                        "mla_timing_write_elapsed_seconds",
                    ),
                    "total": _optional_report_seconds(
                        record_payload,
                        "mla_timing_total_elapsed_seconds",
                    ),
                }
                if not any(mla_timing.values()):
                    mla_timing = {}
                mla_diagnostics: dict[str, object] = {
                    "weights_bytes": _optional_report_nonnegative_int(
                        record_payload,
                        "mla_weights_bytes",
                    ),
                    "key_cache_bytes": _optional_report_nonnegative_int(
                        record_payload,
                        "mla_key_cache_bytes",
                    ),
                    "rope_cache_bytes": _optional_report_nonnegative_int(
                        record_payload,
                        "mla_rope_cache_bytes",
                    ),
                    "value_cache_bytes": _optional_report_nonnegative_int(
                        record_payload,
                        "mla_value_cache_bytes",
                    ),
                    "estimated_peak_bytes": _optional_report_nonnegative_int(
                        record_payload,
                        "mla_estimated_peak_bytes",
                    ),
                    "singleton_kernel": _optional_report_bool(
                        record_payload,
                        "mla_singleton_kernel",
                    ),
                    "split_kernel_timing_enabled": _optional_report_bool(
                        record_payload,
                        "mla_split_kernel_timing_enabled",
                    ),
                    "key_cache_enabled": _optional_report_bool(
                        record_payload,
                        "mla_key_cache_enabled",
                    ),
                    "rope_cache_enabled": _optional_report_bool(
                        record_payload,
                        "mla_rope_cache_enabled",
                    ),
                    "value_cache_enabled": _optional_report_bool(
                        record_payload,
                        "mla_value_cache_enabled",
                    ),
                }
                if not any(mla_diagnostics.values()):
                    mla_diagnostics = {}
                mlp_elapsed = _optional_report_seconds(
                    record_payload,
                    "mlp_elapsed_seconds",
                )
                mlp_stages = {
                    "rmsnorm": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_rmsnorm_elapsed_seconds",
                    ),
                    "router": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_router_elapsed_seconds",
                    ),
                    "moe": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_moe_elapsed_seconds",
                    ),
                    "moe_setup": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_moe_setup_elapsed_seconds",
                    ),
                    "moe_clear": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_moe_clear_elapsed_seconds",
                    ),
                    "expert_read": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_expert_read_elapsed_seconds",
                    ),
                    "expert_kernel": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_expert_kernel_elapsed_seconds",
                    ),
                    "shared": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_shared_elapsed_seconds",
                    ),
                    "residual": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_residual_elapsed_seconds",
                    ),
                    "output": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_output_elapsed_seconds",
                    ),
                    "total": _optional_report_seconds(
                        record_payload,
                        "mlp_timing_total_elapsed_seconds",
                    ),
                }
                if not any(mlp_stages.values()):
                    mlp_stages = {}
                mlp_diagnostics: dict[str, object] = {
                    "preload_selected_enabled": _optional_report_bool(
                        record_payload,
                        "mlp_preload_selected_enabled",
                    ),
                    "preload_selected_bytes": _optional_report_nonnegative_int(
                        record_payload,
                        "mlp_preload_selected_bytes",
                    ),
                    "mxfp4_fused_decode_enabled": _optional_report_bool(
                        record_payload,
                        "mlp_mxfp4_fused_decode_enabled",
                    ),
                }
                if not any(mlp_diagnostics.values()):
                    mlp_diagnostics = {}
                input_in_memory = _optional_report_bool(record_payload, "input_in_memory")
                output_in_memory = _optional_report_bool(
                    record_payload,
                    "output_in_memory",
                )
                is_dense = layer in dense_layer_ids
                budget = budget_by_layer[layer]
                records.append(
                    DecodeLayerRecord(
                        layer=layer,
                        kind="dense" if is_dense else "moe",
                        input_path=Path(input_value),
                        output_path=Path(output_value),
                        command=tuple(cmd),
                        expert_read_bytes=budget.read_bytes_per_token,
                        attention_read_bytes=budget.attention_read_bytes_per_token,
                        cache_read_bytes=budget.decoder_cache_read_bytes,
                        dsa_index_cache_read_bytes=budget.dsa_index_cache_read_bytes,
                        mla_cache_read_bytes=budget.decoder_mla_cache_read_bytes,
                        estimated_peak_bytes=budget.estimated_peak_bytes,
                        router_stage_peak_bytes=budget.router_stage_peak_bytes,
                        moe_stage_peak_bytes=budget.moe_stage_peak_bytes,
                        elapsed_seconds=elapsed_seconds,
                        attention_elapsed_seconds=sum(attention_stages.values()),
                        mlp_elapsed_seconds=mlp_elapsed,
                        attention_stage_elapsed_seconds=attention_stages,
                        mlp_stage_elapsed_seconds=mlp_stages,
                        mlp_diagnostics=mlp_diagnostics,
                        mla_attention_timing_elapsed_seconds=mla_timing,
                        mla_attention_diagnostics=mla_diagnostics,
                        input_in_memory=input_in_memory,
                        output_in_memory=output_in_memory,
                    )
                )
            return build_result()

        for index, layer in enumerate(layer_ids):
            is_last = index + 1 == len(layer_ids)
            layer_output = final_output if is_last else work_root / f"layer_{layer:04d}.f32"
            layer_started = time.perf_counter()
            is_dense = layer in dense_layer_ids
            dsa_mode = "none"
            if dsa_types:
                scheduled_mode = dsa_types[layer]
                if scheduled_mode in {"full", "shared"}:
                    dsa_mode = scheduled_mode
                if skip_dsa_composition and dsa_mode in {"full", "shared"}:
                    dsa_mode = "none"
                if dsa_mode == "shared" and previous_dsa_indices is None:
                    raise DecodeDriverError(
                        f"layer {layer} uses shared DSA indexer without a previous full layer"
                    )
            if dsa_mode in {"full", "shared"}:
                budget = budget_by_layer[layer]
                layer_dir = work_root / f"layer_{layer:04d}_dsa"
                attention_output = layer_dir / "attention_hidden.f32"
                attention_started = time.perf_counter()
                try:
                    attention = run_prefill_attention_block_batch(
                        runner_path=runner,
                        resident_layout_path=resident_layout,
                        cache_layout_path=cache_layout,
                        cache_file_path=cache_file,
                        layer=layer,
                        input_f32_path=current_input,
                        output_dir=layer_dir / "attention",
                        output_f32_path=attention_output,
                        start_position=position,
                        batch_tokens=1,
                        context_length=context_length,
                        num_heads=num_heads,
                        qk_nope_dim=qk_nope_dim,
                        rope_dim=rope_dim,
                        v_head_dim=v_head_dim,
                        kv_lora_dim=kv_lora_dim,
                        mla_kv_b_cache_dir=mla_kv_b_cache_path,
                        mla_key_cache=mla_key_cache,
                        cache_position_offset=cache_position_offset,
                        attention_scale=attention_scale,
                        rope_theta=rope_theta,
                        rope_interleave=rope_interleave,
                        dsa_indexer_mode=dsa_mode,
                        dsa_prev_indices_u32_path=previous_dsa_indices,
                        dsa_index_topk=dsa_index_topk,
                        dsa_index_n_heads=dsa_index_n_heads,
                        dsa_qk_rope_dim=dsa_qk_rope_dim or rope_dim,
                        dsa_rope_interleave=dsa_rope_interleave,
                        dsa_layer_norm_eps=dsa_layer_norm_eps,
                        write_dsa_future_cache=write_dsa_future_cache,
                        rms_norm_eps=rms_norm_eps,
                        max_cache_file_mib=max_cache_file_mib,
                        max_cache_write_mib=max_cache_write_mib,
                        max_cache_read_mib=max_cache_read_mib,
                        max_resident_matrix_mib=max_resident_matrix_mib,
                        max_runner_scratch_mib=max_runner_scratch_mib,
                        echo_runner_output=echo_runner_output,
                    )
                    attention_elapsed = time.perf_counter() - attention_started
                    mlp_started = time.perf_counter()
                    if attention.dsa_indices_u32_path is not None:
                        previous_dsa_indices = attention.dsa_indices_u32_path
                    if is_dense:
                        run_prefill_dense_mlp_block_batch(
                            runner_path=runner,
                            resident_layout_path=resident_layout,
                            layer=layer,
                            input_f32_path=attention.output_path,
                            output_dir=layer_dir / "dense_mlp",
                            output_f32_path=layer_output,
                            batch_tokens=1,
                            rms_norm_eps=rms_norm_eps,
                            max_resident_matrix_mib=max_resident_matrix_mib,
                            max_runner_scratch_mib=max_runner_scratch_mib,
                            echo_runner_output=echo_runner_output,
                        )
                    else:
                        run_prefill_routed_mlp_block_batch(
                            runner_path=runner,
                            expert_layout_path=expert_layout,
                            resident_layout_path=resident_layout,
                            layer=layer,
                            input_f32_path=attention.output_path,
                            output_dir=layer_dir / "routed_mlp",
                            output_f32_path=layer_output,
                            batch_tokens=1,
                            top_k=top_k,
                            max_k=max_k,
                            router_score=router_score,
                            routed_scaling_factor=routed_scaling_factor,
                            norm_topk_prob=norm_topk_prob,
                            no_norm_topk_prob=no_norm_topk_prob,
                            router_n_group=router_n_group,
                            router_topk_group=router_topk_group,
                            ignore_router_bias=ignore_router_bias,
                            include_shared_expert=include_shared_expert,
                            rms_norm_eps=rms_norm_eps,
                            max_slot_mib=max_slot_mib,
                            max_router_mib=max_router_mib,
                            max_runner_scratch_mib=max_runner_scratch_mib,
                            expert_read_advise_merge_gap_kib=expert_read_advise_merge_gap_kib,
                            expert_read_advise_align_kib=expert_read_advise_align_kib,
                            keep_token_files=keep_work_dir,
                            echo_runner_output=echo_runner_output,
                        )
                    mlp_elapsed = time.perf_counter() - mlp_started
                except PrefillExecuteError as exc:
                    raise DecodeDriverError(str(exc)) from exc
                mla_timing = {
                    name: value
                    for name, value in {
                        "input": (
                            attention.mla_attention.mla_timing_input_elapsed_seconds
                        ),
                        "cache_read": (
                            attention.mla_attention.mla_timing_cache_read_elapsed_seconds
                        ),
                        "value_read": (
                            attention.mla_attention.mla_timing_value_read_elapsed_seconds
                        ),
                        "metal_setup": (
                            attention.mla_attention.mla_timing_metal_setup_elapsed_seconds
                        ),
                        "kernel": (
                            attention.mla_attention.mla_timing_kernel_elapsed_seconds
                        ),
                        "write": (
                            attention.mla_attention.mla_timing_write_elapsed_seconds
                        ),
                        "total": (
                            attention.mla_attention.mla_timing_total_elapsed_seconds
                        ),
                    }.items()
                    if value is not None
                }
                mla_diagnostics: dict[str, object] = {
                    "key_cache_bytes": attention.mla_attention.mla_key_cache_bytes,
                    "value_cache_bytes": attention.mla_attention.mla_value_cache_bytes,
                    "estimated_peak_bytes": (
                        attention.mla_attention.estimated_peak_bytes
                    ),
                    "key_cache_enabled": attention.mla_attention.mla_key_cache,
                    "value_cache_enabled": attention.mla_attention.mla_value_cache,
                }
                records.append(
                    DecodeLayerRecord(
                        layer=layer,
                        kind="dense" if is_dense else "moe",
                        input_path=current_input,
                        output_path=layer_output,
                        command=attention.mla_attention.command,
                        expert_read_bytes=budget.read_bytes_per_token,
                        attention_read_bytes=budget.attention_read_bytes_per_token,
                        cache_read_bytes=budget.decoder_cache_read_bytes,
                        dsa_index_cache_read_bytes=budget.dsa_index_cache_read_bytes,
                        mla_cache_read_bytes=budget.decoder_mla_cache_read_bytes,
                        estimated_peak_bytes=budget.estimated_peak_bytes,
                        router_stage_peak_bytes=budget.router_stage_peak_bytes,
                        moe_stage_peak_bytes=budget.moe_stage_peak_bytes,
                        elapsed_seconds=time.perf_counter() - layer_started,
                        attention_elapsed_seconds=attention_elapsed,
                        mlp_elapsed_seconds=mlp_elapsed,
                        attention_stage_elapsed_seconds={
                            "projections": attention.projections_elapsed_seconds,
                            "cache_write": attention.cache_write_elapsed_seconds,
                            "rope": attention.rope_elapsed_seconds,
                            "dsa_indexer": attention.dsa_indexer_elapsed_seconds,
                            "mla_attention": attention.mla_attention_elapsed_seconds,
                            "attention_output": attention.attention_output_elapsed_seconds,
                        },
                        mla_attention_timing_elapsed_seconds=mla_timing,
                        mla_attention_diagnostics=mla_diagnostics,
                        composed=True,
                        dsa_indexer_mode=dsa_mode,
                        dsa_rope_interleave=attention.dsa_rope_interleave,
                        dsa_indices_u32_path=attention.dsa_indices_u32_path,
                    )
                )
                current_input = layer_output
                continue
            cmd = [str(runner)]
            if is_dense:
                cmd.extend(["--run-dense-decoder-layer"])
            else:
                cmd.extend(["--layout", str(expert_layout), "--run-decoder-layer"])
            cmd.extend(
                [
                    "--resident-layout",
                    str(resident_layout),
                    "--cache-layout",
                    str(cache_layout),
                    "--cache-file",
                    str(cache_file),
                    "--layer",
                    str(layer),
                    "--input-f32",
                    str(current_input),
                    "--position",
                    str(position),
                    "--context-length",
                    str(context_length),
                    "--num-heads",
                    str(num_heads),
                    "--qk-nope-dim",
                    str(qk_nope_dim),
                    "--rope-dim",
                    str(rope_dim),
                    "--v-head-dim",
                    str(v_head_dim),
                    "--rms-norm-eps",
                    _num(rms_norm_eps),
                    "--output-f32",
                    str(layer_output),
                    "--max-cache-file-mib",
                    _num(max_cache_file_mib),
                    "--max-cache-read-mib",
                    _num(max_cache_read_mib),
                    "--max-resident-matrix-mib",
                    _num(max_resident_matrix_mib),
                    "--max-runner-scratch-mib",
                    _num(max_runner_scratch_mib),
                ]
            )
            if not is_dense:
                cmd.extend(
                    [
                        "--top-k",
                        str(top_k),
                        "--max-k",
                        str(max_k),
                        "--router-score",
                        router_score,
                        "--max-slot-mib",
                        _num(max_slot_mib),
                        "--max-router-mib",
                        _num(max_router_mib),
                    ]
                )
            if kv_lora_dim is not None:
                cmd.extend(["--kv-lora-dim", str(kv_lora_dim)])
            if mla_kv_b_cache_path is not None:
                cmd.extend(["--mla-kv-b-cache-dir", str(mla_kv_b_cache_path)])
            if cache_position_offset:
                cmd.extend(["--cache-position-offset", str(cache_position_offset)])
            _optional_float_flag(cmd, "--attention-scale", attention_scale)
            if rope_theta != 10000.0:
                cmd.extend(["--rope-theta", _num(rope_theta)])
            if rope_interleave:
                cmd.append("--rope-interleave")
            if not is_dense:
                _optional_float_flag(cmd, "--routed-scaling-factor", routed_scaling_factor)
                if norm_topk_prob:
                    cmd.append("--norm-topk-prob")
                if no_norm_topk_prob:
                    cmd.append("--no-norm-topk-prob")
                _optional_int_flag(cmd, "--router-n-group", router_n_group)
                _optional_int_flag(cmd, "--router-topk-group", router_topk_group)
                if ignore_router_bias:
                    cmd.append("--ignore-router-bias")
                if include_shared_expert:
                    cmd.append("--include-shared-expert")
                if expert_read_advise_merge_gap_kib > 0:
                    cmd.extend(
                        [
                            "--expert-read-advise-merge-gap-kib",
                            str(expert_read_advise_merge_gap_kib),
                        ]
                    )
                if expert_read_advise_align_kib > 0:
                    cmd.extend(
                        [
                            "--expert-read-advise-align-kib",
                            str(expert_read_advise_align_kib),
                        ]
                    )

            _run_command(cmd, echo_output=echo_runner_output)
            budget = budget_by_layer[layer]
            records.append(
                DecodeLayerRecord(
                    layer=layer,
                    kind="dense" if is_dense else "moe",
                    input_path=current_input,
                    output_path=layer_output,
                    command=tuple(cmd),
                    expert_read_bytes=budget.read_bytes_per_token,
                    attention_read_bytes=budget.attention_read_bytes_per_token,
                    cache_read_bytes=budget.decoder_cache_read_bytes,
                    dsa_index_cache_read_bytes=budget.dsa_index_cache_read_bytes,
                    mla_cache_read_bytes=budget.decoder_mla_cache_read_bytes,
                    estimated_peak_bytes=budget.estimated_peak_bytes,
                    router_stage_peak_bytes=budget.router_stage_peak_bytes,
                    moe_stage_peak_bytes=budget.moe_stage_peak_bytes,
                    elapsed_seconds=time.perf_counter() - layer_started,
                )
            )
            current_input = layer_output
    except Exception:
        if keep_work_dir or not created_work_dir:
            keep_work_dir = True
        raise
    finally:
        if created_work_dir and not keep_work_dir:
            shutil.rmtree(work_root, ignore_errors=True)

    return build_result()
