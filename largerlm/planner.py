from __future__ import annotations

import math
import operator
from dataclasses import dataclass

from .config import ModelConfig
from .routed_read import (
    format_routed_read_guard_flag_float,
    suggest_decode_routed_read_guard_flags,
)
from .safetensors import CheckpointStats


class PlannerError(RuntimeError):
    """Raised when model planning inputs would produce unsafe estimates."""


GIB = 1024**3
DEFAULT_SYSTEM_RESERVE_BYTES = 16 * GIB
LARGE_UNIFIED_MEMORY_THRESHOLD_BYTES = 96 * GIB
LARGE_UNIFIED_MEMORY_SYSTEM_RESERVE_BYTES = 24 * GIB


def default_system_reserve_bytes(unified_memory_bytes: int | None) -> int:
    """Choose a conservative live-memory reserve from known unified memory size."""

    if (
        unified_memory_bytes is not None
        and unified_memory_bytes >= LARGE_UNIFIED_MEMORY_THRESHOLD_BYTES
    ):
        return LARGE_UNIFIED_MEMORY_SYSTEM_RESERVE_BYTES
    return DEFAULT_SYSTEM_RESERVE_BYTES


def _numeric_limit(name: str, value: float | int) -> float:
    if isinstance(value, bool):
        raise PlannerError(f"{name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PlannerError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise PlannerError(f"{name} must be finite")
    return parsed


def _positive_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed <= 0:
        raise PlannerError(f"{name} must be positive")
    return parsed


def _nonnegative_limit(name: str, value: float | int) -> float:
    parsed = _numeric_limit(name, value)
    if parsed < 0:
        raise PlannerError(f"{name} must be non-negative")
    return parsed


def _integer_limit(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise PlannerError(f"{name} must be an integer")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise PlannerError(f"{name} must be an integer") from exc


def _positive_integer_limit(name: str, value: object) -> int:
    parsed = _integer_limit(name, value)
    if parsed <= 0:
        raise PlannerError(f"{name} must be positive")
    return parsed


def _nonnegative_integer_limit(name: str, value: object) -> int:
    parsed = _integer_limit(name, value)
    if parsed < 0:
        raise PlannerError(f"{name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class QuantizedMatrixLayout:
    out_dim: int
    in_dim: int
    weight_bits: int = 4
    group_size: int = 64
    scale_bytes: int = 2
    bias_bytes: int = 2

    @property
    def weight_bytes(self) -> int:
        return math.ceil(self.out_dim * self.in_dim * self.weight_bits / 8)

    @property
    def groups_per_row(self) -> int:
        return math.ceil(self.in_dim / self.group_size)

    @property
    def metadata_bytes(self) -> int:
        return self.out_dim * self.groups_per_row * (self.scale_bytes + self.bias_bytes)

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.metadata_bytes


@dataclass(frozen=True)
class ExpertLayout:
    hidden_size: int
    intermediate_size: int
    weight_bits: int = 4
    group_size: int = 64
    scale_bytes: int = 2
    bias_bytes: int = 2

    @property
    def gate(self) -> QuantizedMatrixLayout:
        return QuantizedMatrixLayout(
            self.intermediate_size,
            self.hidden_size,
            self.weight_bits,
            self.group_size,
            self.scale_bytes,
            self.bias_bytes,
        )

    @property
    def up(self) -> QuantizedMatrixLayout:
        return self.gate

    @property
    def down(self) -> QuantizedMatrixLayout:
        return QuantizedMatrixLayout(
            self.hidden_size,
            self.intermediate_size,
            self.weight_bits,
            self.group_size,
            self.scale_bytes,
            self.bias_bytes,
        )

    @property
    def total_bytes(self) -> int:
        return self.gate.total_bytes + self.up.total_bytes + self.down.total_bytes


@dataclass(frozen=True)
class CacheEstimate:
    bytes_per_token: int | None
    indexer_full_layers: int | None
    mla_cache_width: int | None = None
    indexer_bytes_per_token: int | None = None


@dataclass(frozen=True)
class ModelPlan:
    config: ModelConfig
    quant_bits: int
    group_size: int
    expert_layout: ExpertLayout
    routed_expert_disk_bytes_estimate: int
    routed_expert_read_bytes_per_decode_token: int
    checkpoint_stats: CheckpointStats | None
    cache_estimate: CacheEstimate
    resident_bytes_estimate: int | None
    unified_memory_bytes: int | None
    system_reserve_bytes: int
    runtime_buffer_bytes: int
    resident_memory_budget_bytes: int | None
    resident_memory_pressure_bytes: int | None
    resident_memory_headroom_bytes: int | None
    resident_memory_fits_budget: bool | None
    page_cache_budget_bytes: int | None
    max_context_tokens: int | None
    decode_cache_bytes_estimate: int | None
    decode_cache_budget_bytes: int | None
    decode_cache_fits_budget: bool | None
    decode_cache_safe_context_tokens: int | None
    suggested_prepare_flags: dict[str, object] | None
    cold_read_seconds_per_token: float | None
    suggested_launch_guard_flags: dict[str, object] | None
    suggested_decode_guard_flags: dict[str, object] | None
    suggested_launch_profile: dict[str, object] | None
    resident_bytes_estimate_source: str | None = None

    @property
    def using_checkpoint_scan(self) -> bool:
        return self.checkpoint_stats is not None


def estimate_mla_cache(config: ModelConfig, dtype_bytes: int = 2) -> CacheEstimate:
    """Estimate GLM/DeepSeek-style MLA resident KV bytes per token."""

    cache_width = config.mla_cache_width
    if cache_width is None:
        return CacheEstimate(bytes_per_token=None, indexer_full_layers=None)

    full_layers = None
    if config.indexer_types is not None:
        full_layers = sum(1 for item in config.indexer_types if item == "full")

    index_dim = config.index_head_dim or 0
    main = config.num_hidden_layers * cache_width
    index = (full_layers or 0) * index_dim
    return CacheEstimate(
        bytes_per_token=int((main + index) * dtype_bytes),
        indexer_full_layers=full_layers,
        mla_cache_width=cache_width,
        indexer_bytes_per_token=int(index * dtype_bytes),
    )


def _combine_plan_launch_profile(
    *,
    launch_guard_flags: dict[str, object] | None,
    decode_guard_flags: dict[str, object] | None,
) -> dict[str, object] | None:
    sections: dict[str, dict[str, object]] = {}
    argv: list[str] = []
    for section_name, flags in (
        ("launch_guard_flags", launch_guard_flags),
        ("decode_guard_flags", decode_guard_flags),
    ):
        if not isinstance(flags, dict):
            continue
        section_argv = flags.get("argv")
        if not isinstance(section_argv, (list, tuple)) or not section_argv:
            continue
        sections[section_name] = flags
        argv.extend(str(item) for item in section_argv)
    if not argv:
        return None
    return {
        "source": "plan",
        "argv": tuple(argv),
        "sections": sections,
        "argv_safe_to_replay": True,
    }


def _suggest_plan_launch_guard_flags(
    *,
    runtime_buffer_bytes: int,
    system_reserve_bytes: int,
) -> dict[str, object]:
    max_live_mib = runtime_buffer_bytes / 1024**2
    min_free_gib = system_reserve_bytes / 1024**3
    return {
        "source": "plan",
        "require_prepared_memory_profile": True,
        "recommended_max_live_working_set_bytes": int(runtime_buffer_bytes),
        "max_live_working_set_mib": max_live_mib,
        "recommended_min_free_unified_memory_bytes": int(system_reserve_bytes),
        "min_free_unified_memory_gib": min_free_gib,
        "recommended_required_available_memory_bytes": (
            int(runtime_buffer_bytes) + int(system_reserve_bytes)
        ),
        "argv": (
            "--require-prepared-memory-profile",
            "--max-live-working-set-mib",
            format_routed_read_guard_flag_float(max_live_mib),
            "--min-free-unified-memory-gib",
            format_routed_read_guard_flag_float(min_free_gib),
        ),
    }


def _format_gib_from_bytes_ceil(value: int) -> str:
    gib = math.ceil((int(value) / GIB) * 1_000_000) / 1_000_000
    return f"{gib:.6f}".rstrip("0").rstrip(".")


def _suggest_plan_prepare_flags(
    *,
    quant_bits: int,
    group_size: int,
    unified_memory_bytes: int | None,
    system_reserve_bytes: int,
    runtime_buffer_bytes: int,
    target_page_cache_fraction: float,
    decode_cache_budget_bytes: int | None,
    decode_cache_safe_context_tokens: int | None,
    cold_read_gib_per_second: float | None,
) -> dict[str, object] | None:
    if (
        decode_cache_budget_bytes is None
        or decode_cache_safe_context_tokens is None
        or decode_cache_safe_context_tokens <= 0
    ):
        return None
    argv: list[str] = [
        "--auto-context-from-budget",
        "--group-size",
        str(group_size),
        "--max-cache-gib",
        _format_gib_from_bytes_ceil(decode_cache_budget_bytes),
        "--system-reserve-gib",
        format_routed_read_guard_flag_float(system_reserve_bytes / GIB),
        "--runtime-buffer-gib",
        format_routed_read_guard_flag_float(runtime_buffer_bytes / GIB),
        "--page-cache-fraction",
        format_routed_read_guard_flag_float(target_page_cache_fraction),
    ]
    if unified_memory_bytes is not None:
        argv.extend(
            (
                "--unified-memory-gib",
                format_routed_read_guard_flag_float(unified_memory_bytes / GIB),
            )
        )
    if cold_read_gib_per_second is not None:
        argv.extend(
            (
                "--cold-read-gib-s",
                format_routed_read_guard_flag_float(cold_read_gib_per_second),
            )
        )
    return {
        "source": "plan",
        "argv": tuple(argv),
        "argv_safe_to_replay": True,
        "context_selection": "auto_context_from_budget",
        "quant_bits": quant_bits,
        "group_size": group_size,
        "decode_cache_budget_bytes": int(decode_cache_budget_bytes),
        "decode_cache_safe_context_tokens": int(decode_cache_safe_context_tokens),
        "max_cache_gib": decode_cache_budget_bytes / GIB,
        "unified_memory_bytes": unified_memory_bytes,
        "system_reserve_bytes": int(system_reserve_bytes),
        "runtime_buffer_bytes": int(runtime_buffer_bytes),
        "page_cache_fraction": float(target_page_cache_fraction),
        "cold_read_gib_per_second": cold_read_gib_per_second,
    }


def _config_weight_dtype_bytes(config: ModelConfig) -> int:
    return int(config.weight_dtype_bytes)


def estimate_config_resident_bytes(config: ModelConfig) -> int | None:
    """Conservatively estimate non-routed resident bytes from config shapes."""

    weight_bytes = _config_weight_dtype_bytes(config)
    total = 0
    hidden = int(config.hidden_size)
    vocab = config.vocab_size
    if vocab is not None:
        total += vocab * hidden * weight_bytes
        if config.tie_word_embeddings is not True:
            total += vocab * hidden * weight_bytes
    total += hidden * weight_bytes

    if (
        config.q_lora_rank is not None
        and config.kv_lora_rank is not None
        and config.num_attention_heads is not None
        and config.qk_nope_head_dim is not None
        and config.qk_rope_head_dim is not None
        and config.v_head_dim is not None
    ):
        q_lora = int(config.q_lora_rank)
        kv_lora = int(config.kv_lora_rank)
        heads = int(config.num_attention_heads)
        q_out = heads * (int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim))
        kv_a_out = kv_lora + int(config.qk_rope_head_dim)
        kv_b_out = heads * (int(config.qk_nope_head_dim) + int(config.v_head_dim))
        value_out = heads * int(config.v_head_dim)
        attention_per_layer = (
            hidden
            + q_lora * hidden
            + q_lora
            + q_out * q_lora
            + kv_a_out * hidden
            + kv_lora
            + kv_b_out * kv_lora
            + hidden * value_out
            + hidden
        )
        total += config.num_hidden_layers * attention_per_layer * weight_bytes

    moe_layer_set = set(config.moe_layers)
    dense_layers = config.num_hidden_layers - len(moe_layer_set)
    if dense_layers > 0 and config.intermediate_size is not None:
        dense_intermediate = int(config.intermediate_size)
        total += dense_layers * (
            dense_intermediate * hidden
            + dense_intermediate * hidden
            + hidden * dense_intermediate
        ) * weight_bytes

    if config.n_shared_experts and config.n_shared_experts > 0:
        shared_width = int(config.n_shared_experts) * config.moe_hidden_size
        total += len(moe_layer_set) * (
            shared_width * hidden
            + shared_width * hidden
            + hidden * shared_width
        ) * weight_bytes

    if config.n_routed_experts is not None:
        router_weight_bytes = max(4, weight_bytes)
        router_bias_bytes = max(4, weight_bytes)
        total += len(moe_layer_set) * int(config.n_routed_experts) * (
            hidden * router_weight_bytes + router_bias_bytes
        )

    if (
        config.indexer_types is not None
        and config.index_head_dim is not None
        and config.index_n_heads is not None
        and config.q_lora_rank is not None
    ):
        full_layers = sum(1 for item in config.indexer_types if item == "full")
        index_head_dim = int(config.index_head_dim)
        index_n_heads = int(config.index_n_heads)
        q_lora = int(config.q_lora_rank)
        indexer_per_full_layer = (
            index_head_dim * hidden
            + index_n_heads * index_head_dim * q_lora
            + index_n_heads * hidden
            + 2 * index_head_dim
        )
        total += full_layers * indexer_per_full_layer * weight_bytes

    return int(total) if total > 0 else None


def build_plan(
    config: ModelConfig,
    *,
    quant_bits: int = 4,
    group_size: int = 64,
    checkpoint_stats: CheckpointStats | None = None,
    unified_memory_bytes: int | None = None,
    system_reserve_bytes: int | None = None,
    runtime_buffer_bytes: int = 8 * GIB,
    target_page_cache_fraction: float = 0.60,
    cold_read_gib_per_second: float | None = None,
    max_context_tokens: int | None = None,
    max_cache_bytes: int | None = None,
) -> ModelPlan:
    quant_bits = _positive_integer_limit("quant_bits", quant_bits)
    group_size = _positive_integer_limit("group_size", group_size)
    runtime_buffer_bytes = _positive_integer_limit(
        "runtime_buffer_bytes",
        runtime_buffer_bytes,
    )
    target_page_cache_fraction = _numeric_limit(
        "target_page_cache_fraction",
        target_page_cache_fraction,
    )
    if target_page_cache_fraction < 0 or target_page_cache_fraction > 1:
        raise PlannerError("target_page_cache_fraction must be between 0 and 1")
    if unified_memory_bytes is not None:
        unified_memory_bytes = _positive_integer_limit(
            "unified_memory_bytes",
            unified_memory_bytes,
        )
    if system_reserve_bytes is not None:
        system_reserve_bytes = _nonnegative_integer_limit(
            "system_reserve_bytes",
            system_reserve_bytes,
        )
    if max_context_tokens is not None:
        max_context_tokens = _positive_integer_limit(
            "max_context_tokens",
            max_context_tokens,
        )
    if max_cache_bytes is not None:
        max_cache_bytes = _nonnegative_integer_limit(
            "max_cache_bytes",
            max_cache_bytes,
        )
    if cold_read_gib_per_second is not None:
        cold_read_gib_per_second = _positive_limit(
            "cold_read_gib_per_second",
            cold_read_gib_per_second,
        )
    resolved_system_reserve_bytes = (
        default_system_reserve_bytes(unified_memory_bytes)
        if system_reserve_bytes is None
        else int(system_reserve_bytes)
    )
    expert_layout = ExpertLayout(
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_hidden_size,
        weight_bits=quant_bits,
        group_size=group_size,
    )

    routed_disk_est = (
        config.num_moe_layers * config.routed_experts * expert_layout.total_bytes
    )
    per_token_read = (
        config.num_moe_layers * config.experts_per_token * expert_layout.total_bytes
    )

    resident_source = None
    if checkpoint_stats:
        resident = checkpoint_stats.resident_bytes
        resident_source = "checkpoint_scan"
    else:
        resident = estimate_config_resident_bytes(config)
        resident_source = "config_estimate" if resident is not None else None
    page_cache_budget = None
    decode_cache_budget = max_cache_bytes
    resident_memory_budget = None
    resident_memory_pressure = None
    resident_memory_headroom = None
    resident_memory_fits = None
    if unified_memory_bytes is not None:
        resident_for_budget = resident or 0
        resident_memory_budget = max(
            0,
            unified_memory_bytes
            - resolved_system_reserve_bytes
            - runtime_buffer_bytes
        )
        resident_memory_pressure = (
            resolved_system_reserve_bytes + runtime_buffer_bytes + resident_for_budget
        )
        resident_memory_headroom = unified_memory_bytes - resident_memory_pressure
        resident_memory_fits = resident_memory_headroom >= 0
        page_cache_budget = max(
            0,
            int(resident_memory_headroom * target_page_cache_fraction),
        )
        if decode_cache_budget is None:
            decode_cache_budget = max(0, int(resident_memory_headroom - page_cache_budget))

    cache_estimate = estimate_mla_cache(config)
    decode_cache_total = None
    decode_cache_fits = None
    safe_context = None
    if (
        decode_cache_budget is not None
        and cache_estimate.bytes_per_token is not None
        and cache_estimate.bytes_per_token > 0
    ):
        safe_context = max(0, decode_cache_budget // cache_estimate.bytes_per_token)
    if max_context_tokens is not None and cache_estimate.bytes_per_token is not None:
        decode_cache_total = int(max_context_tokens) * cache_estimate.bytes_per_token
        if decode_cache_budget is not None:
            decode_cache_fits = decode_cache_total <= decode_cache_budget

    cold_seconds = None
    if cold_read_gib_per_second and cold_read_gib_per_second > 0:
        cold_seconds = per_token_read / (cold_read_gib_per_second * 1024**3)
    suggested_prepare_flags = _suggest_plan_prepare_flags(
        quant_bits=quant_bits,
        group_size=group_size,
        unified_memory_bytes=unified_memory_bytes,
        system_reserve_bytes=resolved_system_reserve_bytes,
        runtime_buffer_bytes=runtime_buffer_bytes,
        target_page_cache_fraction=target_page_cache_fraction,
        decode_cache_budget_bytes=decode_cache_budget,
        decode_cache_safe_context_tokens=safe_context,
        cold_read_gib_per_second=cold_read_gib_per_second,
    )
    suggested_decode_guard_flags = suggest_decode_routed_read_guard_flags(
        read_bytes_per_token=per_token_read,
        ssd_read_gib_per_second=cold_read_gib_per_second,
        source="plan",
    )
    suggested_launch_guard_flags = _suggest_plan_launch_guard_flags(
        runtime_buffer_bytes=runtime_buffer_bytes,
        system_reserve_bytes=resolved_system_reserve_bytes,
    )
    suggested_launch_profile = _combine_plan_launch_profile(
        launch_guard_flags=suggested_launch_guard_flags,
        decode_guard_flags=suggested_decode_guard_flags,
    )

    return ModelPlan(
        config=config,
        quant_bits=quant_bits,
        group_size=group_size,
        expert_layout=expert_layout,
        routed_expert_disk_bytes_estimate=routed_disk_est,
        routed_expert_read_bytes_per_decode_token=per_token_read,
        checkpoint_stats=checkpoint_stats,
        cache_estimate=cache_estimate,
        resident_bytes_estimate=resident,
        resident_bytes_estimate_source=resident_source,
        unified_memory_bytes=unified_memory_bytes,
        system_reserve_bytes=resolved_system_reserve_bytes,
        runtime_buffer_bytes=runtime_buffer_bytes,
        resident_memory_budget_bytes=resident_memory_budget,
        resident_memory_pressure_bytes=resident_memory_pressure,
        resident_memory_headroom_bytes=resident_memory_headroom,
        resident_memory_fits_budget=resident_memory_fits,
        page_cache_budget_bytes=page_cache_budget,
        max_context_tokens=max_context_tokens,
        decode_cache_bytes_estimate=decode_cache_total,
        decode_cache_budget_bytes=decode_cache_budget,
        decode_cache_fits_budget=decode_cache_fits,
        decode_cache_safe_context_tokens=safe_context,
        suggested_prepare_flags=suggested_prepare_flags,
        cold_read_seconds_per_token=cold_seconds,
        suggested_launch_guard_flags=suggested_launch_guard_flags,
        suggested_decode_guard_flags=suggested_decode_guard_flags,
        suggested_launch_profile=suggested_launch_profile,
    )
