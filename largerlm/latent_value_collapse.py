from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .context1_o_proj_cache import Context1OProjCacheError, plan_context1_o_proj_cache


SCHEMA = "largerlm.latent_value_collapse_plan.v1"


class LatentValueCollapseError(RuntimeError):
    """Raised when the GLM latent-value collapse plan cannot be derived."""


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator > 0 else None


@dataclass(frozen=True)
class LatentValueCollapsePlan:
    prepared_dir: Path
    dtype: str
    dtype_bytes: int
    layer_count: int
    layers: tuple[int, ...]
    hidden_dim: int
    attention_value_dim: int
    num_heads: int
    v_head_dim: int
    kv_lora_dim: int
    current_o_proj_bytes_per_token: int
    context1_cache_bytes: int
    context1_cache_read_ratio: float | None
    context1_runtime_fma_per_token: int
    context1_runtime_fma_ratio: float | None
    exact_per_head_cache_bytes: int
    exact_per_head_cache_read_ratio: float | None
    exact_per_head_int4_floor_bytes: int
    exact_per_head_int4_floor_read_ratio: float | None
    exact_per_head_runtime_fma_per_token: int
    exact_per_head_runtime_fma_ratio: float | None
    current_o_proj_runtime_fma_per_token: int
    break_even_shared_head_groups: int
    independent_attention_head_groups: int
    exact_all_context_cache_recommended: bool
    decision: str

    def to_report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = SCHEMA
        payload["prepared_dir"] = str(self.prepared_dir)
        payload["layers"] = list(self.layers)
        payload["dims"] = {
            "hidden_dim": self.hidden_dim,
            "attention_value_dim": self.attention_value_dim,
            "num_heads": self.num_heads,
            "v_head_dim": self.v_head_dim,
            "kv_lora_dim": self.kv_lora_dim,
        }
        payload["bytes"] = {
            "current_o_proj_per_token": self.current_o_proj_bytes_per_token,
            "context1_cache": self.context1_cache_bytes,
            "exact_per_head_cache": self.exact_per_head_cache_bytes,
            "exact_per_head_int4_floor": self.exact_per_head_int4_floor_bytes,
        }
        payload["ratios"] = {
            "context1_cache_read": self.context1_cache_read_ratio,
            "context1_runtime_fma": self.context1_runtime_fma_ratio,
            "exact_per_head_cache_read": self.exact_per_head_cache_read_ratio,
            "exact_per_head_int4_floor_read": (
                self.exact_per_head_int4_floor_read_ratio
            ),
            "exact_per_head_runtime_fma": self.exact_per_head_runtime_fma_ratio,
        }
        payload["constraints"] = {
            "break_even_shared_head_groups": self.break_even_shared_head_groups,
            "independent_attention_head_groups": self.independent_attention_head_groups,
            "exact_all_context_cache_recommended": (
                self.exact_all_context_cache_recommended
            ),
        }
        return payload


def _decision(
    *,
    num_heads: int,
    break_even_shared_head_groups: int,
    exact_per_head_read_ratio: float | None,
    exact_per_head_int4_floor_read_ratio: float | None,
    exact_per_head_fma_ratio: float | None,
) -> str:
    if num_heads <= break_even_shared_head_groups:
        return (
            "exact_all_context_candidate: attention weights would need to be "
            "shared across no more than the break-even head groups"
        )
    read = (
        f"{exact_per_head_read_ratio:.2f}x"
        if exact_per_head_read_ratio is not None
        else "unknown"
    )
    int4 = (
        f"{exact_per_head_int4_floor_read_ratio:.2f}x"
        if exact_per_head_int4_floor_read_ratio is not None
        else "unknown"
    )
    fma = (
        f"{exact_per_head_fma_ratio:.2f}x"
        if exact_per_head_fma_ratio is not None
        else "unknown"
    )
    return (
        "exact_all_context_per_head_cache_not_viable: GLM decode has "
        f"{num_heads} independent attention heads, above the "
        f"{break_even_shared_head_groups} break-even shared-head groups; exact "
        f"per-head BF16 cache reads would be {read} current o_proj bytes/token "
        f"(ideal int4 floor {int4}) and {fma} current o_proj FMA"
    )


def build_latent_value_collapse_plan(
    prepared_dir: str | Path,
    *,
    dtype: str = "BF16",
    layers: tuple[int, ...] | None = None,
) -> LatentValueCollapsePlan:
    try:
        context1 = plan_context1_o_proj_cache(
            prepared_dir,
            dtype=dtype,
            layers=layers,
        )
    except Context1OProjCacheError as exc:
        raise LatentValueCollapseError(str(exc)) from exc

    layer_count = len(context1.layers)
    if layer_count <= 0:
        raise LatentValueCollapseError("no collapsible layers found")

    context1_runtime_fma = layer_count * context1.hidden_dim * context1.kv_lora_dim
    current_runtime_fma = (
        layer_count * context1.hidden_dim * context1.attention_value_dim
    )
    exact_per_head_runtime_fma = (
        layer_count
        * context1.hidden_dim
        * context1.num_heads
        * context1.kv_lora_dim
    )
    exact_per_head_cache_bytes = context1.total_bytes * context1.num_heads
    exact_per_head_elements = (
        layer_count
        * context1.hidden_dim
        * context1.num_heads
        * context1.kv_lora_dim
    )
    exact_per_head_int4_floor = math.ceil(exact_per_head_elements / 2)
    current_bytes = context1.current_o_proj_storage_per_token
    break_even_groups = max(0, current_bytes // context1.total_bytes)
    exact_read_ratio = _ratio(exact_per_head_cache_bytes, current_bytes)
    exact_int4_ratio = _ratio(exact_per_head_int4_floor, current_bytes)
    exact_fma_ratio = _ratio(exact_per_head_runtime_fma, current_runtime_fma)
    recommended = (
        exact_read_ratio is not None
        and exact_read_ratio < 1.0
        and exact_fma_ratio is not None
        and exact_fma_ratio <= 1.0
    )

    return LatentValueCollapsePlan(
        prepared_dir=Path(prepared_dir),
        dtype=context1.dtype,
        dtype_bytes=context1.dtype_bytes,
        layer_count=layer_count,
        layers=tuple(layer.layer for layer in context1.layers),
        hidden_dim=context1.hidden_dim,
        attention_value_dim=context1.attention_value_dim,
        num_heads=context1.num_heads,
        v_head_dim=context1.v_head_dim,
        kv_lora_dim=context1.kv_lora_dim,
        current_o_proj_bytes_per_token=current_bytes,
        context1_cache_bytes=context1.total_bytes,
        context1_cache_read_ratio=_ratio(context1.total_bytes, current_bytes),
        context1_runtime_fma_per_token=context1_runtime_fma,
        context1_runtime_fma_ratio=_ratio(context1_runtime_fma, current_runtime_fma),
        exact_per_head_cache_bytes=exact_per_head_cache_bytes,
        exact_per_head_cache_read_ratio=exact_read_ratio,
        exact_per_head_int4_floor_bytes=exact_per_head_int4_floor,
        exact_per_head_int4_floor_read_ratio=exact_int4_ratio,
        exact_per_head_runtime_fma_per_token=exact_per_head_runtime_fma,
        exact_per_head_runtime_fma_ratio=exact_fma_ratio,
        current_o_proj_runtime_fma_per_token=current_runtime_fma,
        break_even_shared_head_groups=break_even_groups,
        independent_attention_head_groups=context1.num_heads,
        exact_all_context_cache_recommended=recommended,
        decision=_decision(
            num_heads=context1.num_heads,
            break_even_shared_head_groups=break_even_groups,
            exact_per_head_read_ratio=exact_read_ratio,
            exact_per_head_int4_floor_read_ratio=exact_int4_ratio,
            exact_per_head_fma_ratio=exact_fma_ratio,
        ),
    )
