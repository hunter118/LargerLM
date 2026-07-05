from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ConfigError(ValueError):
    """Raised when a model config is missing fields needed for planning."""


@dataclass(frozen=True)
class ModelConfig:
    """Subset of a Hugging Face config needed for MoE memory and I/O planning."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    weight_dtype: str = "bfloat16"
    weight_dtype_bytes: int = 2
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    max_position_embeddings: int | None = None
    rms_norm_eps: float | None = None
    rope_theta: float | None = None
    index_head_dim: int | None = None
    index_n_heads: int | None = None
    index_topk: int | None = None
    indexer_rope_interleave: bool = False
    rope_interleave: bool = False
    scoring_func: str | None = None
    topk_method: str | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    n_group: int | None = None
    topk_group: int | None = None
    indexer_types: tuple[str, ...] | None = None
    mlp_layer_types: tuple[str, ...] | None = None
    tie_word_embeddings: bool | None = None
    raw: dict[str, Any] | None = None

    @property
    def moe_layers(self) -> list[int]:
        if self.mlp_layer_types is not None:
            if len(self.mlp_layer_types) != self.num_hidden_layers:
                raise ConfigError(
                    "config mlp_layer_types length does not match num_hidden_layers"
                )
            return [
                layer
                for layer, layer_type in enumerate(self.mlp_layer_types)
                if layer_type in {"sparse", "moe", "moe_sparse"}
            ]
        layers: list[int] = []
        freq = max(1, int(self.moe_layer_freq or 1))
        for layer in range(self.num_hidden_layers):
            if layer < int(self.first_k_dense_replace or 0):
                continue
            if (layer - int(self.first_k_dense_replace or 0)) % freq == 0:
                layers.append(layer)
        return layers

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layers)

    @property
    def routed_experts(self) -> int:
        if self.n_routed_experts is None:
            raise ConfigError("config is missing n_routed_experts")
        return int(self.n_routed_experts)

    @property
    def experts_per_token(self) -> int:
        if self.num_experts_per_tok is None:
            raise ConfigError("config is missing num_experts_per_tok")
        return int(self.num_experts_per_tok)

    @property
    def moe_hidden_size(self) -> int:
        if self.moe_intermediate_size is not None:
            return int(self.moe_intermediate_size)
        if self.intermediate_size is not None:
            return int(self.intermediate_size)
        raise ConfigError("config is missing moe_intermediate_size/intermediate_size")

    @property
    def attention_q_head_dim(self) -> int | None:
        if self.qk_nope_head_dim is None or self.qk_rope_head_dim is None:
            return None
        return int(self.qk_nope_head_dim) + int(self.qk_rope_head_dim)

    @property
    def attention_q_projection_output_dim(self) -> int | None:
        q_head = self.attention_q_head_dim
        if self.num_attention_heads is None or q_head is None:
            return None
        return int(self.num_attention_heads) * q_head

    @property
    def attention_kv_a_output_dim(self) -> int | None:
        if self.kv_lora_rank is None or self.qk_rope_head_dim is None:
            return None
        return int(self.kv_lora_rank) + int(self.qk_rope_head_dim)

    @property
    def attention_kv_b_output_dim(self) -> int | None:
        if (
            self.num_attention_heads is None
            or self.qk_nope_head_dim is None
            or self.v_head_dim is None
        ):
            return None
        return int(self.num_attention_heads) * (
            int(self.qk_nope_head_dim) + int(self.v_head_dim)
        )

    @property
    def attention_value_output_dim(self) -> int | None:
        if self.num_attention_heads is None or self.v_head_dim is None:
            return None
        return int(self.num_attention_heads) * int(self.v_head_dim)

    @property
    def mla_cache_width(self) -> int | None:
        return self.attention_kv_a_output_dim

    @property
    def dsa_full_indexer_q_output_dim(self) -> int | None:
        if self.index_head_dim is None or self.index_n_heads is None:
            return None
        return int(self.index_head_dim) * int(self.index_n_heads)


def _get_first(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def _as_int(
    value: Any,
    default: int | None = None,
    *,
    field: str = "integer field",
) -> int | None:
    if value is None:
        return default
    if type(value) is not int:
        raise ConfigError(f"config {field} must be an integer")
    return int(value)


def _as_float(
    value: Any,
    default: float | None = None,
    *,
    field: str = "numeric field",
) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"config {field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ConfigError(f"config {field} must be finite")
    return parsed


def _as_bool(value: Any, default: bool = False, *, field: str = "boolean field") -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"config {field} must be a boolean")
    return value


def _as_optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ConfigError(f"config {field} must be a boolean")
    return bool(value)


def _as_optional_str(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"config {field} must be a string")
    text = value.strip()
    if not text:
        raise ConfigError(f"config {field} must be non-empty")
    return text


def _model_type(raw: dict[str, Any]) -> str:
    value = raw.get("model_type")
    if value is None:
        return "unknown"
    return _as_optional_str(value, field="model_type") or "unknown"


def _eos_token_ids(raw: dict[str, Any]) -> tuple[int, ...]:
    value = raw.get("eos_token_id")
    if value is None:
        return ()

    def parse_token(item: Any, *, list_item: bool = False) -> int:
        if type(item) is not int:
            if list_item:
                raise ConfigError("config eos_token_id entries must be integers")
            raise ConfigError(
                "config eos_token_id must be an integer or a list of integers"
            )
        token = int(item)
        if token < 0:
            if list_item:
                raise ConfigError("config eos_token_id entries must be non-negative")
            raise ConfigError("config eos_token_id must be non-negative")
        return token

    if isinstance(value, list):
        return tuple(dict.fromkeys(parse_token(item, list_item=True) for item in value))
    return (parse_token(value),)


def _weight_dtype(raw: dict[str, Any]) -> tuple[str, int]:
    value = _get_first(raw, ("dtype", "torch_dtype"))
    if value is None:
        return ("bfloat16", 2)
    if not isinstance(value, str):
        raise ConfigError("config dtype must be a string")
    text = value.strip().lower()
    if text.startswith("torch."):
        text = text.removeprefix("torch.")
    aliases = {
        "bfloat16": ("bfloat16", 2),
        "bf16": ("bfloat16", 2),
        "float16": ("float16", 2),
        "fp16": ("float16", 2),
        "f16": ("float16", 2),
        "half": ("float16", 2),
        "float32": ("float32", 4),
        "fp32": ("float32", 4),
        "f32": ("float32", 4),
        "single": ("float32", 4),
        "float64": ("float64", 8),
        "fp64": ("float64", 8),
        "f64": ("float64", 8),
        "double": ("float64", 8),
    }
    if text not in aliases:
        raise ConfigError(
            "config dtype must be one of bfloat16, float16, float32, or float64"
        )
    return aliases[text]


def _rope_theta(raw: dict[str, Any]) -> float | None:
    params = raw.get("rope_parameters")
    if isinstance(params, dict):
        theta = _as_float(params.get("rope_theta"), field="rope_parameters.rope_theta")
        if theta is not None:
            return theta
    return _as_float(_get_first(raw, ("rope_theta", "rope_base")), field="rope_theta")


def _indexer_types(raw: dict[str, Any], num_layers: int) -> tuple[str, ...] | None:
    def validate_schedule_order(
        values: tuple[str, ...],
        field: str,
    ) -> tuple[str, ...]:
        bad_layer = first_shared_indexer_without_previous_full(values)
        if bad_layer is not None:
            raise ConfigError(
                f"config {field} has shared DSA indexer at layer {bad_layer} "
                "before any full indexer layer"
            )
        return values

    def normalize(value: Any) -> str:
        item = str(value).lower()
        aliases = {"f": "full", "s": "shared", "n": "none"}
        item = aliases.get(item, item)
        if item not in {"full", "shared", "none"}:
            raise ConfigError(f"invalid config DSA indexer type {value!r}")
        return item

    def validate_length(
        values: tuple[str, ...],
        field: str,
    ) -> tuple[str, ...]:
        if len(values) != num_layers:
            raise ConfigError(
                f"config {field} length does not match num_hidden_layers"
            )
        return validate_schedule_order(values, field)

    types = raw.get("indexer_types")
    if types is not None:
        if not isinstance(types, list):
            raise ConfigError("config indexer_types must be a list when present")
        return validate_length(tuple(normalize(t) for t in types), "indexer_types")

    pattern = raw.get("index_topk_pattern")
    if isinstance(pattern, str):
        return validate_length(
            tuple(normalize(ch) for ch in pattern),
            "index_topk_pattern",
        )
    if isinstance(pattern, list):
        return validate_length(
            tuple(normalize(t) for t in pattern),
            "index_topk_pattern",
        )

    # GLM MoE DSA checkpoints can define a schedule by frequency/offset. This
    # mirrors the oMLX patch convention and is only used for memory estimation.
    if "index_topk_freq" in raw or "index_skip_topk_offset" in raw:
        freq = _as_int(
            raw.get("index_topk_freq"),
            default=1,
            field="index_topk_freq",
        )
        offset = _as_int(
            raw.get("index_skip_topk_offset"),
            default=2,
            field="index_skip_topk_offset",
        )
        assert freq is not None and offset is not None
        if freq <= 0:
            raise ConfigError("config index_topk_freq must be positive")
        if offset < 0:
            raise ConfigError("config index_skip_topk_offset must be non-negative")
        return tuple(
            "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
            for i in range(num_layers)
        )

    return None


def first_shared_indexer_without_previous_full(
    indexer_types: Iterable[str],
    selected_layers: Iterable[int] | None = None,
) -> int | None:
    types = tuple(str(item).lower() for item in indexer_types)
    layer_ids = range(len(types)) if selected_layers is None else sorted(set(selected_layers))
    saw_full = False
    for layer in layer_ids:
        if layer < 0 or layer >= len(types):
            continue
        mode = types[layer]
        if mode == "full":
            saw_full = True
        elif mode == "shared" and not saw_full:
            return layer
    return None


def _mlp_layer_types(raw: dict[str, Any], num_layers: int) -> tuple[str, ...] | None:
    values = raw.get("mlp_layer_types")
    if values is None:
        return None
    if not isinstance(values, list):
        raise ConfigError("config mlp_layer_types must be a list when present")
    if len(values) != num_layers:
        raise ConfigError("config mlp_layer_types length does not match num_hidden_layers")
    aliases = {
        "dense": "dense",
        "mlp": "dense",
        "dense_mlp": "dense",
        "sparse": "sparse",
        "moe": "sparse",
        "moe_sparse": "sparse",
    }
    normalized: list[str] = []
    for value in values:
        item = str(value).lower()
        try:
            normalized.append(aliases[item])
        except KeyError as exc:
            raise ConfigError(f"invalid config mlp_layer_type {value!r}") from exc
    return tuple(normalized)


def _require_positive(config: ModelConfig, field: str) -> None:
    value = getattr(config, field)
    if value is not None and int(value) <= 0:
        raise ConfigError(f"config {field} must be positive")


def _require_nonnegative(config: ModelConfig, field: str) -> None:
    value = getattr(config, field)
    if value is not None and int(value) < 0:
        raise ConfigError(f"config {field} must be non-negative")


def _validate_config_scalars(config: ModelConfig) -> None:
    for field in (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_layer_freq",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "max_position_embeddings",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "n_group",
        "topk_group",
    ):
        _require_positive(config, field)
    for field in ("n_shared_experts", "first_k_dense_replace"):
        _require_nonnegative(config, field)
    if config.rms_norm_eps is not None:
        rms_norm_eps = float(config.rms_norm_eps)
        if not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ConfigError("config rms_norm_eps must be positive")
    if config.rope_theta is not None:
        rope_theta = float(config.rope_theta)
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ConfigError("config rope_theta must be positive")
    if config.routed_scaling_factor is not None:
        routed_scaling_factor = float(config.routed_scaling_factor)
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ConfigError("config routed_scaling_factor must be positive")
    if (
        config.scoring_func is not None
        and config.scoring_func not in {"sigmoid", "softmax", "raw"}
    ):
        raise ConfigError("config scoring_func must be sigmoid, softmax, or raw")
    if (
        config.n_group is not None
        and config.topk_group is not None
        and config.topk_group > config.n_group
    ):
        raise ConfigError("config topk_group must not exceed n_group")
    if config.first_k_dense_replace > config.num_hidden_layers:
        raise ConfigError(
            "config first_k_dense_replace must not exceed num_hidden_layers"
        )
    if (
        config.n_routed_experts is not None
        and config.num_experts_per_tok is not None
        and config.num_experts_per_tok > config.n_routed_experts
    ):
        raise ConfigError("config num_experts_per_tok must not exceed n_routed_experts")
    if (
        config.index_topk is not None
        and config.max_position_embeddings is not None
        and config.index_topk > config.max_position_embeddings
    ):
        raise ConfigError("config index_topk must not exceed max_position_embeddings")
    if config.indexer_types is not None and any(
        item in {"full", "shared"} for item in config.indexer_types
    ):
        missing = [
            field
            for field in (
                "index_head_dim",
                "index_n_heads",
                "index_topk",
                "q_lora_rank",
            )
            if getattr(config, field) is None
        ]
        if missing:
            raise ConfigError(
                "config DSA indexer schedule requires "
                + ", ".join(missing)
            )


def load_config(path: str | Path) -> ModelConfig:
    """Load a model config from a config file or checkpoint directory."""

    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    if not p.exists():
        raise ConfigError(f"config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ConfigError("config JSON must be an object")

    hidden = _as_int(_get_first(raw, ("hidden_size", "n_embd", "dim")), field="hidden_size")
    layers = _as_int(
        _get_first(raw, ("num_hidden_layers", "n_layer", "num_layers")),
        field="num_hidden_layers",
    )
    if hidden is None or layers is None:
        raise ConfigError("config must include hidden_size and num_hidden_layers")

    weight_dtype, weight_dtype_bytes = _weight_dtype(raw)

    config = ModelConfig(
        model_type=_model_type(raw),
        hidden_size=hidden,
        num_hidden_layers=layers,
        vocab_size=_as_int(
            _get_first(raw, ("vocab_size", "padded_vocab_size", "n_vocab")),
            field="vocab_size",
        ),
        eos_token_ids=_eos_token_ids(raw),
        weight_dtype=weight_dtype,
        weight_dtype_bytes=weight_dtype_bytes,
        intermediate_size=_as_int(raw.get("intermediate_size"), field="intermediate_size"),
        moe_intermediate_size=_as_int(
            _get_first(raw, ("moe_intermediate_size", "moe_ffn_hidden_size")),
            field="moe_intermediate_size",
        ),
        n_routed_experts=_as_int(
            _get_first(raw, ("n_routed_experts", "num_experts", "num_local_experts")),
            field="n_routed_experts",
        ),
        n_shared_experts=_as_int(
            raw.get("n_shared_experts"),
            default=0,
            field="n_shared_experts",
        ),
        num_experts_per_tok=_as_int(
            _get_first(raw, ("num_experts_per_tok", "moe_top_k", "top_k")),
            field="num_experts_per_tok",
        ),
        moe_layer_freq=_as_int(
            raw.get("moe_layer_freq"),
            default=1,
            field="moe_layer_freq",
        )
        or 0,
        first_k_dense_replace=_as_int(
            raw.get("first_k_dense_replace"),
            default=0,
            field="first_k_dense_replace",
        )
        or 0,
        num_attention_heads=_as_int(raw.get("num_attention_heads"), field="num_attention_heads"),
        num_key_value_heads=_as_int(
            raw.get("num_key_value_heads"),
            field="num_key_value_heads",
        ),
        q_lora_rank=_as_int(raw.get("q_lora_rank"), field="q_lora_rank"),
        kv_lora_rank=_as_int(raw.get("kv_lora_rank"), field="kv_lora_rank"),
        qk_nope_head_dim=_as_int(raw.get("qk_nope_head_dim"), field="qk_nope_head_dim"),
        qk_rope_head_dim=_as_int(raw.get("qk_rope_head_dim"), field="qk_rope_head_dim"),
        v_head_dim=_as_int(raw.get("v_head_dim"), field="v_head_dim"),
        max_position_embeddings=_as_int(
            raw.get("max_position_embeddings"),
            field="max_position_embeddings",
        ),
        rms_norm_eps=_as_float(raw.get("rms_norm_eps"), field="rms_norm_eps"),
        rope_theta=_rope_theta(raw),
        index_head_dim=_as_int(raw.get("index_head_dim"), field="index_head_dim"),
        index_n_heads=_as_int(raw.get("index_n_heads"), field="index_n_heads"),
        index_topk=_as_int(raw.get("index_topk"), field="index_topk"),
        indexer_rope_interleave=_as_bool(
            raw.get("indexer_rope_interleave"),
            default=False,
            field="indexer_rope_interleave",
        ),
        rope_interleave=_as_bool(
            raw.get("rope_interleave"),
            default=False,
            field="rope_interleave",
        ),
        scoring_func=_as_optional_str(raw.get("scoring_func"), field="scoring_func"),
        topk_method=_as_optional_str(raw.get("topk_method"), field="topk_method"),
        norm_topk_prob=_as_optional_bool(
            raw.get("norm_topk_prob"),
            field="norm_topk_prob",
        ),
        routed_scaling_factor=_as_float(
            raw.get("routed_scaling_factor"),
            field="routed_scaling_factor",
        ),
        n_group=_as_int(raw.get("n_group"), field="n_group"),
        topk_group=_as_int(raw.get("topk_group"), field="topk_group"),
        indexer_types=_indexer_types(raw, layers),
        mlp_layer_types=_mlp_layer_types(raw, layers),
        tie_word_embeddings=_as_optional_bool(
            raw.get("tie_word_embeddings"),
            field="tie_word_embeddings",
        ),
        raw=raw,
    )
    _validate_config_scalars(config)
    return config
