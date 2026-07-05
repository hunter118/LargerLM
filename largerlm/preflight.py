from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, ModelConfig, load_config
from .hardware import detect_hardware
from .packer import PackerError, build_packed_layout
from .planner import ModelPlan, PlannerError, build_plan, default_system_reserve_bytes
from .resident import (
    ResidentPackerError,
    build_resident_layout,
    normalize_resident_tensor_aliases,
    resident_affine_int4_logical_shape,
    resident_mxfp4_logical_shape,
)
from .safety import DiskBudget, disk_budget
from .safetensors import (
    SafetensorsError,
    TensorMeta,
    categorize_tensor_for_moe_layers,
    iter_tensor_metadata,
    scan_checkpoint,
    tensor_layer_id,
)
from .tokenizer import TokenizerError, load_tokenizer


class PreflightError(RuntimeError):
    """Raised when a GLM checkpoint preflight cannot be completed."""


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class TensorCoverage:
    embedding: bool
    final_norm: bool
    lm_head: bool
    tied_lm_head: bool
    attention_layers_checked: int
    attention_layers_ok: int
    router_layers_checked: int
    router_layers_ok: int
    router_bias_layers_checked: int
    router_bias_layers_found: int
    dense_layers_checked: int
    dense_layers_ok: int
    shared_layers_checked: int
    shared_layers_ok: int
    indexer_layers_checked: int
    indexer_layers_ok: int
    missing_global: tuple[str, ...]
    missing_by_layer: tuple[str, ...]


@dataclass(frozen=True)
class ExpertCoverage:
    checked: bool
    quantization_mode: str
    packed_layers: int
    packed_bytes_estimate: int
    max_expert_slot_bytes: int


@dataclass(frozen=True)
class ResidentCoverage:
    checked: bool
    tensor_count: int
    packed_bytes_estimate: int
    max_tensor_bytes: int


@dataclass(frozen=True)
class TokenizerPreflight:
    found: bool
    loaded: bool
    backend: str | None
    path: Path | None
    eos_token_id: int | None
    error: str | None


@dataclass(frozen=True)
class QuantizationPreflight:
    detected: bool
    source: str | None
    method: str | None
    bits: int | None
    group_size: int | None
    target_bits: int
    target_group_size: int
    bits_match_target: bool | None
    group_size_match_target: bool | None
    compatible_with_mlx_affine_int4: bool | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlmPreflightReport:
    ok: bool
    model_dir: Path
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    moe_layers: int
    routed_experts: int | None
    experts_per_token: int | None
    dsa_index_topk: int | None
    dsa_index_n_heads: int | None
    dsa_index_head_dim: int | None
    dsa_full_indexer_q_output_dim: int | None
    checkpoint_tensor_count: int
    checkpoint_total_bytes: int
    checkpoint_routed_bytes: int
    checkpoint_resident_bytes: int
    checkpoint_ignored_bytes: int
    hardware_chip_name: str
    hardware_unified_memory_bytes: int | None
    hardware_gpu_cores: int | None
    hardware_apple_silicon_generation: int | None
    hardware_apple_silicon_tier: str | None
    effective_unified_memory_bytes: int | None
    effective_unified_memory_source: str
    effective_system_reserve_bytes: int
    tensor_coverage: TensorCoverage
    expert_coverage: ExpertCoverage
    resident_coverage: ResidentCoverage
    tokenizer: TokenizerPreflight
    quantization: QuantizationPreflight
    plan: ModelPlan | None
    disk_budget: DiskBudget | None
    recommended_max_live_working_set_bytes: int
    recommended_min_free_unified_memory_bytes: int
    cold_read_gib_per_second: float | None
    public_glm_5_2_shape: dict[str, object]
    issues: tuple[PreflightIssue, ...]


GLOBAL_EMBED_SUFFIXES = (
    "model.embed_tokens.weight",
    ".embed_tokens.weight",
    "transformer.word_embeddings.weight",
    ".word_embeddings.weight",
)
GLOBAL_NORM_SUFFIXES = (
    "model.norm.weight",
    ".model.norm.weight",
    "transformer.norm.weight",
    ".transformer.norm.weight",
    "norm.weight",
    ".norm.weight",
)
GLOBAL_LM_HEAD_SUFFIXES = ("lm_head.weight", ".lm_head.weight")
ATTENTION_SUFFIXES = (
    ".input_layernorm.weight",
    ".self_attn.q_a_proj.weight",
    ".self_attn.q_a_layernorm.weight",
    ".self_attn.q_b_proj.weight",
    ".self_attn.kv_a_proj_with_mqa.weight",
    ".self_attn.kv_a_layernorm.weight",
    ".self_attn.kv_b_proj.weight",
    ".self_attn.o_proj.weight",
    ".post_attention_layernorm.weight",
)
ATTENTION_KV_B_ALTERNATIVE_SUFFIXES = (
    ".self_attn.embed_q.weight",
    ".self_attn.unembed_out.weight",
)
SHARED_COMPONENTS = ("gate_proj", "up_proj", "down_proj")
DENSE_MLP_COMPONENTS = ("gate_proj", "up_proj", "down_proj")
INDEXER_SUFFIXES = (
    ".self_attn.indexer.wk.weight",
    ".self_attn.indexer.wq_b.weight",
    ".self_attn.indexer.weights_proj.weight",
    ".self_attn.indexer.k_norm.weight",
    ".self_attn.indexer.k_norm.bias",
)
ROUTER_CORRECTION_BIAS_SUFFIX = ".gate.e_score_correction_bias"
_UNSUPPORTED_EXPERT_QUANT_PARTS = {
    "qweight",
    "qzeros",
    "g_idx",
    "absmax",
    "nested_absmax",
    "quant_map",
    "nested_quant_map",
}
_UNSUPPORTED_EXPERT_QUANT_SUBSTRINGS = (
    ".quant_state.",
    "bitsandbytes",
    "bnb_4bit",
)
_ROUTED_EXPERT_NAME_MARKERS = (
    ".experts.",
    ".switch_mlp.",
    ".block_sparse_moe.",
)


def _issue(severity: str, code: str, message: str) -> PreflightIssue:
    return PreflightIssue(severity=severity, code=code, message=message)


def _public_glm_5_2_shape_report(config: ModelConfig) -> dict[str, object]:
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


def _unchecked_tensor_coverage() -> TensorCoverage:
    return TensorCoverage(
        embedding=False,
        final_norm=False,
        lm_head=False,
        tied_lm_head=False,
        attention_layers_checked=0,
        attention_layers_ok=0,
        router_layers_checked=0,
        router_layers_ok=0,
        router_bias_layers_checked=0,
        router_bias_layers_found=0,
        dense_layers_checked=0,
        dense_layers_ok=0,
        shared_layers_checked=0,
        shared_layers_ok=0,
        indexer_layers_checked=0,
        indexer_layers_ok=0,
        missing_global=(),
        missing_by_layer=(),
    )


def _unchecked_tokenizer(error: str | None = None) -> TokenizerPreflight:
    return TokenizerPreflight(
        found=False,
        loaded=False,
        backend=None,
        path=None,
        eos_token_id=None,
        error=error,
    )


_QUANTIZATION_CONTAINER_KEYS = (
    "quantization",
    "quantization_config",
    "quantizationConfig",
    "quantization_config_v2",
)
_QUANTIZATION_BITS_KEYS = (
    "bits",
    "num_bits",
    "nbits",
    "w_bits",
    "weight_bits",
    "precision",
)
_QUANTIZATION_GROUP_KEYS = (
    "group_size",
    "q_group_size",
    "groupsize",
    "weight_group_size",
)
_QUANTIZATION_METHOD_KEYS = (
    "quant_method",
    "quantization_method",
    "quant_type",
    "type",
    "scheme",
    "format",
)


def _quantization_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(operator.index(value))
    except TypeError:
        pass
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit():
            return int(text)
        for marker, bits in (
            ("int4", 4),
            ("4bit", 4),
            ("4-bit", 4),
            ("int8", 8),
            ("8bit", 8),
            ("8-bit", 8),
        ):
            if marker in text:
                return bits
    return None


def _quantization_text(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _quantization_value(raw: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _quantization_method_compatible_with_mlx_affine_int4(
    method: str | None,
) -> bool | None:
    if method is None:
        return None
    normalized = method.strip().lower().replace("_", "-")
    incompatible = (
        "awq",
        "gptq",
        "nf4",
        "bitsandbytes",
        "bnb",
        "gguf",
        "fp8",
    )
    if any(marker in normalized for marker in incompatible):
        return False
    compatible = ("mlx", "affine", "int4", "4bit", "4-bit")
    if any(marker in normalized for marker in compatible):
        return True
    return None


def _quantization_preflight(
    raw_config: dict[str, object] | None,
    *,
    target_bits: int,
    target_group_size: int,
) -> QuantizationPreflight:
    raw = raw_config if isinstance(raw_config, dict) else {}
    source: str | None = None
    metadata: dict[str, object] = {}
    method: str | None = None

    for key in _QUANTIZATION_CONTAINER_KEYS:
        value = raw.get(key)
        if isinstance(value, dict):
            metadata = dict(value)
            source = key
            break
        text = _quantization_text(value)
        if text is not None:
            method = text
            source = key
            break

    bits = _quantization_int(_quantization_value(metadata, _QUANTIZATION_BITS_KEYS))
    group_size = _quantization_int(
        _quantization_value(metadata, _QUANTIZATION_GROUP_KEYS)
    )
    if method is None:
        method = _quantization_text(
            _quantization_value(metadata, _QUANTIZATION_METHOD_KEYS)
        )

    if bits is None:
        if metadata.get("load_in_4bit") is True:
            bits = 4
        elif metadata.get("load_in_8bit") is True:
            bits = 8
        elif method is not None:
            bits = _quantization_int(method)
    if source is None:
        bits = _quantization_int(_quantization_value(raw, _QUANTIZATION_BITS_KEYS))
        group_size = _quantization_int(
            _quantization_value(raw, _QUANTIZATION_GROUP_KEYS)
        )
        method = _quantization_text(
            _quantization_value(raw, _QUANTIZATION_METHOD_KEYS)
        )
        if bits is not None or group_size is not None or method is not None:
            source = "top_level"

    detected = source is not None
    bits_match = None if bits is None else bits == target_bits
    group_match = None if group_size is None else group_size == target_group_size
    method_compatible = _quantization_method_compatible_with_mlx_affine_int4(method)
    compatible = method_compatible
    if bits is not None and group_size is not None:
        compatible = bits == 4 and group_size == target_group_size
        if method_compatible is False:
            compatible = False

    warnings: list[str] = []
    if detected and bits is None:
        warnings.append("config quantization metadata does not declare bit width")
    if detected and group_size is None:
        warnings.append("config quantization metadata does not declare group size")
    if detected and method_compatible is False:
        warnings.append("config quantization method is not MLX affine int4")

    return QuantizationPreflight(
        detected=detected,
        source=source,
        method=method,
        bits=bits,
        group_size=group_size,
        target_bits=target_bits,
        target_group_size=target_group_size,
        bits_match_target=bits_match,
        group_size_match_target=group_match,
        compatible_with_mlx_affine_int4=compatible,
        warnings=tuple(warnings),
    )


def _unsupported_expert_quantized_tensor_examples(
    config: ModelConfig,
    tensors: list[TensorMeta],
    *,
    limit: int = 8,
) -> tuple[str, ...]:
    moe_layers = set(config.moe_layers)
    examples: list[str] = []
    for tensor in tensors:
        layer = tensor_layer_id(tensor.name)
        if layer is None or layer not in moe_layers:
            continue
        lowered = tensor.name.lower()
        if not any(marker in lowered for marker in _ROUTED_EXPERT_NAME_MARKERS):
            continue
        parts = set(lowered.split("."))
        if (
            parts & _UNSUPPORTED_EXPERT_QUANT_PARTS
            or any(
                marker in lowered
                for marker in _UNSUPPORTED_EXPERT_QUANT_SUBSTRINGS
            )
        ):
            examples.append(tensor.name)
            if len(examples) >= limit:
                break
    return tuple(examples)


def _unsupported_expert_quantized_tensor_message(
    examples: tuple[str, ...],
) -> str:
    preview = ", ".join(repr(name) for name in examples)
    return (
        "checkpoint appears to use GPTQ/AWQ/bitsandbytes-style routed expert "
        "tensor names instead of MLX affine-int4 weight/scales/biases; "
        f"examples: {preview}"
    )


def _integer_param(
    name: str,
    value: object,
    issues: list[PreflightIssue],
    *,
    minimum: int | None,
    default: int,
) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        parsed = int(operator.index(value))
    except TypeError:
        issues.append(
            _issue("error", f"invalid_{name}", f"{name} must be an integer")
        )
        return default
    if minimum is not None and parsed < minimum:
        relation = "positive" if minimum == 1 else f">= {minimum}"
        issues.append(
            _issue("error", f"invalid_{name}", f"{name} must be {relation}")
        )
        return default
    return parsed


def _optional_integer_param(
    name: str,
    value: object | None,
    issues: list[PreflightIssue],
    *,
    minimum: int | None,
) -> int | None:
    if value is None:
        return None
    return _integer_param(name, value, issues, minimum=minimum, default=minimum or 0)


def _float_param(
    name: str,
    value: object,
    issues: list[PreflightIssue],
    *,
    minimum: float | None,
    maximum: float | None = None,
    default: float,
) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError
        parsed = float(value)
    except (TypeError, ValueError):
        issues.append(_issue("error", f"invalid_{name}", f"{name} must be numeric"))
        return default
    if not math.isfinite(parsed):
        issues.append(_issue("error", f"invalid_{name}", f"{name} must be finite"))
        return default
    if minimum is not None and parsed < minimum:
        relation = "positive" if minimum > 0 else f">= {minimum:g}"
        issues.append(_issue("error", f"invalid_{name}", f"{name} must be {relation}"))
        return default
    if maximum is not None and parsed > maximum:
        issues.append(
            _issue("error", f"invalid_{name}", f"{name} must be <= {maximum:g}")
        )
        return default
    return parsed


def _optional_float_param(
    name: str,
    value: object | None,
    issues: list[PreflightIssue],
    *,
    minimum: float | None,
) -> float | None:
    if value is None:
        return None
    return _float_param(name, value, issues, minimum=minimum, default=minimum or 0.0)


def _config_int(config: ModelConfig, name: str, issues: list[PreflightIssue]) -> int | None:
    try:
        return int(getattr(config, name))
    except (ConfigError, TypeError, ValueError) as exc:
        issues.append(_issue("error", f"missing_{name}", str(exc)))
        return None


def _is_global(name: str) -> bool:
    return ".layers." not in name


def _contains_layer(name: str, layer: int) -> bool:
    return f".layers.{layer}." in name


def _has_suffix(tensors: list[TensorMeta], suffixes: tuple[str, ...]) -> bool:
    return _find_suffix(tensors, suffixes) is not None


def _find_suffix(
    tensors: list[TensorMeta],
    suffixes: tuple[str, ...],
) -> TensorMeta | None:
    for tensor in tensors:
        if _is_global(tensor.name) and any(
            tensor.name.endswith(suffix) for suffix in suffixes
        ):
            return tensor
    return None


def _has_layer_suffix(tensors: list[TensorMeta], layer: int, suffix: str) -> bool:
    return any(
        _contains_layer(tensor.name, layer) and tensor.name.endswith(suffix)
        for tensor in tensors
    )


def _find_layer_suffix(
    tensors: list[TensorMeta],
    layer: int,
    suffix: str,
) -> TensorMeta | None:
    for tensor in tensors:
        if _contains_layer(tensor.name, layer) and tensor.name.endswith(suffix):
            return tensor
    return None


def _has_router(tensors: list[TensorMeta], layer: int) -> bool:
    return _find_router(tensors, layer) is not None


def _find_router(tensors: list[TensorMeta], layer: int) -> TensorMeta | None:
    for tensor in tensors:
        name = tensor.name
        if not _contains_layer(name, layer):
            continue
        if name.endswith(".gate.weight") or ".mlp.gate.weight" in name:
            return tensor
    return None


def _has_router_correction_bias(tensors: list[TensorMeta], layer: int) -> bool:
    return _find_router_correction_bias(tensors, layer) is not None


def _find_router_correction_bias(
    tensors: list[TensorMeta],
    layer: int,
) -> TensorMeta | None:
    return _find_layer_suffix(tensors, layer, ROUTER_CORRECTION_BIAS_SUFFIX)


def _has_shared_component(tensors: list[TensorMeta], layer: int, component: str) -> bool:
    return _find_shared_component(tensors, layer, component) is not None


def _find_shared_component(
    tensors: list[TensorMeta],
    layer: int,
    component: str,
) -> TensorMeta | None:
    suffix = f".{component}.weight"
    for tensor in tensors:
        name = tensor.name
        if not _contains_layer(name, layer) or not name.endswith(suffix):
            continue
        if ".shared_experts." in name or ".shared_expert." in name:
            return tensor
    return None


def _has_dense_mlp_component(
    tensors: list[TensorMeta],
    layer: int,
    component: str,
) -> bool:
    return _find_dense_mlp_component(tensors, layer, component) is not None


def _find_dense_mlp_component(
    tensors: list[TensorMeta],
    layer: int,
    component: str,
) -> TensorMeta | None:
    suffixes = (
        f".mlp.{component}.weight",
        f".mlp.switch_mlp.{component}.weight",
        f".switch_mlp.{component}.weight",
    )
    for tensor in tensors:
        name = tensor.name
        if not _contains_layer(name, layer):
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            return tensor
    return None


def _resident_logical_shape(
    tensor: TensorMeta,
    tensors_by_name: dict[str, TensorMeta],
    issues: list[PreflightIssue],
) -> tuple[int, ...]:
    try:
        affine_shape = resident_affine_int4_logical_shape(tensor, tensors_by_name)
    except ResidentPackerError as exc:
        issues.append(_issue("error", "resident_affine_int4_layout_invalid", str(exc)))
        return tensor.shape
    return affine_shape or tensor.shape


def _resident_mxfp4_runtime_unsupported_names(
    tensors: list[TensorMeta],
) -> tuple[str, ...]:
    tensors_by_name = {tensor.name: tensor for tensor in tensors}
    names: list[str] = []
    for tensor in tensors:
        if not tensor.name.endswith(".weight"):
            continue
        if ".mlp.experts." in tensor.name or ".mlp.switch_mlp." in tensor.name:
            continue
        try:
            logical_shape = resident_mxfp4_logical_shape(tensor, tensors_by_name)
        except ResidentPackerError:
            continue
        if logical_shape is None:
            continue
        if any(
            tensor.name.endswith(suffix)
            for suffix in ATTENTION_KV_B_ALTERNATIVE_SUFFIXES
        ):
            continue
        if len(logical_shape) != 2:
            names.append(tensor.name)
            continue
        if any(tensor.name.endswith(suffix) for suffix in GLOBAL_EMBED_SUFFIXES):
            continue
        if any(tensor.name.endswith(suffix) for suffix in GLOBAL_LM_HEAD_SUFFIXES):
            continue
    return tuple(names)


def _tensor_coverage(
    config: ModelConfig,
    tensors: list[TensorMeta],
    issues: list[PreflightIssue],
) -> TensorCoverage:
    tensors_by_name = {tensor.name: tensor for tensor in tensors}
    embedding_tensor = _find_suffix(tensors, GLOBAL_EMBED_SUFFIXES)
    final_norm_tensor = _find_suffix(tensors, GLOBAL_NORM_SUFFIXES)
    lm_head_tensor = _find_suffix(tensors, GLOBAL_LM_HEAD_SUFFIXES)
    embedding = embedding_tensor is not None
    final_norm = final_norm_tensor is not None
    lm_head = lm_head_tensor is not None
    tied_allowed = config.tie_word_embeddings is not False
    tied = embedding and not lm_head and tied_allowed
    missing_global: list[str] = []
    if not embedding:
        missing_global.append("embed_tokens.weight")
        issues.append(_issue("error", "missing_embedding", "global embedding tensor not found"))
    if not final_norm:
        missing_global.append("norm.weight")
        issues.append(_issue("error", "missing_final_norm", "global final norm tensor not found"))
    if not lm_head and (not embedding or not tied_allowed):
        missing_global.append("lm_head.weight")
        detail = (
            "tie_word_embeddings=false"
            if embedding and not tied_allowed
            else "tied embeddings are unavailable"
        )
        issues.append(
            _issue(
                "error",
                "missing_lm_head",
                f"lm_head.weight not found and {detail}",
            )
        )
    if not lm_head and embedding and tied_allowed:
        issues.append(
            _issue(
                "warning",
                "tied_lm_head",
                "lm_head.weight not found; final logits will use tied embeddings",
            )
        )

    global_shape_errors: list[str] = []
    hidden = int(config.hidden_size)
    vocab = config.vocab_size
    if embedding_tensor is not None:
        embedding_shape = _resident_logical_shape(
            embedding_tensor,
            tensors_by_name,
            issues,
        )
        if len(embedding_shape) != 2:
            global_shape_errors.append(
                f"embed_tokens.weight shape {list(embedding_shape)} must be 2-D"
            )
        elif vocab is not None and embedding_shape[0] != vocab:
            global_shape_errors.append(
                "embed_tokens.weight vocab rows "
                f"{embedding_shape[0]} != {vocab}"
            )
        elif embedding_shape[1] != hidden:
            global_shape_errors.append(
                "embed_tokens.weight hidden dim "
                f"{embedding_shape[1]} != {hidden}"
            )
    if final_norm_tensor is not None and final_norm_tensor.shape != (hidden,):
        global_shape_errors.append(
            f"norm.weight shape {list(final_norm_tensor.shape)} != [{hidden}]"
        )
    if lm_head_tensor is not None:
        lm_head_shape = _resident_logical_shape(
            lm_head_tensor,
            tensors_by_name,
            issues,
        )
        if len(lm_head_shape) != 2:
            global_shape_errors.append(
                f"lm_head.weight shape {list(lm_head_shape)} must be 2-D"
            )
        elif vocab is not None and lm_head_shape[0] != vocab:
            global_shape_errors.append(
                f"lm_head.weight vocab rows {lm_head_shape[0]} != {vocab}"
            )
        elif lm_head_shape[1] != hidden:
            global_shape_errors.append(
                f"lm_head.weight hidden dim {lm_head_shape[1]} != {hidden}"
            )
    if global_shape_errors:
        preview = "; ".join(global_shape_errors[:8])
        more = (
            ""
            if len(global_shape_errors) <= 8
            else f"; +{len(global_shape_errors) - 8} more"
        )
        issues.append(
            _issue(
                "error",
                "global_shape_mismatch",
                f"global tensor shapes do not match config: {preview}{more}",
            )
        )

    missing_by_layer: list[str] = []
    attention_shape_errors: list[str] = []
    mlp_shape_errors: list[str] = []
    router_shape_errors: list[str] = []
    attention_ok = 0
    router_ok = 0
    router_bias_found = 0
    dense_ok = 0
    shared_ok = 0
    indexer_ok = 0
    moe_layers = config.moe_layers
    moe_layer_set = set(moe_layers)
    dense_layers = [
        layer for layer in range(config.num_hidden_layers) if layer not in moe_layer_set
    ]
    indexer_layers = [
        layer
        for layer, indexer_type in enumerate(config.indexer_types or ())
        if layer < config.num_hidden_layers
        and config.index_head_dim
        and str(indexer_type).lower() == "full"
    ]
    for layer in range(config.num_hidden_layers):
        layer_missing: list[str] = []
        for suffix in ATTENTION_SUFFIXES:
            if _has_layer_suffix(tensors, layer, suffix):
                continue
            if suffix == ".self_attn.kv_b_proj.weight" and all(
                _has_layer_suffix(tensors, layer, alternative)
                for alternative in ATTENTION_KV_B_ALTERNATIVE_SUFFIXES
            ):
                continue
            layer_missing.append(suffix)
        if layer_missing:
            missing_by_layer.extend(f"layer {layer}: {suffix}" for suffix in layer_missing)
        else:
            attention_ok += 1

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
        q_out = int(config.num_attention_heads) * (
            int(config.qk_nope_head_dim) + int(config.qk_rope_head_dim)
        )
        kv_a_out = kv_lora + int(config.qk_rope_head_dim)
        kv_b_out = int(config.num_attention_heads) * (
            int(config.qk_nope_head_dim) + int(config.v_head_dim)
        )
        value_out = int(config.num_attention_heads) * int(config.v_head_dim)
        expected_attention_shapes = {
            ".input_layernorm.weight": (int(config.hidden_size),),
            ".self_attn.q_a_proj.weight": (q_lora, int(config.hidden_size)),
            ".self_attn.q_a_layernorm.weight": (q_lora,),
            ".self_attn.q_b_proj.weight": (q_out, q_lora),
            ".self_attn.kv_a_proj_with_mqa.weight": (
                kv_a_out,
                int(config.hidden_size),
            ),
            ".self_attn.kv_a_layernorm.weight": (kv_lora,),
            ".self_attn.kv_b_proj.weight": (kv_b_out, kv_lora),
            ".self_attn.o_proj.weight": (int(config.hidden_size), value_out),
            ".post_attention_layernorm.weight": (int(config.hidden_size),),
        }
        for layer in range(config.num_hidden_layers):
            for suffix, expected_shape in expected_attention_shapes.items():
                if (
                    suffix == ".self_attn.kv_b_proj.weight"
                    and not _has_layer_suffix(tensors, layer, suffix)
                    and all(
                        _has_layer_suffix(tensors, layer, alternative)
                        for alternative in ATTENTION_KV_B_ALTERNATIVE_SUFFIXES
                    )
                ):
                    continue
                tensor = _find_layer_suffix(tensors, layer, suffix)
                logical_shape = (
                    _resident_logical_shape(tensor, tensors_by_name, issues)
                    if tensor is not None
                    else None
                )
                if logical_shape is not None and logical_shape != expected_shape:
                    attention_shape_errors.append(
                        f"layer {layer}: {suffix} shape "
                        f"{list(logical_shape)} != {list(expected_shape)}"
                    )
            if (
                not _has_layer_suffix(tensors, layer, ".self_attn.kv_b_proj.weight")
                and all(
                    _has_layer_suffix(tensors, layer, alternative)
                    for alternative in ATTENTION_KV_B_ALTERNATIVE_SUFFIXES
                )
            ):
                alternative_shapes = {
                    ".self_attn.embed_q.weight": (
                        int(config.num_attention_heads),
                        kv_lora,
                        int(config.qk_nope_head_dim),
                    ),
                    ".self_attn.unembed_out.weight": (
                        int(config.num_attention_heads),
                        int(config.v_head_dim),
                        kv_lora,
                    ),
                }
                for suffix, expected_shape in alternative_shapes.items():
                    tensor = _find_layer_suffix(tensors, layer, suffix)
                    logical_shape = (
                        _resident_logical_shape(tensor, tensors_by_name, issues)
                        if tensor is not None
                        else None
                    )
                    if logical_shape is not None and logical_shape != expected_shape:
                        attention_shape_errors.append(
                            f"layer {layer}: {suffix} shape "
                            f"{list(logical_shape)} != {list(expected_shape)}"
                        )

    for layer in dense_layers:
        missing = [
            component
            for component in DENSE_MLP_COMPONENTS
            if not _has_dense_mlp_component(tensors, layer, component)
        ]
        if missing:
            missing_by_layer.append(f"layer {layer}: dense MLP {','.join(missing)}")
        else:
            dense_ok += 1
            intermediate = config.intermediate_size
            if intermediate is not None:
                expected_dense_shapes = {
                    "gate_proj": (int(intermediate), int(config.hidden_size)),
                    "up_proj": (int(intermediate), int(config.hidden_size)),
                    "down_proj": (int(config.hidden_size), int(intermediate)),
                }
                for component, expected_shape in expected_dense_shapes.items():
                    tensor = _find_dense_mlp_component(tensors, layer, component)
                    logical_shape = (
                        _resident_logical_shape(tensor, tensors_by_name, issues)
                        if tensor is not None
                        else None
                    )
                    if logical_shape is not None and logical_shape != expected_shape:
                        mlp_shape_errors.append(
                            f"layer {layer}: dense MLP {component} shape "
                            f"{list(logical_shape)} != {list(expected_shape)}"
                        )

    for layer in moe_layers:
        router_tensor = _find_router(tensors, layer)
        if router_tensor is not None:
            router_ok += 1
            if config.n_routed_experts is not None:
                expected_router_shape = (
                    int(config.n_routed_experts),
                    int(config.hidden_size),
                )
                router_shape = _resident_logical_shape(
                    router_tensor,
                    tensors_by_name,
                    issues,
                )
                if router_shape != expected_router_shape:
                    router_shape_errors.append(
                        f"layer {layer}: router gate.weight shape "
                        f"{list(router_shape)} != {list(expected_router_shape)}"
                    )
        else:
            missing_by_layer.append(f"layer {layer}: router gate.weight")
        correction_bias = _find_router_correction_bias(tensors, layer)
        if correction_bias is not None:
            router_bias_found += 1
            if config.n_routed_experts is not None:
                expected_bias_shape = (int(config.n_routed_experts),)
                if correction_bias.shape != expected_bias_shape:
                    router_shape_errors.append(
                        f"layer {layer}: router correction bias shape "
                        f"{list(correction_bias.shape)} != {list(expected_bias_shape)}"
                    )

    topk_method = (config.topk_method or "").lower()
    if (
        moe_layers
        and topk_method == "noaux_tc"
        and router_bias_found != len(moe_layers)
    ):
        issues.append(
            _issue(
                "warning",
                "router_correction_bias_incomplete",
                "topk_method=noaux_tc usually expects "
                "gate.e_score_correction_bias on every MoE layer; "
                f"found {router_bias_found}/{len(moe_layers)}",
            )
        )

    shared_layers_checked = 0
    if (config.n_shared_experts or 0) > 0:
        shared_layers_checked = len(moe_layers)
        for layer in moe_layers:
            missing = [
                component
                for component in SHARED_COMPONENTS
                if not _has_shared_component(tensors, layer, component)
            ]
            if missing:
                missing_by_layer.append(
                    f"layer {layer}: shared expert {','.join(missing)}"
                )
            else:
                shared_ok += 1
                try:
                    shared_hidden = config.moe_hidden_size
                except ConfigError:
                    shared_hidden = None
                if shared_hidden is not None:
                    shared_width = int(config.n_shared_experts or 0) * shared_hidden
                    expected_shared_shapes = {
                        "gate_proj": (shared_width, int(config.hidden_size)),
                        "up_proj": (shared_width, int(config.hidden_size)),
                        "down_proj": (int(config.hidden_size), shared_width),
                    }
                    for component, expected_shape in expected_shared_shapes.items():
                        tensor = _find_shared_component(tensors, layer, component)
                        logical_shape = (
                            _resident_logical_shape(tensor, tensors_by_name, issues)
                            if tensor is not None
                            else None
                        )
                        if logical_shape is not None and logical_shape != expected_shape:
                            mlp_shape_errors.append(
                                f"layer {layer}: shared expert {component} shape "
                                f"{list(logical_shape)} != {list(expected_shape)}"
                            )

    dsa_shape_errors: list[str] = []
    dsa_missing_config: list[str] = []
    if indexer_layers:
        if config.q_lora_rank is None:
            dsa_missing_config.append("q_lora_rank")
        if config.index_n_heads is None:
            dsa_missing_config.append("index_n_heads")
    if dsa_missing_config:
        issues.append(
            _issue(
                "error",
                "missing_dsa_indexer_config",
                "full DSA indexer layers require config fields: "
                + ",".join(dsa_missing_config),
            )
        )

    for layer in indexer_layers:
        missing = [
            suffix
            for suffix in INDEXER_SUFFIXES
            if not _has_layer_suffix(tensors, layer, suffix)
        ]
        if missing:
            missing_by_layer.extend(f"layer {layer}: indexer {suffix}" for suffix in missing)
        else:
            indexer_ok += 1
            if not dsa_missing_config:
                assert config.index_head_dim is not None
                assert config.index_n_heads is not None
                assert config.q_lora_rank is not None
                expected_shapes = {
                    ".self_attn.indexer.wk.weight": (
                        int(config.index_head_dim),
                        int(config.hidden_size),
                    ),
                    ".self_attn.indexer.wq_b.weight": (
                        int(config.index_n_heads) * int(config.index_head_dim),
                        int(config.q_lora_rank),
                    ),
                    ".self_attn.indexer.weights_proj.weight": (
                        int(config.index_n_heads),
                        int(config.hidden_size),
                    ),
                    ".self_attn.indexer.k_norm.weight": (int(config.index_head_dim),),
                    ".self_attn.indexer.k_norm.bias": (int(config.index_head_dim),),
                }
                for suffix, expected_shape in expected_shapes.items():
                    tensor = _find_layer_suffix(tensors, layer, suffix)
                    logical_shape = (
                        _resident_logical_shape(tensor, tensors_by_name, issues)
                        if tensor is not None
                        else None
                    )
                    if logical_shape is not None and logical_shape != expected_shape:
                        dsa_shape_errors.append(
                            f"layer {layer}: indexer {suffix} shape "
                            f"{list(logical_shape)} != {list(expected_shape)}"
                        )

    if dsa_shape_errors:
        preview = "; ".join(dsa_shape_errors[:8])
        more = "" if len(dsa_shape_errors) <= 8 else f"; +{len(dsa_shape_errors) - 8} more"
        issues.append(
            _issue(
                "error",
                "dsa_indexer_shape_mismatch",
                f"DSA indexer tensor shapes do not match config: {preview}{more}",
            )
        )

    if attention_shape_errors:
        preview = "; ".join(attention_shape_errors[:8])
        more = (
            ""
            if len(attention_shape_errors) <= 8
            else f"; +{len(attention_shape_errors) - 8} more"
        )
        issues.append(
            _issue(
                "error",
                "attention_shape_mismatch",
                f"attention tensor shapes do not match config: {preview}{more}",
            )
        )

    if mlp_shape_errors:
        preview = "; ".join(mlp_shape_errors[:8])
        more = "" if len(mlp_shape_errors) <= 8 else f"; +{len(mlp_shape_errors) - 8} more"
        issues.append(
            _issue(
                "error",
                "mlp_shape_mismatch",
                f"MLP tensor shapes do not match config: {preview}{more}",
            )
        )

    if router_shape_errors:
        preview = "; ".join(router_shape_errors[:8])
        more = (
            ""
            if len(router_shape_errors) <= 8
            else f"; +{len(router_shape_errors) - 8} more"
        )
        issues.append(
            _issue(
                "error",
                "router_shape_mismatch",
                f"router tensor shapes do not match config: {preview}{more}",
            )
        )

    if missing_by_layer:
        preview = "; ".join(missing_by_layer[:8])
        more = "" if len(missing_by_layer) <= 8 else f"; +{len(missing_by_layer) - 8} more"
        issues.append(
            _issue(
                "error",
                "missing_resident_tensors",
                f"resident tensor coverage is incomplete: {preview}{more}",
            )
        )

    return TensorCoverage(
        embedding=embedding,
        final_norm=final_norm,
        lm_head=lm_head,
        tied_lm_head=tied,
        attention_layers_checked=config.num_hidden_layers,
        attention_layers_ok=attention_ok,
        router_layers_checked=len(moe_layers),
        router_layers_ok=router_ok,
        router_bias_layers_checked=len(moe_layers),
        router_bias_layers_found=router_bias_found,
        dense_layers_checked=len(dense_layers),
        dense_layers_ok=dense_ok,
        shared_layers_checked=shared_layers_checked,
        shared_layers_ok=shared_ok,
        indexer_layers_checked=len(indexer_layers),
        indexer_layers_ok=indexer_ok,
        missing_global=tuple(missing_global),
        missing_by_layer=tuple(missing_by_layer),
    )


def _tokenizer_preflight(
    model_dir: Path,
    tokenizer_path: str | Path | None,
    *,
    backend: str,
    trust_remote_code: bool,
    load: bool,
    issues: list[PreflightIssue],
) -> TokenizerPreflight:
    root = Path(tokenizer_path) if tokenizer_path is not None else model_dir
    tokenizer_files = (
        root if root.is_file() else root / "tokenizer.json",
        root if root.is_file() else root / "tokenizer.model",
        root if root.is_file() else root / "spiece.model",
        root if root.is_file() else root / "largerlm_tokenizer.json",
        root if root.is_file() else root / "simple_tokenizer.json",
    )
    found = any(path.exists() for path in tokenizer_files)
    if not found:
        issues.append(
            _issue(
                "warning",
                "tokenizer_not_found",
                f"no tokenizer file found under {root}; generate-text needs one",
            )
        )
        return TokenizerPreflight(False, False, None, None, None, None)
    if not load:
        return TokenizerPreflight(True, False, None, root, None, None)
    try:
        tokenizer = load_tokenizer(
            root,
            backend=backend,
            trust_remote_code=trust_remote_code,
        )
    except TokenizerError as exc:
        issues.append(_issue("warning", "tokenizer_load_failed", str(exc)))
        return TokenizerPreflight(True, False, None, root, None, str(exc))
    return TokenizerPreflight(
        True,
        True,
        tokenizer.backend,
        tokenizer.path,
        tokenizer.eos_token_id,
        None,
    )


def preflight_glm_checkpoint(
    model_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    tokenizer_backend: str = "auto",
    trust_remote_code: bool = False,
    load_tokenizer_backend: bool = False,
    quant_bits: int = 4,
    group_size: int = 64,
    quantize_raw_to_int4: bool = False,
    require_public_glm_5_2_shape: bool = False,
    max_context_tokens: int | None = None,
    max_cache_bytes: int | None = None,
    output_dir: str | Path | None = None,
    disk_safety_margin_bytes: int = 16 * 1024**3,
    unified_memory_bytes: int | None = None,
    system_reserve_bytes: int | None = None,
    runtime_buffer_bytes: int = 8 * 1024**3,
    page_cache_fraction: float = 0.60,
    cold_read_gib_per_second: float | None = None,
    prefer_header_manifest: bool = False,
) -> GlmPreflightReport:
    root = Path(model_dir)
    issues: list[PreflightIssue] = []
    try:
        config = load_config(root)
    except ConfigError as exc:
        raise PreflightError(str(exc)) from exc
    if type(require_public_glm_5_2_shape) is not bool:
        issues.append(
            _issue(
                "error",
                "invalid_require_public_glm_5_2_shape",
                "require_public_glm_5_2_shape must be a boolean",
            )
        )
        require_public_glm_5_2_shape = False
    public_shape_report = _public_glm_5_2_shape_report(config)
    public_glm_5_2_gate_failed = False
    if (
        require_public_glm_5_2_shape
        and public_shape_report.get("matches") is not True
    ):
        public_glm_5_2_gate_failed = True
        detail = _public_glm_5_2_shape_failure_detail(public_shape_report)
        suffix = f" ({detail})" if detail else ""
        issues.append(
            _issue(
                "error",
                "public_glm_5_2_shape_mismatch",
                "config does not match the public GLM-5.2 shape"
                f"{suffix}",
            )
        )
    quant_bits = _integer_param(
        "quant_bits",
        quant_bits,
        issues,
        minimum=1,
        default=4,
    )
    if require_public_glm_5_2_shape and quant_bits != 4:
        public_glm_5_2_gate_failed = True
        issues.append(
            _issue(
                "error",
                "public_glm_5_2_requires_4bit",
                "public GLM-5.2 preparation requires quant_bits=4",
            )
        )
    group_size = _integer_param(
        "group_size",
        group_size,
        issues,
        minimum=1,
        default=64,
    )
    quantization = _quantization_preflight(
        config.raw,
        target_bits=quant_bits,
        target_group_size=group_size,
    )
    if quantization.detected:
        if quantize_raw_to_int4:
            if quantization.compatible_with_mlx_affine_int4 is True:
                issues.append(
                    _issue(
                        "warning",
                        "config_quantization_raw_conversion_redundant",
                        "config declares an MLX affine-int4 checkpoint; raw "
                        "conversion expects BF16/F16/F32 expert weights, so "
                        "remove --quantize-bf16-affine-int4 if the tensors are "
                        "already pre-quantized",
                    )
                )
            if (
                quantization.bits_match_target is False
                or quantization.group_size_match_target is False
            ):
                issues.append(
                    _issue(
                        "warning",
                        "config_quantization_ignored_for_raw_conversion",
                        "config quantization metadata differs from requested raw "
                        "conversion output and will be ignored",
                    )
                )
        else:
            if quantization.bits_match_target is False:
                issues.append(
                    _issue(
                        "error",
                        "config_quantization_bits_mismatch",
                        "config quantization bits "
                        f"{quantization.bits} do not match requested quant_bits "
                        f"{quant_bits}",
                    )
                )
            if quantization.group_size_match_target is False:
                issues.append(
                    _issue(
                        "error",
                        "config_quantization_group_size_mismatch",
                        "config quantization group_size "
                        f"{quantization.group_size} does not match requested "
                        f"group_size {group_size}",
                    )
                )
            if quantization.compatible_with_mlx_affine_int4 is False:
                issues.append(
                    _issue(
                        "error",
                        "config_quantization_method_unsupported",
                        "config quantization metadata does not describe an MLX "
                        "affine int4 checkpoint",
                    )
                )
    max_context_tokens = _optional_integer_param(
        "max_context_tokens",
        max_context_tokens,
        issues,
        minimum=1,
    )
    max_cache_bytes = _optional_integer_param(
        "max_cache_bytes",
        max_cache_bytes,
        issues,
        minimum=0,
    )
    disk_safety_margin_valid = True
    try:
        if isinstance(disk_safety_margin_bytes, bool):
            raise TypeError
        disk_safety_margin_bytes = int(operator.index(disk_safety_margin_bytes))
    except TypeError:
        issues.append(
            _issue(
                "error",
                "invalid_disk_safety_margin",
                "disk_safety_margin_bytes must be an integer",
            )
        )
        disk_safety_margin_bytes = 16 * 1024**3
        disk_safety_margin_valid = False
    if disk_safety_margin_bytes < 0:
        issues.append(
            _issue(
                "error",
                "invalid_disk_safety_margin",
                "disk_safety_margin_bytes must be non-negative",
            )
        )
        disk_safety_margin_bytes = 16 * 1024**3
        disk_safety_margin_valid = False
    unified_memory_bytes = _optional_integer_param(
        "unified_memory_bytes",
        unified_memory_bytes,
        issues,
        minimum=1,
    )
    system_reserve_bytes = _optional_integer_param(
        "system_reserve_bytes",
        system_reserve_bytes,
        issues,
        minimum=0,
    )
    runtime_buffer_bytes = _integer_param(
        "runtime_buffer_bytes",
        runtime_buffer_bytes,
        issues,
        minimum=1,
        default=8 * 1024**3,
    )
    page_cache_fraction = _float_param(
        "page_cache_fraction",
        page_cache_fraction,
        issues,
        minimum=0.0,
        maximum=1.0,
        default=0.60,
    )
    cold_read_gib_per_second = _optional_float_param(
        "cold_read_gib_per_second",
        cold_read_gib_per_second,
        issues,
        minimum=0.0,
    )
    if not config.model_type.startswith("glm"):
        issues.append(
            _issue(
                "warning",
                "non_glm_model_type",
                f"model_type is {config.model_type!r}; target path is GLM-oriented",
            )
        )
    if max_context_tokens is not None:
        if (
            config.max_position_embeddings is not None
            and max_context_tokens > int(config.max_position_embeddings)
        ):
            issues.append(
                _issue(
                    "error",
                    "context_exceeds_model_max",
                    "requested max_context_tokens "
                    f"{max_context_tokens} exceeds model max_position_embeddings "
                    f"{config.max_position_embeddings}",
                )
            )

    routed = _config_int(config, "routed_experts", issues)
    experts_per_token = _config_int(config, "experts_per_token", issues)
    _config_int(config, "moe_hidden_size", issues)
    required_attention = {
        "num_attention_heads": config.num_attention_heads,
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
    }
    for name, value in required_attention.items():
        if value is None:
            issues.append(_issue("error", f"missing_{name}", f"config is missing {name}"))

    hw = detect_hardware()
    explicit_unified_memory = unified_memory_bytes is not None
    memory_bytes = unified_memory_bytes if explicit_unified_memory else hw.unified_memory_bytes
    memory_source = (
        "explicit"
        if explicit_unified_memory
        else "detected"
        if hw.unified_memory_bytes is not None
        else "unknown"
    )
    resolved_system_reserve_bytes = (
        default_system_reserve_bytes(memory_bytes)
        if system_reserve_bytes is None
        else int(system_reserve_bytes)
    )
    if public_glm_5_2_gate_failed:
        return GlmPreflightReport(
            ok=False,
            model_dir=root,
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            moe_layers=config.num_moe_layers,
            routed_experts=routed,
            experts_per_token=experts_per_token,
            dsa_index_topk=config.index_topk,
            dsa_index_n_heads=config.index_n_heads,
            dsa_index_head_dim=config.index_head_dim,
            dsa_full_indexer_q_output_dim=config.dsa_full_indexer_q_output_dim,
            checkpoint_tensor_count=0,
            checkpoint_total_bytes=0,
            checkpoint_routed_bytes=0,
            checkpoint_resident_bytes=0,
            checkpoint_ignored_bytes=0,
            hardware_chip_name=hw.chip_name,
            hardware_unified_memory_bytes=hw.unified_memory_bytes,
            hardware_gpu_cores=hw.gpu_cores,
            hardware_apple_silicon_generation=hw.apple_silicon_generation,
            hardware_apple_silicon_tier=hw.apple_silicon_tier,
            effective_unified_memory_bytes=memory_bytes,
            effective_unified_memory_source=memory_source,
            effective_system_reserve_bytes=int(resolved_system_reserve_bytes),
            tensor_coverage=_unchecked_tensor_coverage(),
            expert_coverage=ExpertCoverage(False, "not_checked", 0, 0, 0),
            resident_coverage=ResidentCoverage(False, 0, 0, 0),
            tokenizer=_unchecked_tokenizer(
                "not checked because the public GLM-5.2 gate failed"
            ),
            quantization=quantization,
            plan=None,
            disk_budget=None,
            recommended_max_live_working_set_bytes=int(runtime_buffer_bytes),
            recommended_min_free_unified_memory_bytes=int(resolved_system_reserve_bytes),
            cold_read_gib_per_second=cold_read_gib_per_second,
            public_glm_5_2_shape=public_shape_report,
            issues=tuple(issues),
        )

    try:
        tensors = iter_tensor_metadata(
            root,
            prefer_header_manifest=prefer_header_manifest,
        )
        unsupported_expert_quantized_tensors = (
            _unsupported_expert_quantized_tensor_examples(config, tensors)
        )
        moe_layers = set(config.moe_layers)
        stats = scan_checkpoint(
            root,
            prefer_header_manifest=prefer_header_manifest,
            category_fn=lambda name: categorize_tensor_for_moe_layers(
                name,
                moe_layers,
                num_hidden_layers=config.num_hidden_layers,
            ),
        )
    except SafetensorsError as exc:
        raise PreflightError(str(exc)) from exc

    coverage_tensors = tensors
    try:
        coverage_tensors = normalize_resident_tensor_aliases(tensors, config=config)
    except (ResidentPackerError, ConfigError) as exc:
        issues.append(_issue("error", "resident_alias_expansion_failed", str(exc)))
    tensor_coverage = _tensor_coverage(config, coverage_tensors, issues)
    resident_mxfp4_names = _resident_mxfp4_runtime_unsupported_names(coverage_tensors)
    if resident_mxfp4_names:
        preview = ", ".join(resident_mxfp4_names[:6])
        more = (
            ""
            if len(resident_mxfp4_names) <= 6
            else f", +{len(resident_mxfp4_names) - 6} more"
        )
        issues.append(
            _issue(
                "error",
                "mxfp4_resident_runtime_unsupported",
                "these resident MXFP4 tensors are recognized for metadata and "
                "shape coverage, but still need dedicated runtime paths "
                "(for example non-2D aliases): "
                f"{preview}{more}",
            )
        )

    expert_coverage = ExpertCoverage(False, "unknown", 0, 0, 0)
    try:
        expert_layout, _sources = build_packed_layout(
            root,
            config=config,
            group_size=group_size,
            quantization="largerlm-affine-int4" if quantize_raw_to_int4 else "mlx-affine-int4",
            quantize_raw_to_int4=quantize_raw_to_int4,
            prefer_header_manifest=prefer_header_manifest,
        )
        expert_coverage = ExpertCoverage(
            True,
            expert_layout.quantization,
            len(expert_layout.layers),
            expert_layout.total_bytes,
            max(layer.expert_slot_bytes for layer in expert_layout.layers),
        )
    except (PackerError, ConfigError) as exc:
        if unsupported_expert_quantized_tensors:
            issues.append(
                _issue(
                    "error",
                    "unsupported_expert_quantized_tensor_layout",
                    _unsupported_expert_quantized_tensor_message(
                        unsupported_expert_quantized_tensors
                    ),
                )
            )
        issues.append(_issue("error", "expert_coverage_failed", str(exc)))

    resident_coverage = ResidentCoverage(False, 0, 0, 0)
    try:
        resident_layout, _resident_tensors = build_resident_layout(
            root,
            prefer_header_manifest=prefer_header_manifest,
        )
        resident_coverage = ResidentCoverage(
            True,
            len(resident_layout.tensors),
            resident_layout.total_bytes,
            max((tensor.size for tensor in resident_layout.tensors), default=0),
        )
    except (ResidentPackerError, ConfigError) as exc:
        issues.append(_issue("error", "resident_layout_failed", str(exc)))
    recommended_max_live_working_set_bytes = int(runtime_buffer_bytes) + (
        int(resident_coverage.packed_bytes_estimate)
        if resident_coverage.checked
        else 0
    )

    plan: ModelPlan | None = None
    try:
        plan = build_plan(
            config,
            quant_bits=quant_bits,
            group_size=group_size,
            checkpoint_stats=stats,
            unified_memory_bytes=memory_bytes,
            system_reserve_bytes=resolved_system_reserve_bytes,
            runtime_buffer_bytes=runtime_buffer_bytes,
            target_page_cache_fraction=page_cache_fraction,
            cold_read_gib_per_second=cold_read_gib_per_second,
            max_context_tokens=max_context_tokens,
            max_cache_bytes=max_cache_bytes,
        )
        if plan.decode_cache_fits_budget is False:
            suffix = ""
            if plan.decode_cache_safe_context_tokens is not None:
                suffix = (
                    f"; safe context under current budget is "
                    f"{plan.decode_cache_safe_context_tokens} tokens"
                )
            issues.append(
                _issue(
                    "error",
                    "decode_cache_budget_exceeded",
                    "decode cache estimate exceeds configured cache budget" + suffix,
                )
            )
        if plan.page_cache_budget_bytes == 0:
            issues.append(
                _issue(
                    "warning",
                    "page_cache_budget_zero",
                    "resident/checkpoint/reserve settings leave no modeled OS page-cache budget",
                )
            )
        if plan.resident_memory_fits_budget is False:
            issues.append(
                _issue(
                    "warning",
                    "resident_memory_budget_exceeded",
                    "modeled resident bytes plus runtime and system reserve exceed "
                    "unified memory; lower cache/context targets or use a larger-memory profile",
                )
            )
    except (ConfigError, PlannerError, ValueError) as exc:
        issues.append(_issue("error", "planning_failed", str(exc)))

    total_output_bytes = (
        resident_coverage.packed_bytes_estimate
        + expert_coverage.packed_bytes_estimate
    )
    if plan is not None and plan.decode_cache_bytes_estimate is not None:
        total_output_bytes += plan.decode_cache_bytes_estimate
    budget = None
    if output_dir is not None and disk_safety_margin_valid:
        budget = disk_budget(
            output_dir,
            total_output_bytes,
            safety_margin_bytes=disk_safety_margin_bytes,
        )
        if not budget.ok:
            issues.append(
                _issue(
                    "error",
                    "disk_budget_exceeded",
                    "estimated packed output plus safety margin exceeds free disk",
                )
            )

    tokenizer = _tokenizer_preflight(
        root,
        tokenizer_path,
        backend=tokenizer_backend,
        trust_remote_code=trust_remote_code,
        load=load_tokenizer_backend,
        issues=issues,
    )

    ok = not any(issue.severity == "error" for issue in issues)
    return GlmPreflightReport(
        ok=ok,
        model_dir=root,
        model_type=config.model_type,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        moe_layers=config.num_moe_layers,
        routed_experts=routed,
        experts_per_token=experts_per_token,
        dsa_index_topk=config.index_topk,
        dsa_index_n_heads=config.index_n_heads,
        dsa_index_head_dim=config.index_head_dim,
        dsa_full_indexer_q_output_dim=config.dsa_full_indexer_q_output_dim,
        checkpoint_tensor_count=stats.tensor_count,
        checkpoint_total_bytes=stats.total_bytes,
        checkpoint_routed_bytes=stats.routed_expert_bytes,
        checkpoint_resident_bytes=stats.resident_bytes,
        checkpoint_ignored_bytes=stats.by_category.get("ignored_extra_layers", 0),
        hardware_chip_name=hw.chip_name,
        hardware_unified_memory_bytes=hw.unified_memory_bytes,
        hardware_gpu_cores=hw.gpu_cores,
        hardware_apple_silicon_generation=hw.apple_silicon_generation,
        hardware_apple_silicon_tier=hw.apple_silicon_tier,
        effective_unified_memory_bytes=memory_bytes,
        effective_unified_memory_source=memory_source,
        effective_system_reserve_bytes=int(resolved_system_reserve_bytes),
        tensor_coverage=tensor_coverage,
        expert_coverage=expert_coverage,
        resident_coverage=resident_coverage,
        tokenizer=tokenizer,
        quantization=quantization,
        plan=plan,
        disk_budget=budget,
        recommended_max_live_working_set_bytes=recommended_max_live_working_set_bytes,
        recommended_min_free_unified_memory_bytes=int(resolved_system_reserve_bytes),
        cold_read_gib_per_second=cold_read_gib_per_second,
        public_glm_5_2_shape=public_shape_report,
        issues=tuple(issues),
    )
