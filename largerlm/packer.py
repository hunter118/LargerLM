from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .config import ModelConfig, load_config
from .layout import (
    DEFAULT_EXPERT_COMPONENTS,
    MXFP4_EXPERT_COMPONENTS,
    ComponentLayout,
    LayerLayout,
    PackedExpertsLayout,
    config_sha256,
)
from .safety import (
    DiskBudget,
    check_chunk_budget,
    check_disk_budget,
    check_memory_budget,
    disk_budget,
    estimate_pack_peak_heap_bytes,
)
from .safetensors import TensorMeta, iter_tensor_metadata, tensor_layer_id


class PackerError(RuntimeError):
    """Raised when routed expert weights cannot be packed safely."""


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
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
_EXPERT_COMPONENT_RE = (
    r"(?:(?:gate_proj|up_proj|down_proj|w1|w2|w3)\.(?:weight|scales|biases)"
    r"|(?:gate_up_proj|gate_up|w13)\.(?:weight|scales|biases))"
)
_PER_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*?\.experts\.(?P<expert>\d+)\."
    rf"(?P<component>{_EXPERT_COMPONENT_RE})$"
)
_SWITCH_COMPONENT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*?\.switch_mlp\."
    rf"(?P<component>{_EXPERT_COMPONENT_RE})$"
)
_FUSED_EXPERTS_COMPONENT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*?\.experts\."
    rf"(?P<component>{_EXPERT_COMPONENT_RE})$"
)
EXPERT_COMPONENT_ALIASES = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}
FUSED_GATE_UP_BASES = ("gate_up_proj", "gate_up", "w13")
FUSED_GATE_UP_COMPONENTS = {
    f"{base}.{kind}"
    for base in FUSED_GATE_UP_BASES
    for kind in ("weight", "scales", "biases")
}
RAW_EXPERT_WEIGHTS: tuple[str, ...] = (
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
RAW_EXPERT_DTYPE_BYTES = {
    "BF16": 2,
    "BFLOAT16": 2,
    "F16": 2,
    "FLOAT16": 2,
    "F32": 4,
    "FLOAT32": 4,
}
AFFINE_INT4_WEIGHT_DTYPES = {"U32", "UINT32"}
AFFINE_INT4_META_DTYPES = {"BF16", "BFLOAT16", "F16", "FLOAT16"}
MXFP4_SCALE_DTYPES = {"U8", "UINT8"}
MXFP4_GROUP_SIZE = 32


@dataclass(frozen=True)
class SourceSlice:
    tensor: TensorMeta
    offset: int
    size: int
    shape: tuple[int, ...] | None = None

    @property
    def dtype(self) -> str:
        return self.tensor.dtype

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.shape or self.tensor.shape


@dataclass(frozen=True)
class ExpertComponentSource:
    layer: int
    expert: int
    component: str
    source: SourceSlice


@dataclass(frozen=True)
class PackReport:
    layout: PackedExpertsLayout
    output_dir: Path
    dry_run: bool
    chunk_size: int
    estimated_peak_heap_bytes: int
    max_heap_bytes: int
    raw_quantization_extra_heap_bytes: int
    raw_quantization_max_source_block_bytes: int
    raw_quantization_max_output_block_bytes: int
    raw_quantization_max_rows_per_block: int
    disk_budget: DiskBudget
    disk_checked: bool


@dataclass(frozen=True)
class RawQuantizationBlockBudget:
    extra_heap_bytes: int
    max_source_block_bytes: int
    max_output_block_bytes: int
    max_rows_per_block: int


def _layer_of(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _canonical_expert_component(component: str) -> str:
    base, suffix = component.split(".", 1)
    return f"{EXPERT_COMPONENT_ALIASES.get(base, base)}.{suffix}"


def _fused_gate_up_sources(
    *,
    tensor: TensorMeta,
    config: ModelConfig,
    layer: int,
    expert: int,
    component: str,
    offset: int,
    size: int,
    shape: tuple[int, ...],
) -> list[ExpertComponentSource]:
    hidden = int(config.hidden_size)
    moe_hidden = int(config.moe_hidden_size)
    base, kind = component.rsplit(".", 1)
    dtype = tensor.dtype.upper()
    if kind == "weight" and dtype in AFFINE_INT4_WEIGHT_DTYPES:
        label = f"fused affine-int4 routed expert component {base}.weight"
    elif kind == "weight":
        label = f"fused raw routed expert component {base}.weight"
    else:
        label = f"fused affine-int4 routed expert component {base}.{kind}"
    if len(shape) != 2:
        raise PackerError(
            f"{label} must be 2-D, got shape {shape}"
        )
    if shape[0] != 2 * moe_hidden:
        raise PackerError(
            f"{label} shape {shape} "
            f"does not have expected first dimension {2 * moe_hidden}"
        )
    if kind == "weight" and dtype in AFFINE_INT4_WEIGHT_DTYPES:
        if hidden % 8 != 0:
            raise PackerError("hidden_size must be divisible by 8 for affine-int4")
        expected_shape = (2 * moe_hidden, hidden // 8)
        dtype_bytes = 4
        if shape != expected_shape:
            raise PackerError(
                f"{label} shape {shape} does not match expected {expected_shape}"
            )
    elif kind == "weight":
        expected_shape = (2 * moe_hidden, hidden)
        dtype_bytes = _raw_expert_dtype_bytes(tensor.dtype)
        if shape != expected_shape:
            raise PackerError(
                f"{label} shape {shape} does not match expected {expected_shape}"
            )
    else:
        if dtype not in AFFINE_INT4_META_DTYPES:
            raise PackerError(
                f"{label} dtype {tensor.dtype} does not match expected "
                "16-bit metadata"
            )
        if shape[1] <= 0 or hidden % shape[1] != 0:
            raise PackerError(
                f"{label} shape {shape} is incompatible with hidden_size "
                f"{hidden}"
            )
        dtype_bytes = 2
    row_bytes = shape[1] * dtype_bytes
    component_size = moe_hidden * row_bytes
    expected_size = 2 * component_size
    if size != expected_size:
        raise PackerError(
            f"{label} byte size "
            f"{size} does not match dtype/shape size {expected_size}"
        )
    return [
        ExpertComponentSource(
            layer=layer,
            expert=expert,
            component=f"gate_proj.{kind}",
            source=SourceSlice(
                tensor=tensor,
                offset=offset,
                size=component_size,
                shape=(moe_hidden, shape[1]),
            ),
        ),
        ExpertComponentSource(
            layer=layer,
            expert=expert,
            component=f"up_proj.{kind}",
            source=SourceSlice(
                tensor=tensor,
                offset=offset + component_size,
                size=component_size,
                shape=(moe_hidden, shape[1]),
            ),
        ),
    ]


def _affine_raw_component_dims(
    config: ModelConfig,
    raw_component: str,
) -> tuple[int, int]:
    hidden = int(config.hidden_size)
    moe_hidden = int(config.moe_hidden_size)
    if raw_component in {"gate_proj.weight", "up_proj.weight"}:
        return moe_hidden, hidden
    if raw_component == "down_proj.weight":
        return hidden, moe_hidden
    raise PackerError(f"unknown routed expert component {raw_component}")


def _validate_affine_int4_dims(
    *,
    raw_component: str,
    out_dim: int,
    in_dim: int,
    group_size: int,
) -> None:
    if group_size <= 0:
        raise PackerError("group size must be positive")
    if group_size % 8 != 0:
        raise PackerError(f"group size {group_size} must be divisible by 8")
    if in_dim % 8 != 0:
        raise PackerError(f"{raw_component} input dim {in_dim} is not divisible by 8")
    if in_dim % group_size != 0:
        raise PackerError(
            f"{raw_component} input dim {in_dim} is not divisible by "
            f"group size {group_size}"
        )
    if out_dim <= 0 or in_dim <= 0:
        raise PackerError(f"{raw_component} dimensions must be positive")


def _expected_affine_int4_source_component(
    *,
    config: ModelConfig,
    component: str,
    group_size: int,
) -> tuple[str, int, tuple[int, ...]]:
    base, kind = component.rsplit(".", 1)
    raw_component = f"{base}.weight"
    out_dim, in_dim = _affine_raw_component_dims(config, raw_component)
    _validate_affine_int4_dims(
        raw_component=raw_component,
        out_dim=out_dim,
        in_dim=in_dim,
        group_size=group_size,
    )
    if kind == "weight":
        shape = (out_dim, in_dim // 8)
        return "U32", out_dim * (in_dim // 8) * 4, shape
    if kind in {"scales", "biases"}:
        shape = (out_dim, in_dim // group_size)
        return "BF16", out_dim * (in_dim // group_size) * 2, shape
    raise PackerError(f"unknown routed expert component {component}")


def _component_layouts(
    sources: dict[str, SourceSlice],
    *,
    config: ModelConfig,
    group_size: int,
) -> tuple[ComponentLayout, ...]:
    if _looks_like_mxfp4_scales_only_sources(sources):
        raise PackerError(_mxfp4_scales_only_message())
    offset = 0
    layouts: list[ComponentLayout] = []
    for component in DEFAULT_EXPERT_COMPONENTS:
        source = sources.get(component)
        if source is None:
            raise PackerError(f"missing routed expert component {component}")
        expected_dtype, expected_size, expected_shape = (
            _expected_affine_int4_source_component(
                config=config,
                component=component,
                group_size=group_size,
            )
        )
        dtype = source.dtype.upper()
        if expected_dtype == "U32":
            if dtype not in AFFINE_INT4_WEIGHT_DTYPES:
                raise PackerError(
                    f"routed expert component {component} dtype {source.dtype} "
                    "does not match expected U32 affine-int4 weights"
                )
        elif dtype not in AFFINE_INT4_META_DTYPES:
            raise PackerError(
                f"routed expert component {component} dtype {source.dtype} "
                "does not match expected 16-bit affine-int4 metadata"
            )
        if source.logical_shape != expected_shape:
            raise PackerError(
                f"routed expert component {component} shape "
                f"{source.logical_shape} does not match expected {expected_shape}"
            )
        if source.size != expected_size:
            raise PackerError(
                f"routed expert component {component} byte size {source.size} "
                f"does not match expected {expected_size}"
            )
        layouts.append(
            ComponentLayout(
                name=component,
                offset=offset,
                size=source.size,
                dtype=source.dtype,
                shape=source.logical_shape,
            )
        )
        offset += source.size
    return tuple(layouts)


def _expected_mxfp4_source_component(
    *,
    config: ModelConfig,
    component: str,
) -> tuple[str, int, tuple[int, ...]]:
    base, kind = component.rsplit(".", 1)
    raw_component = f"{base}.weight"
    out_dim, in_dim = _affine_raw_component_dims(config, raw_component)
    if in_dim % 8 != 0:
        raise PackerError(f"{raw_component} input dim {in_dim} is not divisible by 8")
    if in_dim % MXFP4_GROUP_SIZE != 0:
        raise PackerError(
            f"{raw_component} input dim {in_dim} is not divisible by "
            f"MXFP4 group size {MXFP4_GROUP_SIZE}"
        )
    if kind == "weight":
        shape = (out_dim, in_dim // 8)
        return "U32", out_dim * (in_dim // 8) * 4, shape
    if kind == "scales":
        shape = (out_dim, in_dim // MXFP4_GROUP_SIZE)
        return "U8", out_dim * (in_dim // MXFP4_GROUP_SIZE), shape
    raise PackerError(f"unknown MXFP4 routed expert component {component}")


def _mxfp4_component_layouts(
    sources: dict[str, SourceSlice],
    *,
    config: ModelConfig,
) -> tuple[ComponentLayout, ...]:
    offset = 0
    layouts: list[ComponentLayout] = []
    for component in MXFP4_EXPERT_COMPONENTS:
        source = sources.get(component)
        if source is None:
            raise PackerError(f"missing MXFP4 routed expert component {component}")
        expected_dtype, expected_size, expected_shape = (
            _expected_mxfp4_source_component(
                config=config,
                component=component,
            )
        )
        dtype = source.dtype.upper()
        if expected_dtype == "U32":
            if dtype not in AFFINE_INT4_WEIGHT_DTYPES:
                raise PackerError(
                    f"MXFP4 routed expert component {component} dtype {source.dtype} "
                    "does not match expected U32 weights"
                )
        elif dtype not in MXFP4_SCALE_DTYPES:
            raise PackerError(
                f"MXFP4 routed expert component {component} dtype {source.dtype} "
                "does not match expected U8 scales"
            )
        if source.logical_shape != expected_shape:
            raise PackerError(
                f"MXFP4 routed expert component {component} shape "
                f"{source.logical_shape} does not match expected {expected_shape}"
            )
        if source.size != expected_size:
            raise PackerError(
                f"MXFP4 routed expert component {component} byte size "
                f"{source.size} does not match expected {expected_size}"
            )
        layouts.append(
            ComponentLayout(
                name=component,
                offset=offset,
                size=source.size,
                dtype=source.dtype,
                shape=source.logical_shape,
            )
        )
        offset += source.size
    return tuple(layouts)


def _looks_like_mxfp4_scales_only_sources(
    sources: dict[str, SourceSlice],
) -> bool:
    for raw_component in RAW_EXPERT_WEIGHTS:
        base = raw_component.removesuffix(".weight")
        weight = sources.get(f"{base}.weight")
        scales = sources.get(f"{base}.scales")
        biases = sources.get(f"{base}.biases")
        if weight is None or scales is None or biases is not None:
            return False
        if weight.dtype.upper() not in AFFINE_INT4_WEIGHT_DTYPES:
            return False
        if scales.dtype.upper() not in MXFP4_SCALE_DTYPES:
            return False
    return True


def _mxfp4_scales_only_message() -> str:
    return (
        "routed expert components look like MLX MXFP4 "
        "(U32 weights plus U8 scales without affine biases); "
        "metadata-only preflight can inspect this checkpoint, but LargerLM "
        "still needs mlx-mxfp4 expert packing and Metal dequant kernels before "
        "it can execute this layout"
    )


def _raw_expert_dtype_bytes(dtype: str) -> int:
    if dtype.upper() in AFFINE_INT4_WEIGHT_DTYPES:
        raise PackerError(
            "routed expert weights already look like affine-int4; remove "
            "--quantize-bf16-affine-int4 when preparing a pre-quantized MLX "
            "checkpoint"
        )
    try:
        return RAW_EXPERT_DTYPE_BYTES[dtype.upper()]
    except KeyError as exc:
        raise PackerError(
            f"unsupported raw expert dtype {dtype}; expected BF16/F16/F32"
        ) from exc


def _expected_raw_expert_bytes(
    *,
    dtype: str,
    shape: tuple[int, ...],
) -> int:
    total = _raw_expert_dtype_bytes(dtype)
    for dim in shape:
        total *= int(dim)
    return total


def _affine_int4_component_layouts(
    sources: dict[str, SourceSlice],
    *,
    config: ModelConfig,
    group_size: int,
) -> tuple[ComponentLayout, ...]:
    offset = 0
    by_name: dict[str, ComponentLayout] = {}
    for raw_component in RAW_EXPERT_WEIGHTS:
        source = sources.get(raw_component)
        if source is None:
            raise PackerError(f"missing raw routed expert component {raw_component}")
        source_shape = source.logical_shape
        if len(source_shape) != 2:
            raise PackerError(
                f"raw routed expert component {raw_component} must be 2-D, "
                f"got shape {source_shape}"
            )
        expected_bytes = _expected_raw_expert_bytes(
            dtype=source.dtype,
            shape=source_shape,
        )
        if source.size != expected_bytes:
            raise PackerError(
                f"raw routed expert component {raw_component} byte size "
                f"{source.size} does not match dtype/shape size {expected_bytes}"
            )
        out_dim, in_dim = source_shape
        expected_shape = _affine_raw_component_dims(config, raw_component)
        if source_shape != expected_shape:
            raise PackerError(
                f"raw routed expert component {raw_component} shape "
                f"{source_shape} does not match config expected {expected_shape}"
            )
        _validate_affine_int4_dims(
            raw_component=raw_component,
            out_dim=out_dim,
            in_dim=in_dim,
            group_size=group_size,
        )

        packed_cols = in_dim // 8
        groups_per_row = in_dim // group_size
        weight_size = out_dim * packed_cols * 4
        meta_size = out_dim * groups_per_row * 2
        base = raw_component.removesuffix(".weight")
        for name, size, dtype, shape in (
            (f"{base}.weight", weight_size, "U32", (out_dim, packed_cols)),
            (f"{base}.scales", meta_size, "BF16", (out_dim, groups_per_row)),
            (f"{base}.biases", meta_size, "BF16", (out_dim, groups_per_row)),
        ):
            by_name[name] = ComponentLayout(
                name=name,
                offset=offset,
                size=size,
                dtype=dtype,
                shape=shape,
            )
            offset += size

    return tuple(by_name[name] for name in DEFAULT_EXPERT_COMPONENTS)


def _expert_sources_from_tensor(
    tensor: TensorMeta,
    *,
    config: ModelConfig,
    moe_layers: set[int],
) -> list[ExpertComponentSource]:
    per = _PER_EXPERT_RE.search(tensor.name)
    if per:
        layer = int(per.group("layer"))
        if layer not in moe_layers:
            return []
        expert = int(per.group("expert"))
        if expert < 0 or expert >= int(config.routed_experts):
            raise PackerError(
                f"routed expert id {expert} in tensor {tensor.name} is outside "
                f"configured range [0, {int(config.routed_experts)})"
            )
        component = per.group("component")
        if component in FUSED_GATE_UP_COMPONENTS:
            return _fused_gate_up_sources(
                tensor=tensor,
                config=config,
                layer=layer,
                expert=expert,
                component=component,
                offset=tensor.absolute_start,
                size=tensor.nbytes,
                shape=tensor.shape,
            )
        return [
            ExpertComponentSource(
                layer=layer,
                expert=expert,
                component=_canonical_expert_component(component),
                source=SourceSlice(
                    tensor=tensor,
                    offset=tensor.absolute_start,
                    size=tensor.nbytes,
                    shape=tensor.shape,
                ),
            )
        ]

    fused = _SWITCH_COMPONENT_RE.search(tensor.name) or _FUSED_EXPERTS_COMPONENT_RE.search(
        tensor.name
    )
    if not fused:
        return []

    layer = int(fused.group("layer"))
    if layer not in moe_layers:
        return []
    if not tensor.shape or tensor.shape[0] != config.routed_experts:
        raise PackerError(
            "cannot infer expert stride for fused expert tensor "
            f"{tensor.name}; expected first dimension to be {config.routed_experts}, "
            f"got shape {tensor.shape}"
        )
    if tensor.nbytes % config.routed_experts != 0:
        raise PackerError(
            f"tensor {tensor.name} byte size {tensor.nbytes} is not divisible by "
            f"{config.routed_experts} experts"
        )

    stride = tensor.nbytes // config.routed_experts
    expert_shape = tensor.shape[1:] if len(tensor.shape) > 1 else tensor.shape
    component = fused.group("component")
    if component in FUSED_GATE_UP_COMPONENTS:
        result: list[ExpertComponentSource] = []
        for expert in range(config.routed_experts):
            result.extend(
                _fused_gate_up_sources(
                    tensor=tensor,
                    config=config,
                    layer=layer,
                    expert=expert,
                    component=component,
                    offset=tensor.absolute_start + expert * stride,
                    size=stride,
                    shape=expert_shape,
                )
            )
        return result
    canonical_component = _canonical_expert_component(component)
    return [
        ExpertComponentSource(
            layer=layer,
            expert=expert,
            component=canonical_component,
            source=SourceSlice(
                tensor=tensor,
                offset=tensor.absolute_start + expert * stride,
                size=stride,
                shape=expert_shape,
            ),
        )
        for expert in range(config.routed_experts)
    ]


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


def _discover_expert_sources_from_metadata(
    tensors: list[TensorMeta],
    *,
    config: ModelConfig,
) -> dict[tuple[int, int], dict[str, SourceSlice]]:
    moe_layers = set(config.moe_layers)
    result: dict[tuple[int, int], dict[str, SourceSlice]] = {}
    for tensor in tensors:
        for source in _expert_sources_from_tensor(
            tensor,
            config=config,
            moe_layers=moe_layers,
        ):
            key = (source.layer, source.expert)
            components = result.setdefault(key, {})
            if source.component in components:
                raise PackerError(
                    f"duplicate routed expert component {source.component} "
                    f"for layer {source.layer} expert {source.expert}"
                )
            components[source.component] = source.source
    return result


def discover_expert_sources(
    model_dir: str | Path,
    *,
    config: ModelConfig | None = None,
) -> dict[tuple[int, int], dict[str, SourceSlice]]:
    cfg = config or load_config(model_dir)
    tensors = iter_tensor_metadata(model_dir)
    sources = _discover_expert_sources_from_metadata(tensors, config=cfg)
    if not sources:
        unsupported_examples = _unsupported_expert_quantized_tensor_examples(
            cfg,
            tensors,
        )
        if unsupported_examples:
            raise PackerError(
                _unsupported_expert_quantized_tensor_message(unsupported_examples)
            )
    return sources


def build_packed_layout(
    model_dir: str | Path,
    *,
    config: ModelConfig | None = None,
    quantization: str = "mlx-affine-int4",
    group_size: int | None = 64,
    layers: set[int] | None = None,
    quantize_raw_to_int4: bool = False,
    prefer_header_manifest: bool = False,
) -> tuple[PackedExpertsLayout, dict[tuple[int, int], dict[str, SourceSlice]]]:
    cfg = config or load_config(model_dir)
    selected_layers = set(cfg.moe_layers if layers is None else layers)
    tensors = iter_tensor_metadata(
        model_dir,
        prefer_header_manifest=prefer_header_manifest,
    )
    sources = _discover_expert_sources_from_metadata(tensors, config=cfg)
    unsupported_examples = _unsupported_expert_quantized_tensor_examples(
        cfg,
        tensors,
    )
    if quantize_raw_to_int4 and not group_size:
        raise PackerError("group_size is required when quantizing raw experts")

    layer_layouts: list[LayerLayout] = []
    effective_quantization = quantization
    effective_group_size = group_size
    for layer in cfg.moe_layers:
        if layer not in selected_layers:
            continue
        first_key = (layer, 0)
        if first_key not in sources:
            if unsupported_examples:
                raise PackerError(
                    _unsupported_expert_quantized_tensor_message(
                        unsupported_examples
                    )
                )
            raise PackerError(f"missing routed expert sources for layer {layer}")
        if quantize_raw_to_int4:
            components = _affine_int4_component_layouts(
                sources[first_key],
                config=cfg,
                group_size=group_size,
            )
            layout_mode = "largerlm-affine-int4"
            expected_component_order = DEFAULT_EXPERT_COMPONENTS
            effective_quantization = "largerlm-affine-int4"
        else:
            if _looks_like_mxfp4_scales_only_sources(sources[first_key]):
                components = _mxfp4_component_layouts(
                    sources[first_key],
                    config=cfg,
                )
                layout_mode = "mlx-mxfp4"
                expected_component_order = MXFP4_EXPERT_COMPONENTS
                effective_quantization = "mlx-mxfp4"
                effective_group_size = MXFP4_GROUP_SIZE
            else:
                components = _component_layouts(
                    sources[first_key],
                    config=cfg,
                    group_size=group_size or 0,
                )
                layout_mode = quantization
                expected_component_order = DEFAULT_EXPERT_COMPONENTS
        slot_bytes = sum(component.size for component in components)

        for expert in range(cfg.routed_experts):
            key = (layer, expert)
            if key not in sources:
                raise PackerError(f"missing routed expert {expert} for layer {layer}")
            if quantize_raw_to_int4:
                expert_components = _affine_int4_component_layouts(
                    sources[key],
                    config=cfg,
                    group_size=group_size,
                )
            elif layout_mode == "mlx-mxfp4":
                expert_components = _mxfp4_component_layouts(
                    sources[key],
                    config=cfg,
                )
            else:
                expert_components = _component_layouts(
                    sources[key],
                    config=cfg,
                    group_size=group_size or 0,
                )
            expert_slot = sum(component.size for component in expert_components)
            if expert_slot != slot_bytes:
                raise PackerError(
                    f"expert slot size mismatch for layer {layer} expert {expert}: "
                    f"{expert_slot} != {slot_bytes}"
                )
            if len(components) != len(expert_components):
                raise PackerError(
                    f"component count mismatch for layer {layer} expert {expert}"
                )
            for a, b in zip(components, expert_components):
                if a.name != b.name or a.size != b.size or a.dtype != b.dtype or a.shape != b.shape:
                    raise PackerError(
                        f"component layout mismatch for layer {layer} expert {expert} "
                        f"{a.name}"
                    )

        layer_layouts.append(
            LayerLayout(
                layer=layer,
                num_experts=cfg.routed_experts,
                expert_slot_bytes=slot_bytes,
                layer_file=f"layer_{layer:03d}.bin",
                components=components,
            )
        )

    if not layer_layouts:
        raise PackerError("no MoE layers selected for packing")

    layout = PackedExpertsLayout(
        version=1,
        model_type=cfg.model_type,
        config_sha256=config_sha256(model_dir),
        quantization=effective_quantization,
        group_size=effective_group_size,
        num_layers=cfg.num_hidden_layers,
        num_experts=cfg.routed_experts,
        component_order=expected_component_order,
        layers=tuple(layer_layouts),
    )
    return layout, sources


def _copy_slice(
    *,
    src_fd: int,
    dst_fd: int,
    src_offset: int,
    dst_offset: int,
    size: int,
    chunk_size: int,
) -> None:
    copied = 0
    while copied < size:
        to_read = min(chunk_size, size - copied)
        data = os.pread(src_fd, to_read, src_offset + copied)
        if len(data) != to_read:
            raise PackerError(
                f"short read at offset {src_offset + copied}: "
                f"expected {to_read}, got {len(data)}"
            )
        written = os.pwrite(dst_fd, data, dst_offset + copied)
        if written != len(data):
            raise PackerError(
                f"short write at offset {dst_offset + copied}: "
                f"expected {len(data)}, got {written}"
            )
        copied += to_read


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _read_exact(fd: int, offset: int, size: int, chunk_size: int) -> bytearray:
    raw = bytearray(size)
    copied = 0
    while copied < size:
        to_read = min(chunk_size, size - copied)
        data = os.pread(fd, to_read, offset + copied)
        if len(data) != to_read:
            raise PackerError(
                f"short read at offset {offset + copied}: expected {to_read}, "
                f"got {len(data)}"
            )
        raw[copied : copied + to_read] = data
        copied += to_read
    return raw


def _bf16_to_f32(raw: bytes, offset: int) -> float:
    bits = int.from_bytes(raw[offset : offset + 2], "little") << 16
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def _f32_to_bf16(value: float) -> bytes:
    bits = int.from_bytes(struct.pack("<f", float(value)), "little")
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return ((rounded >> 16) & 0xFFFF).to_bytes(2, "little")


def _decode_value(raw: bytes | bytearray, dtype: str, index: int) -> float:
    normalized = dtype.upper()
    if normalized in {"BF16", "BFLOAT16"}:
        return _bf16_to_f32(raw, index * 2)
    if normalized in {"F16", "FLOAT16"}:
        return struct.unpack_from("<e", raw, index * 2)[0]
    if normalized in {"F32", "FLOAT32"}:
        return struct.unpack_from("<f", raw, index * 4)[0]
    raise PackerError(f"unsupported raw expert dtype {dtype}; expected BF16/F16/F32")


def _quantize_affine_int4_matrix(
    raw: bytes | bytearray,
    *,
    dtype: str,
    shape: tuple[int, ...],
    group_size: int,
) -> tuple[bytearray, bytearray, bytearray]:
    if len(shape) != 2:
        raise PackerError(f"expected 2-D raw expert tensor, got shape {shape}")
    out_dim, in_dim = shape
    return _quantize_affine_int4_rows(
        raw,
        dtype=dtype,
        rows=out_dim,
        in_dim=in_dim,
        group_size=group_size,
    )


def _quantize_affine_int4_rows(
    raw: bytes | bytearray,
    *,
    dtype: str,
    rows: int,
    in_dim: int,
    group_size: int,
) -> tuple[bytearray, bytearray, bytearray]:
    if rows <= 0 or in_dim <= 0:
        raise PackerError("raw expert row block dimensions must be positive")
    if group_size <= 0 or group_size % 8 != 0 or in_dim % group_size != 0:
        raise PackerError(
            f"invalid group size {group_size} for raw expert row block "
            f"({rows}, {in_dim})"
        )
    dtype_bytes = _raw_expert_dtype_bytes(dtype)
    expected_raw_bytes = rows * in_dim * dtype_bytes
    if len(raw) != expected_raw_bytes:
        raise PackerError(
            f"raw expert row block has {len(raw)} bytes, expected "
            f"{expected_raw_bytes}"
        )
    packed = bytearray(rows * (in_dim // 8) * 4)
    scales = bytearray(rows * (in_dim // group_size) * 2)
    biases = bytearray(rows * (in_dim // group_size) * 2)

    groups_per_row = in_dim // group_size
    packed_word = 0
    meta_index = 0
    for row in range(rows):
        row_base = row * in_dim
        for group in range(groups_per_row):
            group_base = row_base + group * group_size
            values = [
                _decode_value(raw, dtype, group_base + i)
                for i in range(group_size)
            ]
            if not all(math.isfinite(value) for value in values):
                raise PackerError(
                    "raw expert row block contains non-finite value "
                    f"at row {row}, group {group}"
                )
            lo = min(values)
            hi = max(values)
            scale = (hi - lo) / 15.0
            scales[meta_index * 2 : meta_index * 2 + 2] = _f32_to_bf16(scale)
            biases[meta_index * 2 : meta_index * 2 + 2] = _f32_to_bf16(lo)
            meta_index += 1

            for base in range(0, group_size, 8):
                word = 0
                for lane, value in enumerate(values[base : base + 8]):
                    if scale == 0:
                        q = 0
                    else:
                        q = int(round((value - lo) / scale))
                        q = max(0, min(15, q))
                    word |= (q & 0xF) << (lane * 4)
                struct.pack_into("<I", packed, packed_word * 4, word)
                packed_word += 1

    return packed, scales, biases


def _raw_quantization_rows_per_block(
    *,
    dtype: str,
    in_dim: int,
    chunk_size: int,
) -> int:
    row_source_bytes = in_dim * _raw_expert_dtype_bytes(dtype)
    if row_source_bytes <= 0:
        raise PackerError("raw expert row byte size must be positive")
    return max(1, chunk_size // row_source_bytes)


def _raw_quantization_output_bytes_for_rows(
    *,
    rows: int,
    in_dim: int,
    group_size: int,
) -> int:
    packed_row_bytes = (in_dim // 8) * 4
    meta_row_bytes = (in_dim // group_size) * 2
    return rows * (packed_row_bytes + 2 * meta_row_bytes)


def _estimate_raw_quantization_block_budget(
    *,
    sources: dict[tuple[int, int], dict[str, SourceSlice]],
    chunk_size: int,
    group_size: int,
) -> RawQuantizationBlockBudget:
    max_extra = 0
    max_source_block = 0
    max_output_block = 0
    max_rows_per_block = 0
    for component_sources in sources.values():
        for raw_component in RAW_EXPERT_WEIGHTS:
            src = component_sources[raw_component]
            shape = src.logical_shape
            if len(shape) != 2:
                continue
            out_dim, in_dim = shape
            rows_per_block = min(
                out_dim,
                _raw_quantization_rows_per_block(
                    dtype=src.dtype,
                    in_dim=in_dim,
                    chunk_size=chunk_size,
                ),
            )
            source_block_bytes = rows_per_block * in_dim * _raw_expert_dtype_bytes(
                src.dtype
            )
            output_block_bytes = _raw_quantization_output_bytes_for_rows(
                rows=rows_per_block,
                in_dim=in_dim,
                group_size=group_size,
            )
            max_source_block = max(max_source_block, source_block_bytes)
            max_output_block = max(max_output_block, output_block_bytes)
            max_rows_per_block = max(max_rows_per_block, rows_per_block)
            max_extra = max(
                max_extra,
                max(0, source_block_bytes - chunk_size) + output_block_bytes,
            )
    return RawQuantizationBlockBudget(
        extra_heap_bytes=max_extra,
        max_source_block_bytes=max_source_block,
        max_output_block_bytes=max_output_block,
        max_rows_per_block=max_rows_per_block,
    )


def _write_quantized_raw_expert(
    *,
    root: Path,
    dst_fd: int,
    dst_offset: int,
    component_sources: dict[str, SourceSlice],
    component_layouts: tuple[ComponentLayout, ...],
    open_fds: dict[str, int],
    group_size: int,
    chunk_size: int,
) -> None:
    layout_by_name = {component.name: component for component in component_layouts}
    for raw_component in RAW_EXPERT_WEIGHTS:
        src = component_sources[raw_component]
        out_dim, in_dim = src.logical_shape
        dtype_bytes = _raw_expert_dtype_bytes(src.dtype)
        source_row_bytes = in_dim * dtype_bytes
        rows_per_block = min(
            out_dim,
            _raw_quantization_rows_per_block(
                dtype=src.dtype,
                in_dim=in_dim,
                chunk_size=chunk_size,
            ),
        )
        base = raw_component.removesuffix(".weight")
        weight_layout = layout_by_name[f"{base}.weight"]
        scale_layout = layout_by_name[f"{base}.scales"]
        bias_layout = layout_by_name[f"{base}.biases"]
        weight_row_bytes = (in_dim // 8) * 4
        meta_row_bytes = (in_dim // group_size) * 2
        shard_path = str(root / src.tensor.shard)
        if shard_path not in open_fds:
            open_fds[shard_path] = os.open(shard_path, os.O_RDONLY)
        for row_start in range(0, out_dim, rows_per_block):
            rows = min(rows_per_block, out_dim - row_start)
            raw = _read_exact(
                open_fds[shard_path],
                src.offset + row_start * source_row_bytes,
                rows * source_row_bytes,
                chunk_size,
            )
            weight, scales, biases = _quantize_affine_int4_rows(
                raw,
                dtype=src.dtype,
                rows=rows,
                in_dim=in_dim,
                group_size=group_size,
            )
            for layout, data, row_bytes in (
                (weight_layout, weight, weight_row_bytes),
                (scale_layout, scales, meta_row_bytes),
                (bias_layout, biases, meta_row_bytes),
            ):
                expected_size = rows * row_bytes
                if len(data) != expected_size:
                    raise PackerError(
                        f"generated component {layout.name} row block has "
                        f"{len(data)} bytes, expected {expected_size}"
                    )
                written = os.pwrite(
                    dst_fd,
                    data,
                    dst_offset + layout.offset + row_start * row_bytes,
                )
                if written != len(data):
                    raise PackerError(
                        f"short write for generated component {layout.name}: "
                        f"expected {len(data)}, got {written}"
                    )

def pack_experts(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = True,
    force: bool = False,
    chunk_size: int = 8 * 1024**2,
    max_chunk_size: int = 64 * 1024**2,
    max_heap_bytes: int = 512 * 1024**2,
    disk_safety_margin_bytes: int = 16 * 1024**3,
    layers: set[int] | None = None,
    quantize_raw_to_int4: bool = False,
    group_size: int = 64,
    prefer_header_manifest: bool = False,
) -> PackReport:
    root = Path(model_dir)
    out = Path(output_dir)
    if prefer_header_manifest and not dry_run:
        raise PackerError("prefer_header_manifest is only supported for dry-run packing")
    cfg = load_config(root)
    check_chunk_budget(chunk_size, max_chunk_size)
    base_peak_heap = estimate_pack_peak_heap_bytes(chunk_size=chunk_size)
    layout, sources = build_packed_layout(
        root,
        config=cfg,
        layers=layers,
        group_size=group_size,
        quantization="largerlm-affine-int4" if quantize_raw_to_int4 else "mlx-affine-int4",
        quantize_raw_to_int4=quantize_raw_to_int4,
        prefer_header_manifest=prefer_header_manifest,
    )
    estimated_peak_heap = base_peak_heap
    raw_quantization_budget = RawQuantizationBlockBudget(
        extra_heap_bytes=0,
        max_source_block_bytes=0,
        max_output_block_bytes=0,
        max_rows_per_block=0,
    )
    if quantize_raw_to_int4:
        raw_quantization_budget = _estimate_raw_quantization_block_budget(
            sources=sources,
            chunk_size=chunk_size,
            group_size=layout.group_size or 64,
        )
        estimated_peak_heap = base_peak_heap + raw_quantization_budget.extra_heap_bytes
    check_memory_budget(estimated_peak_heap, max_heap_bytes)
    budget = disk_budget(
        out,
        layout.total_bytes,
        safety_margin_bytes=disk_safety_margin_bytes,
    )

    disk_checked = False
    if not dry_run:
        budget = check_disk_budget(
            out,
            layout.total_bytes,
            safety_margin_bytes=disk_safety_margin_bytes,
        )
        disk_checked = True
        out.mkdir(parents=True, exist_ok=True)
        layout_path = out / "layout.json"
        if layout_path.exists() and not force:
            raise PackerError(f"{layout_path} already exists; use --force to overwrite")

        open_fds: dict[str, int] = {}
        partial_paths: list[Path] = []
        try:
            for layer_layout in layout.layers:
                layer_path = out / layer_layout.layer_file
                if layer_path.exists() and not force:
                    raise PackerError(f"{layer_path} already exists; use --force")
                dst_fd = os.open(layer_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
                partial_paths.append(layer_path)
                try:
                    os.ftruncate(dst_fd, layer_layout.layer_file_bytes)
                    for expert in range(layer_layout.num_experts):
                        expert_base = expert * layer_layout.expert_slot_bytes
                        component_sources = sources[(layer_layout.layer, expert)]
                        if quantize_raw_to_int4:
                            _write_quantized_raw_expert(
                                root=root,
                                dst_fd=dst_fd,
                                dst_offset=expert_base,
                                component_sources=component_sources,
                                component_layouts=layer_layout.components,
                                open_fds=open_fds,
                                group_size=layout.group_size or 64,
                                chunk_size=chunk_size,
                            )
                        else:
                            for component in layer_layout.components:
                                src = component_sources[component.name]
                                shard_path = str(root / src.tensor.shard)
                                if shard_path not in open_fds:
                                    open_fds[shard_path] = os.open(shard_path, os.O_RDONLY)
                                _copy_slice(
                                    src_fd=open_fds[shard_path],
                                    dst_fd=dst_fd,
                                    src_offset=src.offset,
                                    dst_offset=expert_base + component.offset,
                                    size=component.size,
                                    chunk_size=chunk_size,
                                )
                finally:
                    os.close(dst_fd)
            layout.write(out / "layout.json")
        except OSError as exc:
            for path in reversed(partial_paths):
                _remove_partial_file(path)
            raise PackerError(f"failed to pack experts: {exc}") from exc
        except PackerError:
            for path in reversed(partial_paths):
                _remove_partial_file(path)
            raise
        finally:
            for fd in open_fds.values():
                os.close(fd)

    return PackReport(
        layout=layout,
        output_dir=out,
        dry_run=dry_run,
        chunk_size=chunk_size,
        estimated_peak_heap_bytes=estimated_peak_heap,
        max_heap_bytes=max_heap_bytes,
        raw_quantization_extra_heap_bytes=raw_quantization_budget.extra_heap_bytes,
        raw_quantization_max_source_block_bytes=(
            raw_quantization_budget.max_source_block_bytes
        ),
        raw_quantization_max_output_block_bytes=(
            raw_quantization_budget.max_output_block_bytes
        ),
        raw_quantization_max_rows_per_block=(
            raw_quantization_budget.max_rows_per_block
        ),
        disk_budget=budget,
        disk_checked=disk_checked,
    )
